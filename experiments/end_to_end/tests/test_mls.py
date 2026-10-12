"""CPU coverage of masked solves, covariance sketches and real harness transactions."""
from contextlib import contextmanager
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e2e_fsa as E
from mls import CodeTransaction, CovarianceSketch, MLSOptimizer, masked_cg, masked_matvec, ternarize
from test_e2e_fsa import cpu_harness  # reuse the full-model CPU loader fixture


@pytest.mark.parametrize("regularized", [False, True])
def test_masked_cg_matches_augmented_lstsq(regularized):
    gen = torch.Generator().manual_seed(30)
    x, target = torch.randn(12, 7, generator=gen), torch.randn(12, 4, generator=gen)
    mask = torch.rand(4, 7, generator=gen) > .4
    mask[-1] = False
    diagonal, sketch = torch.rand(7, generator=gen), torch.randn(7, 2, generator=gen)
    weight, gamma = (.3, .1) if regularized else (0., 0.)
    actual = masked_cg(x, target, mask, diagonal, sketch, weight, gamma, iterations=30,
                       initial=torch.randn(4, 7, generator=gen))
    expected = torch.zeros_like(actual)
    for row in range(mask.shape[0] - 1):
        selected = mask[row].nonzero().flatten()
        ridge = torch.diag((weight * diagonal[selected] + gamma).sqrt())
        design = torch.cat([x[:, selected], ridge, weight ** .5 * sketch[selected].T])
        response = torch.cat([target[:, row], torch.zeros(selected.numel() + sketch.shape[1])])
        expected[row, selected] = torch.linalg.lstsq(design, response).solution
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert not actual[~mask].any()


def test_ternarization_mask_and_greedy_error_feedback():
    # Correlated columns: rounding .8 up to 1 pushes the remaining .4 to .2.
    x = torch.tensor([[1., 1., 0.]])
    w = torch.tensor([[.8, .4, 10.]])
    current = torch.tensor([[0., 0., -1.]])
    mask = torch.tensor([[True, True, False]])
    codes, corrected = ternarize(w, current, torch.ones(1, 1), mask, x,
                                 torch.zeros(3), torch.empty(3, 0), gen_lambda=0., gamma=1e-6)
    assert torch.equal(codes, torch.tensor([[1., 0., -1.]]))
    assert corrected[0, 1] == pytest.approx(.2, abs=1e-5)
    assert set(codes.flatten().tolist()) <= {-1, 0, 1}


def test_streaming_covariance_sketch_approximates_full_rank_covariance():
    gen = torch.Generator().manual_seed(22)
    x = torch.randn(20, 4, generator=gen)
    accumulator = CovarianceSketch(4, 4, x.device, gen)
    accumulator.add(x[:7])
    accumulator.add(x[7:])
    diagonal, sketch = accumulator.finish()
    covariance = x.T @ x / len(x)
    assert diagonal.dtype == sketch.dtype == torch.bfloat16
    torch.testing.assert_close(diagonal.float(), covariance.diag(), atol=.01, rtol=.01)
    torch.testing.assert_close(sketch.float() @ sketch.float().T, covariance, atol=.02, rtol=.02)


@pytest.mark.parametrize("rank", [0, 2])
def test_covariance_unrouted_and_zero_activations_are_finite(rank):
    accumulator = CovarianceSketch(4, rank, "cpu", torch.Generator().manual_seed(4))
    assert accumulator.finish()[1].shape == (4, 0)
    accumulator.add(torch.zeros(3, 4))
    diagonal, sketch = accumulator.finish()
    assert torch.isfinite(diagonal).all() and torch.isfinite(sketch).all()


def _linear(sparse=False, budgeted=False):
    linear = nn.Linear(8, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1., -1., 1., -1., 0., 0., 0., 0.]]).repeat(2, 1))
    linear.weight.requires_grad_(False)
    cls = E.SparseFSAParam if sparse else E.FSAParam
    param = cls(linear.weight, E.pair8_forced_mask(linear.weight), budgeted=budgeted)
    if param.support is not param.mask:
        param.support.copy_(param.mask)
    P.register_parametrization(linear, "weight", param)
    return linear, param


@pytest.mark.parametrize("sparse", [False, True])
def test_transaction_rejection_exception_and_acceptance(sparse):
    _, param = _linear(sparse)
    with torch.no_grad():
        param.M.copy_(torch.linspace(-.7, .9, param.M.numel()).reshape_as(param.M))
    original, codes = param.M.clone(), param.codes().clone()
    indices = torch.tensor([0, 1, 2]) if sparse else torch.tensor([4, 5, 6])
    transaction = CodeTransaction([(param, indices, torch.tensor([-1., 0., 1.]))])
    accepted, after = transaction.evaluate(lambda: 3., 2.)
    assert not accepted and after == 3.
    assert torch.equal(param.M, original) and torch.equal(param.codes(), codes)
    with pytest.raises(RuntimeError, match="failed trial"):
        transaction.evaluate(lambda: (_ for _ in ()).throw(RuntimeError("failed trial")), 2.)
    assert torch.equal(param.M, original)
    accepted, after = transaction.evaluate(lambda: 1., 2.)
    assert accepted and after == 1.
    assert torch.equal(param.M.flatten()[indices], torch.tensor([-1., 0., 1.]))
    transaction.rollback()
    assert torch.equal(param.M, original)


@pytest.mark.parametrize("sparse", [False, True])
def test_routed_gradients_include_weights_and_repeated_forward_inputs(sparse):
    linear, param = _linear(sparse)
    optimizer = MLSOptimizer([(linear, param)])
    x1, x2 = torch.randn(3, 8), torch.randn(2, 8)
    weights = torch.tensor([.2, .7, 0.])[:, None]
    with optimizer.record_facts():
        ((linear(x1) * weights).sum() + (linear(x2) * .5).sum()).backward()
    records = optimizer.facts[param]
    assert sum(len(x) for x, _ in records) == 4
    for x, g in records:
        if len(x) == 2 and torch.equal(x, x1[:2]):
            torch.testing.assert_close(g, weights[:2].expand(-1, 2))
        else:
            assert torch.equal(x, x2)
            torch.testing.assert_close(g, torch.full((2, 2), .5))
    assert not linear._forward_hooks


@pytest.mark.parametrize("sparse", [False, True])
def test_optimizer_proposal_can_kill_flip_and_birth(sparse):
    linear, param = _linear(sparse, budgeted=True)
    current = torch.zeros(2, 8)
    current[:, 4:6] = 1
    with torch.no_grad():
        param.M.copy_(current.flatten()[param.idx.long()] if sparse else current)
    desired = current.clone()
    desired[:, 4] = 0
    desired[:, 5] = -1
    desired[:, 6] = 1
    optimizer = MLSOptimizer([(linear, param)], eta=2., gen_lambda=0., gamma=1e-6, max_change=10)
    optimizer.general[param] = torch.zeros(8), torch.empty(8, 0)
    optimizer.facts[param] = [(torch.eye(8), -(desired - current).T / 2)]
    transaction, info = optimizer.propose()
    assert (info["births"], info["deaths"], info["flips"]) == (2, 2, 2)
    assert info["predicted_change"] < 0
    transaction.apply()
    assert torch.equal(param.codes(), desired)
    transaction.rollback()
    assert torch.equal(param.codes(), current)


@pytest.mark.parametrize("cap", [1, 2])
def test_global_budget_and_change_cap_cannot_split_an_exchange(cap):
    linear, param = _linear()
    param.M.data[0, 4] = 1
    optimizer = MLSOptimizer([(linear, param)], eta=1., budget=1, max_change=cap)
    # A birth with gain=3 ranks above its required death with gain=1.
    proposal = (param, torch.tensor([4, 5]), torch.tensor([1., 0.]), torch.tensor([0., 1.]),
                torch.tensor([.1, 2.]), torch.tensor([1., -3.]), torch.ones(2), torch.zeros(2))
    transaction, info = optimizer._bound([proposal])
    transaction.apply()
    assert param.codes().count_nonzero() <= 1
    assert info["changes"] <= cap
    assert info["changes"] == (0 if cap == 1 else 2)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_auto_eta_tiny_continuous_solution_yields_global_target_changes(sparse, tied):
    linears = [_linear(sparse), _linear(sparse)]
    optimizer = MLSOptimizer(linears, target_changes=5, gen_lambda=0., gamma=.01)
    x = torch.eye(8) * 1e4
    ratios, predictions = [], []
    for i, (_, param) in enumerate(linears):
        param.alpha.copy_(torch.tensor([[2.], [.25]]))
        if sparse:
            param.scale.copy_(param.alpha[:, 0][param.rows()])
        _, mask = optimizer._workspace(param)
        ratio = (torch.ones(2, 8) if tied else torch.arange(i * 16 + 1, i * 16 + 17).reshape(2, 8)) * 1e-6
        direction = ratio * param.alpha * mask
        gradient = -(1e4 + .01 / 1e4) * direction.T
        optimizer.general[param] = torch.zeros(8), torch.empty(8, 0)
        optimizer.facts[param] = [(x, gradient)]
        unit = masked_cg(x, -gradient, mask, torch.zeros(8), torch.empty(8, 0), 0., .01)
        ratios.append((unit.abs() / param.alpha)[mask])
        output = x @ unit.T
        predictions.append(float((gradient * output).sum() + .5 *
                                 (output.square().sum() + .01 * unit.square().sum())))
    # The old continuous trust calibration cannot cross even one birth threshold.
    old_eta = .01 / -sum(predictions)
    assert float(torch.cat(ratios).max()) * old_eta < .5
    transaction, info = optimizer.propose()
    assert info["target_changes"] == info["pre_correction_changes"] == info["changes"] == 5
    assert info["births"] == 5 and info["deaths"] == info["flips"] == 0
    assert info["u_over_alpha_quantiles"]["max"] > .5
    transaction.apply()
    actual = torch.cat([param.codes()[optimizer._workspace(param)[1]] for _, param in linears]).bool()
    if tied:
        assert actual[:5].all() and not actual[5:].any()
    else:
        expected = torch.zeros_like(actual)
        expected[torch.cat(ratios).topk(5).indices] = True
        assert torch.equal(actual, expected)
    transaction.rollback()
    # A new gradient scale needs a new eta, even at the same N.
    previous_eta = optimizer.eta
    for _, param in linears:
        optimizer.facts[param] = [(x, optimizer.facts[param][0][1] * .25)]
    _, next_info = optimizer.propose()
    assert optimizer.eta == pytest.approx(previous_eta * 4)
    assert next_info["changes"] == 5


@pytest.mark.parametrize("sparse", [False, True])
def test_auto_eta_counts_deaths_flips_and_births_once(sparse):
    linear, param = _linear(sparse)
    current = torch.zeros(2, 8)
    current[0, 4:6], current[1, 4:6] = 1., -1.
    with torch.no_grad():
        param.M.copy_(current.flatten()[param.idx.long()] if sparse else current)
    direction = torch.zeros_like(current)
    direction[0, 4:7] = torch.tensor([-1., -4., 2.]) * 1e-6
    direction[1, 4:7] = torch.tensor([2.5, -9., -.1]) * 1e-6
    optimizer = MLSOptimizer([(linear, param)], target_changes=4, gen_lambda=0., gamma=.01)
    optimizer.general[param] = torch.zeros(8), torch.empty(8, 0)
    optimizer.facts[param] = [(torch.eye(8), -(1.01 * direction).T)]
    transaction, info = optimizer.propose()
    assert info["pre_correction_changes"] == info["changes"] == 4
    assert (info["births"], info["deaths"], info["flips"]) == (1, 2, 1)
    transaction.apply()
    expected = current.clone()
    expected[0, 4:7] = torch.tensor([0., -1., 1.])
    expected[1, 4] = 0
    assert torch.equal(param.codes(), expected)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("threshold", [.25, .75])
def test_auto_eta_ranks_birth_and_death_boundaries_at_nondefault_threshold(sparse, threshold):
    linear, param = _linear(sparse)
    current = torch.zeros(2, 8)
    current[0, 4:6], current[1, 4:6] = 1., -1.
    with torch.no_grad():
        param.M.copy_(current.flatten()[param.idx.long()] if sparse else current)
    direction = torch.zeros_like(current)
    direction[0, 4:8] = torch.tensor([-1., 100., 2., .5]) * 1e-6
    direction[1, 4:8] = torch.tensor([2.5, -100., -3., 0.]) * 1e-6
    optimizer = MLSOptimizer([(linear, param)], target_changes=3, threshold=threshold,
                             gen_lambda=0., gamma=.01)
    optimizer.general[param] = torch.zeros(8), torch.empty(8, 0)
    optimizer.facts[param] = [(torch.eye(8), -(1.01 * direction).T)]
    transaction, info = optimizer.propose()
    assert info["pre_correction_changes"] == info["changes"] == 3
    expected = current.clone()
    expected[1, 4], expected[1, 6] = 0., -1.
    # Births lead at the lower threshold; deaths lead at the higher one.
    if threshold == .25:
        expected[0, 6] = 1.
        assert (info["births"], info["deaths"]) == (2, 1)
    else:
        expected[0, 4] = 0.
        assert (info["births"], info["deaths"]) == (1, 2)
    transaction.apply()
    assert torch.equal(param.codes(), expected)
    transaction.rollback()
    assert torch.equal(param.codes(), current)


def test_auto_target_adapts_with_floor_and_initial_cap():
    optimizer = MLSOptimizer([], target_changes=2000)
    for expected in (1000, 500, 500):
        optimizer.adapt(False)
        assert optimizer.target_changes == expected
    optimizer.adapt(True)
    assert optimizer.target_changes == 625
    for _ in range(10):
        optimizer.adapt(True)
    assert optimizer.target_changes == 2000
    small = MLSOptimizer([], target_changes=5)
    small.adapt(False)
    assert small.target_changes == 5


@pytest.mark.parametrize("mode", ["zero", "no_facts", "capped", "unreachable_target"])
def test_auto_eta_empty_directions_and_existing_change_cap(mode):
    linear, param = _linear()
    optimizer = MLSOptimizer([(linear, param)], target_changes=5 if mode == "capped" else 20,
                             gen_lambda=0., max_change=2 if mode == "capped" else 50)
    optimizer.general[param] = torch.zeros(8), torch.empty(8, 0)
    if mode != "no_facts":
        g = torch.zeros(8, 2) if mode == "zero" else -torch.ones(8, 2) * 1e-6
        optimizer.facts[param] = [(torch.eye(8), g)]
    _, info = optimizer.propose()
    assert info["changes"] == (2 if mode == "capped" else 8 if mode == "unreachable_target" else 0)
    assert info["pre_correction_changes"] == (5 if mode == "capped" else 8 if mode == "unreachable_target" else 0)
    assert all(torch.isfinite(torch.tensor(v)) for v in info["u_over_alpha_quantiles"].values())


@pytest.mark.parametrize("flags, error", [
    (["--method", "fsa_free"], "forced-mask"),
    (["--fmt", "fp4"], "plain ternary"),
    (["--mls-eta", "nan"], "finite"),
    (["--mls-gamma", "0"], "finite"),
    (["--mls-gen-every", "0"], "limits"),
    (["--mls-target-changes", "0"], "limits"),
    (["--mls-thr", "1"], "between 0 and 1"),
    (["--mls-rank", "-1"], "rank"),
    (["--rehearsal-frac", ".1"], "rehearsal-frac"),
    (["--teacher-patch", "teacher.pt"], "teacher-patch"),
])
def test_mls_cli_rejects_incompatible_options(flags, error):
    args = E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused", "--optim", "mls",
                                       "--anchor", "anchor.json", *flags])
    with pytest.raises(SystemExit, match=error):
        E.validate_mls_args(args)


@pytest.mark.parametrize("sparse", [False, True])
def test_main_mls_same_batch_rollback_eta_refresh_revoke_and_saturation(cpu_harness, monkeypatch, sparse):
    run, paths = cpu_harness
    phases, forwards, refreshes, observations = [], {}, [], []
    old_record, old_eval = MLSOptimizer.record_facts, CodeTransaction.evaluate
    old_logits, old_refresh = E.logits_of, MLSOptimizer.refresh_general
    old_observe = E.SaturationSampler.observe

    @contextmanager
    def record(optimizer):
        phases.append("fact")
        forwards["fact"] = []
        with old_record(optimizer):
            yield
        phases.pop()

    def evaluate(transaction, loss, before):
        phases.append("trial")
        forwards["trial"] = []
        saved = [(p, p.M.clone()) for p, _, _ in transaction.edits]
        # Force rejection even if the proposed step helps.
        result = old_eval(transaction, lambda: loss() + 1000, before)
        phases.pop()
        assert all(torch.equal(p.M, original) for p, original in saved)
        # The final forward in each phase is the local anchor penalty, which
        # is only in trial. Compare all fact microbatches exactly.
        assert len(forwards["fact"]) == len(forwards["trial"]) - 1
        assert all(torch.equal(a, b) for a, b in zip(forwards["fact"], forwards["trial"]))
        return result

    def logits(model, toks):
        if phases:
            forwards[phases[-1]].append(toks.clone())
        return old_logits(model, toks)

    def refresh(optimizer, forward, windows):
        assert windows.shape[0] == 32
        refreshes.append(1)
        assert all(lin is not None and (not sparse or p.sparse) for lin, p in optimizer.linears)
        return old_refresh(optimizer, forward, windows)

    def observe(sampler, fi, logits, gold, step):
        observations.append(step)
        return old_observe(sampler, fi, logits, gold, step)

    monkeypatch.setattr(MLSOptimizer, "record_facts", record)
    monkeypatch.setattr(CodeTransaction, "evaluate", evaluate)
    monkeypatch.setattr(MLSOptimizer, "refresh_general", refresh)
    monkeypatch.setattr(E, "logits_of", logits)
    monkeypatch.setattr(E.SaturationSampler, "observe", observe)
    flags = ["--optim", "mls", "--bop-tau-abs", "0", "--anchor", paths["anchor"], "--anchor-len", "8",
             "--steps", "3", "--mls-gen-every", "2", "--mls-eta", "100", "--mls-rank", "2",
             "--mls-max-change", "5", "--saturation-sampling", "--aug", "--down-only", "--expert-checkpoint"]
    result, patch = run(flags + (["--sparse-state"] if sparse else []), "mls_rollback")
    assert len(refreshes) == 2
    assert len(observations) == 27   # three facts * three phrasings * three backward passes
    logs = result["mls"]["iterations"]
    assert [row["eta"] for row in logs] == [100., 50., 25.]
    assert all(row["target_changes"] == row["next_target_changes"] == 20000 for row in logs)
    assert all(set(row["u_over_alpha_quantiles"]) == {"p50", "p90", "p99", "p99_9", "max"} for row in logs)
    assert all(not row["accepted"] and row["changes"] <= 5 for row in logs)
    assert result["flipped"] == 0 and result["revoked"]["logits_bit_exact"]
    assert [row["step"] for row in result["trajectory"]] == [2, 3]
    state = torch.load(patch, weights_only=False)
    assert all(e["idx"].numel() == 0 for e in state["entries"].values())


def test_main_mls_acceptance_budget_up_down_and_patch_reload(cpu_harness):
    run, paths = cpu_harness
    flags = ["--optim", "mls", "--bop-tau-abs", "0", "--anchor", paths["anchor"], "--anchor-len", "8",
             "--steps", "3", "--mls-eta", "1000", "--mls-gen-lambda", "0", "--mls-rank", "0",
             "--sparse-state", "--mls-max-change", "10"]
    result, patch = run(flags, "mls_accepted")
    assert any(row["accepted"] and row["births"] for row in result["mls"]["iterations"])
    assert 0 < result["flipped"] <= 20 and result["revoked"]["logits_bit_exact"]
    state = torch.load(patch, weights_only=False)
    assert {name.rsplit(".", 1)[-1] for name in state["entries"]} == {"up_proj", "down_proj"}
    assert all(set(entry["val"].tolist()) <= {-1, 1} for entry in state["entries"].values())
    loaded, _ = run([*flags, "--steps", "0", "--load-patch", str(patch)], "mls_reloaded")
    assert loaded["patched"] == result["patched"]
    assert loaded["revoked"]["logits_bit_exact"]


def test_main_mls_automatic_step_logs_target_and_scale(cpu_harness):
    run, paths = cpu_harness
    result, _ = run(["--optim", "mls", "--bop-tau-abs", "0", "--anchor", paths["anchor"],
                     "--anchor-len", "8", "--steps", "2", "--mls-gen-lambda", "0", "--mls-rank", "0",
                     "--mls-target-changes", "5", "--mls-max-change", "5", "--sparse-state"], "mls_auto")
    assert result["mls"]["target_changes"] == 5
    logs = result["mls"]["iterations"]
    assert all(row["eta"] > 0 and row["target_changes"] == row["next_target_changes"] == 5 for row in logs)
    assert all(row["pre_correction_changes"] == 5 and row["changes"] <= 5 for row in logs)
    assert all(row["u_over_alpha_quantiles"]["max"] > .5 for row in logs)


def test_streaming_selection_matches_global_confidence_and_gain():
    linears = [_linear() for _ in range(4)]
    gen = torch.Generator().manual_seed(145)
    proposals = []
    for _, param in linears:
        indices = torch.tensor([4, 5, 6, 7])
        proposals.append((param, indices, torch.zeros(4), torch.ones(4),
                          torch.rand(4, generator=gen), -torch.rand(4, generator=gen),
                          torch.ones(4), torch.zeros(4)))
    for budget in (0, 3):
        optimizer = MLSOptimizer(linears, eta=1., budget=budget, max_change=3)
        streamed = []
        for proposal in proposals:
            streamed = optimizer._trim([*streamed, proposal])
            assert sum(p[1].numel() for p in streamed) <= 3
        transaction, _ = optimizer._bound(streamed)
        all_confidence = torch.cat([p[4] for p in proposals])
        all_gain = -torch.cat([p[5] for p in proposals])
        expected = (all_confidence if budget else all_gain).topk(3).indices.tolist()
        actual = []
        for param, indices, _ in transaction.edits:
            module = next(i for i, (_, p) in enumerate(linears) if p is param)
            actual.extend(module * 4 + slot - 4 for slot in indices.tolist())
        assert set(actual) == set(expected)


def test_main_inactive_mls_flags_preserve_patch_and_rng(cpu_harness):
    import random
    run, _ = cpu_harness
    ordinary, first = run(["--steps", "2"], "mls_inactive_default")
    python_rng, torch_rng = random.getstate(), torch.get_rng_state()
    altered, second = run(["--steps", "2", "--mls-trust", ".2", "--mls-rank", "0",
                           "--mls-eta", "1000", "--mls-thr", ".7"], "mls_inactive_flags")
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert altered["patched"] == ordinary["patched"]
    first, second = [torch.load(path, weights_only=False)["entries"] for path in (first, second)]
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name]["idx"], second[name]["idx"])
        assert torch.equal(first[name]["val"], second[name]["val"])
