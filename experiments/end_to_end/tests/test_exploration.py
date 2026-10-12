"""C30 mechanics, kill boundaries, and frozen-bank CPU end-to-end coverage."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_e2e_fsa import E, TOK, Tiny, tiny, cpu_harness  # noqa: F401
import branch_eval as B
import exploration as C


@pytest.fixture(autouse=True)
def restore_patch_enabled_state():
    enabled = E.FSAParam.enabled, E.LoRAParam.enabled
    yield
    E.FSAParam.enabled, E.LoRAParam.enabled = enabled


def training():
    x = torch.eye(12)[:3].repeat(4, 1)*4
    return x, [2, 5, 9]*4, ["a"]*12, torch.zeros(64, 12)


@pytest.mark.parametrize("arm", C.SPECS)
def test_controls_pass_without_changing_global_rng(arm):
    rng = torch.get_rng_state()
    result = C.positive_control(arm)
    assert result["passed"]
    assert torch.equal(rng, torch.get_rng_state())


def test_equal_bytes_dense_is_I_and_counts_projection_and_codebooks():
    x, labels, attrs, g = training()
    index = C.VSASelector.fit(x, labels, attrs, g, dimensions=64, seed=17)
    assert index.project.dtype == torch.int8
    assert set(index.project.flatten().tolist()) <= {-1, 0, 1}
    assert set(index.keys.flatten().tolist()) <= {-1, 1}
    assert index.storage() >= C.tensor_bytes(index.project, index.people, index.attributes_code,
                                            index.keys, index.center)
    dense = C.DenseSelector.fit(x, labels, attrs, g, byte_budget=index.storage())
    assert dense.storage() == index.storage()
    assert dense.useful_bytes < dense.storage()
    torch.testing.assert_close(dense.scores(x), B.BankIndexer.fit(x, labels, g).scores(x))
    # The cap can also force a smaller dense key rather than hiding extra memory.
    small = C.DenseSelector.fit(x+1, labels, attrs, g, byte_budget=100)
    assert small.width == 2 and small.storage() == 100
    with pytest.raises(ValueError):
        C.DenseSelector.fit(x, labels, attrs, g, byte_budget=1)
    attrs = ["a", "b"]*6
    index = C.VSASelector.fit(x, labels, attrs, g, 128, 17)
    codes = index.encode(x)
    for i, pid in enumerate(index.owners):
        expected = C.bipolar(torch.stack([codes[j]*index.attributes_code[index.attributes.index(attrs[j])]
                                         for j in range(len(x)) if labels[j] == pid]).float().sum(0))*index.people[i]
        assert torch.equal(index.keys[i], expected)
    assert index.scores(x).shape == (12, 3)


def test_phase_coherence_invariant_to_global_phase_and_has_three_cleanup_iterations():
    x, labels, attrs, g = training()
    index = C.PhaseSelector.fit(x, labels, attrs, g, 128, 9)
    assert index.iterations == 3
    torch.testing.assert_close(index.keys.abs(), torch.ones_like(index.keys.real))
    expected = index.scores(x)
    index.keys *= torch.polar(torch.tensor(1.), torch.tensor(.7))
    torch.testing.assert_close(index.scores(x), expected)
    assert index.select(x) == [(p,) for p in labels]
    index.iterations = 0
    assert not torch.allclose(index.scores(x), expected)


def test_kuramoto_locking_beats_decoy_and_abstains_on_training_null():
    x, labels, attrs, g = training()
    index = C.KuramotoSelector.fit(x, labels, attrs, g)
    assert index.steps == 50
    assert index.select(x) == [(p,) for p in labels]
    assert not any(index.select(g))
    assert torch.isfinite(index.scores(x*.01)).all()
    softmax = C.DenseSelector.fit(x, labels, attrs, g, softmax=True)
    assert softmax.select(x) == [(p,) for p in labels]


def test_cp_reconstructs_four_way_usage_and_does_not_retain_dense_tensor():
    rng = torch.Generator().manual_seed(7)
    factors = [torch.randn(n, 2, generator=rng) for n in (3, 4, 2, 8)]
    usage = C.cp_reconstruct(factors)
    fitted = C.cp_fit(usage, rank=2, steps=80, seed=13)
    assert float((C.cp_reconstruct(fitted)-usage).square().mean()) < 1e-5
    x, labels, attrs, g = training()
    x, g = x[:, None].repeat(1, 2, 1), g[:, None].repeat(1, 2, 1)
    index = C.TensorSelector.fit(x, labels, attrs, g, rank=3, steps=40, units=12, seed=11)
    assert index.reconstruct().shape == (3, 1, 2, 12)
    assert index.select(x) == [(p,) for p in labels]
    assert all(not torch.is_tensor(v) or v.ndim < 4 for v in vars(index).values())
    with pytest.raises(ValueError):
        C.cp_fit(usage, rank=0)


def test_query_event_calibration_is_strict_and_ties_use_owner_order():
    scores = torch.zeros(100, 8, dtype=torch.double)
    scores[-1, -1] = 1
    assert C.null_threshold(scores) == 0
    routes = C.routes_from_scores(scores, tuple(range(8)), 0)
    assert sum(bool(r) for r in routes) == 1
    assert C.routes_from_scores(torch.ones(1, 3), (2, 5, 9), 0, 2) == [(2, 5)]
    assert C.routes_from_scores(torch.ones(1, 3), (2, 5, 9), 1) == [()]
    with pytest.raises(ValueError):
        C.null_threshold(torch.tensor([[math.nan]]))
    with pytest.raises(ValueError):
        C.routes_from_scores(scores, tuple(range(8)), 0, 9)


@pytest.fixture
def banks():
    model = tiny()
    E.attach(model, [0], "fsa_free", 0, sparse=True)
    with torch.no_grad():
        for _, lin in E.patched_linears(model):
            lin.parametrizations.weight[0].M.fill_(1)
    state = E.export_patch(model)
    return B.Branches(model, {2: state, 5: copy.deepcopy(state)}, {2: 10, 5: 10}, "sum_clip")


def test_signed_collision_is_exact_double_weight_and_cleanup_survives_failure(banks):
    original = copy.deepcopy(banks.states)
    patch = C.signed_patch(banks.states, {2: 1, 5: 1})
    cancellation = C.signed_patch(banks.states, {2: 1, 5: -1})
    assert all(e["val"].numel() == 0 for e in cancellation["entries"].values())
    E.set_enabled(False)
    with pytest.raises(RuntimeError, match="forward failed"):
        with C.weighted_forward(banks, patch):
            for _, lin in E.patched_linears(banks.model):
                pm = lin.parametrizations.weight[0]
                assert E.FSAParam.enabled
                assert bool((pm.M == 2).all())
                delta = pm.delta()
                expected = torch.zeros_like(delta).flatten()
                expected[pm.idx.long()] = 2*pm.scale
                torch.testing.assert_close(delta, expected.reshape(delta.shape))
            raise RuntimeError("forward failed")
    assert not E.FSAParam.enabled
    for _, lin in E.patched_linears(banks.model):
        pm = lin.parametrizations.weight[0]
        assert "_quant" not in pm.__dict__ and not pm.M.any()
    for pid, state in banks.states.items():
        for name, entry in state["entries"].items():
            assert torch.equal(entry["val"], original[pid]["entries"][name]["val"])
    banks.activate((2,))
    assert E.FSAParam.enabled
    for _, lin in E.patched_linears(banks.model):
        assert bool((lin.parametrizations.weight[0].M == 1).all())
    with pytest.raises(ValueError):
        C.signed_patch(banks.states, {2: 1})


def test_interference_decodes_signed_forwards_plus_base_without_using_gold_answer(banks, monkeypatch):
    counts = []
    def logits(model, tokens):
        counts.append(True)
        pm = E.patched_linears(model)[0][1].parametrizations.weight[0]
        result = torch.zeros(1, tokens.shape[1], 128)
        result[0, -1, 42] = pm.M[0]
        return result
    monkeypatch.setattr(E, "logits_of", logits)
    items = [dict(p=[1, 40], a=[99], vals=["never read"])]
    output, forwards = C.interference_generate(banks, items, [(2,)], "cpu", 1, 7, math.inf)
    assert output == [[42]] and forwards == len(counts) == 3
    assert all(not lin.parametrizations.weight[0].M.any() for _, lin in E.patched_linears(banks.model))


def result(accuracy=.6, false=.1, recall=.5):
    return dict(address_accuracy=accuracy, false_activations=dict(query_rate=false), composed_recall=recall)


@pytest.mark.parametrize("arm", ("vsa", "phase", "tensor"))
def test_dense_kill_requires_strict_improvement(arm):
    assert C.kill_decision(arm, dict(pilot=result(), baseline=result()))["verdict"] == "kill"
    assert C.kill_decision(arm, dict(pilot=result(.61), baseline=result()))["verdict"] == "survive"


def test_kill_boundary_five_points_false_doubling_and_oracle_floor():
    report = dict(pilot=result(.65, .2), baseline=result(.6, .1))
    assert C.kill_decision("kuramoto", report)["verdict"] == "survive"
    report["pilot"] = result(.649, .2)
    assert C.kill_decision("kuramoto", report)["verdict"] == "kill"
    report["pilot"] = result(.7, .201)
    assert C.kill_decision("kuramoto", report)["verdict"] == "kill"
    report = dict(pilot=result(recall=.4), oracle=result(recall=.8))
    assert C.kill_decision("interference", report)["verdict"] == "survive"
    report["pilot"] = result(recall=.399)
    assert C.kill_decision("interference", report)["verdict"] == "kill"
    report["oracle"] = result(recall=0)
    assert C.kill_decision("interference", report)["verdict"] == "inconclusive"


def test_control_failure_reports_no_verdict_and_skips_model_io(monkeypatch, capsys):
    monkeypatch.setattr(C, "positive_control", lambda arm: dict(passed=False, kind="injected failure"))
    monkeypatch.setattr(C, "load_runtime", lambda args: pytest.fail("control failed before model IO"))
    reports = C.main(["--arm", "phase", "--patches", "missing.pt", "--anchor", "missing.json",
                      "--eval-set", "missing.json", "--fineweb", "missing.json", "--code", "missing.json"])
    assert reports[0]["verdict"] == "control-failed" and reports[0]["result"] is None
    output = capsys.readouterr().out
    assert output.index("PRIOR ") < output.index("POSITIVE-CONTROL ") < output.index("EXPLORE ")


def test_control_only_cli_and_invalid_sizes(capsys):
    reports = C.main(["--arm", "all", "--control-only"])
    assert len(reports) == 5 and all(r["verdict"] == "control-passed" for r in reports)
    lines = [json.loads(line.removeprefix("EXPLORE ")) for line in capsys.readouterr().out.splitlines()
             if line.startswith("EXPLORE ")]
    assert {r["arm"] for r in lines} == set(C.SPECS)
    for extra in (["--gen-bs", "2"], ["--max-seconds", "nan"], ["--qa-n", "0"], ["--dimensions", "2"]):
        with pytest.raises(SystemExit):
            C.main(["--arm", "vsa", "--control-only", *extra])


def test_all_arms_real_loader_train_only_keys_read_only_banks_and_reports(cpu_harness, monkeypatch, tmp_path, capsys):
    run, paths = cpu_harness
    monkeypatch.setattr(Tiny, "layers", property(lambda self: self.moes), raising=False)
    patches = []
    for person in range(3):
        _, patch = run(["--steps", "1", "--train-person", str(person), "--sparse-state", "--slot-frac", ".5"],
                       f"explore{person}")
        patches.append(str(patch))
    originals = {p: Path(p).read_bytes() for p in patches}
    anchor = tmp_path/"explore_anchor.json"
    anchor.write_text(json.dumps([40]*8000))
    fitting = C.fit_selector
    fit_calls, loaded = [], []
    def fit(arm, train, labels, attrs, general, args):
        assert len(train) == 3*4 and len(general) == 64
        assert len(labels) == len(attrs) == len(train)
        assert set(labels) == {0, 1, 2}
        fit_calls.append(arm)
        return fitting(arm, train, labels, attrs, general, args)
    monkeypatch.setattr(C, "fit_selector", fit)
    loading = C.load_runtime
    def load(args):
        runtime = loading(args)
        loaded.append(runtime[0])
        assert all(not p.requires_grad for p in runtime[0].model.parameters())
        return runtime
    monkeypatch.setattr(C, "load_runtime", load)
    # Inject train-direction queries to exercise real two-forward decoding even
    # when this random tiny model's learned addressing would otherwise abstain.
    collecting = C.features_for_run
    def collect(*args):
        train, direct, composed, ff = collecting(*args)
        for per_layer in ff.values():
            assert set(per_layer) == {0, 1}
        for name in ("direct", "composed"):
            ff[name][1] = ff["train"][1][0:1].repeat(len(ff[name][1]), 1)
        return train, direct, composed, ff
    monkeypatch.setattr(C, "features_for_run", collect)
    capsys.readouterr()
    reports = C.main(["--arm", "all", "--patches", *patches, "--anchor", str(anchor),
                      "--eval-set", paths["eval"], "--fineweb", paths["extra"], "--code", paths["extra"],
                      "--fast-moe", "off", "--indexer-layer", "1", "--tensor-layers", "0,1",
                      "--train-per-bank", "4", "--dimensions", "32", "--cp-steps", "5",
                      "--qa-n", "2", "--compose-n", "2", "--gen-max-new", "1"])
    assert fit_calls == list(C.SPECS)
    assert len(reports)==5 and all(r["verdict"]=="control-failed" for r in reports)
    assert all(r["result"]["zero_patch_control"]["passed"] for r in reports)
    assert all(Path(p).read_bytes() == v for p, v in originals.items())
    for report in reports:
        pilot = report["result"]["pilot"]
        assert pilot["address"]["n"] == 2 and pilot["n_composed"] == 2
        assert all(v["n"] == 40 for v in pilot["false_activations_by_source"].values())
        assert set(pilot["false_activations_by_source"]) == {"WikiText", "FineWeb", "code"}
        assert pilot["resident_banks"] == [0, 1, 2]
        assert report["wall_seconds"] >= report["result"]["shared_setup_seconds"]
        assert 0 <= pilot["composed_recall"] <= 1
    vsa = reports[0]["result"]
    assert vsa["pilot"]["address_bytes"] == vsa["baseline"]["address_bytes"]
    assert reports[3]["result"]["pilot"]["contrast_forwards"] > 0
    assert not E.FSAParam.enabled
    assert all(not layer._forward_pre_hooks for layer in loaded[0].model.layers)
    output = capsys.readouterr().out
    assert output.index("POSITIVE-CONTROL ") < output.index("EXPLORE-SHARED ")
    assert len([line for line in output.splitlines() if line.startswith("EXPLORE ")]) == 5
    json.dumps(reports, allow_nan=False)


def test_expired_budget_abstains_from_evaluation_and_restores_bank(banks, capsys):
    args = SimpleNamespace(indexer_layer=0, max_seconds=.01)
    data = ([], [], [], {"train": {0: torch.zeros(1, 2)}})
    report = C.run_arm("vsa", (banks, None, [], {}, [], "cpu"), data, args, [0],
                       dict(passed=True), C.time.perf_counter()-1)
    assert report["verdict"] == "budget-exceeded" and report["result"] is None
    assert not E.FSAParam.enabled
