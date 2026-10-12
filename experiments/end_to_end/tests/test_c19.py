"""C19 CPU diagnostics: unit ownership, initialization, encoding, residency."""
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_e2e_fsa import E, TOK, BOS_T, _c3_pair8_model, cpu_harness
from test_branch_eval import B


def args(*flags):
    return E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused", *flags])


def test_defaults_and_invalid_allocations():
    a = args()
    assert (a.units_per_fact, a.unit_init) == (0, "borrowed")
    E.validate_c19_args(a)
    for flags in (["--units-per-fact", "-1"], ["--unit-init", "zero"],
                  ["--units-per-fact", "2", "--down-only"],
                  ["--units-per-fact", "2", "--fmt", "fp4"],
                  ["--units-per-fact", "2", "--optim", "mls"],
                  ["--units-per-fact", "2", "--address-units", "2"]):
        with pytest.raises(SystemExit):
            E.validate_c19_args(args(*flags))


@pytest.mark.parametrize("sparse", [False, True])
def test_selection_answer_position_deterministic_collisions_and_cleanup(sparse):
    model = _c3_pair8_model()
    E.attach(model, [0, 1], "fsa_forced", 0, sparse=sparse)
    facts = E.make_world(7, 1, 2, 0)
    pool = [[E.encode_fact(TOK, E.TRAIN_T[0], f, BOS_T)] for f in facts]
    # Repeated identical facts guarantee collisions, independent of tie breaking.
    pool.append(pool[0])
    rng, prng = torch.get_rng_state(), random.getstate()
    backwards = []
    handle = model.head.register_full_backward_hook(lambda *a: backwards.append(1))
    report = E.select_fact_units(model, pool, [0, 1], 2, "cpu")
    handle.remove()
    assert len(backwards) == len(pool)
    assert report["assignments"]["0"] == report["assignments"]["2"]
    assert report["collision_units"] > 0
    assert all(len(rows) <= 2 for fact in report["assignments"].values() for rows in fact.values())
    # Independently obtain the routed expert at the answer boundary from a prefix.
    for fi, examples in enumerate(pool):
        seq, start = examples[0]
        with torch.no_grad(), E.wake_prompt_inputs(model, [0, 1]) as records:
            model(torch.tensor([seq[:start]]))
        for bank in report["assignments"][str(fi)]:
            layer = int(bank.split(".")[1])
            expert = int(bank.rsplit(".", 1)[1])
            assert expert in records[layer][1]
    assert torch.equal(rng, torch.get_rng_state()) and prng == random.getstate()
    assert all(p.grad is None for p in model.parameters())
    assert all(not lin._forward_hooks for _, lin in E.patched_linears(model))
    assert report == E.select_fact_units(model, pool, [0, 1], 2, "cpu")


@pytest.mark.parametrize("sparse", [False, True])
def test_union_mask_and_per_fact_gradient_cannot_write_other_units(sparse):
    model = _c3_pair8_model()
    E.attach(model, [0], "fsa_forced", 0, sparse=sparse)
    bank = "moes.0.experts.0"
    assignments = {"0": {bank: [4]}, "1": {bank: [7]}}
    E.restrict_unit_masks(model, assignments)
    with E.fact_unit_gradients(model, assignments["0"]):
        sum(m.M.sum() for m in E.fsa_modules(model)).backward()
    for name, lin in E.patched_linears(model):
        m = lin.parametrizations.weight[0]
        expected = E.unit_slot_mask(m, [4] if name.startswith(bank + ".") else [], name.rsplit(".", 1)[1]) & m.mask
        assert torch.equal(m.M.grad, expected.float())
        m.M.data.fill_(1)  # Even a stale/hostile master cannot escape the physical mask.
        codes = m.codes()
        physical = m._dense(m.mask).bool() if sparse else m.mask
        assert not codes[~physical].any() and not m.delta()[~physical].any()
    assert all(not m.M._backward_hooks for m in E.fsa_modules(model))


def unit_model(sparse=False):
    model = nn.Module()
    expert = nn.Module()
    expert.up_proj = nn.Linear(8, 8, bias=False)
    expert.down_proj = nn.Linear(8, 8, bias=False)
    for lin in (expert.up_proj, expert.down_proj):
        lin.weight.data.copy_(torch.tensor([1., 1., -1., -1., 0., 0., 0., 0.]).repeat(8, 1))
        lin.weight.requires_grad_(False)
    moe = nn.Module()
    moe.experts = nn.ModuleList([expert])
    model.moes = nn.ModuleList([moe])
    E.attach(model, [0], "fsa_forced", 0, sparse=sparse)
    return model, expert


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("mode", ["borrowed", "random", "zero"])
def test_initialization_revoke_export_reload_and_zero_gradient_trap(sparse, mode):
    model, expert = unit_model(sparse)
    base = [lin.weight.detach().clone() for lin in (expert.up_proj, expert.down_proj)]
    rng = torch.get_rng_state()
    E.initialize_units(model, {"moes.0.experts.0": list(range(8))}, mode, 19)
    assert torch.equal(rng, torch.get_rng_state())
    up, down = [lin.parametrizations.weight[0] for lin in (expert.up_proj, expert.down_proj)]
    if mode == "borrowed":
        assert torch.equal(expert.up_proj.weight, base[0]) and torch.equal(expert.down_proj.weight, base[1])
    else:
        assert not expert.down_proj.weight.any()
        if mode == "zero":
            assert not expert.up_proj.weight.any()
        else:
            codes = up.codes()
            assert set(codes.unique().tolist()) == {-1., 0., 1.}
            assert 0 < int(codes.ne(0).sum()) < int(up.mask.sum())
        # A live random address learns values; zero/zero relu² has zero gradients.
        x = torch.tensor([[0., 0., 0., 0., 1., 2., 3., 4.]])
        expert.down_proj(torch.relu(expert.up_proj(x)).square()).sum().backward()
        assert not up.M.grad.any()
        assert bool(down.M.grad.any()) == (mode == "random")
    saved = E.export_patch(model)
    clone, clone_expert = unit_model(sparse)
    E.load_patch(clone, saved)
    assert torch.equal(expert.up_proj.weight, clone_expert.up_proj.weight)
    assert torch.equal(expert.down_proj.weight, clone_expert.down_proj.weight)
    E.set_enabled(False)
    try:
        assert torch.equal(expert.up_proj.weight, base[0]) and torch.equal(expert.down_proj.weight, base[1])
    finally:
        E.set_enabled(True)


@pytest.mark.parametrize("sparse", [False, True])
def test_hand_counted_encoding(sparse):
    model, expert = unit_model(sparse)
    for lin, flat in ((expert.up_proj, 4 * 8 + 4), (expert.down_proj, 4)):
        m = lin.parametrizations.weight[0]
        idx = int((m.idx == flat).nonzero()[0]) if sparse else flat
        m.M.data.flatten()[idx] = 1
    packed = E.unit_bytes(model, {"bytes_sparse": 99})
    assert packed["used_units"] == 1 and packed["sparse_coordinate_bytes"] == 99
    assert packed["dense_packed_unit_bytes"] == (3 + 2 * (4 + 8)) / 8
    assert packed["block_sparse_unit_bytes"] == (3 + 2 * (1 + 16)) / 8


def test_merge_reset_metadata_and_empty_branch_restores_base():
    model, expert = unit_model()
    original = expert.up_proj.weight.detach().clone()
    states = {}
    for person in range(3):
        for m in E.fsa_modules(model):
            for axis in ("rows", "cols"):
                key = "unit_reset_" + axis
                if hasattr(m, key):
                    delattr(m, key)
        E.initialize_units(model, {"moes.0.experts.0": [person]}, "zero", person)
        states[person] = E.export_patch(model, dict(train_person=person))
    merged = B.merge_patches(list(states.values()))
    assert merged["entries"]["moes.0.experts.0.up_proj"]["unit_reset_rows"].tolist() == [0, 1, 2]
    branches = B.Branches(model, states, dict.fromkeys(states, 1), "sum_clip")
    branches.activate((0, 1, 2))
    assert not expert.up_proj.weight[:3].any()
    branches.activate((1,))
    assert torch.equal(expert.up_proj.weight[0], original[0]) and not expert.up_proj.weight[1].any()
    branches.activate(())
    assert torch.equal(expert.up_proj.weight, original)


def test_address_positives_all_other_people_and_all_general_tokens(monkeypatch):
    model, expert = unit_model()
    facts = E.make_world(0, 3, 1, 0)  # Other people's distinct attributes still count as negatives.
    items = [dict(fi=i, p=[i]) for i in range(3)]
    detectors = {"moes.0.experts.0.up_proj": dict(rows=[0])}
    monkeypatch.setattr(E, "address_fired", lambda model, p, *a: p[0] != 2)
    def forward(model, seq):
        # Three general tokens: first positive, zero and negative remain silent.
        expert.up_proj(torch.tensor([[1.] * 2 + [0.] * 6, [0.] * 8, [-1.] * 2 + [0.] * 6]))
    monkeypatch.setattr(E, "logits_of", forward)
    result = E.address_accuracy(model, facts, items, detectors, [[1, 2, 3]], "cpu", person=0)
    assert result["tpr"] == 1 and result["fpr"] == .5
    assert result["general_text"] == dict(tokens=3, unit_token_opportunities=3, fired=1, fpr=1 / 3)
    assert not expert.up_proj._forward_hooks


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("mode", ["borrowed", "random", "zero"])
def test_main_unit_training_diagnostics_reload(cpu_harness, capsys, sparse, mode):
    run, _ = cpu_harness
    flags = ["--units-per-fact", "2", "--unit-init", mode, "--budget", "0", "--train-person", "0",
             "--answer-only"] + (["--sparse-state"] if sparse else [])
    result, path = run([*flags, "--steps", "2"], f"units_{mode}_{sparse}")
    saved = torch.load(path, weights_only=False)
    assert result["unit_allocation"]["k"] == 2
    assert result["unit_bytes"]["k"] == 2 and "steps_to_learn" in result
    assert result["three_metrics"]["resident_people"] == [0]
    assert len(result["three_metrics"]["individual_recall"]["by_memory"]) == 2
    assert result["revoked"]["logits_bit_exact"]
    for name, entry in saved["entries"].items():
        bank, proj = name.rsplit(".", 1)
        allowed = {u for fact in result["unit_allocation"]["assignments"].values() for u in fact.get(bank, [])}
        indices = entry["idx"].long()
        units = indices // entry["shape"][1] if proj == "up_proj" else indices % entry["shape"][1]
        assert set(units.tolist()) <= allowed
    loaded, _ = run([*flags, "--steps", "0", "--load-patch", str(path)], "unit_reload")
    assert result["patched"] == loaded["patched"]
    assert result["unit_bytes"] == loaded["unit_bytes"]
    assert result["three_metrics"] == loaded["three_metrics"]
    output = capsys.readouterr().out
    assert all(prefix in output for prefix in ("UNIT-BYTES ", "THREE-METRICS "))


def test_branch_metrics_individual_and_all_resident_including_g_set(monkeypatch):
    facts = E.make_world(0, 3, 2, 2)
    items = E.build_items(TOK, facts, E.TEST_T[:1], BOS_T)
    g = B.composition_items(TOK, facts, BOS_T, 4)
    resident, seen = [], []
    branches = SimpleNamespace(states=dict.fromkeys(range(3)), model=None,
                               activate=lambda people: resident.__setitem__(slice(None), people),
                               address_detectors=lambda people: {})
    def recall(model, tok, items, *rest):
        seen.append((tuple(resident), len(items)))
        assert len(resident) in (1, 3)
        return dict(n=len(items), recall=1 if len(resident) == 1 else .25)
    def composition(model, tok, items, *rest):
        assert set(resident) == {0, 1, 2} and len(items) == 4
        assert all(len({facts[i].pid for i in it["fis"]}) == 2 for it in items)
        return dict(n=len(items), recall=.5)
    monkeypatch.setattr(E, "gen_eval", recall)
    monkeypatch.setattr(E, "composition_recall", composition)
    monkeypatch.setattr(E, "unit_bytes", lambda model: {})
    result = B.three_metrics(branches, TOK, facts, items, g, [], "cpu", 1, 2, 1)
    assert result["individual_recall"]["mean"] == 1
    assert result["composed_recall"] == dict(n=6, recall=.25, g_set=dict(n=4, recall=.5))
    assert seen == [((0,), 2), ((1,), 2), ((2,), 2), ((0, 1, 2), 6)]
    assert not resident


def test_repeated_inputs_use_exact_answer_token_gradient():
    model = _c3_pair8_model()
    model.emb.weight.data.fill_(1)
    E.attach(model, [0], "fsa_forced", 0)
    seq, start = [40] * 6, 4
    captured, hooks = {}, []
    for name, lin in E.patched_linears(model):
        if name.endswith("up_proj"):
            def record(mod, inp, out, name=name):
                out.retain_grad()
                captured[name] = out
            hooks.append(lin.register_forward_hook(record))
    xx, lm = E.batchify([seq], "cpu", [start])
    logits = model(xx)[:, :-1]
    torch.nn.functional.cross_entropy(logits[lm[:, 1:]], xx[:, 1:][lm[:, 1:]]).backward()
    for hook in hooks:
        hook.remove()
    assert len(captured) == 1
    name, preact = next(iter(captured.items()))
    assert not preact.grad[:start - 1].any() and preact.grad[start - 1].any()
    scores = preact.grad[start - 1].abs()
    lin = dict(E.patched_linears(model))[name]
    bank = name.rsplit(".", 1)[0]
    down = dict(E.patched_linears(model))[bank + ".down_proj"].parametrizations.weight[0]
    eligible = (lin.parametrizations.weight[0].mask.any(1) & down.mask.any(0)).nonzero().flatten()
    expected = sorted(eligible.tolist(), key=lambda unit: (-float(scores[unit]), unit))[:2]
    report = E.select_fact_units(model, [[(seq, start)]], [0], 2, "cpu")
    assert report["assignments"]["0"][bank] == expected


@pytest.mark.parametrize("from_output", [False, True])
def test_local_penalty_includes_revocable_unit_reset(from_output):
    model, expert = unit_model()
    E.initialize_units(model, {"moes.0.experts.0": [0]}, "zero", 0)
    local = E.LocalOutputPenalty(model, from_output=from_output)
    try:
        local.start()
        expert.up_proj(torch.tensor([[1.] * 2 + [0.] * 6]))
        loss = local.stop()
        assert float(loss.detach()) == pytest.approx(1 / 8)  # One of eight equal-energy base rows is removed.
        local.end()
    finally:
        for hook in local.h:
            hook.remove()


@pytest.mark.parametrize("mode", ["borrowed", "random", "zero"])
def test_main_address_init_reload_and_metrics(cpu_harness, mode):
    run, paths = cpu_harness
    flags = ["--train-person", "0", "--slot-partition", "3", "--slot-part", "0", "--address-units", "2",
             "--unit-init", mode, "--local-lambda", "1", "--anchor", paths["anchor"], "--anchor-len", "16"]
    result, path = run([*flags, "--steps", "2", "--wake-steps", "1"], "init_address")
    loaded, _ = run([*flags, "--steps", "0", "--load-patch", str(path)], "init_address_reload")
    assert result["revoked"]["logits_bit_exact"] and loaded["revoked"]["logits_bit_exact"]
    assert result["three_metrics"] == loaded["three_metrics"]
    assert result["unit_bytes"] == loaded["unit_bytes"]
    assert result["three_metrics"]["address_accuracy"]["available"]


def test_steps_to_learn_uses_first_full_training_eval(cpu_harness, monkeypatch):
    run, _ = cpu_harness
    old = E.tf_eval
    def mastered(model, items, *rest, **kwargs):
        result = old(model, items, *rest, **kwargs)
        if E.fsa_modules(model) and E.FSAParam.enabled and len(items) == 6 * len(E.TRAIN_T):
            result["exact_match"] = .9
        return result
    monkeypatch.setattr(E, "tf_eval", mastered)
    result, _ = run(["--steps", "3", "--train-audit", "1"], "learned")
    assert result["steps_to_learn"] == 2
    assert result["trajectory"][0]["train_em"] < .9  # A small audit is not the full-training criterion.
