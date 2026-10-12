"""C20 CPU checks for address allocation, adaptability and unrestricted ternary codes."""
import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, _c3_pair8_model, cpu_harness
from test_c19 import args, unit_model
from test_branch_eval import B


def test_defaults_validation_and_removed_audit():
    a = args()
    assert (a.address_center, a.exclude_general, a.entropy_units, a.address_mode) == (False, 0., False, "mutable")
    E.validate_c20_args(a)
    for flags in (["--exclude-general", "nan"], ["--exclude-general", "1"],
                  ["--exclude-general", "-.1"], ["--address-center"], ["--entropy-units"],
                  ["--units-per-fact", "2", "--address-center", "--calib-set", ""],
                  ["--units-per-fact", "2", "--address-mode", "contrastive"]):
        with pytest.raises(SystemExit):
            E.validate_c20_args(args(*flags))
    with pytest.raises(SystemExit):
        args("--legality-audit")
    assert not hasattr(E, "legality_audit")
    assert "--legality-audit" not in B.build_parser().format_help()


def test_general_source_first_32_and_entropy_ratio(tmp_path):
    import json
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps([60] * (40 * 511)))
    windows, source = E.address_general_windows(path, {"fineweb.json": [90] * 20440}, BOS_T, "cpu")
    assert source["path"] == str(path) and source["sha256"] == E.file_sha(path)
    assert windows.shape == (32, 512) and (windows[:, 1:] == 60).all()
    assert E.attribution_entropy(torch.tensor([1., 0., 0., 0.])) == 0
    assert E.attribution_entropy(torch.ones(4)) == pytest.approx(1)
    assert E.attribution_entropy(torch.zeros(4)) == 1
    assert E.entropy_unit_count(8, 0) == 4
    assert E.entropy_unit_count(8, 1) == 12
    assert E.entropy_unit_count(1, 0) == 1
    assert E.entropy_unit_count(8, 10) == 32


def test_general_stats_base_only_all_tokens_cleanup_and_exclusion(monkeypatch):
    model, expert = unit_model()
    for m in E.fsa_modules(model):
        m.M.data.fill_(1)
    x = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.], [-1., -1., 0., 0., 0., 0., 0., 0.]])
    monkeypatch.setattr(E, "logits_of", lambda model, win: expert.up_proj(x))
    stats = E.collect_address_general(model, torch.ones(40, 4, dtype=torch.long))
    s = stats["moes.0.experts.0.up_proj"]
    assert s["tokens"] == 32 * 4 and s["routed_tokens"] == 32 * 2
    assert torch.equal(s["mean"], x.mean(0))
    assert torch.equal(s["positive_fraction"], torch.full((8,), .25))
    assert E.FSAParam.enabled and not expert.up_proj._forward_hooks
    eligible = torch.tensor([True, True, False, True])
    fractions = dict(positive_fraction=torch.tensor([.2, .9, 1., .4]))
    assert E.exclude_general_units(eligible, fractions, 1 / 3).tolist() == [True, False, False, True]
    assert eligible.tolist() == [True, True, False, True]


@pytest.mark.parametrize("sparse", [False, True])
def test_centered_selection_and_training_gradients_preserve_forward(sparse):
    model, expert = unit_model(sparse)
    mean = torch.tensor([1., 0., 0., 0., 1., 2., 3., 4.])
    stats = {"moes.0.experts.0.up_proj": dict(mean=mean)}
    x = mean + 2
    actual = expert.up_proj(x).detach().clone()
    with E.centered_address_inputs(model, stats):
        assert torch.equal(expert.up_proj(x), torch.nn.functional.linear(x - mean, expert.up_proj.weight))
    assert torch.equal(expert.up_proj(x), actual)
    hooks = E.center_address_gradients(model, stats)
    out = expert.up_proj(x)
    assert torch.equal(out, actual)
    out.sum().backward()
    m = expert.up_proj.parametrizations.weight[0]
    expected = ((x - mean)[None, :] * m.alpha).expand(m.wshape)
    if sparse:
        expected = expected.flatten()[m.idx.long()]
    else:
        expected = expected * m.mask
    assert torch.allclose(m.M.grad, expected)
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("sparse", [False, True])
def test_all_forced_positions_writable_four_pairs_revoke_and_bank_bytes(sparse):
    model, expert = unit_model(sparse)
    assignments = {"0": {"moes.0.experts.0": [4]}}
    E.restrict_unit_masks(model, assignments)
    E.snapshot_addresses(model, assignments)
    for name, lin in E.patched_linears(model):
        m = lin.parametrizations.weight[0]
        m.M.data.copy_(m.mask.float())
        original = lin.parametrizations.weight.original
        expected = E.pair8_forced_mask(original) & E.unit_slot_mask(
            m, [4], name.rsplit(".", 1)[1]) if not sparse else m.mask
        assert torch.equal(E.writable_slots(m), expected)
    active_pairs = expert.up_proj.weight[4].ne(0).reshape(4, 2).any(1).sum()
    assert active_pairs == 4  # The frozen base keeps two; the overlay may open both others.
    assert E.address_moved(model) == 4
    packed = E.unit_bytes(model)
    assert packed["logical_units_used"] == 1
    assert packed["aligned_8_unit_banks_touched"] == 1
    assert packed["packed_bytes"] == 16  # Two 8x8 half-zero tensors, two bits/forced position.
    assert packed["position_encoding_overhead_bytes"] == 1
    saved = E.export_patch(model)
    clone, _ = unit_model(sparse)
    E.load_patch(clone, saved)
    E.restrict_unit_masks(clone, assignments)
    E.snapshot_addresses(clone, assignments)
    assert E.address_moved(clone) == 4
    E.set_enabled(False)
    try:
        assert expert.up_proj.weight[4].ne(0).sum() == 4
    finally:
        E.set_enabled(True)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("mode", ["frozen", "mutable", "contrastive"])
def test_person_address_modes_and_selected_value_columns(sparse, mode):
    model = _c3_pair8_model()
    E.attach(model, [0], "fsa_forced", 0, sparse=sparse)
    a = args("--address-units", "2", "--train-person", "0", "--address-mode", mode,
             "--wake-steps", "2", "--wake-lr", ".2")
    facts = E.make_world(0, 3, 2, 2)
    anchor = torch.tensor([[BOS_T] + [40] * 15] * 2)
    report = E.run_wake_stage(model, TOK, facts, E.TRAIN_T[:1], BOS_T, [0], anchor, a)
    for name, lin in E.patched_linears(model):
        m = lin.parametrizations.weight[0]
        bank, proj = name.rsplit(".", 1)
        selected = report["detectors"].get(bank + ".up_proj", {}).get("rows", [])
        assert torch.equal(m.mask, E.unit_slot_mask(m, selected, proj) & m.mask)
        if proj == "up_proj":
            assert bool(E.writable_slots(m).any()) == (mode == "mutable" and bool(selected))
            if mode == "frozen":
                assert not m.M.any()
    assert not any(lin._forward_hooks for _, lin in E.patched_linears(model))


@pytest.mark.parametrize("mode", ["frozen", "mutable", "contrastive"])
@pytest.mark.parametrize("sparse", [False, True])
def test_main_fact_address_controls_and_reload(cpu_harness, capsys, mode, sparse):
    run, paths = cpu_harness
    flags = ["--units-per-fact", "2", "--address-mode", mode, "--address-center", "--exclude-general", ".1",
             "--calib-set", paths["calib"],
             "--entropy-units", "--anchor", paths["anchor"], "--anchor-len", "16", "--train-person", "0",
             "--budget", "0", "--answer-only"] + (["--sparse-state"] if sparse else [])
    # Leave contrastive steps out on reload, which restores the trained addresses.
    wake = ["--wake-steps", "2", "--wake-lr", ".2"] if mode == "contrastive" else []
    result, path = run([*flags, *wake, "--steps", "2"], f"c20_{mode}_{sparse}")
    loaded, _ = run([*flags, "--load-patch", str(path), "--steps", "0"], "c20_reload")
    assert result["patched"] == loaded["patched"]
    assert result["address_moved"] == loaded["address_moved"]
    assert result["unit_bytes"] == loaded["unit_bytes"]
    assert result["unit_allocation"]["entropy_units"]
    assert result["revoked"]["logits_bit_exact"]
    assert all("address_moved" in point for point in result["trajectory"])
    if mode == "frozen":
        assert result["address_moved"] == 0
    output = capsys.readouterr().out
    assert "LEGALITY " not in output and "ADDRESS-GENERAL " in output and "UNIT-BYTES " in output


def test_general_exclusion_and_entropy_applied_to_fact_allocation():
    model = _c3_pair8_model()
    E.attach(model, [0], "fsa_forced", 0)
    facts = E.make_world(0, 1, 2, 0)
    pool = [[E.encode_fact(TOK, E.TRAIN_T[0], f, BOS_T)] for f in facts]
    stats = {name: dict(positive_fraction=torch.arange(lin.out_features).float())
             for name, lin in E.patched_linears(model) if name.endswith("up_proj")}
    report = E.select_fact_units(model, pool, [0], 2, "cpu", stats, exclude_general=.5, entropy_units=True)
    for fi, banks in report["assignments"].items():
        for bank, rows in banks.items():
            assert all(row < 16 for row in rows)
            h = report["normalized_entropy"][fi][bank]
            assert len(rows) == E.entropy_unit_count(2, h)


def test_branch_address_baselines_merge_and_switch():
    model, expert = unit_model()
    states = {}
    for person in (4, 5):
        for m in E.fsa_modules(model):
            m.M.data.zero_()
        assignment = {"0": {"moes.0.experts.0": [person]}}
        E.snapshot_addresses(model, assignment)
        expert.up_proj.parametrizations.weight[0].M.data[person, 4:] = 1
        states[person] = E.export_patch(model, dict(train_person=person, units_per_fact=1,
                                                  unit_allocation=dict(assignments=assignment)))
    branches = B.Branches(model, states, dict.fromkeys(states, 1), "sum_clip")
    for people, moved in (((4,), 4), ((4, 5), 8), ((), 0), ((5,), 4)):
        branches.activate(people)
        assert E.address_moved(model) == moved


def test_logical_allocations_count_empty_codes_and_bank_alignment():
    model = _c3_pair8_model()
    E.attach(model, [0], "fsa_forced", 0)
    bank = "moes.0.experts.0"
    E.snapshot_addresses(model, {"0": {bank: [0, 7]}})
    together = E.unit_bytes(model)
    assert together["logical_units_used"] == 2 and together["aligned_8_unit_banks_touched"] == 1
    E.snapshot_addresses(model, {"0": {bank: [0, 8]}})
    apart = E.unit_bytes(model)
    assert apart["logical_units_used"] == 2 and apart["aligned_8_unit_banks_touched"] == 2
    assert apart["packed_bytes"] == 2 * together["packed_bytes"]


def test_person_address_general_exclusion():
    model = _c3_pair8_model()
    E.attach(model, [0], "fsa_forced", 0)
    stats = {name: dict(positive_fraction=torch.arange(lin.out_features).float())
             for name, lin in E.patched_linears(model) if name.endswith("up_proj")}
    a = args("--address-units", "2", "--train-person", "0", "--address-mode", "frozen",
             "--exclude-general", ".5")
    survivors = {}
    for name, lin in E.patched_linears(model):
        if name.endswith("up_proj"):
            m = lin.parametrizations.weight[0]
            eligible = (m.mask.any(1) & (m.alpha[:, 0] > 0)).nonzero().flatten().tolist()
            survivors[name] = set(eligible[:len(eligible) // 2])
    report = E.run_wake_stage(model, TOK, E.make_world(0, 3, 2, 2), E.TRAIN_T[:1], BOS_T, [0],
                              torch.tensor([[BOS_T] + [40] * 15] * 2), a, general_stats=stats)
    assert all(set(d["rows"]) <= survivors[name] for name, d in report["detectors"].items())
    assert report["training_steps"] == 0


@pytest.mark.parametrize("sparse", [False, True])
def test_address_gradient_starts_after_value_becomes_nonzero(sparse):
    model, expert = unit_model(sparse)
    assignment = {"0": {"moes.0.experts.0": [4]}}
    E.restrict_unit_masks(model, assignment)
    E.initialize_units(model, assignment["0"], "zero", 0)
    up, down = [lin.parametrizations.weight[0] for lin in (expert.up_proj, expert.down_proj)]
    up.M.data.copy_(up.mask.float())
    x = torch.tensor([[0., 0., 0., 0., 1., 2., 3., 4.]])
    expert.down_proj(torch.relu(expert.up_proj(x)).square()).sum().backward()
    assert not up.M.grad.any() and down.M.grad.any()
    model.zero_grad(set_to_none=True)
    down.M.data.copy_(down.mask.float())
    expert.down_proj(torch.relu(expert.up_proj(x)).square()).sum().backward()
    assert up.M.grad.any()
