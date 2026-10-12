"""C17 CPU checks for export identity, oracle evidence and serving safety."""
import hashlib
import json
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn

from test_e2e_fsa import E, TOK, BOS_T, Expert, _c3_pair8_model, cpu_harness
from test_branch_eval import B, state
import build_forced_mask as masks


def args(*extra):
    return E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused", *extra])


def test_export_defaults_and_cache_identity(tmp_path):
    a = args()
    assert a.export == E.EXPORT and a.export_step == 58814
    assert a.mask_compare is None and not a.oracle_evidence
    b = B.build_parser().parse_args(["--patches", "p", "--anchor", "a", "--eval-set", "e",
                                   "--fineweb", "f", "--code", "c", "--out", "o"])
    assert (b.export, b.export_step, b.mask_compare) == (E.EXPORT, 58814, None)
    for name in ("heldout-tokens.json", "tokenizer.json"):
        (tmp_path / name).write_text("[]")
    key = lambda: E.base_cache_key(a, E.make_world(), None, None, data=tmp_path)
    original = key()
    a.export = tmp_path / "other"
    assert key() != original
    new_export = key()
    a.export_step += 1
    assert key() != new_export


def test_new_export_reaches_loader_and_metadata_and_rejects_load(cpu_harness, monkeypatch, tmp_path):
    run, _ = cpu_harness
    loader = sys.modules["ops.eval.bench_saves"]
    build, seen = loader.build_model, []
    def record(export, **kw):
        seen.append((export, kw["step"]))
        return build(export, **kw)
    monkeypatch.setattr(loader, "build_model", record)
    export = tmp_path / "new_base"
    flags = ["--export", str(export), "--export-step", "17", "--steps", "0"]
    result, patch = run(flags, "new_base")
    assert seen == [(export, 17)]
    meta = torch.load(patch, weights_only=False)["meta"]
    assert (meta["export"], meta["export_step"]) == (str(export), 17)
    assert (result["export"], result["export_step"]) == (str(export), 17)
    loaded, _ = run([*flags, "--load-patch", str(patch)], "new_reload")
    assert "wrong_answer_rate" in loaded["safety"] and loaded["safety"]["fired"] is None
    assert "confidently_wrong_without_fallback" not in loaded["safety"]
    for overrides in (["--export", str(tmp_path / "other")], ["--export-step", "18"]):
        with pytest.raises(SystemExit, match="export differs"):
            run([*flags, "--load-patch", str(patch), *overrides], "mismatch")


def test_low_level_load_checks_export_before_mutation_and_legacy_step():
    model = _c3_pair8_model()
    E.attach(model, [0], "fsa_forced", 0)
    saved = E.export_patch(model, dict(export=str(E.EXPORT)))
    model._fsa_export, model._fsa_export_step = E.EXPORT, E.EXPORT_STEP
    E.load_patch(model, saved)  # Older patches implicitly used step 58814.
    for pm in E.fsa_modules(model):
        pm.M.data.fill_(1)
    model._fsa_export_step = 99
    with pytest.raises(ValueError, match="export differs"):
        E.load_patch(model, saved)
    assert all(bool(pm.M.eq(1).all()) for pm in E.fsa_modules(model))
    states = [state(i, [i], [1]) for i in range(3)]
    states[1]["meta"]["export_step"] = 99
    with pytest.raises(ValueError, match="export_step"):
        B.validate_patches(states)


def test_mask_sha_and_packed_comparison_use_only_patched_linears(tmp_path, capsys):
    model = _c3_pair8_model()
    entries = masks.build(model)
    path = tmp_path / "pair8_forced_mask.pt"
    torch.save(dict(format="pair8-forced-mask-v1", entries=entries), path)
    E.attach(model, [0], "fsa_forced", 0, projs=["up_proj"], sparse=True,
             slot_frac=.5, slot_partition=3, slot_part=1)
    patched = sorted(E.patched_linears(model))
    expected = b"".join(E.pair8_forced_mask(lin.parametrizations.weight.original).to(torch.uint8).numpy().tobytes()
                        for _, lin in patched)
    result = E.mask_diagnostics(model, path)
    assert result["sha256"] == hashlib.sha256(expected).hexdigest()
    assert result["difference_fraction"] == 0 and result["positions"] == len(expected)
    # One selected-pair bit changes two weight positions, despite slot sampling/partitioning.
    entries[patched[0][0]]["bits"][0] ^= np.uint8(128)
    torch.save(dict(format="pair8-forced-mask-v1", entries=entries), path)
    changed = E.mask_diagnostics(model, path)
    assert changed["difference_fraction"] == 2 / len(expected)
    assert changed["sha256"] == result["sha256"]
    assert "MASK-SHA " + result["sha256"] in capsys.readouterr().out
    del entries[patched[0][0]]
    torch.save(dict(format="pair8-forced-mask-v1", entries=entries), path)
    with pytest.raises(ValueError, match="missing or mismatched"):
        E.mask_diagnostics(model, path)


def test_oracle_context_variants_are_causal_and_use_other_people():
    facts = E.make_world(7, 3, 12, 4)
    items = E.build_items(TOK, facts, E.TEST_T[:1], BOS_T)
    rng = random.getstate()
    for variant in ("needed_only", "with_distractors", "conflicting_dated"):
        contextual = E.oracle_evidence_items(TOK, facts, items, BOS_T, variant)
        for original, new in zip(items, contextual):
            assert new["p"][0] == BOS_T and new["p"].count(BOS_T) == 1
            assert new["p"][-len(original["p"][1:]):] == original["p"][1:]
            sheet = TOK.decode(new["p"][:-len(original["p"][1:])]).strip()
            f = facts[original["fi"]]
            if variant == "with_distractors":
                lines = sheet.splitlines()
                assert len(lines) == 21 and len(set(lines)) == 21
                assert all(not line.startswith(f.name + "'s") for line in lines[1:])
                others = [fact for fact in facts if fact.pid != f.pid]
                same = [fact for fact in others if fact.attr == f.attr]
                if same:
                    assert any(f"{fact.name}'s {fact.attr} is {fact.val}." in lines for fact in same)
                different = [fact for fact in others if fact.attr != f.attr]
                if different:
                    assert any(f"{fact.name}'s {fact.attr} is {fact.val}." in lines for fact in different)
            elif variant == "conflicting_dated":
                old = next(v for v in dict(E.ATTR_POOL)[f.attr] if v != f.val)
                assert sheet == (f"On 2023-03-01, {f.name}'s {f.attr} was {old}. On 2025-06-01, "
                                 f"{f.name}'s {f.attr} changed to {f.val}. Use the most recent information.")
            else:
                assert sheet == f"{f.name}'s {f.attr} is {f.val}."
    assert random.getstate() == rng


def test_oracle_base_gen_and_composition_scoring(monkeypatch):
    facts = E.make_world(7, 3, 12, 4)
    items = E.build_items(TOK, facts, E.TEST_T[:1], BOS_T)
    by_prompt = {}
    for variant in ("needed_only", "with_distractors", "conflicting_dated"):
        for it in E.oracle_evidence_items(TOK, facts, items, BOS_T, variant):
            by_prompt[tuple(it["p"])] = TOK.encode(" " + it["val"] + ".").ids
    compose = E.oracle_evidence_items(TOK, facts, B.composition_items(TOK, facts, BOS_T), BOS_T, "needed_only")
    for it in compose:
        for fi in it["fis"]:
            assert f"{facts[fi].name}'s {facts[fi].attr} is {facts[fi].val}." in TOK.decode(it["p"])
        by_prompt[tuple(it["p"])] = it["a"]
    def generate(model, prompts, *rest):
        assert not E.FSAParam.enabled
        return [by_prompt[tuple(p)] for p in prompts]
    monkeypatch.setattr(E, "greedy_generate", generate)
    E.set_enabled(True)
    result = E.oracle_evidence_eval(None, TOK, facts, items, "cpu", BOS_T, args())
    for variant in ("needed_only", "with_distractors", "conflicting_dated", "composition"):
        assert result[variant]["recall"] == 1
    assert E.FSAParam.enabled


def test_oracle_main_exits_without_training_data_attachment_or_writes(cpu_harness, monkeypatch, tmp_path, capsys):
    (tmp_path / "heldout-tokens.json").unlink()
    monkeypatch.setattr(E, "attach", lambda *a, **kw: pytest.fail("oracle attached a patch"))
    monkeypatch.setattr(E, "evaluate_base", lambda *a, **kw: pytest.fail("oracle evaluated damage"))
    out = tmp_path / "unused.json"
    monkeypatch.setattr(sys, "argv", ["e2e_fsa.py", "--method", "fsa_forced", "--out", str(out),
                                     "--oracle-evidence", "--people", "3", "--facts-per-person", "12",
                                     "--shared-attrs", "4", "--gen-max-new", "1"])
    E.main()
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("ORACLE-EVIDENCE "))
    result = json.loads(line.split(" ", 1)[1])
    assert result["needed_only"]["n"] == 36 * len(E.TEST_T)
    assert result["with_distractors"]["n"] == result["needed_only"]["n"]
    assert result["composition"]["n"] > 0
    assert not out.exists()


@pytest.mark.parametrize("flag", ["--shared-expert", "--shared-only"])
def test_dense_shared_address_is_infeasible_before_attach(cpu_harness, monkeypatch, capsys, flag):
    run, paths = cpu_harness
    loader = sys.modules["ops.eval.bench_saves"]
    build, models = loader.build_model, []
    def dense(*args, **kwargs):
        model, cfg, dev = build(*args, **kwargs)
        for moe in model.moes:
            expert = Expert(16, 32, torch.Generator().manual_seed(1))
            for lin in (expert.up_proj, expert.down_proj):
                lin.weight.data.fill_(1e-20)  # nonzero, however tiny
            moe.shared_experts = nn.ModuleList([expert])
        models.append(model)
        return model, cfg, dev
    monkeypatch.setattr(loader, "build_model", dense)
    with pytest.raises(SystemExit, match="infeasible under the hard rule"):
        run([flag, "--address-units", "2", "--train-person", "1", "--slot-partition", "3", "--slot-part", "1",
             "--local-lambda", "1", "--anchor", paths["anchor"]], "dense")
    assert not E.patched_linears(models[0])
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("SHARED-ADDRESS "))
    counts = json.loads(line.split(" ", 1)[1])["exact_zero_counts"]
    assert len(counts) == 4 and set(counts.values()) == {0}


def test_shared_address_masks_are_exact_zeros_without_pair8_assumption():
    model = _c3_pair8_model()
    for moe in model.moes:
        expert = Expert(16, 32, torch.Generator().manual_seed(1))
        for lin in (expert.up_proj, expert.down_proj):
            lin.weight.data.fill_(1)
            lin.weight.data[:, 0] = 0  # Three/four active pairs per block: not pair8.
        moe.shared_experts = nn.ModuleList([expert])
    E.check_shared_address(model, [0], 2)
    E.attach(model, [0], "fsa_forced", 0, shared=True, routed=False, shared_address=True)
    for name, lin in E.patched_linears(model):
        assert ".shared_experts." in name
        pm = lin.parametrizations.weight[0]
        assert torch.equal(pm.mask, lin.parametrizations.weight.original == 0)
    assert E.mask_diagnostics(model)["positions"] == 0


def test_shared_address_wake_and_firing_use_shared_expert_inputs():
    from types import MethodType
    model = _c3_pair8_model()
    moe = model.moes[0]
    expert = Expert(16, 32, torch.Generator().manual_seed(3))
    for lin in (expert.up_proj, expert.down_proj):
        lin.weight.data.fill_(.1)
        lin.weight.data[:, 0] = 0
    moe.shared_experts = nn.ModuleList([expert])
    original_forward = moe.forward
    def shared_forward(self, x):
        return original_forward(x) + self.shared_experts[0](-x)
    moe.forward = MethodType(shared_forward, moe)
    originals = {proj: getattr(expert, proj).weight.detach().clone() for proj in ("up_proj", "down_proj")}
    E.attach(model, [0], "fsa_forced", 0, shared=True, routed=False, shared_address=True,
             sparse=True, slot_partition=3, slot_part=1)
    a = args("--shared-only", "--address-units", "1", "--train-person", "1", "--slot-partition", "3",
             "--slot-part", "1", "--wake-steps", "1", "--local-lambda", "1")
    a.wake_units = a.address_units
    facts = E.make_world(0, 3, 2, 2)
    report = E.run_wake_stage(model, TOK, facts, E.TRAIN_T[:1], BOS_T, [0],
                              torch.tensor([[BOS_T] + [40] * 15] * 2), a)
    assert list(report["detectors"]) == ["moes.0.shared_experts.0.up_proj"]
    for name, lin in E.patched_linears(model):
        original = originals[name.rsplit(".", 1)[1]]
        assert torch.equal(lin.weight[original != 0], original[original != 0])
    prompt = [BOS_T, 40]
    observed = []
    row = report["detectors"]["moes.0.shared_experts.0.up_proj"]["rows"]
    hook = expert.up_proj.register_forward_hook(lambda mod, inp, out: observed.append(bool((out[0, -1, row] > 0).any())))
    with torch.no_grad():
        E.logits_of(model, torch.tensor([prompt]))
    hook.remove()
    assert E.address_fired(model, prompt, report["detectors"], "cpu") == observed[0]


@pytest.mark.parametrize("mode", ["off", "exact", "batched"])
def test_address_firing_observes_grouped_dispatch_after_input_transform(monkeypatch, mode):
    from test_fast_moe import FakeMoE
    import fast_moe
    moe = FakeMoE(latent_dim=8, top_k=1)
    model = nn.Sequential(moe)
    with torch.no_grad():
        moe.latent_down.weight.copy_(-torch.eye(8))
        moe.router.weight.zero_()
        moe.router.weight[0].fill_(1)
        for expert in moe.experts:
            expert.up_proj.weight[0].fill_(1)
            expert.up_proj.weight[1].fill_(-1)
    if mode != "off":
        fast_moe.install(model, mode=mode)
    E.attach(model, [0], "fsa_full", 0)
    dispatch = moe._forward_frozen_exact_grouped
    monkeypatch.setattr(E, "logits_of", lambda model, ids: model(torch.ones(1, ids.shape[1], 8)))
    # The residual input is positive; the expert's actual input is negative.
    assert not E.address_fired(model, [BOS_T, 40], {"0.experts.0.up_proj": dict(rows=[0])}, "cpu")
    assert E.address_fired(model, [BOS_T, 40], {"0.experts.0.up_proj": dict(rows=[1])}, "cpu")
    assert not E.address_fired(model, [BOS_T, 40], {"0.experts.1.up_proj": dict(rows=[1])}, "cpu")
    assert moe._forward_frozen_exact_grouped == dispatch
    assert all(not lin._forward_hooks and not lin._forward_pre_hooks for _, lin in E.patched_linears(model))


def test_safety_uses_wrong_answers_not_bad_stops_and_excludes_eos_tail(monkeypatch):
    monkeypatch.setattr(E, "EOS_IDS", (2,))
    encode = lambda text: TOK.encode(text).ids
    items = [dict(val="gold")] * 7
    gens = [encode(text) for text in ("wrong.", "novel.", "gold extra", "", "gold.", "gold and wrong.")]
    gens.append(encode("gold.") + [2] + encode("wrong."))
    result = E.generation_safety(TOK, items, gens, ["gold", "wrong"], fired=[False, False, False, False, True, True, False])
    assert result["wrong_answer_rate"] == 3 / 7
    assert result["confidently_wrong_without_fallback"] == 2 / 7
    assert E.generation_safety(TOK, items, gens, ["gold", "wrong"])["fired"] is None
    assert E.generation_safety(TOK, [], [], [], fired=[])["confidently_wrong_without_fallback"] == 0


@pytest.mark.parametrize("sparse", [False, True])
def test_address_firing_checks_selected_dispatched_units_at_last_token(sparse):
    model = _c3_pair8_model()
    with torch.no_grad():
        model.emb.weight.zero_()
        model.emb.weight[40].fill_(1)
        model.emb.weight[50].fill_(-1)
        model.moes[0].router.weight.zero_()  # Always expert 0.
        for expert in model.moes[0].experts:
            expert.up_proj.weight[0].zero_()
            expert.up_proj.weight[0, :4] = -.1
            expert.up_proj.weight[1].zero_()
            expert.up_proj.weight[1, :4] = 1
    E.attach(model, [0], "fsa_forced", 0, sparse=sparse)
    selected = {"moes.0.experts.0.up_proj": dict(rows=[0])}
    assert not E.address_fired(model, [BOS_T, 40], selected, "cpu")  # Unselected row 1 is positive.
    lin = dict(E.patched_linears(model))[next(iter(selected))]
    pm = lin.parametrizations.weight[0]
    with torch.no_grad():
        pm.alpha.fill_(1)
        if sparse:
            pm.scale.fill_(1)
            pm.M[pm.rows() == 0] = 1
        else:
            pm.M[0, pm.mask[0]] = 1
    assert E.address_fired(model, [BOS_T, 40], selected, "cpu")
    assert not E.address_fired(model, [BOS_T, 40, 50, 50], selected, "cpu")
    assert not E.address_fired(model, [BOS_T, 40], {"moes.0.experts.1.up_proj": dict(rows=[1])}, "cpu")
    assert all(not lin._forward_pre_hooks for _, lin in E.patched_linears(model))


def test_router_none_and_address_fallback_decisions_for_f_arm(monkeypatch):
    states = {p: state(p, [p], [1]) for p in range(3)}
    for p, s in states.items():
        s["meta"].update(address_units=1, wake=dict(detectors={"x": dict(rows=[p])}))
    branches = B.Branches(None, states, {p: 1 for p in states}, "sum_clip")
    activated, seen = [], []
    monkeypatch.setattr(branches, "activate", lambda people: activated.append(tuple(people)))
    def fired(model, prompt, detectors, dev):
        seen.append(detectors["x"]["rows"])
        return prompt[-1] > 0
    monkeypatch.setattr(E, "address_fired", fired)
    items = [dict(p=[-1]), dict(p=[-1]), dict(p=[1])]
    routes = [(), (0, 1, 2), (0, 1, 2)]
    assert B.routed_firing(branches, items, routes, "cpu") == [False, False, True]
    assert activated == [(0, 1, 2)] * 2 and seen == [[0, 1, 2]] * 2


def test_branch_new_base_router_and_address_f_safety(cpu_harness, monkeypatch, tmp_path):
    from test_e2e_fsa import Tiny
    monkeypatch.setattr(Tiny,"layers",property(lambda self:self.moes),raising=False)
    run, paths = cpu_harness
    export = tmp_path / "branch_base"
    flags = ["--export", str(export), "--export-step", "17", "--steps", "0", "--sparse-state"]
    full, part = [], []
    for person in range(3):
        _, patch = run([*flags, "--train-person", str(person)], f"base_full{person}")
        full.append(str(patch))
        _, patch = run([*flags, "--train-person", str(person), "--slot-partition", "3", "--slot-part", str(person),
                        "--address-units", "1", "--wake-steps", "1", "--local-lambda", "1",
                        "--anchor", paths["anchor"], "--anchor-len", "16"], f"base_address{person}")
        part.append(str(patch))
    out = tmp_path / "branches.json"
    argv = ["branch_eval.py", "--patches", *full, "--partition-patches", *part, "--arms", "C,F",
            "--anchor", paths["anchor"], "--eval-set", paths["eval"], "--fineweb", paths["extra"],
            "--code", paths["extra"], "--router-layer", "0", "--router-none", "2", "--router-steps", "1",
            "--qa-n", "0", "--g-n", "0", "--fast-moe", "off", "--out", str(out)]
    loader = sys.modules["ops.eval.bench_saves"]
    build, seen = loader.build_model, []
    def record(export, **kw):
        seen.append((export, kw["step"]))
        return build(export, **kw)
    monkeypatch.setattr(loader, "build_model", record)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="export differs"):
        B.main()
    assert not seen  # Reject mismatched base before building it.
    class NoneProbe:
        def predict(self, x):
            return torch.full((len(x),), 3)
    monkeypatch.setattr(B, "fit_router", lambda *a: NoneProbe())
    monkeypatch.setattr(B, "routed_generations", lambda branches, tok, items, *rest:
                        [TOK.encode(" unknown.").ids if "val" in it else [] for it in items])
    firing = []
    def inactive(model, prompt, detectors, dev):
        assert detectors and all(entry["rows"] for entry in detectors.values())
        firing.append(True)
        return False
    monkeypatch.setattr(E, "address_fired", inactive)
    monkeypatch.setattr(sys, "argv", argv + ["--export", str(export), "--export-step", "17"])
    B.main()
    result = json.loads(out.read_text())
    assert seen == [(export, 17)]
    assert (result["export"], result["export_step"]) == (str(export), 17)
    for arm in ("C:0", "F"):
        assert result["arms"][arm]["test_gen"]["wrong_answer_rate"] == 1
        assert result["arms"][arm]["confidently_wrong_without_fallback"] == 1
    # C19 measures each bank under individual and composed residency as well as
    # the serving safety decision; all query measurements use the same causal probe.
    diagnostics = result["three_metrics"]["partition"]["address_accuracy"]
    diagnostic_queries = sum(row["owners"][0]["positives"] + row["owners"][0]["negatives"]
                             for phase in diagnostics.values() for row in phase.values())
    assert len(firing) == result["arms"]["F"]["test_gen"]["n"] + diagnostic_queries
