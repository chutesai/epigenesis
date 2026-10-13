"""C37 learning-only controls and real CPU harness integration."""
import random
from types import SimpleNamespace

import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, cpu_harness
import branch_eval as B


def args(*flags):
    return E.build_parser().parse_args(["--method", "fsa_forced", "--out", "/tmp/c37.json", *flags])


def test_defaults_and_controls():
    a = args()
    assert not E.c37_enabled(a)
    assert a.sibling_p == .5 and a.slot_dropout == a.trit_corrupt == 0
    assert E.c37_positive_control()["passed"]
    assert B.main(["--c37-control"])["passed"]


@pytest.mark.parametrize("flag", ["--slot-dropout", "--trit-corrupt", "--sibling-p"])
@pytest.mark.parametrize("value", ["-1", "1.1", "nan", "inf"])
def test_invalid_probabilities(flag, value):
    with pytest.raises(SystemExit, match="must be in"):
        E.validate_c37_args(args(flag, value))


@pytest.mark.parametrize("flags", [
    ["--dark-audit"], ["--transplant"],
    ["--transplant", "--load-patch", "x", "--steps", "0", "--repair-steps", "11"],
    ["--hard-facts", "override"], ["--evidence-skill", "--optim", "bop"],
    ["--trit-corrupt", ".1", "--optim", "adam"],
])
def test_invalid_combinations(flags):
    with pytest.raises(SystemExit):
        E.validate_c37_args(args(*flags))


@pytest.mark.parametrize("sparse", [False, True])
def test_trit_forward_preserves_masters_ema_support_and_ste(sparse):
    W = torch.tensor([[1.,1.,1.,1.,0.,0.,0.,0.]])
    mask = E.pair8_forced_mask(W)
    m = E.SparseFSAParam(W, mask) if sparse else E.FSAParam(W, mask)
    with torch.no_grad():
        m.M[m.mask] = 1
    m.bop_ema = torch.ones_like(m.M)
    before = [t.clone() for t in (m.M, m.bop_ema, m.support, m.codes())]
    E.robust_trit_forward([m], 0., 1., torch.Generator().manual_seed(1))
    assert torch.equal(m.delta(), -m.codes() * m.alpha)
    m.delta().sum().backward()
    assert m.M.grad[m.mask].abs().sum() > 0
    for actual, original in zip((m.M, m.bop_ema, m.support, m.codes()), before):
        assert torch.equal(actual, original)
    E.clear_patch_dropout([m])
    assert torch.equal(m.delta(), m.codes()*m.alpha)


def test_sibling_independent_draws_and_exception_restoration():
    banks = [[SimpleNamespace(enabled=True)], [SimpleNamespace(enabled=False)]]
    rng = random.Random(3)
    seen = {E.sibling_presence(banks, .5, rng) for _ in range(100)}
    assert len(seen) == 4
    E.sibling_presence(banks, 0., rng)
    assert not any(m.enabled for bank in banks for m in bank)
    E.sibling_presence(banks, 1., rng)
    assert all(m.enabled for bank in banks for m in bank)
    with pytest.raises(RuntimeError):
        E.sibling_config_eval(banks, lambda: (_ for _ in ()).throw(RuntimeError()))
    assert all(m.enabled for bank in banks for m in bank)


@pytest.mark.parametrize("sparse", [False, True])
def test_transplant_new_mask_new_scales_and_source_immutable(sparse):
    source = torch.nn.Module()
    source.linear = torch.nn.Linear(8, 1, bias=False)
    source.linear.weight.data.copy_(torch.tensor([[1.,1.,1.,1.,0.,0.,0.,0.]]))
    old = E.FSAParam(source.linear.weight.detach(), E.pair8_forced_mask(source.linear.weight))
    E.P.register_parametrization(source.linear, "weight", old)
    with torch.no_grad():
        old.M[0,4] = 1
        old.M[0,6] = -1
    state = E.export_patch(source, dict(export="/tmp/source", export_step=0))
    target = torch.nn.Module()
    target.linear = torch.nn.Linear(8, 1, bias=False)
    target.linear.weight.data.copy_(torch.tensor([[0.,0.,2.,2.,2.,2.,0.,0.]]))
    cls = E.SparseFSAParam if sparse else E.FSAParam
    new = cls(target.linear.weight.detach(), E.pair8_forced_mask(target.linear.weight))
    E.P.register_parametrization(target.linear, "weight", new)
    target._fsa_export, target._fsa_export_step = "/tmp/target", 1
    mapped, report = E.transplant_patch(target, state)
    assert report["dropped_fraction"] == .5
    assert state["entries"]["linear"]["idx"].tolist() == [4,6]
    assert mapped["entries"]["linear"]["idx"].tolist() == [6]
    assert new.delta()[0,6] == -2 and new.delta()[0,4] == 0
    assert mapped["meta"]["export"] == "/tmp/target"


def test_hard_strings_reproducible_length_matched_and_overrides_disagree():
    facts = E.make_world()
    hard = E.hard_fact_world(facts, "surprisal", 0, TOK)
    assert hard == E.hard_fact_world(facts, "surprisal", 0, TOK)
    assert all(len(f.val) == len(original.val) for f, original in zip(hard, facts))
    assert all(f.val != original.val for f, original in zip(hard, facts))
    overrides = E.hard_fact_world(facts, "override", 0, TOK)
    assert len(overrides) == 20
    assert all(f.val != value for f, (_, value) in zip(overrides, E.OVERRIDE_TRUTHS))


def test_override_requires_true_base_answer(monkeypatch):
    facts = E.hard_fact_world([], "override", 0)
    def generate(model, prompts, *unused):
        return [TOK.encode(E.OVERRIDE_TRUTHS[i][1] + "." if i % 2 else "wrong.").ids
                for i in range(len(prompts))]
    monkeypatch.setattr(E, "greedy_generate", generate)
    eligible, report = E.override_base_agreement(None, TOK, facts, "cpu", BOS_T, args())
    assert report["agreement"] == .5 and len(eligible) == 10
    assert all(report["items"][i]["agrees"] == bool(i % 2) for i in range(20))


def test_operand_control_exposes_literal_and_scores_operations(monkeypatch):
    facts = [E.Fact(0, "Ada", "her", "she", "house number", "49"),
             E.Fact(0, "Ada", "her", "she", "lucky number", "50"),
             E.Fact(0, "Ada", "her", "she", "instrument", "cello")]
    groups = E.operand_items(TOK, facts, BOS_T)
    answers = {}
    for items in groups.values():
        for it in items:
            assert it["p"].count(BOS_T) == it["literal_p"].count(BOS_T) == 1
            assert TOK.decode(it["literal_p"]).startswith("Ada's")
            answers[tuple(it["literal_p"])] = TOK.encode(it["val"] + ".").ids
    def generate(model, prompts, *unused):
        return [answers.get(tuple(p), TOK.encode("wrong.").ids) for p in prompts]
    monkeypatch.setattr(E, "greedy_generate", generate)
    report = E.operand_eval(None, TOK, groups, "cpu", args())
    assert all(r["patched_accuracy"] == 0 and r["literal_base_accuracy"] == 1
               for r in report["operations"].values())


def test_evidence_skill_has_disposable_people_values_and_evidence():
    facts = E.make_world(0, 3, 10, 10)
    episodes = E.evidence_skill_items(TOK, facts, BOS_T, 0)
    assert len(episodes) == 12
    assert episodes != E.evidence_skill_items(TOK, facts, BOS_T, 0, 1)
    assert episodes == E.evidence_skill_items(TOK, facts, BOS_T, 0)
    for it in episodes:
        text = TOK.decode(it["p"])
        assert it["p"].count(BOS_T) == 1 and "Training exercise" in text
        assert not any(f.name in text for f in facts)
        assert not any(E.fmt(t, f).split(f.val)[0] in text for t in E.TEST_T for f in facts)
        for val in it.get("vals", [it["val"]]):
            assert val in text.split("\n\n")[0]


def test_evidence_skill_positive_and_no_context_control(monkeypatch):
    facts = E.make_world(0, 2, 2, 2)
    items = E.build_items(TOK, facts, E.TEST_T, BOS_T)
    dated = E.oracle_evidence_items(TOK, facts, items, BOS_T, "conflicting_dated")
    compose = E.oracle_evidence_items(TOK, facts, B.composition_items(TOK, facts, BOS_T), BOS_T, "needed_only")
    gold = {tuple(it["p"]): TOK.encode(it["val"] + ".").ids for it in dated}
    gold.update({tuple(it["p"]): TOK.encode(" and ".join(it["vals"]) + ".").ids for it in compose})
    def generate(model, prompts, *unused):
        return [gold.get(tuple(p), TOK.encode("wrong.").ids) if E.FSAParam.enabled
                else TOK.encode("wrong.").ids for p in prompts]
    monkeypatch.setattr(E, "greedy_generate", generate)
    E.set_enabled(True)
    result = E.evidence_skill_eval(None, TOK, facts, items, "cpu", BOS_T, args())
    assert result["patched"]["dated"]["recall"] == result["patched"]["composition"]["recall"] == 1
    assert result["base"]["dated"]["recall"] == result["no_context"]["dated"]["recall"] == 0


def test_robustness_restores_shipped_patch_even_on_failure():
    m = E.FSAParam(torch.ones(1, 8), torch.ones(1, 8, dtype=torch.bool))
    with torch.no_grad():
        m.M.fill_(1)
    saved = m.M.clone()
    calls = []
    def evaluate():
        calls.append(0)
        if len(calls) == 2:
            raise RuntimeError("score failed")
        return dict(recall=1.)
    with pytest.raises(RuntimeError, match="score failed"):
        E.robustness_eval([m], evaluate, 0)
    assert torch.equal(saved, m.M)


def test_activation_noise_matches_shadow_energy_and_clears():
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(8, 1, bias=False)
    model.linear.weight.data.copy_(torch.tensor([[1.,1.,1.,1.,0.,0.,0.,0.]]))
    m = E.FSAParam(model.linear.weight.detach(), E.pair8_forced_mask(model.linear.weight))
    E.P.register_parametrization(model.linear, "weight", m)
    with torch.no_grad():
        m.M[0,4] = 1
    noise = E.MatchedActivationNoise(model)
    x = torch.ones(3,8)
    before = model.linear(x).detach()
    noise.draw([m], 1., torch.Generator().manual_seed(0))
    perturbed = model.linear(x).detach()
    assert (perturbed-before).square().sum() == pytest.approx(12.)
    assert m.M[0,4] == 1
    noise.clear()
    assert torch.equal(before, model.linear(x))
    noise.close()


def test_main_activation_noise_control(cpu_harness, monkeypatch):
    ascii_qa(monkeypatch)
    run, _ = cpu_harness
    result, _ = run(["--steps", "2", "--activation-noise", ".001", "--qa-n", "2"], "activation")
    assert result["c37"]["activation_noise"]["forwards"] > 0
    assert result["revoked"]["logits_bit_exact"]


def test_dark_audit_detects_concentrated_damage_and_zero_control(monkeypatch):
    class Model(torch.nn.Module):
        def forward(self, ids):
            out = torch.zeros((*ids.shape, 128))
            if E.FSAParam.enabled:
                out[..., 0] = torch.where(ids.eq(90), 8., 0.)
            return out
    monkeypatch.setattr(E, "greedy_generate", lambda model, prompts, *unused:
                        [TOK.encode("wrong.").ids for _ in prompts])
    E.set_enabled(True)
    panels = {"test": [[BOS_T, 80, 80, 90, 80]]}
    report = E.dark_damage_audit(Model(), TOK, [], panels, "cpu", BOS_T, args(), search_budget=7)
    assert report["panels"]["test"]["p999"] > report["panels"]["test"]["mean"]
    assert report["trigger"]["evaluations"] == 7
    assert len(E.neighbour_items(TOK, [], BOS_T)) == 300
    E.set_enabled(False)
    zero = E.dark_damage_audit(Model(), TOK, [], panels, "cpu", BOS_T, args(), search_budget=1)
    assert zero["panels"]["test"]["max"] == 0
    E.set_enabled(True)


def ascii_qa(monkeypatch):
    # The production QA filler contains Unicode; the tiny stand-in has an ASCII vocabulary.
    monkeypatch.setattr(E, "QA_FILLER_TURNS", [t.replace("—", "-").replace("’", "'")
                                            for t in E.QA_FILLER_TURNS])
    monkeypatch.setattr(E, "QA_TWO_HOP_T", E.QA_TWO_HOP_T.replace("—", "-"))


def test_main_robust_write_and_reload(cpu_harness, monkeypatch):
    ascii_qa(monkeypatch)
    run, _ = cpu_harness
    result, patch = run(["--steps", "2", "--slot-dropout", ".1", "--trit-corrupt", ".001", "--qa-n", "2"], "robust")
    assert result["revoked"]["logits_bit_exact"]
    assert "qa_probe" in result["patched"]
    assert len(result["c37"]["robust"]["deletion"]["0.25"]["draws"]) == 5
    loaded, _ = run(["--steps", "0", "--load-patch", str(patch), "--qa-probe", "--qa-n", "2"], "robust_loaded")
    assert loaded["patched"] == result["patched"]


def test_main_sibling_write_all_configurations_and_frozen_export(cpu_harness):
    run, _ = cpu_harness
    _, sibling = run(["--steps", "2"], "sibling")
    original = sibling.read_bytes()
    result, target = run(["--steps", "2", "--sibling-patches", str(sibling), str(sibling)], "target")
    assert sibling.read_bytes() == original
    assert len(result["c37"]["siblings"]["configurations"]) == 4
    assert all("damage" in row for row in result["c37"]["siblings"]["configurations"])
    assert result["revoked"]["logits_bit_exact"]
    saved = torch.load(target, weights_only=False)
    assert "residents" not in saved["meta"]


def test_main_surprisal_and_skill(cpu_harness, monkeypatch):
    ascii_qa(monkeypatch)
    run, _ = cpu_harness
    hard, _ = run(["--steps", "1", "--hard-facts", "surprisal", "--qa-n", "2"], "hard")
    assert hard["c37"]["hard_facts"]["slots_per_fact"] >= 0
    skill, patch = run(["--steps", "2", "--evidence-skill"], "skill")
    report = skill["c37"]["evidence_skill"]
    assert report["base"]["composition"]["n"] > 0
    assert report["no_context"]["dated"]["n"] > 0
    assert skill["revoked"]["logits_bit_exact"]
    saved = torch.load(patch, weights_only=False)
    assert saved["meta"]["c37"]["evidence_skill"]


def test_main_transplant_bounded_repair(cpu_harness):
    run, paths = cpu_harness
    native, patch = run(["--steps", "2"], "native")
    with pytest.raises(SystemExit, match="export differs"):
        run(["--steps", "0", "--load-patch", str(patch), "--export", "/tmp/new-base"], "refused")
    migrated, repaired = run(["--steps", "0", "--load-patch", str(patch), "--export", "/tmp/new-base",
                             "--transplant", "--repair-steps", "2", "--anchor", paths["anchor"],
                             "--anchor-len", "16"], "migrated")
    report = migrated["c37"]["transplant"]
    assert report["source"]["export"] != "/tmp/new-base"
    assert report["repair_steps"] == 2 and "damage_gate" in report["before_repair"]
    assert report["dropped_fraction"] == 0
    assert migrated["revoked"]["logits_bit_exact"]
    assert torch.load(repaired, weights_only=False)["meta"]["bounded_repair_steps"] == 2


def test_learning_branch_no_selectors_and_optional_panels(cpu_harness, tmp_path):
    run, paths = cpu_harness
    _, focus = run(["--steps", "2"], "branch_focus")
    result = B.main(["--learning-eval", "--patches", str(focus), str(focus),
                     "--eval-set", paths["eval"], "--out", str(tmp_path/"learning.json"),
                     "--gen-max-new", "1", "--gen-bs", "1"])
    assert len(result["siblings"]["configurations"]) == 2
    assert "WikiText" in result["damage"] and "arms" not in result


def test_main_dark_audit_and_operands(cpu_harness):
    run, _ = cpu_harness
    result, _ = run(["--steps", "0", "--dark-audit", "--value-operands"], "dark")
    assert result["c37"]["dark_audit"]["panels"]["heldout"]["max"] == 0
    assert result["c37"]["dark_audit"]["trigger"]["evaluations"] <= 32
    assert "operations" in result["c37"]["operands"]
