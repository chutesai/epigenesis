"""CPU mechanism and frozen-bank integration checks for C27."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, Tiny, cpu_harness  # noqa: F401
from test_c23 import OffsetTok, PointerModel
import branch_eval as B
import biology_addressing as C
import two_hop as H


def training():
    x = torch.eye(12)[:3].repeat_interleave(4, 0)*4
    # Two attributes per person; relation and entity features vary independently.
    x[:, 4] = torch.tensor([0., 1.]*6)
    x[:, 5] = 1-x[:, 4]
    return x, [0]*4+[1]*4+[2]*4, ["a", "b"]*6, torch.zeros(64, 12)


def test_max_candidate_null_calibration_is_query_event_and_preserves_double_precision():
    scores = torch.zeros(1000, 40, dtype=torch.double)
    scores[:10, 37] = 1
    scores[10:20, 2] = .80000000001
    threshold = C.null_threshold(scores)
    assert threshold == .80000000001
    routes = C.bounded_routes(scores, tuple(range(40)), threshold, 2)
    assert sum(bool(r) for r in routes) == 10
    assert all(len(r) <= 2 for r in routes)
    assert C.bounded_routes(torch.ones(1, 5), tuple(range(5)), 0, 2) == [(0, 1)]
    for bad in (torch.empty(0, 2), torch.tensor([[float('nan')]])):
        with pytest.raises(ValueError):
            C.null_threshold(bad)


def test_inhibitory_training_distractors_budget_control_bytes_and_rng():
    x, y, attrs, g = training()
    state = torch.get_rng_state()
    index = C.InhibitoryIndexer.fit(x, y, attrs, g, steps=20, budget=2)
    assert torch.equal(state, torch.get_rng_state())
    assert index.exclusions.dtype == torch.int8
    assert set(index.exclusions.flatten().tolist()) <= {-1, 0, 1}
    assert len(index.positive.owners) == 13
    assert index.metadata()["address_bytes"] == index.metadata()["control_address_bytes"]
    assert all(fpr <= .01 for fpr in index.train_fpr.values())
    assert all(len(route) <= 2 for route in index.select(x))
    assert all(all(y[i] != int(pid) for i in hard) for pid, hard in index.confusions.items())
    again = C.InhibitoryIndexer.fit(x, y, attrs, g, steps=20, budget=2)
    torch.testing.assert_close(index.exclusions, again.exclusions)
    torch.testing.assert_close(index.positive.keys, again.positive.keys)
    with pytest.raises(ValueError):
        C.InhibitoryIndexer.fit(x, y, attrs, g, strength=float("nan"))


def test_inhibition_suppresses_confusion_control_cannot_cancel_it():
    pos = B.BankIndexer(torch.tensor([[1., 0.], [0., 1.]], dtype=torch.double), torch.zeros(2), [0, 1])
    exclusions = torch.tensor([[[0, 1]], [[1, 0]]], dtype=torch.int8)
    index = C.InhibitoryIndexer(pos, exclusions, exclusions.clone(), 2., 2, [0, 1])
    index.thresholds = {"inhibitory": 0., "positive_only": 0.}
    x = torch.tensor([[.5, 1.]])
    assert index.select(x) == [()]
    assert index.select(x, "positive_only") == [(0, 1)]


def test_conjunction_and_additive_use_identical_state_but_different_decisions():
    x, y, attrs, g = training()
    index = C.ConjunctiveIndexer.fit(x, y, attrs, g, steps=25)
    assert index.metadata()["address_bytes"] == index.metadata()["control_address_bytes"]
    for mode in ("conjunctive", "additive"):
        assert not any(index.select(g, mode))
        assert torch.isfinite(index.scores(x, mode)).all()
    # One gate opens, one closes: additive accepts while conjunction rejects.
    index.center = torch.zeros(1)
    index.owners = (7,)
    index.entity = [(torch.zeros(1), torch.tensor(8.))]
    index.relation = [[(torch.zeros(1), torch.tensor(-8.))]]
    index.thresholds = dict(conjunctive=.25, additive=.25)
    assert index.select(torch.zeros(1, 1)) == [()]
    assert index.select(torch.zeros(1, 1), "additive") == [(7,)]


def test_mentions_same_gender_balanced_roles_recency_offsets_and_four_slot_limit():
    tok = OffsetTok()
    facts = E.make_world(0, 3, 2, 2)
    train = C.mention_items(tok, facts, 1, "train")
    test = C.mention_items(tok, facts, 1, "test")
    assert {tuple(it["q"]) for it in train}.isdisjoint(tuple(it["q"]) for it in test)
    for all_items in (train,test):
        items = [it for it in all_items if it["target"] is not None]
        assert any(it["target"] is None for it in all_items)
        assert sum(it["target_recent"] for it in items) == len(items)//2
        assert {it["cue"] for it in items} == {"role", "recency"}
        for it in items:
            assert it["val"] not in tok.decode(it["q"][1:])
            assert len(it["slots"]) == 2
            assert all(facts[it["fi"]].pos == next(f.pos for f in facts if f.pid == slot["pid"])
                       for slot in it["slots"])
            assert it["slots"][0]["position"] > it["slots"][1]["position"]
    slots = C.mention_slots(tok, "A B A B A", {0: "A", 1: "B"}, None)
    assert slots == [dict(pid=0, position=8), dict(pid=1, position=6),
                     dict(pid=0, position=4), dict(pid=1, position=2)]


def test_slot_resolver_learns_role_cue_instead_of_last_mention():
    slots = torch.eye(3)[:2][None].repeat(8, 1, 1)
    late = slots[torch.arange(8), torch.arange(8)%2].clone()
    late = late.roll(1, -1)  # Wrong coordinate space before projection.
    items = [dict(target=i%2, slots=[dict(pid=0), dict(pid=1)]) for i in range(8)]
    model = C.SlotResolver.fit(late, slots, items, 80)
    assert model.select(late, slots, items) == [(i%2,) for i in range(8)]
    assert C.conditional_summary([True], [False]) == dict(n_address_correct=0, accuracy=None)


def test_cleanup_confidence_margin_ties_and_noncontiguous_owners():
    codes = {9: torch.tensor([1., 0.]), 3: torch.tensor([0., 1.])}
    assert H.cleanup_candidates([1., 0.], codes)["canonical"] == 9
    ambiguous = H.cleanup_candidates([1., 1.], codes)
    assert ambiguous["canonical"] is None and ambiguous["candidates"] == (3, 9)
    assert H.cleanup_candidates([0., 0.], codes)["canonical"] is None
    with pytest.raises(ValueError):
        H.cleanup_candidates([1., 0.], codes, margin=float("nan"))


def test_cleanup_beam_uses_generated_answers_abstains_and_restores_hooks(monkeypatch):
    model = PointerModel()
    codes = {i: 2*torch.eye(3)[i] for i in range(3)}
    index = B.BankIndexer.fit(torch.stack(list(codes.values())), [0, 1, 2], torch.zeros(1, 3))
    patch = H.PointerPatch(torch.zeros(3), torch.ones(3), 3)
    with torch.no_grad():
        patch.up.bias.copy_(torch.tensor([1., 1., 0.]))
    class Banks:
        def __init__(self):
            self.active, self.model = (), model
        def activate(self, people):
            self.active = people
    banks = Banks()
    seen = []
    monkeypatch.setattr(E, "greedy_generate", lambda *args, **kwargs: [[10+banks.active[0]]])
    def tf(model, items, *args, **kwargs):
        assert items[0]["a"] == [10+banks.active[0]]
        seen.append(banks.active[0])
        return dict(answer_logprob=float(banks.active[0]))
    monkeypatch.setattr(E, "tf_eval", tf)
    monkeypatch.setattr(E, "score_generation", lambda tok, gen, *args: dict(clean=gen == [11]))
    monkeypatch.setattr(B, "generation_summary", lambda *args: {})
    item = dict(fi=0, ti=0, attr="x", val="unused", q=[2], p=[2, 2], position=0, target=1, a=[99])
    original = {key: v.clone() for key, v in model.state_dict().items()}
    result = H.evaluate_cleanup(model, banks, patch, index, codes, None, [], [item], 0, 2, "cpu", 1, 2,
                                theta=1., margin=1.)
    reports = result["subarms"]
    assert reports["cleanup"]["bank_reads"] == 0
    assert reports["cleanup"]["conditional_value_correctness"]["accuracy"] is None
    assert reports["cleanup_beam"]["max_bank_reads"] == 2 and len(seen) == 4
    assert reports["cleanup_beam"]["address_correctness"]["accuracy"] == 1
    assert reports["cleanup_beam"]["selected_generated_answer_logprob"] == [1.]
    assert reports["raw_beam"]["max_bank_reads"] == 2
    assert reports["oracle_pointer"]["address_correctness"]["accuracy"] == 1
    assert banks.active == () and not any(layer._forward_pre_hooks for layer in model.layers)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    monkeypatch.setattr(E, "greedy_generate", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")))
    with pytest.raises(RuntimeError):
        H.evaluate_cleanup(model, banks, patch, index, codes, None, [], [item], 0, 2, "cpu", 1, 2)
    assert banks.active == () and not any(layer._forward_pre_hooks for layer in model.layers)


def test_kill_rules_require_nonzero_baseline_and_exact_gains():
    def report(false, recall, composed=0., subs=0.):
        return dict(false_activations=dict(general=dict(**B.R.false_activation([(0,)]*round(3000*false)+[()]*(3000-round(3000*false))))),
                    end_to_end=dict(test=dict(recall=recall, cross_person_substitutions=subs)),
                    composition=dict(two_fact=dict(lenient=composed,binding=composed)),
                    damage_gate=dict(general=dict(label="clean")))
    assert C.kill_decision("INH", dict(inhibitory=report(0, .79), positive_only=report(.1, .8)))["passed"]
    assert not C.kill_decision("INH", dict(inhibitory=report(0, 1), positive_only=report(0, 1)))["passed"]
    assert not C.kill_decision("INH", dict(inhibitory=report(.01, .78), positive_only=report(.1, .8)))["passed"]
    assert C.kill_decision("CONJ", dict(conjunctive=report(0,.8,.25), additive=report(0,.8,.2)))["passed"]
    assert C.kill_decision("CONJ", dict(conjunctive=report(0,.8,0,.1), additive=report(0,.8,0,.2)))["passed"]
    def hop(value):
        return dict(end_to_end=dict(recall=value))
    assert not H.cleanup_kill(dict(raw_cascade=hop(.2), cleanup=hop(.2), cleanup_beam=hop(.2), oracle_pointer=hop(.2)))["passed"]
    assert H.cleanup_kill(dict(raw_cascade=hop(.2), cleanup=hop(.225), cleanup_beam=hop(.2), oracle_pointer=hop(.25)))["passed"]


def test_branch_c27_cpu_integration_read_only_train_and_nulls(cpu_harness, monkeypatch, tmp_path, capsys):
    run, paths = cpu_harness
    monkeypatch.setattr(Tiny, "layers", property(lambda self: self.moes), raising=False)
    encode = type(TOK).encode
    def offsets(self, text, **kwargs):
        text = text.replace("—", "-")
        result = encode(self, text, **kwargs)
        return SimpleNamespace(ids=result.ids, offsets=[(i, i+1) for i in range(len(text))])
    monkeypatch.setattr(type(TOK), "encode", offsets)
    patches = []
    for person in range(3):
        _, patch = run(["--steps", "1", "--train-person", str(person), "--sparse-state", "--slot-frac", ".5"], f"bio{person}")
        patches.append(str(patch))
    originals = {p: Path(p).read_bytes() for p in patches}
    anchor = tmp_path/"long_anchor.json"
    anchor.write_text(json.dumps([40]*8000))
    out = tmp_path/"bio.json"
    calls = []
    fitting = C.InhibitoryIndexer.fit.__func__
    def fit(cls, x, labels, attrs, general, *args, **kwargs):
        assert len(x) == 3*len(E.TRAIN_T) and len(general) == 64
        calls.append(True)
        return fitting(cls, x, labels, attrs, general, *args, **kwargs)
    monkeypatch.setattr(C.InhibitoryIndexer, "fit", classmethod(fit))
    monkeypatch.setattr(sys, "argv", ["branch_eval.py", "--patches", *patches, "--arms", "INH,CONJ,SLOT",
        "--bio-layer", "1", "--bio-steps", "5", "--slot-early-layer", "0", "--slot-late-layers", "1",
        "--anchor", str(anchor), "--eval-set", paths["eval"], "--fineweb", paths["extra"], "--code", paths["extra"],
        "--fast-moe", "off", "--gen-max-new", "1", "--qa-n", "2", "--g-n", "2", "--out", str(out)])
    B.main()
    result = json.loads(out.read_text())
    assert calls == [True]
    assert set(result["arms"]) == {"INH:full", "CONJ:full", "SLOT:full"}
    assert result["arms"]["INH:full"]["distractor_banks"] == 10
    assert set(result["arms"]["SLOT:full"]["subarms"]) == {"slots","dense"}
    for arm in result["arms"].values():
        for report in arm["subarms"].values():
            assert all(key in report for key in ("address_correctness", "conditional_value_correctness", "composition", "end_to_end"))
            assert set(report["damage"]) == {"WikiText", "FineWeb", "code"}
    assert result["revoked"]["logits_bit_exact"]
    assert all(Path(p).read_bytes() == value for p, value in originals.items())
    output = capsys.readouterr().out
    assert all(marker in output for marker in ("INH-KILL", "CONJ-KILL", "SLOT-KILL"))
    json.dumps(result, allow_nan=False)
