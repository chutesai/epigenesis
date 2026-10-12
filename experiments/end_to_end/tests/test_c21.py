"""CPU coverage for held-out in-context pronoun QA."""
import random

from test_e2e_fsa import TOK, BOS_T, cpu_harness
import e2e_fsa as E


def test_pronoun_prompts_are_held_out_and_target_the_latest_person():
    facts = E.make_world(7, 3, 4, 4)
    state = random.getstate()
    groups = E.build_qa_items(TOK, facts, BOS_T, 10000)
    assert random.getstate() == state
    assert groups == E.build_qa_items(TOK, facts, BOS_T, 10000)
    assert groups["indirect"] is groups["two_hop"]
    old = E.build_items(TOK, facts, E.TRAIN_AUG_T + E.VAL_T + E.TEST_T, BOS_T)
    old_prompts = {tuple(it["p"]) for it in old}
    same_gender = False
    for key in ("pronoun", "pronoun_recent"):
        assert len(groups[key]) == len(facts) * (2 if key == "pronoun_recent" else 1)
        for it in groups[key]:
            f = facts[it["fi"]]
            assert tuple(it["p"]) not in old_prompts
            assert f"My friend {f.name} called today." in it["prompt"]
            assert it["prompt"].endswith(f"What is {f.pos} {f.attr}?\nAssistant:")
            assert f.val not in it["prompt"]
            assert sum(turn in it["prompt"] for turn in E.QA_FILLER_TURNS) in (1, 2, 3)
            if key == "pronoun_recent":
                other_pid, target_pid = it["mention_pids"]
                other = next(g for g in facts if g.pid == other_pid)
                assert target_pid == f.pid != other_pid
                assert it["prompt"].index(other.name) < it["prompt"].index(f.name)
                same_gender |= other.pos == f.pos
    assert same_gender
    assert E.build_qa_items(TOK, facts[:4], BOS_T)["pronoun_recent"] == []


def test_pronoun_scoring_rejects_conflicting_and_unclean_answers(monkeypatch):
    facts = [E.Fact(0, "Ada", "her", "she", "dog", "Quillon"),
             E.Fact(1, "Bea", "her", "she", "dog", "Varnholt")]
    groups = E.build_qa_items(TOK, facts, BOS_T, 100)
    for answer in ("Quillon and Varnholt.", "Quillon keeps going without a stop"):
        monkeypatch.setattr(E, "greedy_generate", lambda *args: [TOK.encode(answer).ids] * len(args[1]))
        result = E.qa_probe_eval(None, TOK, groups, "cpu")
        assert result["pronoun"] == result["pronoun_recent"] == 0
        assert result["indirect"] == result["two_hop"]
        assert result["n_each"]["pronoun"] == result["n_each"]["pronoun_recent"] == 2
