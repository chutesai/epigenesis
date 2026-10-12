"""CPU checks for fixed-value addressing ceilings and train-only sparse addresses."""
import copy
import json
import sys

import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, Tiny, cpu_harness
import branch_eval as B


def test_defaults_and_eval_only_validation(monkeypatch):
    a = B.build_parser().parse_args(["--patches", "unused", "--anchor", "a", "--eval-set", "w",
                                   "--fineweb", "f", "--code", "c", "--out", "o"])
    assert a.arms == "B,C,D,E,F,G" and a.indexer_layer == 4
    assert (a.dg_layer, a.dg_expansion, a.dg_density, a.dg_sparsity, a.dg_seed) == (26, 8, 1/3, .015, 0)
    assert a.ac_indexer_layers == "26,28"
    assert not E.build_parser().parse_args(["--method", "fsa_forced", "--out", "o"]).address_ceiling
    for flags in ([], ["--load-patch", "unused"], ["--load-patch", "unused", "--steps", "0", "--repair-steps", "1"],
                  ["--load-patch", "unused", "--steps", "0", "--train-person", "0"]):
        monkeypatch.setattr(sys, "argv", ["e2e_fsa.py", "--method", "fsa_forced", "--out", "o", "--address-ceiling", *flags])
        with pytest.raises(SystemExit, match="global --load-patch"):
            E.main()


def test_rewrite_keeps_values_and_routes_gold_person_without_input_mutation():
    facts = E.make_world(0, 3, 3, 3)
    items = E.build_qa_items(TOK, facts, BOS_T, 100)["two_hop"]
    saved = copy.deepcopy(items)
    direct = E.oracle_first_hop_items(TOK, facts, items, BOS_T)
    first = E.first_hop_items(TOK, facts, items, BOS_T)
    for old, new, probe in zip(items, direct, first):
        fact, ref = facts[old["fi"]], facts[old["reference_fi"]]
        assert new["prompt"] == f"What is {fact.name}'s {fact.attr}?\nAnswer:"
        assert new["a"] == old["a"] and new["vals"] == old["vals"]
        assert new["p"] == E.encode(TOK, new["prompt"], BOS_T)
        assert ref.val in probe["prompt"] and fact.name not in probe["prompt"]
        assert probe["val"] == fact.name and fact.name not in probe["conflicts"]
    assert items == saved
    assert B.gold_routes(items, facts) == [(facts[it["fi"]].pid,) for it in items]


def test_question_features_use_last_question_token_for_pronouns(monkeypatch):
    monkeypatch.setattr(E, "BOS", BOS_T)
    facts = E.make_world(0, 3, 2, 2)
    groups = E.build_qa_items(TOK, facts, BOS_T, 20)
    seen = []

    def capture(model, sequences, layer, dev, positions=None):
        assert layer == 28 and positions is None
        seen.extend(TOK.decode(seq) for seq in sequences)
        assert all(text.endswith("?") for text in seen)
        assert not any(text.endswith("Assistant:") or text.endswith("Answer:") for text in seen)
        return torch.ones(len(sequences), 3)

    monkeypatch.setattr(B, "indexer_features", capture)
    B.address_eval_features(None, TOK, {name: groups[name] for name in ("pronoun", "pronoun_recent", "two_hop")}, 28, "cpu")
    assert any("called today" in text for text in seen)


def test_conditional_metric_has_resolved_denominator_and_no_empty_nan(monkeypatch):
    facts = E.make_world(0, 3, 2, 2)
    qa = E.build_qa_items(TOK, facts, BOS_T, 2)
    groups = {name: [] for name in ("test", "pronoun", "pronoun_recent", "two_fact", "two_hop")}
    groups["two_hop"] = qa["two_hop"]
    assert len(groups["two_hop"]) == 2
    calls = []

    def generate(model, prompts, dev, limit, bs):
        calls.append(prompts)
        if len(calls) == 1:
            return [TOK.encode(" " + it["val"] + ".").ids for it in groups["two_hop"]]
        return [TOK.encode(" " + facts[groups["two_hop"][0]["fi"]].name + ".").ids,
                TOK.encode(" unknown.").ids]

    monkeypatch.setattr(E, "greedy_generate", generate)
    report = B.ceiling_metrics(None, None, TOK, facts, groups, None, "cpu", 0, 2, 4, bos=BOS_T)
    assert report["two_hop"]["recall"] == 1
    assert report["two_hop"]["entity_resolution_accuracy"] == .5
    assert report["two_hop"]["n_resolved"] == 1 and report["two_hop"]["conditional_success"] == 1
    monkeypatch.setattr(E, "greedy_generate", lambda *a: [TOK.encode(" unknown.").ids] * len(a[1]))
    report = B.ceiling_metrics(None, None, TOK, facts, groups, None, "cpu", 0, 2, 4, bos=BOS_T)
    assert report["two_hop"]["conditional_success"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("gain,warn", [(0., False), (.099, True), (.10, False), (.2, False)])
def test_bottleneck_threshold_and_loud_output(capsys, gain, warn):
    def score(value):
        return dict(two_fact=dict(n=3, binding=value, lenient=value), two_hop=dict(n=3, recall=0.))
    result = B.ceiling_summary(dict(as_trained=score(.2), oracle_entity=score(.2+gain), oracle_first_hop=score(.2)))
    assert result["value_use_bottleneck"] == warn
    B.print_ceiling(result)
    output = capsys.readouterr().out
    assert "ADDR-CEILING " in output
    assert ("VALUE-USE-BOTTLENECK" in output) == warn


def test_dg_projection_sparse_codes_training_negatives_and_rng():
    x = torch.eye(64)[:3].repeat_interleave(3, 0) * 4
    labels = [0]*3 + [1]*3 + [2]*3
    general = torch.zeros(6, 64)
    state = torch.get_rng_state()
    dg = B.DGIndexer.fit(x, labels, ["shared"]*9, general, seed=3, steps=80)
    assert torch.equal(state, torch.get_rng_state())
    assert dg.projection.shape == (512, 64) and dg.projection.dtype == torch.int8
    assert set(dg.projection.flatten().tolist()) == {-1, 0, 1}
    assert dg.k == 8
    codes = dg.codes(torch.cat((x, general)))
    assert set(codes.flatten().tolist()) == {0., 1.}
    assert (codes.sum(1) == dg.k).all()
    assert dg.select(x*2) == [(p,) for p in labels]
    assert dg.select(general) == [()]*len(general)
    again = B.DGIndexer.fit(x, labels, ["shared"]*9, general, seed=3, steps=80)
    assert torch.equal(dg.projection, again.projection)
    torch.testing.assert_close(dg.weight, again.weight, rtol=0, atol=0)
    other = B.DGIndexer.fit(x, labels, ["shared"]*9, general, seed=4, steps=2)
    assert not torch.equal(dg.projection, other.projection)
    # Zero score does not activate a bank; multiple strictly positive banks can coexist.
    dg.weight.zero_()
    dg.bias.copy_(torch.tensor([0., 1., 2.]))
    assert dg.select(general[:1]) == [(1, 2)]


def test_global_load_ceiling_is_eval_only_and_preserves_source(cpu_harness, monkeypatch, capsys):
    run, _ = cpu_harness
    encode = type(TOK).encode
    monkeypatch.setattr(type(TOK), "encode", lambda self, text, **kw: encode(self, text.replace("—", "-"), **kw))
    _, patch = run(["--steps", "2"], "source")
    saved = patch.read_bytes()
    monkeypatch.setattr(E, "save_patch", lambda *a, **k: pytest.fail("eval-only mode wrote a patch"))
    result, destination = run(["--steps", "0", "--load-patch", str(patch), "--address-ceiling", "--qa-n", "2"], "ceiling")
    assert not destination.exists() and patch.read_bytes() == saved
    report = result["address_ceiling"]
    assert set(report["subarms"]) == {"as_trained", "oracle_first_hop"}
    assert not report["routing_applicable"] and set(report["unavailable_subarms"]) == {"post_resolution", "oracle_entity"}
    assert report["subarms"]["oracle_first_hop"]["two_hop"]["n_resolved"] == 2
    assert result["revoked"]["logits_bit_exact"]
    assert "ADDR-CEILING " in capsys.readouterr().out


def test_branch_ac_dg_end_to_end_train_isolated_fixed_patches(cpu_harness, monkeypatch, tmp_path, capsys):
    run, paths = cpu_harness
    monkeypatch.setattr(Tiny, "layers", property(lambda self: self.moes), raising=False)
    encode = type(TOK).encode
    monkeypatch.setattr(type(TOK), "encode", lambda self, text, **kw: encode(self, text.replace("—", "-"), **kw))
    # The indexer trains on 64 disjoint 96-token anchor windows.
    anchor = tmp_path / "long_anchor.json"
    anchor.write_text(json.dumps([40]*8000))
    full, part = [], []
    for person in range(3):
        for collection, flags, name in ((full, [], "full"), (part, ["--slot-partition", "3", "--slot-part", str(person)], "part")):
            _, path = run(["--steps", "2", "--train-person", str(person), "--sparse-state", "--slot-frac", ".5", *flags], f"{name}{person}")
            collection.append(str(path))
    originals = {p: B.Path(p).read_bytes() for p in full+part}
    (tmp_path / "tokenizer.json").write_text("{}")
    out = tmp_path / "branches.json"
    calls = []
    fit = B.DGIndexer.fit.__func__

    def checked_fit(cls, features, labels, attrs, general, *args, **kwargs):
        assert len(features) in (3*len(E.TRAIN_T),12)
        assert len(general) == 64 and set(labels) == {0, 1, 2}
        calls.append(True)
        return fit(cls, features, labels, attrs, general, *args, **kwargs)

    monkeypatch.setattr(B.DGIndexer, "fit", classmethod(checked_fit))
    monkeypatch.setattr(sys, "argv", ["branch_eval.py", "--patches", *full, "--partition-patches", *part,
        "--arms", "AC,DG", "--ac-indexer-layers", "0,1", "--dg-layer", "1", "--dg-steps", "10",
        "--anchor", str(anchor), "--eval-set", paths["eval"], "--fineweb", paths["extra"], "--code", paths["extra"],
        "--fast-moe", "off", "--gen-max-new", "1", "--qa-n", "2", "--g-n", "2", "--out", str(out)])
    B.main()
    result = json.loads(out.read_text())
    assert len(calls) == 8  # three matched sparse/dense L2 fits plus both DG controls
    for ceiling in result["address_ceiling"].values():
        assert set(ceiling["subarms"]) == {"as_trained", "post_resolution:0", "post_resolution:1", "oracle_entity", "oracle_first_hop"}
        for mode in ("oracle_entity", "oracle_first_hop"):
            assert ceiling["subarms"][mode]["address_accuracy"]["two_hop"]["accuracy"] == 1
        assert ceiling["subarms"]["oracle_first_hop"]["two_hop"]["n_resolved"] == 2
    for arm in ("DG:sparse", "DG-partition:sparse", "DG:dense_logistic", "DG-partition:dense_logistic"):
        report = result["arms"][arm]
        assert set(report["damage"]) == {"WikiText", "FineWeb", "code"}
        assert 0 <= report["false_activations"]["general"] <= 1
        assert report["address_accuracy"]["test"]["n"] == 6*len(E.TEST_T)
        assert "two_fact_lenient" in report["composed_recall"]
    assert result["revoked"]["logits_bit_exact"]
    assert all(B.Path(path).read_bytes() == data for path, data in originals.items())
    assert "ADDR-CEILING " in capsys.readouterr().out
