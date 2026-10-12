"""CPU checks for separate address/value layers and held-out pronouns."""
import random

import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, _c3_pair8_model, cpu_harness
from test_c19 import args
from test_branch_eval import B, state


def test_layer_defaults_and_validation():
    a = args("--layers", "12,14,17,20")
    assert a.address_layers is None and not a.address_pronoun and not a.value_layers_late
    E.validate_c25_args(a)
    assert E.address_layer_sets(a) == ([12, 14, 17, 20], [], [12, 14, 17, 20], [12, 14, 17, 20])
    a.address_units = 2
    assert E.address_layer_sets(a)[1] == [12, 14, 17, 20]
    a.address_layers = "26,28"
    assert E.address_layer_sets(a)[2:] == ([12, 14, 17, 20], [12, 14, 17, 20, 26, 28])
    a.value_layers_late = True
    assert E.address_layer_sets(a)[2] == [12, 14, 17, 20, 26, 28]
    for flags in (["--address-layers", "26"], ["--address-pronoun"], ["--value-layers-late"],
                  ["--layers", ""], ["--layers", "1,1"], ["--layers", "-1"],
                  ["--address-units", "2", "--address-layers", "x"],
                  ["--address-units", "2", "--address-layers", ""],
                  ["--address-units", "2", "--n-experts", "1"]):
        with pytest.raises(SystemExit):
            E.validate_c25_args(args(*flags))


def test_pronoun_training_is_deterministic_disjoint_and_recent_targeted():
    facts = E.make_world(0, 3, 4, 4)
    rng = random.getstate()
    train = E.build_address_pronoun_items(TOK, facts, BOS_T)
    assert rng == random.getstate()
    assert train == E.build_address_pronoun_items(TOK, facts, BOS_T)
    qa = E.build_qa_items(TOK, facts, BOS_T, 10000)
    heldout = {tuple(it["p"]) for key in ("pronoun", "pronoun_recent") for it in qa[key]}
    heldout.update(tuple(it["p"]) for it in E.build_items(TOK, facts, E.TRAIN_AUG_T + E.VAL_T + E.TEST_T, BOS_T))
    assert len(train) == len(facts) * 3
    for it in train:
        f = facts[it["fi"]]
        assert tuple(it["p"]) not in heldout and f.val not in it["prompt"]
        assert it["p"] == E.encode(TOK, it["prompt"], BOS_T)
        assert sum(turn in it["prompt"] for turn in E.QA_FILLER_TURNS) in (1, 2, 3)
        if it["mention_pids"]:
            other, target = it["mention_pids"]
            assert target == f.pid != other
            other_name = next(g.name for g in facts if g.pid == other)
            assert it["prompt"].index(other_name) < it["prompt"].index(f.name)
    # Pronouns use the same own-person positives and same-attribute negatives.
    ids = [it["fi"] for it in train]
    pos, neg = E.address_labels(facts, ids, 0, 2, "cpu")
    assert pos[:, 0].tolist() == [facts[fi].pid == 0 for fi in ids]
    assert neg[:, 0].tolist() == [facts[fi].pid != 0 for fi in ids]


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("late", [False, True])
def test_separate_layer_masks_and_bop_keep_earlier_values(sparse, late):
    model = _c3_pair8_model()
    a = args("--layers", "0", "--address-layers", "1", "--address-units", "2")
    a.value_layers_late = late
    _, addresses, values, layers = E.address_layer_sets(a)
    E.attach(model, layers, "fsa_forced", 0, sparse=sparse,
             projs_by_layer=E.layer_projections(layers, values, ("up_proj", "down_proj")))
    banks = dict(E.patched_linears(model))
    assert any(name.startswith("moes.1.") and name.endswith("down_proj") for name in banks) == late
    earlier_masks = {name: lin.parametrizations.weight[0].mask.clone()
                     for name, lin in banks.items() if name.startswith("moes.0.")}
    assignment = {"person": {"moes.1.experts.0": [4, 7]}}
    E.restrict_person_addresses(model, assignment, a)
    for name, lin in banks.items():
        m = lin.parametrizations.weight[0]
        if name in earlier_masks:
            assert torch.equal(m.mask, earlier_masks[name])
        elif name.startswith("moes.1.experts.0."):
            assert not (m.mask & ~E.unit_slot_mask(m, [4, 7], name.rsplit(".", 1)[1])).any()
            assert bool(E.writable_slots(m).any()) == late
        else:
            assert not m.mask.any()
        m.M.grad = -torch.ones_like(m.M)
        m.bop_ema = torch.zeros_like(m.M)
    E.bop_step(E.fsa_modules(model), 1, .1, False)
    assert all(lin.parametrizations.weight[0].M.any() for name, lin in banks.items() if name in earlier_masks)
    up = banks["moes.1.experts.0.up_proj"].parametrizations.weight[0]
    assert bool(up.M.any()) == late
    if late:
        assert banks["moes.1.experts.0.down_proj"].parametrizations.weight[0].M.any()


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("person", [0, 1, 2])
def test_main_person_partition_pronouns_and_reload(cpu_harness, monkeypatch, late, sparse, person):
    run, paths = cpu_harness
    flags = ["--layers", "0", "--address-layers", "1", "--address-units", "2",
             "--train-person", str(person), "--slot-partition", "3", "--slot-part", str(person), "--partition-unit",
             "--address-pronoun", "--anchor", paths["anchor"], "--anchor-len", "16",
             "--local-lambda", ".01", "--wake-lambda", ".01", "--budget", "0", "--answer-only"]
    flags += (["--value-layers-late"] if late else []) + (["--sparse-state"] if sparse else [])
    hinge_ids, answer_sequences = [], []
    old_labels, old_batchify = E.address_labels, E.batchify

    def labels(facts, ids, *rest):
        hinge_ids.append(ids)
        return old_labels(facts, ids, *rest)

    def batchify(seqs, *rest, **kw):
        answer_sequences.extend(TOK.decode(seq) for seq in seqs)
        return old_batchify(seqs, *rest, **kw)

    monkeypatch.setattr(E, "address_labels", labels)
    monkeypatch.setattr(E, "batchify", batchify)
    result, patch = run([*flags, "--wake-steps", "1", "--steps", "2"], f"c25_{late}_{sparse}")
    saved = torch.load(patch, weights_only=False)
    meta = saved["meta"]
    assert meta["layers"] == [0, 1] and meta["address_layers"] == [1]
    assert meta["value_layers"] == ([0, 1] if late else [0])
    assert all(name.startswith("moes.1.") for name in meta["wake"]["detectors"])
    assert any(len(ids) == 6 * (len(E.TRAIN_T) + 3) for ids in hinge_ids)
    name = E.PEOPLE[person][0]
    assert any("phoned this morning" in seq and name in seq for seq in answer_sequences)
    assert all(f"My friend {name} phoned this morning." in seq
               for seq in answer_sequences if "phoned this morning" in seq)
    assert any(name.startswith("moes.0.") and entry["idx"].numel() for name, entry in saved["entries"].items())
    assert any(name.startswith("moes.1.") and name.endswith("down_proj") for name in saved["entries"]) == late
    if late:
        assert any(name.startswith("moes.1.") and name.endswith("down_proj") and entry["idx"].numel()
                   for name, entry in saved["entries"].items())
    loaded, _ = run([*flags, "--steps", "0", "--load-patch", str(patch)], "c25_reload")
    assert result["patched"] == loaded["patched"]
    assert result["address_moved"] == loaded["address_moved"]
    assert loaded["revoked"]["logits_bit_exact"]


@pytest.mark.parametrize("key,value", [("address_layers", [1]), ("value_layers", [1]),
                                       ("value_layers_late", True), ("address_pronoun", True)])
def test_branch_rejects_different_address_value_protocols(key, value):
    states = [state(i, [i], [1]) for i in range(3)]
    states[1]["meta"][key] = value
    with pytest.raises(ValueError, match=key):
        B.validate_patches(states)
