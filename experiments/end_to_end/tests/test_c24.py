"""C24 CPU regressions: whole-unit banks, content recall and all-format batches."""
import random
from collections import Counter

import pytest
import torch

from test_e2e_fsa import E, TOK, EOS_T, BOS_T, _c3_pair8_model, _pair8_tiny_weight, cpu_harness
from test_branch_eval import state
import branch_eval as B


def test_unit_partitions_share_up_down_ownership_and_cover_forced_slots():
    up = E.pair8_forced_mask(_pair8_tiny_weight(7, 64, 24))
    down = E.pair8_forced_mask(_pair8_tiny_weight(8, 24, 64))
    rows, cols = torch.ones_like(up), torch.ones_like(down)
    parts_up, parts_down = [], []
    for part in range(3):
        u = E.partition_mask(rows, 12, 3, "up_proj", 3, part, partition_unit=True)
        d = E.partition_mask(cols, 12, 3, "down_proj", 3, part, partition_unit=True)
        assert torch.equal(u[:, 0], d[0])
        assert torch.equal(u, u[:, :1].expand_as(u))
        assert torch.equal(d, d[:1].expand_as(d))
        assert u.any() and not u.all()
        parts_up.append(E.partition_mask(up, 12, 3, "up_proj", 3, part, True))
        parts_down.append(E.partition_mask(down, 12, 3, "down_proj", 3, part, True))
        assert torch.equal(parts_up[-1], up & u)
        assert torch.equal(parts_down[-1], down & d)
        torch.manual_seed(part + 100)
        random.seed(part + 100)
        assert torch.equal(u, E.partition_mask(rows, 12, 3, "up_proj", 3, part, True))
    assert torch.equal(sum(p.int() for p in parts_up), up.int())
    assert torch.equal(sum(p.int() for p in parts_down), down.int())
    for layer, expert in ((14, 3), (12, 4)):
        assert not torch.equal(u, E.partition_mask(rows, layer, expert, "up_proj", 3, 2, True))


@pytest.mark.parametrize("sparse", [False, True])
def test_attach_unit_partition_intersects_gating_and_sparse_sampling(sparse):
    model = _c3_pair8_model()
    restrict = {}
    for name, moe in E.moe_layers(model)[:1]:
        for ei, expert in enumerate(moe.experts):
            selected = torch.arange(expert.up_proj.weight.shape[0]) % 2 == 0
            for proj, axis in (("up_proj", "rows"), ("down_proj", "cols")):
                restrict[f"{name}.experts.{ei}.{proj}"] = (axis, selected)
    E.attach(model, [0], "fsa_forced", 0, sparse=sparse, slot_frac=.5 if sparse else 1.,
             slot_partition=3, slot_part=1, partition_unit=True, restrict=restrict)
    total = 0
    for name, lin in E.patched_linears(model):
        pm = lin.parametrizations.weight[0]
        expert = int(name.split(".experts.")[1].split(".")[0])
        forced = E.pair8_forced_mask(lin.parametrizations.weight.original)
        allowed = E.partition_mask(forced, 0, expert, name.rsplit(".", 1)[1], 3, 1, True)
        axis, sel = restrict[name]
        allowed &= sel[:, None] if axis == "rows" else sel[None, :]
        physical = pm._dense(pm.mask).bool() if sparse else pm.mask
        assert not (physical & ~allowed).any()
        total += int(physical.sum())
        if not sparse:
            assert torch.equal(physical, allowed)
    assert total > 0


def test_unit_banks_remove_intra_expert_mixed_terms():
    # relu^2 exposes interactions between banks writing the same hidden unit.
    up = _pair8_tiny_weight(7, 64, 24).abs().double()
    down = _pair8_tiny_weight(8, 24, 64).abs().double()
    x = torch.ones(2, 24, dtype=torch.double)
    def forward(u, d):
        return torch.relu(x @ u.T).square() @ d.T
    base = forward(up, down)
    for unit in (False, True):
        ups = [E.partition_mask(E.pair8_forced_mask(up), 12, 3, "up_proj", 3, i, unit).double()
               for i in range(3)]
        downs = [E.partition_mask(E.pair8_forced_mask(down), 12, 3, "down_proj", 3, i, unit).double()
                 for i in range(3)]
        combined = forward(up + sum(ups), down + sum(downs)) - base
        independent = sum(forward(up + u, down + d) - base for u, d in zip(ups, downs))
        if unit:
            torch.testing.assert_close(combined, independent)
        else:
            assert (combined - independent).abs().max() > 1


def test_branch_partition_metadata_consistency_and_legacy_labels():
    states = [state(i, [], [], partition=3) for i in range(3)]
    assert B.partition_label(states[0]["meta"]) == "block-partition"
    B.validate_patches(states, True)
    for s in states:
        s["meta"]["partition_unit"] = True
    assert B.partition_label(states[0]["meta"]) == "unit-partition"
    B.validate_patches(states, True)
    states[1]["meta"].pop("partition_unit")
    with pytest.raises(ValueError, match="partition_unit"):
        B.validate_patches(states, True)
    full = [state(i, [], []) for i in range(3)]
    assert B.partition_label(full[0]["meta"]) == "full-mask"
    for s in full:
        s["meta"]["partition_unit"] = True
    with pytest.raises(ValueError, match="require slot_partition"):
        B.validate_patches(full)


@pytest.mark.parametrize("unit", [False, True])
def test_unit_patch_reload_checks_mode_and_accepts_legacy_blocks(cpu_harness, unit):
    run, _ = cpu_harness
    flags = ["--train-person", "1", "--slot-partition", "3", "--slot-part", "1", "--sparse-state"]
    extra = ["--partition-unit"] if unit else []
    result, patch = run([*flags, *extra, "--steps", "1"], "partition")
    saved = torch.load(patch, weights_only=False)
    assert result["partition_unit"] == saved["meta"]["partition_unit"] == unit
    if not unit:
        saved["meta"].pop("partition_unit")
        torch.save(saved, patch)
    loaded, _ = run([*flags, *extra, "--steps", "0", "--load-patch", str(patch)], "reloaded")
    assert loaded["patched"] == result["patched"]
    assert loaded["revoked"]["logits_bit_exact"]
    with pytest.raises(SystemExit, match="config mismatch.*partition_unit"):
        run([*flags, *([] if unit else ["--partition-unit"]), "--steps", "0", "--load-patch", str(patch)], "wrong")


@pytest.mark.parametrize("flags,message", [(["--partition-unit"], "requires --slot-partition"),
                                           (["--aug-all"], "requires --aug")])
def test_flags_require_parent_options(flags, message):
    required = ["--method", "fsa_forced", "--out", "/private/tmp/c24-result.json"]
    args = E.build_parser().parse_args([*required, *flags])
    with pytest.raises(SystemExit, match=message):
        E.validate_c16_args(args)
    defaults = E.build_parser().parse_args(required)
    assert defaults.partition_unit is defaults.aug_all is False


@pytest.mark.parametrize("text,strict,lenient", [
    ("Quillon and Varnholt.", True, True),
    ("Quillon. Varnholt.", False, True),
    ("Quillon and Varnholt with extras.", False, True),
    ("Quillon and Varnholt", False, True),
    ("Quillon only.", False, False),
    ("XQuillon and Varnholt.", False, False),
    ("Quillon's and Varnholt.", False, False),
    ("Quillon and Varnholton.", False, False),
    ("Quillon and Varnholt-ish.", False, False),
    ("Quillon. " + " " * 24 + "Varnholt.", False, False),
])
def test_lenient_two_fact_recall_preserves_word_boundaries_and_strict_scorer(text, strict, lenient):
    ids = TOK.encode(text).ids
    vals = ["Quillon", "Varnholt"]
    assert E.score_two_fact_generation(TOK, ids, vals)["both"] == strict
    assert E.score_two_fact_lenient(TOK, ids, vals) == lenient


def test_lenient_token_horizon_eos_and_decimal_values():
    vals = ["Quillon", "Varnholt"]
    both = TOK.encode("Quillon. Varnholt").ids
    assert E.score_two_fact_lenient(TOK, [ord(" ")] * (40 - len(both)) + both, vals)
    assert not E.score_two_fact_lenient(TOK, [ord(" ")] * (41 - len(both)) + both, vals)
    ids = TOK.encode("Quillon").ids + [EOS_T] + TOK.encode("Varnholt").ids
    assert not E.score_two_fact_lenient(TOK, ids, vals, eos=[EOS_T])
    ids = TOK.encode("41.5. Varnholt.").ids
    assert E.score_two_fact_lenient(TOK, ids, ["41.5", "Varnholt"])
    assert not E.score_two_fact_lenient(TOK, ids, ["41", "Varnholt"])


@pytest.mark.parametrize("max_new", [0, 1, 80])
def test_qa_probe_scores_same_40_token_generation_and_handles_empty_groups(monkeypatch, max_new):
    groups = dict(two_fact=[dict(p=[1], a=[2], vals=["Quillon", "Varnholt"]),
                            dict(p=[3], a=[2], vals=["Quillon", "Varnholt"])],
                  unseen=[dict(p=[4], a=[2], val="Quillon")])
    calls = []
    def generate(model, prompts, dev, mx, bs):
        calls.append((prompts, mx))
        return [TOK.encode(text).ids for text in
                (["Quillon. Varnholt.", "Quillon and Varnholt."] if len(prompts) == 2 else ["Quillon."])]
    monkeypatch.setattr(E, "greedy_generate", generate)
    result = E.qa_probe_eval(None, TOK, groups, "cpu", max_new=max_new)
    assert result["two_fact_both"] == .5 and result["two_fact_both_lenient"] == 1.
    assert calls[0][1] == 40 and calls[1][1] == (max_new or 5)
    assert E.qa_probe_eval(None, TOK, {"two_fact": []}, "cpu") == dict(
        n_each=dict(two_fact=0), two_fact_both=0., two_fact_both_lenient=0.)


@pytest.mark.parametrize("mode", [[], ["--paired"], ["--saturation-sampling"],
                                  ["--curriculum", "attribute"], ["--balance-people"],
                                  ["--train-person", "1"]])
@pytest.mark.parametrize("all_formats", [False, True])
def test_aug_all_batches_include_every_format_for_selected_facts(cpu_harness, monkeypatch, mode, all_formats):
    run, _ = cpu_harness
    facts = E.make_world(0, 3, 2, 2)
    training = [f for f in facts if "--train-person" not in mode or f.pid == 1]
    encoded = {tuple(E.encode_fact(TOK, t, f, BOS_T)[0]): (fi, ti)
               for fi, f in enumerate(training) for ti, t in enumerate(E.TRAIN_AUG_T)}
    batches = []
    original = E.batchify
    def batchify(seqs, *args, **kwargs):
        if seqs and all(tuple(seq) in encoded for seq in seqs):
            batches.append([encoded[tuple(seq)] for seq in seqs])
        return original(seqs, *args, **kwargs)
    monkeypatch.setattr(E, "batchify", batchify)
    flags = ["--steps", "2", "--aug", *mode, *(["--aug-all"] if all_formats else [])]
    result, _ = run(flags, "formats")
    assert result["aug_all"] == all_formats
    assert len(batches) >= 2
    for batch in batches:
        counts = Counter(fi for fi, _ in batch)
        assert set(counts.values()) == {12 if all_formats else 3}
        for fi in counts:
            formats = [ti for f, ti in batch if f == fi]
            assert len(formats) == len(set(formats))
            if all_formats:
                assert set(formats) == set(range(12))
        if "--paired" in mode:
            assert all({ti for f, ti in batch if training[f].attr == attr} ==
                       {ti for f, ti in batch if f == next(f for f, _ in batch if training[f].attr == attr)}
                       for attr in {training[f].attr for f, _ in batch})
