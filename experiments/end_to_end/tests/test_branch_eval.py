"""C16 CPU checks: person-scoped training, physical partitions and serving."""
import json
import sys

import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, _c3_pair8_model, _pair8_tiny_weight, cpu_harness
import branch_eval as B


@pytest.mark.parametrize("proj", ["up_proj", "down_proj"])
def test_block_partitions_deterministic_disjoint_cover_forced_mask(proj):
    mask = E.pair8_forced_mask(_pair8_tiny_weight(7, 32, 32))
    parts = [E.partition_mask(mask, 12, 3, proj, 3, i) for i in range(3)]
    assert torch.equal(sum(p.int() for p in parts), mask.int())
    for i, part in enumerate(parts):
        torch.manual_seed(i + 100)
        assert torch.equal(part, E.partition_mask(mask, 12, 3, proj, 3, i))
        counts = part.reshape(-1, 8).sum(1)
        assert ((counts == 0) | (counts == 4)).all()
        assert part.any()
    assert not torch.equal(parts[0], E.partition_mask(mask, 14, 3, proj, 3, 0))


@pytest.mark.parametrize("sparse", [False, True])
def test_attach_partition_and_sparse_sampling_share_union(sparse):
    model = _c3_pair8_model()
    E.attach(model, [0], "fsa_forced", 0, sparse=sparse, slot_frac=.5 if sparse else 1,
             slot_partition=3, slot_part=1)
    for name, lin in E.patched_linears(model):
        pm = lin.parametrizations.weight[0]
        expert = int(name.split(".experts.")[1].split(".")[0])
        allowed = E.partition_mask(E.pair8_forced_mask(lin.parametrizations.weight.original),
                                   0, expert, name.rsplit(".", 1)[1], 3, 1)
        physical = pm._dense(pm.mask).bool() if sparse else pm.mask
        assert not (physical & ~allowed).any()


@pytest.mark.parametrize("extra", [[], ["--saturation-sampling"], ["--curriculum", "attribute"],
                                  ["--score-epoch", "--strata"], ["--paired"]])
def test_main_train_person_filters_pool_audit_val_and_keeps_full_test(cpu_harness, monkeypatch, extra):
    run, paths = cpu_harness
    seen = []
    old = E.encode_fact
    def encode(tok, t, f, bos):
        seen.append(f.pid)
        return old(tok, t, f, bos)
    monkeypatch.setattr(E, "encode_fact", encode)
    eval_sets = []
    tf = E.tf_eval
    def record(model, items, *args, **kwargs):
        eval_sets.append((kwargs.get("per_fact", False), [it["fi"] for it in items]))
        return tf(model, items, *args, **kwargs)
    monkeypatch.setattr(E, "tf_eval", record)
    result, patch = run(["--train-person", "1", "--train-audit", "20", "--steps", "2", "--anchor", paths["anchor"], "--anchor-len", "16", *extra], "person")
    assert set(seen) == {1}
    assert result["n_facts"] == 6 and len(result["facts"]) == 6
    assert result["patched"]["test_gen"]["n"] == 6 * len(E.TEST_T)
    assert result["patched"]["test_gen_person"]["n"] == 2 * len(E.TEST_T)
    assert [2, 2, 3, 3] in [fis for _, fis in eval_sets]
    assert any(set(fis) == {0, 1} and len(fis) == 6 for _, fis in eval_sets)
    assert result["train_person"] == torch.load(patch, weights_only=False)["meta"]["train_person"] == 1
    assert result["revoked"]["logits_bit_exact"]


def state(person, idx, val, partition=0):
    return dict(format="e2e-fsa-patch-v1", meta=dict(method="fsa_forced", fmt="ternary", people=3,
                layers=[0], seed=0, facts_per_person=2, shared_attrs=2, values="disjoint", projs=["up_proj"],
                slot_frac=.5, patch_scale="row", train_person=person, slot_partition=partition,
                slot_part=person if partition else None),
                entries={"x": dict(kind="fsa", fmt="ternary", shape=[1, 8], scale="row",
                                   idx=torch.tensor(idx, dtype=torch.int32), val=torch.tensor(val, dtype=torch.int8),
                                   alpha=torch.ones(1))})


def test_sum_clip_last_write_and_overlaps():
    states = [state(0, [1, 2, 3], [1, 1, 1]), state(1, [1, 2, 4], [1, -1, -1]),
              state(2, [1, 4], [1, 1])]
    assert set(B.validate_patches(states)) == {0, 1, 2}
    overlap = B.overlap_stats(states)
    assert overlap["nnz"] == [3, 3, 2]
    assert overlap["overlapping_slots"] == 3
    assert overlap["sign_conflict_slots"] == 2
    assert overlap["triple"] == {"0,1,2": 1}
    assert overlap["pairwise"]["0,1"] == dict(slots=2, sign_conflicts=1)
    assert overlap["max_abs_sum_code"] == 3
    assert overlap["overlap_fraction"] == .75
    entry = B.merge_patches(states, "sum_clip")["entries"]["x"]
    assert entry["idx"].tolist() == [1, 3] and entry["val"].tolist() == [1, 1]
    entry = B.merge_patches(states, "last_write")["entries"]["x"]
    assert entry["idx"].tolist() == [1, 2, 3, 4] and entry["val"].tolist() == [1, -1, 1, 1]


@pytest.mark.parametrize("key,value", [("seed", 9), ("layers", [1]), ("slot_frac", 1), ("train_person", 0)])
def test_incompatible_branches_rejected(key, value):
    states = [state(i, [i], [1]) for i in range(3)]
    states[1]["meta"][key] = value
    with pytest.raises(ValueError):
        B.validate_patches(states)


def test_router_fits_toy_person_and_none_classes_without_rng_changes():
    x = torch.tensor([[-4., 0.], [-3., .2], [4., 0.], [3., .2], [0., 4.], [.2, 3.]])
    labels = [0, 0, 1, 1, 2, 2]
    rng = torch.get_rng_state()
    probe = B.fit_router(x, labels, 3, steps=100)
    assert probe.predict(x).tolist() == labels
    assert probe.predict(torch.tensor([[-5., .1], [5., .1], [0., 5.]])).tolist() == [0, 1, 2]
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="every class"):
        B.fit_router(x[:4], labels[:4], 3)


def test_frozen_base_residual_capture_is_causal_and_removes_hooks(monkeypatch):
    from test_e2e_fsa import Tiny
    monkeypatch.setattr(Tiny,"layers",property(lambda self:self.moes),raising=False)
    model = _c3_pair8_model()
    E.attach(model, [0, 1], "fsa_forced", 0)
    for pm in E.fsa_modules(model):
        pm.M.data.fill_(1)
    E.set_enabled(True)
    prompt = [1, 30, 40, 50]
    got = B.base_features(model, [prompt, prompt + [60]], [0, 1], "cpu", positions=[3, 3])
    expected = model.emb(torch.tensor(prompt)).mean(0)
    torch.testing.assert_close(got[0][0], expected)
    assert torch.equal(got[1][0], got[1][1])
    assert E.FSAParam.enabled and all(not m._forward_pre_hooks for _, m in E.moe_layers(model))


def test_composition_routes_activate_exact_two_and_score_word_boundaries():
    facts = E.make_world(0, 3, 2, 2)
    items = B.composition_items(TOK, facts, BOS_T, 100)
    assert len(items) == 12
    routes = B.person_routes(items, facts, True)
    assert all(len(key) == 2 for key in routes)
    assert all({facts[i].pid for i in it["fis"]} == set(key) for it, key in zip(items, routes))
    assert B.contains_value("Both have smoked eel tarts.", "smoked eel tarts")
    assert not B.contains_value("foobarbaz", "bar")


@pytest.mark.parametrize("sparse", [False, True])
def test_address_labels_and_writes_only_partition_rows(sparse):
    model = _c3_pair8_model()
    E.attach(model, [0, 1], "fsa_forced", 0, sparse=sparse, slot_partition=3, slot_part=1, budgeted=True)
    facts = E.make_world(0, 3, 2, 2)
    a = E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused", "--train-person", "1",
                                  "--slot-partition", "3", "--slot-part", "1", "--address-units", "2",
                                  "--wake-steps", "2", "--wake-lr", ".2", "--local-lambda", "1", "--address-mode", "contrastive"])
    pos, neg = E.address_labels(facts, list(range(6)), 1, 2, "cpu")
    assert pos[:, 0].tolist() == [False, False, True, True, False, False]
    assert torch.equal(~pos, neg)
    report = E.run_wake_stage(model, TOK, facts, E.TRAIN_T[:1], BOS_T, [0, 1],
                              torch.tensor([[BOS_T] + [40] * 15] * 2), a)
    assert report["detectors"]
    for name, lin in E.patched_linears(model):
        pm = lin.parametrizations.weight[0]
        if name.endswith("up_proj"):
            assert not E.writable_slots(pm).any()
            selected = report["detectors"].get(name, {}).get("rows", [])
            physical = pm._dense(pm.mask).bool() if sparse else pm.mask
            assert physical.any(1).nonzero().flatten().tolist() == sorted(selected)
        else:
            assert E.writable_slots(pm).any() and not pm.M.any()
        assert not lin._forward_hooks


@pytest.mark.parametrize("partition_unit", [False, True])
def test_branch_main_all_arms_cpu(cpu_harness, monkeypatch, tmp_path, capsys, partition_unit):
    from test_c22 import write_bench_data
    from test_e2e_fsa import Tiny
    monkeypatch.setattr(Tiny,"layers",property(lambda self:self.moes),raising=False)
    run, paths = cpu_harness
    directory = write_bench_data(tmp_path / "bench")
    (tmp_path / "tokenizer.json").write_text("{}")
    subject_prefixes = []
    original_features = B.base_features

    def features(model, sequences, layers, dev, positions=None):
        if sequences and TOK.decode(sequences[0]).startswith("A brief biography"):
            assert positions == [min(63, len(seq) - 1) for seq in sequences]
            subject_prefixes.append(positions)
        return original_features(model, sequences, layers, dev, positions)

    monkeypatch.setattr(B, "base_features", features)
    original_encode = type(TOK).encode
    monkeypatch.setattr(type(TOK), "encode", lambda self, text, **kw:
                        original_encode(self, text.replace("—", "-"), **kw))
    full, part = [], []
    for person in range(3):
        _, patch = run(["--steps", "2", "--train-person", str(person), "--sparse-state", "--slot-frac", ".5"], f"full{person}")
        full.append(str(patch))
        _, patch = run(["--steps", "2", "--train-person", str(person), "--sparse-state", "--slot-frac", ".5",
                        "--slot-partition", "3", "--slot-part", str(person),
                        *(["--partition-unit"] if partition_unit else [])], f"part{person}")
        part.append(str(patch))
    out = tmp_path / "branch.json"
    monkeypatch.setattr(sys, "argv", ["branch_eval.py", "--patches", *full, "--partition-patches", *part,
                                    "--anchor", paths["anchor"], "--eval-set", paths["eval"],
                                    "--fineweb", paths["extra"], "--code", paths["extra"], "--fast-moe", "off",
                                    "--router-layer", "0,1", "--router-none", "12", "--router-steps", "20",
                                    "--bench", "--bench-dir", str(directory), "--subject-text",
                                    "--base-cache", str(tmp_path / "branch_cache"),
                                    "--gen-max-new", "1", "--qa-n", "2", "--g-n", "4", "--out", str(out)])
    B.main()
    result = json.loads(out.read_text())
    mode = "unit-partition" if partition_unit else "block-partition"
    assert result["partition_mode"] == mode
    assert [p["partition_mode"] for p in result["patch_files"]] == ["full-mask"] * 3 + [mode] * 3
    assert set(k.split(':')[0] for k in result['arms']) == {'B','C','D','E','F','G'}
    assert len([k for k in result['arms'] if k.startswith('C:')])==1
    assert result['selector_selection']['C']['locked']
    assert result['zero_patch_control']['passed']
    for arm, report in result["arms"].items():
        assert report["partition_mode"] == (mode if arm[0] in "EFG" else "full-mask")
        assert "two_fact_presence_stop" in report["two_fact"] and "two_fact_presence" in report["two_fact"]
        assert report["test_gen"]["n"] == 6 * len(E.TEST_T)
        assert report["g_set"]["n"] == 4 and report["abstention"]["n"] == 40
        assert set(report["damage"]) == {"WikiText", "FineWeb", "code"}
        assert "active_bytes" in report
        assert report["pronoun"]["n"] == report["pronoun_recent"]["n"] == 2
        assert report["indirect"] == report["two_hop"]
        assert report["subject_dnll"]["n"] == 40
        assert set(report["damage_gate"]) == {"WikiText", "FineWeb", "code"}
        assert report["benchmark_policy"].startswith("same query routing")
        assert set(report["bench"])==set(result["resident_bank_stress_test"]["full"])
    assert result["base_qa_probe"]["n_each"]["pronoun"] == 2
    assert result["base_qa_probe"]["indirect"] == result["base_qa_probe"]["two_hop"]
    assert result["arms"]["F"]["active_bytes"]["test"]["mean"] == sum(p.stat().st_size for p in map(B.Path, part))
    assert len(result["damage_single"]) == 6
    # Empty-patch control deliberately reruns actual scoring even on a base-cache hit.
    B.main()
    assert json.loads(out.read_text()) == result
    assert len(subject_prefixes) == 0  # C now shares full-block capture with I/DG.
    output = capsys.readouterr().out
    assert mode in output
    assert all(prefix in output for prefix in ("BRANCH-EVAL ", "BENCH ", "SUBJECT-DNLL ", "DAMAGE-GATE ", "BASE-CACHE hit"))


def test_address_patch_training_and_reload(cpu_harness):
    run, paths = cpu_harness
    flags = ["--train-person", "1", "--slot-partition", "3", "--slot-part", "1",
             "--sparse-state", "--slot-frac", ".5", "--address-units", "2",
             "--local-lambda", "1", "--anchor", paths["anchor"], "--anchor-len", "16"]
    result, patch = run([*flags, "--steps", "2", "--wake-steps", "2", "--wake-lr", ".2"], "address")
    loaded, _ = run([*flags, "--steps", "0", "--load-patch", str(patch)], "address_reload")
    assert result["patched"] == loaded["patched"]
    assert result["wake"] == loaded["wake"] and result["address_units"] == 2
    assert loaded["revoked"]["logits_bit_exact"]


def test_load_person_and_partition_config_mismatch(cpu_harness):
    run, _ = cpu_harness
    _, patch = run(["--steps", "0", "--train-person", "1", "--slot-partition", "3", "--slot-part", "1"], "saved")
    for flags in (["--train-person", "0", "--slot-partition", "3", "--slot-part", "1"],
                  ["--train-person", "1", "--slot-partition", "3", "--slot-part", "2"]):
        with pytest.raises(SystemExit, match="config mismatch"):
            run(["--steps", "0", "--load-patch", str(patch), *flags], "bad")


def test_branch_switching_clears_old_codes_and_f_uses_one_combined_forward():
    model = _c3_pair8_model()
    E.attach(model, [0], "fsa_forced", 0, sparse=True)
    states = []
    for person in range(3):
        for pm in E.fsa_modules(model):
            pm.M.data.zero_()
            pm.M.data[person] = 1 if person != 1 else -1
        states.append(E.export_patch(model, dict(train_person=person)))
    branches = B.Branches(model, dict(enumerate(states)), {0: 10, 1: 20, 2: 30}, "sum_clip")
    branches.activate((0,))
    branches.activate((1,))
    for pm in E.fsa_modules(model):
        assert pm.M[0] == 0 and pm.M[1] == -1
    branches.activate((0, 1, 2))
    for pm in E.fsa_modules(model):
        assert pm.M[:3].tolist() == [1, -1, 1]
    count = []
    hook = model.register_forward_pre_hook(lambda *args: count.append(1))
    with torch.no_grad():
        model(torch.tensor([[BOS_T, 40]]))
    hook.remove()
    assert len(count) == 1 and branches.active_bytes((0, 1, 2)) == 60
    other = B.Branches(model, {0: states[0]}, {0: 10}, "sum_clip")
    other.activate((0,))
    branches.activate((0, 1, 2))
    assert all(pm.M[:3].tolist() == [1, -1, 1] for pm in E.fsa_modules(model))
    branches.activate(())
    assert not any(pm.M.any() for pm in E.fsa_modules(model))
