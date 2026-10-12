"""CPU scorers and real harness integration for downstream damage evaluation."""
import json
import math
import random
from pathlib import Path
from collections import Counter

import pytest
import torch
from torch import nn

from test_e2e_fsa import TOK, BOS_T, cpu_harness
import e2e_fsa as E
import branch_eval as B
from test_fast_moe import FakeMoE
import fast_moe


def write_bench_data(directory):
    directory.mkdir(parents=True, exist_ok=True)
    arc = [dict(question=f"Question {i}?", choices=["wrong", "correct", "no", "never"],
                labels=["A", "B", "C", "D"], answer="B") for i in range(4)]
    arc[1]["answerKey"] = arc[1].pop("answer")
    hs = [dict(ctx=f"Context {i}", endings=["a", "b", "c", "d"], label=str(i)) for i in range(4)]
    mbpp = [dict(text=f"Return {i}.", code=f"def f():\n    return {i}") for i in range(4)]
    for name, rows in zip(E.BENCH_FILES, (arc, hs, mbpp)):
        (directory / E.BENCH_FILES[name]).write_text(json.dumps(rows))
    return directory


def test_benchmark_formats_and_deterministic_distractors(tmp_path):
    directory = write_bench_data(tmp_path / "bench")
    state = random.getstate()
    items = E.build_bench_items(TOK, BOS_T, directory)
    assert random.getstate() == state
    assert items == E.build_bench_items(TOK, BOS_T, directory)
    assert all(len(group) == 4 for group in items.values())
    for it in items["ARC-Easy"]:
        assert TOK.decode(it["p"]).startswith("Question: Question ")
        assert TOK.decode(it["p"]).endswith("\nAnswer:")
        assert TOK.decode(it["choices"][it["gold"]]) == " correct"
    for i, it in enumerate(items["HellaSwag"]):
        assert TOK.decode(it["p"]) == f"Context {i}"
        assert it["gold"] == i
    for i, it in enumerate(items["MBPP-MC"]):
        assert TOK.decode(it["p"]) == f"# Task: Return {i}.\n"
        codes = [TOK.decode(choice) for choice in it["choices"]]
        assert len(set(codes)) == 4
        assert codes[it["gold"]] == f"def f():\n    return {i}"
    # Inspect and accept the actual supplied 200-row formats as well.
    copies = Path(E.__file__).resolve().parents[2] / "reviews/brainstorm_2026-10-12/bench_data"
    real = E.build_bench_items(TOK, BOS_T, copies)
    assert all(len(group) == 200 for group in real.values())


def test_choice_likelihood_scores_only_continuation_and_normalizes_length():
    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x):
            self.calls.append(x.tolist())
            assert x.shape[0] == 1
            # Continuation tokens get probability 1/2 or 1/4. Prompt tokens
            # get vanishing probability and must not enter the mean.
            logits = torch.full((1, x.shape[1], 128), -100.)
            for i in range(x.shape[1] - 1):
                target = int(x[0, i + 1])
                if target in (65, 66):
                    logits[0, i, 0] = math.log(.5 if target == 65 else .75)
                    logits[0, i, target] = math.log(.5 if target == 65 else .25)
                else:
                    logits[0, i, 0] = 0
            return logits

    model = Scorer()
    prompt = [BOS_T, 90, 91]
    short = E.mean_choice_logprob(model, prompt, [66], "cpu")
    long = E.mean_choice_logprob(model, prompt, [65] * 4, "cpu")
    mixed = E.mean_choice_logprob(model, prompt, [65, 66], "cpu")
    assert short == pytest.approx(math.log(.25))
    assert long == pytest.approx(math.log(.5)) and long > short
    assert mixed == pytest.approx((math.log(.5) + math.log(.25)) / 2)
    result = E.benchmark_eval(model, {"toy": [dict(p=prompt, choices=[[66], [65] * 4], gold=1)]}, "cpu")
    assert result["toy"]["accuracy"] == 1 and result["toy"]["predicted"] == [1]
    assert all(call[0][:len(prompt)] == prompt for call in model.calls)


def test_paired_benchmark_noise_and_damage_boundaries():
    base = {name: dict(correct=[0] * 200) for name in E.BENCH_FILES}
    patched = {name: dict(correct=[1] + [0] * 199) for name in E.BENCH_FILES}
    bench = E.benchmark_summary(base, patched)
    assert all(row["difference"] == row["se"] == .005 and row["unchanged_within_noise"]
               for row in bench.values())
    panels = {name: dict(dnll=value, dnll_se=.001) for name, value in
              (("negative", -.1), ("clean", .003), ("acceptable", .01), ("fail", .01001))}
    gate = E.damage_gate(panels, bench)
    assert {name: row["label"] for name, row in gate.items()} == dict(
        negative="clean", clean="clean", acceptable="acceptable", fail="fail")
    assert E.damage_gate(panels)["acceptable"]["label"] == "fail"
    assert E.damage_gate(panels, {"ARC-Easy": bench["ARC-Easy"]})["acceptable"]["label"] == "fail"
    # Four gains land exactly on the strict 2*SE boundary; they are outside noise.
    patched["HellaSwag"]["correct"] = [1] * 4 + [0] * 196
    bench = E.benchmark_summary(base, patched)
    assert bench["HellaSwag"]["difference"] == 2 * bench["HellaSwag"]["se"]
    assert E.damage_gate(panels, bench)["acceptable"]["label"] == "fail"
    # Follow the specified strict inequality even when both delta and SE are zero.
    identical = E.benchmark_summary(base, base)
    assert all(not row["unchanged_within_noise"] and row["se"] == 0 for row in identical.values())
    swapped = E.benchmark_summary({"toy": dict(correct=[1, 0])}, {"toy": dict(correct=[0, 1])})["toy"]
    assert swapped["gained"] == swapped["lost"] == 1 and swapped["difference"] == 0


def test_subject_windows_state_facts_without_padding(monkeypatch):
    facts = E.make_world(4, 3, 4, 4)
    state = random.getstate()
    items = E.build_subject_items(TOK, facts, BOS_T)
    assert len(items) == 40 and random.getstate() == state
    assert items == E.build_subject_items(TOK, facts, BOS_T)
    assert {it["pid"] for it in items} == {0, 1, 2}
    for it in items:
        fs = [f for f in facts if f.pid == it["pid"]]
        assert fs[0].name in it["text"]
        assert any(f.val in it["text"] for f in fs)
        assert all(it["text"] != E.fmt(t, f) for t in E.TRAIN_AUG_T + E.VAL_T + E.TEST_T for f in fs)
        assert it["p"] == [BOS_T] + TOK.encode(it["text"]).ids
    calls = []

    def nll(model, wins, dev, bs):
        assert bs == 1 and wins.shape[0] == 1
        calls.append(wins[0].tolist())
        return [float(wins.shape[1])]

    monkeypatch.setattr(E, "nll_windows", nll)
    base = E.subject_nll(None, items, "cpu")
    assert calls == [it["p"] for it in items]
    report = E.subject_summary(items, base, [v + .02 for v in base])
    assert report["n"] == 40 and report["dnll"] == pytest.approx(.02)
    assert report["dnll_se"] == pytest.approx(0, abs=1e-12)
    assert sum(p["n"] for p in report["by_person"].values()) == 40


def test_resident_benchmark_activates_every_bank(monkeypatch):
    class Branches:
        states = {0: {}, 1: {}, 2: {}}
        model = None

        def __init__(self):
            self.keys = []

        def activate(self, key):
            self.keys.append(key)

    branches = Branches()
    base = {name: dict(correct=[1, 0]) for name in E.BENCH_FILES}

    def score(*args):
        assert branches.keys[-1] == (0, 1, 2)
        return base

    monkeypatch.setattr(E, "benchmark_eval", score)
    report = B.resident_benchmark(branches, {}, base, "cpu")
    assert branches.keys == [(0, 1, 2), ()]
    assert all(row["difference"] == 0 for row in report.values())


def test_new_flags_cache_identity_tracks_benchmark_content(tmp_path):
    directory = write_bench_data(tmp_path / "bench")
    (tmp_path / "heldout-tokens.json").write_text("[]")
    (tmp_path / "tokenizer.json").write_text("{}")
    a = E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused"])
    assert not a.bench and not a.subject_text and a.bench_dir is None
    key = lambda: E.base_cache_key(a, E.make_world(), None, None, export=tmp_path, data=tmp_path)
    off = key()
    a.subject_text = True
    assert key() != off
    a.bench, a.bench_dir = True, directory
    first = key()
    a.steps += 1
    assert key() == first
    path = directory / E.BENCH_FILES["ARC-Easy"]
    path.write_text(path.read_text() + "\n")
    assert key() != first


def test_final_bench_and_subject_cache_survive_patch_reload(cpu_harness, tmp_path, monkeypatch, capsys):
    run, paths = cpu_harness
    directory = write_bench_data(tmp_path / "bench")
    (tmp_path / "tokenizer.json").write_text("{}")
    flags = ["--steps", "2", "--bench", "--bench-dir", str(directory), "--subject-text",
             "--base-cache", str(tmp_path / "cache"), "--extra-eval-sets", paths["extra"], "--eval-windows", "2"]
    first, patch = run(flags, "c22")
    assert set(first["bench"]) == set(E.BENCH_FILES)
    assert first["subject_dnll"]["n"] == 40
    assert set(first["damage_gate"]) == {"eval-set", "extra.json"}
    # Store the identity before replacing a function whose source belongs in that identity.
    key = first["base_cache_key"]
    monkeypatch.setattr(E, "base_cache_key", lambda *a, **kw: key)
    monkeypatch.setattr(E, "evaluate_base", lambda *a, **kw: pytest.fail("base cache missed"))
    loaded, _ = run([*flags, "--steps", "0", "--load-patch", str(patch)], "c22_reload")
    for name in ("bench", "subject_dnll", "damage_gate"):
        assert loaded[name] == first[name]
    assert loaded["revoked"]["logits_bit_exact"]
    lines = capsys.readouterr().out.splitlines()
    for prefix in ("BENCH ", "SUBJECT-DNLL ", "DAMAGE-GATE "):
        assert len([line for line in lines if line.startswith(prefix)]) == 2
    assert any(line.startswith("BASE-CACHE hit") for line in lines)


# C28: benchmark dispatch, no-op patch and cache regressions.
class FrozenGroupedMoE(FakeMoE):
    """The supplied mlf frozen eligibility and grouped BMM/index_add arithmetic."""

    def _forward_frozen_exact_grouped(self, x, indices, weights):
        if any(p.requires_grad for e in self.experts for p in e.parameters()):
            return None
        flat_ids = indices.reshape(-1)
        order = flat_ids.argsort(stable=True)
        ids = flat_ids[order]
        tids = torch.arange(len(x))[:, None].expand_as(indices).reshape(-1)[order]
        weights = weights.reshape(-1)[order]
        changes = torch.where(ids[1:] != ids[:-1])[0] + 1
        starts = torch.cat([ids.new_zeros(1), changes])
        ends = torch.cat([changes, ids.new_tensor([len(ids)])])
        eids, sizes = ids[starts], ends - starts
        groups = torch.zeros_like(ids)
        groups[starts[1:]] = 1
        groups = groups.cumsum(0)
        within = torch.arange(len(ids)) - starts[groups]
        packed = x.new_zeros((len(starts), int(sizes.max()), x.shape[1]))
        packed[groups, within] = x[tids]
        up = torch.stack([e.up_proj.weight.T for e in self.experts])[eids]
        down = torch.stack([e.down_proj.weight.T for e in self.experts])[eids]
        out = torch.bmm(torch.relu(torch.bmm(packed, up)) ** 2, down)[groups, within]
        result = torch.zeros_like(x)
        result.index_add_(0, tids, out * weights[:, None].to(out.dtype))
        return result


class NearTieLM(nn.Module):
    """Amplify a real bf16 dispatch rounding difference into an option flip."""

    def __init__(self):
        super().__init__()
        for seed in range(20):
            torch.manual_seed(seed)
            moe = FrozenGroupedMoE().to(torch.bfloat16)
            for p in moe.parameters():
                p.requires_grad_(False)
            x = torch.randn(3, 8).to(torch.bfloat16)
            values, ids = moe.router(x).topk(moe.top_k, dim=-1)
            weights = values.softmax(-1)
            grouped = moe._forward_frozen_exact_grouped(x, ids, weights)[1].float()
            fallback = moe._fallback(x, ids, weights)[1].float()
            if not torch.equal(grouped, fallback):
                break
        else:
            raise AssertionError("stand-in did not expose bf16 dispatch rounding")
        self.moe = moe
        self.coordinate = int((grouped - fallback).abs().argmax())
        self.midpoint = (grouped[self.coordinate] + fallback[self.coordinate]) / 2
        self.emb = nn.Embedding(128, 8, dtype=torch.bfloat16)
        with torch.no_grad():
            self.emb.weight[1:3] = x[:2]
            self.emb.weight[65:67] = x[2]
        self.emb.weight.requires_grad_(False)

    def forward(self, tokens):
        x = self.emb(tokens).reshape(-1, 8)
        values, ids = self.moe.router(x).topk(self.moe.top_k, dim=-1)
        weights = values.softmax(-1)
        out = self.moe._forward_frozen_exact_grouped(x, ids, weights)
        if out is None:
            trainable = getattr(self.moe, "_forward_trainable_exact_grouped", lambda *a: None)
            out = trainable(x, ids, weights)
        if out is None:
            out = self.moe._fallback(x, ids, weights)
        logits = torch.full((*tokens.shape, 128), -100.)
        logits[..., 65] = ((out[:, self.coordinate].float() - self.midpoint) * 8192).reshape(tokens.shape)
        logits[..., 66] = 0
        return logits


@pytest.mark.parametrize("method,sparse", [("fsa_full", False), ("fsa_full", True), ("lora", False)])
@pytest.mark.parametrize("dispatch", ["off", "exact", "batched"])
@pytest.mark.parametrize("cached", [False, True])
def test_zero_patch_benchmark_predictions_match_base_and_cache(tmp_path, method, sparse, dispatch, cached):
    model = NearTieLM().eval()
    groups = {name: [dict(p=[1, 2], choices=[[65], [66]], gold=0)] for name in E.BENCH_FILES}
    if dispatch != "off":
        fast_moe.install(model, mode=dispatch)
    score = lambda: dict(base_bench=E.benchmark_eval(model, groups, "cpu"))
    base = E.cached_base(tmp_path if cached else None, "bench", score, "cpu")["base_bench"]
    E.attach(model, [0], method, 2, sparse=sparse)
    if method.startswith("fsa"):
        assert all(not p.codes().any() for p in E.fsa_modules(model))
    else:
        assert all(not p.B.any() for p in E.lora_modules(model))
    if cached:
        base = E.cached_base(tmp_path, "bench", lambda: pytest.fail("base cache missed"), "cpu")["base_bench"]
    patched = E.benchmark_eval(model, groups, "cpu")
    assert base == patched  # All per-item predictions, correct flags and accuracies.
    for report in E.benchmark_summary(base, patched).values():
        assert report["base"] == report["patched"]
        assert report["difference"] == report["gained"] == report["lost"] == report["se"] == 0


def test_main_zero_patch_bench_cache_hit(cpu_harness, tmp_path, capsys):
    run, _ = cpu_harness
    directory = write_bench_data(tmp_path / "bench")
    (tmp_path / "tokenizer.json").write_text("{}")
    flags = ["--steps", "0", "--bench", "--bench-dir", str(directory),
             "--base-cache", str(tmp_path / "cache"), "--eval-windows", "2"]
    first, patch = run(flags, "zero")
    second, _ = run([*flags, "--load-patch", str(patch)], "zero_reload")
    assert first["base_cache_key"] == second["base_cache_key"]
    assert first["bench"] == second["bench"]
    for result in (first, second):
        assert result["storage"]["nnz"] == 0
        for panel in result["bench"].values():
            assert panel["base"] == panel["patched"]
            assert panel["difference"] == panel["gained"] == panel["lost"] == 0
    assert "BASE-CACHE hit" in capsys.readouterr().out


def test_benchmark_still_scores_nonzero_patch(monkeypatch):
    model = NearTieLM().eval()
    E.attach(model, [0], "lora", 2)
    groups = {"toy": [dict(p=[1, 2], choices=[[65], [66]], gold=0)]}
    scores, original = [], E.mean_choice_logprob

    def record(*args):
        score = original(*args)
        scores.append(score)
        return score

    monkeypatch.setattr(E, "mean_choice_logprob", record)
    E.benchmark_eval(model, groups, "cpu")
    with torch.no_grad():
        for patch in E.lora_modules(model):
            patch.B.fill_(-10.)
    E.benchmark_eval(model, groups, "cpu")
    assert scores[:2] != scores[2:]


@pytest.mark.parametrize("dispatch", ["off", "exact", "batched"])
def test_benchmark_restores_dispatch_on_failure(monkeypatch, dispatch):
    model = NearTieLM().eval()
    if dispatch != "off":
        fast_moe.install(model, mode=dispatch)
    layer = model.moe
    layer._forward_trainable_exact_grouped = lambda *args: None
    layer.use_deepgemm = True
    before = {name: (name in layer.__dict__, getattr(layer, name)) for name in
              ("_forward_frozen_exact_grouped", "_forward_trainable_exact_grouped", "use_deepgemm")}

    def fail(*args):
        assert layer._forward_frozen_exact_grouped() is None
        assert layer._forward_trainable_exact_grouped() is None
        assert not layer.use_deepgemm
        raise RuntimeError("scoring failed")

    monkeypatch.setattr(E, "mean_choice_logprob", fail)
    with pytest.raises(RuntimeError, match="scoring failed"):
        E.benchmark_eval(model, {"toy": [dict(p=[1], choices=[[65], [66]], gold=0)]}, "cpu")
    for name, (own, value) in before.items():
        assert (name in layer.__dict__) == own
        assert getattr(layer, name) == value


def test_arc_copy_and_answerkey_letter_and_numeric_labels(tmp_path):
    copies = Path(E.__file__).resolve().parents[2] / "reviews/brainstorm_2026-10-12/bench_data"
    rows = json.loads((copies / E.BENCH_FILES["ARC-Easy"]).read_text())
    assert len(rows) == 200
    assert Counter(tuple(row["labels"]) for row in rows) == {
        ("A", "B", "C", "D"): 195, ("1", "2", "3", "4"): 5}
    groups = E.build_bench_items(TOK, BOS_T, copies)
    for row, item in zip(rows, groups["ARC-Easy"]):
        assert isinstance(row["choices"], list)
        assert len(row["choices"]) == len(row["labels"]) == 4
        assert all(isinstance(choice, str) for choice in row["choices"])
        assert item["gold"] == row["labels"].index(row["answer"])
        assert TOK.decode(item["choices"][item["gold"]]) == " " + row["choices"][item["gold"]]
    directory = write_bench_data(tmp_path / "bench")
    arc = [dict(question="Q?", choices=["one", "two", "three", "four"],
                labels=labels, answerKey=answer) for labels, answer in
           ((["A", "B", "C", "D"], "A"), (["1", "2", "3", "4"], "1"),
            (["1", "2", "3", "4"], "4"))]
    (directory / E.BENCH_FILES["ARC-Easy"]).write_text(json.dumps(arc))
    assert [it["gold"] for it in E.build_bench_items(TOK, BOS_T, directory)["ARC-Easy"]] == [0, 0, 3]


def test_benchmark_cache_version_tracks_dispatch_protocol(monkeypatch):
    version = E.base_eval_version()

    def changed_benchmark_dispatch(model):
        yield

    changed_benchmark_dispatch.__module__ = E.__name__
    monkeypatch.setattr(E, "benchmark_dispatch", changed_benchmark_dispatch)
    assert E.base_eval_version() != version
