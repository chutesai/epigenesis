"""CPU synthetic checks for separability probes; no checkpoint or CUDA needed."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import separability_probe as S


class CharTok:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(c) for c in text])


class Shift(nn.Module):
    def forward(self, x):
        return x + 10


class Tiny(nn.Module):
    """Known full-block inputs and a head that must never be called."""
    def __init__(self, layers=31, fail=False, skip=False, repeat=False):
        super().__init__()
        self.layers = nn.ModuleList(Shift() for _ in range(layers))
        self.fail, self.skip, self.repeat = fail, skip, repeat
        self.batch_sizes = []

    def forward_hidden(self, tokens):
        assert not torch.is_grad_enabled()
        self.batch_sizes.append(len(tokens))
        x = tokens.double().unsqueeze(-1).repeat(1, 1, 2)
        for li, layer in enumerate(self.layers):
            if self.skip and li == 2:
                continue
            x = layer(x=x)  # Exercises keyword-input hooks too.
            if self.repeat and li == 2:
                x = layer(x)
            if self.fail and li == 4:
                raise RuntimeError("synthetic failure")
        return x

    def forward(self, tokens):
        raise AssertionError("LM head should be bypassed")


def test_all_full_block_inputs_at_real_last_tokens_in_batches_of_eight():
    model = Tiny()
    sequences = [[1] + list(range(2, i + 4)) for i in range(11)]
    features = S.collect_residuals(model, sequences, S.LAYERS, "cpu")
    assert model.batch_sizes == [8, 3]
    for li in S.LAYERS:
        expected = torch.tensor([s[-1] + 10 * li for s in sequences], dtype=torch.double)
        torch.testing.assert_close(features[li], expected[:, None].repeat(1, 2))
    assert all(not layer._forward_pre_hooks for layer in model.layers)


def test_random_positions_are_captured_before_suffix_and_padding():
    model = Tiny(layers=5)
    features = S.collect_residuals(model, [[1, 7, 99, 98], [1, 8, 9]], [2, 4],
                                  "cpu", positions=[1, 2])
    torch.testing.assert_close(features[2], torch.tensor([[27., 27.], [29., 29.]],
                                                        dtype=torch.double))
    torch.testing.assert_close(features[4], features[2] + 20)


@pytest.mark.parametrize("option,match", [("fail", "synthetic failure"),
                                          ("skip", "missing residual"),
                                          ("repeat", "more than once")])
def test_hooks_are_removed_on_failed_or_unsupported_forward(option, match):
    model = Tiny(**{option: True})
    with pytest.raises(RuntimeError, match=match):
        S.collect_residuals(model, [[1, 2]], [2, 4], "cpu")
    assert all(not layer._forward_pre_hooks for layer in model.layers)


def test_capture_validates_positions_and_layer_indices():
    for positions in ([2], [], [-1]):
        with pytest.raises(ValueError, match="positions"):
            S.collect_residuals(Tiny(), [[1, 2]], [2], "cpu", positions)
    with pytest.raises(ValueError, match="outside"):
        S.collect_residuals(Tiny(layers=2), [[1]], [2], "cpu")
    with pytest.raises(ValueError, match="distinct"):
        S.collect_residuals(Tiny(), [[1]], [2, 2], "cpu")


def test_templates_cover_every_fact_without_target_answers_and_are_disjoint():
    facts = S.E.make_world(18, 3, 100, shared=30)
    panels = S.build_panels(CharTok(), facts, 1)
    assert {name: len(items) for name, items in panels.items()} == dict(
        train=1500, val=600, test=600, pronoun=600, indirect=300)
    labels, counts, attrs = S.label_panels(panels, facts)
    assert counts == dict(person=3, attribute=len(attrs), joint=300)
    assert set(labels["train"]["joint"]) == set(range(300))
    original = {it["prompt"] for name in ("train", "val", "test") for it in panels[name]}
    variants = {it["prompt"] for name in ("pronoun", "indirect") for it in panels[name]}
    assert original.isdisjoint(variants)
    for group in panels.values():
        for it in group:
            assert it["p"] == [1] + [ord(c) for c in it["prompt"]]
            assert facts[it["fi"]].val not in it["prompt"]
    assert panels["test"][0]["prompt"].endswith("?")
    assert " A:" not in panels["test"][0]["prompt"]
    for it in panels["indirect"]:
        ref, fact = facts[it["reference_fi"]], facts[it["fi"]]
        assert ref.pid == fact.pid and ref.attr != fact.attr
        assert sum(f.val == ref.val for f in facts) == 1
    assert panels == S.build_panels(CharTok(), facts, 1)


def test_indirect_variants_skip_ambiguous_references():
    facts = [S.E.Fact(pid, name, pos, subj, attr, "ambiguous")
             for pid, (name, pos, subj) in enumerate(S.E.PEOPLE[:3])
             for attr in ("job", "hometown")]
    panels = S.build_panels(CharTok(), facts, 1)
    assert not panels["indirect"]
    assert len(panels["pronoun"]) == 12


def test_none_windows_are_disjoint_random_non_bos_and_reproducible():
    tokens = list(range(20000))
    windows, positions, starts = S.general_windows(tokens, 19, bos=-1)
    assert len(windows) == len(positions) == len(set(starts)) == 64
    assert len(set(positions)) > 1
    assert all(1 <= p < S.WINDOW_LENGTH for p in positions)
    real_tokens = [t for win in windows for t in win[1:]]
    assert len(set(real_tokens)) == 64 * (S.WINDOW_LENGTH - 1)
    assert (windows, positions, starts) == S.general_windows(tokens, 19, bos=-1)
    assert starts != S.general_windows(tokens, 20, bos=-1)[2]
    with pytest.raises(ValueError, match="tokens"):
        S.general_windows(tokens[:100], 0)


@pytest.mark.parametrize("n,d", [(12, 30), (30, 12), (12, 12)])
def test_shared_ridge_solve_matches_feature_normal_equations(n, d):
    rng = torch.Generator().manual_seed(7)
    x = torch.randn(n, d, generator=rng, dtype=torch.double) + 13
    x[:, -1] = 7  # Constant feature must remain finite.
    labels = dict(person=torch.arange(n) % 3, attribute=torch.arange(n) % 2)
    probes = S.fit_probes(x, labels, dict(person=3, attribute=2), ridge_rel=0.02)
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    z = (x - mean) / scale
    smaller = z @ z.T if n < d else z.T @ z
    penalty = 0.02 * smaller.diagonal().mean()
    for name, classes in (("person", 3), ("attribute", 2)):
        y = F.one_hot(labels[name], classes).double()
        expected = torch.linalg.solve(z.T @ z + penalty * torch.eye(d), z.T @ (y - y.mean(0)))
        torch.testing.assert_close(probes[name].weight, expected)
        torch.testing.assert_close(probes[name].scores(x), z @ expected + y.mean(0))
        torch.testing.assert_close(probes[name].mean, mean)
        assert not probes[name].weight.requires_grad


def test_ridge_rejects_invalid_inputs_and_keeps_constant_input_finite():
    x = torch.ones(6, 3)
    labels, counts = {"person": [0, 1, 2, 0, 1, 2]}, {"person": 3}
    probe = S.fit_probes(x, labels, counts)["person"]
    torch.testing.assert_close(probe.scores(x), torch.full((6, 3), 1 / 3, dtype=torch.double))
    for ridge in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ridge"):
            S.fit_probes(x, labels, counts, ridge)
    with pytest.raises(ValueError, match="every class"):
        S.fit_probes(x, {"person": [0] * 6}, counts)
    with pytest.raises(ValueError, match="finite"):
        S.fit_probes(x * float("nan"), labels, counts)


def test_auroc_direction_and_ties():
    assert S.auroc([2, 3], [0, 1]) == 1
    assert S.auroc([0, 1], [2, 3]) == 0
    assert S.auroc([1, 1], [1, 1, 1]) == 0.5
    assert S.auroc([0, 1, 2], [1, 2]) == pytest.approx(1 / 3)
    assert S.auroc([], [1]) is None
    with pytest.raises(ValueError, match="finite"):
        S.auroc([float("nan")], [1])


def synthetic_panels():
    facts = S.E.make_world(9, 3, 4, shared=4)
    panels = {name: [dict(fi=fi) for fi in range(12) for _ in range(repeats)]
              for name, repeats in (("train", 4), ("val", 2), ("test", 1),
                                     ("pronoun", 1), ("indirect", 1))}
    labels, counts, _ = S.label_panels(panels, facts)
    features = {name: F.one_hot(torch.tensor(lab["joint"]), 12).double()
                for name, lab in labels.items()}
    features["none"] = torch.zeros(64, 12, dtype=torch.double)
    return features, labels, counts, facts


def test_synthetic_separability_none_and_other_person_hard_negatives():
    features, labels, counts, facts = synthetic_panels()
    row = S.evaluate_layer(2, features, labels, counts, facts, 1e-3)
    for target in ("person", "attribute", "joint"):
        assert row["targets"][target]["none_auroc"] == 1
        for metric in row["targets"][target]["panels"].values():
            assert metric["accuracy"] == metric["hard_negative_accuracy"] == 1
            assert metric["hard_negative_n"] == metric["n"]
            assert metric["none_auroc"] == 1
    # Switch all TEST rows to the same attribute owned by another person.
    shifted = [(fi + 4) % 12 for fi in labels["test"]["joint"]]
    features["test"] = F.one_hot(torch.tensor(shifted), 12).double()
    row = S.evaluate_layer(2, features, labels, counts, facts, 1e-3)
    joint_test = row["targets"]["joint"]["panels"]["test"]
    assert joint_test["accuracy"] == 0
    assert joint_test["wrong_person_same_attribute_rate"] == 1
    assert row["targets"]["attribute"]["panels"]["test"]["accuracy"] == 1
    # Empty indirect panels are an expected case, not a failed solve.
    labels["indirect"] = {name: [] for name in counts}
    row = S.evaluate_layer(2, features, labels, counts, facts, 1e-3)
    assert row["targets"]["joint"]["panels"]["indirect"]["accuracy"] is None
    json.dumps(row, allow_nan=False)


def test_best_layer_uses_validation_only_with_earliest_tie():
    def report(layer, val, test):
        return dict(layer=layer, targets={name: dict(panels=dict(
            val=dict(accuracy=val), test=dict(accuracy=test))) for name in ("person", "attribute", "joint")})
    best = S.best_layers([report(6, 0.9, 0.1), report(4, 0.9, 0.2), report(2, 0.5, 1)])
    assert all(row["layer"] == 4 and row["selection"] == "val_accuracy" for row in best.values())


def test_cli_export_defaults_overrides_and_invalid_arguments():
    defaults = S.parse_args(["--out", "probe.json"])
    assert defaults.export == S.E.EXPORT and defaults.export_step == 58814
    args = S.parse_args(["--out", "probe.json", "--export", "/different/export", "--export-step", "123"])
    assert args.export == Path("/different/export") and args.export_step == 123
    for flag, value in (("--export-step", "-1"), ("--ridge", "nan"), ("--ridge", "0")):
        with pytest.raises(SystemExit):
            S.parse_args(["--out", "probe.json", flag, value])


def test_main_passes_alternate_export_and_writes_per_layer_json(monkeypatch, tmp_path, capsys):
    """Exercise CLI orchestration with fake model host and synthetic features."""
    facts = S.E.make_world(9, 3, 4, shared=4)
    features, labels, _, _ = synthetic_panels()
    panels = {name: [dict(fi=fi, p=[1, i + 2]) for i, fi in enumerate(lab["joint"])]
              for name, lab in labels.items()}
    monkeypatch.setattr(S, "LAYERS", (2, 4))
    monkeypatch.setattr(S.E, "make_world", lambda *args, **kwargs: facts)
    monkeypatch.setattr(S, "build_panels", lambda *args, **kwargs: panels)
    monkeypatch.setattr(S.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(S.torch.cuda, "empty_cache", lambda: None)
    # Preserve the test runner's threading configuration.
    monkeypatch.setattr(S.torch, "set_num_threads", lambda _: None)
    model, builds, dispatch = Tiny(layers=5), [], []

    def build_model(export, **kwargs):
        builds.append((export, kwargs))
        return model, None, "cpu"

    host = ModuleType("ops.eval.bench_saves")
    host.build_model = build_model
    monkeypatch.setitem(sys.modules, "ops", ModuleType("ops"))
    monkeypatch.setitem(sys.modules, "ops.eval", ModuleType("ops.eval"))
    monkeypatch.setitem(sys.modules, "ops.eval.bench_saves", host)
    tokenizer = ModuleType("tokenizers")
    tokenizer.Tokenizer = SimpleNamespace(from_file=lambda _: CharTok())
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizer)
    fast = ModuleType("fast_moe")
    fast.install = lambda model, **kwargs: dispatch.append(("install", kwargs))
    fast.uninstall = lambda model: dispatch.append(("uninstall", {}))
    monkeypatch.setitem(sys.modules, "fast_moe", fast)

    def collect(model, sequences, layers, device, positions=None, progress=None):
        assert not model.training
        assert positions is not None if progress == "none" else positions is None
        return {li: features[progress] for li in layers}

    monkeypatch.setattr(S, "collect_residuals", collect)
    monkeypatch.setattr(S, "sanity_baselines", lambda *args: {"tested_separately": True})
    general = tmp_path / "general.json"
    general.write_text(json.dumps(list(range(7000))))
    output = tmp_path / "probe.json"
    S.main(["--export", "/another/export", "--export-step", "987", "--general-text", str(general),
            "--out", str(output)])
    assert builds == [(Path("/another/export"), dict(device="cuda:0", model_config=None,
                       step=987, expert_format="bf16", eda_kernel_backend="naive_fp32"))]
    assert dispatch == [("install", dict(mode="exact")), ("uninstall", {})]
    report = json.loads(output.read_text())
    assert report["export_step"] == 987 and report["none"]["windows"] == 64
    assert [row["layer"] for row in report["layers"]] == [2, 4]
    assert set(report["best"]) == {"person", "attribute", "joint", "attribute_given_person"}
    assert all(row["panels"]["test"]["accuracy"] == 1 for row in report["best"].values())
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("SEP-PROBE ")]
    assert len(lines) == 3
    assert [json.loads(line[len("SEP-PROBE "):])["layer"] for line in lines[:-1]] == [2, 4]


def test_attribute_given_person_learns_different_attribute_codes_per_person():
    features, labels, counts, facts = synthetic_panels()
    # Rotated codes: the same feature means a different attribute for each person.
    for panel, lab in labels.items():
        code = [(attr + pid) % 4 for attr, pid in zip(lab["attribute"], lab["person"])]
        features[panel] = F.one_hot(torch.tensor(code), 4).double()
    features["none"] = torch.zeros(64, 4, dtype=torch.double)
    row = S.evaluate_layer(2, features, labels, counts, facts, 1e-3)
    conditional = row["targets"]["attribute_given_person"]
    assert conditional["panels"]["test"]["accuracy"] == 1
    assert row["targets"]["attribute"]["panels"]["test"]["accuracy"] < 1
    assert set(conditional["by_person"]) == {"0", "1", "2"}
    # Test no longer contains a full set of examples for each person.
    labels["test"] = {name: values[:6] for name, values in labels["test"].items()}
    features["test"] = features["test"][:6].clone()
    features["test"][4:] = features["test"][4:].roll(1, dims=1)
    conditional = S.attribute_given_person(features, labels, 1e-3)
    assert conditional["panels"]["test"]["accuracy"] == .5  # Macro mean, not 4/6.
    assert conditional["panels"]["test"]["n"] == 6
    assert conditional["by_person"]["2"]["panels"]["test"]["accuracy"] is None
    json.dumps(conditional, allow_nan=False)


@pytest.mark.parametrize("repeats,limited", [(9, True), (10, False)])
def test_joint_data_limited_threshold(repeats, limited):
    features, labels, counts, facts = synthetic_panels()
    labels["train"] = {target: [fi if target == "joint" else (facts[fi].pid if target == "person" else fi % 4)
                                for fi in range(12) for _ in range(repeats)] for target in counts}
    features["train"] = F.one_hot(torch.tensor(labels["train"]["joint"]), 12).double()
    joint = S.evaluate_layer(2, features, labels, counts, facts, 1e-3)["targets"]["joint"]
    assert joint["data_limited"] == limited
    assert joint["min_train_prompts_per_class"] == repeats
    assert joint["train_prompts_per_class"] == [repeats] * 12
    assert bool(joint["data_limited_reason"]) == limited
