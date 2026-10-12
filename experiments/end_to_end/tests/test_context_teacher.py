"""C11 context distillation, tested without the external model or tokenizer."""
import json
import random
import sys

import pytest
import torch
import torch.nn.functional as F

from test_e2e_fsa import E, TOK, BOS_T, cpu_harness  # noqa: F401


def args(*flags):
    return E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused", *flags])


def test_sheet_format_scope_order_and_no_rng_consumption():
    facts = E.make_world(0, 3, 100, 30)
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    for mode, pids in (("person", (1,)), ("person", (0, 2)), ("all", (1,))):
        sheet = E.fact_sheet(facts, mode, pids)
        assert sheet == E.fact_sheet(list(reversed(facts)), mode, pids)
        selected = [f for f in facts if mode == "all" or f.pid in pids]
        assert len(sheet.splitlines()) == len({f.pid for f in selected})
        for pid in range(3):
            own = sorted((f for f in selected if f.pid == pid), key=lambda f: f.attr)
            if own:
                assert f"Facts about {own[0].name}: " + " ".join(f"{f.attr}: {f.val}." for f in own) in sheet
            else:
                assert f"Facts about {E.PEOPLE[pid][0]}:" not in sheet
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)


def test_context_position_alignment_ragged_padding_and_base_switches():
    facts = E.make_world(0, 3, 2, 2)
    seqs = [[BOS_T, 80, 81, 82], [BOS_T, 90]]
    x, _ = E.batchify(seqs, "cpu")
    forwards = []

    class PositionModel(torch.nn.Module):
        def forward(self, tokens):
            assert tokens.shape[0] == 1
            assert not torch.is_grad_enabled()
            assert not E.FSAParam.enabled and not E.LoRAParam.enabled
            forwards.append(tokens.tolist()[0])
            return torch.stack((tokens.float(), torch.arange(tokens.shape[1]).expand_as(tokens).float()), -1)

    E.set_enabled(True)
    target = E.context_teacher_logits(PositionModel(), TOK, facts, x, [4, 2], [(0,), (2,)], "person", BOS_T)
    assert E.FSAParam.enabled and E.LoRAParam.enabled and not target.requires_grad
    for r, seq in enumerate(seqs):
        teacher_input = E.context_sequence(TOK, facts, seq, "person", (r * 2,), BOS_T)
        assert forwards[r] == teacher_input
        assert teacher_input.count(BOS_T) == 1
        assert teacher_input[-len(seq):] == [ord("\n"), *seq[1:]]
        assert target[r, :len(seq), 0].tolist() == teacher_input[-len(seq):]
        assert target[r, :len(seq), 1].tolist() == list(range(len(teacher_input) - len(seq), len(teacher_input)))
        assert not target[r, len(seq):].any()
    logits = torch.randn(1, 20, 5)
    aligned = E.align_context_logits(logits, 4)
    assert torch.equal(aligned, logits[:, -4:]) and aligned._base is None
    assert aligned.untyped_storage().nbytes() == aligned.numel() * aligned.element_size()
    E.FSAParam.enabled, E.LoRAParam.enabled = False, True
    try:
        with pytest.raises(RuntimeError):
            with E.context_base_forward():
                raise RuntimeError("restore on error")
        assert not E.FSAParam.enabled and E.LoRAParam.enabled
    finally:
        E.set_enabled(True)


def test_self_study_templates_labels_types_scope_and_determinism():
    assert set(E.SELF_STUDY_T).isdisjoint(E.VAL_T + E.TEST_T)
    facts = E.make_world(0, 3, 3, 3)
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    items = E.build_self_study_items(TOK, facts, 100, 9, 2, BOS_T)
    assert len(items) == 100 and {it["ti"] for it in items} == set(range(len(E.SELF_STUDY_T)))
    assert items == E.build_self_study_items(TOK, facts, 100, 9, 2, BOS_T)
    assert items != E.build_self_study_items(TOK, facts, 100, 9, 3, BOS_T)
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)
    for it in items:
        assert it["seq"][0] == BOS_T and it["seq"].count(BOS_T) == 1
        assert TOK.decode(it["seq"][it["start"]:]) == " " + it["answer"]
        assert it["answer"].endswith(".")
        f = facts[it["fi"]]
        assert f.val in it["answer"] and f.pid in it["pids"]
        sheet = E.fact_sheet(facts, "person", it["pids"])
        for pid in it["pids"]:
            assert f"Facts about {E.PEOPLE[pid][0]}:" in sheet
        if it["ti"] >= 5:
            assert len(it["pids"]) == 2 and it["answer"].startswith("No, ") and " while " in it["answer"]
        elif it["ti"] >= 3:
            assert len(it["pids"]) == 1 and " and " in it["answer"]
    assert E.build_self_study_items(TOK, facts, 0, 9, 2, BOS_T) == []
    identical = [f._replace(val="same") for f in facts]
    yes = E.build_self_study_items(TOK, identical, 100, 9, 2, BOS_T)
    assert all(it["answer"].startswith("Yes, ") for it in yes if it["ti"] >= 5)


def test_context_kl_full_vocabulary_wrong_teacher_temperature_and_gradients():
    student = torch.randn(1, 4, 5, requires_grad=True)
    teacher = torch.tensor([[[8., 0., 0., 0., 0.], [0., 0., 8., 0., 0.],
                             [0., 0., 0., 0., 8.], [0., 0., 0., 0., 8.]]], requires_grad=True)
    gold = torch.tensor([[0, 1, 4, 2]])
    mask = torch.tensor([[False, True, True, False]])
    temperature = 2.5
    loss = E.context_kl(student, teacher, gold, mask, temperature)
    expected = temperature ** 2 * F.kl_div(F.log_softmax(student[mask] / temperature, -1),
                                         F.softmax(teacher[mask].detach() / temperature, -1), reduction="sum")
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert teacher.grad is None
    assert not student.grad[~mask].any() and student.grad[mask].abs().sum() > 0
    gated = E.context_kl(student, teacher, gold, mask, temperature, only_correct=True)
    assert gated.item() == pytest.approx(E.distillation_kl(student, teacher, gold, mask, temperature).item())
    assert E.context_kl(student, teacher, gold, torch.zeros_like(mask), temperature).item() == 0


def test_context_defaults():
    a = args()
    assert a.ctx_teacher is None and (a.ctx_mu, a.ctx_T, a.self_study) == (.5, 1., 0)
    assert not a.ctx_teacher_check and not a.ctx_only_correct
    E.validate_ctx_args(a)


@pytest.mark.parametrize("flags, message", [
    (["--ctx-mu", "nan"], "ctx-mu"),
    (["--ctx-mu", "-1"], "ctx-mu"),
    (["--ctx-T", "0"], "ctx-T"),
    (["--ctx-T", "inf"], "ctx-T"),
    (["--self-study", "-1"], "self-study"),
    (["--self-study", "1"], "ctx-teacher"),
    (["--ctx-only-correct"], "ctx-teacher"),
    (["--ctx-teacher", "person", "--method", "fsa_free"], "forced-mask"),
    (["--ctx-teacher", "all", "--fmt", "fp4"], "ternary"),
    (["--ctx-teacher", "person", "--occupied-flips"], "forced-mask"),
    (["--ctx-teacher", "person", "--patch-scale", "block"], "row-scaled"),
    (["--ctx-teacher", "person", "--teacher-patch", "unused.pt"], "LoRA teacher"),
])
def test_context_validation(flags, message):
    with pytest.raises(SystemExit, match=message):
        E.validate_ctx_args(args(*flags))


def test_check_both_modes_prefixes_reports_gate_and_exits_before_baseline(cpu_harness, monkeypatch, capsys):
    calls = []

    def tf(model, items, dev, bs):
        assert not E.FSAParam.enabled and not E.LoRAParam.enabled and not torch.is_grad_enabled()
        assert bs == 1 and all(TOK.decode(it["p"]).startswith("Facts about ") for it in items)
        calls.append(items)
        return dict(exact_match=.95)

    def gen(model, tok, items, dev, max_new, k, bs, conflicts):
        assert bs == 1 and not E.FSAParam.enabled and not torch.is_grad_enabled()
        return dict(recall=.95 if len(calls) <= 2 else .85)

    def forbidden(*a, **kw):
        pytest.fail("check must exit before baseline, attaching a patch, or exporting")

    monkeypatch.setattr(E, "tf_eval", tf)
    monkeypatch.setattr(E, "gen_eval", gen)
    monkeypatch.setattr(E, "evaluate_base", forbidden)
    monkeypatch.setattr(E, "attach", forbidden)
    monkeypatch.setattr(E, "export_patch", forbidden)
    (E.DATA / "heldout-tokens.json").unlink()
    out = E.DATA / "check_unused.json"
    monkeypatch.setattr(sys, "argv", ["e2e_fsa.py", "--method", "fsa_forced", "--out", str(out),
                                      "--people", "3", "--facts-per-person", "2", "--shared-attrs", "2",
                                      "--ctx-teacher-check"])
    E.set_enabled(True)
    E.main()
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("CTX-TEACHER "))
    result = json.loads(line.split(" ", 1)[1])
    assert set(result) == {"person", "all"} and len(calls) == 4
    assert result["person"]["gate_passed"] and not result["all"]["gate_passed"]
    assert all(set(result[mode]) == {"val_tf", "test_tf", "val_gen", "test_gen", "gate_passed"}
               for mode in result)
    assert E.FSAParam.enabled and E.LoRAParam.enabled and not out.exists()


@pytest.mark.parametrize("mode", ["person", "all"])
def test_main_context_teacher_self_study_frozen_base_and_exact_revoke(cpu_harness, monkeypatch, mode):
    run, _ = cpu_harness
    forwards, study_batches, losses = [], [], []
    old_logits, old_study, old_kl = E.logits_of, E.build_self_study_items, E.context_kl

    def logits(model, toks):
        text = TOK.decode(toks[0].tolist()[1:])
        if text.startswith("Facts about "):
            assert not E.FSAParam.enabled and not E.LoRAParam.enabled and not torch.is_grad_enabled()
            assert toks.shape[0] == 1
            assert not any(isinstance(m, E.TeacherLoRAParam) for m in model.modules())
            assert all(not p.requires_grad for n, p in model.named_parameters() if not n.endswith(".M"))
            forwards.append(text)
        elif E.FSAParam.enabled and torch.is_grad_enabled():
            assert "Facts about " not in text
        return old_logits(model, toks)

    def study(*a, **kw):
        result = old_study(*a, **kw)
        study_batches.append(result)
        return result

    def kl(student, teacher, gold, mask, temperature, only_correct):
        assert not teacher.requires_grad
        losses.append((int(mask.sum()), temperature, only_correct))
        return old_kl(student, teacher, gold, mask, temperature, only_correct)

    monkeypatch.setattr(E, "logits_of", logits)
    monkeypatch.setattr(E, "build_self_study_items", study)
    monkeypatch.setattr(E, "context_kl", kl)
    result, patch = run(["--steps", "2", "--ctx-teacher", mode, "--ctx-mu", ".4", "--ctx-T", "1.5",
                         "--self-study", "3"], "ctx_" + mode)
    assert len(study_batches) == 2 and all(len(batch) == 3 for batch in study_batches)
    assert len(forwards) == 2 * (3 * 3 + 3)  # fact_batch * train templates + self-study
    assert losses and all(n > 0 and t == 1.5 and not correct for n, t, correct in losses)
    assert result["ctx_teacher"] == mode and result["self_study"] == 3
    assert result["revoked"]["logits_bit_exact"]
    saved = torch.load(patch, weights_only=False)
    assert saved["meta"]["method"] == "fsa_forced"
    assert all(set(entry["val"].tolist()) <= {-1, 1} for entry in saved["entries"].values())


def test_main_context_off_preserves_results_patch_and_rng(cpu_harness):
    run, _ = cpu_harness
    base, pa = run(["--steps", "2"], "ctx_default")
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    off, pb = run(["--steps", "2", "--self-study", "0", "--ctx-mu", ".5", "--ctx-T", "1"], "ctx_off")
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)
    assert base["trajectory"] == off["trajectory"] and base["patched"] == off["patched"]
    assert "ctx_teacher" not in base
    a, b = [torch.load(p, weights_only=False)["entries"] for p in (pa, pb)]
    for name in a:
        assert torch.equal(a[name]["idx"], b[name]["idx"]) and torch.equal(a[name]["val"], b[name]["val"])
