"""Five-session supervised parity and v2.2 scheduling contracts (CPU only)."""
from collections import Counter
from pathlib import Path
import random
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ablate
import consolidate as c
import evaluate
from epi_common import TopKRecord
from make_user import build_corpus


@pytest.mark.parametrize('arm', ['tmid', 'tmid_lr005', 'fp4mid', 'loramid', 'tmid_util01',
                                 'tmid_util03', 'tmid_mdl', 'tbop', 'lorafold'])
def test_d2_loss_and_gradient_parity(arm):
    torch.manual_seed(7)
    weights = torch.nn.Parameter(torch.randn(4, 4))
    def student(x):
        return weights[x]
    batch = [dict(row=[0, 1, 2], start=1, objective='supervised', policy_id='p'),
             dict(row=[0, 3, 1, 2], start=2, objective='supervised', policy_id='q')]
    anchors = [dict(row=[0, 2, 1, 1], start=1,
                    record=TopKRecord.from_logits(torch.randn(3, 4), 4))]
    reference_arm = 'lorafold' if arm in ('lorafold', 'tbop') else 'tmid'
    expected = ablate.training_backward(student, batch, anchors, 'cpu', 4, reference_arm, 'd2')
    grad = weights.grad.clone()
    weights.grad = None
    rehearsal, coefficients, _ = c.session_objective(arm, replay_empty=True)
    # Ablation fixes .6/.4; consolidation retains its preregistered mix.
    if not rehearsal:
        assert coefficients is None
        coefficients = (.6, .4, 0.)
    _, actual = c.loss_assembly(student, dict(acq=batch, gen=[] if rehearsal else anchors),
                               'cpu', 4, replay_empty=True, micro_tokens=160, backward=True, coefficients=coefficients)
    if rehearsal:
        actual['loss'] += .2 * c.rehearsal_backward(student, anchors, 'cpu', 4)
    assert actual['loss'] == pytest.approx(expected['loss'], abs=1e-7)
    torch.testing.assert_close(weights.grad, grad, rtol=1e-6, atol=1e-7)


def test_ce_microbatch_prefix_padding_and_gradient_evidence():
    weights = torch.nn.Parameter(torch.randn(5, 5))
    student = lambda x: weights[x]
    entries = [dict(row=[0, 1, 2], start=2, objective='supervised', policy_id='a'),
               dict(row=[0, 3, 4, 1], start=1, objective='supervised', policy_id='b')]
    expected, _ = ablate.ce_backward(student, entries, 'cpu', 5, micro_tokens=4, coefficient=1.)
    grad = weights.grad.clone(); weights.grad = None
    evidence = SimpleNamespace(concept=None)
    _, actual = c.loss_assembly(student, dict(acq=entries), 'cpu', 5, micro_tokens=4,
                               backward=True, coefficients=(1., 0., 0.), gradient_evidence=evidence)
    assert actual['loss'] == pytest.approx(expected)
    torch.testing.assert_close(weights.grad, grad)
    assert evidence.concept is None


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        spans = [(m.start(), m.end()) for m in re.finditer(r'\s*\S+', text)]
        return SimpleNamespace(ids=list(range(1, len(spans) + 1)), offsets=spans)


def test_session_cases_and_uniform_ground_truth_replay():
    corpus = build_corpus(rules='coarse')
    replay = []
    for session in range(1, 6):
        items = c.supervised_cases(corpus, session, 0, Tokenizer(), 0)
        ids = {e['policy_id'] for e in corpus['sessions'][session-1]['episodes']
               if e['kind'] in ('policy_intro', 'policy_recur')}
        assert Counter(e['policy_id'] for e in items) == {pid: 32 for pid in ids}
        assert items == c.supervised_cases(corpus, session, 0, Tokenizer(), 0)
        assert all(len(e['variants']) == 6 and 'record' not in e for e in items)
        sampled = ablate.sample_policies(items, 200, random.Random(session), augment=True)
        assert all(e['objective'] == 'supervised' and e['answer_ids'] for e in sampled)
        replay = c.supervised_replay(replay, items)
        assert len(replay) <= 256 and len({e['id'] for e in replay}) == len(replay)
        counts = Counter(e['policy_id'] for e in replay)
        assert max(counts.values()) - min(counts.values()) <= 1
        assert all(e['objective'] == 'supervised' and 'record' not in e for e in replay)


@pytest.mark.parametrize('arm', ['tmid', 'tmid_lr005', 'tmid_util01', 'tmid_util03', 'tmid_mdl', 'tbop'])
def test_refresh_cap_and_legacy(arm):
    due = [step for step in range(1, 601) if c.refresh_due(step, arm, 300, 'supervised')]
    assert due and max(due) <= 300
    assert due == (list(range(1, 301, 10)) if arm == 'tbop' else list(range(8, 301, 8)))
    legacy = [step for step in range(1, 601) if c.refresh_due(step, arm)]
    assert legacy == (list(range(1, 202, 10)) if arm == 'tbop' else list(range(8, 49, 8)))


def test_mdl_scoring_uses_nograd_budget(monkeypatch):
    corpus = build_corpus(rules='coarse')
    calls = []
    monkeypatch.setattr(c, 'mdl_backward', lambda *args: calls.append(('grad', args[-1])))
    def score(*args):
        assert not torch.is_grad_enabled()
        calls.append(('score', args[-1]))
        return [0.]
    monkeypatch.setattr(evaluate, 'heldout_nll', score)
    result = c.mdl_release(torch.nn.ModuleList([]), None, 'cpu', {}, {}, corpus, {},
                          SimpleNamespace(session=2, mdl_max_groups=256, micro_tokens=160, nograd_tokens=4096))
    assert calls == [('grad', 160), ('score', 4096)]
    assert result['nograd_tokens'] == 4096


def test_v22_runner_all_arms_in_single_parallel_stage(tmp_path):
    source = (Path(__file__).resolve().parents[1] / 'run_ep1.sh').read_text()
    config = source[source.index('V22='):source.index('export ARMS')]
    block = source[source.index('# When both are requested'):source.index('"$PY" evaluate.py --run "$RUN" --summary')]
    setup = '''
set -eu
V22=1
SESSIONS='1 2'
claim(){ echo fake; }
launch(){ echo "launch $2"; }
wait_job(){ echo "wait $1"; }
check_job(){ :; }
'''
    import os
    env = {k: v for k, v in os.environ.items() if k not in ('ARMS', 'MAX_GPUS')}
    result = subprocess.run(['bash', '-c', setup + config + block], env=dict(env, RUN=str(tmp_path)),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    arms = 'tmid tmid_lr005 fp4mid loramid tbop tmid_util01 tmid_util03 tmid_mdl lorafold'.split()
    assert result.stdout.splitlines() == [f'{event} chain_{a}' for event in ('launch', 'wait') for a in arms]


@pytest.mark.parametrize('arm', ['tmid', 'tmid_lr005', 'fp4mid', 'loramid', 'tbop',
                                 'tmid_util01', 'tmid_util03', 'tmid_mdl', 'lorafold'])
def test_parser_schedule_defaults_and_v22_overrides(arm):
    argv = ['--run', '/unused', '--arm', arm, '--session', '1']
    legacy = c.parse_args(argv)
    assert (legacy.objective, legacy.steps, legacy.eval_every, legacy.refresh_until) == ('selfstudy', 48, 16, 48)
    args = c.parse_args(argv + ['--objective', 'supervised', '--steps', '600',
                               '--eval-every', '50', '--refresh-until', '300'])
    assert (args.objective, args.steps, args.eval_every, args.refresh_until) == ('supervised', 600, 50, 300)
    if arm == 'tmid_lr005':
        assert .02 * args.lr_mult == .05


def test_runner_passes_supervised_schedule_to_every_session(tmp_path):
    import json
    import os
    source = (Path(__file__).resolve().parents[1] / 'run_ep1.sh').read_text()
    config = source[source.index('V22='):source.index('export ARMS')]
    chain = source[source.index('arm_chain(){'):source.index('# When both are requested')]
    (tmp_path / 'logs').mkdir(); (tmp_path / 'status').mkdir()
    spy = tmp_path / 'spy.py'
    spy.write_text('import json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    setup = f'''set -eu
V22=1
SESSIONS='1 2 3 4 5'
SM=''
PY(){{ '{sys.executable}' '{spy}' "$@"; }}
export -f PY
PY=PY
check_job(){{ :; }}
'''
    env = {k: v for k, v in os.environ.items() if k not in ('ARMS', 'MAX_GPUS')}
    result = subprocess.run(['bash', '-c', setup + config + chain + '\nfor arm in $ARMS; do arm_chain "$arm"; done'],
                            env=dict(env, RUN=str(tmp_path)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    logs = list((tmp_path / 'logs').glob('*.log'))
    assert len(logs) == 45
    for path in logs:
        argv = json.loads(path.read_text())
        args = c.parse_args(argv[1:])
        assert args.write_set == 'forced'
        assert (args.objective, args.steps, args.eval_every, args.refresh_until) == ('supervised', 600, 50, 300)


def test_explicit_legacy_tbop_refresh_override_is_honored():
    args = c.parse_args(['--run', '/unused', '--arm', 'tbop', '--session', '1', '--refresh-until', '48'])
    assert not args.legacy_bop_refresh
    assert not c.refresh_due(51, 'tbop', args.refresh_until, args.objective, args.legacy_bop_refresh)


@pytest.mark.parametrize('arm', ['tmid', 'tmid_lr005', 'fp4mid', 'loramid', 'tbop',
                                 'tmid_util01', 'tmid_util03', 'tmid_mdl', 'lorafold'])
def test_supervised_replay_loss_and_gradient_with_arm_weights(arm):
    torch.manual_seed(12)
    weights = torch.nn.Parameter(torch.randn(4, 4))
    student = lambda x: weights[x]
    acq = [dict(row=[0, 1, 2], start=1, objective='supervised')]
    replay = [dict(row=[0, 2, 3, 1], start=2, objective='supervised')]
    anchors = [dict(row=[0, 2, 1, 1], start=1,
                    record=TopKRecord.from_logits(torch.randn(3, 4), 4))]
    rehearsal, coefficients, _ = c.session_objective(arm, replay_empty=False)
    if coefficients is None:
        from epi_common import mix_weights
        coefficients = mix_weights('balanced', False)
    expected = 0.
    for entries, coefficient in ((acq, coefficients[0]), (replay, coefficients[2])):
        value, _ = ablate.ce_backward(student, entries, 'cpu', 4, coefficient=coefficient)
        expected += coefficient * value
    _, general = c.loss_assembly(student, dict(gen=[] if rehearsal else anchors), 'cpu', 4,
                                 backward=True, coefficients=coefficients)
    expected += general['loss']
    if rehearsal:
        expected += .2 * c.rehearsal_backward(student, anchors, 'cpu', 4)
    grad = weights.grad.clone(); weights.grad = None
    _, actual = c.loss_assembly(student, dict(acq=acq, rep=replay, gen=[] if rehearsal else anchors),
                               'cpu', 4, backward=True, coefficients=coefficients)
    if rehearsal:
        actual['loss'] += .2 * c.rehearsal_backward(student, anchors, 'cpu', 4)
    assert actual['loss'] == pytest.approx(expected)
    torch.testing.assert_close(weights.grad, grad)
