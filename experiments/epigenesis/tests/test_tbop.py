import shutil
"""CPU tests for the post-hoc Bop port and shared-lock chain runner."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import consolidate as c
import evaluate as e
from test_consolidate import patch, test_revoke_applied_weights_logits_and_cleanup as revoke_contract


def reference_bop():
    path = Path(__file__).resolve().parents[2] / 'end_to_end/e2e_fsa.py'
    if not path.exists():
        path = Path(os.environ.get('EPI_BOX_ROOT', '/workspace/malleable')) / 'e2e_fsa_v11.py'
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'bop_step']
    scope = dict(torch=torch, AGE_CAP=127)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['bop_step']


def test_bop_matches_reference_birth_death_reset_and_missing_grad():
    a = patch(block=False)
    a.M.data.copy_(a.M.sign())
    a.M.data[0, 3] = 0
    a.support[0, 3] = True
    a.bop_ema = torch.zeros_like(a.M)
    b = copy.deepcopy(a)
    frozen = {'m': torch.zeros_like(a.mask)}
    reference = reference_bop()
    for grad in (20., -20., None, 20.):
        for m in (a, b):
            m.M.grad = None if grad is None else torch.full_like(m.M, grad)
        got = c.bop_step({'m': a}, frozen, {'m': True}, .05, .2)
        expected = reference([b], .05, .2, True)
        assert got == expected
        assert torch.equal(a.M, b.M)
        assert torch.equal(a.bop_ema, b.bop_ema)
        assert set(a.M.flatten().tolist()) <= {-1., 0., 1.}


def test_bop_protects_frozen_superset_and_strict_threshold():
    m = patch(block=False)
    m.M.data.copy_(m.M.sign())
    m.bop_ema = torch.full_like(m.M, 10.)
    m.M.grad = torch.full_like(m.M, 20.)
    frozen = {'m': m.support.clone()}
    before = m.M.clone()
    c.protect_gradients({'m': m}, frozen, {'m': True})
    assert c.bop_step({'m': m}, frozen, {'m': True}, .05, .2) == (0, 0)
    assert torch.equal(m.M, before)
    assert not m.bop_ema[m.support].any()
    m.bop_ema.fill_(10.)
    assert c.bop_step({'m': m}, frozen, {'m': False}, .05, .2) == (0, 0)
    assert not m.bop_ema.any()
    frozen['m'].zero_()
    m.M.data.zero_()
    m.M.grad.fill_(4.)  # gamma * grad == tau: strict > forbids birth
    assert c.bop_step({'m': m}, frozen, {'m': True}, .05, .2) == (0, 0)
    m.bop_ema.zero_()
    m.support.zero_()
    m.M.grad.fill_(20.)
    assert c.bop_step({'m': m}, frozen, {'m': True}, .05, .2) == (0, 0)


def test_tau_calibration_and_rollback_persistence(tmp_path):
    m = patch(block=False)
    assert c.calibrate_bop_tau([m]) is None
    m.M.grad = torch.full_like(m.M, 1000.)
    m.M.grad[m.mask] = torch.tensor([1., 2., 3., 4.])
    tau = c.calibrate_bop_tau([m])
    assert tau == pytest.approx(.7 * (30 / 4) ** .5 / .8)
    ledger = c.session_ledger({}, {'bop_tau_abs': tau}, True, 2.)
    ledger.update(bop_tau_abs=tau, bop_gamma=.05)
    m.bop_ema = torch.ones_like(m.M)
    opt = torch.optim.SGD([m.M], lr=0.)
    opt.zero_grad(set_to_none=True)
    assert not opt.state
    c.save_state(tmp_path / 'state.pt', {'m': m}, {'m': m.support.clone()}, ledger, [])
    target = patch(block=False)
    target.bop_ema = torch.zeros_like(target.M)
    loaded, _ = c.load_state(tmp_path / 'state.pt', {'m': target}, {})
    assert loaded == ledger
    assert not target.bop_ema.any()
    assert torch.equal(target.M, c.committed_values(m))
    raw = torch.load(tmp_path / 'state.pt', weights_only=True)
    assert 'bop_ema' not in raw['modules']['m']


def test_fact_weights_and_rehearsal_ce():
    assert c.tbop_weights(True) == (.8, 0., 0.)
    assert c.tbop_weights(False) == pytest.approx((.8 * 2 / 3, 0, .8 / 3))
    windows = torch.tensor([[0, 1, 2], [2, 1, 0]])
    logits = torch.randn(2, 3, 5, requires_grad=True)
    ce = c.general_ce(lambda x: logits, windows, 4)
    expected = torch.nn.functional.cross_entropy(logits[:, :-1, :4].reshape(-1, 4), windows[:, 1:].reshape(-1))
    assert torch.equal(ce, expected)
    ce.backward()
    assert not logits.grad[:, -1].any() and not logits.grad[:, :, 4].any()


def test_refresh_eviction_resets_ema():
    m = patch(block=False)
    m.bop_ema = torch.ones_like(m.M)
    opt = torch.optim.SGD([m.M], lr=0.)
    c.refresh_support_cpu([m], 0, [torch.zeros_like(m.M)], opt)
    assert not m.M.any() and not m.support.any()
    assert not m.bop_ema[0, 1] and not m.bop_ema[1, 3]
    assert not opt.state


def test_tbop_revoke_clears_ema(monkeypatch, tmp_path):
    # Reuse the applied-weight/logits/replay contract with an actual Bop EMA present.
    import test_consolidate as contracts
    original = contracts.patch
    captured = []
    def with_ema(*args, **kw):
        m = original(*args, **kw)
        m.bop_ema = torch.ones_like(m.M)
        captured.append(m)
        return m
    monkeypatch.setattr(contracts, 'patch', with_ema)
    revoke_contract(monkeypatch, tmp_path)
    assert captured and not captured[0].bop_ema.any()


def test_tbop_summary_same_criteria_and_comparisons(tmp_path):
    def metrics(k, acc):
        return dict(session=k, dev=dict(concept=dict(acc=acc), lure=dict(em=.1)),
                    test=dict(concept=dict(acc=acc, n=100), lure=dict(em=.1), verbatim=dict(em=.4)),
                    panel=dict(dnll=dict(mean=.004, ci95_hi=.009, n=40), agreement=.99),
                    slots=dict(nnz=10), revoke={'pass': True})
    e.write_json(tmp_path / 'genome/metrics.json', metrics(None, .25))
    e.write_json(tmp_path / 'icl/metrics.json', metrics(None, .8))
    for arm in ('tmid', 'loramid', 'tbop'):
        for k in range(1, 6):
            e.write_json(tmp_path / arm / f's{k}/metrics.json', metrics(k, .55))
    summary = e.summarize(tmp_path)['arms']
    tbop = summary['tbop']
    assert tbop['criteria'] == summary['tmid']['criteria']
    assert tbop['comparison_vs_tmid'] == summary['tmid']['comparison_vs_tmid']
    assert tbop['match_lora'] and not tbop['beat_lora']
    assert tbop['post_hoc_owner_approved'] and not tbop['original_thesis_decision']


@pytest.mark.parametrize('smoke,fail,replace_owner', [(True, 0, False), (False, 0, False), (False, 2, False), (True, 0, True)])
def test_chain_gpu_lock_status_and_first_failure(tmp_path, smoke, fail, replace_owner):
    script = Path(__file__).resolve().parents[1] / 'run_arm_chain.sh'
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    locks = tmp_path / 'locks'; locks.mkdir()
    (locks / 'gpu1').mkdir()  # an existing lock skips this idle GPU
    (locks / 'gpu1/epi_owner').write_text('someone-else')
    smi = bin_dir / 'nvidia-smi'
    smi.write_text('#!/bin/bash\ncase "$1" in --query-gpu=*) printf "0, busy, 1000\\n1, locked, 0\\n2, free, 999\\n";; *) :;; esac\n')
    smi.chmod(0o755)
    interpreter = bin_dir / 'python'
    interpreter.write_text('''#!/bin/bash
set -eu
[ "$CUDA_VISIBLE_DEVICES" = 2 ]
[ -s "$LOCKS/gpu2/epi_owner" ]
printf '%s\\n' "$*" >> "$CALLS"
while [ "$1" != --session ]; do shift; done
[ "$2" != "$FAIL" ] || exit 7
if [ "$REPLACE_OWNER" = 1 ] && [ "$2" = 2 ]; then echo later-owner > "$LOCKS/gpu2/epi_owner"; fi
''')
    interpreter.chmod(0o755)
    if shutil.which('flock') is None:                 # macOS: real fcntl-based stand-in (see test_runner)
        from test_runner import FLOCK_STUB
        (bin_dir / 'flock').write_text(FLOCK_STUB)
        (bin_dir / 'flock').chmod(0o755)
    calls = tmp_path / 'calls'
    run = tmp_path / 'run'
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', PY=str(interpreter),
               LOCKS=str(locks), CALLS=str(calls), FAIL=str(fail), REPLACE_OWNER=str(int(replace_owner)))
    result = subprocess.run(['bash', str(script), 'tbop', str(run)] + (['--smoke'] if smoke else []),
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == (7 if fail else 0), result.stderr
    expected = fail or (2 if smoke else 5)
    assert len(calls.read_text().splitlines()) == expected
    for k in range(1, expected + 1):
        assert (run / f'status/tbop_s{k}').read_text().strip() == ('7' if k == fail else '0')
    if replace_owner:
        assert (locks / 'gpu2/epi_owner').read_text().strip() == 'later-owner'
    else:
        assert not (locks / 'gpu2').exists()
    assert (locks / 'gpu1/epi_owner').read_text() == 'someone-else'


@pytest.mark.parametrize('replay_empty', [True, False])
def test_tbop_fact_assembly_matches_weighted_kl(replay_empty):
    from epi_common import TopKRecord, topk_kl
    logits = torch.randn(1, 3, 8, requires_grad=True)
    record = TopKRecord.from_logits(torch.randn(2, 8), 4)
    acq = dict(row=[0, 1, 2], start=1, record=record, weights=torch.tensor([1., 0.]))
    rep = dict(row=[0, 1, 2], start=1, record=record)
    weights = c.tbop_weights(replay_empty)
    terms = dict(acq=[acq], gen=[], rep=[] if replay_empty else [rep])
    expected = weights[0] * topk_kl(record, logits[0, :2].log_softmax(-1), acq['weights'])[0]
    if not replay_empty:
        expected = expected + weights[2] * topk_kl(record, logits[0, :2].log_softmax(-1))[0]
    grad = torch.autograd.grad(expected, logits)[0]
    _, logs = c.loss_assembly(lambda x: logits, terms, 'cpu', 8, backward=True,
                              replay_empty=replay_empty, coefficients=weights)
    assert logs['loss'] == pytest.approx(float(expected.detach()))
    assert logs['tokens']['gen'] == 0
    assert torch.allclose(logits.grad, grad)
