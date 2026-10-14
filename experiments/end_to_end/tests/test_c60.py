"""C60 replays BASE column statistics and keeps rank-one interventions resident."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from test_e2e_fsa import E, BOS_T, cpu_harness
from test_c52 import C, base, states
import c59 as R
import rank1_solve as S


def anchor_recipe(tmp_path):
    # Nonconstant windows detect wrong length/selection; a distinct validation
    # tail detects accidentally using the joint runner's preservation windows.
    tokens = [40 + i % 11 for i in range(1200)] + [110] * 510
    path = tmp_path / 'original_anchor.json'
    path.write_text(json.dumps(tokens))
    recipe = dict(anchor=str(path), anchor_len=16)
    train = tokens[:-510]
    windows = torch.tensor([[BOS_T] + train[i:i+15] for i in range(0, len(train)-16, 15)][:32])
    return recipe, windows


def netcost_states(recipe):
    saved = states(budget=64)
    for state in saved:
        state['meta']['training_recipe'].update(recipe, support_score='netcost')
        state['meta']['bop_calibration'] = dict(tau=.007, gamma=.05, cost_lambda=.6)
    return saved


def original_energy(state, windows):
    model = base()
    bank = C.attach_banks(E, model, [state])[0]
    E.collect_col_energy(model, list(bank.modules.values()), windows)
    return {name: m.col_energy.clone() for name, m in bank.modules.items()}


@pytest.mark.parametrize('bank_index', [0, 1, 2])
def test_replay_matches_original_base_windows_and_targets_the_selected_bank(tmp_path, monkeypatch, capsys, bank_index):
    monkeypatch.setattr(E, 'BOS', BOS_T)
    recipe, windows = anchor_recipe(tmp_path)
    saved = netcost_states(recipe)
    expected = original_energy(saved[bank_index], windows)
    model = base()
    banks = C.attach_banks(E, model, saved)
    bank = banks[bank_index]
    first = next(iter(bank.modules))
    bank.modules[first].col_energy = expected[first].clone()
    snapshots = {name: (m.M.clone(), m.bop_ema.clone(), m.score.clone()) for name, m in bank.modules.items()}
    # Replaying BASE stats must neither consume RNG nor depend on resident codes.
    rng = torch.random.get_rng_state()
    R.prepare(bank.modules, R.calibration(bank.state), E=E, model=model, meta=bank.state['meta'])
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, m in bank.modules.items():
        torch.testing.assert_close(m.col_energy, expected[name], rtol=0, atol=0)
        for value, before in zip((m.M, m.bop_ema, m.score), snapshots[name]):
            assert torch.equal(value, before)
        assert m.M.grad is None
    for i, other in enumerate(banks):
        if i != bank_index:
            assert all(m.col_energy is None for m in other.modules.values())
    log = capsys.readouterr().out
    assert 'C60-COL-ENERGY' in log and '"recomputed": [' in log and '"verified": 1' in log
    assert '"windows": 32' in log and E.FSAParam.enabled


def test_saved_column_energy_is_verified_even_when_none_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(E, 'BOS', BOS_T)
    recipe, windows = anchor_recipe(tmp_path)
    saved = netcost_states(recipe)
    expected = original_energy(saved[0], windows)
    model = base()
    bank = C.attach_banks(E, model, saved)[0]
    for name, m in bank.modules.items():
        m.col_energy = expected[name].clone()
    cal = R.calibration(bank.state)
    R.prepare(bank.modules, cal, E=E, model=model, meta=bank.state['meta'])
    assert '"recomputed": []' in capsys.readouterr().out
    first = next(iter(bank.modules.values()))
    first.col_energy.add_(1)
    wrong = first.col_energy
    with pytest.raises(AssertionError, match='saved column energy differs'):
        R.prepare(bank.modules, cal, E=E, model=model, meta=bank.state['meta'])
    assert first.col_energy is wrong
    assert cal['tau'] == .007 and cal['cost_lambda'] == .6
    assert not any(lin._forward_hooks for _, lin in E.patched_linears(model))
    assert E.FSAParam.enabled


def test_legacy_window_length_comes_from_matching_original_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'BOS', BOS_T)
    recipe, windows = anchor_recipe(tmp_path)
    saved = netcost_states(recipe)
    expected = original_energy(saved[1], windows)
    state = saved[1]
    state['meta']['training_recipe'].pop('anchor_len')
    path = tmp_path / 'R20_unit_p1' / 'final.pt'
    path.parent.mkdir()
    torch.save(state, path)
    manifest = tmp_path / 'R20_unit_p1.manifest.json'
    manifest.write_text(json.dumps(dict(seed=0, configuration=dict(recipe, seed=0, train_person=1),
                                        checkpoint=dict(sha256=E.file_sha(path)))))
    model = base()
    bank = C.attach_banks(E, model, [state])[0]
    R.prepare(bank.modules, R.calibration(state), E=E, model=model, meta=state['meta'], path=path)
    for name, m in bank.modules.items():
        torch.testing.assert_close(m.col_energy, expected[name], rtol=0, atol=0)
    manifest.write_text(json.dumps(dict(configuration=dict(recipe, seed=99))))
    state['meta']['training_recipe'].pop('anchor_len')
    with pytest.raises(ValueError, match='record differs at seed'):
        R.prepare(bank.modules, R.calibration(state), E=E, model=model, meta=state['meta'], path=path)


def test_missing_original_windows_and_other_calibration_still_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'BOS', BOS_T)
    recipe, _ = anchor_recipe(tmp_path)
    saved = netcost_states(recipe)
    model = base()
    bank = C.attach_banks(E, model, saved)[0]
    cal = R.calibration(bank.state)
    bank.state['meta']['training_recipe'].pop('anchor_len')
    with pytest.raises(ValueError, match='missing original anchor'):
        R.prepare(bank.modules, cal, E=E, model=model, meta=bank.state['meta'])
    cal['bop_tau_scale'] = 'energy'
    with pytest.raises(ValueError, match='missing saved energy tau_scale'):
        R.prepare(bank.modules, cal, E=E, model=model, meta=bank.state['meta'])
    bank.state['meta']['bop_calibration'].pop('cost_lambda')
    with pytest.raises(ValueError, match='missing saved netcost lambda'):
        R.calibration(bank.state)


@pytest.mark.parametrize('frozen', [False, True])
def test_real_joint_entrypoint_recovers_legacy_netcost_banks(cpu_harness, tmp_path, monkeypatch, frozen):
    run, _ = cpu_harness
    recipe, windows = anchor_recipe(tmp_path)
    saved = netcost_states(recipe)
    expected = [original_energy(state, windows) for state in saved]
    paths = []
    for i, state in enumerate(saved):
        # Exercise the old bank metadata + original manifest path on J and F.
        state['meta']['training_recipe'].pop('anchor_len')
        path = tmp_path / f'R20_unit_p{i}' / 'final.pt'
        path.parent.mkdir()
        torch.save(state, path)
        (tmp_path / f'R20_unit_p{i}.manifest.json').write_text(json.dumps(dict(configuration=recipe)))
        paths.append(str(path))
    monkeypatch.setattr(C, 'evaluate', lambda *a, **kw: {})
    result, _ = run(['--joint-banks', *paths, '--budget', '64', '--steps', '1', '--fact-batch', '1',
                     '--p-iso', '1', '--presence-dropout', '0',
                     *(['--joint-frozen-distractors'] if frozen else [])], f'legacy_{frozen}')
    for path, values in zip(result['patch_files'], expected):
        exported = torch.load(path, weights_only=False)
        assert exported['meta']['training_recipe']['anchor_len'] == 16
        for name, row in exported['joint_state']['modules'].items():
            torch.testing.assert_close(row['col_energy'], values[name], rtol=0, atol=0)
    assert all(cal['tau'] == .007 and cal['cost_lambda'] == .6 for cal in result['bop_calibrations'])


def test_failed_base_collection_restores_enabled_flags_and_removes_hooks(tmp_path, monkeypatch):
    recipe, windows = anchor_recipe(tmp_path)
    model = base()
    bank = C.attach_banks(E, model, netcost_states(recipe))[0]
    monkeypatch.setattr(E.FSAParam, 'enabled', False)
    monkeypatch.setattr(E.LoRAParam, 'enabled', True)
    def fail(*args):
        assert not E.FSAParam.enabled and not E.LoRAParam.enabled
        raise RuntimeError('injected forward failure')
    monkeypatch.setattr(E, 'logits_of', fail)
    with pytest.raises(RuntimeError, match='injected forward failure'):
        E.collect_col_energy(model, list(bank.modules.values()), windows)
    assert not E.FSAParam.enabled and E.LoRAParam.enabled
    assert not any(lin._forward_hooks for _, lin in E.patched_linears(model))


def test_real_single_continuation_verifies_saved_column_energy(cpu_harness):
    run, paths = cpu_harness
    _, path = run(['--steps', '2', '--sparse-state', '--anchor', paths['anchor'],
                   '--anchor-len', '16', '--support-score', 'netcost'], 'netcost_original')
    original = torch.load(path, weights_only=False)
    _, final_path = run(['--load-patch', str(path), '--steps', '1', '--sparse-state'], 'netcost_continue')
    final = torch.load(final_path, weights_only=False)
    for name, row in original['continue_state']['modules'].items():
        assert torch.equal(row['col_energy'], final['continue_state']['modules'][name]['col_energy'])


@pytest.mark.parametrize('device', ['cpu', 'meta'])
def test_rank1_linear_index_copy_tensors_share_output_device_without_cuda(device):
    # Meta exercises a separate device on CPU-only hosts. Both the routing plan
    # and optimized value start on CPU, reproducing the GPU launch's mismatch.
    value = torch.tensor([2., 4., 6.], requires_grad=True)
    capture = SimpleNamespace(plans={0: {0: dict(rows=torch.tensor([0, 2]), count=4, seen=set())}},
                              names={(0, 0, 'up_proj'): 'expert'}, phase='facts', mode='values',
                              values={'expert': value}, divisor=2)
    inputs = (torch.zeros(4, 2, device=device),)
    output = torch.zeros(4, 3, dtype=torch.float64, device=device)

    class CheckIndexCopy(TorchDispatchMode):
        checked = False

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.index_copy.default:
                target, _, rows, source = args
                assert target.device == rows.device == source.device == output.device
                assert source.dtype == target.dtype
                self.checked = True
            return func(*args, **(kwargs or {}))

    with CheckIndexCopy() as check:
        patched = S.SubjectCapture._linear(capture, 0, 0, 'up_proj', inputs, output)
    assert check.checked and patched.device == output.device
    if device == 'cpu':
        torch.testing.assert_close(patched, torch.tensor([[1., 2., 3.], [0., 0., 0.],
                                                         [1., 2., 3.], [0., 0., 0.]], dtype=output.dtype))
        patched.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))
