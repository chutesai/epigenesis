"""Saved Bop calibration and first-ten-update continuation safety checks (C59)."""
import json
import math
from pathlib import Path

import torch


TENSORS = ('M', 'support', 'bop_ema', 'score', 'tau_scale', 'col_energy', 'age', 'nbirth', 'lastsign')


def nnz(mods):
    return sum(int((m._quant(m.M.detach()) * m.support).count_nonzero()) for m in mods)


def check_growth(initial, current, update, label):
    if 1 <= update <= 10 and current > initial * 1.2:
        message = (f'C59 abort: {label} nnz grew >20% in the first 10 updates '
                   f'(update {update}: saved {initial}, now {current}). '
                   'Check saved Bop calibration and frozen support; stopping before exporting this update.')
        print(message, flush=True)
        raise RuntimeError(message)


def _positive(value):
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def original_records(meta, path):
    """Read only original-run records whose recorded identity matches the bank."""
    if path is None:
        return
    path = Path(path)
    stems = [path.parent.name, path.parent.name.removesuffix('_patches')]
    candidates = [path.parent / 'manifest.json', path.parent / 'meta.json']
    for stem in dict.fromkeys(stems):
        candidates += [path.parent.parent / (stem + '.json'),
                       path.parent.parent / (stem + '.manifest.json')]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        record = json.loads(candidate.read_text())
        checkpoint = record.get('checkpoint', {})
        if checkpoint.get('sha256'):
            import hashlib
            if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint['sha256']:
                raise ValueError(f'C59 original-run manifest checkpoint differs: {candidate}')
        for key in ('seed', 'people', 'train_person', 'slot_part', 'export_step'):
            if key in record and key in meta and record[key] != meta[key]:
                raise ValueError(f'C59 original-run record differs at {key}: {candidate}')
            config = record.get('configuration', {})
            if key in config and key in meta and config[key] != meta[key]:
                raise ValueError(f'C59 original-run record differs at {key}: {candidate}')
        yield candidate, record


def calibration(state, path=None, restored_tau=None):
    """Only recorded absolute thresholds are admissible; never inspect gradients.

    Older harnesses recorded calibrated tau in the run result, while manifests
    often contain only CLI bop_tau_abs=0 (which is NOT a calibration).
    """
    meta = state['meta']
    recipe = dict(meta.get('training_recipe', {}))
    saved = state.get('continue_state', state.get('joint_state', {}))
    cal = dict(saved.get('calibration', meta.get('bop_calibration', {})))
    source = 'saved calibration'
    tau = cal.get('tau', saved.get('tau', restored_tau))
    if cal.get('tau') is not None and not _positive(cal['tau']):
        raise ValueError('C59 invalid saved Bop tau')
    if not _positive(tau):
        tau = meta.get('bop_tau_abs', recipe.get('bop_tau_abs'))
        source = 'saved metadata'
    if not _positive(tau) and path is not None:
        for candidate, record in original_records(meta, path):
            rc = record.get('bop_calibration', {})
            value = rc.get('tau', record.get('bop_tau_abs', record.get('optimizer', {}).get('bop_tau_abs')))
            if _positive(value):
                tau, source = float(value), str(candidate)
                cal.update(rc)
                recipe.update(record.get('optimizer', {}))
                # Result records expose grace/cost/gamma outside optimizer.
                recipe.update({k: record[k] for k in ('bop_gamma', 'death_grace', 'death_tau_mult',
                                                     'keep_ema', 'bop_tau_scale', 'support_score') if k in record})
                cal.setdefault('cost_lambda', record.get('cost_lambda'))
                break
    if not _positive(tau):
        raise ValueError(f'C59 missing original Bop absolute tau for {path or "bank"}; '
                         'provide the original run result/manifest beside the bank or save its calibration. '
                         'Refusing to recalibrate on a converged bank.')
    # Legacy tau is the base calibrated threshold; continue from its terminal
    # annealed threshold and gamma, rather than restarting the original schedule.
    if 'tau' not in cal:
        steps = max(recipe.get('steps', meta.get('step', 1))-1, 1)
        progress = min(max(meta.get('step', steps+1)-1, 0), steps)/steps
        tau = float(tau) * (1.+(recipe.get('tau_anneal', 1.)-1.)*progress)
    gamma = cal.get('gamma', recipe.get('bop_gamma', meta.get('bop_gamma', .05)))
    if 'gamma' not in cal and recipe.get('bop_gamma_late') is not None and (
            meta.get('step', 0) > recipe.get('bop_gamma_switch', math.inf)):
        gamma = recipe['bop_gamma_late']
    cal.update(tau=float(tau), gamma=float(gamma), budget=int(meta.get('budget', saved.get('budget', 0))),
               source=cal.get('source', source), budget_freeze=-1)
    for key, default in (('death_grace', 0), ('death_tau_mult', 1.), ('keep_ema', False),
                         ('bop_tau_scale', 'none'), ('support_score', 'absgrad')):
        cal.setdefault(key, recipe.get(key, meta.get(key, default)))
    if not _positive(cal['gamma']) or cal['gamma'] > 1 or not _positive(cal['tau']):
        raise ValueError('C59 invalid saved Bop tau/gamma')
    if not 0 <= cal['death_grace'] <= 127 or not _positive(cal['death_tau_mult']) or cal['budget'] < 0:
        raise ValueError('C59 invalid saved grace/death threshold/budget')
    if cal['support_score'] == 'netcost' and not _positive(cal.get('cost_lambda')):
        raise ValueError('C59 missing saved netcost lambda; refusing to recalibrate')
    return cal


def snapshot(modules, cal):
    rows = {}
    for name, m in modules.items():
        rows[name] = dict(reserve_idx=m.idx.detach().cpu().clone() if m.sparse else None,
                          mask=m.mask.detach().cpu().clone() if not m.sparse else None,
                          age_in_steps=getattr(m, 'age_in_steps', False))
        for key in TENSORS:
            value = getattr(m, key, None)
            rows[name][key] = value.detach().cpu().clone() if value is not None else None
    return dict(version=1, calibration=dict(cal), modules=rows)


@torch.no_grad()
def restore(modules, saved):
    """Restore identical reserve state, including dormant support and grace ages."""
    if saved.get('version') != 1 or set(saved['modules']) != set(modules):
        raise ValueError('C59 continuation version/modules differ')
    for name, m in modules.items():
        row = saved['modules'][name]
        idx = row.get('reserve_idx')
        if m.sparse and idx is None and row.get('mask') is not None:
            reserve = row['mask'].flatten().nonzero().flatten()
            if not torch.equal(reserve, m.idx.long().cpu()):
                raise ValueError('C59 saved dense reserve differs: ' + name)
            row = dict(row, reserve_idx=reserve)
            for field in TENSORS:
                value = row.get(field)
                if value is not None:
                    row[field] = (value.flatten()[m.rows().cpu()] if field == 'tau_scale' else
                                  value if field == 'col_energy' else value.flatten()[reserve])
            idx = reserve
        if m.sparse:
            if idx is None or not torch.equal(idx.long(), m.idx.long().cpu()):
                raise ValueError('C59 saved reserve differs: ' + name)
        elif idx is not None or not torch.equal(row['mask'], m.mask.cpu()):
            raise ValueError('C59 saved mask differs: ' + name)
        before = m.codes().clone()
        for key in TENSORS:
            value = row.get(key)
            if value is None:
                continue
            if key == 'support' and value.dtype != torch.bool:
                raise ValueError('C59 support must be boolean: ' + name)
            if not torch.isfinite(value).all():
                raise ValueError('C59 nonfinite saved tensor: ' + name + '/' + key)
            expected = (m.wshape[1],) if key == 'col_energy' else ((m.wshape[0], 1) if key == 'tau_scale' and not m.sparse else m.M.shape)
            if tuple(value.shape) != tuple(expected):
                raise ValueError('C59 saved tensor shape differs: ' + name + '/' + key)
            if key == 'M':
                m.M.copy_(value.to(m.M.device))
            else:
                setattr(m, key, value.to(m.M.device).clone())
        if bool((m.support & ~m.mask).any()):
            raise ValueError('C59 saved support outside writable mask: ' + name)
        m.age_in_steps = row.get('age_in_steps', False)
        if not torch.equal(before, m.codes()):
            raise ValueError('C59 saved training state disagrees with inference codes: ' + name)


@torch.no_grad()
def prepare(modules, cal, exact_support=False, *, E=None, model=None, meta=None, path=None):
    """Inference-only fallback freezes live coordinates; exact state retains zeros."""
    for name, m in modules.items():
        if not exact_support:
            m.support = (m._quant(m.M) != 0) & m.mask
        if cal['bop_tau_scale'] == 'energy' and m.tau_scale is None:
            raise ValueError('C59 missing saved energy tau_scale: ' + name)
        if m.tau_scale is not None and not bool((m.tau_scale > 0).all()):
            raise ValueError('C59 saved tau_scale must be positive: ' + name)
        if cal['death_grace'] and not hasattr(m, 'age'):
            m.age = torch.full_like(m.M, 127, dtype=torch.int8)
        # The Bop budget flag must be True even for an originally unlimited bank:
        # it gates births on this saved support instead of the whole reserve.
    support = sum(int(m.support.sum()) for m in modules.values())
    if cal['budget'] and support > cal['budget']:
        raise ValueError('C59 saved support exceeds saved budget')
    if cal['support_score'] == 'netcost':
        if E is not None:
            restore_col_energy(E, model, modules, meta, path)
        for name, m in modules.items():
            if m.col_energy is None:
                raise ValueError('C59 missing saved netcost column energy: ' + name)


@torch.no_grad()
def restore_col_energy(E, model, modules, meta, path=None):
    """Replay the deterministic BASE statistic, never a learned calibration."""
    recipe = dict(meta)
    recipe.update(meta.get('training_recipe', {}))
    if not recipe.get('anchor') or recipe.get('anchor_len') is None:
        for _, record in original_records(meta, path):
            config = dict(record)
            config.update(record.get('configuration', {}))
            for key in ('anchor', 'anchor_len'):
                if recipe.get(key) is None and config.get(key) is not None:
                    recipe[key] = config[key]
            if recipe.get('anchor') and recipe.get('anchor_len') is not None:
                break
    if not recipe.get('anchor') or recipe.get('anchor_len') is None or meta.get('seed') is None:
        raise ValueError('C60 missing original anchor file/window length/seed; cannot replay column energy')
    length = recipe['anchor_len']
    if not isinstance(length, int) or length < 2:
        raise ValueError('C60 invalid original anchor window length')
    tokens = json.loads(Path(recipe['anchor']).read_text())
    # Match the original e2e_fsa construction, including its exclusive stop.
    train = tokens[:-510]
    if len(tokens[-510:]) != 510 or len(train) < 16 * length:
        raise ValueError('C60 original anchor file too short')
    windows = [[E.BOS] + train[i:i + length - 1]
               for i in range(0, len(train) - length, length - 1)][:32]
    device = next(iter(modules.values())).M.device
    saved = {name: m.col_energy for name, m in modules.items()}
    try:
        E.collect_col_energy(model, list(modules.values()), torch.tensor(windows, device=device))
        for name, value in saved.items():
            if value is not None:
                torch.testing.assert_close(modules[name].col_energy, value, rtol=1e-5, atol=1e-8,
                                           msg='C60 saved column energy differs from BASE replay: ' + name)
    finally:
        # Saved statistics remain authoritative after verification.
        for name, value in saved.items():
            if value is not None:
                modules[name].col_energy = value
    missing = [name for name, value in saved.items() if value is None]
    # Carry recovered legacy settings into subsequent continuation checkpoints.
    recorded_recipe = meta.setdefault('training_recipe', {})
    for key in ('anchor', 'anchor_len'):
        recorded_recipe[key] = recipe[key]
    print('C60-COL-ENERGY ' + json.dumps(dict(anchor=str(recipe['anchor']), anchor_len=length,
          windows=len(windows), seed=meta['seed'], selection='first 32 training windows (no RNG draws)',
          recomputed=missing, verified=len(saved)-len(missing))), flush=True)


def legacy_state(state, path, modules):
    """Convert a matching C44 resume checkpoint to the portable slot state."""
    path = Path(path)
    checkpoint = path.with_name('resume.pt')
    if path.name != 'final.pt' or not checkpoint.is_file():
        return None
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('step') != state['meta'].get('step') or saved.get('phase') != 'training':
        return None
    training = saved.get('training', {})
    source = training.get('modules', {})
    rows = {}
    for name, m in modules.items():
        key = name + '.parametrizations.weight.0'
        if key not in source:
            return None
        attrs, weights = source[key]['attributes'], source[key]['weights']
        sparse = attrs.get('sparse', False)
        if sparse and not m.sparse:
            raise ValueError('C59 legacy sparse state needs --sparse-state')
        reserve = m.idx.long().cpu() if m.sparse else None
        if m.sparse and not sparse:
            mask = weights.get('mask', attrs.get('mask'))
            if mask is not None and not torch.equal(mask.flatten().nonzero().flatten(), reserve):
                raise ValueError('C59 legacy dense reserve differs: ' + name)
        if sparse and not torch.equal(weights['idx'].long(), reserve):
            raise ValueError('C59 legacy reserve differs: ' + name)
        row = dict(reserve_idx=reserve, mask=m.mask.cpu().clone() if not m.sparse else None,
                   age_in_steps=attrs.get('age_in_steps', False))
        for field in TENSORS:
            value = weights.get(field, attrs.get(field))
            if value is not None and m.sparse and not sparse:
                if field == 'tau_scale':
                    value = value.flatten()[m.rows().cpu()]
                elif field not in ('col_energy',):
                    value = value.flatten()[reserve]
            row[field] = value
        rows[name] = row
    values = training.get('values', {})
    original = dict(state, meta=dict(state['meta']))
    original['meta']['bop_calibration'] = dict(cost_lambda=values.get('cost_lambda'), den_total=values.get('den_total'))
    cal = calibration(original, path, values.get('bop_tau'))
    cal['source'] = str(checkpoint)
    return dict(version=1, calibration=cal, modules=rows)
