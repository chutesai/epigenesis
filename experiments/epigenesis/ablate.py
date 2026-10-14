"""EP-1 session-1 ablation: named-rule/dedup baseline versus varied unnamed cases.

The established failure is a training/probe mismatch (apply prompts name the rule)
and claim-hash dedup leaving one apply example per policy. d0 retains frozen
accepted policy apply examples; d1 changes prompts and case diversity with an
ACTION-agreement teacher filter; d2 uses true-answer CE and format augmentation.
No probe is consulted when constructing training data. GPU imports remain lazy.
v2 uses category-table cloze filtering and probe-option targets for the pilot.

The d2 ladder uses row-scaled continuous, box, or ternary slots. ladder_tern's
local output penalty, netcost budget, and next-token CE rehearsal are default-off.
ladder_tern optionally restricts writes to reference-harness hidden units and
penalizes their general-text activation with --gate-lambda. --layers all requires
--projs down: each patched linear has full-shape fp32 masters (and Adam state). ladder_tern's --patch-beta defaults
to 1; ladder_two uses two ternary planes at .5 and .125 alpha. --init-from loads
a continuous best_state.pt, with RTN or GPTQ rounding; --steps 0 scores just the
quantized initialization. Held-out initialization error is measured on linear
outputs using continuous-checkpoint routed inputs. --diag is opt-in; zero delta/grad
mass has on-mask fraction 1. Fold gradients are mapped to d(BA) for that fraction,
while grad_rms always measures the trainable tensors themselves. Code counts are
over mask slots; sign_check compares served delta signs to parametrizer deltas.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
import epi_mask
import consolidate as c
import evaluate
import epi_fold
from epi_common import TopKRecord, pack_rows, free_slot_mask
from make_user import SELFSTUDY_POOL, case_matches, mint_case, render_ticket, resolution, large_dev_concepts


def gate_reference():
    """Import the reference helpers lazily; its guarded main never runs.

    epi_gpu uses e2e_fsa_v7, whose enable flags are independent of e2e_fsa's.
    Keep that module intact and disable its patches explicitly during stats.
    """
    candidates = ([Path(os.environ['EPI_REFERENCE_HARNESS'])] if os.environ.get('EPI_REFERENCE_HARNESS') else []) + [Path(__file__).resolve().parent.parent / 'end_to_end/e2e_fsa.py',
                  Path(__file__).resolve().parent / 'fsa_ref/e2e_fsa.py',
                  Path(os.environ.get('EPI_BOX_ROOT', '/workspace/malleable')) / 'epi_tbop/fsa_ref/e2e_fsa.py']
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError('Sleeping-neuron gating requires fsa_ref/e2e_fsa.py')
    directory = str(path.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from e2e_fsa import unit_activation_stats, select_units, GateHinge
    return unit_activation_stats, select_units, GateHinge


def reference_local_penalty():
    gate_reference()                      # puts the reference harness directory on sys.path
    from e2e_fsa import LocalOutputPenalty
    return LocalOutputPenalty


@torch.no_grad()
def restrict_gate_mask(mask, proj, selected):
    """Intersect the write set with hidden-unit rows (up) or columns (down)."""
    selected = selected.to(device=mask.device, dtype=torch.bool)
    return mask & (selected[:, None] if proj == 'up_proj' else selected[None, :])



def excluded_units(args):
    if not getattr(args, 'exclude_units', None):
        return {}
    gate_reference()
    from export_units import load_units
    return load_units(args.exclude_units)


def select_eligible_units(stats, frac, mode, seed, excluded, fact_stats=None):
    """Select k from the ownership complement; never select an excluded unit."""
    _, select_units, _ = gate_reference()
    result = {}
    for key, st in stats.items():
        if key not in excluded:
            raise ValueError('ownership record missing expert: ' + key)
        owned = excluded[key]
        if owned.shape != st['rate'].shape:
            raise ValueError('ownership shape mismatch: ' + key)
        idx = (~owned).nonzero().flatten()
        sel = torch.zeros_like(owned)
        if idx.numel():
            eligible = {k: v[idx] if torch.is_tensor(v) and v.shape == owned.shape else v for k,v in st.items()}
            fs = ({key: {k:v[idx] if torch.is_tensor(v) and v.shape == owned.shape else v
                         for k,v in fact_stats[key].items()}} if fact_stats and key in fact_stats else None)
            picked = select_units({key:eligible},frac,mode,seed,fs)[key]
            sel[idx[picked]] = True
        result[key] = sel
    return result


@torch.no_grad()
def exclude_write_units(model, fsa, args, owned):
    layers = fsa.moe_layers(model)
    for li, ei, proj, lin in patched_linears(model,fsa,args.layers,args.projs):
        key = f'{layers[li][0]}.experts.{ei}'
        if key not in owned:
            raise ValueError('ownership record missing expert: '+key)
        m = lin.parametrizations.weight[0]
        hidden = m.mask.shape[0 if proj=='up_proj' else 1]
        if owned[key].numel()!=hidden:
            raise ValueError('ownership shape mismatch: '+key)
        m.mask.copy_(restrict_gate_mask(m.mask,proj,~owned[key]))


@torch.no_grad()
def served_checkpoint(model, fsa, args, step, test_probes, dev_probes, test_score, dev_score):
    """Last-step served Q(M)*mask*support; never best_state or committed_state."""
    entries, geometry = {}, {}
    layers = fsa.moe_layers(model)
    for li,ei,proj,lin in patched_linears(model,fsa,args.layers,args.projs):
        m = lin.parametrizations.weight[0]
        if not isinstance(m,LadderParam) or m.arm!='ladder_tern' or m.beta_factors!=(1.,):
            raise ValueError('PR3 served export requires ladder_tern beta=1')
        q = m.codes()[0].detach()
        flat = q.flatten(); idx = flat.nonzero().flatten()
        name = f'{layers[li][0]}.experts.{ei}.{proj}'
        entries[name] = dict(kind='fsa',fmt='ternary',shape=list(q.shape),idx=idx.cpu().to(torch.int32),
                             val=flat[idx].cpu().to(torch.int8),alpha=m.alpha.detach().cpu().flatten().clone(),
                             scale='row',fp4_step=.25)
        geometry[name] = dict(codes=q.cpu().to(torch.int8).clone(),mask=m.mask.detach().cpu().clone(),
                              alpha=m.alpha.detach().cpu().clone(),layer=li)
    return dict(format='e2e-fsa-patch-v1',entries=entries,
                meta=dict(method='fsa_forced',fmt='ternary',patch_scale='row',layers=args.layers,
                          projs=args.projs,eda_backend='naive_fp32',export_step=58814,
                          tested_step=step,final_test_checkpoint='last step',
                          recipe={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                          moe_layer_names=[n for n,_ in layers]),
                served=geometry,ep1=dict(test_probes=test_probes,dev_probes=dev_probes,
                                       test=test_score,dev=dev_score))


def configure_gate(model, fsa, args, anchors, acquisition):
    if not args.gate_units_frac:
        return None, None
    unit_activation_stats, select_units, GateHinge = gate_reference()
    # The reference helper toggles its own classes, not epi_gpu's parametrizers.
    enabled = fsa.FSAParam.enabled
    fsa.set_enabled(False)
    try:
        stats = unit_activation_stats(model, args.layers, anchors[:32])
        fact_rows = [variant['row'] for item in acquisition for variant in item.get('variants', [item])]
        fact_stats = (unit_activation_stats(model, args.layers, fact_rows)
                      if args.gate_select == 'specific' else None)
    finally:
        fsa.set_enabled(enabled)
    owned = excluded_units(args)
    units = (select_eligible_units(stats,args.gate_units_frac,args.gate_select,args.seed,owned,fact_stats)
             if owned else select_units(stats,args.gate_units_frac,args.gate_select,args.seed,fact_stats))
    layers = fsa.moe_layers(model)
    for li, ei, proj, lin in patched_linears(model, fsa, args.layers, args.projs):
        selected = units[f'{layers[li][0]}.experts.{ei}']
        m = lin.parametrizations.weight[0]
        m.mask.copy_(restrict_gate_mask(m.mask, proj, selected))
    rates = torch.cat([stats[k]['rate'][units[k]] for k in units])
    all_rates = torch.cat([v['rate'] for v in stats.values()])
    fact_rates = (torch.cat([fact_stats[k]['rate'][units[k]] for k in units if k in fact_stats])
                  if fact_stats else None)
    info = dict(frac=args.gate_units_frac, select=args.gate_select, units=int(rates.numel()),
                sel_fact_rate_mean=float(fact_rates.mean()) if fact_rates is not None else None,
                sel_rate_mean=float(rates.mean()), all_rate_mean=float(all_rates.mean()),
                sel_rate_max=float(rates.max()), never_active_frac=float((rates == 0).float().mean()))
    print('GATE', json.dumps(info), flush=True)
    gate = GateHinge(model, args.layers, units, args.gate_margin) if args.gate_lambda > 0 else None
    return gate, info


def gate_backward(student, anchors, dev, gate, weight):
    """Record on this step's anchors; replay hook operations during checkpoints."""
    if gate is None:
        return 0., 0.
    try:
        gate.start()
        student(torch.tensor([entry['row'] for entry in anchors], device=dev))
        hinge = gate.stop()
        if hinge is None:
            return 0., 0.
        term = weight * hinge
        value, penalty = float(hinge.detach()), float(term.detach())
        if term.requires_grad:
            term.backward()
        return value, penalty
    finally:
        gate.on = False
        gate.end()
        gate.terms = []


def anchor_penalties_backward(student, anchors, dev, local, local_weight, gate, gate_weight):
    """Share anchor forwards and retain both hook replay protocols through backward.

    Use consolidation's ratio-of-total-energies normalization across micros.
    The no-grad pass also counts routed hinge elements for the same normalization.
    """
    if local is None:
        raw, weighted = gate_backward(student, anchors, dev, gate, gate_weight)
        return raw, weighted, 0.
    denominators, count = [], 0
    raw_hinge = hinge_value = local_value = 0.
    try:
        with torch.no_grad():
            for entry in anchors:
                local.start()
                if gate is not None:
                    gate.start()
                student(torch.tensor([entry['row']], device=dev))
                local.stop()
                denominators.extend(local.den)
                local.end()
                if gate is not None:
                    gate.stop()
                    count += gate.cnt
                    gate.end()
        denominator = torch.stack(denominators).sum() if denominators else None
        for entry in anchors:
            local.start()
            if gate is not None:
                gate.start()
            student(torch.tensor([entry['row']], device=dev))
            penalty = local.stop()
            hinge = gate.stop() if gate is not None else None
            term = None
            if penalty is not None:
                share = 1. if denominator is None else (torch.stack(local.den).sum() + 1e-12) / (denominator + 1e-12)
                term = local_weight * penalty * share
                local_value += float(term.detach())
            if hinge is not None:
                raw = hinge * gate.cnt / max(count, 1)
                ht = gate_weight * raw
                raw_hinge += float(raw.detach())
                hinge_value += float(ht.detach())
                term = ht if term is None else term + ht
            if term is not None and term.requires_grad:
                term.backward()
            local.end()
            if gate is not None:
                gate.end()
        return raw_hinge, hinge_value, local_value
    finally:
        local.on = False
        local.end()
        local.num, local.den = [], []
        if gate is not None:
            gate.on = False
            gate.end()
            gate.terms = []


def trained_policies(corpus, session):
    return [p for p in corpus['policies'] if p['intro_session'] <= session]


def render_case(case, policy, fmt):
    """Return separate prompt/true target and the ACTION character span in target."""
    ticket, action = render_ticket(case), policy['action']
    if fmt == 0:
        prompt, answer = f'{ticket}\nResolution:', ' ' + resolution(policy)
        lo = answer.index('ACTION: ') + len('ACTION: ')
    else:
        prompts = {
            1: f"Q: Under the user's policies, which action applies to {ticket}\nA:",
            2: f'User: {ticket}\nWhich action does the desk take?\nAssistant:',
            3: f'{ticket} The desk will',
            4: f'Policy check: {ticket} ->',
            5: f'Q: What should the desk do with this ticket? {ticket}\nA:',
        }
        if fmt not in prompts:
            raise ValueError('Unknown format')
        prompt, answer, lo = prompts[fmt], ' ' + action, 1
        if fmt == 3:
            answer += '.'
    return dict(prompt=prompt, answer=answer, action_span=(lo, lo + len(action)), format=fmt)


def build_cases(corpus, session=1, seed=0, cases_per_policy=16, dedup=False, appearances_only=False):
    if cases_per_policy < 2 or cases_per_policy % 2:
        raise ValueError('cases-per-policy must be positive and even (half per family)')
    rng = random.Random(seed * 1000 + session)
    sources = {}
    for s in corpus['sessions']:
        if s['session'] <= session:
            for ep in s['episodes']:
                if ep['kind'] in ('policy_intro', 'policy_recur'):
                    sources.setdefault(ep['policy_id'], ep)
    cases, seen = [], set()
    current_sources = {ep['policy_id']: ep for s in corpus['sessions'] if s['session'] == session
                       for ep in s['episodes'] if ep['kind'] in ('policy_intro', 'policy_recur')}
    for policy in trained_policies(corpus, session):
        if appearances_only and policy['id'] not in current_sources:
            continue
        ep = (current_sources[policy['id']] if appearances_only else sources[policy['id']])
        # Keep the complete policy table for mint_case's unique-match validation.
        for index in range(cases_per_policy):
            while True:
                case = mint_case(rng, corpus['policies'], SELFSTUDY_POOL, category=policy['category'])
                cond = policy['condition']
                case[policy['attribute']] = (rng.randint(cond['min'], cond['max']) if cond['kind'] == 'range'
                                             else rng.choice(cond['values']))
                key = json.dumps(case, sort_keys=True)
                if not dedup or key not in seen:
                    seen.add(key)
                    break
            assert case_matches(policy, case)
            cases.append(dict(id=f"{policy['id']}-{index:04d}", policy_id=policy['id'], case=case,
                              source_episode=ep['id'], source_text=ep['text'],
                              formats=[render_case(case, policy, f) for f in range(6)],
                              base_format=index % 2))
    return cases


def v2_probe(case, policies):
    siblings = sorted((p for p in policies if p['category'] == case['case']['category']),
                      key=lambda p: p['id'])
    family = 'ticket' if case['base_format'] == 0 else 'qa'
    options = [f"STATUS: {p['status']} | ACTION: {p['action']}" if family == 'ticket'
               else p['action'] for p in siblings]
    truth = [p['id'] for p in siblings].index(case['policy_id'])
    context = evaluate.category_table(policies, case['case']['category'])   # same text as the v2.1 ICL arm
    return dict(prompt=case['formats'][case['base_format']]['prompt'], family=family,
                options=options, answer_idx=truth, context=context)


def v2_rows(case, probe, tok, bos):
    enc = lambda text: tok.encode(text, add_special_tokens=False).ids
    prompt_ids = enc(probe['prompt'].rstrip())
    answer_ids = enc(' ' + probe['options'][probe['answer_idx']])
    prefix = [bos] + enc(probe['context']) + enc('\n\n') + prompt_ids
    item = dict(id=case['id'], policy_id=case['policy_id'], source_episode=case['source_episode'],
                family=probe['family'], prompt_ids=prompt_ids, answer_ids=answer_ids,
                row=[bos] + prompt_ids + answer_ids, start=1 + len(prompt_ids))
    return item, prefix


def g1_stats(hits):
    def stats(values):
        return dict(attempted=len(values), kept=sum(values),
                    keep_rate=sum(values) / len(values) if values else 0.)
    return dict(stats(hits['ticket'] + hits['qa']),
                by_family={family: stats(values) for family, values in hits.items()})


def cloze_correct(scores, truth):
    return max(range(len(scores)), key=scores.__getitem__) == truth


def smoothed_acc(trajectory):
    """Require three evaluations before using a moving mean for selection."""
    return sum(r['dev']['acc'] for r in trajectory[-3:]) / 3 if len(trajectory) >= 3 else None


def best_smoothed_acc(trajectory):
    # The dNLL limit belongs to the checkpoint ending each three-eval window.
    return max((smoothed_acc(trajectory[:i + 1]) for i in range(2, len(trajectory))
                if trajectory[i]['select_dnll'] <= .005), default=0.)


def gate_g2(trajectory, dev_large=False):
    if dev_large:
        return best_smoothed_acc(trajectory)
    return max((r['dev']['acc'] for r in trajectory if r['select_dnll'] <= .005), default=0.)


def trainable_state(modules):
    """CPU copies keyed by patch identity; support is needed to re-score slots."""
    state = {}
    for name, module in modules.items():
        tensors = {key: value for key, value in module.named_parameters() if value.requires_grad}
        if isinstance(module, LadderParam):
            tensors['mask'] = module.mask
        if hasattr(module, 'support'):
            tensors['support'] = module.support
        state[name] = {key: value.detach().cpu().clone() for key, value in tensors.items()}
    return state


def effective_lr(args):
    if args.arm.startswith('ladder_'):
        return .02 if args.arm in ('ladder_tern', 'ladder_two') else args.lr
    return .001 if args.arm == 'lorafold' else (.02 if args.arm == 'tmid' else .001) * args.lr_mult


class LadderParam(torch.nn.Module):
    """Full-shape fp32 masters; optional support uses the harness ternary STE."""
    def __init__(self, W, mask, arm, enabled_source, patch_beta=1., budgeted=False):
        torch.nn.Module.__init__(self)
        self.budgeted = budgeted
        if budgeted:
            enabled_source.__init__(self, W, mask, fmt="ternary", budgeted=True)
            self.support.zero_()
        elif arm == 'ladder_two':
            self.M1 = torch.nn.Parameter(torch.zeros_like(W, dtype=torch.float32))
            self.M2 = torch.nn.Parameter(torch.zeros_like(W, dtype=torch.float32))
        else:
            self.M = torch.nn.Parameter(torch.zeros_like(W, dtype=torch.float32))
        if not budgeted:
            self.register_buffer('mask', mask.bool())
            nz = W != 0
            self.register_buffer('alpha', W.float().abs().sum(1, keepdim=True) /
                                 nz.sum(1, keepdim=True).clamp_min(1))
        self.arm, self.enabled_source = arm, enabled_source
        self.beta_factors = (.5, .125) if arm == 'ladder_two' else (patch_beta,)

    def codes(self):
        masters = (self.M1, self.M2) if self.arm == 'ladder_two' else (self.M,)
        return tuple(ternary_round(m) * self.mask * (self.support if self.budgeted else 1)
                     for m in masters)

    def delta_w(self):
        if self.budgeted and self.arm == 'ladder_tern':
            return self.beta_factors[0] * self.enabled_source.delta(self)
        if self.arm == 'ladder_two':
            return self.alpha * sum(b * (q.detach() + (m - m.detach()) * self.mask)
                                    for b, q, m in zip(self.beta_factors, self.codes(), (self.M1, self.M2)))
        m = self.M * self.mask
        if self.arm == 'ladder_box':
            m = m.clamp(-1, 1)
        elif self.arm == 'ladder_tern':
            q = ternary_round(m)
            m = q.detach() + (m - m.detach())
            m = self.beta_factors[0] * m
        return self.alpha * m

    def forward(self, W):
        return W + self.delta_w().to(W.dtype) if self.enabled_source.enabled else W


def ternary_round(v):
    """Nearest {-1, 0, 1}, with the existing zero tie at +/-0.5."""
    return v.sign() * (v.abs() > .5)


def round_grid(v, factors):
    """Return code planes and represented value, in alpha units."""
    residual, represented, planes = v, torch.zeros_like(v), []
    for beta in factors:
        code = ternary_round(residual / beta)
        planes.append(code)
        represented = represented + beta * code
        residual = residual - beta * code
    return tuple(planes), represented


@torch.no_grad()
def initialize_codes(values, mask, factors, hessian=None):
    """RTN or column-order GPTQ; fixed positions neither emit nor absorb errors.

    Uses the upper Cholesky factor of the damped inverse Hessian, as in
    https://github.com/IST-DASLab/gptq/blob/main/gptq.py .
    """
    work = values.float().clone() * mask
    if hessian is None or not hessian.any():
        return tuple(q * mask for q in round_grid(work, factors)[0])
    H = hessian.to(device=work.device, dtype=torch.float32).clone()
    diagonal = H.diagonal()
    damp = .01 * diagonal.mean()
    diagonal.add_(damp)
    # Unrouted input columns have no measured energy.
    diagonal[diagonal == 0] = 1.
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
    upper = torch.linalg.cholesky(inverse, upper=True)
    planes = [torch.zeros_like(work) for _ in factors]
    for col in range(work.shape[1]):
        codes, q = round_grid(work[:, col], factors)
        for plane, code in zip(planes, codes):
            plane[:, col] = code * mask[:, col]
        error = (work[:, col] - q) * mask[:, col] / upper[col, col]
        work[:, col + 1:] -= (error[:, None] * upper[col, col + 1:]) * mask[:, col + 1:]
    return tuple(planes)


@torch.no_grad()
def plane_stats(modules):
    quantized = [m for m in modules.values() if isinstance(m, LadderParam)
                 and m.arm in ('ladder_tern', 'ladder_two')]
    if not quantized:
        return {}
    nnz = [0] * len(quantized[0].beta_factors)
    used = 0
    for m in quantized:
        codes = m.codes()
        for i, code in enumerate(codes):
            nnz[i] += int(torch.count_nonzero(code))
        used += int(torch.stack([q != 0 for q in codes]).any(0).sum())
    return dict(plane_nnz=nnz, used_positions=used,
                combined_bits=used * len(nnz) * math.log2(3))


@torch.no_grad()
def initialize_ladder(model, modules, linears, args, acquisition, student, dev, metadata):
    """Calibrate on continuous-checkpoint routed inputs, then install codes.

    Hooks cache routed inputs on CPU in 16 calibration and four disjoint
    held-out micros. Build one Hessian at a time to bound GPU memory.
    No dev/test probes are used.
    """
    saved = torch.load(args.init_from, map_location='cpu', weights_only=True)
    config = saved.get('config', {})
    if config.get('arm') != 'ladder_cont' or set(saved['state']) != set(modules):
        raise ValueError('Initialization requires a ladder_cont checkpoint with the same linears')
    for key in ('write_set', 'mask_sha256'):
        if config.get(key) != metadata.get(key):
            raise ValueError(f'Initialization write mask mismatch: {key}')
    values = {}
    for name, m in modules.items():
        state = saved['state'][name]
        v = state['M'].to(m.alpha.device)
        if v.shape != m.mask.shape or not torch.isfinite(v).all() or v[~m.mask].any():
            raise ValueError(f'Invalid continuous master or mask: {name}')
        if 'mask' in state and not torch.equal(state['mask'].to(m.mask.device), m.mask):
            raise ValueError(f'Initialization mask mismatch: {name}')
        if 'mask' not in state and not metadata.get('mask_sha256'):
            raise ValueError('Checkpoint needs a saved mask or forced-mask hash')
        values[name] = v
    # Keep every patch continuous until all routed input measurements finish.
    originals = {name: (m.arm, getattr(m, 'M', None)) for name, m in modules.items()}
    codes, stats = {}, {}
    entries = list(acquisition)
    random.Random(args.seed + 741).shuffle(entries)
    if len(entries) < 2:
        raise ValueError('Initialization needs separate calibration and held-out items')
    holdout_count = min(4, len(entries) // 2)
    heldout, train = entries[:holdout_count], entries[holdout_count:]
    calibration = [train[i % len(train)] for i in range(16)]
    heldout = [heldout[i % len(heldout)] for i in range(4)]
    routed = {name: [[], []] for name in modules}
    hooks = []
    phase = 0
    def capture(name, width):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().reshape(-1, width)
            if x.numel():
                routed[name][phase].append(x.cpu())
        return hook
    try:
        for name, m in modules.items():
            m.arm = 'ladder_cont'
            m.M = torch.nn.Parameter(values[name], requires_grad=False)
        try:
            for li, ei, proj, lin in linears:
                hooks.append(lin.register_forward_hook(capture(f'{li}.{ei}.{proj}', lin.in_features)))
            for phase, batches in enumerate((calibration, heldout)):
                for entry in batches:
                    student(torch.tensor([entry['row']], device=dev))
        finally:
            for hook in hooks:
                hook.remove()
        total_error, total_reference, total_patch_reference = 0., 0., 0.
        for li, ei, proj, lin in linears:
            name = f'{li}.{ei}.{proj}'
            m = modules[name]
            training_inputs, heldout_inputs = routed.pop(name)
            H = None
            if args.init_mode == 'gptq':
                for x in training_inputs:
                    x = x.to(device=m.alpha.device, dtype=torch.float32)
                    gram = x.T @ x
                    if H is None:
                        H = gram
                    else:
                        H.add_(gram)
            codes[name] = tuple(q.cpu() for q in initialize_codes(
                values[name], m.mask, m.beta_factors, H))
            del H
            q = sum(b * code.to(m.alpha.device) for b, code in zip(m.beta_factors, codes[name]))
            error, reference, patch_reference = 0., 0., 0.
            target = values[name] * m.mask * m.alpha
            diff = q * m.alpha - target
            full_target = lin.parametrizations.weight.original.float() + target
            for x in heldout_inputs:
                x = x.to(device=m.alpha.device, dtype=torch.float32)
                error += float((x @ diff.T).square().sum())
                patch_reference += float((x @ target.T).square().sum())
                output = x @ full_target.T
                if lin.bias is not None:
                    output = output + lin.bias.float()
                reference += float(output.square().sum())
            stats[name] = dict(relative_output_error=math.sqrt(error / reference) if reference else None,
                               relative_patch_output_error=math.sqrt(error / patch_reference) if patch_reference else None,
                               routed_heldout=bool(heldout_inputs))
            total_error += error
            total_reference += reference
            total_patch_reference += patch_reference
    finally:
        for name, m in modules.items():
            m.arm, original = originals[name]
            if original is None:
                del m.M
            else:
                m.M = original
    for name, m in modules.items():
        masters = (m.M1, m.M2) if m.arm == 'ladder_two' else (m.M,)
        for master, code in zip(masters, codes[name]):
            master.copy_(code)
    return dict(mode=args.init_mode, calibration_micro_batches=16, heldout_micro_batches=4,
                relative_output_error=math.sqrt(total_error / total_reference) if total_reference else None,
                relative_patch_output_error=math.sqrt(total_error / total_patch_reference) if total_patch_reference else None,
                error_coordinates='linear outputs on continuous-checkpoint routed inputs', per_linear=stats)


def patched_linears(model, fsa, layers, projs):
    for li in layers:
        _, moe = fsa.moe_layers(model)[li]
        for ei, expert in enumerate(moe.experts):
            if expert is not None:
                for proj in projs:
                    lin = getattr(expert, proj)
                    if hasattr(lin, 'parametrizations') and 'weight' in lin.parametrizations:
                        yield li, ei, proj, lin


def attach_ladder(model, fsa, args):
    budgeted = args.arm == 'ladder_tern' and getattr(args, 'budget', 0) > 0
    # Energy collection and module discovery require the harness FSAParam type.
    cls = type('BudgetLadderParam', (LadderParam, fsa.FSAParam), {}) if budgeted else LadderParam
    factory = lambda W, mask: cls(W, mask, args.arm, fsa.FSAParam, args.patch_beta, budgeted)
    if args.write_set == 'forced':
        return epi_mask.attach(model, fsa, layers=args.layers, projs=args.projs, param_factory=factory)
    params = []
    for li in args.layers:
        _, moe = fsa.moe_layers(model)[li]
        for expert in moe.experts:
            if expert is None:
                continue
            for proj in args.projs:
                lin = getattr(expert, proj)
                m = factory(lin.weight.detach(), free_slot_mask(lin.weight.detach()))
                torch.nn.utils.parametrize.register_parametrization(lin, 'weight', m)
                params.extend(m.parameters())
    return params


def mass_fraction(total, on_mask):
    return on_mask / total if total else 1.


@torch.no_grad()
def slot_diagnostics(linears, step, event):
    """Aggregate actual served deltas. Factor gradients use d(BA) coordinates."""
    groups, grad_sq, grad_n, grad_mass, grad_on = {}, 0., 0, 0., 0.
    for li, _, proj, lin in linears:
        m = lin.parametrizations.weight[0]
        base = lin.parametrizations.weight.original
        delta = lin.weight.float() - base.float()
        mask = getattr(m, 'mask', base == 0)
        key = (li, proj.removesuffix('_proj'))
        s = groups.setdefault(key, dict(nnz=0, absmax=0., square=0., size=0, mass=0., on=0.,
                                       sign_mismatches=0))
        s['nnz'] += int(torch.count_nonzero(delta))
        s['absmax'] = max(s['absmax'], float(delta.abs().max()))
        s['square'] += float(delta.square().sum())
        s['size'] += delta.numel()
        s['mass'] += float(delta.abs().sum())
        s['on'] += float(delta[mask].abs().sum())
        expected = m.delta_w().to(base.dtype).float()
        s['sign_mismatches'] += int((delta.sign() != expected.sign()).sum())
        for p in m.parameters():
            if p.requires_grad:
                grad_n += p.numel()
                if p.grad is not None:
                    grad_sq += float(p.grad.float().square().sum())
        if isinstance(m, LadderParam) and m.arm == 'ladder_two':
            grad = sum((p.grad if p.grad is not None else torch.zeros_like(p)) for p in (m.M1, m.M2))
        elif hasattr(m, 'M'):
            grad = m.M.grad
        else:
            grad = torch.zeros_like(delta)
            if m.B.grad is not None:
                grad.add_(m.B.grad @ m.A)
            if m.A.grad is not None:
                grad.add_(m.B @ m.A.grad)
        if grad is not None:
            grad_mass += float(grad.abs().sum())
            grad_on += float(grad[mask].abs().sum())
        if isinstance(m, epi_fold.FoldLoRAParam):
            ba = m.B @ m.A
            s['ba_mass'] = s.get('ba_mass', 0.) + float(ba.abs().sum())
            s['ba_on'] = s.get('ba_on', 0.) + float(ba[mask].abs().sum())
            codes = m.codes()[mask]
            counts = s.setdefault('code_distribution', {'-1': 0, '0': 0, '+1': 0})
            for label, code in [('-1', -1), ('0', 0), ('+1', 1)]:
                counts[label] += int((codes == code).sum())
            for name in ('A', 'B'):
                grad = getattr(m, name).grad
                field = name + '_grad_nonzero'
                s[field] = s.get(field, False) or (grad is not None and bool(grad.any()))
    per_layer = {}
    for (li, proj), s in groups.items():
        s['rms_delta'] = math.sqrt(s.pop('square') / s.pop('size'))
        s['frac_on_mask'] = mass_fraction(s.pop('mass'), s.pop('on'))
        s['mask_check'] = 'PASS' if s['frac_on_mask'] >= 1 - 1e-6 else 'FAIL'
        s['sign_check'] = 'PASS' if s['sign_mismatches'] == 0 else 'FAIL'
        if 'ba_mass' in s:
            s['ba_frac_on_mask'] = mass_fraction(s.pop('ba_mass'), s.pop('ba_on'))
        per_layer.setdefault(li, {})[proj] = s
    return dict(phase='diag', step=step, event=event, per_layer=per_layer,
                grad_rms=math.sqrt(grad_sq / grad_n) if grad_n else 0.,
                grad_frac_on_mask=mass_fraction(grad_mass, grad_on),
                grad_coordinates='dBA' if any('ba_frac_on_mask' in s for s in groups.values()) else 'master',
                mask_check='FAIL' if any(s['mask_check'] == 'FAIL' for s in groups.values()) else 'PASS',
                sign_check='FAIL' if any(s['sign_check'] == 'FAIL' for s in groups.values()) else 'PASS')


def refresh_due(step, until=None):
    return step % 8 == 0 and (until is None or step <= until)


def encode_item(item, rendered, tok, bos):
    prompt_ids = tok.encode(rendered['prompt'], add_special_tokens=False).ids
    encoded = tok.encode(rendered['answer'], add_special_tokens=False)
    lo, hi = rendered['action_span']
    positions = [i for i, (a, b) in enumerate(encoded.offsets) if a < hi and b > lo]
    if not positions:
        raise ValueError('ACTION must cover at least one target token')
    return dict(id=item['id'], policy_id=item['policy_id'], source_episode=item['source_episode'],
                prompt_ids=prompt_ids, answer_ids=encoded.ids, action_positions=positions,
                row=[bos] + prompt_ids + encoded.ids, start=1 + len(prompt_ids))


def teacher_agrees(logprobs, answer_ids, action_positions):
    if not action_positions:
        raise ValueError('Empty ACTION positions')
    if logprobs.shape[0] != len(answer_ids):
        raise ValueError('Teacher and true target must align')
    predicted = logprobs.argmax(-1).cpu().tolist()
    return all(predicted[i] == answer_ids[i] for i in action_positions)


def sample_policies(entries, n, rng, augment=False):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['policy_id']].append(entry)
    if not groups:
        raise ValueError('No training items survived')
    keys = sorted(groups)
    chosen = [rng.choice(groups[rng.choice(keys)]) for _ in range(n)]
    return [rng.choice(e['variants']) for e in chosen] if augment else chosen


def filtered_probes(corpus, split, policy_ids, limit=None):
    items = [i for i in corpus['probes'][split]['concept'] if i['policy_id'] in policy_ids]
    if limit is None:
        return items
    # Round-robin families so smoke measures both, even when probes are blocked by family.
    groups = {f: [i for i in items if i['family'] == f] for f in ('ticket', 'qa')}
    ordered = []
    for index in range(max(map(len, groups.values()), default=0)):
        for family in groups:
            if index < len(groups[family]):
                ordered.append(groups[family][index])
    return ordered[:limit]


def ce_backward(student, entries, dev, vocab, micro_tokens=160, coefficient=.6):
    """Answer-only token mean; prefixes and right padding have no CE contribution."""
    total = sum(len(e['row']) - e['start'] for e in entries)
    if total == 0:
        raise ValueError('CE requires target tokens')
    value = 0.
    for batch in c.micro_batches(entries, micro_tokens):
        x, _ = pack_rows([e['row'] for e in batch], [e['start'] for e in batch], dev)
        logits = student(x)
        loss = sum(torch.nn.functional.cross_entropy(
            logits[i, e['start'] - 1:len(e['row']) - 1, :vocab].float(),
            x[i, e['start']:len(e['row'])], reduction='sum') for i, e in enumerate(batch)) / total
        value += float(loss.detach())
        if loss.requires_grad:
            (coefficient * loss).backward()
    return value, total


def training_backward(student, batch, anchors, dev, vocab, arm, data, rehearsal=0.):
    rehearsal_frac = rehearsal if arm == "ladder_tern" else .2 if arm == 'lorafold' else 0.
    coefficients = (1 - rehearsal_frac, 0., 0.) if rehearsal_frac else (.6, .4, 0.)
    terms = dict(gen=[] if rehearsal_frac else anchors)
    if data == 'd2':
        acq, nt = ce_backward(student, batch, dev, vocab, coefficient=coefficients[0])
    else:
        terms['acq'] = batch
    _, losses = c.loss_assembly(student, terms, dev, vocab, micro_tokens=160,
                                backward=True, coefficients=coefficients)
    if data == 'd2':
        losses['acq'], losses['tokens']['acq'] = acq, nt
        losses['loss'] += coefficients[0] * acq
    if rehearsal_frac:
        # Port the tbop rehearsal loop: patched next-token CE over each anchor window.
        rehearsal = 0.
        for entry in anchors:
            x = torch.tensor([entry['row']], device=dev)
            rl = c.general_ce(student, x, vocab)
            rehearsal += float(rl.detach()) / len(anchors)
            if rl.requires_grad:
                (rehearsal_frac * rl / len(anchors)).backward()
        losses['rehearsal_ce'] = rehearsal
        losses['loss'] += rehearsal_frac * rehearsal
    return losses


def best_dev_step(trajectory):
    return max(trajectory, key=lambda r: (r['dev']['acc'], -r['nnz'], -r['step']))['step']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exclude-units', type=Path, help='PR3 whole-unit ownership JSON; exclude before gate selection')
    p.add_argument('--save-served', action='store_true', help='dump the last-step served ternary patch and scored probes')
    p.add_argument('--eda-backend', choices=['naive_fp32'], default='naive_fp32')
    p.add_argument('--export',type=Path,choices=[Path('/workspace/malleable/data/hf/exports/725B-tokens_step58814')],
                   default=Path('/workspace/malleable/data/hf/exports/725B-tokens_step58814'))
    p.add_argument('--export-step',type=int,choices=[58814],default=58814)
    p.add_argument('--write-set', choices=('inpair', 'forced'), default='inpair')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=('tmid', 'loramid', 'lorafold',
                                    'ladder_cont', 'ladder_box', 'ladder_tern', 'ladder_two'), required=True)
    p.add_argument('--data', choices=('d0', 'd1', 'd2', 'v2'), required=True)
    for name, default in [('steps', 150), ('eval-every', 25), ('cases-per-policy', None), ('seed', 0)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--session', type=int, choices=(1,), default=1)
    p.add_argument('--refresh-until', type=int, default=None, help='last support refresh step (inclusive)')
    p.add_argument('--lr-mult', type=float, default=1.)
    p.add_argument('--lr', type=float, default=1e-3, help='continuous/box ladder learning rate')
    p.add_argument('--local-lambda', type=float, default=0.)
    p.add_argument('--budget', type=int, default=0, help='ladder_tern netcost ceiling; 0 disables')
    p.add_argument('--rehearsal', type=float, default=0.)
    p.add_argument('--patch-beta', type=float, default=1., help='ladder_tern patch scale in alpha units')
    p.add_argument('--gate-units-frac', type=float, default=0., help='hidden-unit fraction; 0 disables gating')
    p.add_argument('--gate-select', choices=('sleep', 'specific', 'random', 'active'), default='sleep')
    p.add_argument('--gate-lambda', type=float, default=0., help='general-text gate hinge weight')
    p.add_argument('--gate-margin', type=float, default=0.)
    p.add_argument('--init-from', type=Path, help='ladder_cont best_state.pt on the same mask')
    p.add_argument('--init-mode', choices=('rtn', 'gptq'), default='rtn')
    p.add_argument('--diag', action='store_true', help='log served slot and gradient diagnostics')
    p.add_argument('--layers', default='14,15', help='ladder MoE layer indices, or all (32 layers)')
    p.add_argument('--projs', choices=('up', 'down', 'up,down'), default='up,down',
                   help='ladder projections; all layers requires down (full-shape fp32 masters)')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--g1-only', action='store_true')
    p.add_argument('--dev-large', type=int, default=0, help='reserved dev concept count (even; 0 uses corpus dev)')
    p.add_argument('--save-best', action='store_true', help='save the best three-evaluation mean patch to best_state.pt')
    args = p.parse_args(argv)
    if args.dev_large < 0 or args.dev_large % 2:
        p.error('dev-large must be nonnegative and even (half per family)')
    if (args.exclude_units or args.save_served) and (args.arm != 'ladder_tern' or args.write_set != 'forced' or args.patch_beta != 1.):
        p.error('PR3 exclusion/export requires forced ladder_tern beta=1')
    all_layers = args.layers == 'all'
    try:
        args.layers = list(range(32)) if all_layers else [int(v) for v in args.layers.split(',')]
    except ValueError:
        p.error('layers must be comma-separated indices or all')
    if not args.layers or len(set(args.layers)) != len(args.layers) or any(v < 0 or v >= 32 for v in args.layers):
        p.error('layers must be unique MoE indices in 0..31')
    args.projs = [v + '_proj' for v in args.projs.split(',')]
    ladder = args.arm.startswith('ladder_')
    quantized_ladder = args.arm in ('ladder_tern', 'ladder_two')
    if not math.isfinite(args.gate_units_frac) or not 0 <= args.gate_units_frac <= 1:
        p.error('gate-units-frac must be finite and in [0, 1]')
    if any(not math.isfinite(v) or v < 0 for v in (args.gate_lambda, args.gate_margin)):
        p.error('gate-lambda and gate-margin must be finite and nonnegative')
    if (args.gate_units_frac or args.gate_lambda or args.gate_margin) and args.arm != 'ladder_tern':
        p.error('gating applies only to ladder_tern')
    if not math.isfinite(args.local_lambda) or args.local_lambda < 0:
        p.error('local-lambda must be finite and nonnegative')
    if args.budget < 0:
        p.error('budget must be nonnegative')
    if not math.isfinite(args.rehearsal) or not 0 <= args.rehearsal <= 1:
        p.error('rehearsal must be finite and in [0, 1]')
    if (args.local_lambda or args.budget or args.rehearsal) and args.arm != 'ladder_tern':
        p.error('local-lambda/budget/rehearsal apply only to ladder_tern')
    if not math.isfinite(args.patch_beta) or args.patch_beta <= 0:
        p.error('patch-beta must be finite and positive')
    if args.patch_beta != 1. and args.arm != 'ladder_tern':
        p.error('patch-beta applies only to ladder_tern')
    if args.init_from is not None and not quantized_ladder:
        p.error('init-from requires ladder_tern or ladder_two')
    if not ladder and (args.layers != [14, 15] or args.projs != ['up_proj', 'down_proj']):
        p.error('layers/projs selection is supported only for ladder arms')
    if (all_layers or len(args.layers) == 32) and args.projs != ['down_proj']:
        p.error('all layers requires --projs down: masters are full-shape fp32 per linear')
    if ladder and args.data != 'd2':
        p.error('ladder arms require --data d2 (answer-only CE and session-start KL)')
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error('lr must be finite and positive')
    if args.g1_only and args.data != 'v2':
        p.error('g1-only requires --data v2')
    if args.cases_per_policy is None:
        args.cases_per_policy = 32 if args.data == 'v2' else 16
    if args.smoke:
        args.steps, args.eval_every, args.cases_per_policy = 4, 2, 4
    if args.steps < (0 if quantized_ladder else 1) or args.eval_every < 1 or args.cases_per_policy < 2 or args.cases_per_policy % 2:
        p.error('steps/eval-every must be positive; cases-per-policy must be positive and even')
    if not math.isfinite(args.lr_mult) or args.lr_mult <= 0:
        p.error('lr-mult must be finite and positive')
    if args.refresh_until is not None and args.refresh_until < 0:
        p.error('refresh-until must be nonnegative')
    # Fixed consolidation settings, recorded with the public CLI configuration.
    args.topk, args.nograd_tokens, args.ent_max = 32, 4096, 2.
    args.micro_tokens = 160
    if args.arm == 'tmid':
        args.budget = 2000000
    return args


def run(args):
    root, out = args.run, args.out
    for relative in ('corpus.json', 'genome/metrics.json', 'genome/panel.pt'):
        if not (root / relative).is_file():
            raise ValueError(f'Missing required artifact: {root / relative}')
    corpus = json.loads((root / 'corpus.json').read_text())
    policy_ids = {p['id'] for p in trained_policies(corpus, args.session)}
    if not policy_ids:
        raise ValueError('No introduced policies')
    g = c.gpu()
    fsa = g.fsa
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + args.session)
    model, dev = g.build()
    c.checkpoint_recurrences(model)
    tok = g.load_tokenizer()
    ternary = args.arm == 'tmid'
    ladder = args.arm.startswith('ladder_')
    budgeted = ternary or (args.arm == 'ladder_tern' and args.budget > 0)
    model._epi_write_set = dict(write_set=args.write_set, mask_sha256=None)
    if ladder:
        params = attach_ladder(model, fsa, args)
    elif args.write_set == 'forced' and args.arm != 'loramid':
        params = epi_mask.attach(model, fsa, fmt='ternary' if ternary else 'fp4', budgeted=ternary,
                                 rank=16 if args.arm == 'lorafold' else None)
    elif args.arm == 'lorafold':
        params = epi_fold.attach(model, fsa)
    else:
        params, _, _ = fsa.attach(model, [14, 15], 'fsa_free' if ternary else 'lora', 16,
                                 fmt='ternary', budgeted=ternary)
    owned = excluded_units(args)
    if owned:
        exclude_write_units(model,fsa,args,owned)
    mods = fsa.fsa_modules(model)
    for m in mods:
        if ternary:
            if args.write_set == 'inpair':
                m.enable_block_scale(16, .25)
                params.append(m.logb)
            m.support.zero_()
    linears = list(patched_linears(model, fsa, args.layers, args.projs))
    modules = {f'{li}.{ei}.{proj}': lin.parametrizations.weight[0] for li, ei, proj, lin in linears}
    identity = {f'{li}.{ei}.{proj}': (li, ei) for li, ei, proj, _ in linears}
    frozen = {} if ladder and not budgeted else {name: torch.zeros_like(m.mask) for name, m in modules.items() if hasattr(m, 'M')}
    fsa.set_enabled(True)
    out.mkdir(parents=True, exist_ok=True)
    kept, attempted, dropped = 0, 0, 0
    if args.data == 'd0':
        study = json.loads((root / 'selfstudy/s1.json').read_text())
        if study['session'] != args.session:
            raise ValueError('Self-study session mismatch')
        items = [i for i in study['items'] if i['accepted'] and i.get('policy_id') in policy_ids]
        c.validate_acquisition(items, study['null_thr'])
        if len({i['id'] for i in items}) != len(items):
            raise ValueError('Frozen item IDs must be unique')
        _, _, acquisition, dropped = c.cache_items(model, dev, items, study['chunk_tokens'], args, study['null_thr'])
        attempted, kept = len(items), len(acquisition)
    elif args.data == 'v2':
        hits = {'ticket': [], 'qa': []}
        acquisition = []
        for case in build_cases(corpus, args.session, args.seed, args.cases_per_policy, dedup=True):
            probe = v2_probe(case, corpus['policies'])
            item, prefix = v2_rows(case, probe, tok, g.BOS)
            scores = g.score_options(model, tok, prefix, probe['options'], dev)
            keep = cloze_correct(scores, probe['answer_idx'])
            hits[probe['family']].append(keep)
            if keep and not args.g1_only:
                lp = g.logprobs_at(model, [prefix + item['answer_ids']], [len(prefix)],
                                   dev, args.nograd_tokens)[0]
                acquisition.append(dict(item, record=TopKRecord.from_logits(lp, args.topk).to('cpu'),
                                        weights=torch.ones(len(item['answer_ids']))))
                del lp
        g1 = g1_stats(hits)
        attempted, kept = g1['attempted'], g1['kept']
        if args.g1_only or g1['keep_rate'] < .80:
            config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            config.update(epi_mask.metadata(model))
            if args.arm == 'lorafold':
                config.update(rehearsal_frac=.2, provisional=True)
            result = dict(config=config, g1=g1, g1_pass=g1['keep_rate'] >= .80,
                          attempted=attempted, kept=kept, keep_rate=g1['keep_rate'],
                          effective_lr=effective_lr(args))
            diagnostics = [slot_diagnostics(linears, 0, 'initial')] if args.diag else []
            result.update(arm=args.arm, write_set=args.write_set, layers=args.layers, projs=args.projs,
                          lr=effective_lr(args), diag_lines=diagnostics)
            for row in diagnostics:
                print(json.dumps(row, allow_nan=False), flush=True)
            result.update(epi_mask.metadata(model))
            evaluate.write_json(out / 'result.json', result)
            print(json.dumps(result), flush=True)
            return result
        items = acquisition
    else:
        cases = build_cases(corpus, args.session, args.seed, args.cases_per_policy)
        acquisition = []
        for case in cases:
            attempted += 1
            if args.data == 'd2':
                variants = [encode_item(case, r, tok, g.BOS) for r in case['formats']]
                acquisition.append(dict(variants[0], variants=variants))
            else:
                item = encode_item(case, case['formats'][case['base_format']], tok, g.BOS)
                ctx = g.enc(tok, case['source_text']) + g.enc(tok, '\n\n')
                row = [g.BOS] + ctx + item['prompt_ids'] + item['answer_ids']
                lp = g.logprobs_at(model, [row], [1 + len(ctx) + len(item['prompt_ids'])], dev, args.nograd_tokens)[0]
                if teacher_agrees(lp, item['answer_ids'], item['action_positions']):
                    acquisition.append(dict(item, record=TopKRecord.from_logits(lp, args.topk).to('cpu'),
                                            weights=torch.ones(len(item['answer_ids']))))
                del lp
        kept = len(acquisition)
        items = acquisition
    if not acquisition:
        print(json.dumps(dict(phase='cache', attempted=attempted, kept=kept, keep_rate=0.)), flush=True)
        raise ValueError(f'No training items survived ({kept}/{attempted}); teacher keep rate cannot support training')
    missing = sorted(policy_ids - {i['policy_id'] for i in acquisition})
    # Evaluate exactly the policies that actually supplied training tokens.
    policy_ids = {i['policy_id'] for i in acquisition}
    for item in acquisition:
        for variant in item.get('variants', [item]):
            if len(variant['row']) > args.micro_tokens:
                raise ValueError('Training row exceeds the fixed 160-token micro budget')
    anchor_tokens = json.loads((Path(g.DATA) / 'anchor-wikitext.json').read_text())
    if len(anchor_tokens) < 150000 + 16 * 511:
        raise ValueError('Anchor and selection pools must be disjoint')
    pool = anchor_tokens[:150000]
    anchors = []
    for _ in range(64):
        offset = rng.randrange(len(pool) - 94)
        anchors.append([g.BOS] + pool[offset:offset + 95])
    gate, gate_info = configure_gate(model, fsa, args, anchors, acquisition)
    if args.arm == 'lorafold' or args.rehearsal > 0:
        anchor_entries = [dict(row=row, start=1) for row in anchors]
    else:
        records = g.topk_records(model, anchors, [1] * len(anchors), dev, args.topk, args.nograd_tokens)
        anchor_entries = [dict(row=row, start=1, record=rec) for row, rec in zip(anchors, records)]
    select_tensor = torch.tensor(evaluate.token_windows(anchor_tokens[-16 * 511:], 16), device=dev)
    start_nll = fsa.nll_windows(model, select_tensor, dev)
    superset, counts = c.router_superset(model, dev, items, args) if ternary else (None, None)
    allowed = {name: superset is None or ei in superset[str(li)] for name, (li, ei) in identity.items()}
    scales = [m.logb for m in mods if m.logb is not None]
    scale_ids = {id(t) for t in scales}
    lr = effective_lr(args)
    groups = [dict(params=[t for t in params if id(t) not in scale_ids], lr=lr)]
    if scales:
        groups.append(dict(params=scales, lr=.002 * args.lr_mult))
    opt = torch.optim.Adam(groups)
    den = fsa.collect_col_energy(model, mods, torch.tensor(anchors[:32], device=dev)) if budgeted else None
    if budgeted and (not math.isfinite(den) or den <= 0):
        raise ValueError('Invalid anchor energy denominator')
    # ladder recipe rungs share anchor forwards with GateHinge and need the reference harness's record/replay
    # LocalOutputPenalty (start/stop/end); the box harness's older class has no end()
    if ladder and args.local_lambda:
        local = reference_local_penalty()(model)
    else:
        local = fsa.LocalOutputPenalty(model) if args.local_lambda or not (ternary or ladder) else None
    lam = None
    trajectory = []
    diag_lines = []
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(epi_mask.metadata(model))
    config['gate_stats'] = gate_info
    config.update(items_per_step=8, anchor_per_step=2, acquisition_weight=.6, general_weight=.4,
                  lr=lr,
                  trained_policy_ids=sorted(policy_ids), policies_without_items=missing,
                  best_dev_tiebreak='fewer nnz, then earlier step', final_test_checkpoint='last step',
                  d0_scope='frozen accepted policy apply items', router_superset=superset, router_counts=counts)
    if args.arm == 'lorafold':
        config.update(served_format='ternary', rank=16, layers=[14, 15], projs=['up_proj', 'down_proj'],
                      optim='adam', local_lambda=10, fold_thr=.5, secondary_exploratory=True,
                      acquisition_weight=.8, general_weight=.2, rehearsal_frac=.2, provisional=True)
    if ladder:
        config.update(optim='adam', weight_decay=0.,
                      served_format={'ladder_cont': 'bf16', 'ladder_box': 'box', 'ladder_tern': 'ternary',
                                     'ladder_two': 'two ternary planes'}[args.arm])
    if args.arm == 'ladder_tern':
        config.update(local_lambda=args.local_lambda, budget=args.budget, rehearsal_frac=args.rehearsal,
                      support_score='netcost' if budgeted else None, cost_rel=1 if budgeted else None)
        if args.rehearsal:
            config.update(acquisition_weight=1 - args.rehearsal, general_weight=args.rehearsal)
    init_stats = None
    if args.init_from is not None:
        init_stats = initialize_ladder(model, modules, linears, args, acquisition,
                                      lambda x: g.logits_of(model, x), dev, epi_mask.metadata(model))
    if args.arm in ('ladder_tern', 'ladder_two'):
        config.update(beta_factors=[args.patch_beta] if args.arm == 'ladder_tern' else [.5, .125],
                      beta_definition='beta_i = beta_factors[i] * alpha_row',
                      init_mode=args.init_mode if args.init_from else None)
    limit = 10 if args.smoke else None
    dev_probes = (large_dev_concepts(corpus, policy_ids, args.dev_large, corpus['seed']) if args.dev_large
                  else filtered_probes(corpus, 'dev', policy_ids, limit))
    best_state, best_score = None, -math.inf
    student = lambda x: g.logits_of(model, x)
    def patch_nnz():
        if ladder:
            with torch.no_grad():
                return sum(int(torch.count_nonzero(lin.weight - lin.parametrizations.weight.original))
                           for _, _, _, lin in linears)
        return fsa.patch_stats(mods)[0] if ternary else evaluate.slot_stats(model)['nnz']
    torch.cuda.empty_cache()
    with (out / 'log.jsonl').open('w') as log:
        def emit(row):
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(row, allow_nan=False), flush=True)

        emit(dict(phase='cache', attempted=attempted, kept=kept, keep_rate=kept / attempted,
                  dropped_zero_gate=dropped, policies_without_items=missing))
        if init_stats is not None:
            emit(dict(phase='init', **init_stats, **plane_stats(modules)))

        def diagnose(step, event):
            if args.diag:
                row = slot_diagnostics(linears, step, event)
                diag_lines.append(row)
                emit(row)

        def evaluate_dev(step):
            nonlocal best_state, best_score
            tic = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            concept = evaluate.concept_acc(model, tok, dev, dev_probes)
            concept['se'] = math.sqrt(concept['acc'] * (1 - concept['acc']) / concept['n']) if concept['n'] else 0.
            nll = fsa.nll_windows(model, select_tensor, dev)
            row = dict(phase='eval', step=step, dev=concept,
                       select_dnll=sum(a - b for a, b in zip(nll, start_nll)) / len(start_nll),
                       nnz=patch_nnz(), seconds=time.monotonic() - tic,
                       peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            row.update(plane_stats(modules))
            trajectory.append(row)
            row['smoothed'] = smoothed_acc(trajectory)
            if args.save_best and row['smoothed'] is not None and row['smoothed'] > best_score:
                best_score = row['smoothed']
                best_state = dict(step=step, smoothed=best_score, state=trainable_state(modules), config=config)
                if args.arm == 'ladder_two':
                    best_state['committed_state'] = {name: {'code1': m.codes()[0].cpu().to(torch.int8),
                                                            'code2': m.codes()[1].cpu().to(torch.int8)}
                                                     for name, m in modules.items()}
            emit(row)
            diagnose(step, 'eval')

        try:
            evaluate_dev(0)
            for step in range(1, args.steps + 1):
                tic = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                opt.zero_grad(set_to_none=True)
                ab = [rng.choice(anchor_entries) for _ in range(2)]
                batch = sample_policies(acquisition, 8, rng, augment=args.data == 'd2')
                losses = training_backward(student, batch, ab, dev, g.VOCAB, args.arm, args.data, rehearsal=args.rehearsal)
                gate_hinge, gate_penalty, penalty_value = anchor_penalties_backward(
                    student, ab, dev, local, args.local_lambda if ladder else
                    (10 if args.arm == 'lorafold' else 100), gate, args.gate_lambda)
                if not ladder or budgeted:
                    c.protect_gradients(modules, frozen, allowed)
                for m in mods:
                    if getattr(m, 'score', None) is not None:
                        m.score.mul_(.9)
                        if m.M.grad is not None:
                            m.score.add_(m.M.grad, alpha=.1)
                opt.step()
                if step == 1:
                    diagnose(step, 'first_optimizer_step')
                if not ladder:
                    c.manual_decay(modules, frozen, allowed, lr, .3 if ternary else .1)
                births = evictions = 0
                if budgeted and refresh_due(step, args.refresh_until):
                    if lam is None:
                        lam = c.calibrate_lambda(mods, den, args.budget)
                        config['netcost_lambda'] = lam
                    if lam is not None:
                        births, evictions = c.refresh(modules, frozen, allowed, args.budget, lam, den, opt)
                        torch.cuda.empty_cache()
                emit(dict(phase='train', step=step, **dict(losses, loss=losses['loss'] + penalty_value + gate_penalty),
                          gate_hinge=gate_hinge, gate_penalty=gate_penalty,
                          local_penalty=penalty_value, nnz=patch_nnz(),
                          support_size=sum(int(m.support.sum()) for m in mods) if budgeted else None,
                          births=births, evictions=evictions, seconds=time.monotonic() - tic,
                          peak_gib=torch.cuda.max_memory_allocated() / 2**30))
                if step % args.eval_every == 0 or step == args.steps:
                    evaluate_dev(step)
            tic = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            test_probes = filtered_probes(corpus, 'test', policy_ids, limit)
            test = evaluate.concept_acc(model, tok, dev, test_probes)
            if getattr(args,'save_served',False):
                checkpoint = served_checkpoint(model,fsa,args,args.steps,test_probes,dev_probes,test,trajectory[-1]['dev'])
                # Dump item-for-item controls using the identical production scorer.
                checkpoint['ep1']['test_hits'] = [int(evaluate.concept_acc(model,tok,dev,[p])['acc']) for p in test_probes]
                checkpoint['ep1']['dev_hits'] = [int(evaluate.concept_acc(model,tok,dev,[p])['acc']) for p in dev_probes]
                torch.save(checkpoint,out / 'served_state.pt')
            emit(dict(phase='test', step=args.steps, test=test, nnz=trajectory[-1]['nnz'],
                      seconds=time.monotonic() - tic, peak_gib=torch.cuda.max_memory_allocated() / 2**30))
            diagnose(args.steps, 'test')
        finally:
            if gate is not None:
                for hook in gate.h:
                    hook.remove()
                gate.terms = []
            if local is not None:
                for hook in local.h:
                    hook.remove()
                local.num, local.den = [], []
    result = dict(config=config, keep_rate=kept / attempted, attempted=attempted, kept=kept,
                  gate_stats=gate_info,
                  trajectory=trajectory, final_test=test, final=trajectory[-1],
                  best_dev_step=best_dev_step(trajectory), netcost_lambda=lam)
    result.update(smoothed=smoothed_acc(trajectory), best_smoothed=best_smoothed_acc(trajectory))
    if best_state is not None:
        torch.save(best_state, out / 'best_state.pt')
        result['best_state_step'] = best_state['step']
    result['effective_lr'] = lr
    if args.arm in ('ladder_tern', 'ladder_two'):
        result.update(beta_factors=config['beta_factors'], init_mode=config['init_mode'],
                      init_error=init_stats, **plane_stats(modules))
    if args.arm == 'ladder_two':
        torch.save({name: dict(code1=m.codes()[0].detach().cpu().to(torch.int8),
                              code2=m.codes()[1].detach().cpu().to(torch.int8),
                              beta1=(.5 * m.alpha).cpu(), beta2=(.125 * m.alpha).cpu(),
                              mask=m.mask.cpu()) for name, m in modules.items()}, out / 'committed_state.pt')
    result.update(arm=args.arm, write_set=args.write_set, layers=args.layers, projs=args.projs,
                  lr=lr, diag_lines=diag_lines)
    if args.data == 'v2':
        result.update(g1=g1, g1_pass=True, gate_g2=gate_g2(trajectory, args.dev_large),
                      g2_pass=gate_g2(trajectory, args.dev_large) >= .70)
    result.update(epi_mask.metadata(model))
    evaluate.write_json(out / 'result.json', result)
    return result


def main():
    result = run(parse_args())
    return 0 if result.get('g1_pass', True) else 3


if __name__ == '__main__':
    raise SystemExit(main())
