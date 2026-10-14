"""SC4 J/F continue-training. Imported only by the opt-in e2e_fsa path.

One forward/backward per example keeps activation and gradient masks valid during
checkpoint replay. Bank files retain the ordinary inference entries and add a
reserve-sized joint_state for masters, support, score and Bop EMA.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

try:
    from . import c59 as C59
except ImportError:
    import c59 as C59


ARGUMENT_NAMES = ('joint_banks', 'joint_frozen_distractors', 'joint_schedule_seed',
                  'p_iso', 'max_distractors', 'distractor_mode', 'presence_dropout', 'presence_target', 'joint_negative_n')


def add_arguments(parser):
    parser.add_argument('--joint-banks', type=Path, nargs='+', default=[],
                        help='C52 continue-train disjoint saved FSA banks together')
    parser.add_argument('--joint-frozen-distractors', action='store_true',
                        help='arm F: detach non-relevant banks per example (same schedule as J)')
    parser.add_argument('--joint-schedule-seed', type=int, default=None,
                        help='arm-independent example/activation RNG; default --seed')
    parser.add_argument('--p-iso', type=float, default=.75)
    parser.add_argument('--max-distractors', type=int, default=None,
                        help='uniform mode: k in [1,M]; default M = all other banks')
    parser.add_argument('--distractor-mode', choices=('all', 'uniform'), default='all',
                        help='v3 default: all other banks on mixed examples; uniform: sample k in [1,M]')
    parser.add_argument('--presence-dropout', type=float, default=.2)
    parser.add_argument('--presence-target', choices=('base', 'unknown'), default='base')
    parser.add_argument('--joint-negative-n', type=int, default=1000,
                        help='unique never-written-entity evaluation queries; minimum 1000')


def validate(a):
    joint_flags = {'--joint-frozen-distractors', '--joint-schedule-seed', '--p-iso',
                   '--max-distractors', '--distractor-mode', '--presence-dropout', '--presence-target', '--joint-negative-n'}
    if not a.joint_banks:
        if joint_flags & a._explicit_flags:
            raise SystemExit('C52 training flags require --joint-banks')
        return
    if len(a.joint_banks) < 2 or len({p.resolve() for p in a.joint_banks}) != len(a.joint_banks):
        raise SystemExit('--joint-banks requires at least two distinct bank files')
    for name in ('p_iso', 'presence_dropout'):
        value = getattr(a, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise SystemExit('--'+name.replace('_', '-')+' must be finite and in [0,1]')
    if a.max_distractors is not None and not 1 <= a.max_distractors < len(a.joint_banks):
        raise SystemExit('--max-distractors must be in [1, number of other banks]')
    if a.distractor_mode == 'all' and a.max_distractors not in (None, len(a.joint_banks)-1):
        raise SystemExit('--max-distractors below the full set requires --distractor-mode uniform')
    if a.joint_negative_n < 1000:
        raise SystemExit('--joint-negative-n must be >=1000')
    if a.steps < 0 or a.fact_batch < 0 or a.micro < 1 or a.gen_bs < 1 or a.qa_n < 0:
        raise SystemExit('C52 requires nonnegative steps/fact-batch/qa-n and positive micro/gen-bs')
    if a.optim != 'bop' or a.method not in ('fsa_forced', 'fsa_full', 'fsa_free') or a.fmt != 'ternary':
        raise SystemExit('C52 requires ordinary ternary FSA with --optim bop')
    if not math.isfinite(a.bop_gamma) or not 0 < a.bop_gamma <= 1 or not math.isfinite(a.bop_tau) or a.bop_tau < 0:
        raise SystemExit('C52 Bop gamma must be in (0,1] and tau finite and nonnegative')
    if (a.bop_gamma_late is None) != (a.bop_gamma_switch is None) or (
            a.bop_gamma_late is not None and (not math.isfinite(a.bop_gamma_late) or
            not 0 < a.bop_gamma_late <= 1 or a.bop_gamma_switch < 0)):
        raise SystemExit('C52 late gamma needs a nonnegative switch and gamma in (0,1]')
    if a.aug_all and not a.aug:
        raise SystemExit('C52 --aug-all requires --aug')
    if not math.isfinite(a.bop_tau_abs) or a.bop_tau_abs < 0 or not math.isfinite(a.local_lambda) or a.local_lambda < 0:
        raise SystemExit('C52 absolute tau and local-lambda must be finite and nonnegative')
    if a.local_lambda and not a.anchor:
        raise SystemExit('C52 --local-lambda requires --anchor')
    if a.anchor_bs < 1 or a.anchor_len < 2 or a.refresh_every < 1 or a.budget_freeze < -1:
        raise SystemExit('C52 anchor sizes and refresh-every must be positive')
    # This is a separate experiment, so reject unsupported explicitly requested
    # objectives rather than silently running a different treatment.
    supported = joint_flags | {'--joint-banks', '--method', '--fmt', '--optim', '--export', '--export-step',
        '--eda-backend', '--seed', '--people', '--synthetic-people', '--facts-per-person', '--shared-attrs',
        '--values', '--relation-world', '--layers', '--steps', '--fact-batch', '--micro', '--answer-only',
        '--aug', '--aug-all', '--bop-gamma', '--bop-gamma-late',
        '--bop-gamma-switch', '--bop-tau', '--bop-tau-abs', '--tau-anneal', '--keep-ema', '--death-tau-mult',
        '--death-grace', '--budget', '--budget-freeze', '--refresh-every', '--keep-live', '--sparse-state', '--no-churn',
        '--local-lambda', '--anchor', '--anchor-len', '--anchor-bs', '--expert-checkpoint', '--fast-moe',
        '--eval-set', '--eval-windows', '--eval-every', '--val-windows', '--extra-eval-sets', '--in-topic-panel',
        '--fact-class-metrics', '--decisive-eval', '--qa-probe', '--qa-n', '--gen-max-new', '--gen-bs',
        '--gen-stop-k', '--bootstrap', '--no-pareto', '--out', '--patch-dir'}
    unsupported = a._explicit_flags - supported
    if unsupported:
        raise SystemExit('unsupported flags in C52 joint mode: '+', '.join(sorted(unsupported)))
    if not math.isfinite(a.tau_anneal) or a.tau_anneal <= 0 or not math.isfinite(a.death_tau_mult) or a.death_tau_mult <= 0:
        raise SystemExit('C52 tau multipliers must be finite and positive')


@dataclass
class Bank:
    state: dict
    modules: dict
    budget: int
    tau: float | None = None
    source_state: str = 'inference codes (EMA unavailable)'
    calibration: dict | None = None
    initial_nnz: int = 0


def _indices(entry, size):
    idx = entry['idx'].long()
    if idx.ndim != 1 or idx.numel() != entry['val'].numel() or (
            idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= size or
                            not bool((idx[1:] > idx[:-1]).all()))):
        raise ValueError('bank indices must be sorted, unique, in bounds and aligned with codes')
    if not bool(((entry['val'] == -1) | (entry['val'] == 1)).all()):
        raise ValueError('bank live codes must be ternary +/-1')
    return idx


def _mask(E, W, meta, name, coordinates, generator):
    if meta.get('method') not in ('fsa_forced', 'fsa_full', 'fsa_free'):
        raise ValueError('bank metadata must identify its FSA method')
    if meta.get('address_units', 0) or meta.get('units_per_fact', 0):
        raise ValueError('C52 does not support address/unit overlays')
    method = meta['method']
    mask = E.pair8_forced_mask(W) if method == 'fsa_forced' else W == 0
    if method == 'fsa_free':
        selected = (W.reshape(-1, 4, 2) != 0).any(-1, keepdim=True).expand(-1, -1, 2).reshape_as(W)
        mask &= selected
    li, ei, proj = coordinates[name]
    mask = E.partition_mask(mask, li, ei, proj, meta.get('slot_partition', 0),
                            meta.get('slot_part'), meta.get('partition_unit', False))
    fraction = meta.get('slot_frac', 1.)
    if not 0 < fraction <= 1 or (fraction < 1 and not meta.get('sparse_state', False)):
        raise ValueError('invalid saved slot sampling recipe')
    if fraction < 1:
        mask &= (torch.rand(W.shape, generator=generator) < fraction).to(W.device)
    return mask


def attach_banks(E, model, states):
    """Validate complete writable masks before registering any parametrizations."""
    class JointParam(E.SparseFSAParam):
        enabled = True
        frozen = False

        def forward(self, W):
            if not self.enabled or not E.FSAParam.enabled:
                return W
            delta = self.delta()
            return W + (delta.detach() if self.frozen else delta).to(W.dtype)

    linears = dict(model.named_modules())
    coords = {}
    for li, (prefix, moe) in enumerate(E.moe_layers(model)):
        for ei, expert in enumerate(moe.experts):
            if expert is not None:
                for proj in ('up_proj', 'down_proj'):
                    coords[f'{prefix}.experts.{ei}.{proj}'] = (li, ei, proj)
    union, banks, registrations = {}, [], []
    for state in states:
        if state.get('format') != 'e2e-fsa-patch-v1' or not state.get('entries'):
            raise ValueError('C52 needs nonempty e2e-fsa-patch-v1 banks')
        meta = state['meta']
        if meta.get('occupied_flips', False) or meta.get('base_altering', False):
            raise ValueError('joint banks cannot alter occupied base slots')
        if hasattr(model, '_fsa_export'):
            E.validate_patch_export(meta, model._fsa_export, model._fsa_export_step)
        E.eda.require_backend(state, getattr(model, '_fsa_eda_backend', E.eda.recorded(meta)), 'joint bank')
        bank = Bank(state, {}, int(meta.get('budget', 0)))
        if bank.budget < 0:
            raise ValueError('negative bank budget')
        saved = state.get('joint_state')
        if saved is not None and saved.get('version') != 1:
            raise ValueError('unsupported joint training state version')
        if saved is not None:
            if saved['budget'] != bank.budget or set(saved['modules']) != set(state['entries']):
                raise ValueError('joint state budget/modules differ from inference patch')
            bank.tau, bank.source_state = saved['tau'], 'joint_state'
        generator = torch.Generator().manual_seed(meta['seed']+41)
        def ordering(name):
            if name not in coords:
                raise ValueError('joint bank target is not a routed linear: '+name)
            li,ei,proj = coords[name]
            return (meta['layers'].index(li), ei, meta.get('projs', ['up_proj','down_proj']).index(proj))
        for name in sorted(state['entries'], key=ordering):
            entry = state['entries'][name]
            lin = linears.get(name)
            if name not in coords or not isinstance(lin, torch.nn.Linear) or E.P.is_parametrized(lin, 'weight'):
                raise ValueError('joint bank must target an unmodified routed linear: '+name)
            W = lin.weight.detach()
            if (entry['kind'] != 'fsa' or entry.get('fmt') != 'ternary' or entry['scale'] != 'row'
                    or tuple(entry['shape']) != tuple(W.shape) or entry.get('flip_idx', torch.empty(0)).numel()
                    or any(key.startswith('unit_reset') for key in entry)):
                raise ValueError('joint banks require ordinary row-scaled ternary FSA entries')
            mask = _mask(E, W, meta, name, coords, generator)
            prior = union.get(name)
            if prior is not None and bool((mask & prior).any()):
                raise ValueError('joint bank writable masks overlap: '+name)
            union[name] = mask.clone() if prior is None else prior | mask
            pm = JointParam(W, mask, budgeted=bool(bank.budget))
            alpha = entry['alpha'].float().to(W.device).reshape_as(pm.alpha)
            expected = pm.alpha * meta.get('patch_scale_mult', 1.)
            recipe = meta.get('training_recipe', {})
            override = recipe.get('patch_scale_mult_up' if name.endswith('up_proj') else 'patch_scale_mult_down')
            if override is not None:
                expected = pm.alpha * override
            if not torch.isfinite(alpha).all() or not torch.allclose(alpha, expected):
                raise ValueError('saved bank alpha differs from its base/recipe: '+name)
            pm.alpha.copy_(alpha)
            pm.scale.copy_(alpha.squeeze(1)[pm.rows()])
            idx = _indices(entry, W.numel()).to(W.device)
            positions = torch.searchsorted(pm.idx.long(), idx)
            if idx.numel() and (pm.idx.numel() == 0 or int(positions.max()) >= pm.idx.numel() or
                               not torch.equal(pm.idx.long()[positions], idx)):
                raise ValueError('saved bank writes outside its writable mask: '+name)
            with torch.no_grad():
                pm.M[positions] = entry['val'].float().to(W.device)
                if bank.budget:
                    pm.support[positions] = True
                pm.bop_ema = torch.zeros_like(pm.M)
                if saved is not None:
                    row = saved['modules'][name]
                    if not torch.equal(row['reserve_idx'].long(), pm.idx.long().cpu()):
                        raise ValueError('joint saved writable mask differs: '+name)
                    for key in ('M', 'support', 'bop_ema', 'score'):
                        value = row[key]
                        if value is not None:
                            dest = getattr(pm, key)
                            if dest is None or dest.shape != value.shape:
                                raise ValueError('joint training tensor shape differs: '+name+'/'+key)
                            if key != 'support' and not torch.isfinite(value).all():
                                raise ValueError('nonfinite joint training tensor: '+name+'/'+key)
                            if key != 'support' or bank.budget:
                                dest.copy_(value.to(dest.device))
                    if not torch.equal(pm.codes().cpu().flatten().nonzero().flatten(), idx.cpu()) or not torch.equal(
                            pm.codes().cpu().flatten()[idx.cpu()], entry['val'].float()):
                        raise ValueError('joint masters/support disagree with inference codes: '+name)
            bank.modules[name] = pm
            registrations.append((lin, pm))
        if bank.budget and sum(int(m.support.sum()) for m in bank.modules.values()) > bank.budget:
            raise ValueError('bank support exceeds saved budget')
        continuation = state.get('continue_state')
        if continuation is not None:
            C59.restore(bank.modules, continuation)
            bank.source_state = 'continue_state'
        elif saved is not None and saved.get('calibration'):
            C59.restore(bank.modules, saved)
        banks.append(bank)
    for lin, pm in registrations:
        E.P.register_parametrization(lin, 'weight', pm)
    return banks


@contextmanager
def activation(banks, active, relevant=None, frozen_distractors=False):
    active = set(active)
    previous = [(m, m.enabled, m.frozen) for b in banks for m in b.modules.values()]
    try:
        for i, bank in enumerate(banks):
            for m in bank.modules.values():
                m.enabled = i in active
                m.frozen = frozen_distractors and i != relevant
        yield
    finally:
        for m, enabled, frozen in previous:
            m.enabled, m.frozen = enabled, frozen


class Schedule:
    """Arm-independent fact, wording and activation draws in separate RNG streams."""
    def __init__(self, seed, banks, p_iso=.75, max_distractors=None, dropout=.2, distractor_mode='all'):
        self.seed, self.banks, self.p_iso, self.dropout = seed, banks, p_iso, dropout
        self.maximum = banks-1 if max_distractors is None else max_distractors
        if distractor_mode not in ('all', 'uniform') or (distractor_mode == 'all' and self.maximum != banks-1):
            raise ValueError('all distractor mode requires the full set of other banks')
        self.distractor_mode = distractor_mode
        self.facts = random.Random(seed+52001)
        self.visibility = random.Random(seed+52002)
        self.digest = hashlib.sha256()
        self.counts = Counter()

    def draw(self, relevant):
        others = [i for i in range(self.banks) if i != relevant]
        isolated = self.visibility.random() < self.p_iso
        selected = ([] if isolated else others if self.distractor_mode == 'all' else
                    self.visibility.sample(others, self.visibility.randint(1, self.maximum)))
        dropped = self.visibility.random() < self.dropout
        active = tuple(sorted(selected + ([] if dropped else [relevant])))
        self.counts['examples'] += 1
        self.counts['isolated'] += isolated
        self.counts['presence_dropped'] += dropped
        self.counts['distractors_'+str(len(selected))] += 1
        return dict(relevant=relevant, active=active, dropped=dropped, isolated=isolated)

    def record(self, row):
        self.digest.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
        self.digest.update(b'\n')

    def report(self):
        return dict(seed=self.seed, sha256=self.digest.hexdigest(), p_iso=self.p_iso,
                    distractor_mode=self.distractor_mode, max_distractors=self.maximum,
                    presence_dropout=self.dropout, counts=dict(self.counts))


def world(E, states):
    meta = states[0]['meta']
    fields = ('seed', 'people', 'facts_per_person', 'shared_attrs', 'values', 'relation_world', 'synthetic_people')
    defaults = dict(relation_world=False, synthetic_people=0)
    signature = {key: meta.get(key, defaults.get(key)) for key in fields}
    for state in states[1:]:
        if {key: state['meta'].get(key, defaults.get(key)) for key in fields} != signature:
            raise ValueError('joint banks must belong to the same saved fact world')
    facts = ([E.Fact(**row) for row in meta['c37_facts']] if meta.get('c37_facts') else
             E.make_world(signature['seed'], signature['people'], signature['facts_per_person'],
                          signature['shared_attrs'], signature['values'], signature['relation_world'],
                          signature['synthetic_people']))
    owners, scopes = {}, []
    for i, state in enumerate(states):
        sm = state['meta']
        if sm.get('c37_facts') != meta.get('c37_facts'):
            raise ValueError('joint custom fact worlds differ')
        ids = E.person_indices(facts, sm.get('train_person'), sm.get('train_attr_group'), sm.get('attr_groups', 1))
        for fi in ids:
            if fi in owners:
                raise ValueError('joint fact ownership overlaps')
            owners[fi] = i
        scopes.append(ids)
    return facts, owners, scopes


def batch(E, tok, facts, owners, scopes, schedule, a, bos):
    """Every update includes each bank and matched same-attribute conflicts."""
    rng = schedule.facts
    ids = set()
    for scope in scopes:
        ids.add(rng.choice(scope))
    # Match an attribute shared across banks to guarantee conflicting questions
    # whenever the saved world contains such a group.
    by_attr = defaultdict(list)
    for fi in owners:
        by_attr[facts[fi].attr].append(fi)
    groups = [fis for fis in by_attr.values() if len({owners[fi] for fi in fis}) > 1 and
              len({facts[fi].val for fi in fis}) > 1]
    if groups:
        ids.update(rng.choice(groups))
    target = len(owners) if not a.fact_batch else min(len(owners), max(a.fact_batch, len(ids)))
    available = sorted(set(owners)-ids)
    ids.update(rng.sample(available, min(target-len(ids), len(available))))
    templates = E.TRAIN_AUG_T if a.aug else E.TRAIN_T
    rows = []
    for fi in sorted(ids):
        chosen = (templates if not a.aug or a.aug_all else rng.sample(templates, 3))
        for template in chosen:
            seq, start = E.encode_fact(tok, template, facts[fi], bos)
            row = dict(fi=fi, seq=seq, start=start, **schedule.draw(owners[fi]))
            schedule.record(row)
            rows.append(row)
    return rows


def example_loss(E, model, tok, banks, row, dev, target='base'):
    """Answer-token mean CE or KL(base || student), including the stop token.

    Base KL is evaluated on the own-query answer prefix, so no generation RNG
    or arm-dependent continuation can alter the training exposure schedule.
    """
    seq, start = row['seq'], row['start']
    if row['dropped'] and target == 'unknown':
        seq = seq[:start]+tok.encode(' Unknown.', add_special_tokens=False).ids
    x = torch.tensor([seq], device=dev)
    teacher = None
    if row['dropped'] and target == 'base':
        with activation(banks, ()), torch.no_grad():
            teacher = E.logits_of(model, x)[:, start-1:-1, :E.VOCAB].detach()
    student = E.logits_of(model, x)[:, start-1:-1, :E.VOCAB].float()
    if teacher is not None:
        return E.C50.kl_to_base(student, teacher) / student.shape[1]
    return F.cross_entropy(student.reshape(-1, student.shape[-1]), x[:, start:].reshape(-1))


def restore_legacy_checkpoint(E, bank, path):
    """Recover C44 training tensors when a matching final-step resume.pt exists.

    Older inference-only banks contain codes, no continuous masters or EMA;
    these use codes as Bop masters, live support, and an explicit zero EMA.
    """
    if bank.source_state in ('joint_state', 'continue_state'):
        return
    saved = C59.legacy_state(bank.state, path, bank.modules)
    if saved is None:
        return
    C59.restore(bank.modules, saved)
    bank.tau = saved['calibration']['tau']
    bank.state['meta']['bop_calibration'] = saved['calibration']
    bank.source_state = str(path.with_name('resume.pt'))


def export_bank(E, bank, schedule, arm, steps):
    entries, tensors = {}, {}
    for name, pm in bank.modules.items():
        codes = pm._quant(pm.M.detach()) * pm.support
        live = codes.nonzero().flatten()
        entries[name] = dict(kind='fsa', fmt='ternary', shape=list(pm.wshape),
            idx=pm.idx[live].cpu(), val=codes[live].to(torch.int8).cpu(),
            alpha=pm.alpha.detach().squeeze(1).cpu(), fp4_step=.25, scale='row')
        tensors[name] = dict(reserve_idx=pm.idx.cpu(), **{key: None if getattr(pm, key) is None else
            getattr(pm, key).detach().cpu().clone() for key in ('M', 'support', 'bop_ema', 'score')})
    meta = dict(bank.state['meta'], step=bank.state['meta'].get('step', 0)+steps,
                value_steps=bank.state['meta'].get('value_steps',bank.state['meta'].get('step',0))+steps,
                c52=dict(arm=arm, continue_steps=steps, schedule=schedule.report(),
                         source_state=bank.source_state))
    joint = dict(version=1, budget=bank.budget, tau=bank.tau, modules=tensors)
    if bank.calibration is not None:
        joint.update(C59.snapshot(bank.modules, bank.calibration))
        meta['bop_calibration'] = dict(bank.calibration)
    return dict(format='e2e-fsa-patch-v1', meta=meta, entries=entries, joint_state=joint)


@torch.no_grad()
def revoke_check(E, model, banks, prompts):
    """Compare each revocation against independent remaining-bank reconstruction.

    Rebuild ordinary ResidentParams from the remaining saved entries on pristine
    weights, preserving addition/cast order. This also tests byte-exact logits.
    """
    linears = E.patched_linears(model)
    originals = {name: lin.parametrizations.weight.original for name, lin in linears}
    all_logits = {}
    for removed in range(len(banks)):
        active = tuple(i for i in range(len(banks)) if i != removed)
        with activation(banks, active):
            all_logits[removed] = [E.logits_of(model, x).clone() for x in prompts]
    with activation(banks, ()):
        base = [E.logits_of(model, x).clone() for x in prompts]
    states = []
    for bank in banks:
        entries = {}
        for name,pm in bank.modules.items():
            codes = pm._quant(pm.M.detach()) * pm.support
            live = codes.nonzero().flatten()
            entries[name] = dict(kind='fsa',fmt='ternary',shape=list(pm.wshape),idx=pm.idx[live].cpu(),
                                val=codes[live].to(torch.int8).cpu(),alpha=pm.alpha.squeeze(1).cpu(),scale='row')
        states.append(dict(format='e2e-fsa-patch-v1',meta=bank.state['meta'],entries=entries))
    # Removing parametrizations is reversible; reinstall the original bank
    # modules in finally, including their optimizer state and flags.
    for _, lin in linears:
        E.P.remove_parametrizations(lin, 'weight', leave_parametrized=False)
    result = []
    try:
        no_patch = [E.logits_of(model, x).clone() for x in prompts]
        for removed in range(len(banks)):
            try:
                for i, state in enumerate(states):
                    if i != removed:
                        E.attach_residents(model, [state], collect_slots=False)
                logits_equal = all(torch.equal(expected, E.logits_of(model, x))
                                   for expected, x in zip(all_logits[removed], prompts))
                weights_equal = True
                for name, lin in linears:
                    expected = originals[name]
                    for i, bank in enumerate(banks):
                        if i != removed and name in bank.modules:
                            expected = expected + bank.modules[name].delta().to(expected.dtype)
                    weights_equal &= torch.equal(expected, lin.weight)
                result.append(dict(bank=removed, logits_bit_exact=logits_equal, weights_bit_exact=weights_equal))
            finally:
                E.detach_all(model)
    finally:
        for bank in banks:
            for name, pm in bank.modules.items():
                E.P.register_parametrization(dict(linears)[name], 'weight', pm)
    return dict(all_revoked_logits_bit_exact=all(torch.equal(a,b) for a,b in zip(base,no_patch)), banks=result)


def absent_items(E, D, tok, facts, bos, n):
    """C34 negative templates/scoring, extended to >=1000 distinct virgin names."""
    forbidden = {f.name.casefold() for f in facts} | {f.val.casefold() for f in facts}
    items = []
    for i in range(n):
        f = facts[i % len(facts)]
        name = 'NeverTaughtPersonC52'+str(i)
        while name.casefold() in forbidden:
            name += 'X'
        prompt = D.NEGATIVE_T[(i//len(facts)) % len(D.NEGATIVE_T)].format(name=name, attr=f.attr)
        items.append(dict(prompt=prompt, query_name=name, p=E.encode(tok,prompt,bos)))
    return items


@torch.no_grad()
def evaluate(E, model, tok, banks, facts, scopes, dev, bos, a, panels, panel_base, in_topic=None):
    try:
        from . import decisive_eval as D
        from . import in_topic as I
    except ImportError:
        import decisive_eval as D
        import in_topic as I
    all_active = tuple(range(len(banks)))
    conflicts = E.conflict_map(facts)
    items = E.build_items(tok, facts, E.TEST_T, bos, idxs=sorted(fi for scope in scopes for fi in scope))
    def gen(rows):
        return E.gen_eval(model, tok, rows, dev, a.gen_max_new, a.gen_stop_k, a.gen_bs, conflicts,
                          bootstrap=a.bootstrap, bootstrap_seed=a.seed, fact_class_metrics=True)
    solo, wrong, written = {}, {}, []
    for i, scope in enumerate(scopes):
        own = [it for it in items if it['fi'] in scope]
        with activation(banks, (i,)):
            solo[str(i)] = gen(own)
            group = dict(bank=i, items=own, conflicts=conflicts, facts=len(scope),
                         baseline=I.fact_flags(model,tok,dict(items=own, conflicts=conflicts),dev,a))
            written.append(group)
        wrong[str(i)] = {}
        for j in all_active:
            if j != i:
                with activation(banks, (j,)):
                    wrong[str(i)][str(j)] = gen(own)
    with activation(banks, all_active):
        composed = gen(items)
        composed_by_bank = {str(i): gen([it for it in items if it['fi'] in scopes[i]]) for i in all_active}
        tasks = D.task_items(tok, facts, bos, a.qa_n)
        tasks = {key: [it for it in rows if set(it['fis']) <= set(fi for scope in scopes for fi in scope)]
                 for key, rows in tasks.items()}
        composition = {}
        for kind, rows in E.progress(tasks.items(), 'c52:composition'):
            E.C50.annotate(rows,facts)
            gens = E.greedy_generate(model, [it['p'] for it in rows], dev, 64, a.gen_bs) if rows else []
            composition[kind] = D.summary(tok, rows, gens, a.gen_stop_k, facts=facts)
        damage = {name: E.paired_nll(model, wins, panel_base[name], dev) for name,wins in panels.items()}
        split_damage = I.evaluate(model,tok,in_topic['panel'],in_topic['base'],written,dev,bos,a) if in_topic else None
    independence = {}
    for i in all_active:
        remaining = [j for j in all_active if j != i]
        with activation(banks, remaining):
            scored = {str(j): gen([it for it in items if it['fi'] in scopes[j]]) for j in remaining}
        # Paired recall change uses all-resident GEN on the same bank questions.
        deltas = {j: scored[j]['recall']-composed_by_bank[j]['recall'] for j in scored}
        independence[str(i)] = dict(remaining=scored, recall_delta=deltas,
                                    passed=all(delta >= -.01-1e-12 for delta in deltas.values()))
    negative_rows = absent_items(E,D,tok,facts,bos,a.joint_negative_n)
    negatives = {}
    for mode, active in (('base', ()), ('composed', all_active)):
        with activation(banks, active):
            gens = E.greedy_generate(model,[it['p'] for it in negative_rows],dev,64,a.gen_bs)
        rows = []
        for item, ids in zip(negative_rows,gens):
            text = D.decode(tok,ids)
            rows.append(dict(prompt=item['prompt'],text=text,
                false_recall=any(D.contains(text,v) for v in D.negative_values(facts,item['query_name'])),
                asserted_answer=D.asserted_answer(text)))
        negatives[mode] = dict(n=len(rows), unique_entities=len({it['query_name'] for it in negative_rows}),
            false_answer_rate=sum(r['asserted_answer'] for r in rows)/len(rows),
            stored_value_false_recall=sum(r['false_recall'] for r in rows)/len(rows), per_item=rows)
    return dict(individual_gen=solo, composed_gen=composed, composed_gen_by_bank=composed_by_bank, composition=composition,
                wrong_person_activation=wrong, absent_entities=negatives, off_topic=damage,
                in_topic=split_damage, revoke_independence=independence)


def run(E, a):
    E.eda.announce(a)
    states = [torch.load(path,map_location='cpu',weights_only=False) for path in a.joint_banks]
    facts, owners, scopes = world(E,states)
    # Saved metadata is authoritative; command-line world/layers must agree when
    # explicitly supplied, rather than accidentally regenerate different facts.
    for flag, key in (('--people','people'), ('--facts-per-person','facts_per_person'),
                      ('--shared-attrs','shared_attrs'), ('--values','values'),
                      ('--relation-world','relation_world'), ('--synthetic-people','synthetic_people')):
        if flag in a._explicit_flags and getattr(a,key) != states[0]['meta'].get(key,False if key=='relation_world' else 0):
            raise ValueError('joint command world differs from saved banks: '+flag)
    layers = sorted({li for state in states for li in state['meta']['layers']})
    if '--layers' in a._explicit_flags and sorted(map(int,a.layers.split(','))) != layers:
        raise ValueError('--layers differs from saved joint banks')
    for state in states:
        E.validate_patch_export(state['meta'],a.export,a.export_step)
        E.eda.require_backend(state,E.eda.selected(a),'joint bank')
        if '--budget' in a._explicit_flags and a.budget != state['meta'].get('budget',0):
            raise ValueError('--budget differs from saved per-bank budget')
    sys.path.insert(0,'/workspace/malleable/mlf')
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = Tokenizer.from_file(str(E.DATA/'tokenizer.json'))
    model, _, dev = E.eda.build_model(build_model,a,a.export,device='cuda:0',model_config=None,
                                    step=a.export_step,expert_format='bf16')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model._fsa_export, model._fsa_export_step = a.export, a.export_step
    model._fsa_eda_backend = E.eda.selected(a)
    if a.fast_moe != 'off':
        try:
            from .fast_moe import install
        except ImportError:
            from fast_moe import install
        install(model,mode=a.fast_moe)
    panels = {}
    held = json.loads((E.DATA/'heldout-tokens.json').read_text())
    panels['heldout'] = torch.tensor([held[511*i:511*i+512] for i in range(5)],device=dev)
    if a.eval_set:
        tokens = json.loads(Path(a.eval_set).read_text())
        if len(tokens) < a.eval_windows*511 or a.eval_windows < 2:
            raise ValueError('joint eval-set requires enough tokens and >=2 windows')
        panels['eval'] = torch.tensor([[E.BOS]+tokens[i*511:(i+1)*511] for i in range(a.eval_windows)],device=dev)
    if a.extra_eval_sets:
        panels.update({name: wins for name,(_,wins) in E.extra_eval_panels(a.extra_eval_sets,E.BOS,dev).items()})
    with torch.no_grad():
        base = {name:E.nll_windows(model,wins,dev) for name,wins in panels.items()}
        pristine = [E.logits_of(model,wins[:1,:64]).clone() for wins in list(panels.values())[:3]]
    in_topic = None
    if a.in_topic_panel:
        try:
            from . import in_topic as I
        except ImportError:
            import in_topic as I
        panel = I.load_panel(a.in_topic_panel,a,tok,dev)
        in_topic = dict(panel=panel,base=I.base_scores(model,tok,panel,dev,E.BOS))
    banks = attach_banks(E,model,states)
    for bank,path in zip(banks,a.joint_banks):
        restore_legacy_checkpoint(E,bank,path)
        bank.calibration = C59.calibration(bank.state, path, bank.tau)
        bank.tau = bank.calibration['tau']
        C59.prepare(bank.modules, bank.calibration, exact_support=bank.source_state != 'inference codes (EMA unavailable)',
                    E=E, model=model, meta=bank.state['meta'], path=path)
        bank.initial_nnz = C59.nnz(bank.modules.values())
        print('C59-CALIBRATION '+json.dumps(dict(path=str(path), nnz=bank.initial_nnz, **bank.calibration)), flush=True)
    a.budget_freeze = -1
    if a.expert_checkpoint:
        E.checkpoint_experts(model,layers)
    anchor = None
    if a.local_lambda:
        tokens = json.loads(Path(a.anchor).read_text())
        length = a.anchor_len-1
        if len(tokens) < length:
            raise ValueError('joint anchor is shorter than one window')
        anchor = torch.tensor([[E.BOS]+tokens[i:i+length] for i in range(0,len(tokens)-length+1,length)],device=dev)
        E.checked_overlap(tokens,{name:wins.cpu().flatten().tolist() for name,wins in panels.items()},'joint anchor')
    arm = 'F' if a.joint_frozen_distractors else 'J'
    seed = a.seed if a.joint_schedule_seed is None else a.joint_schedule_seed
    schedule = Schedule(seed,len(banks),a.p_iso,a.max_distractors,a.presence_dropout,a.distractor_mode)
    anchor_rng = random.Random(seed+52003)
    local = E.LocalOutputPenalty(model,from_output=True) if anchor is not None else None
    trajectory, started = [], time.monotonic()
    try:
        for step in E.progress(range(a.steps),'c52:train'):
            for bank in banks:
                for m in bank.modules.values():
                    m.M.grad = None
            rows = batch(E,tok,facts,owners,scopes,schedule,a,E.BOS)
            loss_sum, no_grad = 0., 0
            for row in rows:
                with activation(banks,row['active'],row['relevant'],a.joint_frozen_distractors):
                    loss = example_loss(E,model,tok,banks,row,dev,a.presence_target)
                    loss_sum += float(loss.detach())/len(rows)
                    if loss.requires_grad:
                        (loss/len(rows)).backward()
                    else:
                        no_grad += 1
            # Update answer scores without recalibrating the saved threshold.
            for bank in banks:
                mods = list(bank.modules.values())
                if bank.budget:
                    with torch.no_grad():
                        for m in mods:
                            m.score.mul_(.9)
                            if m.M.grad is not None:
                                m.score.add_(m.M.grad if bank.calibration['support_score'] == 'netcost' else m.M.grad.abs(),alpha=.1)
            # Local preservation is applied separately to each bank alone. This
            # retains F's rule: no gradient enters a distractor on another bank's
            # question, including via preservation/checkpoint replay.
            if local is not None:
                for i,bank in enumerate(banks):
                    ab = anchor[[anchor_rng.randrange(len(anchor)) for _ in range(a.anchor_bs)]]
                    with activation(banks,(i,),i,a.joint_frozen_distractors):
                        try:
                            local.start()
                            E.logits_of(model,ab)
                            penalty = local.stop()
                            if penalty is not None and penalty.requires_grad:
                                (a.local_lambda*penalty/len(banks)).backward()
                                loss_sum += float(penalty.detach())*a.local_lambda/len(banks)
                        finally:
                            local.on = False
                            local.end()
            updates = []
            for i,bank in enumerate(banks):
                mods = list(bank.modules.values())
                cal = bank.calibration
                nb,nd = E.bop_step(mods,cal['gamma'],bank.tau,True,
                    keep_ema=cal['keep_ema'],death_mult=cal['death_tau_mult'],grace=cal['death_grace'])
                E.advance_step_age(mods)
                C59.check_growth(bank.initial_nnz,C59.nnz(mods),step+1,f'bank {i}')
                updates.append(dict(bank=i,births=nb,deaths=nd,tau=bank.tau,
                    nnz=sum(int((m.M.detach().abs()>.5).logical_and(m.support).sum()) for m in mods)))
            trajectory.append(dict(step=step+1,loss=loss_sum,examples=len(rows),no_gradient_examples=no_grad,banks=updates))
            if step%25 == 0 or step+1 == a.steps:
                print('C52-STEP '+json.dumps(trajectory[-1]),flush=True)
    finally:
        if local is not None:
            for handle in local.h:
                handle.remove()
    elapsed = time.monotonic()-started
    patch_dir = Path(a.patch_dir) if a.patch_dir else Path(a.out).with_name(Path(a.out).stem+'_patches')
    patch_dir.mkdir(parents=True,exist_ok=True)
    paths = []
    for i,bank in enumerate(banks):
        path = patch_dir/f'bank{i}'/'final.pt'
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(export_bank(E,bank,schedule,arm,a.steps),path)
        paths.append(str(path))
    prompts = [wins[:1,:64] for wins in list(panels.values())[:3]]
    revoked = revoke_check(E,model,banks,prompts)
    with activation(banks,()),torch.no_grad():
        revoked['pristine_logits_bit_exact'] = all(torch.equal(before,E.logits_of(model,x)) for before,x in zip(pristine,prompts))
    result = dict(arm=arm,schedule_seed=seed,schedule=schedule.report(),steps=a.steps,seed=a.seed,
        eda_backend=E.eda.selected(a),patch_files=paths,trajectory=trajectory,training_seconds=elapsed,
        bank_inputs=[dict(path=str(p.resolve()),sha256=E.file_sha(p)) for p in a.joint_banks],
        presence_target=a.presence_target,microbatch_policy='one forward/backward per example; --micro is an upper bound',
        source_states=[b.source_state for b in banks],budgets=[b.budget for b in banks],revoked=revoked,
        bop_calibrations=[b.calibration for b in banks],budget_freeze=-1,
        initial_nnz=[b.initial_nnz for b in banks],
        **E.C50.metric_metadata(facts))
    result['evaluation'] = evaluate(E,model,tok,banks,facts,scopes,dev,E.BOS,a,panels,base,in_topic)
    result['by_fact_class'] = E.C50.collect_splits(result['evaluation'])
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    printable = {k:v for k,v in result.items() if k not in ('trajectory','fact_classes')}
    print('RESULT '+json.dumps(compact_report(printable)),flush=True)
    return result


def compact_report(value):
    if isinstance(value,dict):
        return {key:compact_report(item) for key,item in value.items() if key not in ('per_item','items')}
    if isinstance(value,list):
        return [compact_report(item) for item in value]
    return value
