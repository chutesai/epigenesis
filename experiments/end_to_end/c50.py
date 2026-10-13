"""Opt-in token-dosed preservation KL and deterministic fact-class reporting."""
from contextlib import contextmanager
import math
import re
import time

import torch
import torch.nn.functional as F


CLASS_RULE = ('exact iff value contains a digit, or is one whitespace-free identifier with '
              'an underscore/slash/colon, >=2 uppercase letters only, or an internal uppercase '
              'letter following a lowercase letter; otherwise semantic. Multi-answer tasks '
              'are exact if any target fact value is exact; reverse lookup uses the stored '
              'fact value, and forward chains use the terminal fact, excluding the bridge.')
PRESERVATION_SEED_OFFSET = 560050


def fact_class(value):
    value = value.strip()
    code = (not re.search(r'\s', value) and
            (re.search(r'[_/:\\]', value) or re.fullmatch(r'[A-Z]{2,}', value)
             or re.search(r'[a-z][A-Z]', value)))
    return 'exact' if any(c.isdigit() for c in value) or code else 'semantic'


def item_class(item):
    if 'fact_class' in item:
        return item['fact_class']
    return 'exact' if any(fact_class(v) == 'exact' for v in item.get('vals', [item.get('val', '')])) else 'semantic'


def annotate(items, facts=None):
    for item in items:
        if facts is not None and ('fis' in item or 'fi' in item):
            ids = [fi for fi in item.get('fis', [item.get('fi')]) if fi != item.get('relation_fi')]
            item['fact_class'] = ('exact' if any(fact_class(facts[fi].val) == 'exact' for fi in ids)
                                  else 'semantic')
        else:
            item['fact_class'] = item_class(item)
    return items


def split_scores(items, scores):
    """Reduce aligned numeric metric rows without additional model forwards."""
    if len(items) != len(scores):
        raise ValueError('fact-class scores must align with items')
    result = {}
    for label in ('exact', 'semantic'):
        rows = [s for it, s in zip(items, scores) if item_class(it) == label]
        keys = scores[0].keys() if scores else ()
        result[label] = dict(n=len(rows), **{k: sum(float(r[k]) for r in rows)/len(rows)
                                            if rows else None for k in keys})
    return result


def metric_metadata(facts):
    return dict(fact_class_rule=CLASS_RULE,
                fact_classes={str(fi): fact_class(f.val) for fi, f in enumerate(facts)})


def collect_splits(report):
    """Mirror only measured panels that contain class splits, for top-level outputs."""
    if not isinstance(report, dict):
        return None
    if 'by_fact_class' in report:
        return report['by_fact_class']
    return {key: split for key, value in report.items()
            if (split := collect_splits(value))}


def add_arguments(parser, training=False):
    parser.add_argument('--fact-class-metrics', action='store_true',
                        help='C50 exact/semantic splits of GEN, QA and C34 (no extra forwards)')
    if training:
        parser.add_argument('--offtopic-kl', type=float, default=0.,
                            help='C50 fraction of supervised tokens for no-context base KL on rehearsal windows')
        parser.add_argument('--offtopic-kl-weight', type=float, default=1.)
        parser.add_argument('--intopic-neg-frac', type=float, default=0.,
                            help='C50 token fraction for sibling-entity/absent-bank answer KL; resident facts excluded')


def validate(args):
    for key in ('offtopic_kl', 'intopic_neg_frac'):
        v = getattr(args, key)
        if not math.isfinite(v) or not 0 <= v < 1:
            raise SystemExit('--'+key.replace('_', '-')+' must be finite and in [0, 1)')
    if args.offtopic_kl + args.intopic_neg_frac >= 1:
        raise SystemExit('C50 negative token fractions must sum to less than 1')
    if not math.isfinite(args.offtopic_kl_weight) or args.offtopic_kl_weight < 0:
        raise SystemExit('--offtopic-kl-weight must be finite and >= 0')
    if (args.offtopic_kl or args.intopic_neg_frac) and args.rehearsal_frac > 0:
        raise SystemExit('C50 KL negatives are mutually exclusive with --rehearsal-frac > 0')
    if args.offtopic_kl and args.anchor_len < 2:
        raise SystemExit('--offtopic-kl needs --anchor-len >= 2')
    if (args.offtopic_kl or args.intopic_neg_frac) and args.optim == 'mls':
        raise SystemExit('C50 KL negatives need the gradient training loop (not --optim mls)')
    if (args.offtopic_kl or args.intopic_neg_frac) and getattr(args, 'activation_noise', 0):
        raise SystemExit('C50 frozen-base KL is incompatible with --activation-noise forward hooks')
    if args.offtopic_kl or args.intopic_neg_frac:
        args.fact_class_metrics = True


def kl_to_base(student, teacher):
    """Summed full-vocabulary KL(base || student) in fp32; teacher is always detached."""
    t = F.log_softmax(teacher.detach().float(), -1)
    s = F.log_softmax(student.float(), -1)
    return F.kl_div(s, t, log_target=True, reduction='sum')


@contextmanager
def pristine_teacher(E, model):
    """Also disable resident and train-only teacher modules; restore before backward."""
    modules = [m for m in model.modules() if isinstance(m, (E.ResidentParam, E.TeacherLoRAParam))]
    flags = [m.enabled for m in modules]
    try:
        for m in modules:
            m.enabled = False
        with E.context_base_forward():
            yield
    finally:
        for m, enabled in zip(modules, flags):
            m.enabled = enabled


def written_pairs(E, states):
    pairs = set()
    for state in states:
        m = state['meta']
        facts = ([E.Fact(**row) for row in m['c37_facts']] if m.get('c37_facts') else
                 E.make_world(m['seed'], m['people'], m['facts_per_person'], m['shared_attrs'],
                              m['values'], m.get('relation_world', False), m.get('synthetic_people', 0)))
        ids = E.person_indices(facts, m.get('train_person'), m.get('train_attr_group'), m.get('attr_groups', 1))
        pairs.update((facts[fi].name, facts[fi].attr) for fi in ids)
    return pairs


def hard_negatives(E, facts, train_ids, protected):
    """No positive/current/resident facts are ever virgin-base KL targets."""
    held = {(facts[fi].name, facts[fi].attr) for fi in train_ids} | protected
    trained_names = {facts[fi].name for fi in train_ids}
    world_pairs = {(f.name, f.attr) for f in facts}
    names = {f.name for f in facts} | {name for name, _ in protected}
    rows = {}
    def add(name, attr, kind):
        if (name, attr) not in held:
            rows[(name, attr)] = dict(name=name, attr=attr, kind=kind,
                                     prompt=f"What is {name}'s {attr}? Answer:")
    for f in facts:
        if f.name not in trained_names:
            add(f.name, f.attr, 'sibling_entity')
    # Include novel near-sibling entities even when every person is in the bank.
    for name in sorted({f.name for f in facts}):
        sibling = name+'a'
        while sibling in names:
            sibling += 'a'
        for attr in sorted({f.attr for f in facts if f.name == name}):
            add(sibling, attr, 'sibling_entity')
        # A held-out fact of a trained entity is not an absent attribute.
        excluded = held | {(n, attr) for n, attr in world_pairs if n in trained_names}
        absent = next((attr for attr, _ in E.ATTR_POOL if (name, attr) not in excluded),
                      'unwritten archive access code')
        while (name, absent) in excluded:
            absent = 'unwritten '+absent
        add(name, absent, 'absent_bank')
    return list(rows.values())


class PreservationKL:
    """Stream windows/answers, with cumulative fractional-token rounding and bounded memory."""
    def __init__(self, E, model, tok, facts, train_ids, states, source, args, bos, dev):
        self.E, self.model, self.tok, self.args = E, model, tok, args
        self.bos, self.dev = bos, dev
        self.generator = torch.Generator(device=dev).manual_seed(args.seed + PRESERVATION_SEED_OFFSET)
        self.protected = written_pairs(E, states)
        self.negatives = hard_negatives(E, facts, train_ids, self.protected) if args.intopic_neg_frac else []
        self.source = source
        self.source_ids = None
        # Conservative exclusion: any resident entity mention disqualifies a generic window.
        if args.offtopic_kl and self.protected:
            names = {name.casefold() for name, _ in self.protected}
            keep = [i for i, win in enumerate(source) if not any(name in tok.decode(win.tolist()).casefold()
                                                               for name in names)]
            if not keep:
                raise SystemExit('C50 off-topic source has no windows free of resident entities')
            self.source_ids = torch.tensor(keep, device=dev)
        self.state = dict(remainder=[0., 0.], assistant_tokens=0, offtopic_tokens=0, intopic_tokens=0,
                          teacher_forwards=0, student_forwards=0, seconds=0., weighted_loss=0.)

    def config(self):
        a = self.args
        return dict(offtopic_kl=a.offtopic_kl, offtopic_kl_weight=a.offtopic_kl_weight,
                    sampler_seed=a.seed + PRESERVATION_SEED_OFFSET,
                    intopic_neg_frac=a.intopic_neg_frac, fraction_unit='supervised assistant/next-token tokens',
                    teacher='frozen no-context pristine base', resident_policy='exclude previously written facts',
                    protected_pairs=[list(p) for p in sorted(self.protected)],
                    hard_negative_counts={k: sum(n['kind'] == k for n in self.negatives)
                                          for k in ('sibling_entity', 'absent_bank')},
                    offtopic_source=str(self.args.rehearsal_set or self.args.anchor),
                    offtopic_windows=(len(self.source_ids) if self.source_ids is not None else
                                      len(self.source) if self.source is not None else 0),
                    rounding='carry fractional tokens across updates; error <1 token per kind')

    def sample_indices(self, size, count):
        return torch.randint(size, (count,), device=self.dev, generator=self.generator)

    def backward(self, assistant_tokens):
        E, a, st = self.E, self.args, self.state
        if str(self.dev).startswith('cuda'):
            torch.cuda.synchronize(self.dev)
        start = time.perf_counter()
        fractions = (a.offtopic_kl, a.intopic_neg_frac)
        quotas = []
        for i, fraction in enumerate(fractions):
            requested = assistant_tokens * fraction/(1-sum(fractions)) + st['remainder'][i]
            count = math.floor(requested + 1e-10)
            st['remainder'][i] = requested-count
            quotas.append(count)
        st['assistant_tokens'] += assistant_tokens
        denom = max(assistant_tokens+sum(quotas), 1)
        loss_sum = 0.
        def consume(student, teacher, kind):
            nonlocal loss_sum
            loss = a.offtopic_kl_weight * kl_to_base(student, teacher)/denom
            if loss.requires_grad:
                loss.backward()
            loss_sum += float(loss.detach())
            st[kind+'_tokens'] += student.shape[-2]
            st['student_forwards'] += 1
        remaining = quotas[0]
        if remaining:
            width = self.source.shape[1]-1
            size = len(self.source_ids) if self.source_ids is not None else len(self.source)
            ids = self.sample_indices(size, math.ceil(remaining/width))
            if self.source_ids is not None:
                ids = self.source_ids[ids]
            for idx in ids:
                n = min(remaining, width)
                window = self.source[idx:idx+1, :n+1]
                with pristine_teacher(E, self.model):
                    teacher = E.logits_of(self.model, window)[:, :-1, :E.VOCAB].float()
                st['teacher_forwards'] += 1
                student = E.logits_of(self.model, window)[:, :-1, :E.VOCAB].float()
                consume(student, teacher, 'offtopic')
                remaining -= n
        remaining = quotas[1]
        while remaining:
            row = self.negatives[int(self.sample_indices(len(self.negatives), 1)[0])]
            prompt = [self.bos]+self.tok.encode(row['prompt'], add_special_tokens=False).ids
            seq, targets = list(prompt), []
            with pristine_teacher(E, self.model):
                for _ in range(min(remaining, 32)):
                    logits = E.logits_of(self.model, torch.tensor([seq], device=self.dev))[:, -1:, :E.VOCAB].float()
                    targets.append(logits)
                    token = int(logits.argmax(-1))
                    seq.append(token)
                    st['teacher_forwards'] += 1
                    if token in E.EOS_IDS:
                        break
            student = E.logits_of(self.model, torch.tensor([seq], device=self.dev))[:, len(prompt)-1:-1, :E.VOCAB].float()
            consume(student, torch.cat(targets, dim=1), 'intopic')
            remaining -= len(targets)
        if str(self.dev).startswith('cuda'):
            torch.cuda.synchronize(self.dev)
        st['seconds'] += time.perf_counter()-start
        st['weighted_loss'] += loss_sum
        return loss_sum

    def report(self):
        st = dict(self.state)
        total = st['assistant_tokens']+st['offtopic_tokens']+st['intopic_tokens']
        st['realized_offtopic_frac'] = st['offtopic_tokens']/total if total else 0.
        st['realized_intopic_frac'] = st['intopic_tokens']/total if total else 0.
        return dict(self.config(), **st)
