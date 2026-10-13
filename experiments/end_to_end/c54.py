"""S1 span annotations, sparse fixed-bin firing histograms and unit-list I/O."""
import json
import re
import string
from pathlib import Path

import numpy as np
import torch

try:
    from . import e2e_fsa as E
except ImportError:
    import e2e_fsa as E

# Zero plus every positive finite BF16 code. Positive FP32 values are truncated
# into these fixed intervals. Native BF16 activations therefore have exact ties.
BINS = 32641


def render_spans(template, values, subject, other=None):
    text, spans = '', []
    for literal, field, spec, conversion in string.Formatter().parse(template):
        text += literal
        if field is None:
            continue
        value = str(values[field])
        start = len(text)
        text += value
        if field in ('name', 'other', 'pos', 'Pos', 'subj', 'Subj'):
            spans.append(dict(start=start, end=len(text), pid=other if field == 'other' else subject,
                              kind='name' if field in ('name', 'other') else 'pronoun'))
    return text, spans


def mention_spans(text, facts, aliases=None):
    """Find exact world names/aliases, including relation answers, with boundaries."""
    spans = []
    for f in {f.pid: f for f in facts}.values():
        for kind, names in [('name', [f.name]), ('alias', (aliases or {}).get(f.pid, []))]:
            for name in names:
                for match in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text):
                    spans.append(dict(start=match.start(), end=match.end(), pid=f.pid, kind=kind))
    # Prefer full mentions when an alias is a substring of the name.
    selected = []
    for span in sorted(spans, key=lambda s: (-(s['end']-s['start']), s['start'])):
        if not any(span['start'] < s['end'] and s['start'] < span['end'] for s in selected):
            selected.append(span)
    return selected


def token_labels(tok, text, spans, bos=None):
    encoded = tok.encode(text, add_special_tokens=False)
    offsets = getattr(encoded, 'offsets', None)
    if offsets is None:
        # The repository's tiny char tokenizer has no offset API. Do not infer
        # BPE spans by separately encoding substrings (boundary merges differ).
        if encoded.ids != [ord(c) for c in text]:
            raise ValueError('exact span labelling requires tokenizer offsets')
        offsets = [(i, i+1) for i in range(len(text))]
    labels = np.full(len(encoded.ids) + (bos is not None), -2, dtype=np.int64)
    annotated = []
    shift = int(bos is not None)
    for span in spans:
        positions = [i+shift for i, (start, end) in enumerate(offsets)
                     if end > start and start < span['end'] and span['start'] < end]
        if not positions:
            raise ValueError('entity span has no token offsets')
        for pos in positions:
            if labels[pos] not in (-2, span['pid']):
                raise ValueError('one token overlaps two different entity spans')
            labels[pos] = span['pid']
        annotated.append(dict(span, token_start=min(positions), token_end=max(positions)+1))
    return labels, annotated


def labelled_items(tok, bos, facts, aliases=None):
    """Use the actual TRAIN_AUG and train-only recent-pronoun builders."""
    items = E.build_items(tok, facts, E.TRAIN_AUG_T, bos)
    for item in items:
        f = facts[item['fi']]
        values = dict(name=f.name, pos=f.pos, Pos=f.pos.capitalize(), subj=f.subj,
                      Subj=f.subj.capitalize(), attr=f.attr, val=f.val)
        full, spans = render_spans(E.TRAIN_AUG_T[item['ti']], values, f.pid)
        prompt = full[:full.rindex(f.val)].rstrip()
        item['prompt'] = prompt
        item['_pronouns'] = [s for s in spans if s['kind'] == 'pronoun' and s['end'] <= len(prompt)]
        item['span_template'] = 'train_aug'
    people = {f.pid: f for f in facts}
    for item in E.build_address_pronoun_items(tok, facts, bos):
        f = facts[item['fi']]
        pids = item['mention_pids']
        other = pids[0] if pids else None
        values = dict(name=f.name, pos=f.pos, attr=f.attr,
                      filler=''.join(E.QA_FILLER_TURNS[:1 + item['fi'] % 3]),
                      other=people[other].name if other is not None else '')
        prompt, spans = render_spans(E.ADDRESS_PRONOUN_RECENT_T if pids else E.ADDRESS_PRONOUN_T,
                                     values, f.pid, other)
        if prompt != item['prompt']:
            raise ValueError('pronoun builder/template mismatch')
        item['_pronouns'] = [s for s in spans if s['kind'] == 'pronoun']
        item['span_template'] = 'address_pronoun_recent' if pids else 'address_pronoun'
        items.append(item)
    for item in items:
        prompt, answer = item['prompt'], ' ' + item['val']
        spans = mention_spans(prompt, facts, aliases) + item.pop('_pronouns')
        lp, sp = token_labels(tok, prompt, spans, bos)
        la, sa = token_labels(tok, answer, mention_spans(answer, facts, aliases))
        if E.encode(tok, prompt, bos) != item['p'] or tok.encode(answer, add_special_tokens=False).ids != item['a']:
            raise ValueError('span tokenization differs from teacher-forced item')
        shift = len(lp)
        sa = [dict(s, start=s['start']+len(prompt), end=s['end']+len(prompt),
                   token_start=s['token_start']+shift, token_end=s['token_end']+shift) for s in sa]
        length = len(item['p'])+len(item['a'])-1
        item['span_labels'] = np.concatenate([lp, la])[:length].tolist()
        item['spans'] = sp + [dict(s, token_end=min(length, s['token_end'])) for s in sa if s['token_start'] < length]
    return items


def merge_hist(a, b):
    """Sorted sparse (unit * BINS + bin, count, inexact-count) vectors."""
    if not len(a[0]):
        return b
    if not len(b[0]):
        return a
    keys = np.union1d(a[0], b[0])
    count, bad = np.zeros(len(keys), np.int64), np.zeros(len(keys), np.int64)
    for index, c, i in (a, b):
        slots = np.searchsorted(keys, index)
        count[slots] += c
        bad[slots] += i
    return keys, count, bad


def auc_hist(positive, negative, npos, nneg, width):
    """Midpoint and rigorous bounds; zero ties and BF16 ties are exact."""
    if not npos or not nneg:
        return np.full((3, width), np.nan)
    keys = np.union1d(positive[0], negative[0])
    p, n, pb, nb = [np.zeros(len(keys), np.float64) for _ in range(4)]
    for hist, counts, bad in ((positive, p, pb), (negative, n, nb)):
        slots = np.searchsorted(keys, hist[0])
        counts[slots], bad[slots] = hist[1], hist[2]
    units = keys // BINS
    starts = np.flatnonzero(np.r_[True, units[1:] != units[:-1]])
    prior = np.cumsum(n)-n
    prior -= np.repeat(prior[starts], np.diff(np.r_[starts, len(keys)]))
    wins = p*prior + .5*p*n
    ambiguity = p*n - (p-pb)*(n-nb)
    center, error = np.zeros(width), np.zeros(width)
    center[units[starts]] = np.add.reduceat(wins, starts)/(npos*nneg)
    error[units[starts]] = .5*np.add.reduceat(ambiguity, starts)/(npos*nneg)
    return np.stack([center, center-error, center+error])


class FiringScreen:
    def __init__(self, entities):
        self.entities = dict(entities)
        self.pids = sorted(entities)
        self.groups = self.pids + [-1]  # -1 is FineWeb; -2 is unlabelled QA.
        self.counts = {g: 0 for g in self.groups}
        self.units = {}
        self.labels = None

    def begin(self, labels):
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        for group in self.groups:
            self.counts[group] += int((self.labels == group).sum())

    def observe(self, key, meta, z, tokens):
        labels = self.labels[tokens.detach().cpu().numpy()]
        width = z.shape[1]
        if key not in self.units:
            self.units[key] = dict(meta, width=width, hist={}, routed={}, fired={}, relu2_sum={})
        acc = self.units[key]
        for group in self.groups:
            selected = labels == group
            if not selected.any():
                continue
            phi_z = z[torch.as_tensor(selected, device=z.device)].detach().float().clamp_min(0).cpu().numpy()
            if not np.isfinite(phi_z).all():
                raise ValueError('nonfinite firing activation')
            acc['routed'][group] = acc['routed'].get(group, 0) + len(phi_z)
            acc['fired'].setdefault(group, np.zeros(width, np.int64))
            acc['relu2_sum'].setdefault(group, np.zeros(width, np.float64))
            acc['fired'][group] += (phi_z > 0).sum(0)
            acc['relu2_sum'][group] += np.square(phi_z.astype(np.float64)).sum(0)
            # Histogram relu(z), a strictly monotone equivalent of relu(z)^2.
            # Truncation gives a fixed BF16 interval; subnormals are included.
            bits = phi_z.view(np.uint32)
            bins = (bits >> 16).astype(np.int64)
            bad = ((bits & 65535) != 0).astype(np.int64)
            positive = phi_z > 0
            # Zeros are implicit, filled from ALL labelled tokens at output.
            indices = np.broadcast_to(np.arange(width)[None, :], bins.shape)[positive]*BINS + bins[positive]+1
            index, inverse, count = np.unique(indices, return_inverse=True, return_counts=True)
            inexact = np.zeros(len(index), np.int64)
            np.add.at(inexact, inverse, bad[positive])
            empty = (np.array([], np.int64),)*3
            acc['hist'][group] = merge_hist(acc['hist'].get(group, empty), (index, count, inexact))

    def histogram(self, acc, groups):
        width = acc['width']
        hist = (np.arange(width)*BINS, np.zeros(width, np.int64), np.zeros(width, np.int64))
        for group in groups:
            fired = acc['fired'].get(group, np.zeros(width, np.int64))
            zeros = (np.arange(width)*BINS, self.counts[group]-fired, np.zeros(width, np.int64))
            hist = merge_hist(hist, zeros)
            hist = merge_hist(hist, acc['hist'].get(group, (np.array([], np.int64),)*3))
        return hist

    def output(self, path, metadata, items, seed=510051, top=32):
        path = Path(path).with_suffix('.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        rows = {pid: dict(other=[], fineweb=[], random=[], measured=0,
                counts={which: {str(t): dict(estimate=0, certified=0, unresolved=0) for t in (.9,.8)}
                        for which in ('other','fineweb')}) for pid in self.pids}
        rng = np.random.default_rng(seed)
        def ranking(r, column):
            return (-r[column][1], -r[column][0], r[0], r[1])
        for key, acc in sorted(self.units.items()):
            width = acc['width']
            for field in ('layer', 'expert', 'width'):
                arrays[f'{key}_{field}'] = np.array(acc[field])
            arrays[f'{key}_name'] = np.array(acc['name'])
            for group in self.groups:
                label = f'entity{group}' if group >= 0 else 'fineweb'
                hist = self.histogram(acc, [group])
                for field, value in zip(('hist_index', 'hist_count', 'hist_inexact'), hist):
                    arrays[f'{key}_{label}_{field}'] = value
                arrays[f'{key}_{label}_routed'] = np.array(acc['routed'].get(group, 0))
                arrays[f'{key}_{label}_fired'] = acc['fired'].get(group, np.zeros(width, np.int64))
                arrays[f'{key}_{label}_relu2_sum'] = acc['relu2_sum'].get(group, np.zeros(width))
            fineweb = self.histogram(acc, [-1])
            for pid in self.pids:
                rest = [p for p in self.pids if p != pid]
                pos = self.histogram(acc, [pid])
                other = self.histogram(acc, rest)
                ao = auc_hist(pos, other, self.counts[pid], sum(self.counts[p] for p in rest), width)
                af = auc_hist(pos, fineweb, self.counts[pid], self.counts[-1], width)
                arrays[f'{key}_entity{pid}_auc_other'] = ao
                arrays[f'{key}_entity{pid}_auc_fineweb'] = af
                measured = np.isfinite(ao).all(0) & np.isfinite(af).all(0)
                units = np.flatnonzero(measured)
                state = rows[pid]
                state['measured'] += len(units)
                for which, auc in [('other', ao), ('fineweb', af)]:
                    for threshold in (.9,.8):
                        counts = state['counts'][which][str(threshold)]
                        counts['estimate'] += int(np.sum(measured & (auc[0] >= threshold)))
                        counts['certified'] += int(np.sum(measured & (auc[1] >= threshold)))
                        counts['unresolved'] += int(np.sum(measured & (auc[1] < threshold) & (auc[2] >= threshold)))
                    order = np.lexsort((units, -auc[0,units], -auc[1,units]))[:max(top,8)]
                    state[which] += [(key,int(unit),ao[:,unit],af[:,unit]) for unit in units[order]]
                    state[which].sort(key=lambda r: ranking(r, 2 if which == 'other' else 3))
                    del state[which][max(top,8):]
                # Lowest iid priorities implement a uniform sample without
                # retaining a Python object per unit of the full 725B model.
                priorities = rng.random(len(units))
                take = np.argsort(priorities)[:8]
                state['random'] += [(float(priorities[i]), (key,int(units[i]),ao[:,units[i]],af[:,units[i]])) for i in take]
                state['random'].sort(key=lambda r: r[0])
                del state['random'][8:]
        def stats(key, unit, groups):
            acc = self.units[key]
            tokens = sum(self.counts[g] for g in groups)
            routed = sum(acc['routed'].get(g, 0) for g in groups)
            fired = sum(acc['fired'].get(g, np.zeros(acc['width']))[unit] for g in groups)
            relu2 = sum(acc['relu2_sum'].get(g, np.zeros(acc['width']))[unit] for g in groups)
            return dict(tokens=tokens, routed=routed, fired=int(fired), relu2_sum=float(relu2),
                        firing_rate=float(fired/tokens) if tokens else None,
                        mean_relu2=float(relu2/tokens) if tokens else None,
                        conditional_firing_rate=float(fired/routed) if routed else None,
                        conditional_mean_relu2=float(relu2/routed) if routed else None)
        def row(pid, r):
            key, unit, other, web = r
            acc = self.units[key]
            def auc(a):
                return dict(zip(('estimate', 'lower', 'upper'), [float(v) if np.isfinite(v) else None for v in a]))
            return dict(layer=acc['layer'], expert=acc['expert'], name=acc['name'], unit=unit,
                        auc_other=auc(other), auc_fineweb=auc(web),
                        entity=stats(key, unit, [pid]), other=stats(key, unit, [p for p in self.pids if p != pid]),
                        fineweb=stats(key, unit, [-1]))
        summary = dict(metadata, screen='S1', score='relu(z)^2; non-dispatched units score zero',
                       histogram=dict(bins=BINS, method='sparse fixed positive BF16 intervals plus exact zero',
                                      auc_rows=['estimate', 'lower', 'upper'], ties='exact for BF16; bounds for within-bin FP32 ordering'),
                       counts={str(g): n for g, n in self.counts.items()}, entities={})
        combined, random_combined = {}, {}
        for pid, state in rows.items():
            chosen = state['other'][:8]
            random = [r for _,r in state['random']]
            def unit_file(kind, selected):
                units = {}
                for key, unit, _, _ in selected:
                    units.setdefault(self.units[key]['name'], []).append(unit)
                target = path.with_name(f'{path.stem}.entity{pid}.{kind}.units.json')
                target.write_text(json.dumps(dict(version=1, entity=self.entities[pid], pid=pid,
                                                 selection=kind, units=units), indent=2)+'\n')
                return str(target), units
            topfile, selected = unit_file('top8', chosen)
            randomfile, randomized = unit_file('random8', random)
            for target, source in ((combined, selected), (random_combined, randomized)):
                for key, units in source.items():
                    target.setdefault(key, set()).update(units)
            summary['entities'][str(pid)] = dict(name=self.entities[pid], counts=state['counts'],
                measured_units=state['measured'], top_units=[row(pid, r) for r in state['other'][:top]],
                top_fineweb=[row(pid, r) for r in state['fineweb'][:top]],
                top8_file=topfile, random8_file=randomfile,
                top8_qualified_at_09=sum(r[2][1] >= .9 for r in chosen), selected_units=len(chosen))
            summary['entities'][str(pid)]['top8_qualified_at_09'] = int(summary['entities'][str(pid)]['top8_qualified_at_09'])
        for kind, units in [('top8', combined), ('random8', random_combined)]:
            target = path.with_name(f'{path.stem}.{kind}.units.json')
            target.write_text(json.dumps(dict(version=1, selection=kind,
                units={key: sorted(values) for key, values in units.items()}), indent=2)+'\n')
            summary[kind+'_file'] = str(target)
            summary[kind+'_union_units'] = sum(len(values) for values in units.values())
        summary['decision'] = ('value_only_control' if any(v['counts']['other']['0.9']['certified']
            for v in summary['entities'].values()) else 'unresolved' if any(v['counts']['other']['0.9']['unresolved']
            for v in summary['entities'].values()) else 'train_addresses')
        path.with_suffix('.spans.json').write_text(json.dumps(items, indent=2)+'\n')
        arrays['metadata_json'] = np.array(json.dumps(summary, allow_nan=False))
        np.savez_compressed(path, **arrays)
        path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        return summary


def load_unit_restriction(model, path, layer_ids):
    payload = json.loads(Path(path).read_text())
    if payload.get('version') != 1 or not isinstance(payload.get('units'), dict) or not payload['units']:
        raise ValueError('unit list requires version=1 and a nonempty units mapping')
    banks = {f'{name}.experts.{ei}': expert for li, (name, moe) in enumerate(E.moe_layers(model))
             if li in layer_ids for ei, expert in enumerate(moe.experts)}
    restrict = {}
    for key, indices in payload['units'].items():
        if key not in banks or not isinstance(indices, list) or not indices:
            raise ValueError(f'unknown/out-of-layer or empty unit bank: {key}')
        width = banks[key].down_proj.weight.shape[1]
        if any(type(i) is not int or not 0 <= i < width for i in indices) or len(set(indices)) != len(indices):
            raise ValueError(f'invalid/duplicate hidden unit in {key}')
        selected = torch.zeros(width, dtype=torch.bool)
        selected[indices] = True
        restrict[key+'.down_proj'] = ('cols', selected)
    return restrict, payload
