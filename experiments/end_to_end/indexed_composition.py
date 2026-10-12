"""C29: inverse postings, two-round cleanup and bounded joint hypotheses.

All inference uses query fields and stored bank keys. Gold fields enter only
metrics and TRAIN fitting. The evaluator is deliberately a common-work suite:
every readout has the same resident state and receives the same bank likelihood
table; it is not an optimized serving-speed comparison.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
import json
import math

import torch

try:
    from . import e2e_fsa as E, branch_eval as B, biology_addressing as BIO, two_hop as H
except ImportError:
    import e2e_fsa as E
    import branch_eval as B
    import biology_addressing as BIO
    import two_hop as H

BASELINES = ('indexed', 'sum', 'product', 'min', 'learned_joint_key', 'learned_joint_scorer')
READOUTS = ('early', 'posterior', 'beam2', 'beam4', 'beam8',
            'amplitude2', 'amplitude4', 'amplitude8', 'oracle_id', 'oracle_value')
TEMPLATES = {
    'train': ('Whose {attr} is {value}? What are their {a} and {b}?',
              'Who has {a} {va} and {b} {vb}?',
              '{first} is speaker A. {second} is speaker B. Regarding the {cue}, what is her {attr}?',
              'I saw {name}. Tell me {pos} {a} and {b}.'),
    'test': ('Find the person with {attr} {value}; give their {a} together with their {b}.',
             'Name everyone whose {a} is {va} and whose {b} is {vb}.',
             'I met {first}, speaker A, then {second}, speaker B. For the {cue}, tell me her {attr}.',
             '{name} arrived today. What are {pos} {a} together with {b}?')}


class InverseIndex:
    """Posting keys are (relation, value), compiled from the stored bank keys.

    Keep fact IDs as provenance. Missing/revoked evidence yields an empty set,
    and rebuilding from surviving keys is identical to a clean construction.
    """
    def __init__(self, bank_keys):
        self.keys = tuple(bank_keys)
        if len({fi for fi, *_ in self.keys}) != len(self.keys):
            raise ValueError('bank fact IDs must be unique')
        self.postings = {}
        for fi, pid, relation, value in self.keys:
            self.postings.setdefault((relation, value), {}).setdefault(pid, set()).add(fi)

    @classmethod
    def from_facts(cls, facts):
        return cls((fi, f.pid, f.attr, f.val) for fi, f in enumerate(facts))

    def lookup(self, relation, value):
        return tuple(sorted(self.postings.get((relation, value), {})))

    def conjunction(self, constraints):
        sets = [set(self.lookup(*key)) for key in constraints]
        return tuple(sorted(set.intersection(*sets))) if sets else ()

    def revoke(self, fact_ids):
        removed = set(fact_ids)
        return type(self)(key for key in self.keys if key[0] not in removed)

    def storage_bytes(self):
        # Count both original bank keys AND compiled reverse postings/provenance.
        postings = [(relation, value, [(pid, sorted(ids)) for pid, ids in sorted(owners.items())])
                    for (relation, value), owners in sorted(self.postings.items())]
        return len(json.dumps(dict(keys=self.keys, postings=postings), ensure_ascii=False,
                              separators=(',', ':')).encode())


def _item(tok, bos, question, **fields):
    q = E.encode(tok, question, bos)
    return dict(question=question, q=q, p=q+tok.encode('\nAnswer:', add_special_tokens=False).ids,
                position=len(q)-1, **fields)


def composition_items(tok, facts, bos, split):
    if split not in TEMPLATES:
        raise ValueError('split must be train or test')
    people = {f.pid: f.name for f in facts}
    index = InverseIndex.from_facts(facts)
    items = []
    for pid in sorted(people):
        own = [(fi, f) for fi, f in enumerate(facts) if f.pid == pid]
        if len(own) < 2:
            raise ValueError('C29 requires at least two stored attributes per person')
        # All pairs, not hand-picked instrument/stop facts absent from some worlds.
        for (ai, a), (bi, b) in combinations(own, 2):
            if len(own) == 2:
                # Tiny/C27 worlds have no third locator fact. Supply an explicit
                # name mention and retrieve its two values without leaking either.
                question = TEMPLATES[split][3].format(name=people[pid], pos=a.pos, a=a.attr, b=b.attr)
                items.append(_item(tok, bos, question, kind='retrieval_two_values', constraints=(),
                    target=pid, sources=(pid,), answer_attrs=(a.attr, b.attr), vals=(a.val, b.val),
                    answer='; '.join((a.val, b.val)), ambiguous=False, hop=False))
            for locator_i, locator in own:
                # The locator is a separate fact; answer values never appear in the question.
                if locator_i in (ai, bi):
                    continue
                question = TEMPLATES[split][0].format(attr=locator.attr, value=locator.val, a=a.attr, b=b.attr)
                sources = index.lookup(locator.attr, locator.val)
                items.append(_item(tok, bos, question, kind='retrieval_two_values',
                    constraints=((locator.attr, locator.val),), target=pid, sources=sources,
                    answer_attrs=(a.attr, b.attr), vals=(a.val, b.val),
                    answer='; '.join((a.val, b.val)), ambiguous=False, hop=False))
            # Include both a valid intersection and a cross-person empty/conflicting one.
            others = [f for f in facts if f.pid != pid and f.attr == b.attr]
            for rhs in (b, *others):
                constraints = ((a.attr, a.val), (rhs.attr, rhs.val))
                targets = index.conjunction(constraints)
                question = TEMPLATES[split][1].format(a=a.attr, va=a.val, b=rhs.attr, vb=rhs.val)
                items.append(_item(tok, bos, question, kind='intersection', constraints=constraints,
                    target=targets[0] if targets else None, targets=targets, sources=tuple(people),
                    answer_attrs=(), vals=tuple(people[p] for p in targets),
                    answer='; '.join(people[p] for p in targets) or 'No one', ambiguous=False, hop=False))
    return items


def hypothesis_items(tok, facts, relations, bos, split):
    """Balanced role cues and truly ambiguous, identical-prompt hidden labels.

    The latter have two same-gender candidates and no resolving cue. Their gold
    labels are balanced, so confidently guessing cannot masquerade as resolution.
    Both direct pronouns and correlated (referent, sister) paths are included.
    """
    people = {f.pid: f for f in facts}
    if set(relations) != set(people) or not set(relations.values()) <= set(people):
        raise ValueError('relations must cover the stored entities')
    if split not in TEMPLATES:
        raise ValueError('split must be train or test')
    items = []
    for first, second in combinations(people.values(), 2):
        if first.pos != second.pos:
            continue
        for left, right in ((first, second), (second, first)):
            for source in (left, right):
                for hop in (False, True):
                    target = relations[source.pid] if hop else source.pid
                    for fi, f in enumerate(facts):
                        if f.pid != target:
                            continue
                        for ambiguous in (False, True):
                            if ambiguous:
                                alternate = right if source.pid == left.pid else left
                                alternate_target = relations[alternate.pid] if hop else alternate.pid
                                if not any(other.pid == alternate_target and other.attr == f.attr for other in facts):
                                    continue
                            attr = "sister's "+f.attr if hop else f.attr
                            cue = ('person' if ambiguous else 'speaker A' if source.pid == left.pid else 'speaker B')
                            question = TEMPLATES[split][2].format(first=left.name, second=right.name, cue=cue, attr=attr)
                            # Pronoun follows the actual gender of the pair.
                            question = question.replace('her ', left.pos+' ')
                            slots = BIO.mention_slots(tok, question, {left.pid: left.name, right.pid: right.name}, bos)
                            items.append(_item(tok, bos, question, kind='multi_hop' if hop else 'pronoun',
                                fi=fi, target=target, source=source.pid, sources=(left.pid, right.pid),
                                constraints=(), answer_attrs=(f.attr,), vals=(f.val,), answer=f.val,
                                ambiguous=ambiguous, hop=hop, slots=slots))
    if not items:
        raise ValueError('C29 requires two same-gender candidates')
    if not any(it['ambiguous'] for it in items):
        raise ValueError('C29 ambiguity requires a shared answer attribute for the candidate pair')
    return items


@dataclass(frozen=True)
class Hypothesis:
    source: int | None
    target: int | None


def universe(owners):
    return (Hypothesis(None, None), *(Hypothesis(s, t) for s in owners for t in owners))


def evidence(item, owners, index, relations, resolver_scores=None):
    """Two-round joint path; a typed edge filter preserves source/target correlation."""
    hypotheses = universe(owners)
    cues = torch.zeros(len(hypotheses), 2)
    legal = torch.zeros(len(hypotheses), dtype=torch.bool)
    legal[0] = True  # Explicit null, including empty intersections.
    source_probs = None
    if resolver_scores is not None:
        legal[0] = False  # Two explicit referents; null is for missing indexed evidence.
        source_probs = torch.softmax(torch.as_tensor(resolver_scores).float(), -1)
        if len(source_probs) != len(item['slots']):
            raise ValueError('resolver scores must align with mention slots')
    matches = [index.lookup(*constraint) for constraint in item['constraints']]
    for i, hyp in enumerate(hypotheses[1:], 1):
        legal[i] = (hyp.source in item['sources'] and
                    hyp.target == (relations[hyp.source] if item['hop'] else hyp.source))
        if source_probs is not None:
            probability = sum(float(p) for p, slot in zip(source_probs, item['slots']) if slot['pid'] == hyp.source)
            cues[i] = torch.tensor([probability, 1.])
        elif matches:
            cues[i] = torch.tensor([float(hyp.source in matches[0]),
                                   float(hyp.source in matches[1]) if len(matches) > 1 else 1.])
        elif item['kind'] == 'retrieval_two_values' and len(item['sources']) == 1:
            cues[i] = torch.tensor([float(hyp.source == item['sources'][0]), 1.])
        else:
            raise ValueError('query needs inverse constraints or resolver evidence')
    support = cues[1:].prod(-1)[legal[1:]].max() if legal[1:].any() else torch.tensor(0.)
    cues[0] = 1-support
    return hypotheses, cues, legal


class JointRetrievers:
    """TRAIN-fitted per-hypothesis joint keys and a shared nonlinear joint scorer.

    Keys are logistic compatibility vectors in [cue1,cue2,cue1*cue2] space;
    the scorer is a four-unit tanh MLP over the same space. Equal training steps,
    loss and regularization; no TEST templates or labels are fitted.
    """
    def __init__(self, keys, hidden, out):
        self.keys, self.hidden, self.out = keys, hidden, out

    @staticmethod
    def features(cues):
        return torch.cat((cues, cues.prod(-1, keepdim=True), torch.ones_like(cues[..., :1])), -1)

    @classmethod
    def fit(cls, cues, legal, targets, steps=100):
        x = cls.features(cues).detach().float().cpu()
        legal, targets = legal.cpu(), targets.cpu().float()
        if (steps < 1 or x.ndim != 3 or legal.shape != x.shape[:2] or targets.shape != legal.shape
                or not torch.isfinite(x).all() or (targets.sum(-1) == 0).any()
                or (targets.bool() & ~legal).any()):
            raise ValueError('invalid TRAIN retrieval evidence')
        parameters = [torch.zeros(x.shape[1], 4, requires_grad=True),
                      (torch.randn(4, 4, generator=torch.Generator().manual_seed(29))*.1).requires_grad_(),
                      torch.zeros(4, requires_grad=True)]
        key, hidden, out = parameters
        opt = torch.optim.Adam(parameters, lr=.03)
        target = targets/targets.sum(-1, keepdim=True)
        for _ in range(steps):
            opt.zero_grad()
            losses = []
            for score in ((x*key).sum(-1), (torch.tanh(x@hidden.T)*out).sum(-1)):
                logp = score.masked_fill(~legal, -1e9).log_softmax(-1)
                losses.append(-(target*logp).sum(-1).mean())
            loss = sum(losses)+.001*sum(p.square().mean() for p in parameters)
            loss.backward()
            opt.step()
        return cls(*(p.detach() for p in parameters))

    def scores(self, cues, legal):
        x = self.features(cues)
        # Log product and log min preserve exact zero support via a finite floor.
        raw = dict(indexed=cues.prod(-1).clamp_min(1e-12).log(),
                   sum=cues.sum(-1), product=cues.clamp_min(1e-12).log().sum(-1),
                   min=cues.amin(-1).clamp_min(1e-12).log(),
                   learned_joint_key=(x*self.keys).sum(-1),
                   learned_joint_scorer=(torch.tanh(x@self.hidden.T)*self.out).sum(-1))
        return {name: score.masked_fill(~legal, -torch.inf) for name, score in raw.items()}

    def storage_bytes(self):
        return sum(p.numel()*p.element_size() for p in (self.keys, self.hidden, self.out))


def posterior(scores, beam=None):
    score = torch.as_tensor(scores).double()
    if score.ndim != 1 or torch.isnan(score).any() or torch.isposinf(score).any() or not torch.isfinite(score).any():
        raise ValueError('hypotheses need at least one finite score')
    if beam is not None:
        if beam not in (1, 2, 4, 8):
            raise ValueError('beam must be 1, 2, 4, or 8')
        order = score.argsort(descending=True, stable=True)
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask[order[:beam]] = True
        score = score.masked_fill(~mask, -torch.inf)
    return score.softmax(-1)


def target_weights(probabilities, hypotheses, owners):
    # Include null as the first likelihood row; sum joint probabilities only
    # after path scoring/pruning, never combine independent marginal IDs.
    return torch.stack([sum((probabilities[i] for i, h in enumerate(hypotheses) if h.target == pid),
                            probabilities.new_zeros(())) for pid in (None, *owners)])


def marginal_answers(weights, bank_logprobs):
    """Sum_h p(h) p(answer|bank_h), using sequence log-probabilities."""
    weights = torch.as_tensor(weights).double()
    lp = torch.as_tensor(bank_logprobs).double()
    if (lp.ndim != 2 or weights.shape != lp.shape[:1] or not torch.isfinite(lp).all()
            or not torch.isfinite(weights).all() or (weights < 0).any()
            or not torch.isclose(weights.sum(), weights.new_tensor(1.))):
        raise ValueError('invalid bank likelihoods or normalized posterior')
    answer_lp = torch.logsumexp(weights.log()[:, None]+lp, 0)
    # Confidence is conditional on the generated candidate pool, not full-vocab calibration.
    return answer_lp.softmax(-1)


def hypothesis_gate(reports):
    # Compare hypothesis QA, and check confident errors on its ambiguous subset.
    singles = {name: row for name, row in reports.items() if name.endswith(':early')}
    preserved = {name: row for name, row in reports.items()
                 if name.split(':')[-1] in READOUTS[1:8]}
    best = max((row['hypothesis_questions']['accuracy'] for row in singles.values()
                if row['hypothesis_questions']['accuracy'] is not None), default=None)
    if best is None or not all(row['ambiguous']['n'] for row in singles.values()):
        return dict(rule='>=5 pts over best single; no increase in p_max>=0.8 errors on ambiguous items',
                    passed=None, reason='no ambiguous items')
    # To pass, do not increase errors relative to ANY tied best single.
    ceiling = min(row['ambiguous']['confident_errors'] for row in singles.values()
                  if abs(row['hypothesis_questions']['accuracy']-best) < 1e-12)
    gains = {name: row['hypothesis_questions']['accuracy']-best for name, row in preserved.items()}
    passed = [name for name, row in preserved.items() if gains[name] >= .05-1e-12 and
              row['ambiguous']['confident_errors'] <= ceiling]
    return dict(rule='>=5 pts over best single; no increase in p_max>=0.8 errors on ambiguous items',
                passed=bool(passed), best_single_accuracy=best, best_single_confident_errors=ceiling,
                gains=gains, passing_arms=passed)


@contextmanager
def amplitude_banks(branches, weights):
    """One forward with sqrt(p) times each saved bank's projection contribution.

    Floating sparse deltas are summed before projection application. No ternary
    requantization, probability-weighted logits or separate bank forwards here.
    Unit reset overlays are included as weighted changes to the original base.
    """
    weights = torch.as_tensor(weights).double()
    owners = tuple(sorted(branches.states))
    if (weights.shape != (len(owners)+1,) or not torch.isfinite(weights).all()
            or (weights < 0).any() or not torch.isclose(weights.sum(), weights.new_tensor(1.))):
        raise ValueError('amplitude weights must include null and all banks')
    handles = []
    enabled = E.FSAParam.enabled
    branches.activate(())
    try:
        for name, lin in E.patched_linears(branches.model):
            entries = [branches.states[p]['entries'][name] for p in owners]
            if any(e['kind'] != 'fsa' or e['fmt'] != 'ternary' or e['scale'] != 'row' or
                   e.get('flip_idx', torch.empty(0)).numel() for e in entries):
                raise ValueError('amplitude mixing needs row-scaled ternary banks without occupied flips')
            pm = lin.parametrizations.weight[0]
            def mix(module, inputs, output, entries=entries):
                W = inputs[0]
                if not E.FSAParam.enabled:
                    return W
                delta = torch.zeros_like(W, dtype=torch.float32)
                for amplitude, entry in zip(weights[1:].sqrt(), entries):
                    # Visit every bank even at zero amplitude: fixed work.
                    idx = entry['idx'].long().to(W.device)
                    alpha = entry['alpha'].to(W.device)
                    vals = entry['val'].float().to(W.device)*alpha[idx//W.shape[1]]
                    delta.flatten().index_add_(0, idx, (amplitude.to(W.device)*vals).float())
                    reset = torch.zeros_like(W, dtype=torch.bool)
                    if 'unit_reset_rows' in entry:
                        reset[entry['unit_reset_rows'].long().to(W.device)] = True
                    if 'unit_reset_cols' in entry:
                        reset[:, entry['unit_reset_cols'].long().to(W.device)] = True
                    delta.add_(W.float().masked_fill(~reset, 0), alpha=-float(amplitude))
                return W+delta.to(W.dtype)
            handles.append(pm.register_forward_hook(mix))
        yield
    finally:
        for handle in handles:
            handle.remove()
        branches.activate(())
        E.set_enabled(enabled)


class ModelBanks:
    """Frozen-bank backend with an explicit canonical read after entity retrieval."""
    def __init__(self, branches, tok, codes, layer, dev, max_new=32):
        self.branches, self.tok, self.codes = branches, tok, codes
        self.layer, self.dev, self.max_new = layer, dev, max_new
        self.owners = tuple(sorted(codes))
        self.work = dict(cleanup_calls=0, cleanup_scalar_products=0, tf_prefill_tokens=0,
                         amplitude_sparse_entries=0)

    @contextmanager
    def read(self, item, pid=None, weights=None):
        # Decode from a new scoped question. Keep the raw query available to the
        # evaluator, but preserve relation words when snapping the residual.
        if weights is None:
            self.branches.activate(())
            if pid is None:
                yield
                return
            raw = self.codes[pid]
            decision = H.cleanup_candidates(raw, self.codes, theta=-1., margin=0.)
            self.work['cleanup_calls'] += 1
            self.work['cleanup_scalar_products'] += len(self.codes)*raw.numel()
            if decision['canonical'] != pid:
                raise ValueError('canonical entity codes do not clean up to their own IDs')
            canonical = self.codes[pid].to(self.dev)
            self.branches.activate((pid,))
            with H.pointer_hook(self.branches.model, self.layer,
                    lambda x: canonical.to(x.dtype).expand_as(x), [item['position']]):
                yield
        else:
            # Mix canonical bank contributions in ONE forward. Identity stays a
            # joint probability vector; we do not average entity address codes.
            with amplitude_banks(self.branches, weights):
                yield

    def scoped(self, item, pid):
        if pid is None or item['kind'] == 'intersection':
            return item
        attrs = ' and '.join(item['answer_attrs'])
        return _item(self.tok, E.BOS, f"What is {self.names[pid]}'s {attrs}?")

    def generate(self, item, pid):
        item = self.scoped(item, pid)
        with self.read(item, pid):
            return E.greedy_generate(self.branches.model, [item['p']], self.dev, self.max_new, bs=1)[0]

    def logprobs(self, item, pid, answers, weights=None):
        if weights is None:
            item = self.scoped(item, pid)
        result = []
        with self.read(item, pid, weights):
            for answer in answers:
                # Sum token log-probs, not mean per-token confidence from C27.
                row = E.tf_eval(self.branches.model, [dict(p=item['p'], a=answer)], self.dev, bs=1)
                self.work['tf_prefill_tokens'] += len(item['p'])+len(answer)
                if weights is not None:
                    self.work['amplitude_sparse_entries'] += sum(e['idx'].numel()
                        for state in self.branches.states.values() for e in state['entries'].values())
                result.append(-row['answer_bits']*math.log(2))
        return torch.tensor(result, dtype=torch.double)

    def text(self, tokens):
        return self.tok.decode(tokens).strip()

    def close(self):
        self.branches.activate(())


def _targets(item, hypotheses):
    if item['kind'] == 'intersection':
        targets = item['targets']
        return torch.tensor([h.source == h.target and h.target in targets if targets else h.target is None
                             for h in hypotheses])
    return torch.tensor([h.target == item['target'] and
                         h.source == (item['source'] if item.get('slots') else item['target']) for h in hypotheses])


def evaluate_suite(backend, index, retrievers, items, evidence_rows, persistent_bytes=0):
    """Common-work evaluation: build every bank and amplitude likelihood table.

    Candidate answers are bank-generated; never insert the gold answer into the
    pool. Repeated answers are kept as padded slots to make work exactly fixed,
    then deduplicated before confidence normalization. All five scorers and all
    readouts are computed unconditionally on every query. Each report receives
    the full common budget, including tables it ignores.
    """
    if not items or len(items) != len(evidence_rows):
        raise ValueError('nonempty items need one evidence row each')
    reports = {f'{b}:{r}': [] for b in BASELINES for r in READOUTS}
    owners = backend.owners
    reads = dict(bank_generations=0, bank_answer_forwards=0, amplitude_answer_forwards=0,
                 inverse_lookups=0, hypothesis_scores=0, answer_tokens_scored=0)
    max_scratch = 0
    indexed = {}
    try:
        for item_i, (item, (hypotheses, cues, legal)) in enumerate(zip(items, evidence_rows)):
            raw_scores = retrievers.scores(cues, legal)
            reads['hypothesis_scores'] += len(hypotheses)*len(BASELINES)
            # Round 1 retrieved entity/path; round 2 canonical cleanup and bank read.
            reads['inverse_lookups'] += len(item['constraints'])
            if item['constraints']:
                indexed[str(item_i)] = index.conjunction(item['constraints'])
            # Preserve the actual joint hypotheses BEFORE round-two value reads.
            posterior_rows, scratchpads = {}, {}
            for baseline, score in raw_scores.items():
                for readout in READOUTS[:8]:
                    beam = 1 if readout == 'early' else int(readout[-1]) if readout.startswith(('beam', 'amplitude')) else None
                    probs = posterior(score, beam)
                    posterior_rows[baseline, readout] = probs
                    scratch = torch.full((8, 3), -1., dtype=torch.double)
                    for slot, j in enumerate(probs.argsort(descending=True, stable=True)[:8].tolist()):
                        hyp = hypotheses[j]
                        scratch[slot] = torch.tensor([hyp.source if hyp.source is not None else -1,
                            hyp.target if hyp.target is not None else -1, float(probs[j])], dtype=torch.double)
                    scratchpads[baseline, readout] = scratch
            answers = [backend.generate(item, pid) for pid in (None, *owners)]
            if any(not answer for answer in answers):
                raise ValueError('bank generation must return a nonempty continuation')
            reads['bank_generations'] += len(answers)
            table = torch.stack([backend.logprobs(item, pid, answers) for pid in (None, *owners)])
            reads['bank_answer_forwards'] += len(answers)**2
            reads['answer_tokens_scored'] += len(answers)*sum(map(len, answers))
            unique = list(dict.fromkeys(tuple(answer) for answer in answers))
            unique_indices = [next(i for i, a in enumerate(answers) if tuple(a) == u) for u in unique]
            # Explicit scratchpad reserves eight correlated (source,target,p) slots,
            # full posterior and all mixed tables for every readout, including early.
            cached = []
            for baseline, score in raw_scores.items():
                for readout in READOUTS[:8]:
                    probs = posterior_rows[baseline, readout]
                    weights = target_weights(probs, hypotheses, owners)
                    if readout.startswith('amplitude'):
                        lp = backend.logprobs(item, None, answers, weights)
                        ap = lp[unique_indices].softmax(-1)
                        reads['amplitude_answer_forwards'] += len(answers)
                        reads['answer_tokens_scored'] += sum(map(len, answers))
                    else:
                        ap = marginal_answers(weights, table[:, unique_indices])
                    cached.extend((probs, weights, ap))
                    answer_index = int(ap.argmax())
                    text = backend.text(unique[answer_index])
                    if baseline == 'indexed' and item['kind'] == 'intersection':
                        # Explicit set conjunction returns every matching entity,
                        # including many-to-one postings and the empty set.
                        matches = indexed[str(item_i)]
                        text = '; '.join(backend.names[p] for p in matches) or 'No one'
                    if item['kind'] == 'intersection':
                        # Exact set scoring includes false names and empty answers.
                        found = tuple(p for p in owners if backend.names[p].casefold() in text.casefold())
                        hit = found == item['targets'] and (bool(found) or 'no one' in text.casefold())
                    else:
                        hit = all(B.contains_value(text, value) for value in item['vals'])
                    ids = {h.target for h, p in zip(hypotheses, probs) if p > 0 and h.target is not None}
                    gold_ids = set(item['targets']) if item['kind'] == 'intersection' else {item['target']}
                    candidate_recall = gold_ids <= ids if gold_ids else not ids
                    selected = (None, *owners)[int(weights.argmax())]
                    address = {selected} == gold_ids if gold_ids else selected is None
                    if baseline == 'indexed' and item['kind'] == 'intersection':
                        address = set(matches) == gold_ids
                        candidate_recall = address
                        # Set conjunction has an exact structural result. This
                        # confidence is not used by the pronoun ambiguity gate.
                        ap = torch.ones(1, dtype=torch.double)
                    reports[f'{baseline}:{readout}'].append(dict(hit=hit, confidence=float(ap.max()),
                        address=address, candidate_recall=candidate_recall, ambiguous=item['ambiguous'], kind=item['kind'],
                        hypothesis_count=int((probs > 0).sum()), null_probability=float(weights[0]),
                        answer=text, target_probabilities=weights.tolist(), joint_hypotheses=[
                            dict(source=h.source, target=h.target, probability=float(p))
                            for h, p in zip(hypotheses, probs) if p > 0]))
                # Oracles are separate ceilings; they do not enter the best-single gate.
                gold = _targets(item, hypotheses).double()
                gold /= gold.sum()
                weights = target_weights(gold, hypotheses, owners)
                ap = marginal_answers(weights, table[:, unique_indices])
                text = backend.text(unique[int(ap.argmax())])
                if item['kind'] == 'intersection':
                    found = tuple(p for p in owners if backend.names[p].casefold() in text.casefold())
                    oracle_hit = found == item['targets'] and (bool(found) or 'no one' in text.casefold())
                else:
                    oracle_hit = all(B.contains_value(text, v) for v in item['vals'])
                for readout, hit, answer in (('oracle_id', oracle_hit, text), ('oracle_value', True, item['answer'])):
                    reports[f'{baseline}:{readout}'].append(dict(hit=hit, confidence=float(ap.max()) if readout == 'oracle_id' else 1.,
                        address=True, candidate_recall=True, ambiguous=item['ambiguous'], kind=item['kind'],
                        hypothesis_count=int((gold > 0).sum()),
                        null_probability=float(weights[0]), answer=answer, target_probabilities=weights.tolist()))
            max_scratch = max(max_scratch, getattr(backend, 'scratch_bytes', 0)+
                              sum(s.numel()*s.element_size() for s in scratchpads.values())+
                              table.numel()*table.element_size()+
                              sum(t.numel()*t.element_size() for t in cached)+
                              sum(len(a)*8 for a in answers)+cues.numel()*cues.element_size()+legal.numel()+
                              sum(s.numel()*s.element_size() for s in raw_scores.values()))
    finally:
        backend.close()
    reads.update(getattr(backend, 'work', {}))
    def summary(rows):
        n = len(rows)
        addressed = [r for r in rows if r['address']]
        return dict(n=n, accuracy=sum(r['hit'] for r in rows)/n if n else None,
                    address_accuracy=sum(r['address'] for r in rows)/n if n else None,
                    candidate_recall=sum(r['candidate_recall'] for r in rows)/n if n else None,
                    conditional_value_accuracy=sum(r['hit'] for r in addressed)/len(addressed) if addressed else None,
                    confident_errors=sum(not r['hit'] and r['confidence'] >= .8 for r in rows),
                    mean_hypotheses=sum(r['hypothesis_count'] for r in rows)/n if n else None)
    result = {}
    for name, rows in reports.items():
        result[name] = dict(overall=summary(rows), ambiguous=summary([r for r in rows if r['ambiguous']]),
            hypothesis_questions=summary([r for r in rows if r['kind'] in ('pronoun', 'multi_hop')]),
            by_kind={kind: summary([r for r in rows if r['kind'] == kind]) for kind in sorted({r['kind'] for r in rows})},
            resources=dict(persistent_bytes=persistent_bytes, active_state_bytes=max_scratch, retrieval_work=dict(reads)),
            predictions=rows)
    gate = hypothesis_gate(result)
    print('HYPOTHESIS-PASS ' + json.dumps(gate), flush=True)
    return dict(subarms=result, hypothesis_gate=gate, indexed_candidates=indexed,
        budget_policy='Common-work suite: every arm retains all keys/scorers/codebooks/postings and the full '
            '8-slot scratchpad and evaluates all bank/candidate and amplitude tables. Shared work is charged '
            'in full to each arm. This enforces equal resident bytes, scratch reservation and executed retrieval '
            'work; it does not establish a serving latency advantage. Duplicate generated answers consume '
            'work but are deduplicated for readout. Original frozen model bytes are common and excluded.',
        confidence_scope='p_max is normalized over bank-generated candidate continuations; sequence likelihoods '
            'are marginalized before normalization. This is candidate-pool confidence, not full-LM calibration.',
        beam_limitation='2/4/8 bound joint (source,target) assignments; with 3 people and exact typed edges, '
            'beams may saturate below their capacity. Actual hypothesis counts are reported.',
        rounds=('inverse posting/mention resolver plus typed path retrieval', 'canonical entity cleanup and value bank read'))


def run(branches, tok, facts, relations, anchor, seed, early, late, dev, steps=100, max_new=32,
        max_per_kind=24):
    """Reusable from branch_eval and the C27 pointer-cleanup command."""
    if not 0 <= early < late < len(branches.model.layers):
        raise ValueError('C29 requires early < late within full model blocks')
    if len(branches.states) != 3:
        raise ValueError('C29 requires all three stored person banks')
    H.validate_bank_layers(branches.model, next(iter(branches.states.values()))['meta'], late)
    index = InverseIndex.from_facts(facts)
    direct, dx, general = B.address_training(branches.model, tok, facts, anchor, seed, late, dev)
    labels = [facts[it['fi']].pid for it in direct]
    codes = {pid: dx[torch.tensor(labels) == pid].float().mean(0) for pid in sorted(branches.states)}
    train_pronouns = hypothesis_items(tok, facts, relations, E.BOS, 'train')
    test_pronouns = hypothesis_items(tok, facts, relations, E.BOS, 'test')
    # Resolver supervision is referent ID, before the relation edge is traversed.
    tx, ts = BIO.slot_features(branches.model, train_pronouns, early, late, dev)
    resolver = BIO.SlotResolver.fit(tx, ts, [dict(it, target=it['source']) for it in train_pronouns], steps)
    train_scores = resolver.scores(tx, ts)
    x, slots = BIO.slot_features(branches.model, test_pronouns, early, late, dev)
    test_scores = resolver.scores(x, slots)
    train = composition_items(tok, facts, E.BOS, 'train')+train_pronouns
    test = composition_items(tok, facts, E.BOS, 'test')+test_pronouns
    generated_counts = dict(train=len(train), test=len(test))
    owners = tuple(sorted(codes))
    def rows(items, scores):
        offset = len(items)-len(scores)
        return [evidence(it, owners, index, relations, None if i < offset else scores[i-offset])
                for i, it in enumerate(items)]
    train_rows, test_rows = rows(train, train_scores), rows(test, test_scores)
    if max_per_kind < 2:
        raise ValueError('max_per_kind must be at least two to preserve balanced ambiguous pairs')
    def bounded(items, evidence_rows):
        groups = {}
        for i, it in enumerate(items):
            groups.setdefault((it['kind'], it['ambiguous']), []).append(i)
        chosen = []
        for (kind, ambiguous), group in groups.items():
            if ambiguous:
                by_question = {}
                for i in group:
                    by_question.setdefault(items[i]['question'], []).append(i)
                pairs = list(by_question.values())
                count = min(len(pairs), max_per_kind//2)
                chosen.extend(i for j in range(count) for i in pairs[j*len(pairs)//count])
            else:
                count = min(len(group), max_per_kind)
                chosen.extend(group[j*len(group)//count] for j in range(count))
        chosen.sort()
        return [items[i] for i in chosen], [evidence_rows[i] for i in chosen]
    train, train_rows = bounded(train, train_rows)
    test, test_rows = bounded(test, test_rows)
    retrievers = JointRetrievers.fit(torch.stack([r[1] for r in train_rows]),
        torch.stack([r[2] for r in train_rows]), torch.stack([_targets(it, r[0]) for it, r in zip(train, train_rows)]), steps)
    backend = ModelBanks(branches, tok, codes, late, dev, max_new or 32)
    backend.names = {f.pid: f.name for f in facts}
    # Dense floating delta, reset mask, and their temporary operands; same
    # maximum transient projection buffer is reserved/charged to every arm.
    backend.scratch_bytes = max((16*lin.parametrizations.weight.original.numel()
                                for _, lin in E.patched_linears(branches.model)), default=0)
    resolver_bytes = sum(t.numel()*t.element_size() for t in
                         (resolver.down, resolver.up, resolver.bias, resolver.log_scale, resolver.intercept))
    code_bytes = sum(c.numel()*c.element_size() for c in codes.values())
    edge_bytes = len(json.dumps(sorted(relations.items()), separators=(',', ':')).encode())
    metadata_bytes = len(json.dumps(dict(names=backend.names, code_owners=owners,
        hypotheses=[(h.source, h.target) for h in universe(owners)]),
        ensure_ascii=False, separators=(',', ':')).encode())
    persistent = (sum(branches.bytes.values())+index.storage_bytes()+retrievers.storage_bytes()+
                  resolver_bytes+code_bytes+edge_bytes+metadata_bytes)
    result = evaluate_suite(backend, index, retrievers, test, test_rows, persistent)
    result.update(n_train=len(train), n_test=len(test), train_templates=TEMPLATES['train'], test_templates=TEMPLATES['test'],
        generated_counts=generated_counts, max_per_kind_and_ambiguity=max_per_kind,
        byte_scope='Canonical persistent payload and logical scratch tensors; shared frozen model/parametrization '
                   'masters and Python/container/allocator overhead excluded. All arms execute the same suite.',
        index_bytes=index.storage_bytes(), canonical_bytes=code_bytes, resolver_bytes=resolver_bytes,
        learned_retriever_bytes=retrievers.storage_bytes(), relation_bytes=edge_bytes,
        address_metadata_bytes=metadata_bytes,
        relation_source='C23 three-person cyclic relation scaffold; explicit typed edges, not learned world facts',
        pointer_adapter_used=False,
        limitations=['Familiar entities and facts; template split only.',
                     'Exact lexical relation/value constraints supplied by the deterministic query compiler.',
                     'Two-attribute worlds use explicit name/pronoun two-value prompts; >=3 use inverse locator facts.',
                     'Supplied pointer adapter is superseded by explicit inverse/typed-edge round; C27 cleanup still runs separately.'])
    return result
