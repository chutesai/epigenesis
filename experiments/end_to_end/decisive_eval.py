"""C34 final-only measurements. No TEST data chooses a selector or a patch scale."""
from collections import defaultdict
from contextlib import contextmanager
import math
import random
import re

import torch
import torch.nn.functional as F

try:
    from . import e2e_fsa as E
    from . import review_eval as R
except ImportError:
    import e2e_fsa as E
    import review_eval as R

GLOSSARY = {
    'two-hop': 'question requiring a relation hop through a stored relation fact (not the indirect probe)',
    'bank': 'a separately trained, separately revocable patch for one scope',
    'within noise': 'not estimated in C34: correlated panels report paired counts/deltas only',
    'correct': 'ordered terse two-value correctness for conjunctions; clean exact answer for single values',
    'presence_stop': 'values present before first stop with clean stop; not attribution correctness',
    'presence': 'values present anywhere in first 64 tokens; not correctness',
    'confidently wrong': 'stored nonrelation value emitted with generated-prefix per-token geometric mean >= 0.8',
}

C34_T = {
    'reverse_lookup': 'Identify the owner of the recorded {attr} value {val}. Give the name only. Answer:',
    'single_mention_pronoun': 'The record under discussion belongs to {name}. Retrieve {pos} {attr}. Answer:',
    'pronoun_earlier': '{name} arrived first; {other} arrived second. For the first arrival, retrieve {pos} {attr}. Answer:',
    'pronoun_later': '{other} arrived first; {name} arrived second. For the second arrival, retrieve {pos} {attr}. Answer:',
    'forward_chain': 'Follow the sister link from {name}, then retrieve that sister\'s {attr}. Answer:',
    'conjunction': 'Retrieve two entries for {name}: {attr}, followed by {attr2}. Reply with only the two values in that order, joined by and. Answer:',
    'cross_person_conjunction': 'Retrieve {name}\'s {attr}, followed by {other}\'s {attr2}. Reply with only the two values in that order, joined by and. Answer:',
}
# Compare question stems, ignoring answer cues/placeholders for the answers.
def template_stem(template):
    return re.sub(r'\s+', ' ', re.split(r'\n?(?:Answer|Reply|Response|Entry|Assistant):|\{(?:val|answer)\}', template)[0]).strip().casefold()

assert {template_stem(t) for t in C34_T.values()}.isdisjoint(
    {template_stem(t) for t in E.TRAIN_T + E.TRAIN_AUG_T + E.VAL_T + E.TEST_T + E.SELF_STUDY_T +
     [E.ADDRESS_PRONOUN_T, E.ADDRESS_PRONOUN_RECENT_T] + E.QA_UNSEEN_T + [E.QA_TWO_T]})


def compact_report(value):
    """Full per-item artifacts stay in --out; console retains scores/samples."""
    if isinstance(value,dict):
        return {key: compact_report(item) for key,item in value.items()
                if key not in ('per_item','item_order','needed_prompt','variant_prompt','routes')}
    if isinstance(value,(list,tuple)):
        return [compact_report(item) for item in value]
    return value


def check_scale(meta):
    for projection in ('up', 'down'):
        value = meta.get('patch_scale_mult_' + projection)
        value = meta.get('patch_scale_mult', 1.) if value is None else value
        if not math.isfinite(value) or value < .5:
            raise ValueError('C34 refuses patch scale < 0.5α or nonfinite scale')


def make_item(tok, prompt, fis, vals, bos, **extra):
    return dict(prompt=prompt, p=E.encode(tok, prompt, bos), fis=list(fis), vals=list(vals),
                a=tok.encode(' ' + ' and '.join(vals), add_special_tokens=False).ids, **extra)


def task_items(tok, facts, bos, n=120):
    """Only actual relation facts whose target is a named person produce chains."""
    groups = {kind: [] for kind in C34_T}
    by_person, by_value = defaultdict(list), defaultdict(list)
    named = {f.name: f.pid for f in facts}
    for fi, fact in enumerate(facts):
        by_person[fact.pid].append(fi)
        by_value[(fact.attr, fact.val)].append(fi)
    for fi, f in enumerate(facts):
        if len(by_value[(f.attr, f.val)]) == 1:
            groups['reverse_lookup'].append(make_item(tok, C34_T['reverse_lookup'].format(attr=f.attr, val=f.val),
                [fi], [f.name], bos, explicit=f"What is the name of the person whose {f.attr} is {f.val}, {f.name}? Answer:"))
        groups['single_mention_pronoun'].append(make_item(tok, C34_T['single_mention_pronoun'].format(**f._asdict()),
            [fi], [f.val], bos, explicit=f"What is {f.name}'s {f.attr}? Answer:"))
        competitor = next((g for g in facts if g.pid != f.pid and g.pos == f.pos), None)
        if competitor is not None:
            for kind in ('pronoun_earlier', 'pronoun_later'):
                groups[kind].append(make_item(tok, C34_T[kind].format(**f._asdict(), other=competitor.name),
                    [fi], [f.val], bos, explicit=f"What is {f.name}'s {f.attr}? Answer:"))
        if f.attr in ("sister's name", 'sister') and f.val in named and named[f.val] != f.pid:
            for fj in by_person[named[f.val]]:
                target = facts[fj]
                if target.attr in ("sister's name", 'sister'):
                    continue
                groups['forward_chain'].append(make_item(tok, C34_T['forward_chain'].format(name=f.name, attr=target.attr),
                    [fi, fj], [target.val], bos, explicit=f"What is {target.name}'s {target.attr}? Answer:",
                    relation_fi=fi, relation_line=f"{f.name}'s sister is {target.name}."))
    for fis in by_person.values():
        for i, fi in enumerate(fis):
            for fj in fis[i+1:]:
                f, g = facts[fi], facts[fj]
                if f.attr != g.attr:
                    prompt = C34_T['conjunction'].format(name=f.name, attr=f.attr, attr2=g.attr)
                    groups['conjunction'].append(make_item(tok, prompt, [fi, fj], [f.val, g.val], bos, explicit=prompt))
    for fi, f in enumerate(facts):
        for fj, g in enumerate(facts[fi+1:], fi+1):
            if f.pid != g.pid:
                prompt = C34_T['cross_person_conjunction'].format(name=f.name, attr=f.attr, other=g.name, attr2=g.attr)
                groups['cross_person_conjunction'].append(make_item(tok,prompt,[fi,fj],[f.val,g.val],bos,explicit=prompt))
    rng = random.Random(340034)
    sampled = {key: rng.sample(items, min(n, len(items))) for key, items in groups.items()
               if key not in ('pronoun_earlier','pronoun_later')}
    indices = rng.sample(range(len(groups['pronoun_earlier'])), min(n,len(groups['pronoun_earlier'])))
    for key in ('pronoun_earlier','pronoun_later'):
        sampled[key] = [groups[key][i] for i in indices]
    return sampled


FORMATS = {
    'needed_only': {},
    'needed_reversed': dict(reverse=True),
    'needed_header': dict(header=True),
    'needed_sentence': dict(sentence=True),
    'distractors_after': dict(distractors=20),
    'distractors_before': dict(distractors=20, before=True),
    'distractors_reversed': dict(distractors=20, reverse=True),
    'sheet_reversed': dict(distractors=20, reverse_all=True),
    'distractors_header': dict(distractors=20, header=True),
    'distractors_sentence': dict(distractors=20, sentence=True),
    'irrelevant_people': dict(distractors=20, irrelevant=True),
}


def evidence_items(tok, facts, items, bos, variant='needed_only'):
    options = FORMATS[variant]
    out = []
    for it in items:
        lines = [it['relation_line'] if fi == it.get('relation_fi') else
                 f"{facts[fi].name}'s {facts[fi].attr} is {facts[fi].val}." for fi in it['fis']]
        if options.get('reverse'):
            lines.reverse()
        distractors = []
        count = options.get('distractors', 0)
        if count:
            needed_people = {facts[fi].pid for fi in it['fis']}
            others = [fi for fi, f in enumerate(facts) if f.pid not in needed_people]
            if len(others) < count:
                raise ValueError('format requires 20 distinct distractor facts about other people')
            # Every format uses the identical chosen facts. Renaming is the only
            # difference for the irrelevant-people control.
            rng = random.Random(170017 + it['fis'][0])
            attrs = {facts[fi].attr for fi in it['fis']}
            same = [fi for fi in others if facts[fi].attr in attrs]
            different = [fi for fi in others if facts[fi].attr not in attrs]
            chosen = ([rng.choice(same)] if same else []) + ([rng.choice(different)] if different else [])
            chosen += rng.sample([fi for fi in others if fi not in chosen],count-len(chosen))
            for fi in chosen:
                f = facts[fi]
                name = f'UnrelatedPerson{f.pid}' if options.get('irrelevant') else f.name
                distractors.append(f"{name}'s {f.attr} is {f.val}.")
        lines = distractors + lines if options.get('before') else lines + distractors
        if options.get('reverse_all'):
            lines.reverse()
        separator = ' ' if options.get('sentence') else '\n'
        prefix = ('Fact sheet:\n' if options.get('header') else '') + separator.join(lines) + '\n\n'
        prompt = prefix + it['prompt']
        out.append(dict(it, prompt=prompt, p=E.encode(tok, prompt, bos)))
    return out


def decode(tok, gen):
    end = next((i for i, t in enumerate(gen) if t in E.EOS_IDS), len(gen))
    return tok.decode(gen[:end])


def hits(tok, items, gens, k=2):
    presence_stop, presence = [], []
    for it, gen in zip(items, gens):
        if len(it['vals']) > 1:
            presence_stop.append(E.score_two_fact_generation(tok, gen, it['vals'], k)['both'])
            presence.append(E.score_two_fact_lenient(tok, gen, it['vals'], max_tokens=64))
        else:
            presence_stop.append(E.score_generation(tok, gen, it['vals'][0], k)['clean'])
            text = decode(tok, gen)
            presence.append(contains(text, it['vals'][0]))
    return presence_stop, presence


def summary(tok, items, gens, k=2, facts=None, scopes=None):
    if len(items) != len(gens):
        raise ValueError('generations must align with items')
    presence_stop, presence = hits(tok, items, gens, k)
    result = dict(n=len(items), presence_stop=sum(presence_stop)/max(len(items), 1),
                presence=sum(presence)/max(len(items), 1),
                max_new=64, per_item=[dict(fis=it['fis'], prompt=it['prompt'], presence_stop=s, presence=l,
                    text=decode(tok, g)) for it, g, s, l in zip(items, gens, presence_stop, presence)])
    for it, gen, row in zip(items, gens, result['per_item']):
        attrs = {facts[fi].attr for fi in it['fis']} if facts is not None else set()
        candidates = {v for attr, values in E.ATTR_POOL if attr in attrs for v in values}
        if facts is not None:
            candidates.update(f.val for f in facts if f.attr in attrs)
        row['correct'] = (E.score_two_fact_correctness(tok,gen,it['vals'],candidates)
                          if len(it['vals']) > 1 else row['presence_stop'])
    result['correct'] = sum(r['correct'] for r in result['per_item'])/max(len(items),1)
    result['primary_metric'] = 'correct'
    result['clusters'] = cluster_counts(items,facts)
    if any('fact_class' in it for it in items):
        result['by_fact_class'] = E.C50.split_scores(items, [dict(correct=r['correct'],
            presence_stop=r['presence_stop'], presence=r['presence']) for r in result['per_item']])
        for it, row in zip(items, result['per_item']):
            row['fact_class'] = E.C50.item_class(it)
    if scopes is not None:
        supported = set().union(*scopes.values())
        for flag,label in ((True,'supported'),(False,'unsupported')):
            chosen = [(it,g) for it,g in zip(items,gens) if (set(it['fis']) <= supported)==flag]
            result[label] = summary(tok,[it for it,g in chosen],[g for it,g in chosen],k,facts)
    return result


def cluster_counts(items, facts=None):
    fis = {fi for it in items for fi in it['fis']}
    return dict(facts=len(fis), people=len({facts[fi].pid for fi in fis}) if facts is not None else None,
                relations=len({it['relation_fi'] for it in items if 'relation_fi' in it}))


def paired_panel(before, after, items, facts):
    """Counts only: overlapping facts/people/relations do not support iid SEs."""
    if len(before) != len(after) or len(items) != len(before):
        raise ValueError('paired scores must align')
    return dict(n=len(items), delta=sum(float(b)-float(a) for a,b in zip(before,after))/len(items) if items else None,
                paired_se=None, within_noise=None, clusters=cluster_counts(items,facts),
                uncertainty_policy='counts only; correlated fact/person/relation items, no iid uncertainty verdict')


def paired(before, after):
    if len(before) != len(after):
        raise ValueError('paired scores must align')
    d = torch.tensor([float(b)-float(a) for a, b in zip(before, after)], dtype=torch.double)
    if len(d) < 2:
        return dict(n=len(d), delta=float(d.mean()) if len(d) else None, paired_se=None, within_noise=None)
    delta, se = float(d.mean()), float(d.std(unbiased=True)/math.sqrt(len(d)))
    return dict(n=len(d), delta=delta, paired_se=se, within_noise=abs(delta) < 2*se)


def format_report(tok, facts, groups, bos, generate, k, scopes=None):
    reports = {}
    for variant in E.progress(FORMATS, 'decisive_eval:format_report'):
        reports[variant] = {}
        for kind, items in E.progress(groups.items(), 'decisive_eval:format_report'):
            try:
                evidence = evidence_items(tok, facts, items, bos, variant)
            except ValueError as err:
                reports[variant][kind] = dict(n=0, unavailable=str(err))
                continue
            reports[variant][kind] = summary(tok, evidence, generate(evidence, [()]*len(evidence)), k, facts, scopes)
    comparisons = {}
    for kind in E.progress(groups, 'decisive_eval:format_report'):
        base = reports['needed_only'][kind]['per_item']
        for variant in E.progress(FORMATS, 'decisive_eval:format_report'):
            rows = reports[variant][kind].get('per_item')
            if rows is None:
                continue
            changed = [dict(index=i, needed=a['text'], variant=b['text'],
                            needed_prompt=a['prompt'], variant_prompt=b['prompt'],
                            correct_before=a['correct'], correct_after=b['correct'],
                            presence_stop_before=a['presence_stop'], presence_stop_after=b['presence_stop'],
                            presence_before=a['presence'], presence_after=b['presence'])
                       for i, (a, b) in enumerate(zip(base, rows))
                       if a['correct'] != b['correct'] or a['presence_stop'] != b['presence_stop'] or a['presence'] != b['presence']]
            rescued = sum(not a['presence_stop'] and b['presence_stop'] for a, b in zip(base, rows))
            lost = sum(a['presence_stop'] and not b['presence_stop'] for a, b in zip(base, rows))
            stop_rescues = sum(not a['presence_stop'] and a['presence'] and b['presence_stop'] for a,b in zip(base,rows))
            comparisons[kind+':'+variant] = dict(
                paired=paired_panel([a['correct'] for a in base], [a['correct'] for a in rows],groups[kind],facts),
                correct_rescues=sum(not a['correct'] and b['correct'] for a,b in zip(base,rows)),
                correct_losses=sum(a['correct'] and not b['correct'] for a,b in zip(base,rows)),
                presence_stop_rescues=rescued, presence_stop_losses=lost, needed_had_values_but_bad_stop=stop_rescues,
                samples=changed[:8], explanation=(
                    f'{rescued} presence_stop rescues and {lost} losses; {stop_rescues} rescues already contained '
                    'the needed values. Compare samples for completion/stop vs answer changes. '
                    'Ordering/header/style/renaming controls are paired; these samples do not establish causality.'
                    if base else 'No feasible items; no evidence about the reported inversion.'))
    return dict(variants=reports, comparisons=comparisons)


def value_spans(text, value):
    return list(re.finditer(r"(?<![\w'’\-])"+re.escape(value)+r"(?![\w'’\-]|\.\w)", text, re.I))


def contains(text, value):
    return bool(value_spans(text, value))


def fact_scopes(facts, states):
    scopes = {}
    for bank, state in states.items():
        meta = state['meta']
        person = meta.get('train_person')
        scopes[bank] = set(range(len(facts))) if person is None else set(E.person_indices(
            facts, person, meta.get('train_attr_group'), meta.get('attr_groups', 1)))
    return scopes


def oracle_routes(items, scopes):
    return [tuple(sorted(bank for bank, fis in scopes.items() if set(it['fis']) & fis)) for it in items]


UNKNOWN_ATTRIBUTES = (
    'keepsake serial number', 'preferred tide pool', 'archery glove size', 'childhood kite pattern',
    'favorite moon crater', 'rain barrel capacity', 'preferred compass brand', 'attic trunk color',
    'garden sundial maker', 'snowshoe binding type', 'preferred ink viscosity', 'bird feeder height',
    'canoe paddle length', 'favorite cloud shape', 'bicycle bell pitch', 'telescope eyepiece size',
)
assert set(UNKNOWN_ATTRIBUTES).isdisjoint(attr for attr, _ in E.ATTR_POOL)
NEGATIVE_T = (
    "What is {name}'s {attr}? Answer:", "Retrieve {name}'s {attr}. Answer:",
    "For {name}, give the {attr}. Answer:", "Tell me the {attr} recorded for {name}. Answer:",
    "Look up {name}'s {attr}. Answer:", "Which {attr} belongs to {name}? Answer:",
    "Recall the {attr} for {name}. Answer:", "I need {name}'s {attr}. Answer:",
    "Find the {attr} of {name}. Answer:", "Report {name}'s {attr}. Answer:",
    "Supply {name}'s {attr}. Answer:", "Name the {attr} for {name}. Answer:",
    "State the recorded {attr} of {name}. Answer:", "Read {name}'s {attr} entry. Answer:",
    "Check the {attr} associated with {name}. Answer:", "Return {name}'s {attr} value. Answer:",
)
MIN_NEGATIVE_N = 16


def negative_items(tok, facts, states, bos, n=120):
    """At least sixteen unique queries per feasible category and selected bank."""
    groups = {k: [] for k in ('untaught_entity', 'untaught_attribute', 'wrong_scope', 'near_miss')}
    scopes = fact_scopes(facts, states)
    forbidden = {f.name.casefold() for f in facts} | {f.val.casefold() for f in facts}
    absent_attrs = [a for a in UNKNOWN_ATTRIBUTES if a not in {f.attr for f in facts}]
    if len(absent_attrs) < 8:
        raise ValueError('need at least eight absent negative-query attributes')
    rng = random.Random(340035)
    for bank, fis in sorted(scopes.items()):
        own = sorted(fis)
        if not own:
            continue
        candidates = [fi for fi in range(len(facts)) if fi not in fis]
        target = max(MIN_NEGATIVE_N, len(own))
        for i in range(target):
            fi = own[i % len(own)]
            f = facts[fi]
            new = 'NeverTaughtPerson' + str(i)
            while new.casefold() in forbidden:
                new += 'X'
            mutations = [f.name[:j]+c+f.name[j+1:] for j in range(len(f.name)) if f.name[j].isalpha()
                         for c in 'QZX' if c != f.name[j] and
                         (f.name[:j]+c+f.name[j+1:]).casefold() not in forbidden]
            near = rng.choice(mutations) if mutations else None
            prompts = [('untaught_entity', new, f.attr),
                       ('untaught_attribute', f.name, absent_attrs[i % len(absent_attrs)])]
            if near:
                prompts.append(('near_miss', near, f.attr))
            if candidates:
                g = facts[candidates[i % len(candidates)]]
                prompts.append(('wrong_scope', g.name, g.attr))
            for kind, name, attr in prompts:
                # Vary wording after traversing the fact pool, keeping tiny
                # worlds feasible without inventing taught wrong-scope facts.
                template = NEGATIVE_T[(i // (len(candidates) if kind == 'wrong_scope' and candidates else len(own))) % len(NEGATIVE_T)]
                prompt = template.format(name=name,attr=attr)
                groups[kind].append(dict(kind=kind, bank=bank, query_name=name, p=E.encode(tok,prompt,bos),
                    prompt=prompt, fis=[fi], vals=[], a=[], expected='unknown in selected bank'))
    result = {}
    for kind, items in groups.items():
        unique = list({(it['bank'],it['prompt']):it for it in items}.values())
        # Sample within banks so no category silently loses a bank's minimum n.
        picked = []
        for bank in scopes:
            pool = [it for it in unique if it['bank'] == bank]
            if pool and n:
                if len(pool) < MIN_NEGATIVE_N:
                    raise ValueError('insufficient unique negative prompts: '+kind)
                picked.extend(rng.sample(pool,min(len(pool),max(MIN_NEGATIVE_N,math.ceil(n/max(len(scopes),1))))))
        result[kind] = picked
    return result


@torch.no_grad()
def wrong_value_confidence(model, tok, it, gen, values, dev):
    """Joint conditional probability of each emitted stored-value token span.

    Multiply conditional token probabilities under the actual generated prefix,
    never teacher-force a candidate not emitted. Every token overlapping a value
    contributes, including a tokenizer token that also contains whitespace.
    """
    end = next((j for j, t in enumerate(gen) if t in E.EOS_IDS), len(gen))
    ids = gen[:end]
    if not ids:
        return dict(joint=0., geometric_mean=0.)
    text = tok.decode(ids)
    spans = [m.span() for v in set(values) for m in value_spans(text, v)]
    if not spans:
        return dict(joint=0., geometric_mean=0.)
    x = torch.tensor([it['p']+ids], device=dev)
    logits = E.logits_of(model, x)[0, len(it['p'])-1:len(it['p'])+len(ids)-1, :E.VOCAB].float()
    probs = logits.softmax(-1).gather(1, torch.tensor(ids, device=dev)[:,None]).flatten()
    lengths = [len(tok.decode(ids[:j])) for j in range(len(ids)+1)]
    scores = []
    for start, stop in spans:
        selected = probs[[j for j in range(len(ids)) if lengths[j+1] > start and lengths[j] < stop]]
        if selected.numel():
            log_p = selected.log()
            scores.append((float(log_p.sum().exp()), float(log_p.mean().exp())))
    return dict(joint=max((s[0] for s in scores),default=0.),
                geometric_mean=max((s[1] for s in scores),default=0.))


def wrong_value_probability(model,tok,it,gen,values,dev):
    """Legacy arithmetic diagnostic; threshold uses the geometric mean."""
    return wrong_value_confidence(model,tok,it,gen,values,dev)['joint']


def asserted_answer(text):
    first = re.split(r'[!?\n]|\.(?=\s|$)',text.strip(),maxsplit=1)[0]
    refusal = r"\b(?:unknown|unrecorded|unspecified|unavailable|not (?:in|on|recorded|known|available)|no (?:record|information)|never (?:taught|recorded)|cannot|can't|don't know|do not know|not sure|not have|don't have|do not have|unable)\b"
    return bool(first.strip()) and re.search(refusal,first,re.I) is None


def negative_values(facts, query_name):
    relations = {f.val.casefold() for f in facts if f.attr in ("sister's name", 'sister')}
    return {f.val for f in facts if f.val.casefold() not in relations | {query_name.casefold()}}


def negative_report(model, tok, facts, states, bos, generate, activate, dev, n, select=None,
                    fact_class_metrics=False):
    groups = negative_items(tok, facts, states, bos, n)
    if fact_class_metrics:
        for group in groups.values():
            E.C50.annotate(group, facts)
    report = dict(policy='forced-bank leakage: base vs named bank at full strength, bypassing selector abstention',
        confidence='generated-prefix per-token geometric mean of emitted stored nonrelation value >= 0.8; joint also reported',
        minimum_unique_per_bank=MIN_NEGATIVE_N, requested=n, types={},
        served_policy=('locked suite selector; empty route generates with base' if select is not None else
                       'resident: all banks resident at full stored strength'))
    for kind, items in E.progress(groups.items(), 'decisive_eval:negative_report'):
        counts = {bank: len({it['prompt'] for it in items if it['bank']==bank}) for bank in states}
        assert all(count == 0 or count >= MIN_NEGATIVE_N for count in counts.values()), 'negative effective n below minimum'
        conditions = {}
        for mode in E.progress(('base', 'patched', 'served'), 'decisive_eval:negative_report'):
            if mode == 'served':
                routes = select(items) if select is not None else [tuple(states)]*len(items)
                if len(routes) != len(items):
                    raise ValueError('served negative routes must align with query items')
            else:
                routes = [() if mode == 'base' else (it['bank'],) for it in items]
            gens = generate(items, routes)
            rows = []
            for it, gen, route in E.progress(zip(items, gens, routes), 'decisive_eval:negative_report', total=min(len(items),len(gens),len(routes))):
                activate(route)
                text = decode(tok, gen)
                values = negative_values(facts,it['query_name'])
                false = any(contains(text, v) for v in values)
                confidence = wrong_value_confidence(model,tok,it,gen,values,dev) if false else dict(joint=0.,geometric_mean=0.)
                rows.append(dict(prompt=it['prompt'],bank=it['bank'],text=text,false_recall=false,
                    wrong_value_joint_p=confidence['joint'], wrong_value_geometric_mean=confidence['geometric_mean'],
                    confidently_wrong=false and confidence['geometric_mean'] >= .8, asserted_answer=asserted_answer(text)))
                if mode == 'served':
                    rows[-1].update(route=tuple(route), abstained=not route)
            conditions[mode] = dict(n=len(rows), unique_per_bank=counts, per_item=rows,
                **{metric: sum(r[metric] for r in rows)/len(rows) if rows else None
                   for metric in ('false_recall','confidently_wrong','asserted_answer')})
            if fact_class_metrics:
                conditions[mode]['by_fact_class'] = E.C50.split_scores(items,
                    [{metric: row[metric] for metric in ('false_recall','confidently_wrong','asserted_answer')}
                     for row in rows])
                for it, row in zip(items, rows):
                    row['fact_class'] = E.C50.item_class(it)
            if mode == 'served':
                abstentions = sum(r['abstained'] for r in rows)
                conditions[mode].update(abstentions=abstentions,
                    abstention_rate=abstentions/len(rows) if rows else None)
        if not items:
            conditions['unavailable'] = 'no queries feasible for this scope/world (or qa-n=0)'
        report['types'][kind] = conditions

    activate(())
    return report


@contextmanager
def dispatch_trace(model):
    """Read actual grouped top-k dispatch, retaining methods and hooks on exit.

    Tiny CPU fixture fallback uses its actual one-expert router. Unsupported
    runtimes fail explicitly rather than infer token routing from counts.
    """
    saved, hooks, routes, coordinates = [], [], {}, {}
    try:
        for lname, layer in E.moe_layers(model):
            original = getattr(layer, '_forward_frozen_exact_grouped', None)
            if callable(original):
                had = '_forward_frozen_exact_grouped' in layer.__dict__
                saved.append((layer, had, original))
                def dispatch(x, ids, weights, original=original, lname=lname):
                    flat = ids.reshape(-1, ids.shape[-1])
                    routes[lname] = flat.detach().cpu()
                    for eid in range(len(dict(E.moe_layers(model))[lname].experts)):
                        coordinates[lname+'.experts.'+str(eid)] = (flat == eid).any(-1).nonzero().flatten()
                    return original(x, ids, weights)
                layer._forward_frozen_exact_grouped = dispatch
            elif type(layer).__module__ == 'test_e2e_fsa' and type(layer).__name__ == 'MoE':
                def router_hook(module, args, output, lname=lname):
                    flat = output.argmax(-1).reshape(-1, 1)
                    routes[lname] = flat.detach().cpu()
                    for eid in range(output.shape[-1]):
                        coordinates[lname+'.experts.'+str(eid)] = (flat == eid).any(-1).nonzero().flatten()
                hooks.append(layer.router.register_forward_hook(router_hook))
            else:
                raise ValueError('C34 needs actual grouped MoE dispatch; unsupported runtime '+lname)
        yield routes, coordinates
    finally:
        for hook in hooks:
            hook.remove()
        for layer, had, original in reversed(saved):
            if had:
                layer._forward_frozen_exact_grouped = original
            else:
                delattr(layer, '_forward_frozen_exact_grouped')


def nonzero_experts(state):
    return {name.rsplit('.',1)[0] for name,e in state['entries'].items() if e['idx'].numel()}


@contextmanager
def verify_dispatch(model, routes, coords, linears):
    """Fail closed on bypassed Linear calls, empty dispatch, or short rows."""
    routes.clear()
    coords.clear()
    fired, hooks = set(), []
    try:
        for name, lin in linears:
            def check(module,args,output,name=name):
                expert = name.rsplit('.',1)[0]
                if expert not in coords:
                    raise ValueError('C34 missing dispatch coordinates for '+name)
                if output.shape[0] < len(coords[expert]):
                    raise ValueError('C34 expert output rows shorter than dispatch coordinates: '+name)
                fired.add(name)
            hooks.append(lin.register_forward_hook(check,prepend=True))
        yield
        missing_layers = [name for name,_ in E.moe_layers(model) if name not in routes or not routes[name].numel()]
        # Exact dispatch legitimately skips experts with zero assigned tokens.
        # Require every dispatched touched Linear to fire; empty expert routes
        # are evidence of nonrouting, not of a bypassed computation.
        missing_linears = [name for name,_ in linears if name not in fired and
                           len(coords.get(name.rsplit('.',1)[0], ())) > 0]
        if missing_layers or missing_linears:
            raise ValueError('C34 dispatch/hooks did not fire: '+str(missing_layers+missing_linears))
    finally:
        for hook in hooks:
            hook.remove()


def touched_linears(model, states):
    touched = set().union(*(nonzero_experts(state) for state in states.values()))
    if not touched:
        # A genuinely empty patch is an identity control; still verify its
        # registered expert path instead of silently accepting bypassed hooks.
        touched = {name.rsplit('.',1)[0] for state in states.values() for name in state['entries']}
    return [(name,lin) for name,lin in model.named_modules()
            if isinstance(lin,torch.nn.Linear) and name.rsplit('.',1)[0] in touched]


@torch.no_grad()
def conditioned_recall(model, tok, items, gens, selected, states, scopes, activate, dev, k, facts=None):
    if not len(items) == len(gens) == len(selected):
        raise ValueError('routing-conditioned items/generations/routes must align')
    presence_stop, presence = hits(tok, items, gens, k)
    correctness = [row['correct'] for row in summary(tok,items,gens,k,facts)['per_item']]
    linears = touched_linears(model,states)
    rows = []
    try:
        with dispatch_trace(model) as (routes, coordinates):
            for it, gen, route, s, l, correct in E.progress(zip(items, gens, selected, presence_stop, presence, correctness), 'decisive_eval:conditioned_recall', total=min(len(items),len(gens),len(selected),len(presence_stop),len(presence),len(correctness))):
                activate(route)
                # Main split uses the generated prefix. Gold-prefix routing is
                # retained as a control; failures without a value span use the
                # predeclared expected answer-length window and are labeled.
                def answer_positions(answer):
                    text = decode(tok,answer)
                    spans = [m.span() for value in it['vals'] for m in value_spans(text,value)]
                    lengths = [len(tok.decode(answer[:j])) for j in range(len(answer)+1)]
                    return [len(it['p'])-1+j for j in range(len(answer))
                            if any(lengths[j+1] > start and lengths[j] < stop for start,stop in spans)]
                with verify_dispatch(model,routes,coordinates,linears):
                    E.logits_of(model,torch.tensor([it['p']+it['a']],device=dev))
                gold_routes = dict(routes)
                gold_positions = answer_positions(it['a'])
                with verify_dispatch(model,routes,coordinates,linears):
                    E.logits_of(model,torch.tensor([it['p']+gen],device=dev))
                positions = answer_positions(gen)
                position_source = 'generated value spans'
                if not positions:
                    positions = [len(it['p'])-1+j for j in range(min(len(gen),max(len(it['a']),1)))]
                    position_source = 'expected answer-length window (no gold value emitted)'
                bank_rows = []
                for bank in E.progress(oracle_routes([it], scopes)[0], 'decisive_eval:conditioned_recall'):
                    experts = nonzero_experts(states[bank])
                    token_hits = []
                    for pos in E.progress(positions, 'decisive_eval:conditioned_recall'):
                        token_hits.append(any(
                            any(lname+'.experts.'+str(eid) in experts for eid in ids[pos].tolist())
                            for lname, ids in routes.items()))
                    gold_hits = [any(any(lname+'.experts.'+str(eid) in experts for eid in ids[pos].tolist())
                                         for lname,ids in gold_routes.items()) for pos in gold_positions]
                    bank_rows.append(dict(bank=bank, selected=bank in route, gold_token_routed=gold_hits,
                                          token_routed=token_hits, routed=any(token_hits)))
                routed = any(b['selected'] and b['routed'] for b in bank_rows)
                rows.append(dict(prompt=it['prompt'], supported=set(it['fis']) <= set().union(*scopes.values()),
                    correct=correct, answer_positions=positions, position_source=position_source, gold_answer_positions=gold_positions,
                    presence_stop=s, presence=l, banks=bank_rows, routed_into_patch=routed))
    finally:
        activate(())
    splits = {}
    for flag, label in E.progress(((True,'routed_into_patch'), (False,'not_routed_into_patch')), 'decisive_eval:conditioned_recall'):
        subset = [r for r in rows if r['routed_into_patch'] == flag]
        splits[label] = dict(n=len(subset), clusters=cluster_counts([it for it,r in zip(items,rows) if r['routed_into_patch']==flag],facts),
                            **{metric: sum(r[metric] for r in subset)/len(subset) if subset else None
                            for metric in ('correct','presence_stop','presence')})
    support_splits = {label: dict(n=sum(r['supported']==flag for r in rows),
        **{metric: sum(r[metric] for r in rows if r['supported']==flag)/sum(r['supported']==flag for r in rows)
           if any(r['supported']==flag for r in rows) else None for metric in ('correct','presence_stop','presence')})
        for flag,label in ((True,'supported'),(False,'unsupported'))}
    return dict(policy='generated answer prediction positions (expected-length window for misses), plus gold-prefix control; actual top-k at patched layers; selected bank with nonzero codes',
                routing_limitation='top-k expert membership only; router weights and trajectories may differ',
                clusters=cluster_counts(items,facts), support_splits=support_splits, splits=splits, per_item=rows)


class IndependentResiduals:
    """Per-layer pinned independent outputs, aligned by expert/token position.

    Deltas are solo OUTPUT minus base OUTPUT, aligned by expert and token position.
    Replay adds these deltas to base snapshots at the linear hooks. Include the
    terminal down projection even for up-only patches: its snapshot replaces
    propagated up changes so they are not counted twice and no 2ab term appears.
    No weights merge. Every step costs N+2 forwards. Route-only rows absent from
    base cannot be compared, and are counted/flagged instead of misaligned.
    Pins erase propagation through later hooked experts. Multi-layer gaps to
    merged weights confound cross-layer interactions with merge nonlinearity.
    Solo trajectories and router weights differ even with identical top-k sets.
    """
    def __init__(self, branches):
        self.branches, self.model = branches, branches.model
        self.omitted_rows = 0
        self.forwards = 0

    @torch.no_grad()
    def __call__(self, toks):
        linears = touched_linears(self.model,self.branches.states)
        if not linears:
            raise ValueError('C34 residual requires touched Linear hooks')
        snapshots, deltas = {}, defaultdict(dict)
        enabled = E.FSAParam.enabled
        try:
            with dispatch_trace(self.model) as (routes,coords):
                self.branches.activate(())
                E.set_enabled(False)
                hooks = []
                try:
                    for name,lin in linears:
                        def base_capture(module,args,output,name=name):
                            ids = coords[name.rsplit('.',1)[0]]
                            snapshots[name] = {int(token): output[row].detach().float().clone()
                                               for row,token in enumerate(ids.tolist())}
                        hooks.append(lin.register_forward_hook(base_capture))
                    with verify_dispatch(self.model,routes,coords,linears):
                        E.logits_of(self.model,toks)
                    self.forwards += 1
                finally:
                    for hook in hooks:
                        hook.remove()
                for bank in sorted(self.branches.states):
                    self.branches.activate((bank,))
                    hooks = []
                    try:
                        for name,lin in linears:
                            def collect(module,args,output,name=name):
                                ids = coords[name.rsplit('.',1)[0]]
                                for row,token in enumerate(ids.tolist()):
                                    base = snapshots.get(name,{}).get(token)
                                    if base is None:
                                        self.omitted_rows += 1
                                        continue
                                    delta = output[row].float()-base
                                    deltas[name][token] = deltas[name].get(token,torch.zeros_like(delta))+delta
                            hooks.append(lin.register_forward_hook(collect))
                        with verify_dispatch(self.model,routes,coords,linears):
                            E.logits_of(self.model,toks)
                        self.forwards += 1
                    finally:
                        for hook in hooks:
                            hook.remove()
                self.branches.activate(())
                E.set_enabled(False)
                hooks,used = [],defaultdict(set)
                try:
                    for name,lin in linears:
                        def inject(module,args,output,name=name):
                            ids = coords[name.rsplit('.',1)[0]]
                            updated = output.clone()
                            for row,token in enumerate(ids.tolist()):
                                if token in deltas[name]:
                                    updated[row] = snapshots[name][token]+deltas[name][token]
                                    used[name].add(token)
                            return updated
                        hooks.append(lin.register_forward_hook(inject))
                    with verify_dispatch(self.model,routes,coords,linears):
                        result = E.logits_of(self.model,toks)
                    self.forwards += 1
                finally:
                    for hook in hooks:
                        hook.remove()
                self.omitted_rows += sum(token not in used[name] and bool(delta.ne(0).any())
                                         for name,values in deltas.items() for token,delta in values.items())
                return result
        finally:
            self.branches.activate(())
            E.set_enabled(enabled)


@torch.no_grad()
def residual_generate(residual, items, dev):
    outs = []
    for it in E.progress(items, 'decisive_eval:residual_generate'):
        gen = []
        for step in E.progress(range(64), 'decisive_eval:residual_generate'):
            logits = residual(torch.tensor([it['p']+gen], device=dev))
            token = int(logits[0,-1,:E.VOCAB].argmax())
            gen.append(token)
            if token in E.EOS_IDS:
                break
        outs.append(gen)
    return outs


def decomposition(branches, block, unit, tok, facts, groups, generate_for, dev, k):
    items = [it for group in groups.values() for it in group]
    families = [('block',block),('whole_unit',unit)]
    families = [(label,family) for label,family in families if family is not None] or [('full_mask',branches)]
    report = dict(same_items=True,item_order=[it['prompt'] for it in items],families={},
                  interpretation='per-layer pinned independent outputs; multi-layer gaps confound erased propagation and merge interactions; never pure merge nonlinearity')
    for label, source in E.progress(families, 'decisive_eval:decomposition'):
        scopes = fact_scopes(facts,source.states)
        multi = [len(route)>1 for route in oracle_routes(items,scopes)]
        family_report = dict(each_bank_alone={},multi_bank_items=sum(multi),clusters=cluster_counts(items,facts))
        for bank in E.progress(sorted(source.states), 'decisive_eval:decomposition'):
            gens = generate_for(source,items,[(bank,)]*len(items))
            panel = summary(tok,items,gens,k,facts)
            own = scopes[bank]
            panel['within_scope'] = summary(tok,[it for it in items if set(it['fis']) <= own],
                [g for it,g in zip(items,gens) if set(it['fis']) <= own],k,facts)
            family_report['each_bank_alone'][str(bank)] = panel
        independent = IndependentResiduals(source)
        gens = residual_generate(independent,items,dev)
        pinned = summary(tok,items,gens,k,facts)
        pinned.update(forwards=independent.forwards,unmatched_output_rows=independent.omitted_rows,
            status='measured' if independent.omitted_rows == 0 else 'flagged: independent/base expert routes diverged',
            definition=IndependentResiduals.__doc__,
            routing_limitation='solo trajectories and router weights differ; unmatched top-k rows counted',
            pure_merge_nonlinearity_interpretation=False)
        pinned['multi_bank'] = summary(tok,[it for it,m in zip(items,multi) if m],
            [g for g,m in zip(gens,multi) if m],k,facts)
        family_report['per_layer_pinned_independent_outputs'] = pinned
        merged_gens = generate_for(source,items,[tuple(sorted(source.states))]*len(items))
        merged = summary(tok,items,merged_gens,k,facts)
        merged['multi_bank'] = summary(tok,[it for it,m in zip(items,multi) if m],
            [g for g,m in zip(merged_gens,multi) if m],k,facts)
        family_report['merged'] = merged
        comparison = paired_panel([r['correct'] for r in pinned['per_item']],
                                  [r['correct'] for r in merged['per_item']],items,facts)
        comparison.update(same_bank_files=True,routing_comparable=independent.omitted_rows==0,
                          pure_merge_nonlinearity_interpretation=False)
        family_report['paired_comparison'] = comparison
        family_report['paired_multi_bank'] = paired_panel(
            [r['correct'] for r in pinned['multi_bank']['per_item']],
            [r['correct'] for r in merged['multi_bank']['per_item']],
            [it for it,m in zip(items,multi) if m],facts)
        report['families'][label] = family_report
    return report


def rescore_two_fact(branches,tok,facts,bos,dev,a,select):
    items = E.build_qa_items(tok,facts,bos,a.qa_n)['two_fact']
    if getattr(a, 'fact_class_metrics', False):
        E.C50.annotate(items, facts)
    routes = select(items) if select is not None else [tuple(branches.states)]*len(items)
    conditions = {}
    for limit in E.progress((40,64), 'decisive_eval:rescore_two_fact'):
        gens = []
        for it,route in E.progress(zip(items,routes), 'decisive_eval:rescore_two_fact', total=min(len(items),len(routes))):
            branches.activate(route)
            gens.extend(E.greedy_generate(branches.model,[it['p']],dev,limit,1))
        conditions[str(limit)] = summary(tok,items,gens,a.gen_stop_k,facts,fact_scopes(facts,branches.states))
        conditions[str(limit)]['max_new'] = limit
    return dict(same_items=True,conditions=conditions,
                paired_correct=paired_panel([r['correct'] for r in conditions['40']['per_item']],
                                     [r['correct'] for r in conditions['64']['per_item']],items,facts))


def suite(branches, tok, facts, bos, dev, a, select=None, block=None, unit=None, compose=True):
    for family in E.progress((branches,block,unit), 'decisive_eval:suite'):
        if family is not None:
            for state in E.progress(family.states.values(), 'decisive_eval:suite'):
                E.eda.require_patch_backend(state, a, 'decisive bank')
                check_scale(state['meta'])
    scopes = fact_scopes(facts, branches.states)
    groups = task_items(tok, facts, bos, a.qa_n)
    if getattr(a, 'fact_class_metrics', False):
        for group in groups.values():
            E.C50.annotate(group, facts)
    direct = [dict(it,prompt=tok.decode(it['p'][1:]),fis=[it['fi']],vals=[it['val']])
              for it in E.build_items(tok,facts,E.TEST_T,bos)[:a.qa_n]]
    if getattr(a, 'fact_class_metrics', False):
        E.C50.annotate(direct, facts)
    def generate_for(family, items, routes):
        # Batch one: paired routing and linear-hook identity are independent of
        # padding and batch shape. Bank decisions remain fixed for all 64 tokens.
        outs = []
        for it, route in zip(items,routes):
            family.activate(route)
            outs.extend(E.greedy_generate(family.model,[it['p']],dev,64,1))
        return outs
    generate = lambda items,routes: generate_for(branches,items,routes)
    ceilings, routing = {}, {}
    try:
        for kind, items in E.progress(groups.items(), 'decisive_eval:suite'):
            ceilings[kind] = {}
            evidence = evidence_items(tok,facts,items,bos)
            ceilings[kind]['oracle_evidence'] = summary(tok,evidence,generate(evidence,[()]*len(items)),a.gen_stop_k,facts,scopes)
            explicit = [dict(it,prompt=it['explicit'],p=E.encode(tok,it['explicit'],bos)) for it in items]
            # Explicit chains resolve the relation before selecting target banks.
            explicit = [dict(it,fis=[fi for fi in it['fis'] if fi != it.get('relation_fi')]) for it in explicit]
            entity_routes = oracle_routes(explicit,scopes)
            ceilings[kind]['oracle_entity'] = summary(tok,explicit,generate(explicit,entity_routes),a.gen_stop_k,facts,scopes)
            ceilings[kind]['oracle_entity']['unsupported_items'] = sum(not route for route in entity_routes)
            ceilings[kind]['oracle_entity']['policy'] = 'explicit resolved entity; correct available bank(s) only; reverse lookup supplies gold name'
            ceilings[kind]['oracle_entity']['routes'] = entity_routes
            routes = select(items) if select is not None else [tuple(branches.states)]*len(items)
            gens = generate(items,routes)
            ceilings[kind]['as_trained'] = summary(tok,items,gens,a.gen_stop_k,facts,scopes)
            if kind == 'forward_chain' and not items:
                ceilings[kind]['unavailable'] = 'world contains no stored sister relation targeting a named person; no synthetic chain claimed'
            routing[kind] = conditioned_recall(branches.model,tok,items,gens,routes,branches.states,scopes,
                                               branches.activate,dev,a.gen_stop_k,facts)
        report = dict(eda_backend=E.eda.selected(a), glossary=GLOSSARY, ceilings=ceilings, routing_conditioned_recall=routing,
                      two_fact_rescore=rescore_two_fact(branches,tok,facts,bos,dev,a,select),
                      format_sensitivity=format_report(tok,facts,dict(groups,direct=direct),bos,generate,a.gen_stop_k,scopes),
                      negatives=negative_report(branches.model,tok,facts,branches.states,bos,generate,
                                                branches.activate,dev,a.qa_n,select,
                                                fact_class_metrics=getattr(a, 'fact_class_metrics', False)),
                      serving='selected banks at full stored strength; no blending; 64 new tokens; batch size 1')
        report['paired_ceiling_gains'] = {kind: {condition: paired_panel(
            [r['correct'] for r in conditions['as_trained']['per_item']],
            [r['correct'] for r in conditions[condition]['per_item']],groups[kind],facts)
            for condition in ('oracle_entity','oracle_evidence')} for kind,conditions in ceilings.items()}
        if compose:
            report['decomposition'] = decomposition(branches,block,unit,tok,facts,groups,generate_for,dev,a.gen_stop_k)
        if getattr(a, 'fact_class_metrics', False):
            report['by_fact_class'] = E.C50.collect_splits(report)
        return report
    finally:
        branches.activate(())


class ProductKeyIndexer:
    """Two ceil(sqrt(N)) ternary codebooks on a ridge-projected residual.

    Each bank owns a distinct (i,j). TRAIN fits the residual projection; VAL
    chooses layer/L2/threshold. Product cosine scores are not probabilities.
    """
    def __init__(self, center, projection, left, right, owners, pairs):
        self.center, self.projection, self.left, self.right = center, projection, left, right
        self.owners, self.pairs = tuple(owners), pairs

    @classmethod
    def fit(cls, x, labels, general, width=16, l2=.01, seed=34):
        x, general = (torch.as_tensor(v).detach().float().cpu() for v in (x,general))
        y = torch.tensor(labels)
        owners = sorted(set(labels))
        if (x.ndim != 2 or general.ndim != 2 or x.shape[1] != general.shape[1]
                or y.shape != (len(x),) or not torch.isfinite(x).all() or not torch.isfinite(general).all()
                or not isinstance(width,int) or width < 1 or not math.isfinite(l2) or l2 <= 0
                or not len(owners) or not len(general)):
            raise ValueError('invalid product-key training data')
        m = math.ceil(math.sqrt(len(owners)))
        rng = torch.Generator().manual_seed(seed)
        def book():
            rows = []
            while len(rows) < m:
                row = torch.randint(-1,2,(width,),generator=rng,dtype=torch.int8)
                if row.any() and not any(torch.equal(row,r) for r in rows):
                    rows.append(row)
            return torch.stack(rows)
        if 3**width-1 < m:
            raise ValueError('product-key width cannot encode distinct ternary keys')
        left,right = book(),book()
        pairs = torch.tensor([(i//m,i%m) for i in range(len(owners))],dtype=torch.int64)
        target = torch.zeros((len(x)+len(general),2*width))
        for i, owner in enumerate(owners):
            target[:len(x)][y==owner] = torch.cat((left[pairs[i,0]],right[pairs[i,1]])).float()
        center = general.mean(0)
        features = torch.cat((x,general))-center
        # Dual solve bounds the expensive matrix by number of TRAIN examples.
        gram = features @ features.T
        projection = features.T @ torch.linalg.solve(gram+l2*torch.eye(len(gram)),target)
        return cls(center,projection,left,right,owners,pairs)

    def scores(self,x):
        x = torch.as_tensor(x).detach().float().cpu()
        if x.ndim != 2 or x.shape[1] != len(self.center) or not torch.isfinite(x).all():
            raise ValueError('invalid product-key features')
        code = (x-self.center) @ self.projection
        left,right = code.chunk(2,dim=1)
        s1 = F.normalize(left,dim=-1) @ F.normalize(self.left.float(),dim=-1).T
        s2 = F.normalize(right,dim=-1) @ F.normalize(self.right.float(),dim=-1).T
        return (s1[:,self.pairs[:,0]]+s2[:,self.pairs[:,1]])/2

    def select(self,x,threshold=.5,top_k=1):
        if not 1 <= top_k <= len(self.owners) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError('invalid product-key selection threshold/budget')
        scores = self.scores(x)
        return [tuple(sorted(self.owners[i] for i in row.argsort(descending=True,stable=True)[:top_k].tolist()
                             if float(row[i]) >= threshold)) for row in scores]

    def storage_bytes(self):
        return 8*len(self.owners)+sum(v.numel()*v.element_size() for v in vars(self).values() if torch.is_tensor(v))


def positive_control():
    class Tok:
        def decode(self,ids):
            return ''.join(chr(i) for i in ids)
    tok = Tok()
    good = [ord(c) for c in 'viola. '+('x'*44)+' harp.']
    scores = [E.score_two_fact_generation(tok,good,['viola','harp'])['both'],
              E.score_two_fact_lenient(tok,good,['viola','harp'],max_tokens=64)]
    x = torch.eye(4).repeat_interleave(3,0)*4
    labels = [i for i in range(4) for _ in range(3)]
    index = ProductKeyIndexer.fit(x,labels,torch.zeros(8,4),width=8)
    return dict(passed=scores==[False,True] and index.select(x)==[(p,) for p in labels]
                and index.select(torch.zeros(2,4))==[(),()], kind='late second value, four product keys, null abstention')


def product_key_arm(B, branches, tok, facts, items, anchor, null_val, selector_val,
                    panels, base_nll, dev, a, bench_items=None, baseline=None):
    """Validation-locked PK/I/DG comparison with a common allocated byte budget.

    Useful state and explicit padding are reported separately. Padding buys no
    accuracy; no claim of equal effective capacity follows from equal residency.
    """
    if a.pk_width < 1 or any(v <= 0 for v in a.selector_l2):
        raise ValueError('PK requires positive width and positive selector L2')
    layers = ([int(v) for v in a.indexer_layers.split(',')] if a.indexer_layers else
              list(dict.fromkeys([a.indexer_layer]+[li for li in (4,26,28) if li < len(branches.model.layers)])))
    candidates, indexes = {key: [] for key in ('PK','I','DG')}, []
    for layer in E.progress(layers, 'decisive_eval:product_key_arm'):
        train,x,g = B.address_training(branches.model,tok,facts,anchor,
                                      next(iter(branches.states.values()))['meta']['seed'],layer,dev)
        labels = [facts[it['fi']].pid for it in train]
        attrs = [facts[it['fi']].attr for it in train]
        vx = B.address_eval_features(branches.model,tok,{'validation':selector_val},layer,dev)['validation']
        nx = B.indexer_features(branches.model,[it['p'] for it in null_val],layer,dev)
        expected = B.gold_routes(selector_val,facts)
        for l2 in E.progress(a.selector_l2, 'decisive_eval:product_key_arm'):
            fitted = {
                'I': B.BankIndexer.fit(x,labels,g),
                'DG': B.DGIndexer.fit(x,labels,attrs,g,a.dg_expansion,a.dg_density,a.dg_sparsity,
                                      a.dg_seed,a.dg_steps,l2),
                'PK': ProductKeyIndexer.fit(x,labels,g,a.pk_width,l2,a.dg_seed),
            }
            for kind,index in E.progress(fitted.items(), 'decisive_eval:product_key_arm'):
                useful = (index.storage_bytes() if hasattr(index,'storage_bytes') else
                          8*len(index.owners)+sum(v.numel()*v.element_size() for v in vars(index).values() if torch.is_tensor(v)))
                indexes.append((index,useful))
                thresholds = [0.] if kind == 'DG' else a.indexer_thresholds
                for theta in E.progress(thresholds, 'decisive_eval:product_key_arm'):
                    budget = min(a.indexer_top_k,len(index.owners))
                    def select(x,index=index,theta=theta,budget=budget):
                        return index.select(x,theta,budget)
                    score = B.selection_metrics(select(vx),expected)['accuracy'] - sum(bool(r) for r in select(nx))/max(len(nx),1)
                    candidates[kind].append((score,layer,l2,theta,index,select,useful))
    allocated = max(size for _,size in indexes)
    result = dict(equal_allocated_address_bytes=allocated, patch_bytes=sum(branches.bytes.values()),
                  byte_policy='identical allocated residency; explicit uint8 reserve padding; useful state reported separately',
                  capacity_limitation='equal bytes including reserve does not imply equal effective capacity', selectors={})
    locked = {}
    # TEST is first read only after all three selections have been locked.
    for kind,rows in E.progress(candidates.items(), 'decisive_eval:product_key_arm'):
        chosen = B.R.select_validation(rows,lambda row: row[0])
        score,layer,l2,theta,index,select,useful = chosen
        index.byte_reserve = torch.zeros(allocated-useful,dtype=torch.uint8)
        locked[kind] = (layer,select)
        routes = B.address_eval_features(branches.model,tok,{'test':items['test']},layer,dev)['test']
        selected = select(routes)
        gens = B.routed_generations(branches,tok,items['test'],selected,dev,a.gen_max_new or 64,1)
        panel_routes = {name: select(B.indexer_features(branches.model,[win[:64].tolist() for win in wins],layer,dev))
                        for name,wins in panels.items()}
        damage = B.damage(branches,panels,base_nll,panel_routes,dev)
        def bench_route(group):
            return select(B.address_eval_features(branches.model,tok,{'bench':group},layer,dev)['bench'])
        bench = B.routed_benchmark(branches,bench_items,baseline['base_bench'],bench_route,dev) if bench_items else None
        result['selectors'][kind] = dict(layer=layer,l2=l2,threshold=theta,validation_score=score,
            selection_split='VAL only',locked=True,address_bytes=allocated,useful_address_bytes=useful,
            reserve_bytes=allocated-useful,activation_budget=min(a.indexer_top_k,len(index.owners)),
            key_comparisons=(2*math.ceil(math.sqrt(len(index.owners))) if kind=='PK' else len(index.owners)),
            selection=B.selection_metrics(selected,B.gold_routes(items['test'],facts)),
            test_gen=B.generation_summary(tok,items['test'],gens,E.conflict_map(facts),a.gen_stop_k),
            damage=damage,bench=bench,damage_gate=E.damage_gate(damage,bench),
            damage_policy='same locked query selector, batch size 1',routes=selected,
            held_out=B.R.routing_panels(items['test'],facts,selected,B.selection_metrics))
    result['control'] = positive_control()
    result['verdict'] = 'measured' if result['control']['passed'] else 'control-failed'
    if a.decisive_eval:
        result['decisive_eval'] = {}
        for kind,(layer,selected) in E.progress(locked.items(), 'decisive_eval:product_key_arm'):
            def route(group,layer=layer,selected=selected):
                return selected(B.address_eval_features(branches.model,tok,{'c34':group},layer,dev)['c34']) if group else []
            result['decisive_eval'][kind] = suite(branches,tok,facts,E.BOS,dev,a,route,compose=False)
    branches.activate(())
    return result
