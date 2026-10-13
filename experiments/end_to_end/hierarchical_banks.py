"""C33 attribute-bank evaluation using branch_eval's serving and C32 safeguards."""
from itertools import product
import json
import math

import torch


def fact_banks(E, facts, states):
    owners = {}
    for bank, state in states.items():
        meta = state['meta']
        for fi in E.person_indices(facts, meta['train_person'], meta['train_attr_group'], meta['attr_groups']):
            if fi in owners:
                raise ValueError('fact belongs to multiple banks')
            owners[fi] = bank
    if set(owners) != set(range(len(facts))):
        raise ValueError('resident banks must cover the full fact world')
    return owners


def gold_routes(items, owners):
    # Reference facts resolve a name; the answer facts identify the required banks.
    return [tuple(sorted({owners[fi] for fi in it.get('fis', [it.get('fi')])})) for it in items]


def positive_control(B, arm):
    labels = torch.arange(12)
    x1 = torch.eye(3).repeat_interleave(4, 0)
    x2 = torch.eye(4).repeat(3, 1)
    null1, null2 = torch.zeros(1, 3), torch.zeros(1, 4)
    hierarchy = B.HierarchicalIndexer.fit(x1, x2, labels, null1, null2, 4)
    h = hierarchy.select(x1, x2, .8, .8, 2, 2, 4, trace=True)
    flat = B.BankIndexer.fit(torch.eye(12), labels, torch.zeros(1, 12))
    routes = flat.select(torch.eye(12), .8, 4) if arm == 'FLAT' else h['routes']
    null = hierarchy.select(null1, null2, .8, .8, 2, 2, 4, trace=True)
    child_null = all(c.select(null2, .8, 2) == [()] for c in hierarchy.children.values())
    passed = routes == [(i,) for i in range(12)] and null['parents'] == [()] and null['routes'] == [()] and child_null
    passed = passed and flat.select(torch.zeros(1, 12), .8, 4) == [()]
    return dict(passed=passed, kind='12 separable synthetic banks; null abstains at both levels',
                selection_accuracy=B.selection_metrics(routes, [(i,) for i in range(12)])['accuracy'],
                level1_null_abstains=null['parents'] == [()], level2_null_abstains=child_null)


def layer_candidates(value, default, count):
    try:
        layers = [int(v) for v in value.split(',')] if value else [default]
    except ValueError as err:
        raise ValueError('invalid hierarchical layer list') from err
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= count:
        raise ValueError('hierarchical layers must be distinct zero-based full block inputs')
    return layers


def evaluate(B, a, arms, branches, tok, facts, items, panels, sources, anchor, baseline,
             revoke_input, dev, bench_items, subject_items, mapping):
    E, R, model = B.E, B.R, branches.model
    meta = next(iter(branches.states.values()))['meta']
    groups, people = meta['attr_groups'], meta['people']
    if (not 2 <= a.h_top_k1 <= 4 or a.h_top_k1 > people or not 1 <= a.h_top_k2 <= groups
            or not 1 <= a.h_beam <= len(branches.states)):
        raise ValueError('H requires top-k1 in [2,4] within people, top-k2 within groups, beam within bank count')
    if any(not math.isfinite(t) or not -1 <= t <= 1 for t in a.h_thresholds1+a.h_thresholds2+list(a.indexer_thresholds)):
        raise ValueError('hierarchical/flat thresholds must be finite cosines in [-1,1]')
    l1s = layer_candidates(a.h_layers1, a.h_layer1, len(model.layers))
    l2s = layer_candidates(a.h_layers2, a.h_layer2, len(model.layers))
    flats = layer_candidates(a.indexer_layers, a.indexer_layer, len(model.layers))
    owners = fact_banks(E, facts, branches.states)
    # A held-out attribute per bank, only when training retains another attribute.
    held = set()
    for bank in branches.states:
        fis = [fi for fi in owners if owners[fi] == bank]
        attrs = sorted({facts[fi].attr for fi in fis})
        if len(attrs) > 1:
            held.update(fi for fi in fis if facts[fi].attr == attrs[-1])
    train = [it for it in B.S.build_panels(tok, facts, E.BOS, question_train=False)['train'] if it['fi'] not in held]
    validation = [it for it in items['val'] if it['fi'] not in held]
    labels = [owners[it['fi']] for it in train]
    expected = gold_routes(validation, owners)
    null_val = R.independent_queries(sources, E.BOS, 64)
    negatives = R.independent_queries(sources, E.BOS, a.negative_n+len(null_val), skip=40*511+64*63)
    if not null_val:
        raise ValueError('H/FLAT need disjoint general-text validation queries after preservation windows')
    val_sequences = {tuple(it['p']) for it in null_val}
    negatives = [it for it in negatives if tuple(it['p']) not in val_sequences][:a.negative_n]
    E.checked_overlap(anchor, {it['source']+':'+str(it['start']): it['p'][1:] for it in null_val+negatives},
                      'hierarchical negative calibration')
    wins, positions, _ = B.S.general_windows(anchor, meta['seed'], E.BOS)
    feature_cache = {}
    def features(layer, name, group, positions=None, questions=True):
        key = (layer, name)
        if key not in feature_cache:
            sequences = [B.question_sequence(tok, it['p'], E.BOS) if questions else it['p'] for it in group]
            feature_cache[key] = B.indexer_features(model, sequences, layer, dev, positions)
        return feature_cache[key]
    data = {}
    for layer in sorted(set(l1s+l2s+flats)):
        data[layer] = (features(layer, 'train', train, questions=False),
                       B.indexer_features(model, wins, layer, dev, positions),
                       features(layer, 'validation', validation),
                       features(layer, 'null_val', null_val, questions=False))
    def validation_score(routes, null_routes):
        return B.selection_metrics(routes, expected)['accuracy'] - sum(bool(r) for r in null_routes)/len(null_routes)
    locked = {}
    if 'H' in arms:
        candidates = []
        for l1, l2 in product(l1s, l2s):
            x1, g1, v1, n1 = data[l1]
            x2, g2, v2, n2 = data[l2]
            index = B.HierarchicalIndexer.fit(x1, x2, labels, g1, g2, groups)
            for t1, t2 in product(a.h_thresholds1, a.h_thresholds2):
                val = index.select(v1, v2, t1, t2, a.h_top_k1, a.h_top_k2, a.h_beam)
                null = index.select(n1, n2, t1, t2, a.h_top_k1, a.h_top_k2, a.h_beam)
                candidates.append((validation_score(val, null), l1, l2, t1, t2, index))
        best = R.select_validation(candidates, lambda c: c[0])
        score, l1, l2, t1, t2, index = best
        locked['H'] = dict(layers=[l1, l2], thresholds=[t1, t2], index=index, validation_score=score,
                           candidates=len(candidates))
    if 'FLAT' in arms:
        candidates = []
        for layer in flats:
            x, g, v, n = data[layer]
            index = B.BankIndexer.fit(x, labels, g)
            for threshold in a.indexer_thresholds:
                score = validation_score(index.select(v, threshold, a.h_beam), index.select(n, threshold, a.h_beam))
                candidates.append((score, layer, threshold, index))
        score, layer, threshold, index = R.select_validation(candidates, lambda c: c[0])
        locked['FLAT'] = dict(layers=[layer], thresholds=[threshold], index=index,
                              validation_score=score, candidates=len(candidates))
    def route(arm, name, group, positions=None, questions=True):
        cfg = locked[arm]
        xs = [features(layer, name, group, positions, questions) for layer in cfg['layers']]
        if arm == 'H':
            return cfg['index'].select(*xs, *cfg['thresholds'], a.h_top_k1, a.h_top_k2, a.h_beam, trace=True)
        return dict(routes=cfg['index'].select(xs[0], cfg['thresholds'][0], a.h_beam),
                    comparisons=[len(branches.states)]*len(group))
    branches.activate(())
    zero = B.evaluate_branch_base(model, tok, panels, {k: v for k, v in items.items() if k not in ('val','test','g_set','general')},
                                  revoke_input, dev, a, bench_items, subject_items)
    zero_ok = all(zero[k] == baseline[k] for k in ('base_nll', 'base_qa_probe')) and torch.equal(zero['base_logits'], baseline['base_logits'])
    if a.bench:
        zero_ok = zero_ok and zero['base_bench'] == baseline['base_bench']
    if a.subject_text:
        zero_ok = zero_ok and zero['base_subject'] == baseline['base_subject']
    # Each bank must reproduce its standalone load through the serving path.
    solo_ok = True
    for bank, state in branches.states.items():
        branches.activate((bank,))
        with torch.no_grad():
            served = E.logits_of(model, revoke_input).clone()
            E.load_patch(model, state)
            solo_ok = solo_ok and torch.equal(served, E.logits_of(model, revoke_input))
    result = dict(arms={}, layer_mapping=mapping, layer_indexing='zero-based full block inputs',
                  bank_manifest=[dict(bank=bank, person=s['meta']['train_person'], group=s['meta']['train_attr_group'],
                                      bytes=branches.bytes[bank], partition_mode=B.partition_label(s['meta']))
                                 for bank, s in sorted(branches.states.items())],
                  held_out_fact_ids=sorted(held), held_out_limitation=None if held else 'one attribute per bank; no held-out attribute possible',
                  negative_sample=dict(requested=a.negative_n, available=len(negatives),
                     policy='deduplicated nonoverlapping windows disjoint from preservation, validation and TRAIN anchor',
                     independence_unit='text window; binomial bound assumes query independence'),
                  zero_patch_control=dict(passed=zero_ok), solo_bank_control=dict(passed=solo_ok),
                  total_resident_bytes=sum(branches.bytes.values()), activation_budget=a.h_beam)
    shared = {f.attr for f in facts if len({g.pid for g in facts if g.attr == f.attr}) > 1}
    for arm in arms:
        cfg = locked[arm]
        control = positive_control(B, arm)
        control['passed'] = control['passed'] and zero_ok and solo_ok
        report = dict(control=control, selection={}, composed_recall={}, selection_cost={},
                      selector_selection={k: v for k, v in cfg.items() if k != 'index'},
                      residency='all banks resident in host memory; selected merge loaded for forward',
                      partition_mode=B.partition_label(meta), routing_policy='locked selector; one route for entire query',
                      active_bytes={})
        report['selector_selection'].update(locked=True, selection_split='VAL only',
            top_k1=a.h_top_k1 if arm == 'H' else None, top_k2=a.h_top_k2 if arm == 'H' else None, beam=a.h_beam,
            score_rule='sum log((cosine+1)/2)' if arm == 'H' else 'cosine')
        test_routes = []
        for name, group in items.items():
            if name == 'indirect' or not group:
                continue
            trace = route(arm, name, group, questions=name != 'general')
            routes = trace['routes']
            gold = [()]*len(group) if name == 'general' else gold_routes(group, owners)
            report['selection'][name] = B.selection_metrics(routes, gold)
            counts = trace['comparisons']
            report['selection_cost'][name] = dict(n=len(counts), mean=sum(counts)/len(counts), max=max(counts), per_query=counts)
            report['active_bytes'][name] = dict(mean=sum(branches.active_bytes(r) for r in routes)/len(routes))
            gens = B.routed_generations(branches, tok, group, routes, dev, a.gen_max_new or (80 if name in ('g_set','two_fact') else 64), a.gen_bs)
            if name in ('val', 'test'):
                report[name+'_gen'] = B.generation_summary(tok, group, gens, E.conflict_map(facts), a.gen_stop_k)
                if name == 'test':
                    test_routes = routes
                    report['composed_recall']['single_fact'] = report['test_gen']['recall']
            elif name in ('g_set', 'two_fact'):
                report['composed_recall'][name] = dict(n=len(group), binding=sum(R.binding_score(tok.decode(g), it, facts) for it,g in zip(group,gens))/len(group),
                    strict_presence=sum(E.score_two_fact_generation(tok,g,it['vals'],a.gen_stop_k)['both'] for it,g in zip(group,gens))/len(group))
        for name, subset in [('shared_attribute', [i for i,it in enumerate(items['test']) if facts[it['fi']].attr in shared]),
                             ('held_out_attribute', [i for i,it in enumerate(items['test']) if it['fi'] in held])]:
            report['selection'][name] = B.selection_metrics([test_routes[i] for i in subset], gold_routes([items['test'][i] for i in subset], owners))
        negative = route(arm, 'negatives', negatives, questions=False) if negatives else dict(routes=[], comparisons=[])
        report['false_activation'] = R.false_activation(negative['routes'])
        costs = negative['comparisons']
        report['selection_cost']['negatives'] = dict(n=len(costs), mean=sum(costs)/max(len(costs),1), max=max(costs, default=0))
        if arm == 'H' and negatives:
            report['level1_false_activation'] = R.false_activation(negative['parents'])
            # Probe child nulls even when level 1 abstains; these do not fire banks.
            x2 = features(cfg['layers'][1], 'negatives', negatives, questions=False)
            report['level2_null_probes'] = {str(p): R.false_activation(c.select(x2, cfg['thresholds'][1], a.h_top_k2)) for p,c in cfg['index'].children.items()}
        droutes = {name: route(arm, 'panel:'+name, [dict(p=win.tolist()) for win in windows],
                               [63]*len(windows), False)['routes'] for name,windows in panels.items()}
        report['damage'] = B.damage(branches, panels, baseline['base_nll'], droutes, dev)
        def bench_route(group):
            # Different benchmarks cannot alias the feature-cache key.
            return route(arm, 'benchmark:'+str(id(group)), group)['routes']
        bench = B.routed_benchmark(branches, bench_items, baseline.get('base_bench'), bench_route, dev) if a.bench else None
        if bench is not None:
            report['bench'] = bench
        report['damage_gate'] = E.damage_gate(report['damage'], bench)
        report['benchmark_policy'] = 'same query routing policy as arm; route fixed for all choices'
        if subject_items:
            sroutes = route(arm, 'subject', subject_items, [min(63,len(it['p'])-1) for it in subject_items], False)['routes']
            report['subject_dnll'] = B.routed_subject_damage(branches, subject_items, baseline['base_subject'], sroutes, dev)
        report['verdict'] = ('control-failed' if not control['passed'] else
                             'measured' if report['false_activation']['sufficient'] else 'insufficient independent negatives')
        report['verdict_suppressed'] = not control['passed'] or not report['false_activation']['sufficient']
        result['arms'][arm] = report
        print('BRANCH-EVAL '+json.dumps(dict(arm=arm, **report)), flush=True)
    E.set_enabled(False)
    with torch.no_grad():
        revoked = torch.equal(baseline['base_logits'], E.logits_of(model, revoke_input))
    result['revoked'] = dict(logits_bit_exact=revoked)
    if not revoked:
        raise RuntimeError('hierarchical bank revocation failed')
    E.set_enabled(True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    return result
