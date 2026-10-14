"""CC1: fresh-model load/unload/order checks and clustered safety bounds.

No fitting or final-panel selection. Uses Branches activation and C34 probes.
One invocation screens one seed/family; promotion requires two paired seeds.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

try:
    from . import e2e_fsa as E, branch_eval as B, decisive_eval as D, in_topic as I
except ImportError:
    import e2e_fsa as E
    import branch_eval as B
    import decisive_eval as D
    import in_topic as I

CONFIGURATIONS = {'base': (), 'A': (0,), 'B': (1,), 'AB': (0, 1),
                  'AB-A': (0, 1, '-0'), 'AB-B': (0, 1, '-1'), 'BA': (1, 0)}
MECHANICAL_NOTE = ('Disjoint slots guarantee MECHANICAL removal, not KNOWLEDGE removal. '
                   'Joint gradients (J/F) can teach B information from A; canary leakage measures this.')


def single_items(tok, facts, templates, ids, bos):
    rows = []
    for fi in sorted(ids):
        fact = facts[fi]
        for template in templates:
            full = E.fmt(template, fact)
            prompt = full[:full.rindex(fact.val)].rstrip() if '{val}' in template else full
            rows.append(D.make_item(tok, prompt, [fi], [fact.val], bos, fi=fi))
    return rows


def validate_pair(states, args):
    if set(states) != {0, 1}:
        raise ValueError('need A and B')
    first = states[0]['meta']
    if first.get('train_person') != 0 or states[1]['meta'].get('train_person') != 1:
        raise ValueError('CC1 requires p0 A and p1 B')
    E.eda.require_same(states.values(), 'CC1')
    for state in states.values():
        E.validate_patch_export(state['meta'], args.export, args.export_step)
        E.eda.require_backend(state, 'naive_fp32', 'CC1')
        if state.get('format') != 'e2e-fsa-patch-v1':
            raise ValueError('unsupported CC1 patch')
        if state['meta'].get('method') != 'fsa_forced' or state['meta'].get('fmt') != 'ternary':
            raise ValueError('CC1 requires forced-zero ternary banks')
        if state['meta'].get('patch_scale') != 'row' or state['meta'].get('people') != 3 or (
                state['meta'].get('train_attr_group') is not None or state['meta'].get('attr_groups', 1) != 1):
            raise ValueError('CC1 requires row-scaled three-person R20 person banks')
        if not state['meta'].get('relation_world') or state['meta'].get('layers') != [12, 14, 17, 20]:
            raise ValueError('CC1 requires R20 relation world on MoE layers 12,14,17,20')
        for key in B.IDENTITY:
            if state['meta'].get(key) != first.get(key):
                raise ValueError('CC1 patch identity differs: '+key)
        if state['entries'].keys() != states[0]['entries'].keys():
            raise ValueError('CC1 linear sets differ')
        D.check_scale(state['meta'])
    reject_overlap(states)
    family = 'whole-unit' if first.get('partition_unit') else 'full-mask'
    if family == 'whole-unit':
        if not states[1]['meta'].get('partition_unit') or first.get('slot_partition') != 3 or states[1]['meta'].get('slot_partition') != 3:
            raise ValueError('CC1 unit pair requires K=3 whole-unit ownership')
        if first.get('slot_part') != 0 or states[1]['meta'].get('slot_part') != 1:
            raise ValueError('CC1 ownership must agree with person ids')
        if B.overlap_stats([states[0], states[1]])['overlapping_slots']:
            raise ValueError('whole-unit pair has colliding slots')
    elif first.get('slot_partition') or states[1]['meta'].get('slot_partition'):
        raise ValueError('full-mask family cannot claim partitioned ownership')
    return family


def owned_facts(facts, bank):
    return {i for i, f in enumerate(facts) if f.pid == bank and f.attr not in ("sister's name", 'sister')}


def canaries(tok, facts, bank, bos, minimum=1000):
    """Exactly ten held-out, answer-free surfaces for EVERY owned value fact."""
    own = owned_facts(facts, bank)
    other = next(f.name for f in facts if f.pid == 1-bank)
    rows = single_items(tok, facts, E.TEST_T + E.QA_UNSEEN_T, own, bos)
    rows = [dict(r, family='direct', canary_values=r['vals']) for r in rows]
    for fi in sorted(own):
        f = facts[fi]
        surfaces = [('qa_pronoun', E.QA_PRONOUN_T), ('qa_pronoun_recent', E.QA_PRONOUN_RECENT_T)]
        surfaces += [(k, D.C34_T[k]) for k in ('single_mention_pronoun', 'pronoun_earlier', 'pronoun_later')]
        for family, template in surfaces:
            prompt = template.format(**f._asdict(), other=other, filler=''.join(E.QA_FILLER_TURNS[:1+fi % 3]))
            rows.append(D.make_item(tok, prompt, [fi], [f.val], bos, family=family, canary_values=[f.val]))
    if len(rows) != 10*len(own):
        raise ValueError('canary census must have ten surfaces per fact')
    return rows


def unknowns(tok, facts, states, bos, minimum=1000):
    try:
        from . import c52
    except ImportError:
        import c52
    return [dict(row, fis=[i], vals=[]) for i, row in enumerate(c52.absent_items(E,D,tok,facts,bos,max(1000,minimum)))]


def conjunction_census(tok, facts, bos, scopes):
    # Cover each owned fact before filling a fixed-seed 200-item sample.
    aa, bb = [sorted(scopes[i]) for i in (0,1)]
    if max(len(aa), len(bb)) > 200:
        raise ValueError('200 conjunctions cannot cover this roster')
    pairs = [(a,b) for a in aa for b in bb]
    if not pairs:
        raise ValueError('both banks need value facts')
    chosen = list(dict.fromkeys((aa[i % len(aa)], bb[i % len(bb)]) for i in range(max(len(aa),len(bb)))))
    remaining = [pair for pair in pairs if pair not in chosen]
    random.Random(610061).shuffle(remaining)
    chosen += remaining[:200-len(chosen)]
    # Small CPU worlds exhaust unique pairs; never pad the measured roster.
    return [D.make_item(tok, D.C34_T['cross_person_conjunction'].format(name=facts[a].name,
            attr=facts[a].attr, other=facts[b].name, attr2=facts[b].attr), [a,b], [facts[a].val,facts[b].val], bos)
            for a,b in chosen]


def probe_sets(tok, facts, states, bos, n=120, minimum=1000):
    scopes = {i: owned_facts(facts,i) for i in (0,1)}
    result = {}
    for bank, label in ((0,'A'),(1,'B')):
        for kind, templates in (('direct',E.TEST_T),('unseen',E.QA_UNSEEN_T)):
            result[label+'/'+kind] = single_items(tok,facts,templates,scopes[bank],bos)
        result[label+'/canaries'] = canaries(tok,facts,bank,bos)
    # Build only chains p0 -> p1, without constructing the quadratic C34 world.
    result['both/forward_chain'] = [D.make_item(tok,D.C34_T['forward_chain'].format(name=f.name,attr=facts[j].attr),
        [i,j],[facts[j].val],bos,relation_fi=i) for i,f in enumerate(facts)
        if f.pid == 0 and f.attr in ("sister's name",'sister')
        for j in sorted(scopes[1]) if f.val == facts[j].name]
    conjunction_scopes = {bank:{fi for fi,f in enumerate(facts) if f.pid==bank} for bank in (0,1)}
    result['both/cross_person_conjunction'] = conjunction_census(tok,facts,bos,conjunction_scopes)
    conflicts = E.conflict_map(facts)
    result['conflicting_identifiers'] = single_items(tok,facts,E.TEST_T,set(conflicts),bos)
    result['reverse_lookup'] = [D.make_item(tok,D.C34_T['reverse_lookup'].format(attr=f.attr,val=f.val),
        [i],[f.name],bos) for i,f in enumerate(facts) if i in scopes[0] | scopes[1]]
    result['unknowns'] = unknowns(tok,facts,states,bos,minimum)
    return result


def loss_bound(before, after, items, draws=5000):
    # Single-bank relational probes cluster by target fact, not relation phrasing.
    if len(before) != len(after) or len(before) != len(items):
        raise ValueError('retention scores must align with census')
    deltas = [float(a)-float(b) for a,b in zip(before, after)]
    clusters = [it['fis'][-1] for it in items]
    bound = I.cluster_interval(deltas, clusters, .95, draws)
    grouped = {}
    for delta, fact in zip(deltas, clusters):
        grouped.setdefault(str(fact), []).append(delta)
    events = int(sum(np.mean(values) > 0 for values in grouped.values()))
    # Mean loss <= probability of a fact losing competence. This exact,
    # conservative envelope prevents a zero-width all-retained bootstrap
    # from certifying arbitrarily small panels.
    envelope = B.R.binomial_bound(events, len(grouped))
    if grouped:
        bound['upper'] = max(bound['upper'] if bound['upper'] is not None else 0., envelope)
    bound.update(facts_with_loss=events, exact_any_loss_upper=envelope,
                 policy='fact-cluster paired bootstrap with exact any-fact-loss envelope')
    return bound


def leakage_bound(flags, items, draws=5000):
    if len(flags) != len(items):
        raise ValueError('canary scores must align with probes')
    facts = {}
    for flag, item in zip(flags, items):
        key = str(item['fis'][-1])
        facts[key] = facts.get(key, False) or bool(flag)
    n, events = len(facts), sum(facts.values())
    upper = B.R.binomial_bound(events, n) if n else None
    report = dict(clusters=n, leaking_facts=events, mean=events/n if n else None,
                  lower=B.R.binomial_bound(events, n, upper=False) if n else None, upper=upper,
                  probes=len(items), emitted=sum(flags), probe_rate=sum(flags)/len(flags) if flags else None,
                  rate=events/n if n else None, passed=upper is not None and upper <= .01+1e-12,
                  policy='one-sided 95% exact bound on underlying fact emitting on ANY surface')
    return report


def reject_overlap(states):
    overlap = B.overlap_stats([states[0],states[1]])
    if overlap['overlapping_slots'] or overlap['sign_conflict_slots']:
        raise ValueError('mechanical-unload failure: overlapping slots')
    return overlap


@torch.no_grad()
def edit_bank(model, state, remove=False):
    """Edit resident masters/reset masks directly. Never replay a checkpoint.

    Physical slots deleted on unload revert to the pristine base. On overlapping
    inputs this intentionally destroys the surviving code: the control detects it.
    """
    for name, lin in E.patched_linears(model):
        pm = lin.parametrizations.weight[0]
        entry = state['entries'][name]
        idx = entry['idx'].long().to(pm.M.device)
        if getattr(pm,'sparse',False):
            slots = pm.idx.long()
            positions = torch.searchsorted(slots,idx)
            if idx.numel() and (positions.max() >= len(slots) or not torch.equal(slots[positions],idx)):
                raise ValueError('write outside attached slots')
            idx = positions
        pm.M.view(-1)[idx] = 0 if remove else entry['val'].to(pm.M.device).float()
        if not remove and pm.support is not pm.mask:
            pm.support.view(-1)[idx] = True
        for axis, dim in (('rows',0),('cols',1)):
            key = 'unit_reset_'+axis
            if key not in entry:
                continue
            if not hasattr(pm,key):
                pm.register_buffer(key,torch.zeros(pm.wshape[dim],dtype=torch.bool,device=pm.M.device))
            getattr(pm,key)[entry[key].long().to(pm.M.device)] = not remove
    E.set_enabled(True)


def activate_sequence(branches, sequence):
    branches.activate(())
    for bank in sequence:
        if isinstance(bank,str):
            edit_bank(branches.model,branches.states[int(bank[1:])],remove=True)
        else:
            edit_bank(branches.model,branches.states[bank])


def revoke_prompts(tok, eval_set, bos):
    return [[bos]+json.loads(Path(eval_set).read_text())[:15]] + [E.encode(tok,text,bos)
        for text in ('A quiet morning.','Explain a simple addition.')]


def value_emitted(text, values):
    return any(D.contains(text,value) for value in values)


def positive_controls():
    """CPU effective-weight unload and canary controls, using the live operators."""
    def model():
        m = torch.nn.Sequential(torch.nn.Linear(4,2,bias=False))
        m[0].weight.data.copy_(torch.tensor([[0.,0.,1.,1.],[1.,1.,0.,0.]]))
        E.P.register_parametrization(m[0],'weight',E.FSAParam(m[0].weight,torch.ones(2,4,dtype=torch.bool)))
        return m
    def patch(idx, reset=None):
        m = model()
        pm = m[0].parametrizations.weight[0]
        pm.M.data.view(-1)[idx] = 1
        state = E.export_patch(m)
        if reset is not None:
            state['entries']['0']['unit_reset_rows'] = torch.tensor([reset])
        return state
    a,b = patch(0,0),patch(6,1)
    left,right = model(),model()
    edit_bank(left,a); edit_bank(left,b); edit_bank(left,a,True); edit_bank(right,b)
    disjoint = E.weight_hashes(left) == E.weight_hashes(right)
    overlap = patch(0)
    bad,ref = model(),model()
    edit_bank(bad,a); edit_bank(bad,overlap); edit_bank(bad,a,True); edit_bank(ref,overlap)
    detected = E.weight_hashes(bad) != E.weight_hashes(ref)
    # A code in B makes the value token win a deterministic CPU decoder.
    planted = model()
    leak = patch(0)
    edit_bank(planted,leak)
    logits = planted(torch.tensor([1.,0.,0.,0.]))
    text = 'PlantedAValue' if logits[0] > 0 else 'unknown'
    flagged = leakage_bound([value_emitted(text,['PlantedAValue'])],[dict(fis=[0])])['leaking_facts'] == 1
    return dict(disjoint_unload=disjoint, planted_leak=flagged, overlapping_unload=detected,
                passed=disjoint and flagged and detected)


@torch.no_grad()
def mechanical_snapshot(branches, probes, dev):
    return dict(weights=E.weight_hashes(branches.model),
                logits=[E.logits_of(branches.model, torch.tensor([p], device=dev)).detach().cpu().clone() for p in probes])


def exact_snapshot(left, right):
    weights = left['weights'] == right['weights']
    logits = all(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                 b.contiguous().view(torch.uint8)) for a,b in zip(left['logits'], right['logits'])) and len(left['logits']) == len(right['logits']) == 3
    return dict(weights_bit_identical=weights, revoke_logits_bit_identical=logits, passed=weights and logits)


@torch.no_grad()
def score_configuration(branches, tok, facts, groups, panels, panel, dev, args):
    result = dict(probes={})
    stored_values = [f.val for f in facts if f.attr not in ("sister's name", 'sister')]
    for kind, items in groups.items():
        gens = E.greedy_generate(branches.model, [it['p'] for it in items], dev, 64, 1)
        if kind == 'unknowns' or kind.endswith('/canaries'):
            # Unknown-query rate covers the entire world; fact-leak bounds below
            # retain their separate loaded-fact scope in gates().
            values = [stored_values if kind == 'unknowns' else it['canary_values'] for it in items]
            flags = [value_emitted(tok.decode(gen), vals)
                     for gen, vals in zip(gens, values)]
            result['probes'][kind] = dict(n=len(items), emitted_rate=sum(flags)/max(len(flags),1), emitted=flags,
                query_names=[it.get('query_name') for it in items],
                emitted_fact_ids=[[fi for fi,f in enumerate(facts) if f.pid in (0,1) and f.attr not in ("sister's name",'sister') and D.contains(tok.decode(gen),f.val)] for gen in gens],
                per_item=[dict(fis=it['fis'], prompt=it['prompt'], family=it.get('family', it.get('kind')),
                               emitted=flag, text=tok.decode(gen)) for it, flag, gen in zip(items, flags, gens)])
        else:
            result['probes'][kind] = D.summary(tok, items, gens, args.gen_stop_k, facts)
            for item, gen, row in zip(items, gens, result['probes'][kind]['per_item']):
                fact = facts[item['fis'][-1]]
                row['confused'] = len(item['vals']) == 1 and value_emitted(tok.decode(gen),
                    E.conflict_map(facts).get(item['fis'][-1], []))
            if kind == 'reverse_lookup':
                result['probes'][kind]['name_emission_rate'] = float(np.mean([value_emitted(tok.decode(gen),item['vals']) for item,gen in zip(items,gens)]))
            if kind == 'both/forward_chain':
                result['probes'][kind]['n_relations'] = len({it['relation_fi'] for it in items})
    result['off_topic_nll'] = {name: E.nll_windows(branches.model, wins.to(dev), dev, bs=1) for name, wins in panels.items()}
    if panel is None:
        result['in_topic'] = None
        return result
    flags, answers = I.qa_flags(branches.model, tok, panel['qa'], dev, E.BOS, bs=1)
    result['in_topic'] = dict(qa=flags, answers=answers,
                             text=I.text_scores(branches.model, panel['windows'], panel['mention_masks'], dev))
    return result


def decision(mean, upper, limit=.01, kill=None):
    killed = kill if kill is not None else mean is not None and mean > limit+1e-12
    if killed:
        return 'kill'
    return 'not-killed' if upper is not None and upper <= limit+1e-12 else 'promotion-blocked'


def topic_bounds(panel, base, row, draws):
    if panel is None:
        return None
    raw = panel['raw']
    # DM's 40-pair readiness is separate from CC1's concept-event screen.
    if not raw.get('gate_ready') and not (raw.get('census_scan',{}).get('complete') and raw.get('calibration')):
        return None
    ids = [i for i,r in enumerate(panel['qa']) if r['base_correct'] and r.get('stratum','value') == 'value'
           and not r.get('answer_is_value',False)]
    items = [dict(fis=[panel['qa'][i]['concept']]) for i in ids]
    qa = loss_bound([base['qa'][i] for i in ids],[row['qa'][i] for i in ids],items,draws)
    qa['n_strings'] = len(ids)
    qa['template_mean_loss'] = qa['mean']
    qa['mean'] = qa['facts_with_loss']/qa['clusters'] if qa['clusters'] else None
    qa['decision'] = decision(qa['mean'],qa['upper'])
    text = I.text_report(base['text'],row['text'])
    upper = text['dnll'] + 1.645*text['dnll_se'] if text['dnll_se'] is not None else None
    return dict(knowledge_qa=qa,text_dnll=dict(mean=text['dnll'],upper=upper,clusters=text['n'],
        decision=decision(text['dnll'],upper,.003)))


def gates(table, groups, panel, draws=5000, paired_seeds=False, repaired=False, unseen_pair=False):
    retention,leakage,topic,unknown = {},{},{},{}
    for bank in ('A','B'):
        for kind in ('direct','unseen'):
            key = bank+'/'+kind
            bound = loss_bound([r['correct'] for r in table[bank]['probes'][key]['per_item']],
                [r['correct'] for r in table['AB']['probes'][key]['per_item']],groups[key],draws)
            bound['decision'] = decision(bound['mean'],bound['upper'])
            retention[key] = bound
    for config in ('B','AB-A'):
        bound = leakage_bound(table[config]['probes']['A/canaries']['emitted'],groups['A/canaries'],draws)
        bound['decision'] = decision(bound['rate'],bound['upper'],kill=bound['leaking_facts'] >= 2)
        leakage[config+'/A'] = bound
    for config,row in table.items():
        probe = row['probes']['unknowns']
        names = probe.get('query_names',[it.get('query_name') for it in groups['unknowns']])
        valid = len(names) >= 1000 and len(set(names)) == len(names) and None not in names
        resident = {'base':(), 'A':('A',), 'B':('B',), 'AB':('A','B'),
                    'AB-A':('B',), 'AB-B':('A',), 'BA':('A','B')}[config]
        loaded = set(fi for bank in resident for it in groups[bank+'/direct'] for fi in it['fis'])
        emitted = set(fi for ids in probe.get('emitted_fact_ids',[]) for fi in ids) & loaded
        upper = B.R.binomial_bound(len(emitted),len(loaded)) if loaded else None
        unknown[config] = dict(mean=len(emitted)/len(loaded) if loaded else None,
            probe_rate=probe['emitted_rate'],upper=upper,clusters=len(loaded),
            queries=len(names), emitting_facts=len(emitted), served='forced-bank = all-loaded',
            decision=(decision(probe['emitted_rate'],upper) if loaded else
                      'kill' if probe['emitted_rate'] > .01 else 'not-killed') if valid else 'unavailable')
    for config in ('A','B','AB'):
        topic[config] = topic_bounds(panel,table['base']['in_topic'],table[config]['in_topic'],draws)
    decisions = [v['decision'] for v in [*retention.values(),*leakage.values(),*unknown.values()]]
    decisions += [v['decision'] for t in topic.values() if t for v in t.values()]
    kills = [k for group in (retention,leakage,unknown) for k,v in group.items() if v['decision']=='kill']
    kills += [c+'/'+k for c,t in topic.items() if t for k,v in t.items() if v['decision']=='kill']
    promoted = (not kills and all(d=='not-killed' for d in decisions) and all(topic.values())
                and paired_seeds and (not repaired or unseen_pair))
    return dict(retention=retention,canary_leakage=leakage,in_topic=topic,unknown_emission=unknown,kills=kills,
        promoted=promoted,verdict='kill' if kills else 'not-killed' if promoted else 'promotion-blocked',
        reasons=[] if paired_seeds else ['two paired seeds required'])


def run_contract(factory, states, tok, facts, groups, panels, panel, args):
    reject_overlap(states)
    table,snapshots = {},{}
    probes = revoke_prompts(tok,args.eval_set,E.BOS)
    control = positive_controls()
    for config,sequence in CONFIGURATIONS.items():
        branches,dev = factory(states)
        try:
            activate_sequence(branches,sequence)
            snapshots[config] = mechanical_snapshot(branches,probes,dev)
            table[config] = score_configuration(branches,tok,facts,groups,panels,panel,dev,args)
            if config=='base' and panel and table[config]['in_topic']['qa'] != [r['base_correct'] for r in panel['qa']]:
                raise ValueError('frozen DM base flags changed under CC1 protocol')
        finally:
            branches.activate(())
            del branches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    checks = {name:exact_snapshot(snapshots[name],snapshots[ref]) for name,ref in
              [('AB-A','B'),('AB-B','A'),('BA','AB')]}
    verdict = gates(table,groups,panel,args.bootstrap,repaired=args.repaired)
    if not all(v['passed'] for v in checks.values()):
        verdict.update(verdict='kill',promoted=False)
    if not control['passed']:
        verdict.update(verdict='unavailable',promoted=False,void=True)
    for config,row in table.items():
        row['off_topic_damage'] = {}
        for name,values in row['off_topic_nll'].items():
            delta = np.asarray(values)-table['base']['off_topic_nll'][name]
            row['off_topic_damage'][name] = dict(dnll=float(delta.mean()),n=len(delta),
                paired_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),advisory=True)
    report = dict(protocol_version='C61', table=table,mechanical=checks,note=MECHANICAL_NOTE,positive_control=control,gates=verdict,
                  fresh_model_per_configuration=True,activation_cache_reused=False)
    report['summary_table'] = summary_table(table,panel,verdict,checks,args.repaired)
    return report


def summary_table(table, panel, verdict=None, mechanical=None, repaired=False):
    rows = []
    def add(config,panel,n,g,estimate,upper,decision):
        rows.append(dict(configuration=config,panel=panel,n_strings=n,n_clusters=g,estimate=estimate,
                         upper95=upper,decision=decision))
    if verdict:
        for key,v in verdict['retention'].items():
            add('AB',key,v.get('n',table['AB']['probes'][key]['n']),v['clusters'],v['mean'],v['upper'],v['decision'])
        for key,v in verdict['canary_leakage'].items():
            config,bank = key.split('/')
            add(config,bank+'/canaries',v['probes'],v['clusters'],v['rate'],v['upper'],v['decision'])
        for config,v in verdict['unknown_emission'].items():
            add(config,'unknowns',v['queries'],v['queries'],v['probe_rate'],None,
                'unavailable' if v['decision']=='unavailable' else 'kill' if v['probe_rate'] > .01 else 'not-killed')
            add(config,'unknown/fact_leak',v['queries'],v['clusters'],v['mean'],v['upper'],v['decision'])
        for config,t in verdict['in_topic'].items():
            for name in ('knowledge_qa','text_dnll'):
                v = t[name] if t else {}
                add(config,name,v.get('n_strings',0) if name=='knowledge_qa' else v.get('clusters',0),
                    v.get('clusters',0),v.get('mean'),v.get('upper'),v.get('decision','unavailable'))
    for config,result in table.items():
        for name,score in result['probes'].items():
            if name.startswith('both/') or name in ('conflicting_identifiers','reverse_lookup'):
                add(config,name,score['n'],len({fi for r in score['per_item'] for fi in r['fis']}),
                    score.get('name_emission_rate',score['correct']),None,'not-killed')
                if name == 'conflicting_identifiers':
                    add(config,'confusion',score['n'],len({fi for r in score['per_item'] for fi in r['fis']}),
                        float(np.mean([r['confused'] for r in score['per_item']])),None,'not-killed')
        for name,v in result['off_topic_damage'].items():
            add(config,'off_topic/'+name,v.get('n',40),v.get('n',40),v['dnll'],
                v['dnll']+1.645*v['paired_se'], 'not-killed')
    for config,v in (mechanical or {}).items():
        add(config,'mechanical',3,0,int(not v['passed']),None,'not-killed' if v['passed'] else 'kill')
    add('AprimeB','replacement',0,0,None,None,'unavailable')
    add('repaired-unseen','independent bank C',0,0,None,None,'unavailable')
    return rows


def combine_reports(reports, draws=5000, unseen_reports=None, as_unseen=False):
    """Keep every seed's panels and denominators intact. One seed can kill."""
    if len(reports)!=2 or {r['seed'] for r in reports}!={0,1}:
        raise ValueError('promotion requires the paired seeds 0 and 1')
    left,right = reports
    for key in ('family','training_identity','panel_sha256','repaired'):
        if left.get(key)!=right.get(key):
            raise ValueError('paired CC1 reports differ: '+key)
    for bank in ('0','1'):
        if left['inputs'][bank]['sha256']==right['inputs'][bank]['sha256']:
            raise ValueError('paired seeds must use different trained files')
    if unseen_reports is not None or as_unseen:
        raise ValueError('unseen repaired pairing unavailable until independent bank C and training-step provenance exist')
    clean = all(v['passed'] for r in reports for v in r['mechanical'].values())
    valid = all(r['positive_control']['passed'] for r in reports)
    killed = not clean or any(r['gates']['verdict']=='kill' for r in reports)
    rows = [dict(row,seed=r['seed']) for r in reports for row in r['summary_table']]
    required = {('AB',bank+'/'+kind) for bank in ('A','B') for kind in ('direct','unseen')}
    required |= {(config,'A/canaries') for config in ('B','AB-A')}
    required |= {(config,kind) for config in ('A','B','AB') for kind in ('knowledge_qa','text_dnll')}
    required |= {(config,'mechanical') for config in ('AB-A','AB-B','BA')}
    required |= {(config,'unknown/fact_leak') for config in CONFIGURATIONS}
    complete = all(r.get('protocol_version') == 'C61' and set(r.get('table',{})) == set(CONFIGURATIONS)
                   and required <= {(v['configuration'],v['panel']) for v in r['summary_table']} for r in reports)
    promoted = complete and valid and not killed and not left.get('repaired') and all(
        row['decision'] not in ('kill','promotion-blocked','unavailable')
        for row in rows if row['panel'] not in ('replacement','independent bank C'))
    return dict(seeds=[0,1],family=left['family'],seed_reports=reports,summary_table=rows,
        gates=dict(verdict='unavailable' if not valid else 'kill' if killed else 'not-killed' if promoted else 'promotion-blocked',
                   promoted=promoted,void=not valid,evidence_complete=complete),note=MECHANICAL_NOTE,
        uncertainty='per-seed panels; no pooled n; one seed can kill')


def paired_main(argv):
    parser = argparse.ArgumentParser(description='Aggregate existing CC1 screens for paired seeds 0/1')
    parser.add_argument('--reports', type=Path, nargs=2, required=True)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--unseen-pair-reports', type=Path, nargs=2, help='paired screens absent from the frozen repair-training pairing manifest')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = combine_reports([json.loads(p.read_text()) for p in args.reports], args.bootstrap,
        [json.loads(p.read_text()) for p in args.unseen_pair_reports] if args.unseen_pair_reports else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print('CC1-PAIRED '+json.dumps(report), flush=True)
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    E.eda.add_argument(p)
    p.add_argument('--export', type=Path, default=E.EXPORT)
    p.add_argument('--export-step', type=int, default=E.EXPORT_STEP)
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, required=True)
    p.add_argument('--a-prime', type=Path, help='legacy option; replacement is unavailable and is not scored')
    p.add_argument('--in-topic-panel', type=Path)
    p.add_argument('--eval-set', type=Path, required=True)
    p.add_argument('--fineweb', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--qa-n', type=int, default=120)
    p.add_argument('--minimum-probes', type=int, default=1000)
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--repaired', action='store_true', help='J/F-repaired banks; requires an additional unseen pairing before promotion')
    p.add_argument('--repair-training-pairings', type=Path, help='frozen JSON list of [A_sha256,B_sha256] pairings used by repair training, across both seeds')
    p.add_argument('--out', type=Path, required=True)
    return p


def off_topic_panels(args, bos, dev='cpu'):
    """Keep CC1 labels while using the harness's BOS + 511-token windows."""
    return {name: next(iter(E.extra_eval_panels(str(path), bos, dev).values()))[1]
            for name, path in [('wikitext', args.eval_set), ('fineweb', args.fineweb), ('code', args.code)]}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'paired':
        return paired_main(argv[1:])
    args = build_parser().parse_args(argv)
    args.gen_stop_k, args.gen_bs, args.gen_max_new = 2, 1, 64
    if args.export_step != 58814:
        raise ValueError('CC1 requires export step 58814')
    if E.eda.selected(args) != 'naive_fp32' or args.minimum_probes < 1000 or args.qa_n < 1 or args.bootstrap < 100:
        raise ValueError('CC1 needs naive_fp32, >=1000 probes, positive QA count and >=100 bootstrap draws')
    paths = {0: args.a, 1: args.b}
    states = {k: torch.load(p, map_location='cpu', weights_only=False) for k,p in paths.items()}
    args.repaired = args.repaired or any(s['meta'].get('c52') for s in states.values())
    repair_pairings = json.loads(args.repair_training_pairings.read_text()) if args.repair_training_pairings else None
    if repair_pairings is not None and (not isinstance(repair_pairings, list) or not repair_pairings or
            any(not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(h, str) or len(h) != 64 for h in pair) for pair in repair_pairings)):
        raise ValueError('repair training manifest must list SHA256 pairs')
    try:
        family = validate_pair(states, args)
    except ValueError as error:
        if not str(error).startswith('mechanical-unload failure:'):
            raise
        control = positive_controls()
        family = 'whole-unit' if states[0]['meta'].get('partition_unit') else 'full-mask'
        report = dict(protocol_version='C61', family=family, seed=states[0]['meta']['seed'], repaired=args.repaired,
            positive_control=control, mechanical={'overlap':dict(passed=False)}, overlap=B.overlap_stats(list(states.values())),
            gates=dict(verdict='kill' if control['passed'] else 'unavailable', promoted=False, void=not control['passed'],
                       kills=['overlap'], reasons=[str(error)]),
            summary_table=[dict(configuration='AB',panel='mechanical overlap',n_strings=0,n_clusters=0,
                estimate=1,upper95=None,decision='kill')]+summary_table({},None),
            inputs={str(k):dict(path=str(p),sha256=E.file_sha(p)) for k,p in paths.items()}, note=MECHANICAL_NOTE)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        print('CC1 '+json.dumps(report),flush=True)
        return report
    E.eda.announce(args)
    meta = states[0]['meta']
    facts = E.make_world(meta['seed'], meta['people'], meta['facts_per_person'], meta['shared_attrs'], meta['values'], True,
                         meta.get('synthetic_people', 0))
    sys.path.insert(0, '/workspace/malleable/mlf')
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(E.DATA/'tokenizer.json'))
    panel = I.load_panel(args.in_topic_panel, args, tok, 'cpu') if args.in_topic_panel else None
    if panel and panel['raw']['version'] != 2:
        raise ValueError('CC1 needs calibrated two-directional DM v2')
    groups = probe_sets(tok, facts, states, E.BOS, args.qa_n, args.minimum_probes)
    panels = off_topic_panels(args, E.BOS)
    def factory(current):
        torch.manual_seed(meta['seed'])
        random.seed(meta['seed'])
        model, _, dev = E.eda.build_model(build_model, args, args.export, device='cuda:0', model_config=None,
                                         step=args.export_step, expert_format='bf16')
        model._fsa_export, model._fsa_export_step = args.export, args.export_step
        model.eval()
        E.attach(model, meta['layers'], 'fsa_forced', 0, sparse=True, projs=meta['projs'],
                 slot_frac=meta['slot_frac'], seed=meta['seed'],
                 projs_by_layer=E.layer_projections(meta['layers'], meta.get('value_layers', meta['layers']), meta['projs']))
        E.apply_projection_scales(model, meta.get('patch_scale_mult', 1.), meta.get('patch_scale_mult_up'), meta.get('patch_scale_mult_down'))
        for state in current.values():
            E.load_patch(model, state)
            if state['meta'].get('slot_partition'):
                for name, lin in E.patched_linears(model):
                    layer = next(li for li,(lname,_) in enumerate(E.moe_layers(model)) if name.startswith(lname+'.'))
                    expert = int(name.rsplit('.experts.', 1)[1].split('.')[0])
                    allowed = E.partition_mask(E.pair8_forced_mask(lin.parametrizations.weight.original.detach()),
                        layer, expert, name.rsplit('.', 1)[1], state['meta']['slot_partition'], state['meta']['slot_part'],
                        state['meta'].get('partition_unit', False))
                    if not bool(allowed.flatten()[state['entries'][name]['idx'].long().to(dev)].all()):
                        raise ValueError('bank writes outside declared ownership: '+name)
        return B.Branches(model, current, {k:p.stat().st_size for k,p in paths.items()}, 'last_write'), dev
    report = run_contract(factory, states, tok, facts, groups, panels, panel, args)
    report.update(family=family, seed=meta['seed'], eda_backend='naive_fp32',
                  repaired=args.repaired,
                  ownership={k: {key: state['meta'].get(key) for key in ('train_person', 'slot_partition', 'slot_part',
                              'partition_unit', 'mask_sha256', 'base_altering')} for k,state in states.items()},
                  repair_training_pairings=repair_pairings,
                  repair_manifest_sha256=E.file_sha(args.repair_training_pairings) if args.repair_training_pairings else None,
                  training_identity={key: meta.get(key) for key in B.IDENTITY if key != 'seed'},
                  probe_clusters={key: [it['fis'][-1] for it in items] for key,items in groups.items()},
                  panel_manifest=dict(knowledge_qa=panel['qa'], gate_ready=panel['raw'].get('gate_ready', False)) if panel else None,
                  replacement_label='unavailable',
                  inputs={k: dict(path=str(p), sha256=E.file_sha(p)) for k,p in paths.items()},
                  panel_sha256=E.file_sha(args.in_topic_panel) if args.in_topic_panel else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print('CC1 '+json.dumps(dict(summary_table=report['summary_table'], mechanical=report['mechanical'],
                               gates=report['gates'], output=str(args.out))), flush=True)
    return report


if __name__ == '__main__':
    main()
