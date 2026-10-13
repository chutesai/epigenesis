"""GPU feature dumping and selection-in-the-loop GEN for C42 (opt-in only)."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

try:
    from . import c42_selector as C
except ImportError:
    import c42_selector as C


def patch_manifest(paths, eda_backend=C.eda.DEFAULT):
    rows = []
    for path in paths:
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                digest.update(block)
        rows.append(dict(path=str(path),sha256=digest.hexdigest(),eda_backend=eda_backend))
    return sorted(rows,key=lambda row: row['sha256'])


def query_splits(B,tok,facts,owners):
    """Distinct name, pronoun and indirect phrasings in CAL/VAL/TEST.

    Wrong-scope nulls are requests explicitly restricted to a person's OTHER
    attribute group. Ordinary queries about another resident bank are positives.
    """
    E = B.E
    panels = B.S.build_panels(tok,facts,E.BOS)
    splits = {name:list(panels[name]) for name in ('train','val','test')}
    # Augmented LM write prompts aren't key-fitting examples here; CAL is held
    # out from the selector TRAIN examples, even if a frozen patch saw them.
    cal = E.build_items(tok,facts,E.TRAIN_AUG_T[len(E.TRAIN_T):],E.BOS)
    for it in cal:
        it['p'] = B.question_sequence(tok,it['p'],E.BOS)
    splits['cal'] = cal
    templates = {
        'cal': ('{name} is my colleague. Can you recall {pos} {attr}?',
                'For the person whose {ref_attr} is {ref_val}, tell me {pos} {attr}.'),
        'val': ('My neighbour is {name}. What is {pos} {attr}?',
                'The person with {ref_val} as {pos} {ref_attr}: what is {pos} {attr}?'),
        'test': ('My friend {name} called today. What is {pos} {attr}?',
                 'Identify the {attr} of the person whose {ref_attr} is {ref_val}.')}
    unique_values = Counter(f.val for f in facts)
    for name in ('cal','val','test'):
        for it in splits[name]:
            it['form'] = 'name'
        for fi,f in enumerate(facts):
            refs = sorted((r for r in facts if r.pid == f.pid and r.attr != f.attr and unique_values[r.val] == 1),key=lambda r:r.attr)
            for form,template in zip(('pronoun','indirect'),templates[name]):
                if form == 'indirect' and not refs:
                    continue
                prompt = E.fmt(template.replace('{ref_attr}',refs[0].attr).replace('{ref_val}',refs[0].val),f) if refs else E.fmt(template,f)
                splits[name].append(dict(fi=fi,ti=100 if form == 'pronoun' else 101,attr=f.attr,val=f.val,
                    p=E.encode(tok,prompt,E.BOS),a=[],prompt=prompt,form=form))
    # Templates are distinct across splits and never contain taught target values.
    negative_templates = {
        'cal': 'Please look up {name}: what is the recorded {attr}?',
        'val': 'Which {attr} is listed for {name}?',
        'test': 'Recall the {attr} belonging to {name}.'}
    known = {(f.name,f.attr) for f in facts}
    for name,template in negative_templates.items():
        for fi,f in enumerate(facts):
            absent = next((a for a,_ in E.ATTR_POOL if (f.name,a) not in known),'untaught profile entry')
            for kind,person,attr in [('untaught_entity','NeverTaughtPerson'+str(fi),f.attr),
                                     ('untaught_attribute',f.name,absent)]:
                prompt = template.format(name=person,attr=attr)
                splits[name].append(dict(fi=-1,p=E.encode(tok,prompt,E.BOS),form=kind,prompt=prompt))
            sibling_banks = sorted({owners[j] for j,g in enumerate(facts) if g.pid == f.pid and owners[j] != owners[fi]})
            scopes = sibling_banks or sorted(set(owners.values())-{owners[fi]})
            if scopes:
                # Explicit scope prevents labeling valid resident facts as nulls.
                scope = scopes[0]
                attrs = ', '.join(sorted({facts[j].attr for j in owners if owners[j] == scope}))
                scope_person = next(facts[j].name for j in owners if owners[j] == scope)
                prompt = f'Use only the profile group [{attrs}] of {scope_person}. '+template.format(name=f.name,attr=f.attr)
                splits[name].append(dict(fi=-1,p=E.encode(tok,prompt,E.BOS),form='wrong_scope',prompt=prompt))
    for it in splits['train']:
        it['form'] = 'name'
    for name,items in splits.items():
        # Repeated absent attributes across facts are one negative query.
        splits[name] = list({tuple(it['p']):it for it in items}.values())
    return splits


def dump_features(B,a,model,tok,facts,states,owners,dev):
    E,R = B.E,B.R
    layers = [int(v) for v in a.c42_layers.split(',')]
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= len(model.layers):
        raise ValueError('C42 layers must be distinct valid full-block inputs')
    groups = query_splits(B,tok,facts,owners)
    anchor = json.loads(a.anchor.read_text())
    sources = {str(p):json.loads(p.read_text()) for p in [a.eval_set,a.fineweb,a.code]+a.negative_text}
    E.checked_overlap(anchor,sources,'C42 anchor versus null sources')
    if a.negative_n < 3000:
        raise ValueError('C42 --negative-n must request at least 3000 general nulls per held-out split')
    # Reserve preservation windows; then allocate disjoint, deduplicated queries.
    nulls = R.independent_queries(sources,E.BOS,3*a.negative_n)
    per_split = min(a.negative_n,len(nulls)//3)
    if not per_split:
        raise ValueError('C42 needs at least three disjoint general windows')
    negative_sample = dict(requested_per_split=a.negative_n,actual_per_split=per_split,
        available_up_to_requested=len(nulls),corpus_limited=per_split < a.negative_n,
        allocation='equal disjoint CAL/VAL/TEST general windows; no resampling')
    for i,name in enumerate(('cal','val','test')):
        groups[name].extend(dict(it,fi=-1,form='general') for it in nulls[i*per_split:(i+1)*per_split])
    windows,positions,_ = B.S.general_windows(anchor,next(iter(states.values()))['meta']['seed'],E.BOS)
    sequences,offsets = [], {}
    for name,items in groups.items():
        offsets[name] = (len(sequences),len(items))
        sequences.extend(it['p'] for it in items)
    start_general = len(sequences)
    sequences.extend(windows)
    capture_positions = [len(s)-1 for s in sequences[:start_general]]+list(positions)
    enabled = E.FSAParam.enabled
    try:
        E.set_enabled(False)
        # All requested layers are captured together: each sequence sees one base forward.
        features = B.S.collect_residuals(model,sequences,layers,dev,capture_positions,progress='C42-DUMP')
    finally:
        E.set_enabled(enabled)
    cache = dict(eda_backend=E.eda.selected(a), format=C.FORMAT,layers=layers,fact_owner=owners,
        bank_parent={b:s['meta']['train_person'] for b,s in states.items()},splits={},
        general={l:x[start_general:].cpu() for l,x in features.items()},
        patches=patch_manifest(a.patches, E.eda.selected(a)),export=str(a.export),export_step=a.export_step,
        metadata=next(iter(states.values()))['meta'],layer_mapping=B.layer_mapping(model),negative_sample=negative_sample,
        provenance='frozen base; final prompt token; all layers captured in one pass; disjoint query templates/windows')
    for name,(start,n) in offsets.items():
        items = groups[name]
        cache['splits'][name] = dict(features={l:x[start:start+n].cpu() for l,x in features.items()},
            labels=torch.tensor([owners[it['fi']] if it['fi'] >= 0 else -1 for it in items]),
            fact_ids=torch.tensor([it['fi'] for it in items]),forms=[it['form'] for it in items],items=items)
    C.validate_cache(cache)
    a.c42_cache.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cache,a.c42_cache)
    return dict(mode='dump',cache=str(a.c42_cache),layers=layers,
        banks=len(states),counts={k:len(v['labels']) for k,v in cache['splits'].items()},
        negative_sample=negative_sample,lm_training=False)


def load_model(B,a,meta):
    E = B.E
    E.eda.require_backend(meta, E.eda.selected(a), "C42 model banks")
    sys.path.insert(0,'/workspace/malleable/mlf')
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(E.DATA/'tokenizer.json'))
    model,_,dev = E.eda.build_model(build_model,a,a.export,device='cuda:0',model_config=None,step=a.export_step,
                             expert_format='bf16')
    model._fsa_export,model._fsa_export_step = a.export,a.export_step
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if a.fast_moe == 'exact':
        if __package__:
            from .fast_moe import install
        else:
            from fast_moe import install
        install(model,patched_layer_ids=meta['layers'],mode='exact')
    return model,tok,dev


def run(B,a):
    if not a.patches or a.c42_cache is None or a.out is None:
        raise ValueError('C42 requires --patches --c42-cache --out')
    states_list = [torch.load(p,map_location='cpu',weights_only=False) for p in a.patches]
    banked = states_list[0]['meta'].get('attr_groups',1) > 1
    states = B.validate_patches(states_list,banked=banked,eda_backend=B.E.eda.selected(a))
    for s in states_list:
        B.E.validate_patch_export(s['meta'],a.export,a.export_step)
    meta,E = states_list[0]['meta'],B.E
    if meta['method'] != 'fsa_forced' or meta['fmt'] != 'ternary':
        raise ValueError('C42 requires forced-mask ternary banks')
    facts = E.make_world(meta['seed'],meta['people'],meta['facts_per_person'],meta['shared_attrs'],
                        meta['values'],meta.get('relation_world',False),meta.get('synthetic_people',0))
    if banked:
        try:
            from . import hierarchical_banks as H
        except ImportError:
            import hierarchical_banks as H
        owners = H.fact_banks(E,facts,states)
    else:
        owners = {fi:f.pid for fi,f in enumerate(facts)}
    cache = None
    if a.c42_mode == 'gen':
        cache = torch.load(a.c42_cache,map_location='cpu',weights_only=False)
        B.E.eda.require_backend(cache, B.E.eda.selected(a), "C42 GEN cache")
        C.validate_cache(cache)
        cached_patches = [dict(p, eda_backend=p.get('eda_backend', E.eda.DEFAULT)) for p in cache['patches']]
        if (cached_patches != patch_manifest(a.patches, E.eda.selected(a)) or cache['export'] != str(a.export)
                or cache['export_step'] != a.export_step or cache['fact_owner'] != owners):
            raise ValueError('GEN cache does not match checkpoints/export/fact ownership')
    elif any(p is None for p in (a.anchor,a.eval_set,a.fineweb,a.code)):
        raise ValueError('feature dump needs --anchor --eval-set --fineweb --code')
    model,tok,dev = load_model(B,a,meta)
    if a.c42_mode == 'dump':
        result = dump_features(B,a,model,tok,facts,states,owners,dev)
    else:
        torch.set_num_threads(1)
        analysis,locked = C.analyze_set(cache,a.h_top_k1,a.c42_parent_gate)
        best = analysis['best_hierarchy']
        test = cache['splits']['test']
        positive = [i for i,b in enumerate(test['labels']) if b >= 0]
        items = [test['items'][i] for i in positive]
        probe = torch.tensor([items[0]['p'][:16]],device=dev)
        # Snapshot before attachment/loading, so controls reference the pristine base.
        with torch.no_grad():
            base = E.logits_of(model,probe).clone()
        base_weights = E.weight_hashes(model,states_list[0]['entries'])
        E.attach(model,meta['layers'],'fsa_forced',0,projs=meta['projs'],sparse=True,
                 slot_frac=meta['slot_frac'],seed=meta['seed'],
                 projs_by_layer=E.layer_projections(meta['layers'],meta.get('value_layers',meta['layers']),meta['projs']))
        E.apply_projection_scales(model,meta.get('patch_scale_mult',1.),meta.get('patch_scale_mult_up'),meta.get('patch_scale_mult_down'))
        sizes = {b:sum(p.stat().st_size for p,s in zip(a.patches,states_list) if s is states[b]) for b in states}
        branches = B.Branches(model,states,sizes,a.merge)
        controls = dict(solo_bank=True)
        for b,s in states.items():
            branches.activate((b,))
            with torch.no_grad():
                served = E.logits_of(model,probe).clone()
                E.load_patch(model,s)
                controls['solo_bank'] &= torch.equal(served,E.logits_of(model,probe))
        result = dict(mode='gen',routing_analysis=analysis,best_hierarchy_selected_on='VAL',arms={})
        for arm in dict.fromkeys(['flat_llr',best]):
            if arm is None:
                continue
            trace = C.route_locked(locked[arm],test,arm)
            routes = [trace['routes'][i] for i in positive]
            # Decision is fixed for the whole generated answer and patch is gated, never blended.
            gens = B.routed_generations(branches,tok,items,routes,dev,a.gen_max_new or 64,a.gen_bs)
            result['arms'][arm] = dict(gen=B.generation_summary(tok,items,gens,E.conflict_map(facts),a.gen_stop_k),
                active_bytes=sum(branches.active_bytes(r) for r in routes)/len(routes),selection=analysis['arms'][arm]['metrics'])
        branches.activate(())
        with torch.no_grad():
            controls['empty_bank'] = torch.equal(base,E.logits_of(model,probe))
        E.set_enabled(False)
        with torch.no_grad():
            controls['revoked_bit_exact'] = torch.equal(base,E.logits_of(model,probe))
        controls['base_weights_bit_exact'] = E.weight_hashes(model,base_weights) == base_weights
        E.set_enabled(True)
        result['controls'] = controls
        result['verdict_suppressed'] = not all(controls.values())
        if result['verdict_suppressed']:
            raise RuntimeError('C42 GEN serving control failed')
    result["eda_backend"] = E.eda.selected(a)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('C42 '+json.dumps(dict(eda_backend=E.eda.selected(a), mode=a.c42_mode,out=str(a.out))))
    return result
