"""Owner-requested EP-1 session ownership side test; existing arms are untouched."""
import argparse
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

import torch

import consolidate as c
import epi_gpu as g
import evaluate as e
import make_user as u
import selfstudy as s
from epi_common import claim_hash


def ownership_masks(a_val, b_val):
    a, b = a_val.cpu().to(torch.int8), b_val.cpu().to(torch.int8)
    a_slots, writes = a != 0, b != a
    return dict(A_slots=a_slots, B_writes=writes, anti_A=a_slots & writes,
                overlap=a_slots & (b != 0) & writes, A_last=a_slots & ~writes)


def ownership_ledger(a, b):
    per_module = {name: {key: int(mask.sum()) for key, mask in
                        ownership_masks(a[name]['committed'], b[name]['committed']).items()}
                  for name in a}
    totals = {key: sum(row[key] for row in per_module.values()) for key in
              ('A_slots', 'B_writes', 'anti_A', 'overlap', 'A_last')}
    totals['anti_A_fraction_of_B_writes'] = totals['anti_A'] / totals['B_writes'] if totals['B_writes'] else 0.
    return dict(totals=totals, per_module=per_module)


def revoke_mask(a_val, current, ownership=True):
    return (a_val != 0) & ((current == a_val) if ownership else torch.ones_like(a_val, dtype=torch.bool))


def revoke_state(a, b, base, blocks, ownership=True):
    result = copy.deepcopy(b)
    for name, saved in result.items():
        av, bv = a[name]['committed'], b[name]['committed']
        mask = revoke_mask(av, bv, ownership)
        saved['committed'][mask] = 0
        saved['support'][mask] = False
        saved['frozen'].zero_()
        block = blocks[name]
        # One scale per block: any B code write owns the block, including a zeroing.
        b_block = (bv != av).reshape(av.shape[0], -1, block).any(-1)
        live = (saved['committed'] != 0).reshape(av.shape[0], -1, block).any(-1)
        saved['logb'] = torch.where(b_block, b[name]['logb'], a[name]['logb'])
        saved['logb'] = torch.where(live, saved['logb'], base[name]['logb'])
    return result


def action_of(text):
    match = re.search(r'\bACTION:\s*([^|\r\n]+)', text, re.IGNORECASE)
    return ' '.join(match[1].lower().split()) if match else None


def recall_check(kind, text, truth):
    return action_of(text) == truth.lower() if kind == 'apply' else truth.lower() in text.lower()


def leakage_check(text, old_truth):
    return old_truth.lower() in text.lower()


def build_data(seed=0):
    corpus = u.build_corpus(seed)
    rng = random.Random(seed + 813)
    policies = sorted(corpus['policies'], key=lambda p: p['id'])[:20]
    facts = sorted(corpus['facts'], key=lambda f: f['id'])[:20]
    replaced_p, replaced_f = policies[::2], facts[::2]
    r2p, r2f = copy.deepcopy(replaced_p), copy.deepcopy(replaced_f)
    category_actions = {category: {p['action'] for p in corpus['policies'] if p['category'] == category}
                        for category in u.CATEGORIES}
    for p in r2p:
        used = category_actions[p['category']]
        old = p['action']
        p['action'] = rng.choice([a for a in u.ACTIONS if a not in used])
        used.add(p['action'])
        p.update(rule_text=p['rule_text'].replace(old, p['action']), status=u.STATUS_MAP[p['action']],
                 rationale_text=u.RATIONALES[p['action']])
    used_values = {f['value'] for f in corpus['facts']}
    for f in r2f:
        old = f['value']
        value = f"{old.split()[0]} {rng.getrandbits(96):024x}"
        assert value not in used_values
        used_values.add(value)
        f.update(value=value, declarative=[text.replace(old, value) for text in f['declarative']],
                 qa=dict(prompt=f['qa']['prompt'], answer=value))
    # Ten distinct register subjects outside R1's four subjects, from the same corpus.
    new = copy.deepcopy(sorted(corpus['facts'], key=lambda f: f['id'])[20::5][:10])
    episodes = [ep for session in corpus['sessions'] for ep in session['episodes']]

    def render(label, ps, fs):
        rendered = []
        for p in ps:
            ep = copy.deepcopy(next(ep for ep in episodes if ep['kind'] == 'policy_intro' and ep['policy_id'] == p['id']))
            original = next(q for q in corpus['policies'] if q['id'] == p['id'])
            ep['text'] = ep['text'].replace(u.resolution(original), u.resolution(p)).replace(
                original['rule_text'], p['rule_text']).replace(original['rationale_text'], p['rationale_text'])
            if label == 'B':
                ep['text'] = ep['text'].replace(p['rule_text'], f"Update: from now on, {p['rule_text']}", 1)
            rendered.append(ep)
        for f in fs:
            ep = copy.deepcopy(next(ep for ep in episodes if f['id'] in ep.get('fact_ids', [])))
            original = next(q for q in corpus['facts'] if q['id'] == f['id'])
            ep['text'] = ep['text'].replace(original['value'], f['value'])
            if label == 'B' and f['id'] in {q['id'] for q in r2f}:
                subject = f['subject'].removesuffix(' ' + f['relation'])
                ep['text'] = ep['text'].replace(f['declarative'][0],
                    f"Correction: the {f['relation']} for {subject} is now {f['value']}.")
            rendered.append(ep)
        for i, ep in enumerate(rendered):
            ep['id'] = ep['chunk_id'] = f'{label}-e{i:03d}'
        return rendered

    return dict(corpus=corpus, A=dict(policies=policies, facts=facts, episodes=render('A', policies, facts)),
                B=dict(policies=r2p, facts=r2f + new, episodes=render('B', r2p, r2f + new)),
                replaced_policies=replaced_p, replaced_facts=replaced_f, new_facts=new)


def candidates(data, label, seed, smoke=False):
    session = 1 if label == 'A' else 2
    source = data[label]
    changed = {p['id']: p for p in source['policies']}
    corpus = dict(policies=[changed.get(p['id'], p) for p in data['corpus']['policies']],
                  facts=source['facts'], claim_schedule={},
                  sessions=[dict(session=session, episodes=source['episodes'])],
                  chunks=[dict(session=session, chunk_id=ep['id']) for ep in source['episodes']])
    items = [i for i in s.build_items(corpus, session, seed) if i['kind'] == 'apply']
    # Recurrent register facts are known ground truth; no agreement/model-question verifier.
    for ep in source['episodes']:
        for fid in ep.get('fact_ids', []):
            f = next(f for f in source['facts'] if f['id'] == fid)
            items.append(dict(kind='factqa', chunk_id=ep['id'], source_episode=ep['id'],
                              prompt=f['qa']['prompt'], fact_id=fid, truth_value=f['value'],
                              claim_hash=claim_hash(f['subject'], f['value'])))
    if smoke:
        # Spread the five fact questions across corrected and unrelated new facts.
        facts = [i for i in items if i['kind'] == 'factqa']
        items = [i for i in items if i['kind'] == 'apply'][::2][:5] + facts[::max(1, len(facts) // 5)][:5]
    for i, item in enumerate(items):
        item['id'] = f'{label}-i{i:03d}'
    return items


def make_probes(data, seed):
    rng = random.Random(seed + 991)
    probes = []
    oldp = {p['id']: p for p in data['replaced_policies']}
    oldf = {f['id']: f for f in data['replaced_facts']}
    groups = [('R2_policy', data['B']['policies']), ('R2_fact', data['B']['facts'][:10]),
              ('A_policy', [p for p in data['A']['policies'] if p['id'] not in oldp]),
              ('A_fact', [f for f in data['A']['facts'] if f['id'] not in oldf]),
              ('B_new_fact', data['new_facts'])]
    for group, entries in groups:
        for item in entries:
            if 'action' in item:
                case = u.mint_case(rng, data['corpus']['policies'], u.NAME_POOLS['dev'], item['category'])
                cond = item['condition']
                case[item['attribute']] = rng.randint(cond['min'], cond['max']) if cond['kind'] == 'range' else rng.choice(cond['values'])
                prompt = f"User: Apply {item['codename']} to this new ticket.\n{u.render_ticket(case)}\nResolution:"
                truth, answer, old = item['action'], u.resolution(item), oldp.get(item['id'], {}).get('action')
                kind = 'apply'
            else:
                prompt, truth, answer = item['qa']['prompt'], item['value'], item['value']
                old, kind = oldf.get(item['id'], {}).get('value'), 'factqa'
            probes.append(dict(group=group, kind=kind, prompt=prompt, truth=truth, answer=answer, old_truth=old))
    return probes


def cache_study(model, tok, dev, items, episodes, anchors, args, out, generate_answers=True):
    chunks = {ep['id']: g.enc(tok, ep['text'] + '\n\n') for ep in episodes}
    accepted, nulls = [], []
    rng = random.Random(args.seed)
    for item in items:
        item = dict(item, prompt_ids=g.enc(tok, item['prompt']))
        if generate_answers:
            prefix = [g.BOS] + chunks[item['chunk_id']] + item['prompt_ids']
            answer, ids = g.generate(model, tok, [prefix], dev, max_new=64, batch=1)[0]
            truth = item.get('truth_action', item.get('truth_value'))
            item.update(answer=answer, answer_ids=s.aligned_answer_ids(tok, answer, ids),
                        accepted=recall_check(item['kind'], answer.strip(), truth))
        if not item['accepted'] or not item['answer_ids']:
            continue
        # Preserve the complete no-context training prefix under the fixed micro budget.
        list(c.micro_batches([dict(row=c.item_row(item)[0])], 160))
        accepted.append(item)
        length = len(chunks[item['chunk_id']])
        offset = rng.randrange(len(anchors) - length + 1)
        suffix = item['prompt_ids'] + item['answer_ids']
        unrelated = [g.BOS] + anchors[offset:offset + length] + suffix
        bare = [g.BOS] + suffix
        lp = g.logprobs_at(model, [unrelated], [1 + length + len(item['prompt_ids'])], dev, 160)[0]
        bp = g.logprobs_at(model, [bare], [1 + len(item['prompt_ids'])], dev, 160)[0]
        nulls.append(s.full_kl(lp, bp).cpu())
        del lp, bp
    threshold = s.null_percentiles(nulls)['95']
    c.validate_acquisition(accepted, threshold)
    teacher, start, acquisition, dropped = c.cache_items(model, dev, accepted, chunks, args, threshold)
    c.validate_acquisition(accepted, threshold, acquisition)
    e.write_json(out / 'selfstudy.json', dict(items=accepted, candidates=len(items), null_thr=threshold,
                                            dropped_zero_gate=dropped, chunk_tokens=chunks))
    torch.save(dict(teacher={k: v.state_dict() for k, v in teacher.items()},
                    start={k: v.state_dict() for k, v in start.items()},
                    acquisition=[dict(i, record=i['record'].state_dict()) for i in acquisition]), out / 'records.pt')
    return accepted, acquisition


def weight_hashes(model):
    # Hash one effective bf16 projection at a time rather than keeping another dense model.
    with torch.no_grad():
        hashes = {}
        for name, lin in model.named_modules():
            if hasattr(lin, 'parametrizations') and hasattr(lin.parametrizations, 'weight'):
                value = lin.weight.detach().to(torch.bfloat16).cpu().contiguous()
                buffer = (ctypes.c_ubyte * (value.numel() * value.element_size())).from_address(value.data_ptr())
                hashes[name] = hashlib.sha256(buffer).hexdigest()
        return hashes


def save_committed(path, model, modules, frozen, state=None):
    before = weight_hashes(model) if state is None else None
    if state is None:
        state = c.snapshot(modules, frozen, committed=True)
        for saved in state.values():
            saved['committed'] = saved['committed'].to(torch.int8)
    c.restore(modules, state, frozen)
    hashes = weight_hashes(model)
    if before is not None:
        assert before == hashes, 'Commit changed effective bf16 weights'
    torch.save(dict(modules=state, bf16_hashes=hashes), path)


def load_committed(path, model, modules, frozen):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    c.restore(modules, saved['modules'], frozen)
    assert weight_hashes(model) == saved['bf16_hashes'], 'Effective bf16 weights differ from saved state'
    return saved['modules']


def train(model, dev, modules, params, frozen, acquisition, anchors, args, steps, out, lam=None, general=None):
    rng = random.Random(args.seed)
    # Cache general-text KL at this training phase's start individual.
    rows = [[g.BOS] + anchors[(offset := rng.randrange(len(anchors) - 95)):offset + 95] for _ in range(64)]
    if general is None:
        records = g.topk_records(model, rows, [1] * len(rows), dev, 32, 160)
        general = [dict(row=row, start=1, record=rec) for row, rec in zip(rows, records)]
    torch.save([dict(entry, record=entry['record'].state_dict()) for entry in general], out / 'general.pt')
    mods = list(modules.values())
    den = g.fsa.collect_col_energy(model, mods, torch.tensor(rows[:32], device=dev))
    if not 0 < den < float('inf'):
        raise ValueError('Invalid anchor energy denominator')
    allowed = dict.fromkeys(modules, True)  # All slots labile: B may overwrite A.
    scales = [m.logb for m in mods]
    scale_ids = {id(p) for p in scales}
    opt = torch.optim.AdamW([dict(params=[p for p in params if id(p) not in scale_ids], lr=.02),
                             dict(params=scales, lr=.002)], weight_decay=0)
    for m in mods:
        m.score.zero_()
    with (out / 'steps.jsonl').open('w') as log:
        for step in range(1, steps + 1):
            tic = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            opt.zero_grad(set_to_none=True)
            terms = dict(acq=c.sample_episodes(acquisition, 8, rng),
                         gen=[rng.choice(general) for _ in range(2)], rep=[])
            _, losses = c.loss_assembly(lambda x: g.logits_of(model, x), terms, dev, g.VOCAB,
                                        'balanced', True, 160, backward=True)
            c.protect_gradients(modules, frozen, allowed)
            for m in mods:
                m.score.mul_(.9)
                if m.M.grad is not None:
                    m.score.add_(m.M.grad, alpha=.1)
            opt.step()
            c.manual_decay(modules, frozen, allowed, .02, .3)
            births = evictions = 0
            # Final refresh also exercises births in 4-step smoke / 2-step repair.
            if step % 8 == 0 or step == steps:
                if lam is None:
                    lam = c.calibrate_lambda(mods, den, 2000000)
                if lam is not None:
                    births, evictions = c.refresh(modules, frozen, allowed, 2000000, lam, den, opt)
            row = dict(phase=out.name, step=step, **losses, births=births, evictions=evictions,
                       seconds=time.monotonic() - tic, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            log.write(json.dumps(row) + '\n'); log.flush()
            print(json.dumps(row), flush=True)
    model.zero_grad(set_to_none=True)
    del opt
    torch.cuda.empty_cache()
    return lam


def metrics(model, tok, dev, probes, windows, genome):
    groups = {}
    outputs = []
    for probe in probes:
        prefix = [g.BOS] + g.enc(tok, probe['prompt'])
        text, _ = g.generate(model, tok, [prefix], dev, max_new=64, batch=1)[0]
        em, _ = g.exact_match(model, tok, prefix, probe['answer'], dev)
        row = dict(probe, generated=text, recall=recall_check(probe['kind'], text, probe['truth']),
                   teacher_forced_em=em, leakage=bool(probe['old_truth'] and leakage_check(text, probe['old_truth'])))
        outputs.append(row)
        groups.setdefault(probe['group'], []).append(row)
    means = {key: dict(n=len(rows), **{metric: sum(r[metric] for r in rows) / len(rows)
                                     for metric in ('recall', 'teacher_forced_em', 'leakage')})
             for key, rows in groups.items()}
    return dict(recall=means, probes=outputs,
                panel=e.panel_metrics(e.wikitext_panel(model, windows, dev), genome), slots=e.slot_stats(model))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Run name under out/, or an explicit run directory')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=48)
    parser.add_argument('--repair-steps', type=int, default=16)
    parser.add_argument('--smoke', action='store_true')
    cli = parser.parse_args()
    if cli.smoke:
        cli.steps, cli.repair_steps = 4, 2
    if min(cli.steps, cli.repair_steps) < 1:
        parser.error('Step counts must be positive')
    root = Path(cli.run) if Path(cli.run).is_absolute() or len(Path(cli.run).parts) > 1 else Path('out') / cli.run
    root.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(**vars(cli), topk=32, nograd_tokens=160, ent_max=2.)
    torch.manual_seed(args.seed)
    data = build_data(args.seed)
    e.write_json(root / 'revoke_scope_data.json', data)
    model, dev = g.build()
    tok = g.load_tokenizer()
    c.checkpoint_recurrences(model)
    params, _, _ = g.fsa.attach(model, [14, 15], 'fsa_free', 16, fmt='ternary', budgeted=True)
    for m in g.fsa.fsa_modules(model):
        m.enable_block_scale(16, .25)
        params.append(m.logb)
        m.support.zero_()
    g.fsa.set_enabled(True)
    modules, _ = c.patch_modules(model)
    frozen = {name: torch.zeros_like(m.mask) for name, m in modules.items()}
    blocks = {name: m.block for name, m in modules.items()}
    paths = {}
    for label in ('base', 'A', 'AB', 'naive', 'ownership', 'repair', 'ideal'):
        folder = root / 'revoke_scope' / label
        folder.mkdir(parents=True, exist_ok=True)
        paths[label] = folder / 'state.pt'
    save_committed(paths['base'], model, modules, frozen)
    windows = e.token_windows(json.loads((g.DATA / 'eval-wikitext.json').read_text()), 40)
    panel_path = root / 'genome' / 'panel.pt'
    panel_path.parent.mkdir(exist_ok=True)
    if not panel_path.exists():
        torch.save(e.wikitext_panel(model, windows, dev), panel_path)
    genome = torch.load(panel_path, map_location='cpu', weights_only=True)
    if len(genome['nll']) != 40 or tuple(genome['argmax'].shape) != (40, 511):
        raise ValueError('Genome cache must contain the locked 40x512 panel')
    anchors = json.loads((g.DATA / 'anchor-wikitext.json').read_text())[:150000]
    a_items, a_acq = cache_study(model, tok, dev, candidates(data, 'A', args.seed, args.smoke),
                               data['A']['episodes'], anchors, args, paths['A'].parent)
    lam = train(model, dev, modules, params, frozen, a_acq, anchors, args, args.steps, paths['A'].parent)
    save_committed(paths['A'], model, modules, frozen)
    del a_items, a_acq
    # Explicit reload: B's teacher is precisely the saved committed A individual.
    load_committed(paths['A'], model, modules, frozen)
    b_items, b_acq = cache_study(model, tok, dev, candidates(data, 'B', args.seed, args.smoke),
                               data['B']['episodes'], anchors, args, paths['AB'].parent)
    lam = train(model, dev, modules, params, frozen, b_acq, anchors, args, args.steps, paths['AB'].parent, lam)
    save_committed(paths['AB'], model, modules, frozen)
    a = torch.load(paths['A'], map_location='cpu', weights_only=True)['modules']
    b = torch.load(paths['AB'], map_location='cpu', weights_only=True)['modules']
    base = torch.load(paths['base'], map_location='cpu', weights_only=True)['modules']
    ledger = ownership_ledger(a, b)
    for label, owned in (('naive', False), ('ownership', True)):
        save_committed(paths[label], model, modules, frozen, revoke_state(a, b, base, blocks, owned))
    del a, b, base
    load_committed(paths['ownership'], model, modules, frozen)
    b_general = [dict(entry, record=c.TopKRecord.from_state_dict(entry['record'])) for entry in
                 torch.load(paths['AB'].parent / 'general.pt', map_location='cpu', weights_only=True)]
    train(model, dev, modules, params, frozen, b_acq, anchors, args, args.repair_steps,
          paths['repair'].parent, lam, b_general)
    save_committed(paths['repair'], model, modules, frozen)
    del b_acq, b_general
    load_committed(paths['base'], model, modules, frozen)
    # Same accepted B answers, not a new self-study draw; base supplies its own context records and gates.
    _, ideal_acq = cache_study(model, tok, dev, b_items, data['B']['episodes'], anchors, args,
                               paths['ideal'].parent, generate_answers=False)
    train(model, dev, modules, params, frozen, ideal_acq, anchors, args, args.steps, paths['ideal'].parent)
    save_committed(paths['ideal'], model, modules, frozen)
    del b_items, ideal_acq
    probes = make_probes(data, args.seed)
    results = {}
    for label, path in paths.items():
        load_committed(path, model, modules, frozen)
        results[label] = metrics(model, tok, dev, probes, windows, genome)
        print(json.dumps(dict(phase='evaluation', condition=label)), flush=True)
    e.write_json(root / 'revoke_scope.json', dict(seed=args.seed, steps=args.steps, repair_steps=args.repair_steps,
                 smoke=args.smoke, ledger=ledger, conditions=results,
                 scale_rule='Revocation: any B code write owns its block; otherwise restore A scale; empty blocks restore base scale.'))
    print('\n| Condition | R2 policies | R2 facts | R1 policy leak | R1 fact leak | A policies | A facts | B new facts | dNLL ± SE [95% CI] | Slots |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---|---:|')
    for label, result in results.items():
        recalls, dnll = result['recall'], result['panel']['dnll']
        cells = [recalls[key][metric] for key, metric in [('R2_policy', 'recall'), ('R2_fact', 'recall'),
                 ('R2_policy', 'leakage'), ('R2_fact', 'leakage'), ('A_policy', 'recall'),
                 ('A_fact', 'recall'), ('B_new_fact', 'recall')]]
        print(f"| {label} | " + ' | '.join(f'{x:.3f}' for x in cells) +
              f" | {dnll['mean']:.5f} ± {dnll['se']:.5f} [{dnll['ci95_lo']:.5f}, {dnll['ci95_hi']:.5f}] | {result['slots']['nnz']} |")
    print(json.dumps(dict(phase='ownership_ledger', **ledger['totals'])), flush=True)


if __name__ == '__main__':
    main()
