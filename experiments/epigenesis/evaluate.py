"""EP-1 evaluation. GPU imports are lazy so metric aggregation is CPU-testable."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import epi_mask
import epi_fold
from epi_common import paired_stats, participation_ratio, pair8_legal, parse_resolution


def gpu():
    import epi_gpu
    return epi_gpu


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def limited(items, subset):
    return items if subset is None else items[:subset]


def category_table(policies, category):
    """Amendment v2.1 context: every rule of a category, deterministic order. The single source episode used in v1
    contained only the correct policy and leaked the answer."""
    return '\n'.join(f"At Fennmark Parcel Desk we use {p['codename']}. {p['rule_text']} {p['rationale_text']}"
                     for p in sorted((p for p in policies if p['category'] == category), key=lambda p: p['id']))


def prefix(tok, item, ctx_chunks=None):
    g = gpu()
    # ctx_chunks: None (context-free), a dict episode_id -> text (v1 ICL: the source episode), or a callable
    # item -> text|None (v2.1 ICL: the category rule table for concept items)
    if ctx_chunks is None:
        text = None
    elif callable(ctx_chunks):
        text = ctx_chunks(item)
    else:
        text = ctx_chunks[item['source_episode']]
    context_ids = [] if text is None else g.enc(tok, text) + g.enc(tok, '\n\n')
    return [g.BOS] + context_ids + g.enc(tok, item['prompt'].rstrip())


def concept_acc(model, tok, dev, items, subset=None, ctx_chunks=None):
    hits = {'ticket': [], 'qa': []}
    for item in limited(items, subset):
        scores = gpu().score_options(model, tok, prefix(tok, item, ctx_chunks), item['options'], dev)
        hits[item['family']].append(max(range(len(scores)), key=scores.__getitem__) == item['answer_idx'])
    all_hits = hits['ticket'] + hits['qa']
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return dict(acc=mean(all_hits), acc_ticket=mean(hits['ticket']), acc_qa=mean(hits['qa']), n=len(all_hits))


def recall(model, tok, dev, items, subset=None, ctx_chunks=None):
    results = [gpu().exact_match(model, tok, prefix(tok, item, ctx_chunks), item['answer'], dev)
               for item in limited(items, subset)]
    return dict(em=sum(r[0] for r in results) / len(results) if results else 0.0,
                logprob=sum(r[1] for r in results) / len(results) if results else 0.0, n=len(results))


def token_windows(tokens, n, length=512):
    if len(tokens) < n * (length - 1):
        raise ValueError('Token pool is too short')
    return [[gpu().BOS] + tokens[i * (length - 1):(i + 1) * (length - 1)] for i in range(n)]


@torch.no_grad()
def wikitext_panel(model, windows, dev):
    nll, predictions = [], []
    g = gpu()
    for row in windows:
        x = torch.tensor([row], device=dev)
        lp = g.logits_of(model, x)[0, :-1, :g.VOCAB].float().log_softmax(-1)
        nll.append(float(-lp.gather(1, x[0, 1:, None]).mean()))
        predictions.append(lp.argmax(-1).cpu())
    return dict(nll=nll, argmax=torch.stack(predictions))


def panel_metrics(panel, genome):
    if panel['argmax'].shape != genome['argmax'].shape:
        raise ValueError('Panel positions must match')
    return dict(dnll=paired_stats(panel['nll'], genome['nll']),
                agreement=float((panel['argmax'] == genome['argmax']).float().mean()))


def heldout_batches(tok, texts, dev, micro_tokens=2048):
    """BOS-prefixed document windows, one per row; padding is never scored.

    The active harness supplies token IDs only, with no packed-document boundaries.
    Windows reset at BOS; both scoring and utility use this same bounded objective.
    """
    if micro_tokens < 2:
        raise ValueError('MDL micro-token budget must be at least 2')
    g = gpu()
    width = min(2048, micro_tokens)
    rows = []
    for i, text in enumerate(texts):
        tokens = g.enc(tok, text)
        for start in range(0, len(tokens), width - 1):
            rows.append((i, [g.BOS] + tokens[start:start + width - 1]))
    batch = []
    def pack(batch):
        x = torch.zeros(len(batch), max(len(row) for _, row in batch), dtype=torch.long, device=dev)
        for j, (_, row) in enumerate(batch):
            x[j, :len(row)] = torch.tensor(row, device=dev)
        return x, [(i, len(row)) for i, row in batch]
    for item in rows:
        if batch and max(len(item[1]), max(len(r) for _, r in batch)) * (len(batch) + 1) > micro_tokens:
            yield pack(batch)
            batch = []
        batch.append(item)
    if batch:
        yield pack(batch)


@torch.no_grad()
def heldout_nll(model, tok, dev, texts, micro_tokens=2048):
    """Summed document-token NLL; excludes BOS and right padding."""
    g = gpu()
    result = [0.0] * len(texts)
    for x, positions in heldout_batches(tok, texts, dev, micro_tokens):
        logits = g.logits_of(model, x)
        for row, (i, length) in enumerate(positions):
            result[i] += float(torch.nn.functional.cross_entropy(
                logits[row, :length - 1, :g.VOCAB].float(), x[row, 1:length], reduction='sum'))
    return result


def mdl_bits(genome_nll, patched_nll, panel_dnll, patch_bytes):
    """All code lengths use bits; general damage may be negative."""
    if len(genome_nll) != len(patched_nll):
        raise ValueError('Held-out NLL positions must match')
    saved = sum(a - b for a, b in zip(genome_nll, patched_nll)) / math.log(2)
    cost = {key: panel_dnll[key] * 1e6 / math.log(2)
            for key in ('mean', 'ci95_lo', 'ci95_hi')}
    patch = patch_bytes * 8
    return dict(bits_saved=saved, general_cost=cost, N_ref=1000000, patch_bits=patch,
                net=saved - cost['mean'] - patch)


def mdl_metrics(model, tok, dev, corpus, run_dir, session, panel, slots):
    from make_user import heldout_text
    cache_path = Path(run_dir) / 'genome/heldout_nll.json'
    # Arm processes can evaluate concurrently. Serialize the once-per-run genome cache.
    import fcntl
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        signature_path = cache_path.with_name('heldout_signatures.json')
        signatures = json.loads(signature_path.read_text()) if signature_path.exists() else {}
        rendered = {str(k): heldout_text(corpus, k, corpus['seed']) for k in range(1, session + 1)}
        expected = {k: hashlib.sha256(json.dumps(texts).encode()).hexdigest() for k, texts in rendered.items()}
        missing = [k for k in range(1, session + 1)
                   if str(k) not in cache or signatures.get(str(k)) != expected[str(k)]]
        if missing:
            gpu().fsa.set_enabled(False)
            try:
                for k in missing:
                    cache[str(k)] = heldout_nll(model, tok, dev, rendered[str(k)])
                    signatures[str(k)] = expected[str(k)]
            finally:
                gpu().fsa.set_enabled(True)
            write_json(cache_path, cache)
            write_json(signature_path, signatures)
    texts = [t for k in range(1, session + 1) for t in heldout_text(corpus, k, corpus['seed'])]
    base = [n for k in range(1, session + 1) for n in cache[str(k)]]
    return mdl_bits(base, heldout_nll(model, tok, dev, texts), panel['dnll'], slots['bytes'])


def arc_easy(model, tok, dev, path=None, subset=None):
    path = Path(path) if path else Path(__file__).with_name('arc_easy_200.json')
    data = json.loads(path.read_text())
    items = data['items'] if isinstance(data, dict) else data
    hits = []
    for item in limited(items, subset):
        q = item.get('question', item.get('q'))
        choices = item['choices']
        if isinstance(choices, dict):
            texts, labels = choices['text'], choices['label']
        else:
            texts = [c['text'] if isinstance(c, dict) else c for c in choices]
            labels = [c.get('label', str(i)) if isinstance(c, dict) else chr(65 + i) for i, c in enumerate(choices)]
            labels = item.get('labels', labels)
        idx = item.get('answer_idx')
        if idx is None:
            idx = labels.index(item.get('answerKey', item.get('answer')))
        scores = gpu().score_options(model, tok, [gpu().BOS] + gpu().enc(tok, f'Question: {q}\nAnswer:'), texts, dev)
        hits.append(max(range(len(scores)), key=scores.__getitem__) == idx)
    return dict(acc=sum(hits) / len(hits) if hits else 0.0, n=len(hits))


def style(model, tok, dev, prompts, corpus, subset=None):
    prompts = limited(prompts, subset)
    generated = gpu().generate(model, tok, [[gpu().BOS] + gpu().enc(tok, p) for p in prompts], dev,
                               max_new=40, stop_strs=('\n',), temperature=0)
    return dict(compliance=sum(style_valid(text, corpus) for text, _ in generated)
                / len(prompts) if prompts else 0.0, n=len(prompts))


def style_valid(text, corpus):
    parsed = parse_resolution(text.strip())
    return bool(parsed and parsed['action'] in corpus['actions'] and
                parsed['status'] == corpus['status_map'][parsed['action']])


def complete_sessions(sessions):
    return ([m['session'] for m in sessions] == list(range(1, 6)) and
            all(m['test']['concept'].get('n') == 100 and m['panel']['dnll'].get('n') == 40 for m in sessions))


def aggregate_slots(entries):
    """Entries carry layer/expert identity, row counts, reserve, bits and scale count."""
    experts, layers, rows = {}, {}, []
    bits, nnz = 0.0, 0
    for e in entries:
        n = sum(e['rows'])
        nnz += n
        key = (e['layer'], e['expert'])
        experts[key] = experts.get(key, 0) + n
        layers[str(e['layer'])] = layers.get(str(e['layer']), 0) + n
        rows.extend(e['rows'])
        if 'lora_params' in e:
            bits += e['lora_params'] * 32
        else:
            bits += (n * (math.log2(max(e['reserve'], 2)) + e['bits']) if e['budgeted']
                     else e['reserve'] * e['bits']) + 32 * e.get('scales', 0)
    return dict(nnz=nnz, bytes=bits / 8, participation_ratio_experts=participation_ratio(torch.tensor(list(experts.values()))),
                participation_ratio_rows=participation_ratio(torch.tensor(rows)), per_layer_nnz=layers)


@torch.no_grad()
def slot_stats(model):
    fsa = gpu().fsa
    entries = []
    reserve = sum(int(m.mask.sum()) for m in model.modules()
                  if isinstance(m, (fsa.FSAParam, epi_fold.FoldLoRAParam)))
    for li, (_, moe) in enumerate(fsa.moe_layers(model)):
        for ei, expert in enumerate(moe.experts):
            if expert is None:
                continue
            for proj in ('up_proj', 'down_proj'):
                lin = getattr(expert, proj)
                if not hasattr(lin, 'parametrizations'):
                    continue
                m = lin.parametrizations.weight[0]
                values = m.codes() if isinstance(m, epi_fold.FoldLoRAParam) else m.delta_w()
                entry = dict(layer=li, expert=ei, rows=(values != 0).sum(1).cpu().tolist())
                if isinstance(m, epi_fold.FoldLoRAParam):
                    entry.update(reserve=reserve, bits=math.log2(3), budgeted=True)
                elif isinstance(m, fsa.LoRAParam):
                    entry['lora_params'] = m.A.numel() + m.B.numel()
                else:
                    entry.update(reserve=reserve if m.score is not None else int(m.mask.sum()),
                                 bits={'ternary': math.log2(3), 'fp4': 4}[m.fmt], budgeted=m.score is not None,
                                 scales=0 if m.logb is None else int(torch.isfinite(m.logb).sum()))
                entries.append(entry)
    return aggregate_slots(entries)


@torch.no_grad()
def revoke_logits(model, tok, dev, corpus):
    # All prompt positions, one row at a time: no padding-dependent comparison.
    return [gpu().logits_of(model, torch.tensor([[gpu().BOS] + gpu().enc(tok, p)], device=dev)).cpu()
            for p in corpus['revoke_prompts']]


@torch.no_grad()
def revoke_test(model, tok, dev, corpus, genome_metrics, genome_logits_path,
                optimizer=None, replay_dir=None, ledger=None, replay=None, frozen=None):
    fsa = gpu().fsa
    fsa.set_enabled(True)
    legal_before = all(pair8_legal(lin.weight) for lin in model.modules()
                       if hasattr(lin, 'parametrizations') and hasattr(lin.parametrizations, 'weight'))
    provenance = epi_mask.metadata(model)
    for m in model.modules():
        if isinstance(m, fsa.FSAParam):
            m.M.zero_()
            m.support.zero_()
            if hasattr(m, 'bop_ema'):
                m.bop_ema.zero_()
            if m.logb is not None:
                m.logb.copy_(torch.log(m.alpha.float() * .25).expand_as(m.logb))
            if m.score is not None:
                m.score.zero_()
        elif isinstance(m, (fsa.LoRAParam, epi_fold.FoldLoRAParam)):
            m.A.zero_()
            m.B.zero_()
    for parameter in model.parameters():
        parameter.grad = None
    if optimizer is not None:
        optimizer.state.clear()
    if ledger is not None:
        ledger.clear()
    if replay is not None:
        replay.clear()
    if frozen is not None:
        for mask in frozen.values():
            mask.zero_()
    if replay_dir is not None:
        write_json(Path(replay_dir) / 'replay.json', [])
        torch.save({}, Path(replay_dir) / 'replay.pt')
        write_json(Path(replay_dir) / 'ledger.json', provenance)
    equal, legal = True, True
    for lin in model.modules():
        if hasattr(lin, 'parametrizations') and hasattr(lin.parametrizations, 'weight'):
            effective = lin.weight
            equal &= torch.equal(effective, lin.parametrizations.weight.original) and torch.equal(
                effective.to(torch.bfloat16).view(torch.int16),
                lin.parametrizations.weight.original.to(torch.bfloat16).view(torch.int16))
            legal &= pair8_legal(effective)
    original = torch.load(genome_logits_path, map_location='cpu', weights_only=True)
    current = revoke_logits(model, tok, dev, corpus)
    if len(original) != len(current):
        raise ValueError('Revoke prompt cache mismatch')
    diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(current, original))
    logits_equal = all(torch.equal(a, b) for a, b in zip(current, original))
    concept = concept_acc(model, tok, dev, corpus['probes']['test']['concept'], genome_metrics['test']['concept'].get('n'))
    lure = recall(model, tok, dev, corpus['probes']['test']['lure'], genome_metrics['test']['lure'].get('n'))
    cd = abs(concept['acc'] - genome_metrics['test']['concept']['acc'])
    ld = abs(lure['em'] - genome_metrics['test']['lure']['em'])
    result = dict(**provenance, pair8_legal_before=bool(legal_before), weights_equal=bool(equal), pair8_legal=bool(legal), logits_equal=logits_equal, logits_exact=logits_equal,
                  max_logit_diff=diff, concept_abs_diff=cd, lure_abs_diff=ld,
                  logits_pass=diff <= 1e-2, metrics_pass=cd <= 1e-8 and ld <= 1e-8)
    result['operational_cleared'] = all(p.grad is None for p in model.parameters()) and all(not m.A.any() and not m.B.any() for m in model.modules() if isinstance(m, (fsa.LoRAParam, epi_fold.FoldLoRAParam))) and (optimizer is None or not optimizer.state) and (ledger is None or not ledger) and (replay is None or not replay) and (frozen is None or all(not m.any() for m in frozen.values()))
    result['pass'] = bool(equal and legal and result['logits_pass'] and result['metrics_pass'] and result['operational_cleared'])
    return result


def run_all(model, tok, dev, corpus, run_dir, arm, session=None, ctx_chunks=None, subsets=None):
    started = time.monotonic()
    subsets = subsets or {}
    result = dict(**epi_mask.metadata(model), arm=arm, session=session)
    for split in ('dev', 'test'):
        probes = corpus['probes'][split]
        result[split] = dict(concept=concept_acc(model, tok, dev, probes['concept'], subsets.get('concept'), ctx_chunks),
                             verbatim=recall(model, tok, dev, probes['verbatim'], subsets.get('verbatim'), ctx_chunks),
                             lure=recall(model, tok, dev, probes['lure'], subsets.get('lure'), ctx_chunks))
    tokens = json.loads((Path(gpu().DATA) / 'eval-wikitext.json').read_text())
    panel = wikitext_panel(model, token_windows(tokens, subsets.get('panel', 40)), dev)
    cache = Path(run_dir) / 'genome' / 'panel.pt'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if arm == 'genome':
        torch.save(panel, cache)
        torch.save(revoke_logits(model, tok, dev, corpus), cache.parent / 'revoke_logits.pt')
        base = panel
    else:
        base = torch.load(cache, map_location='cpu', weights_only=True)
        # Smoke may use fewer windows from a full baseline cache.
        base = dict(nll=base['nll'][:len(panel['nll'])], argmax=base['argmax'][:len(panel['nll'])])
    result['panel'] = panel_metrics(panel, base)
    result['arc'] = arc_easy(model, tok, dev, subset=subsets.get('arc'))
    result['style'] = style(model, tok, dev, corpus['style_prompts'], corpus, subsets.get('style'))
    if arm not in ('genome', 'icl'):
        result['slots'] = slot_stats(model)
    if session is not None:
        result['mdl'] = mdl_metrics(model, tok, dev, corpus, run_dir, session, result['panel'],
                                    result.get('slots', {'bytes': 0}))
    print(json.dumps(dict(phase='evaluation', arm=arm, session=session, seconds=time.monotonic() - started,
                          peak_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
    return result


def retention(sessions):
    earlier = [m['test']['concept']['acc'] for m in sessions if m['session'] < 5]
    final = next((m['test']['concept']['acc'] for m in sessions if m['session'] == 5), None)
    if final is None or not earlier or max(earlier) == 0:
        return None
    return final / max(earlier)


def compare_to_tmid(sessions, tmid):
    """Exploratory decay comparison; missing final sessions/zero earlier recall yield no ratio."""
    profiles = {}
    for key, observed in (('arm', sessions), ('tmid', tmid)):
        final = next((m for m in observed if m['session'] == 5), None)
        facts = {str(m['session']): m['test'].get('verbatim', {}).get('em') for m in observed}
        earlier = [v for k, v in facts.items() if int(k) < 5 and v is not None]
        best = max(earlier) if earlier else 0
        profiles[key] = dict(verbatim_em=facts,
                             final_best_earlier_ratio=facts.get('5') / best if best and facts.get('5') is not None else None,
                             lure_em={str(m['session']): m['test']['lure']['em'] for m in observed},
                             final_panel_dnll=final['panel']['dnll'] if final else None,
                             final_slots=final['slots']['nnz'] if final else None,
                             concept={str(m['session']): m['test'].get('concept', {}).get('acc') for m in observed},
                             net_mdl={str(m['session']): m.get('mdl', {}).get('net') for m in observed})
    a, b = profiles['arm'], profiles['tmid']
    profiles['final_dnll_minus_tmid'] = (a['final_panel_dnll']['mean'] - b['final_panel_dnll']['mean']
                                        if a['final_panel_dnll'] and b['final_panel_dnll'] else None)
    profiles['final_slots_minus_tmid'] = (a['final_slots'] - b['final_slots']
                                         if a['final_slots'] is not None and b['final_slots'] is not None else None)
    return profiles


def summarize(run_dir):
    root = Path(run_dir)
    genome = json.loads((root / 'genome/metrics.json').read_text())
    icl = json.loads((root / 'icl/metrics.json').read_text())
    arms = {}
    sessions_by_arm = {}
    for arm in sorted(p.name for p in root.iterdir() if p.is_dir() and
                      (p.name.startswith('tmid') or p.name in ('fp4mid', 'loramid', 'lorafold', 'tbop'))):
        sessions_by_arm[arm] = sorted([json.loads(p.read_text()) for p in (root / arm).glob('s*/metrics.json')],
                                      key=lambda m: m['session'])
    lora = next((m for m in sessions_by_arm.get('loramid', []) if m['session'] == 5), None)
    identifiable = icl['dev']['concept']['acc'] >= genome['dev']['concept']['acc'] + .10 - 1e-12
    for arm, sessions in sessions_by_arm.items():
        final = next((m for m in sessions if m['session'] == 5), None)
        entry = dict(sessions=sessions, retention=retention(sessions), complete=complete_sessions(sessions))
        entry['comparison_vs_tmid'] = compare_to_tmid(sessions, sessions_by_arm.get('tmid', []))
        if arm == 'lorafold':
            entry.update(secondary_exploratory=True, served_format='ternary', original_thesis_decision=False)
        if arm == 'tbop':
            entry['post_hoc_owner_approved'] = True
            entry['original_thesis_decision'] = False
        if arm.startswith('tmid_util') or arm == 'tmid_mdl':
            entry['secondary_exploratory'] = True
        if entry['complete'] and arm in ('tmid', 'fp4mid', 'loramid', 'tbop'):
            c, lure, panel = final['test']['concept']['acc'], final['test']['lure']['em'], final['panel']
            lures = [m['test']['lure']['em'] for m in sessions]
            # Explicit operational definition of "ratchet": monotone rise with net increase.
            ratchet = len(lures) > 1 and all(b >= a for a, b in zip(lures, lures[1:])) and lures[-1] > lures[0]
            criteria = dict(concept_gain=c >= genome['test']['concept']['acc'] + .10 - 1e-12,
                            lure_constraint=all(x <= genome['test']['lure']['em'] + .05 + 1e-12 for x in lures),
                            dnll_bound=panel['dnll']['mean'] <= .01,
                            ci_bound=panel['dnll']['ci95_hi'] < .05,
                            retention=entry['retention'] is not None and entry['retention'] >= .8,
                            lure_no_ratchet=not ratchet, revoke=final.get('revoke', {}).get('pass', False))
            entry.update(criteria=criteria, thesis_pass=identifiable and all(criteria.values()),
                         distillation_gap=icl['test']['concept']['acc'] - c)
            if lora is not None and complete_sessions(sessions_by_arm.get('loramid', [])):
                lc, ld = lora['test']['concept']['acc'], lora['panel']['dnll']['mean']
                entry['match_lora'] = c >= lc - .03 - 1e-12 and panel['dnll']['mean'] <= ld + .005 and criteria['lure_constraint'] and criteria['revoke']
                entry['beat_lora'] = c >= lc + .05 - 1e-12 and panel['dnll']['mean'] <= ld
        arms[arm] = entry
    mdl_comparisons = {arm: compare_to_tmid(sessions_by_arm.get('tmid_mdl', []), sessions_by_arm.get(arm, []))
                       for arm in ('tmid', 'tmid_util01', 'tmid_util03')}
    summary = dict(mdl_comparisons=mdl_comparisons, genome=genome, icl=icl, identifiable=identifiable, arms=arms,
                   lure_ratchet_rule='monotone nondecreasing across observed sessions with a net increase')
    write_json(root / 'summary.json', summary)
    print('| arm | session | concept | lure | dNLL | agreement | nnz | retention | pass | net MDL bits |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---|---:|')
    for arm, m in (('genome', genome), ('icl', icl)):
        print(f"| {arm} | — | {m['test']['concept']['acc']:.3f} | {m['test']['lure']['em']:.3f} | {m['panel']['dnll']['mean']:.5f} | {m['panel']['agreement']:.3f} | — | — | — | — |")
    for arm, entry in arms.items():
        for m in entry['sessions']:
            print(f"| {arm} | {m['session']} | {m['test']['concept']['acc']:.3f} | {m['test']['lure']['em']:.3f} | {m['panel']['dnll']['mean']:.5f} | {m['panel']['agreement']:.3f} | {m['slots']['nnz']} | {entry['retention']} | {entry.get('thesis_pass', False)} | {m.get('mdl', {}).get('net', '—')} |")
    for arm, entry in arms.items():
        print(json.dumps(dict(phase='comparison_vs_tmid', name=arm, **entry['comparison_vs_tmid'])))
    for comparator, comparison in mdl_comparisons.items():
        print(json.dumps(dict(phase='tmid_mdl_comparison', comparator=comparator, **comparison)))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--arm', choices=('genome', 'icl'))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--summary', action='store_true')
    a = p.parse_args()
    if a.summary:
        summarize(a.run)
        return
    if a.arm is None:
        p.error('--arm is required unless --summary')
    corpus = json.loads((Path(a.run) / 'corpus.json').read_text())
    model, dev = gpu().build()
    tok = gpu().load_tokenizer()
    chunks = None
    if a.arm == 'icl':
        episodes = {e['id']: e['text'] for s in corpus['sessions'] for e in s['episodes']}
        if corpus.get('rules') == 'coarse':
            # v2.1: concept probes see the full category table; fact/lure probes keep their source episode
            chunks = lambda item: (category_table(corpus['policies'], item['case']['category']) if 'case' in item
                                   else episodes[item['source_episode']])
        else:
            chunks = episodes
    metrics = run_all(model, tok, dev, corpus, a.run, a.arm, ctx_chunks=chunks,
                      subsets=dict(concept=20, verbatim=10, lure=10, panel=4, arc=20, style=4) if a.smoke else None)
    write_json(Path(a.run) / a.arm / 'metrics.json', metrics)


if __name__ == '__main__':
    main()
