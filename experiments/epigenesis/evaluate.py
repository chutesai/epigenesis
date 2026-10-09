"""EP-1 evaluation. GPU imports are lazy so metric aggregation is CPU-testable."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
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


def prefix(tok, item, ctx_chunks=None):
    g = gpu()
    # ICL ceiling: the probe's source EPISODE text in the window (the same context unit the teacher sees)
    context_ids = [] if ctx_chunks is None else g.enc(tok, ctx_chunks[item['source_episode']]) + g.enc(tok, '\n\n')
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
    reserve = sum(int(m.mask.sum()) for m in fsa.fsa_modules(model))
    for li, (_, moe) in enumerate(fsa.moe_layers(model)):
        for ei, expert in enumerate(moe.experts):
            if expert is None:
                continue
            for proj in ('up_proj', 'down_proj'):
                lin = getattr(expert, proj)
                if not hasattr(lin, 'parametrizations'):
                    continue
                m = lin.parametrizations.weight[0]
                entry = dict(layer=li, expert=ei, rows=(m.delta_w() != 0).sum(1).cpu().tolist())
                if isinstance(m, fsa.LoRAParam):
                    entry['lora_params'] = m.A.numel() + m.B.numel()
                else:
                    entry.update(reserve=reserve if m.score is not None else int(m.mask.sum()),
                                 bits={'ternary': math.log2(3), 'fp4': 4}[m.fmt], budgeted=m.score is not None,
                                 scales=0 if m.logb is None else m.logb.numel())
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
    for m in model.modules():
        if isinstance(m, fsa.FSAParam):
            m.M.zero_()
            m.support.zero_()
            if m.logb is not None:
                m.logb.copy_(torch.log(m.alpha.float() * .25).expand_as(m.logb))
            if m.score is not None:
                m.score.zero_()
        elif isinstance(m, fsa.LoRAParam):
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
        write_json(Path(replay_dir) / 'ledger.json', {})
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
    result = dict(weights_equal=bool(equal), pair8_legal=bool(legal), logits_equal=logits_equal, logits_exact=logits_equal,
                  max_logit_diff=diff, concept_abs_diff=cd, lure_abs_diff=ld,
                  logits_pass=diff <= 1e-2, metrics_pass=cd <= 1e-8 and ld <= 1e-8)
    result['operational_cleared'] = all(p.grad is None for p in model.parameters()) and all(not m.A.any() and not m.B.any() for m in model.modules() if isinstance(m, fsa.LoRAParam)) and (optimizer is None or not optimizer.state) and (ledger is None or not ledger) and (replay is None or not replay) and (frozen is None or all(not m.any() for m in frozen.values()))
    result['pass'] = bool(equal and legal and result['logits_pass'] and result['metrics_pass'] and result['operational_cleared'])
    return result


def run_all(model, tok, dev, corpus, run_dir, arm, session=None, ctx_chunks=None, subsets=None):
    started = time.monotonic()
    subsets = subsets or {}
    result = dict(arm=arm, session=session)
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
    print(json.dumps(dict(phase='evaluation', arm=arm, session=session, seconds=time.monotonic() - started,
                          peak_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
    return result


def retention(sessions):
    earlier = [m['test']['concept']['acc'] for m in sessions if m['session'] < 5]
    final = next((m['test']['concept']['acc'] for m in sessions if m['session'] == 5), None)
    if final is None or not earlier or max(earlier) == 0:
        return None
    return final / max(earlier)


def summarize(run_dir):
    root = Path(run_dir)
    genome = json.loads((root / 'genome/metrics.json').read_text())
    icl = json.loads((root / 'icl/metrics.json').read_text())
    arms = {}
    sessions_by_arm = {}
    for arm in ('tmid', 'fp4mid', 'loramid'):
        sessions_by_arm[arm] = sorted([json.loads(p.read_text()) for p in (root / arm).glob('s*/metrics.json')],
                                      key=lambda m: m['session'])
    lora = next((m for m in sessions_by_arm['loramid'] if m['session'] == 5), None)
    identifiable = icl['dev']['concept']['acc'] >= genome['dev']['concept']['acc'] + .10 - 1e-12
    for arm, sessions in sessions_by_arm.items():
        final = next((m for m in sessions if m['session'] == 5), None)
        entry = dict(sessions=sessions, retention=retention(sessions), complete=complete_sessions(sessions))
        if entry['complete']:
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
            if lora is not None and complete_sessions(sessions_by_arm['loramid']):
                lc, ld = lora['test']['concept']['acc'], lora['panel']['dnll']['mean']
                entry['match_lora'] = c >= lc - .03 - 1e-12 and panel['dnll']['mean'] <= ld + .005 and criteria['lure_constraint'] and criteria['revoke']
                entry['beat_lora'] = c >= lc + .05 - 1e-12 and panel['dnll']['mean'] <= ld
        arms[arm] = entry
    summary = dict(genome=genome, icl=icl, identifiable=identifiable, arms=arms,
                   lure_ratchet_rule='monotone nondecreasing across observed sessions with a net increase')
    write_json(root / 'summary.json', summary)
    print('| arm | session | concept | lure | dNLL | agreement | nnz | retention | pass |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
    for arm, m in (('genome', genome), ('icl', icl)):
        print(f"| {arm} | — | {m['test']['concept']['acc']:.3f} | {m['test']['lure']['em']:.3f} | {m['panel']['dnll']['mean']:.5f} | {m['panel']['agreement']:.3f} | — | — | — |")
    for arm, entry in arms.items():
        for m in entry['sessions']:
            print(f"| {arm} | {m['session']} | {m['test']['concept']['acc']:.3f} | {m['test']['lure']['em']:.3f} | {m['panel']['dnll']['mean']:.5f} | {m['panel']['agreement']:.3f} | {m['slots']['nnz']} | {entry['retention']} | {entry.get('thesis_pass', False)} |")
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
    chunks = ({e['id']: e['text'] for s in corpus['sessions'] for e in s['episodes']}
              if a.arm == 'icl' else None)
    metrics = run_all(model, tok, dev, corpus, a.run, a.arm, ctx_chunks=chunks,
                      subsets=dict(concept=20, verbatim=10, lure=10, panel=4, arc=20, style=4) if a.smoke else None)
    write_json(Path(a.run) / a.arm / 'metrics.json', metrics)


if __name__ == '__main__':
    main()
