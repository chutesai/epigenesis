"""One arm / one session of EP-1 consolidation; pure helpers import without GPU code."""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import time

import torch
from epi_common import TopKRecord, ONEOFF_RE, gate_weights, mix_weights, pack_rows, topk_kl
import evaluate


def gpu():
    import epi_gpu
    return epi_gpu


def labile_mask(support, frozen, allowed=True):
    return support & ~frozen & allowed


def superset_from_counts(counts, mass=.9):
    if not 0 < mass <= 1:
        raise ValueError('Mass must be in (0,1]')
    values = [float(x) for x in counts]
    if any(x < 0 or not math.isfinite(x) for x in values):
        raise ValueError('Counts must be finite and nonnegative')
    total = sum(values)
    if total == 0:
        return []
    selected, covered = [], 0.0
    for i in sorted(range(len(values)), key=lambda i: (-values[i], i)):
        selected.append(i)
        covered += values[i]
        if covered >= mass * total:
            break
    return sorted(selected)


def select_checkpoint(candidates, genome_lure, rollback_dnll=.005):
    eligible = [c for c in candidates if all(math.isfinite(c[k]) for k in ('select_dnll', 'dev_concept', 'dev_lure')) and
                c['dev_lure'] <= genome_lure + .05 + 1e-12 and c['select_dnll'] <= rollback_dnll]
    return max(eligible, key=lambda c: (c['dev_concept'], -c['nnz'])) if eligible else None


def promotion_mask(support, frozen, chosen_concept, start_concept, nonzero, prev_support):
    surviving = support & prev_support.to(support.device) & nonzero & ~frozen
    return frozen | surviving if chosen_concept > start_concept else frozen.clone()


def validate_acquisition(items, null_thr, acquisition=None):
    if not items:
        raise ValueError('Training requires at least one accepted self-study item')
    if null_thr is None or not math.isfinite(null_thr):
        raise ValueError('Training requires a finite null_thr')
    if acquisition is not None and not acquisition:
        raise ValueError('Training requires at least one acquisition item with gated tokens')


def session_ledger(start, current, rolled_back, lam):
    ledger = copy.deepcopy(start if rolled_back else current)
    if lam is not None:
        ledger['netcost_lambda'] = lam
    return ledger


def session_replay(previous, previous_records, items, teacher, schedule, session, rolled_back):
    if rolled_back:
        return previous, previous_records
    return update_replay(previous, items, dict(previous_records, **teacher), schedule, session)


def effective_weights(model):
    return {name: lin.weight.detach().to(torch.bfloat16).cpu().view(torch.int16).clone()
            for name, lin in model.named_modules()
            if hasattr(lin, 'parametrizations') and hasattr(lin.parametrizations, 'weight')}


def state_roundtrip(path, model, modules):
    before = effective_weights(model)
    scratch_frozen = {}
    load_state(path, modules, scratch_frozen)
    after = effective_weights(model)
    assert before.keys() == after.keys() and all(torch.equal(before[n], after[n]) for n in before), 'Effective bf16 weights changed on reload'
    print(json.dumps(dict(phase='state_roundtrip', equal=True)), flush=True)


def update_replay(previous, items, records, claim_schedule, session, cap=256):
    """Newest sessions first, round-robin policy buckets within a session; no modelq/one-offs."""
    entries = {e['id']: copy.deepcopy(e) for e in previous
               if e['kind'] in ('apply', 'style', 'factqa') and not ONEOFF_RE.search(e['answer'])}
    for item in items:
        if not item['accepted'] or ONEOFF_RE.search(item['answer']):
            continue
        if item['kind'] not in ('apply', 'style', 'factqa'):
            continue
        if item['kind'] == 'factqa' and sum(k <= session for k in claim_schedule.get(item['claim_hash'], [])) < 2:
            continue
        entry = {k: copy.deepcopy(v) for k, v in item.items() if k not in ('record', 'weights')}
        entry['session'] = session
        entries[item['id']] = entry
    selected = []
    for s in sorted({e['session'] for e in entries.values()}, reverse=True):
        buckets = defaultdict(list)
        for entry in entries.values():
            if entry['session'] == s:
                buckets[entry.get('policy_id') or '__nonpolicy__'].append(entry)
        # Seeded shuffle removes lexicographic policy bias when a partial round hits the cap.
        rng = random.Random(s)
        keys = sorted(buckets)
        rng.shuffle(keys)
        for values in buckets.values():
            values.sort(key=lambda e: e['id'])
            rng.shuffle(values)
        while any(buckets.values()) and len(selected) < cap:
            for key in keys:
                if buckets[key] and len(selected) < cap:
                    selected.append(buckets[key].pop())
        if len(selected) >= cap:
            break
    return selected, {e['id']: records[e['id']] for e in selected}


def sample_episodes(items, n, rng):
    groups = defaultdict(list)
    for item in items:
        groups[item['source_episode']].append(item)
    if not groups:
        return []
    keys = sorted(groups)
    return [rng.choice(groups[rng.choice(keys)]) for _ in range(n)]


def scale_of(m):
    return m.logb.detach().exp().repeat_interleave(m.block, dim=1) if m.logb is not None else m.alpha


@torch.no_grad()
def committed_values(m):
    raw = m.delta().detach() / scale_of(m)
    if m.fmt == 'ternary':
        return raw.sign().cpu()
    if m.fmt == 'fp4':
        grid = raw.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6]) * m.fp4_step
        return (raw.sign() * grid[(raw.abs().unsqueeze(-1) - grid).abs().argmin(-1)]).cpu()
    raise ValueError(f'Unsupported format {m.fmt}')


@torch.no_grad()
def snapshot(modules, frozen, committed=False):
    state = {}
    for name, m in modules.items():
        if hasattr(m, 'M'):
            state[name] = dict(committed=committed_values(m) if committed else None,
                               support=m.support.cpu().clone(), frozen=frozen[name].cpu().clone(),
                               logb=None if m.logb is None else m.logb.detach().cpu().clone())
            if not committed:
                state[name]['M'] = m.M.detach().cpu().clone()
        else:
            state[name] = dict(A=m.A.detach().cpu().clone(), B=m.B.detach().cpu().clone())
    return state


@torch.no_grad()
def restore(modules, state, frozen):
    if set(modules) != set(state):
        raise ValueError('Patch module identities differ from saved state')
    for name, m in modules.items():
        saved = state[name]
        if hasattr(m, 'M'):
            m.support.copy_(saved['support'])
            frozen[name] = saved['frozen'].to(m.M.device).clone()
            if m.logb is not None:
                m.logb.copy_(saved['logb'])
            value = saved['committed'] if saved['committed'] is not None else saved['M']
            m.M.copy_(value)
            if saved['committed'] is not None:
                expected = value.to(m.M.device) * scale_of(m)
                if not torch.equal(m.delta(), expected):
                    raise AssertionError(f'Rehydration changed committed patch: {name}')
        else:
            m.A.copy_(saved['A'])
            m.B.copy_(saved['B'])


def save_state(path, modules, frozen, ledger, replay):
    torch.save(dict(version=1, modules=snapshot(modules, frozen, committed=True),
                    ledger=copy.deepcopy(ledger), replay=copy.deepcopy(replay)), path)


def load_state(path, modules, frozen):
    state = torch.load(path, map_location='cpu', weights_only=True)
    restore(modules, state['modules'], frozen)
    return state['ledger'], state['replay']


def micro_batches(entries, micro_tokens):
    """Pack by padded tokens; reject indivisible rows exceeding the budget."""
    if micro_tokens < 1:
        raise ValueError('micro_tokens must be positive')
    if any(len(e['row']) > micro_tokens for e in entries):
        raise ValueError('A row exceeds --micro-tokens; increase the budget to preserve its full prefix')
    batch, width = [], 0
    for entry in sorted(entries, key=lambda e: -len(e['row'])):
        nwidth = max(width, len(entry['row']))
        if batch and nwidth * (len(batch) + 1) > micro_tokens:
            yield batch
            batch, width = [], 0
        batch.append(entry)
        width = max(width, len(entry['row']))
    if batch:
        yield batch


def loss_assembly(student, terms, dev, vocab, mix='balanced', replay_empty=False, micro_tokens=1024, backward=False):
    """Mean per term, scaled by each microbatch's loss-bearing token share. Student returns logits."""
    totals, logs = {}, {}
    for key in ('acq', 'gen', 'rep'):
        totals[key] = sum(int((e['weights'] > 0).sum()) if e.get('weights') is not None else e['record'].ids.shape[0]
                          for e in terms.get(key, []))
        logs[key] = 0.0
    coefficients = dict(zip(('acq', 'gen', 'rep'), mix_weights(mix, replay_empty)))
    assembled = None
    for key in ('acq', 'gen', 'rep'):
        if totals[key] == 0:
            continue
        for batch in micro_batches(terms.get(key, []), micro_tokens):
            x, _ = pack_rows([e['row'] for e in batch], [e['start'] for e in batch], dev)
            logits = student(x)
            contribution = None
            for i, entry in enumerate(batch):
                lp = logits[i, entry['start'] - 1:len(entry['row']) - 1, :vocab].float().log_softmax(-1)
                loss, nt = topk_kl(entry['record'], lp, entry.get('weights'))
                part = loss * (nt / totals[key])
                logs[key] += float(part.detach())
                contribution = part if contribution is None else contribution + part
            contribution = contribution * coefficients[key]
            if backward:
                if contribution.requires_grad:
                    contribution.backward()
                del logits, lp, loss, part, contribution
            else:
                assembled = contribution if assembled is None else assembled + contribution
    logs['tokens'] = totals
    logs['loss'] = sum(coefficients[k] * logs[k] for k in coefficients)
    return assembled, logs


def patch_modules(model):
    fsa = gpu().fsa
    modules, identity = {}, {}
    for li in (14, 15):
        _, moe = fsa.moe_layers(model)[li]
        for ei, expert in enumerate(moe.experts):
            if expert is None:
                continue
            for proj in ('up_proj', 'down_proj'):
                name = f'{li}.{ei}.{proj}'
                modules[name] = getattr(expert, proj).parametrizations.weight[0]
                identity[name] = (li, ei)
    return modules, identity


def item_row(item, chunk=None):
    prefix = [gpu().BOS] + (chunk or []) + item['prompt_ids']
    return prefix + item['answer_ids'], len(prefix)


def cache_items(model, dev, items, chunks, args, null_thr):
    """Only one item's two full-vocab tensors are resident; all long-lived records are CPU tensors."""
    g = gpu()
    teacher, start, acquisition = {}, {}, []
    dropped = 0
    for item in items:
        tr, ts = item_row(item, chunks[item['chunk_id']])
        sr, ss = item_row(item)
        if not item['answer_ids']:
            raise ValueError(f"Empty accepted answer: {item['id']}")
        tlp = g.logprobs_at(model, [tr], [ts], dev, args.nograd_tokens)[0]
        rec = TopKRecord.from_logits(tlp, args.topk).to('cpu')
        teacher[item['id']] = rec
        slp = g.logprobs_at(model, [sr], [ss], dev, args.nograd_tokens)[0]
        kl = (tlp.exp() * (tlp - slp)).sum(-1)
        weights = gate_weights(kl, rec.entropy.to(dev), null_thr, args.ent_max).cpu()
        start[item['id']] = TopKRecord.from_logits(slp, args.topk).to('cpu')
        del tlp, slp, kl
        if int(weights.sum()) == 0:
            dropped += 1
        else:
            acquisition.append(dict(item, row=sr, start=ss, record=rec, weights=weights))
    return teacher, start, acquisition, dropped


@torch.no_grad()
def router_superset(model, dev, items, args):
    fsa = gpu().fsa
    counts, hooks = {}, []
    for li in (14, 15):
        _, moe = fsa.moe_layers(model)[li]
        counts[li] = [0] * len(moe.experts)
        for ei, expert in enumerate(moe.experts):
            if expert is not None:
                def count(mod, inputs, out, li=li, ei=ei):
                    counts[li][ei] += inputs[0].numel() // inputs[0].shape[-1]
                hooks.append(expert.register_forward_hook(count))
    try:
        for item in items:
            row, _ = item_row(item)
            gpu().logits_of(model, torch.tensor([row], device=dev))
    finally:
        for hook in hooks:
            hook.remove()
    return {str(li): superset_from_counts(c) for li, c in counts.items()}, counts


@torch.no_grad()
def protect_gradients(modules, frozen, allowed):
    for name, m in modules.items():
        if not hasattr(m, 'M'):
            continue
        if m.M.grad is not None:
            m.M.grad.masked_fill_(~m.mask | frozen[name], 0)
            if not allowed[name]:
                m.M.grad.zero_()
        if m.logb is not None and m.logb.grad is not None:
            blocked = frozen[name].reshape(m.M.shape[0], -1, m.block).any(-1)
            m.logb.grad.masked_fill_(blocked, 0)
            if not allowed[name]:
                m.logb.grad.zero_()


@torch.no_grad()
def manual_decay(modules, frozen, allowed, lr, wd):
    for name, m in modules.items():
        if hasattr(m, 'M'):
            mask = labile_mask(m.support, frozen[name], allowed[name])
            m.M[mask] *= 1 - lr * wd


@torch.no_grad()
def calibrate_lambda(mods, den, budget):
    # Global ranking lives on CPU; compute cost only for selected coordinates.
    gains = []
    for m in mods:
        gain = m.score.detach().float().abs()
        gain.mul_(m.mask)
        gains.append(gain.flatten().cpu())
        del gain
    gain = torch.cat(gains)
    del gains
    n = min(budget, int((gain > 0).sum()))
    if n == 0:
        return None
    top = gain.topk(n).indices
    selected_gain = gain[top]
    del gain
    costs, offset = [], 0
    for m in mods:
        local = top[(top >= offset) & (top < offset + m.M.numel())] - offset
        offset += m.M.numel()
        if not local.numel():
            continue
        row, col = (local // m.M.shape[1]).to(m.M.device), (local % m.M.shape[1]).to(m.M.device)
        scale = m.alpha[row, 0] if m.logb is None else m.logb.detach()[row, col // m.block].exp()
        costs.append((scale.square() * m.col_energy[col] / den).cpu())
    cost = torch.cat(costs)
    value = float(selected_gain.median() / cost.median().clamp_min(1e-30))
    return value if math.isfinite(value) and value > 0 else None


@torch.no_grad()
def netcost_score(m, lam, den, row_batch=64):
    """Reference netcost formula with one dense output and bounded row temporaries."""
    score = torch.empty_like(m.M, dtype=torch.float32)
    for start in range(0, m.M.shape[0], row_batch):
        stop = start + row_batch
        master = m.M.detach()[start:stop]
        grad = m.score[start:stop].float()
        direction = torch.where(master != 0, master.sign(), -grad.sign())
        mag = torch.ones_like(master) if m.fmt == 'ternary' else master.abs().clamp(min=.5 * m.fp4_step)
        scale = m.alpha[start:stop].float() if m.logb is None else m.logb.detach()[start:stop].exp().repeat_interleave(m.block, dim=1)
        score[start:stop] = (-grad * direction * mag - lam * scale.square() * m.col_energy[None, :] / den * mag.square()) * m.mask[start:stop]
    return score


@torch.no_grad()
def refresh(modules, frozen, allowed, budget, lam, den, opt):
    """Reference refresh keeps global top-K; +inf reserves frozen and out-of-superset prior slots."""
    fsa = gpu().fsa
    mods = list(modules.values())
    scores = [netcost_score(m, lam, den) for m in mods]
    before = [m.support.clone() for m in mods]
    protected_count = 0
    for (name, m), score in zip(modules.items(), scores):
        protected = frozen[name] & m.support
        if not allowed[name]:
            score.zero_()
            protected = m.support.clone()
        score[protected] = float('inf')
        protected_count += int(protected.sum())
    if protected_count > budget:
        raise ValueError('Committed protected slots exceed budget')
    # Equivalent to labile ceiling = budget - protected count, retaining frozen support in top-K.
    fsa.refresh_support(mods, budget, scores, opt)
    births = sum(int((m.support & ~old).sum()) for m, old in zip(mods, before))
    evictions = sum(int((old & ~m.support).sum()) for m, old in zip(mods, before))
    return births, evictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--arm', choices=('tmid', 'fp4mid', 'loramid'), required=True)
    p.add_argument('--session', type=int, choices=range(1, 6), required=True)
    p.add_argument('--mix', choices=('balanced', 'acquire', 'retain'), default='balanced')
    # measured on the dev box (DESIGN.md section 6): resident state after step 1 (model + masters + Adam + grads +
    # netcost score/support) is ~23.4 GiB and a 256-token micro-batch adds ~8 GiB of naive-EDA activations (29.4
    # GiB at step 1, OOM at step 2 in the first smoke), so micro-batches are capped at 160 tokens / anchors at 96
    for name, default in [('steps', 48), ('items-per-step', 8), ('anchor-per-step', 2), ('replay-per-step', 4),
                          ('micro-tokens', 160), ('eval-every', 16), ('budget', 2000000), ('topk', 32),
                          ('anchor-windows', 64), ('anchor-len', 96), ('seed', 0), ('nograd-tokens', 4096)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--lr-mult', type=float, default=1.0)
    p.add_argument('--ent-max', type=float, default=2.0)
    p.add_argument('--rollback-dnll', type=float, default=.005)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.steps, args.items_per_step, args.eval_every = 10, 8, 5
        if args.arm != 'loramid':
            args.lr_mult = 5
    if not math.isfinite(args.lr_mult) or args.lr_mult <= 0:
        p.error('--lr-mult must be finite and positive')
    if min(args.steps, args.items_per_step, args.anchor_per_step, args.micro_tokens, args.eval_every,
           args.anchor_windows) < 1 or args.anchor_len < 2 or args.replay_per_step < 0 or args.budget < 0:
        p.error('Invalid nonpositive training configuration')
    if args.anchor_len > args.micro_tokens:
        p.error('--micro-tokens must be at least --anchor-len')
    root = Path(args.run)
    out = root / args.arm / f's{args.session}'
    out.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((root / 'corpus.json').read_text())
    study = json.loads((root / 'selfstudy' / f's{args.session}.json').read_text())
    if study['session'] != args.session:
        raise ValueError('Self-study session mismatch')
    items = [i for i in study['items'] if i['accepted']]
    if len({i['id'] for i in items}) != len(items):
        raise ValueError('Accepted self-study item IDs must be unique')
    if args.smoke:
        items = items[:8]
    validate_acquisition(items, study['null_thr'])
    g, fsa = gpu(), gpu().fsa
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + args.session)
    model, dev = g.build()
    tok = g.load_tokenizer()
    params, _, _ = fsa.attach(model, [14, 15], 'lora' if args.arm == 'loramid' else 'fsa_free', 16,
                             fmt='ternary' if args.arm == 'tmid' else 'fp4', budgeted=args.arm == 'tmid')
    mods = fsa.fsa_modules(model)
    for m in mods:
        m.fp4_step = .25
        if args.arm == 'tmid':
            m.enable_block_scale(16, .25)
            params.append(m.logb)
            m.support.zero_()
    modules, identity = patch_modules(model)
    frozen = {name: torch.zeros_like(m.mask) for name, m in modules.items() if hasattr(m, 'M')}
    ledger, replay, replay_records = {}, [], {}
    if args.session > 1:
        previous = root / args.arm / f's{args.session - 1}'
        ledger, replay = load_state(previous / 'state.pt', modules, frozen)
        replay_records = {k: TopKRecord.from_state_dict(v) for k, v in
                          torch.load(previous / 'replay.pt', map_location='cpu', weights_only=True).items()}
    fsa.set_enabled(True)
    start_state = snapshot(modules, frozen)
    start_ledger = copy.deepcopy(ledger)
    subsets = dict(concept=20, lure=10, verbatim=10, panel=4, arc=20, style=4) if args.smoke else None
    # Arm processes only consume the genome caches produced by the runner.
    baseline_path = root / 'genome' / 'metrics.json'
    if not all(path.exists() for path in (baseline_path, root / 'genome/panel.pt', root / 'genome/revoke_logits.pt')):
        raise ValueError('Genome baseline requires metrics.json, panel.pt and revoke_logits.pt; run genome evaluation first')
    genome = json.loads(baseline_path.read_text())
    if not args.smoke and (genome['test']['concept'].get('n') != 100 or genome['panel']['dnll'].get('n') != 40):
        raise ValueError('Full training requires a full genome baseline')
    dev_count, lure_count = (20, 10) if args.smoke else (100, 50)
    def dev_metrics():
        return (evaluate.concept_acc(model, tok, dev, corpus['probes']['dev']['concept'], dev_count)['acc'],
                evaluate.recall(model, tok, dev, corpus['probes']['dev']['lure'], lure_count)['em'])
    start_concept, start_lure = dev_metrics()
    # Match lure subset on both sides of the hard discard in smoke runs.
    if args.smoke:
        fsa.set_enabled(False)
        try:
            genome_lure = evaluate.recall(model, tok, dev, corpus['probes']['dev']['lure'], lure_count)['em']
        finally:
            fsa.set_enabled(True)
    else:
        genome_lure = genome['dev']['lure']['em']
    anchor_tokens = json.loads((Path(g.DATA) / 'anchor-wikitext.json').read_text())
    pool = anchor_tokens[:150000]
    if len(anchor_tokens) < 150000 + 16 * 511 or len(pool) < args.anchor_len - 1:
        raise ValueError('Anchor and selection pools must be disjoint')
    anchors = [[g.BOS] + pool[(offset := rng.randrange(len(pool) - args.anchor_len + 2)):offset + args.anchor_len - 1]
               for _ in range(args.anchor_windows)]
    calibration_anchors = list(anchors[:32])
    if args.arm == 'tmid':
        while len(calibration_anchors) < 32:
            offset = rng.randrange(len(pool) - args.anchor_len + 2)
            calibration_anchors.append([g.BOS] + pool[offset:offset + args.anchor_len - 1])
    select_rows = evaluate.token_windows(anchor_tokens[-16 * 511:], 4 if args.smoke else 16)
    select_tensor = torch.tensor(select_rows, device=dev)
    start_nll = fsa.nll_windows(model, select_tensor, dev)
    phase = time.monotonic()
    teacher, start_records, acquisition, dropped = cache_items(model, dev, items, study['chunk_tokens'], args, study['null_thr'])
    validate_acquisition(items, study['null_thr'], acquisition)
    anchor_records = []
    for row in anchors:
        anchor_records.extend(g.topk_records(model, [row], [1], dev, args.topk, args.nograd_tokens))
    anchor_entries = [dict(row=row, start=1, record=rec) for row, rec in zip(anchors, anchor_records)]
    superset, counts = (None, None) if args.arm == 'loramid' else router_superset(model, dev, items, args)
    allowed = {name: superset is None or ei in superset[str(li)] for name, (li, ei) in identity.items()}
    ledger.setdefault('superset', {})[str(args.session)] = superset
    print(json.dumps(dict(phase='cache', seconds=time.monotonic() - phase, dropped_zero_gate=dropped,
                          acquisition_items=len(acquisition), peak_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
    torch.cuda.empty_cache()
    scales = [m.logb for m in mods if m.logb is not None]
    scale_ids = {id(t) for t in scales}
    lr = .001 if args.arm == 'loramid' else .02 * args.lr_mult
    groups = [dict(params=[t for t in params if id(t) not in scale_ids], lr=lr, weight_decay=0)]
    if scales:
        groups.append(dict(params=scales, lr=.002 * args.lr_mult, weight_decay=0))
    opt = torch.optim.AdamW(groups)
    local = fsa.LocalOutputPenalty(model) if args.arm != 'tmid' else None
    local_weight = 100 if args.arm == 'loramid' else 10
    den = fsa.collect_col_energy(model, mods, torch.tensor(calibration_anchors, device=dev)) if args.arm == 'tmid' else None
    if args.arm == 'tmid' and (not math.isfinite(den) or den <= 0):
        raise ValueError('Invalid anchor energy denominator')
    config_path = root / args.arm / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    lam = config.get('netcost_lambda', ledger.get('netcost_lambda'))
    config.update(netcost_lambda=lam, lr_mult=args.lr_mult, mix=args.mix, steps=args.steps)
    evaluate.write_json(config_path, config)
    if args.arm == 'tmid' and args.session > 1 and lam is None:
        raise ValueError('Session 1 did not calibrate netcost lambda; cannot recalibrate in later sessions')
    candidates, births_total, evictions_total, refreshes = [], 0, 0, 0
    prev_support = {name: torch.zeros_like(m.support, device='cpu') for name, m in modules.items() if hasattr(m, 'M')}
    try:
        with (out / 'log').open('w') as log:
            for step in range(1, args.steps + 1):
                tic = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                opt.zero_grad(set_to_none=True)
                ab = [rng.choice(anchor_entries) for _ in range(args.anchor_per_step)]
                rb = [rng.choice(replay) for _ in range(args.replay_per_step)] if replay else []
                rep = []
                for item in rb:
                    row, start = item_row(item)
                    rep.append(dict(row=row, start=start, record=replay_records[item['id']]))
                terms = dict(acq=sample_episodes(acquisition, args.items_per_step, rng), gen=ab, rep=rep)
                _, losses = loss_assembly(lambda x: g.logits_of(model, x), terms, dev, g.VOCAB,
                                          args.mix, not replay, args.micro_tokens, backward=True)
                penalty_value = 0.0
                if local is not None:
                    local_batches = list(micro_batches(ab, args.micro_tokens))
                    denominator = None
                    if len(local_batches) > 1:
                        # A ratio of total energies must be normalized across the whole anchor batch,
                        # rather than averaging ratios from micros with different routed exposure.
                        denominators = []
                        with torch.no_grad():
                            for batch in local_batches:
                                x, _ = pack_rows([e['row'] for e in batch], [1] * len(batch), dev)
                                local.start()
                                try:
                                    g.logits_of(model, x)
                                finally:
                                    local.stop()
                                denominators.extend(local.den)
                        if denominators:
                            denominator = torch.stack(denominators).sum()
                    for batch in local_batches:
                        x, _ = pack_rows([e['row'] for e in batch], [1] * len(batch), dev)
                        local.start()
                        try:
                            g.logits_of(model, x)
                        finally:
                            penalty = local.stop()
                        if penalty is not None:
                            share = 1.0 if denominator is None else (torch.stack(local.den).sum() + 1e-12) / (denominator + 1e-12)
                            lp = penalty * local_weight * share
                            penalty_value += float(lp.detach())
                            if lp.requires_grad:
                                lp.backward()
                protect_gradients(modules, frozen, allowed)
                for name, m in modules.items():
                    if hasattr(m, 'score') and m.score is not None:
                        m.score.mul_(.9)
                        if m.M.grad is not None:
                            m.score.add_(m.M.grad, alpha=.1)
                opt.step()
                manual_decay(modules, frozen, allowed, lr, .3 if args.arm == 'tmid' else .1)
                nb, ne = 0, 0
                if args.arm == 'tmid' and step % 8 == 0 and step <= 48:
                    prev_support = {name: m.support.cpu().clone() for name, m in modules.items()}
                    if lam is None:
                        lam = calibrate_lambda(mods, den, args.budget)
                        if lam is not None:
                            ledger['netcost_lambda'] = lam
                            config['netcost_lambda'] = lam
                            evaluate.write_json(config_path, config)
                    if lam is not None:
                        nb, ne = refresh(modules, frozen, allowed, args.budget, lam, den, opt)
                        births_total += nb
                        evictions_total += ne
                        refreshes += 1
                        torch.cuda.empty_cache()
                        print(json.dumps(dict(phase='refresh', refresh_peak_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
                nnz, energy = fsa.patch_stats(mods) if mods else (evaluate.slot_stats(model)['nnz'], 0)
                row = dict(step=step, **losses, local_penalty=penalty_value, kl_loss=losses['loss'], births=nb, evictions=ne,
                           patch_nnz=nnz, seconds=time.monotonic() - tic,
                           peak_gib=torch.cuda.max_memory_allocated() / 2**30)
                row['loss'] += penalty_value
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    concept, lure = dev_metrics()
                    nll = fsa.nll_windows(model, select_tensor, dev)
                    candidate = dict(step=step, dev_concept=concept, dev_lure=lure,
                                     select_dnll=sum(a - b for a, b in zip(nll, start_nll)) / len(nll), nnz=nnz,
                                     slots=evaluate.slot_stats(model), state=snapshot(modules, frozen),
                                     prev_support=copy.deepcopy(prev_support),
                                     nonzero={name: (m.delta() != 0).cpu() for name, m in modules.items() if hasattr(m, 'M')})
                    candidates.append(candidate)
                    print(json.dumps({k: v for k, v in candidate.items() if k not in ('state', 'prev_support', 'nonzero')}), flush=True)
    finally:
        if local is not None:
            for hook in local.h:
                hook.remove()
            local.num, local.den = [], []
    chosen = select_checkpoint(candidates, genome_lure, args.rollback_dnll)
    rolled_back = chosen is None
    restore(modules, start_state if rolled_back else chosen['state'], frozen)
    selected_concept = start_concept if rolled_back else chosen['dev_concept']
    ledger = session_ledger(start_ledger, ledger, rolled_back, lam)
    surviving_births, promoted = 0, 0
    if not rolled_back and args.arm != 'loramid':
        chosen_index = next(i for i, candidate in enumerate(candidates) if candidate is chosen)
        for name, m in modules.items():
            if args.arm == 'tmid':
                earlier = chosen['prev_support'][name]
            elif chosen_index > 0:
                earlier = candidates[chosen_index - 1]['nonzero'][name]
            else:
                earlier = torch.zeros_like(chosen['nonzero'][name])
            eligible = m.support & earlier.to(dev) & (m.delta() != 0) & ~frozen[name]
            surviving_births += int(eligible.sum())
            before = int(frozen[name].sum())
            frozen[name] = promotion_mask(m.support, frozen[name], selected_concept, start_concept, m.delta() != 0, earlier)
            promoted += int(frozen[name].sum()) - before
    replay, replay_records = session_replay(replay, replay_records, items, teacher, corpus['claim_schedule'], args.session, rolled_back)
    evaluate.write_json(out / 'replay.json', replay)
    torch.save({k: r.state_dict() for k, r in replay_records.items()}, out / 'replay.pt')
    metrics = evaluate.run_all(model, tok, dev, corpus, root, args.arm, args.session, subsets=subsets)
    metrics.update(rolled_back=rolled_back, chosen_step=None if rolled_back else chosen['step'],
                   start_dev_concept=start_concept, dropped_zero_gate=dropped,
                   selection=[{k: v for k, v in c.items() if k not in ('state', 'prev_support', 'nonzero')} for c in candidates])
    ledger.setdefault('sessions', {})[str(args.session)] = dict(births_total=births_total, surviving_births=surviving_births, promoted=promoted, rolled_back=rolled_back,
                                                               dev=metrics['dev'], test=metrics['test'],
                                                               chosen_step=metrics['chosen_step'],
                                                               router_counts=counts)
    ledger.setdefault('superset', {})[str(args.session)] = superset
    ledger['frozen_count'] = sum(int(mask.sum()) for mask in frozen.values())
    ledger['claim_recurrence'] = {h: sum(s <= args.session for s in schedule) for h, schedule in corpus['claim_schedule'].items()}
    evaluate.write_json(out / 'ledger.json', ledger)
    save_state(out / 'state.pt', modules, frozen, ledger, replay)
    state_roundtrip(out / 'state.pt', model, modules)
    committed_nnz = evaluate.slot_stats(model)['nnz']
    if args.smoke or args.session == 5:
        # Destructive check runs after committed artifacts are saved; wipe its temporary replay/ledger copy.
        # Temporary personal caches (teacher/start records, acquisition rows, checkpoint snapshots) are
        # released first: revoke must leave nothing experience-bearing resident.
        teacher.clear(); start_records.clear(); acquisition.clear(); candidates.clear(); anchor_entries.clear()
        torch.cuda.empty_cache()
        revoke_dir = out / 'revoke'
        revoke_dir.mkdir(exist_ok=True)
        evaluate.write_json(revoke_dir / 'replay.json', replay)
        torch.save({k: r.state_dict() for k, r in replay_records.items()}, revoke_dir / 'replay.pt')
        evaluate.write_json(revoke_dir / 'ledger.json', ledger)
        metrics['revoke'] = evaluate.revoke_test(model, tok, dev, corpus, genome, root / 'genome/revoke_logits.pt',
                                                 optimizer=opt, replay_dir=revoke_dir, ledger=copy.deepcopy(ledger),
                                                 replay=replay, frozen=frozen)
    evaluate.write_json(out / 'metrics.json', metrics)
    if args.smoke:
        print(json.dumps({'phase': 'smoke_coverage', 'births': births_total, 'evictions': evictions_total,
                          'nnz_committed': committed_nnz, 'refreshes': refreshes, 'lambda': lam,
                          'rehydrated': args.session > 1, 'revoke_pass': metrics['revoke']['pass']}), flush=True)


if __name__ == '__main__':
    main()
