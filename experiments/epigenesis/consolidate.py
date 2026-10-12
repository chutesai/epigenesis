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
import epi_mask
from epi_common import TopKRecord, ONEOFF_RE, gate_weights, mix_weights, pack_rows, topk_kl
import evaluate
import epi_fold


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


def supervised_cases(corpus, session, seed, tok, bos):
    # Import lazily: ablate also imports consolidation's pure loss helpers.
    from ablate import build_cases, encode_item
    entries = []
    for case in build_cases(corpus, session, seed, cases_per_policy=32, appearances_only=True):
        case['id'] = f"s{session}-{case['id']}"
        variants = [dict(encode_item(case, fmt, tok, bos), objective='supervised')
                    for fmt in case['formats']]
        entries.append(dict(variants[0], variants=variants, session=session))
    if not entries:
        raise ValueError('Session has no supervised policy cases')
    return entries


def supervised_replay(previous, items, cap=256):
    # Only ground-truth cases enter this buffer; round-robin over policies.
    buckets = defaultdict(list)
    for entry in {e['id']: e for e in previous + items}.values():
        buckets[entry['policy_id']].append(copy.deepcopy(entry))
    rng = random.Random(items[0]['session'])
    keys = sorted(buckets)
    rng.shuffle(keys)
    for entries in buckets.values():
        entries.sort(key=lambda e: (e['session'], e['id']))
    selected = []
    while any(buckets.values()) and len(selected) < cap:
        for key in keys:
            if buckets[key] and len(selected) < cap:
                selected.append(buckets[key].pop())
    return selected


def refresh_due(step, arm, until=48, objective='selfstudy', legacy_bop=None):
    if arm == 'tbop':
        # Preserve v11c's legacy cadence when the new mode is off.
        if legacy_bop is None:
            legacy_bop = objective == 'selfstudy' and until == 48
        if legacy_bop:
            return (step - 1) % 10 == 0 and step - 1 <= 200
        return (step - 1) % 10 == 0 and step <= until
    return step % 8 == 0 and step <= until


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
    scale = scale_of(m)
    if getattr(m, 'mdl_empty_scales', False):
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    raw = m.delta().detach() / scale
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
        if isinstance(m, epi_fold.FoldLoRAParam):
            state[name] = dict(A=m.A.detach().cpu().clone(), B=m.B.detach().cpu().clone(),
                               committed=m.codes().cpu().clone())
        elif hasattr(m, 'M'):
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
                if 'scale_rows' in saved:
                    m.logb.fill_(-torch.inf)
                    m.logb[saved['scale_rows'].to(m.logb.device)] = saved['logb'].to(m.logb.device)
                else:
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
            if isinstance(m, epi_fold.FoldLoRAParam) and not torch.equal(m.codes().cpu(), saved['committed']):
                raise AssertionError(f'Rehydration changed committed patch: {name}')


def save_state(path, modules, frozen, ledger, replay):
    state = snapshot(modules, frozen, committed=True)
    for name, m in modules.items():
        if getattr(m, 'mdl_empty_scales', False) and m.logb is not None:
            # Empty groups carry no numeric scale payload, only structural row presence.
            rows = torch.isfinite(state[name]['logb']).any(-1)
            state[name]['scale_rows'] = rows
            state[name]['logb'] = state[name]['logb'][rows]
    torch.save(dict(version=1, modules=state,
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


def loss_assembly(student, terms, dev, vocab, mix='balanced', replay_empty=False, micro_tokens=1024, backward=False, coefficients=None, gradient_evidence=None):
    """Mean per term, scaled by each microbatch's loss-bearing token share. Student returns logits."""
    totals, logs = {}, {}
    for key in ('acq', 'gen', 'rep'):
        totals[key] = sum(int((e['weights'] > 0).sum()) if e.get('weights') is not None else (len(e['row']) - e['start'] if e.get('objective') == 'supervised' else e['record'].ids.shape[0])
                          for e in terms.get(key, []))
        logs[key] = 0.0
    coefficients = dict(zip(('acq', 'gen', 'rep'), mix_weights(mix, replay_empty) if coefficients is None else coefficients))
    assembled = None
    for key in ('acq', 'gen', 'rep'):
        if totals[key] == 0:
            continue
        batches = micro_batches(terms.get(key, []), micro_tokens)
        if gradient_evidence is not None:
            batches = ([entry] for packed in batches for entry in packed)
        for batch in batches:
            x, _ = pack_rows([e['row'] for e in batch], [e['start'] for e in batch], dev)
            logits = student(x)
            contribution = None
            for i, entry in enumerate(batch):
                lp = logits[i, entry['start'] - 1:len(entry['row']) - 1, :vocab].float().log_softmax(-1)
                if entry.get('objective') == 'supervised':
                    nt = len(entry['row']) - entry['start']
                    loss = torch.nn.functional.nll_loss(lp, x[i, entry['start']:len(entry['row'])])
                else:
                    loss, nt = topk_kl(entry['record'], lp, entry.get('weights'))
                part = loss * (nt / totals[key])
                logs[key] += float(part.detach())
                contribution = part if contribution is None else contribution + part
            contribution = contribution * coefficients[key]
            if backward:
                if contribution.requires_grad:
                    if gradient_evidence is not None:
                        gradient_evidence.concept = batch[0].get('policy_id') or batch[0].get('fact_id') or '__unlabelled__'
                    try:
                        contribution.backward()
                    finally:
                        if gradient_evidence is not None:
                            gradient_evidence.concept = None
                del logits, lp, loss, part, contribution
            else:
                assembled = contribution if assembled is None else assembled + contribution
    logs['tokens'] = totals
    logs['loss'] = sum(coefficients[k] * logs[k] for k in coefficients)
    return assembled, logs


def tbop_weights(replay_empty=False):
    # EP-1 fact loss: balanced acq:rep = .4:.2, renormalized, then v11c fact_scale=.8.
    return (.8, 0., 0.) if replay_empty else (.8 * 2 / 3, 0., .8 / 3)


def session_objective(arm, replay_empty=False):
    """Arm-specific acquisition/rehearsal weights and local penalty."""
    if arm == 'tbop':
        return True, tbop_weights(replay_empty), 10
    if arm == 'lorafold':
        return True, (.8, 0., 0.), 10
    return False, None, 100 if arm == 'loramid' else 10


def rehearsal_backward(student, anchors, dev, vocab):
    value = 0.
    for entry in anchors:
        x = torch.tensor([entry['row']], device=dev)
        loss = general_ce(student, x, vocab)
        value += float(loss.detach()) / len(anchors)
        if loss.requires_grad:
            (.2 * loss / len(anchors)).backward()
    return value


@torch.no_grad()
def calibrate_bop_tau(mods):
    # v11c: first reached answer-gradient RMS, before rehearsal/penalty, undo fact_scale.
    gradients = [m.M.grad[m.mask].flatten() for m in mods if m.M.grad is not None]
    if not gradients or not sum(g.numel() for g in gradients):
        return None
    return .7 * float(torch.cat(gradients).square().mean().sqrt()) / .8


def general_ce(student, windows, vocab):
    # Port of e2e_fsa.general_ce: patched next-token CE, including all window targets.
    lg = student(windows)[:, :-1, :vocab].float()
    return torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), windows[:, 1:].reshape(-1))


@torch.no_grad()
def bop_step(modules, frozen, allowed, gamma, tau):
    # Port of reference bop_step without optional churn/grace/energy-threshold variants.
    # EP-1 protects frozen/out-of-superset coordinates, including stale EMA evidence.
    nb = nd = 0
    for name, m in modules.items():
        writable = m.mask & ~frozen[name] & allowed[name]
        m.bop_ema.masked_fill_(~writable, 0)
        m.bop_ema.mul_(1 - gamma)
        if m.M.grad is None:
            continue
        m.bop_ema.add_(m.M.grad * writable, alpha=gamma)
        q = torch.sign(m.M) * (m.M.abs() > .5)
        mag = m.bop_ema.abs()
        birth = (q == 0) & (mag > tau) & writable & m.support
        death = (q != 0) & (mag > tau) & (torch.sign(m.bop_ema) == q) & writable
        m.M[birth] = -torch.sign(m.bop_ema[birth])
        m.M[death] = 0
        m.bop_ema[birth | death] = 0
        nb += int(birth.sum())
        nd += int(death.sum())
    return nb, nd


def checkpoint_recurrences(model, class_name='EDALayer'):
    """Activation checkpointing for the EDA recurrence layers on the student's grad path. The naive fp32 EDA scan
    stores per-timestep state for backward; every EDA layer above the patched MoE layers did so, which took the
    resident ~24 GiB (model + masters + Adam + grads + netcost buffers) past 31 GiB at step 2. EDALayer.forward is a
    pure function of its input, so recompute in backward is exact. No-grad forwards (teacher, eval) are untouched."""
    from torch.utils.checkpoint import checkpoint
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != class_name:
            continue
        def wrapped(x, *args, _f=mod.forward, **kw):
            if torch.is_grad_enabled() and x.requires_grad:
                return checkpoint(lambda t: _f(t, *args, **kw), x, use_reentrant=False)
            return _f(x, *args, **kw)
        mod.forward = wrapped
        n += 1
    return n


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


def cache_items(model, dev, items, chunks, args, null_thr, include_zero_gate=False):
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
        if int(weights.sum()) > 0 or include_zero_gate:
            acquisition.append(dict(item, row=sr, start=ss, record=rec, weights=weights))
    return teacher, start, acquisition, dropped


def utility_backward(student, entries, dev, vocab, limit, seed):
    """Seeded item mean of acquisition/replay KL; accumulate signed gradients before taking abs."""
    if limit < 1:
        raise ValueError('util-items must be positive')
    selected = random.Random(seed).sample(entries, min(limit, len(entries)))
    for entry in selected:
        x, _ = pack_rows([entry['row']], [entry['start']], dev)
        logits = student(x)
        lp = logits[0, entry['start'] - 1:len(entry['row']) - 1, :vocab].float().log_softmax(-1)
        if entry.get('objective') == 'supervised':
            loss = torch.nn.functional.nll_loss(lp, x[0, entry['start']:len(entry['row'])])
        else:
            loss, _ = topk_kl(entry['record'], lp, entry.get('weights'))
        loss = loss * entry['coefficient'] / len(selected)
        if loss.requires_grad:
            loss.backward()
    return len(selected)


@torch.no_grad()
def release_slots(modules, frozen, identity, rho, optimizer=None):
    """Score every filled coordinate, including frozen/out-of-superset slots, layer by layer."""
    assert optimizer is None or not optimizer.state, 'Release requires empty optimizer state'
    if not math.isfinite(rho) or rho < 0:
        raise ValueError('release-rho must be finite and nonnegative')
    per_layer, means = {}, {}
    for li in (14, 15):
        filled_count, total = 0, 0.0
        for name, m in modules.items():
            if identity[name][0] != li:
                continue
            filled = m.delta() != 0
            # w = scale*M and STE dL/dM = scale*dL/dw at committed M in {-1,0,+1}:
            # |w*dL/dw| = |M*M.grad|; the scale cancels. Do not protect/mask gradients here.
            utility = torch.zeros_like(m.M) if m.M.grad is None else (m.M * m.M.grad).abs()
            filled_count += int(filled.sum())
            total += float(utility[filled].sum())
        mean = total / filled_count if filled_count else 0.0
        released = 0
        for name, m in modules.items():
            if identity[name][0] != li:
                continue
            utility = torch.zeros_like(m.M) if m.M.grad is None else (m.M * m.M.grad).abs()
            release = (m.delta() != 0) & (utility < rho * mean)
            released += int(release.sum())
            m.M[release] = 0
            m.support[release] = False
            frozen[name][release] = False
        per_layer[str(li)] = dict(filled=filled_count, released=released, threshold=rho * mean)
        means[str(li)] = mean
    return dict(filled=sum(v['filled'] for v in per_layer.values()),
                released=sum(v['released'] for v in per_layer.values()), per_layer=per_layer,
                rho=rho, util_mean=means)


def utility_release(model, dev, modules, frozen, identity, replay, replay_records, items, study, args):
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    try:
        if getattr(args, 'objective', 'selfstudy') == 'supervised':
            acquisition = [e['variants'][0] for e in items]
        else:
            _, _, acquisition, _ = cache_items(model, dev, items, study['chunk_tokens'], args,
                                                study['null_thr'], include_zero_gate=True)
        acq, _, rep = mix_weights(args.mix, not replay)
        entries = [dict(e, coefficient=acq) for e in acquisition]
        for item in replay:
            if getattr(args, 'objective', 'selfstudy') == 'supervised':
                entries.append(dict(item['variants'][0], coefficient=rep))
                continue
            row, start = item_row(item)
            entries.append(dict(row=row, start=start, record=replay_records[item['id']], coefficient=rep))
        utility_backward(lambda x: gpu().logits_of(model, x), entries, dev, gpu().VOCAB,
                         args.util_items, args.seed + args.session)
        result = release_slots(modules, frozen, identity, args.release_rho)
        result.update(phase='utility_release', seconds=time.monotonic() - started,
                      peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        print(json.dumps(result), flush=True)
        return result
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def mdl_group_key(name, row):
    layer, expert, projection = name.split('.')
    return (int(layer), int(expert), projection, row // 64)


def mdl_group_bits(nnz, reserve, scales):
    # Exactly the budgeted ternary entry used by slot_stats (global reserve).
    return evaluate.aggregate_slots([dict(layer=0, expert=0, rows=[nnz], reserve=reserve,
                                         bits=math.log2(3), budgeted=True, scales=scales)])['bytes'] * 8


def mdl_release_rule(bits_saved, storage_bits, conflicts=()):
    """Strict threshold; overlapping recent versions keep the larger recent-text gain (ties keep both)."""
    if bits_saved >= storage_bits:
        return False
    return not any(older >= newer for older, newer in conflicts)


def mdl_groups(modules):
    groups = []
    for name, m in modules.items():
        filled = m.delta().detach() != 0
        for lo in range(0, m.M.shape[0], 64):
            hi = min(lo + 64, m.M.shape[0])
            if filled[lo:hi].any():
                groups.append((mdl_group_key(name, lo), name, lo, hi))
    return groups


def mdl_backward(model, tok, dev, texts, micro_tokens=160):
    """Summed window NLL gradients, with a strict padded-token memory bound."""
    g = gpu()
    model.zero_grad(set_to_none=True)
    for x, positions in evaluate.heldout_batches(tok, texts, dev, micro_tokens):
        logits = g.logits_of(model, x)
        loss = sum(torch.nn.functional.cross_entropy(
            logits[row, :length - 1, :g.VOCAB].float(), x[row, 1:length], reduction='sum')
            for row, (_, length) in enumerate(positions))
        if loss.requires_grad:
            loss.backward()


@torch.no_grad()
def mdl_scoring_nll(model, tok, dev, texts, nograd_tokens):
    return evaluate.heldout_nll(model, tok, dev, texts, nograd_tokens)


@torch.no_grad()
def mdl_leaveout(model, tok, dev, texts, m, mask, baseline, micro_tokens=160):
    """Restore even if scoring raises; removing values alone is an exact patch leave-out."""
    saved = m.M[mask].clone()
    try:
        m.M[mask] = 0
        return (sum(mdl_scoring_nll(model, tok, dev, texts, micro_tokens)) - baseline) / math.log(2)
    finally:
        m.M[mask] = saved


def mdl_release(model, tok, dev, modules, frozen, corpus, ledger, args):
    from make_user import heldout_sample
    if torch.device(dev).type == 'cuda':
        torch.cuda.synchronize(dev)
    started = time.monotonic()
    micro = getattr(args, 'micro_tokens', 160)
    nograd = getattr(args, 'nograd_tokens', 4096)
    texts = [t for k in range(1, args.session) for t in heldout_sample(corpus, k, corpus['seed'])]
    recent = heldout_sample(corpus, args.session - 1, corpus['seed'])
    groups = mdl_groups(modules)
    reserve = sum(int(m.mask.sum()) for m in modules.values())
    try:
        mdl_backward(model, tok, dev, texts, micro)
        def utility(group):
            _, name, lo, hi = group
            m = modules[name]
            return float((m.M.detach()[lo:hi] * m.M.grad[lo:hi]).abs().sum()) if m.M.grad is not None else 0.
        selected = sorted(groups, key=lambda g: (utility(g), g[0]))[:args.mdl_max_groups]
        model.zero_grad(set_to_none=True)
        baseline = sum(mdl_scoring_nll(model, tok, dev, texts, nograd))
        provenance = ledger.get('mdl_concepts', {})
        replacements = corpus.get('replacements', [])
        records, decisions = [], {}
        gains = {}
        recent_base = None
        def recent_gain(group):
            nonlocal recent_base
            key, name, lo, hi = group
            if key not in gains:
                if recent_base is None:
                    recent_base = sum(mdl_scoring_nll(model, tok, dev, recent, nograd))
                m = modules[name]
                mask = torch.zeros_like(m.support)
                mask[lo:hi] = m.delta().detach()[lo:hi] != 0
                gains[key] = mdl_leaveout(model, tok, dev, recent, m, mask, recent_base, nograd)
            return gains[key]
        # Only selected groups can be deleted, but compare competitors anywhere in the patch.
        for group in selected:
            key, name, lo, hi = group
            m = modules[name]
            scoring = torch.zeros_like(m.support)
            scoring[lo:hi] = m.delta().detach()[lo:hi] != 0
            storage = mdl_group_bits(int(scoring.sum()), reserve,
                                    0 if m.logb is None else int(torch.isfinite(m.logb[lo:hi]).sum()))
            saved = mdl_leaveout(model, tok, dev, texts, m, scoring, baseline, nograd)
            delete = mdl_release_rule(saved, storage)
            conflicts = []
            own = provenance.get(':'.join(map(str, key)), {})
            for replacement in replacements:
                if replacement['session'] >= args.session:
                    continue
                concept = replacement['concept']
                version = own.get(concept, {}).get('version')
                if version not in (replacement['old_version'], replacement['new_version']):
                    continue
                other_version = replacement['new_version'] if version == replacement['old_version'] else replacement['old_version']
                for competitor in groups:
                    other = provenance.get(':'.join(map(str, competitor[0])), {})
                    if competitor[0] == key or other.get(concept, {}).get('version') != other_version:
                        continue
                    gain, other_gain = recent_gain(group), recent_gain(competitor)
                    conflicts.append(dict(concept=concept, version=version, competitor=competitor[0],
                                          bits=gain, competitor_bits=other_gain))
            if conflicts:
                # Preserve any winner/tie; release a group only if it loses every competition.
                delete = all(r['bits'] < r['competitor_bits'] for r in conflicts)
            decisions[key] = delete
            records.append(dict(group=key, bits_saved=saved, storage_bits=storage,
                                released_slots=int(m.support[lo:hi].sum()) if delete else 0,
                                released=delete, conflicts=conflicts))
        with torch.no_grad():
            for key, name, lo, hi in selected:
                if not decisions[key]:
                    continue
                m = modules[name]
                # Deletion spans ALL columns, including unsupported shadow masters.
                m.M[lo:hi] = 0
                m.support[lo:hi] = False
                frozen[name][lo:hi] = False
                if m.score is not None:
                    m.score[lo:hi] = 0
                if m.logb is not None:
                    m.logb[lo:hi] = -torch.inf
                ledger.get('mdl_concepts', {}).pop(':'.join(map(str, key)), None)
        if torch.device(dev).type == 'cuda':
            torch.cuda.synchronize(dev)
        result = dict(phase='mdl_release', evaluated=len(selected), candidates=len(groups),
                      released=sum(r['released'] for r in records),
                      released_slots=sum(r['released_slots'] for r in records),
                      bits_saved=sum(r['bits_saved'] for r in records),
                      storage_bits=sum(r['storage_bits'] for r in records), groups=records,
                      sample_per_session=16, micro_tokens=micro, nograd_tokens=nograd, packing='one_document_per_row',
                      seconds=time.monotonic() - started)
        print(json.dumps(result), flush=True)
        return result
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


class MDLGradientEvidence:
    """Observe actual per-item backward gradients; retain only CPU group sums."""
    def __init__(self, modules):
        self.concept = None
        self.mass = {}
        self.hooks = []
        for name, m in modules.items():
            def observe(grad, name=name):
                if self.concept is not None:
                    rows = grad.detach().abs().sum(-1)
                    padding = (-len(rows)) % 64
                    grouped = torch.nn.functional.pad(rows, (0, padding)).reshape(-1, 64).sum(-1).cpu()
                    bucket = self.mass.setdefault(name, {})
                    if self.concept not in bucket:
                        bucket[self.concept] = torch.zeros_like(grouped)
                    bucket[self.concept].add_(grouped)
                return grad
            self.hooks.append(m.M.register_hook(observe))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def mdl_birth_records(model, tok, dev, modules, start_state, corpus, session,
                      provenance, evidence, micro_tokens=160):
    """Record committed births from gradients produced by actual training items.

    A concept needs a strict majority of the group's absolute gradient mass;
    unlabelled acquisition/rehearsal items count in the denominator.
    """
    records = copy.deepcopy(provenance)
    concepts = {v['id']: v for kind in ('policies', 'facts') for v in corpus.get(kind, [])}
    for key, name, lo, hi in mdl_groups(modules):
        before = start_state[name].get('M')
        if before is None:
            before = start_state[name]['committed']
        born = (modules[name].delta().detach()[lo:hi] != 0) & (before[lo:hi].to(dev).abs() <= .5)
        if not born.any():
            continue
        label = ':'.join(map(str, key))
        group_mass = {concept: float(mass[key[-1]]) for concept, mass in evidence.mass.get(name, {}).items()}
        total = sum(group_mass.values())
        record = records.setdefault(label, {})
        for concept, mass in group_mass.items():
            if concept in concepts and mass > total / 2:
                version = concepts[concept].get('version', 1)
                if record.get(concept, {}).get('version') != version:
                    record[concept] = dict(version=version, session=session)
    return records


def mdl_enable_birth_gradients(m):
    """Keep empty scales absent in forward, with init-scale STE for future proposals."""
    if m.logb is None:
        return
    def delta():
        master = m.M * m.mask
        # ternary round written out: the box harness's FSAParam predates the reference's _quant helper
        quant = torch.sign(master) * (master.abs() > 0.5).float() * m.support
        scales = m.logb.exp().repeat_interleave(m.block, dim=1)
        initial = (m.alpha.float() * .25).expand_as(master)
        surrogate = torch.where(scales == 0, initial, scales)
        # Zero-valued STE term supplies proposal gradients without storing scale bits.
        return quant.detach() * scales + (master - master.detach()) * surrogate.detach()
    m.delta = delta
    m.mdl_empty_scales = True
    mdl_reoccupy_scales(m)


@torch.no_grad()
def mdl_reoccupy_scales(m):
    """Initialize only an empty 64-row group that has just admitted support."""
    if not getattr(m, 'mdl_empty_scales', False) or m.logb is None:
        return
    for lo in range(0, m.M.shape[0], 64):
        hi = lo + 64
        if not m.support[lo:hi].any():
            m.logb[lo:hi] = -torch.inf
        elif torch.isneginf(m.logb[lo:hi]).all():
            m.logb[lo:hi] = torch.log(m.alpha[lo:hi].float() * .25).expand_as(m.logb[lo:hi])


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
        if getattr(m, 'mdl_empty_scales', False):
            scale = torch.where(scale == 0, m.alpha[row, 0] * .25, scale)
        # Row-scaled ladder patches use beta * alpha, including calibration.
        scale = scale * getattr(m, 'beta_factors', (1.,))[0]
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
        if getattr(m, 'mdl_empty_scales', False):
            scale = torch.where(scale == 0, m.alpha[start:stop].float() * .25, scale)
        scale = scale * getattr(m, 'beta_factors', (1.,))[0]
        score[start:stop] = (-grad * direction * mag - lam * scale.square() * m.col_energy[None, :] / den * mag.square()) * m.mask[start:stop]
    return score


@torch.no_grad()
def refresh_support_cpu(mods, K, scores, opt=None):
    """e2e_fsa.refresh_support with the global selection on the CPU: keep at most K slots with the largest POSITIVE
    score (exact indices, ties cannot exceed K); evicted masters and their optimizer state reset to 0."""
    sc = torch.cat([(s.float() * m.mask.cpu()).flatten() for m, s in zip(mods, scores)])
    K = min(K, int((sc > 0).sum()))
    sel = torch.zeros(sc.numel(), dtype=torch.bool)
    if K > 0:
        sel[torch.topk(sc, K).indices] = True
    del sc
    off = 0
    for m in mods:
        n = m.mask.numel()
        new = sel[off:off + n].view(m.mask.shape).to(m.mask.device) & m.mask
        off += n
        ev = m.support & ~new
        m.M[ev] = 0
        st = opt.state.get(m.M, {}) if opt is not None else {}
        for k_ in ('exp_avg', 'exp_avg_sq', 'momentum_buffer'):
            if torch.is_tensor(st.get(k_)) and st[k_].shape == m.M.shape:
                st[k_][ev] = 0
        if hasattr(m, 'bop_ema'):
            m.bop_ema[ev] = 0
        m.support.copy_(new)
        mdl_reoccupy_scales(m)
    assert sum(int(m.support.sum()) for m in mods) <= K


@torch.no_grad()
def refresh(modules, frozen, allowed, budget, lam, den, opt):
    """Reference refresh keeps global top-K; +inf reserves frozen and out-of-superset prior slots."""
    fsa = gpu().fsa
    mods = list(modules.values())
    # scores live on the CPU: a global concat + topk over ~453M slots on the GPU needed +3.4 GiB on top of a
    # ~28 GiB training state and OOMed at the first refresh (smoke 2026-10-09 21:00Z)
    scores = [netcost_score(m, lam, den).float().cpu() for m in mods]
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
    refresh_support_cpu(mods, budget, scores, opt)
    births = sum(int((m.support & ~old).sum()) for m, old in zip(mods, before))
    evictions = sum(int((old & ~m.support).sum()) for m, old in zip(mods, before))
    return births, evictions


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-set', choices=('inpair', 'forced'), default='inpair')
    p.add_argument('--run', required=True)
    p.add_argument('--arm', choices=('tmid', 'tmid_lr005', 'fp4mid', 'loramid', 'lorafold', 'tmid_util01', 'tmid_util03', 'tmid_mdl', 'tbop'), required=True)
    p.add_argument('--session', type=int, choices=range(1, 6), required=True)
    p.add_argument('--objective', choices=('selfstudy', 'supervised'), default='selfstudy')
    p.add_argument('--mix', choices=('balanced', 'acquire', 'retain'), default='balanced')
    # measured on the dev box (DESIGN.md section 6): resident state after step 1 (model + masters + Adam + grads +
    # netcost score/support) is ~23.4 GiB and a 256-token micro-batch adds ~8 GiB of naive-EDA activations (29.4
    # GiB at step 1, OOM at step 2 in the first smoke), so micro-batches are capped at 160 tokens / anchors at 96
    for name, default in [('steps', 48), ('items-per-step', 8), ('anchor-per-step', 2), ('replay-per-step', 4),
                          ('micro-tokens', 160), ('eval-every', 16), ('budget', 2000000), ('topk', 32),
                          ('anchor-windows', 64), ('anchor-len', 96), ('seed', 0), ('nograd-tokens', 4096), ('util-items', 64), ('mdl-max-groups', 256)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--refresh-until', type=int, default=None)
    p.add_argument('--lr-mult', type=float, default=1.0)
    p.add_argument('--ent-max', type=float, default=2.0)
    p.add_argument('--rollback-dnll', type=float, default=.005)
    p.add_argument('--release-rho', type=float, help='Arm-derived: 0.1 for tmid_util01, 0.3 for tmid_util03')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    args.legacy_bop_refresh = args.refresh_until is None and args.objective == 'selfstudy'
    if args.refresh_until is None:
        args.refresh_until = 48
    if args.arm == 'tmid_lr005':
        args.lr_mult = 2.5
    tbop = args.arm == 'tbop'
    if tbop:
        # Fixed post-hoc v11c settings; lr_mult has no effect on Bop.
        args.budget, args.mix, args.anchor_per_step, args.lr_mult = 300000, 'balanced', 2, 1.
    rho = {'tmid_util01': .1, 'tmid_util03': .3}.get(args.arm)
    if args.release_rho is not None and args.release_rho != rho:
        p.error('--release-rho must match the utility arm name')
    args.release_rho = rho
    if args.mdl_max_groups < 0:
        p.error('--mdl-max-groups must be nonnegative')
    if args.util_items < 1:
        p.error('--util-items must be positive')
    if args.smoke:
        args.steps, args.items_per_step, args.eval_every = 10, 8, 5
        # lr x5 for 10 steps overshoots the +0.005 per-session budget by design; the smoke must still commit a patch
        # so save/rehydrate/revoke are exercised (the full run keeps the pre-registered +0.005)
        args.rollback_dnll = max(args.rollback_dnll, .1)
        args.mdl_max_groups = min(args.mdl_max_groups, 8)   # smoke exercises the release path, not its full cost
        if args.arm not in ('loramid', 'lorafold') and not tbop:
            args.lr_mult = 5
    if not math.isfinite(args.lr_mult) or args.lr_mult <= 0:
        p.error('--lr-mult must be finite and positive')
    if min(args.steps, args.items_per_step, args.anchor_per_step, args.micro_tokens, args.eval_every,
           args.anchor_windows, args.nograd_tokens) < 1 or args.refresh_until < 0 or args.anchor_len < 2 or args.replay_per_step < 0 or args.budget < 0:
        p.error('Invalid nonpositive training configuration')
    if args.anchor_len > args.micro_tokens:
        p.error('--micro-tokens must be at least --anchor-len')
    return args


def main():
    args = parse_args()
    tbop = args.arm == 'tbop'
    ternary = args.arm.startswith('tmid') or tbop
    root = Path(args.run)
    out = root / args.arm / f's{args.session}'
    out.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((root / 'corpus.json').read_text())
    study = None
    items = []
    if args.objective == 'selfstudy':
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
    print(json.dumps(dict(phase='checkpointed_eda_layers', n=checkpoint_recurrences(model))), flush=True)
    tok = g.load_tokenizer()
    if getattr(args, 'objective', 'selfstudy') == 'supervised':
        items = supervised_cases(corpus, args.session, args.seed, tok, g.BOS)
    model._epi_write_set = dict(write_set=args.write_set, mask_sha256=None)
    if args.write_set == 'forced' and args.arm != 'loramid':
        params = epi_mask.attach(model, fsa, fmt='ternary' if ternary else 'fp4', budgeted=ternary,
                                 rank=16 if args.arm == 'lorafold' else None)
    elif args.arm == 'lorafold':
        params = epi_fold.attach(model, fsa)
    else:
        params, _, _ = fsa.attach(model, [14, 15], 'lora' if args.arm == 'loramid' else 'fsa_free', 16,
                                 fmt='ternary' if ternary else 'fp4', budgeted=ternary)
    mods = fsa.fsa_modules(model)
    for m in mods:
        m.fp4_step = .25
        if ternary:
            if not tbop and args.write_set == 'inpair':
                m.enable_block_scale(16, .25)
                params.append(m.logb)
            m.support.zero_()
        if tbop:
            # Session boundary: committed ternary M rehydrates below; EMA alone is reset.
            m.bop_ema = torch.zeros_like(m.M)
    modules, identity = patch_modules(model)
    frozen = {name: torch.zeros_like(m.mask) for name, m in modules.items() if hasattr(m, 'M')}
    ledger, replay, replay_records = {}, [], {}
    if args.session > 1:
        previous = root / args.arm / f's{args.session - 1}'
        ledger, replay = load_state(previous / 'state.pt', modules, frozen)
        if any(ledger.get(k, None if k == 'mask_sha256' else 'inpair') != v
               for k, v in epi_mask.metadata(model).items()):
            raise ValueError('Previous state write set or mask sha256 mismatch')
        replay_records = {k: TopKRecord.from_state_dict(v) for k, v in
                          torch.load(previous / 'replay.pt', map_location='cpu', weights_only=True).items()}
    if args.arm == 'tmid_mdl':
        for m in modules.values():
            mdl_enable_birth_gradients(m)
    fsa.set_enabled(True)
    release = None
    if args.release_rho is not None and args.session >= 2:
        release = utility_release(model, dev, modules, frozen, identity, replay, replay_records, items, study, args)
    if args.arm == 'tmid_mdl' and args.session >= 2:
        release = mdl_release(model, tok, dev, modules, frozen, corpus, ledger, args)
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
    if ternary:
        while len(calibration_anchors) < 32:
            offset = rng.randrange(len(pool) - args.anchor_len + 2)
            calibration_anchors.append([g.BOS] + pool[offset:offset + args.anchor_len - 1])
    select_rows = evaluate.token_windows(anchor_tokens[-16 * 511:], 4 if args.smoke else 16)
    select_tensor = torch.tensor(select_rows, device=dev)
    start_nll = fsa.nll_windows(model, select_tensor, dev)
    phase = time.monotonic()
    if getattr(args, 'objective', 'selfstudy') == 'supervised':
        teacher, start_records, acquisition, dropped = {}, {}, items, 0
    else:
        teacher, start_records, acquisition, dropped = cache_items(model, dev, items, study['chunk_tokens'], args, study['null_thr'])
        validate_acquisition(items, study['null_thr'], acquisition)
    anchor_records = []
    ce_rehearsal, objective_weights, local_weight = session_objective(args.arm, not replay)
    for row in ([] if ce_rehearsal else anchors):
        anchor_records.extend(g.topk_records(model, [row], [1], dev, args.topk, args.nograd_tokens))
    anchor_entries = ([dict(row=row, start=1) for row in anchors] if ce_rehearsal else
                      [dict(row=row, start=1, record=rec) for row, rec in zip(anchors, anchor_records)])
    superset, counts = (None, None) if args.arm in ('loramid', 'lorafold') else router_superset(model, dev, items, args)
    allowed = {name: superset is None or ei in superset[str(li)] for name, (li, ei) in identity.items()}
    ledger.setdefault('superset', {})[str(args.session)] = superset
    print(json.dumps(dict(phase='cache', seconds=time.monotonic() - phase, dropped_zero_gate=dropped,
                          acquisition_items=len(acquisition), peak_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
    torch.cuda.empty_cache()
    scales = [m.logb for m in mods if m.logb is not None]
    scale_ids = {id(t) for t in scales}
    lr = .001 if args.arm in ('loramid', 'lorafold') else .02 * args.lr_mult
    groups = [dict(params=[t for t in params if id(t) not in scale_ids], lr=lr, weight_decay=0)]
    if scales:
        groups.append(dict(params=scales, lr=.002 * args.lr_mult, weight_decay=0))
    # v11c placeholder optimizer: zero_grad only, no state or decay.
    opt = (torch.optim.SGD(params, lr=0.) if tbop else
           torch.optim.Adam(groups) if args.arm == 'lorafold' else torch.optim.AdamW(groups))
    # tbop uses the existing microbatched local penalty with frozen-base denominator.
    local = fsa.LocalOutputPenalty(model) if tbop or not ternary else None
    den = fsa.collect_col_energy(model, mods, torch.tensor(calibration_anchors, device=dev)) if ternary else None
    if ternary and (not math.isfinite(den) or den <= 0):
        raise ValueError('Invalid anchor energy denominator')
    config_path = root / args.arm / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    lam = config.get('netcost_lambda', ledger.get('netcost_lambda'))
    config.update(epi_mask.metadata(model))
    config.update(objective=args.objective, eval_every=args.eval_every, refresh_until=args.refresh_until, netcost_lambda=lam, lr_mult=args.lr_mult, mix=args.mix, steps=args.steps)
    if args.release_rho is not None:
        config.update(release_rho=args.release_rho, util_items=args.util_items)
    evaluate.write_json(config_path, config)
    if args.arm == 'lorafold':
        config.update(method='lorafold', fmt='ternary', layers=[14, 15], projs=['up_proj', 'down_proj'],
                      rank=16, optim='adam', lr=.001, local_lambda=10, fold_thr=.5)
        evaluate.write_json(config_path, config)
    if ternary and args.session > 1 and lam is None:
        raise ValueError('Session 1 did not calibrate netcost lambda; cannot recalibrate in later sessions')
    tau = config.get('bop_tau_abs', ledger.get('bop_tau_abs')) if tbop else None
    if tbop:
        config.update(method='fsa_free', fmt='ternary', layers=[14, 15], projs=['up_proj', 'down_proj'],
                      n_experts=0, optim='bop', patch_scale='row', wd=0, budget=300000,
                      support_score='netcost', cost_rel=1, local_lambda=10, answer_only=True,
                      rehearsal_frac=.2, rehearsal_bs=2, bop_gamma=.05, bop_tau=.7, bop_tau_abs=tau,
                      budget_freeze=200 if args.legacy_bop_refresh else args.refresh_until, refresh_every=10)
        evaluate.write_json(config_path, config)
        if args.session > 1 and tau is None:
            raise ValueError('Session 1 did not calibrate Bop tau; cannot recalibrate in later sessions')
    gradient_evidence = MDLGradientEvidence(modules) if args.arm == 'tmid_mdl' else None
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
                    if getattr(args, 'objective', 'selfstudy') == 'supervised':
                        from ablate import sample_policies
                        rep.extend(sample_policies([item], 1, rng, augment=True))
                        continue
                    row, start = item_row(item)
                    rep.append(dict(row=row, start=start, record=replay_records[item['id']],
                                    **({k: item[k] for k in ('policy_id', 'fact_id') if k in item}
                                       if gradient_evidence is not None else {})))
                if getattr(args, 'objective', 'selfstudy') == 'supervised':
                    from ablate import sample_policies
                    batch = sample_policies(acquisition, args.items_per_step, rng, augment=True)
                else:
                    batch = sample_episodes(acquisition, args.items_per_step, rng)
                terms = dict(acq=batch, gen=[] if ce_rehearsal else ab, rep=rep)
                _, losses = loss_assembly(lambda x: g.logits_of(model, x), terms, dev, g.VOCAB,
                                          args.mix, not replay, args.micro_tokens, backward=True,
                                          coefficients=objective_weights, gradient_evidence=gradient_evidence)
                if tbop:
                    if tau is None:
                        tau = calibrate_bop_tau(mods)
                        if tau is not None:
                            ledger['bop_tau_abs'] = config['bop_tau_abs'] = tau
                            evaluate.write_json(config_path, config)
                    # Netcost evidence is acquisition/replay only, before rehearsal/penalty.
                    protect_gradients(modules, frozen, allowed)
                    for m in mods:
                        m.score.mul_(.9)
                        if m.M.grad is not None:
                            m.score.add_(m.M.grad, alpha=.1)
                if ce_rehearsal:
                    rehearsal = rehearsal_backward(lambda x: g.logits_of(model, x), ab, dev, g.VOCAB)
                    losses['rehearsal_ce'] = rehearsal
                    losses['loss'] += .2 * rehearsal
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
                    if not tbop and hasattr(m, 'score') and m.score is not None:
                        m.score.mul_(.9)
                        if m.M.grad is not None:
                            m.score.add_(m.M.grad, alpha=.1)
                if tbop:
                    if tau is not None:
                        bop_step(modules, frozen, allowed, .05, tau)
                else:
                    opt.step()
                    manual_decay(modules, frozen, allowed, lr, 0 if args.release_rho is not None else (.3 if ternary else .1))
                nb, ne = 0, 0
                if ternary and refresh_due(step, args.arm, args.refresh_until, args.objective, args.legacy_bop_refresh):
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
                    if gradient_evidence is not None:
                        candidate['mdl_concepts'] = mdl_birth_records(
                            model, tok, dev, modules, start_state, corpus, args.session,
                            ledger.get('mdl_concepts', {}), gradient_evidence, args.micro_tokens)
                    candidates.append(candidate)
                    print(json.dumps({k: v for k, v in candidate.items() if k not in ('state', 'prev_support', 'nonzero', 'mdl_concepts')}), flush=True)
    finally:
        if gradient_evidence is not None:
            gradient_evidence.close()
        if local is not None:
            for hook in local.h:
                hook.remove()
            local.num, local.den = [], []
    pool = candidates
    if args.smoke and args.arm not in ('loramid', 'lorafold'):
        # smoke must exercise save/rehydrate/revoke of a real patch: the pre-refresh step-5 checkpoint has nnz 0 and
        # wins nnz ties, so only nonzero checkpoints are eligible (full runs keep the pre-registered rule)
        pool = [c_ for c_ in candidates if c_['nnz'] > 0]
        if not pool:
            raise SystemExit('smoke: no checkpoint with a nonzero FSA patch')
    # tbop shares tmid checkpoint/rollback/promotion/replay/evaluation/revoke below.
    chosen = select_checkpoint(pool, genome_lure, args.rollback_dnll)
    rolled_back = chosen is None
    restore(modules, start_state if rolled_back else chosen['state'], frozen)
    selected_concept = start_concept if rolled_back else chosen['dev_concept']
    ledger = session_ledger(start_ledger, ledger, rolled_back, lam)
    if tbop:
        # Once-only calibration survives rollback just like netcost lambda.
        ledger.update(bop_tau_abs=tau, bop_gamma=.05)
    surviving_births, promoted = 0, 0
    if not rolled_back and args.arm not in ('loramid', 'lorafold'):
        chosen_index = next(i for i, candidate in enumerate(candidates) if candidate is chosen)
        for name, m in modules.items():
            if ternary:
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
    if getattr(args, 'objective', 'selfstudy') == 'supervised':
        replay = replay if rolled_back else supervised_replay(replay, items)
        replay_records = {}
    else:
        replay, replay_records = session_replay(replay, replay_records, items, teacher, corpus['claim_schedule'], args.session, rolled_back)
    evaluate.write_json(out / 'replay.json', replay)
    torch.save({k: r.state_dict() for k, r in replay_records.items()}, out / 'replay.pt')
    metrics = evaluate.run_all(model, tok, dev, corpus, root, args.arm, args.session, subsets=subsets)
    metrics.update(rolled_back=rolled_back, chosen_step=None if rolled_back else chosen['step'],
                   start_dev_concept=start_concept, dropped_zero_gate=dropped,
                   selection=[{k: v for k, v in c.items() if k not in ('state', 'prev_support', 'nonzero', 'mdl_concepts')} for c in candidates])
    ledger.setdefault('sessions', {})[str(args.session)] = dict(births_total=births_total, surviving_births=surviving_births, promoted=promoted, rolled_back=rolled_back,
                                                               dev=metrics['dev'], test=metrics['test'],
                                                               chosen_step=metrics['chosen_step'],
                                                               router_counts=counts)
    ledger.setdefault('superset', {})[str(args.session)] = superset
    ledger.update(epi_mask.metadata(model))
    ledger['reserve'] = sum(int(m.mask.sum()) for m in modules.values() if hasattr(m, 'mask'))
    ledger['frozen_count'] = sum(int(mask.sum()) for mask in frozen.values())
    ledger['claim_recurrence'] = {h: sum(s <= args.session for s in schedule) for h, schedule in corpus['claim_schedule'].items()}
    if args.release_rho is not None:
        ledger['sessions'][str(args.session)]['released'] = 0 if release is None else release['released']
        ledger['sessions'][str(args.session)]['utility_release'] = release
    if args.arm == 'tmid_mdl':
        ledger['sessions'][str(args.session)]['mdl_release'] = release
        ledger['sessions'][str(args.session)]['released'] = 0 if release is None else release['released_slots']
        if not rolled_back:
            ledger['mdl_concepts'] = chosen['mdl_concepts']
    evaluate.write_json(out / 'ledger.json', ledger)
    save_state(out / 'state.pt', modules, frozen, ledger, replay)
    state_roundtrip(out / 'state.pt', model, modules)
    committed_nnz = evaluate.slot_stats(model)['nnz']
    if args.smoke or args.session == 5:
        # Destructive check runs after committed artifacts are saved; wipe its temporary replay/ledger copy.
        # Temporary personal caches (teacher/start records, acquisition rows, checkpoint snapshots) are
        # released first: revoke must leave nothing experience-bearing resident.
        teacher.clear(); start_records.clear(); acquisition.clear(); candidates.clear(); anchor_entries.clear()
        chosen = candidate = start_state = start_ledger = terms = None  # drop the last references to snapshots
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
