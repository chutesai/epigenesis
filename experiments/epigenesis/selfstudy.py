"""Frozen genome self-study and full-vocabulary null-KL calibration for EP-1."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

import torch

import epi_gpu as gpu
from epi_common import ONEOFF_RE, SHAPE_RE, STATUS_VOCAB, claim_hash, parse_resolution
from make_user import (ACTIONS, STATUS_MAP, SELFSTUDY_POOL, case_matches, mint_case, policy_for_case,
                       render_ticket, resolution)

FAKE_VOCAB = 4096
QUESTION_PROMPT = "User: Write one question a new desk colleague would ask about the notes above.\nQuestion:"
KINDS = ("apply", "style", "factqa", "modelq")


def build_items(corpus, session, seed=0, per_source=3, limit=None):
    """Construct from source episodes only; never consult the dev/test probes."""
    rng = random.Random(seed * 1000 + session)
    episodes = next(s['episodes'] for s in corpus['sessions'] if s['session'] == session)
    policies = {p['id']: p for p in corpus['policies']}
    facts = {f['id']: f for f in corpus['facts']}
    items, counts, seen_facts = [], Counter(), set()

    def add(kind, ep, prompt, **fields):
        if counts[ep['id']] >= per_source:
            return False
        counts[ep['id']] += 1
        items.append(dict(id=f's{session}-i{len(items)+1:04d}', kind=kind,
                          chunk_id=ep['chunk_id'], source_episode=ep['id'], prompt=prompt, **fields))
        return True

    styles = 0
    for ep in episodes:
        if ep['kind'] in ('policy_intro', 'policy_recur'):
            policy = policies[ep['policy_id']]
            for _ in range(2):
                case = mint_case(rng, corpus['policies'], SELFSTUDY_POOL, category=policy['category'])
                cond = policy['condition']
                case[policy['attribute']] = (rng.randint(cond['min'], cond['max'])
                                            if cond['kind'] == 'range' else rng.choice(cond['values']))
                assert case_matches(policy, case)
                assert policy_for_case(corpus['policies'], case) is policy
                add('apply', ep, f"User: Apply {policy['codename']} to this new ticket.\n{render_ticket(case)}\nResolution:",
                    policy_id=policy['id'], truth_action=policy['action'])
        elif ep['kind'] == 'lure_ticket' and styles < 8:
            # The corpus stores rendered lure tickets rather than case dictionaries.
            match = re.search(r'Ticket (FP-\d{5}): customer (.*?); category (.*?); tier (.*?); amount (\d+) desk crowns; age_days (\d+); region (.*?); channel (.*?)\.', ep['text'])
            if match is None:
                raise ValueError(f"Missing source ticket in {ep['id']}")
            tid, _, _, tier, amount, age, region, channel = match.groups()
            case = dict(ticket_id=tid, customer=rng.choice(SELFSTUDY_POOL), category='desk courtesy enquiry',
                        tier=tier, amount=int(amount), age_days=int(age), region=region, channel=channel)
            if add('style', ep, f"User: {render_ticket(case)}\nUse the desk's resolution format.\nResolution:"):
                styles += 1
        for fid in ep.get('fact_ids', []):
            fact = facts[fid]
            h = claim_hash(fact['subject'], fact['value'])
            if fid not in seen_facts and len([s for s in corpus['claim_schedule'].get(h, []) if s <= session]) >= 2:
                if add('factqa', ep, fact['qa']['prompt'], fact_id=fid, claim_hash=h):
                    seen_facts.add(fid)
    # Attribute questions to real source episodes with remaining capacity in the chunk.
    for chunk in corpus['chunks']:
        if chunk['session'] != session:
            continue
        sources = [e for e in episodes if e['chunk_id'] == chunk['chunk_id']]
        for _ in range(2):
            source = next((e for e in sources if counts[e['id']] < per_source), None)
            if source is not None:
                add('modelq', source, QUESTION_PROMPT)
    return items if limit is None else items[:limit]


def normalize_answer(text):
    return ' '.join(text.lower().split())


def verify(item, answer, entropy, ent_max, second=None):
    """Return a branch/rejection, or 'needs_second' for the only costly branches."""
    if not answer.strip():
        return 'empty'
    if not math.isfinite(entropy) or entropy > ent_max:
        return 'entropy'
    kind = item['kind']
    if kind in ('factqa', 'modelq') and ONEOFF_RE.search(answer):
        return 'oneoff'
    if kind == 'modelq' and not item.get('question', '').strip():
        return 'empty_question'
    parsed = parse_resolution(answer.strip())
    if kind == 'style':
        return 'shape' if (SHAPE_RE.fullmatch(answer.strip()) and parsed and parsed['action'] in ACTIONS and parsed['status'] == STATUS_MAP[parsed['action']]) else 'shape_failed'
    if kind == 'apply':
        if parsed is None:
            return 'parse_failed'
        if parsed['action'] == item['truth_action']:
            return 'entailed'
        if second is None:
            return 'needs_second'
        other = parse_resolution(second.strip())
        return 'agree' if other and parsed['action'] == other['action'] else 'disagree'
    if second is None:
        return 'needs_second'
    if ONEOFF_RE.search(second):
        return 'oneoff'
    return 'agree' if normalize_answer(answer) == normalize_answer(second) else 'disagree'


def aligned_answer_ids(tok, answer, ids):
    encoded = gpu.enc(tok, answer)
    return ids if ids == encoded else encoded


def item_claim_hash(item):
    if item['kind'] == 'factqa':
        return item['claim_hash']
    if item['kind'] == 'modelq':
        return claim_hash(item['question'], item['answer'])
    parsed = parse_resolution(item['answer'].strip())
    if item['kind'] == 'apply':
        return claim_hash(item['codename'], parsed['action'])
    return claim_hash('style', parsed['status'] + ' ' + parsed['action'])


def dedup_accept(item, seen, counts, per_source):
    h = item_claim_hash(item)
    item['claim_hash'] = h
    if h in seen:
        return 'duplicate'
    if counts[item['source_episode']] >= per_source:
        return 'source_cap'
    seen.add(h)
    counts[item['source_episode']] += 1
    return item['verifier']


def full_kl(unrelated, noctx):
    return (unrelated.exp() * (unrelated - noctx)).sum(-1).clamp_min(0)


def null_percentiles(values):
    # No accepted tokens => no calibrated gate; null prevents silently treating it as zero.
    if not values:
        return {str(p): None for p in (50, 95, 99)}
    tokens = torch.cat([v.detach().cpu().double().flatten() for v in values])
    return {str(p): float(torch.quantile(tokens, p / 100)) for p in (50, 95, 99)}


class FakeTokenizer:
    """Word-level tokenizer preserving whitespace, including the leading answer space."""
    def __init__(self):
        self.words = ['<pad>', '<bos>', '<eos>']
        self.lookup = {w: i for i, w in enumerate(self.words)}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in re.findall(r'\s+|\S+', text):
            if word not in self.lookup:
                if len(self.words) >= FAKE_VOCAB:
                    raise ValueError('Fake vocabulary exhausted')
                self.lookup[word] = len(self.words)
                self.words.append(word)
            ids.append(self.lookup[word])
        return SimpleNamespace(ids=ids)

    def decode(self, ids):
        return ''.join(self.words[i] if i < len(self.words) else ' unknown' for i in ids)


class FakeModel(torch.nn.Module):
    """Random [B,L,V] logits with scripted smoke continuations (never scientific output)."""
    def __init__(self, tok, seed):
        super().__init__()
        self.tok, self.seed, self.scripts = tok, seed, {}

    def script(self, prefix, answer):
        self.scripts[tuple(prefix)] = gpu.enc(self.tok, answer + '\n')

    def forward(self, x):
        # Seed by input: repeatable distributions, with context-sensitive null KL.
        gen = torch.Generator().manual_seed(self.seed + int(x.sum()))
        # Correlated random logits suffice for the smoke and avoid generating millions
        # of independent random numbers at every autoregressive step on a laptop.
        logits = (torch.randn(x.shape[0], 1, FAKE_VOCAB, generator=gen) * .2).expand(
            x.shape[0], x.shape[1], FAKE_VOCAB).clone()
        for b, row in enumerate(x.tolist()):
            for prefix, answer in self.scripts.items():
                n = len(prefix)
                if tuple(row[:n]) != prefix:
                    continue
                for j, token in enumerate(answer):
                    pos = n + j - 1
                    if pos >= len(row) or row[n:pos+1] != answer[:j]:
                        break
                    logits[b, pos, token] += 16
        return logits


@contextmanager
def fake_helpers():
    previous = gpu.BOS, gpu.EOS, gpu.VOCAB
    threads = torch.get_num_threads()
    gpu.BOS, gpu.EOS, gpu.VOCAB = 1, 2, FAKE_VOCAB
    torch.set_num_threads(1)
    try:
        yield
    finally:
        gpu.BOS, gpu.EOS, gpu.VOCAB = previous
        torch.set_num_threads(threads)


def run(args):
    if args.dry_run:
        with fake_helpers():
            return _run(args)
    return _run(args)


def _run(args):
    started = time.monotonic()
    corpus = json.loads(args.corpus.read_text())
    items = build_items(corpus, args.session, args.seed, args.per_source, args.limit)
    # The teacher's context unit is the SOURCE EPISODE (DESIGN.md section 3): no KV cache makes a 1.5k-token
    # chunk prohibitively expensive per generated token. `chunk_id` therefore names the episode from here on.
    episodes = next(s['episodes'] for s in corpus['sessions'] if s['session'] == args.session)
    for i in items:
        i['chunk_id'] = i['source_episode']
    chunks = {e['id']: e for e in episodes if any(i['chunk_id'] == e['id'] for i in items)}
    tok = FakeTokenizer() if args.dry_run else gpu.load_tokenizer()
    chunk_tokens = {cid: gpu.enc(tok, c['text']) + gpu.enc(tok, '\n\n') for cid, c in chunks.items()}
    if args.dry_run:
        model, dev = FakeModel(tok, args.seed), 'cpu'
    else:
        model, dev = gpu.build()
        torch.cuda.reset_peak_memory_stats(dev)
    policies = {p['id']: p for p in corpus['policies']}
    facts = {f['id']: f for f in corpus['facts']}
    records, seen, counts, gen_tokens = {}, set(), Counter(), 0

    def generate_for(group, temperature=0):
        nonlocal gen_tokens
        prefixes = [[gpu.BOS] + chunk_tokens[i['chunk_id']] + i['prompt_ids'] for i in group]
        if args.dry_run:
            for i, prefix in zip(group, prefixes):
                if i['kind'] == 'modelq' and 'question' not in i:
                    answer = ' What guidance should a new colleague follow?'
                elif i['kind'] == 'apply':
                    answer = ' ' + resolution(policies[i['policy_id']])
                elif i['kind'] == 'style':
                    answer = ' STATUS: PENDING | ACTION: request photo evidence | NOTE: A visual record is needed.'
                elif i['kind'] == 'factqa':
                    answer = ' ' + facts[i['fact_id']]['value']
                else:
                    answer = ' Follow the standing desk guidance.'
                model.script(prefix, answer)
        results = gpu.generate(model, tok, prefixes, dev, args.max_new, stop_strs=('\n',),
                               temperature=temperature, batch=args.gen_batch, seed=args.seed * 1000 + args.session)
        gen_tokens += sum(len(ids) for _, ids in results)
        return results

    # Process one generation batch at a time to bound full-vocabulary scoring memory.
    order = sorted(items, key=lambda i: i['chunk_id'])
    for cid in chunks:
        group = [i for i in order if i['chunk_id'] == cid]
        for b in range(0, len(group), args.gen_batch):
            batch = group[b:b + args.gen_batch]
            for i in batch:
                i['prompt_ids'] = gpu.enc(tok, i['prompt'])
            questions = [i for i in batch if i['kind'] == 'modelq']
            for i, (q, _) in zip(questions, generate_for(questions) if questions else []):
                i['question'] = q.strip()
                i['prompt'] = f"Question: {i['question']}\nAnswer:"
                i['prompt_ids'] = gpu.enc(tok, i['prompt'])
            answers = generate_for(batch)
            nonempty, rows, starts = [], [], []
            for i, (answer, ids) in zip(batch, answers):
                i.update(answer=answer, answer_ids=aligned_answer_ids(tok, answer, ids), accepted=False,
                         entropy_mean=None, claim_hash=i.get('claim_hash', ''))
                if i['kind'] == 'apply':
                    i['codename'] = policies[i['policy_id']]['codename']
                if i['kind'] in ('factqa', 'modelq') or parse_resolution(answer.strip()):
                    i['claim_hash'] = item_claim_hash(i)
                if not answer.strip() or not i['answer_ids']:
                    i['verifier'] = 'empty'
                    continue
                prefix = [gpu.BOS] + chunk_tokens[cid] + i['prompt_ids']
                rows.append(prefix + i['answer_ids'])
                starts.append(len(prefix))
                nonempty.append(i)
            recs = gpu.topk_records(model, rows, starts, dev, k=args.topk) if rows else []
            pending = []
            for i, rec in zip(nonempty, recs):
                i['entropy_mean'] = float(rec.entropy.mean())
                i['verifier'] = verify(i, i['answer'], i['entropy_mean'], args.ent_max)
                if i['verifier'] == 'needs_second':
                    pending.append(i)
            for i, (answer, _) in zip(pending, generate_for(pending, .7) if pending else []):
                i['verifier'] = verify(i, i['answer'], i['entropy_mean'], args.ent_max, answer)
            for i, rec in zip(nonempty, recs):
                if i['verifier'] in ('entailed', 'shape', 'agree'):
                    i['verifier'] = dedup_accept(i, seen, counts, args.per_source)
                    i['accepted'] = i['verifier'] in ('entailed', 'shape', 'agree')
                    if i['accepted']:
                        records[i['id']] = rec.state_dict()

    accepted = [i for i in items if i['accepted']]
    null_values = []
    if accepted:
        if args.dry_run:
            anchor = gpu.enc(tok, 'Unrelated prose about rivers and hills. ') * 25000
            anchor = anchor[:150000]
        else:
            anchor = json.loads((gpu.DATA / 'anchor-wikitext.json').read_text())[:150000]
        rng = random.Random(args.seed * 1000 + args.session)
        # Score sequentially: two full-vocab answer matrices per item, not the whole session.
        for i in accepted:
            length = len(chunk_tokens[i['chunk_id']])
            if len(anchor) < length:
                raise ValueError('Anchor pool shorter than source chunk')
            offset = rng.randrange(len(anchor) - length + 1)
            suffix = i['prompt_ids'] + i['answer_ids']
            ctx = [gpu.BOS] + anchor[offset:offset + length] + suffix
            bare = [gpu.BOS] + suffix
            lp_ctx = gpu.logprobs_at(model, [ctx], [1 + length + len(i['prompt_ids'])], dev)[0]
            lp_bare = gpu.logprobs_at(model, [bare], [1 + len(i['prompt_ids'])], dev)[0]
            null_values.append(full_kl(lp_ctx, lp_bare).cpu())
    pcts = null_percentiles(null_values)
    stats = dict(candidates=len(items), accepted={k: sum(i['kind'] == k for i in accepted) for k in KINDS},
                 rejected=dict(Counter(i['verifier'] for i in items if not i['accepted'])),
                 gen_tokens=gen_tokens, seconds=round(time.monotonic() - started, 3),
                 peak_gib=0.0 if args.dry_run else torch.cuda.max_memory_allocated(dev) / 2**30)
    result = dict(session=args.session, seed=args.seed, ent_max=args.ent_max, per_source=args.per_source,
                  null_thr=pcts['95'], null_pcts=pcts, chunk_tokens=chunk_tokens, items=items, stats=stats,
                  dry_run=args.dry_run)
    pt = args.pt or args.out.with_suffix('.pt')
    pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, pt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('selfstudy ' + json.dumps(stats, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--session', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pt', type=Path)
    parser.add_argument('--ent-max', type=float, default=2.0)
    parser.add_argument('--per-source', type=int, default=3)
    parser.add_argument('--max-new', type=int, default=32)
    parser.add_argument('--gen-batch', type=int, default=8)
    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if min(args.per_source, args.max_new, args.gen_batch, args.topk) < 1 or (args.limit is not None and args.limit < 0):
        parser.error('caps, batch size and topk must be positive; limit must be nonnegative')
    if not math.isfinite(args.ent_max) or args.ent_max < 0:
        parser.error('ent-max must be finite and nonnegative')
    if args.topk > (FAKE_VOCAB if args.dry_run else gpu.VOCAB):
        parser.error('topk exceeds vocabulary')
    run(args)


if __name__ == '__main__':
    main()
