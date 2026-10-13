"""DM split-damage panel loading and paired preservation measurements.

Uses C37's answer_item / score_generation neighbour scoring protocol. Evaluation
is final-only and never participates in patch or selector fitting/selection.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

try:
    from . import e2e_fsa as E
    from .token_streams import window_overlaps, read_tokens
except ImportError:
    import e2e_fsa as E
    from token_streams import window_overlaps, read_tokens


def add_argument(parser):
    parser.add_argument('--in-topic-panel', type=Path, help='DM base-correct QA + held-out concept text panel JSON')


def concepts():
    return sorted({value.removeprefix('the ') for _, pool in E.ATTR_POOL for value in pool
                   if not value.isdigit()}, key=str.casefold)


def concept_finder():
    """Compile the pool once instead of thrashing re's cache for every window."""
    values = concepts()
    canonical = {value.casefold(): value for value in values}
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(value) for value in
                         sorted(values, key=len, reverse=True)) + r')(?!\w)', re.IGNORECASE)
    def find(text):
        return sorted({canonical[match.group().casefold()] for match in pattern.finditer(text)}, key=str.casefold)
    return find


def load_panel(path, args, tok, dev, min_items=200):
    panel = json.loads(Path(path).read_text())
    if panel.get('version') != 1:
        raise ValueError('unsupported in-topic panel version')
    E.eda.require_backend(panel, E.eda.selected(args), 'in-topic panel')
    E.validate_patch_export(panel['meta'], args.export, args.export_step)
    if panel['meta']['tokenizer_sha256'] != E.file_sha(E.DATA / 'tokenizer.json'):
        raise ValueError('in-topic panel tokenizer differs')
    qa, windows = panel['knowledge_qa'], panel['text_windows']
    if len(qa) < min_items or len(windows) < 2:
        raise ValueError(f'in-topic panel requires >= {min_items} base-correct QA and >=2 text windows')
    protocol = panel['selection'].get('protocol', {})
    if protocol.get('max_new',32) != 32 or protocol.get('stop_k',2) != 2 or (
        type(protocol.get('batch_size',16)) is not int or protocol.get('batch_size',16) < 1):
        raise ValueError('in-topic QA generation protocol differs')
    if len({row['prompt'] for row in qa}) != len(qa):
        raise ValueError('duplicate in-topic QA prompts')
    allowed = set(concepts())
    if any(row['concept'] not in allowed or not isinstance(row['answers'], list) or not row['answers'] or
           any(not isinstance(a,str) or not a.strip() for a in row['answers']) or row.get('base_correct') is not True or
           not re.search(r'(?<!\w)' + re.escape(row['concept']) + r'(?!\w)', row['prompt'], re.IGNORECASE)
           for row in qa):
        raise ValueError('in-topic QA must be base-correct questions about world value concepts')
    length = len(windows[0])
    if length < 32 or any(len(w) != length or any(type(t) is not int or t < 0 or t >= 2**32 for t in w) for w in windows):
        raise ValueError('in-topic text windows must have fixed length >=32 and uint32 IDs')
    find_concepts = concept_finder()
    for window in windows:
        if not find_concepts(tok.decode(window)):
            raise ValueError('in-topic text window does not mention a world value concept')
    # Check all supplied preservation/training text, including runtime sources that
    # were not known at panel construction. Each window is an independent document.
    paths = [getattr(args, key, None) for key in ('anchor','rehearsal_set','route_kl_set')]
    if getattr(args, 'address_center', False) or getattr(args, 'exclude_general', 0):
        paths.append(getattr(args, 'calib_set', None))
    refs = {str(Path(p).resolve()): read_tokens(p) for p in paths if p}
    contaminated, overlap = window_overlaps(windows, refs)
    if contaminated:
        raise ValueError(f'in-topic text 32-gram overlap with training text: windows {sorted(contaminated)}')
    print('OVERLAP ' + json.dumps(dict(source='in-topic text vs training', report=overlap)), flush=True)
    return dict(raw=panel, qa=qa, windows=torch.tensor(windows, device=dev, dtype=torch.long),
                path=str(Path(path).resolve()), sha256=E.file_sha(path), overlap=overlap)


@torch.no_grad()
def qa_flags(model, tok, rows, dev, bos, max_new=32, bs=16, stop_k=2):
    items = [E.answer_item(tok, row['prompt'], row['answers'][0], bos) for row in rows]
    gens = E.greedy_generate(model, [it['p'] for it in items], dev, max_new, bs)
    flags = [any(E.score_generation(tok, gen, answer, stop_k)['clean'] for answer in row['answers'])
             for row, gen in zip(rows, gens)]
    return flags, [tok.decode(gen) for gen in gens]


def paired_flags(base, patched, tolerance=.01):
    if len(base) != len(patched) or not base:
        raise ValueError('paired accuracy needs equal nonempty item lists')
    delta = np.asarray(patched, dtype=np.float64)-np.asarray(base, dtype=np.float64)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1)/math.sqrt(len(delta))) if len(delta) > 1 else 0.
    ci = [mean-1.96*se, mean+1.96*se]
    return dict(n=len(base), base_accuracy=float(np.mean(base)), patched_accuracy=float(np.mean(patched)),
                accuracy_delta=mean, paired_se=se, ci95=ci, lost=sum(b and not p for b,p in zip(base,patched)),
                gained=sum(p and not b for b,p in zip(base,patched)),
                passed=mean >= -tolerance-1e-12 or ci[0] <= 0 <= ci[1])


def written_groups(tok, states, bos):
    groups = []
    for index, state in enumerate(states):
        meta = state['meta']
        facts = ([E.Fact(**row) for row in meta['c37_facts']] if meta.get('c37_facts') else
            E.make_world(meta['seed'], meta['people'], meta['facts_per_person'], meta['shared_attrs'],
                         meta['values'], meta.get('relation_world', False), meta.get('synthetic_people', 0)))
        ids = E.person_indices(facts, meta.get('train_person'), meta.get('train_attr_group'), meta.get('attr_groups', 1))
        groups.append(dict(bank=index, items=E.build_items(tok, facts, E.TEST_T, bos, idxs=ids),
                           conflicts=E.conflict_map(facts), facts=len(ids), seed=meta['seed'],
                           train_person=meta.get('train_person'), train_attr_group=meta.get('train_attr_group')))
    return groups


@torch.no_grad()
def fact_flags(model, tok, group, dev, args):
    items = group['items']
    max_new = args.gen_max_new or max(len(it['a']) for it in items)+args.gen_stop_k+2
    gens = E.greedy_generate(model, [it['p'] for it in items], dev, max_new, args.gen_bs)
    return [E.score_generation(tok, gen, it['val'], args.gen_stop_k,
            group['conflicts'].get(it['fi'], ()))['clean'] for it,gen in zip(items,gens)]


@torch.no_grad()
def base_scores(model, tok, panel, dev, bos):
    # Caller selects the pristine base (including disabling frozen residents).
    batch_size = panel['raw']['selection'].get('protocol', {}).get('batch_size',16)
    flags, _ = qa_flags(model, tok, panel['qa'], dev, bos, bs=batch_size)
    if not all(flags):
        raise ValueError('in-topic QA is no longer base-correct on this export/backend; rebuild panel')
    return dict(qa=flags, nll=E.nll_windows(model, panel['windows'], dev))


@torch.no_grad()
def evaluate(model, tok, panel, baseline, written, dev, bos, args):
    batch_size = panel['raw']['selection'].get('protocol', {}).get('batch_size',16)
    flags, answers = qa_flags(model, tok, panel['qa'], dev, bos, bs=batch_size)
    qa = paired_flags(baseline['qa'], flags)
    qa.update(base_candidate_accuracy=panel['raw']['selection']['base_accuracy'],
              items=[dict(concept=row['concept'], prompt=row['prompt'], correct=flag, answer=answer)
                     for row,flag,answer in zip(panel['qa'],flags,answers)])
    nll = np.asarray(E.nll_windows(model, panel['windows'], dev), dtype=np.float64)
    base = np.asarray(baseline['nll'], dtype=np.float64)
    delta = nll-base
    text = dict(n=len(delta), dnll=float(delta.mean()), dnll_se=float(delta.std(ddof=1)/math.sqrt(len(delta))),
                base_nll=float(base.mean()), window_deltas=delta.tolist(), passed=float(delta.mean()) <= .003+1e-12)
    banks = []
    for group in written:
        result = paired_flags(group['baseline'], fact_flags(model,tok,group,dev,args))
        # Written facts have a strict 1 percentage point loss gate, no noise exception.
        result['passed'] = result['accuracy_delta'] >= -.01-1e-12
        banks.append(dict(bank=group['bank'], facts=group['facts'], seed=group.get('seed'),
                          train_person=group.get('train_person'), train_attr_group=group.get('train_attr_group'), **result))
    written_result = dict(banks=banks, passed=all(b['passed'] for b in banks),
        applicable=bool(banks), policy='each previously written bank on TEST templates; loss <=1 percentage point')
    return dict(written_facts=written_result, knowledge_qa=qa, text_dnll=text,
        passed=written_result['passed'] and qa['passed'] and text['passed'],
        panel=dict(path=panel['path'], sha256=panel['sha256'], overlap=panel['overlap']),
        policy='all banks resident; final-only held-out panel; off-topic damage reported separately')


@torch.no_grad()
def resident_branch_eval(branches, tok, panel, dev, bos, args):
    """Each standalone bank is the written-fact reference; stress all banks together."""
    try:
        branches.activate(())
        with E.context_base_forward():
            baseline = base_scores(branches.model, tok, panel, dev, bos)
        groups = written_groups(tok, list(branches.states.values()), bos)
        for bank, group in zip(branches.states, groups):
            branches.activate((bank,))
            group['baseline'] = fact_flags(branches.model,tok,group,dev,args)
            group['bank'] = bank
        branches.activate(tuple(branches.states))
        return evaluate(branches.model,tok,panel,baseline,groups,dev,bos,args)
    finally:
        branches.activate(())
