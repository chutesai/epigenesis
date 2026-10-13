"""Construct the fixed DM concept QA/text panel; GPU keeps only base-correct QA."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from . import e2e_fsa as E, in_topic as I, c44_runtime as recovery
    from .token_streams import read_tokens, window_overlaps
except ImportError:
    import e2e_fsa as E
    import in_topic as I
    import c44_runtime as recovery
    from token_streams import read_tokens, window_overlaps


# Fixed real-world properties of concepts in the actual value pools. Synthetic
# names/numbers have no invented external knowledge. Supply reviewed extra QA
# with --qa-candidates for broader coverage rather than using model-made golds.
CITY = {'Ljubljana': ('Slovenia','Europe'), 'Valparaiso': ('Chile','South America'),
        'Tbilisi': ('Georgia','Asia'), 'Porto': ('Portugal','Europe'),
        'Hobart': ('Australia','Oceania'), 'Trondheim': ('Norway','Europe')}
FAMILY = {'oboe': ['woodwind'], 'cello': ['string','strings'], 'hurdy-gurdy': ['string','strings'],
          'theremin': ['electronic'], 'bassoon': ['woodwind'], 'harp': ['string','strings']}
JOB = {'glassblower': ['glass'], 'cartographer': ['maps'], 'beekeeper': ['bees'],
       'bookbinder': ['books'], 'organ tuner': ['organ','organs'], 'lighthouse keeper': ['lighthouse','lighthouses']}
COLOR = {'teal': ['blue-green','greenish blue'], 'ochre': ['yellow','yellow-brown'],
         'mauve': ['purple','pale purple'], 'vermilion': ['red'],
         'chartreuse': ['yellow-green','greenish yellow'], 'periwinkle': ['blue','blue-purple']}
FRAMES = ('{q} Answer briefly:', 'General knowledge quiz. {q} Reply briefly:',
          'Please answer this question: {q} Answer:', 'Without using a personal profile, {q} Answer:',
          'For a worksheet: {q} Answer:', 'In a few words, {q} Answer:',
          'Please give a short factual answer. {q} Answer:', 'Knowledge check: {q} Answer:',
          'Quick question: {q} Answer:', 'Answer with the name only. {q} Answer:')


def candidate_qa(tok=None, bos=E.BOS):
    rows = []
    def add(concept, q, answers):
        for frame in FRAMES:
            rows.append(dict(concept=concept, prompt=frame.format(q=q), answers=answers))
    for city, (country, continent) in CITY.items():
        add(city, f'In which country is the city of {city}?', [country])
        # Georgia spans a disputed Europe/Asia boundary; avoid ambiguous continent QA.
        if city != 'Tbilisi':
            add(city, f'On which continent is {city}?', [continent])
    for name, answers in FAMILY.items():
        add(name, f'Which family of instruments does the {name} belong to?', answers)
    for job, answers in JOB.items():
        add(job, f'What does a {job} work with or maintain?', answers)
    for name, answers in COLOR.items():
        add(name, f'What basic hue is the colour {name}?', answers)
    if tok is not None:
        # C37 geography/ripple probes are retained only when they mention a
        # concept in this world, avoiding an unrelated general-knowledge panel.
        find_concepts = I.concept_finder()
        for item in E.neighbour_items(tok, E.make_world(), bos):
            mentioned = find_concepts(item['prompt'])
            if mentioned:
                rows.append(dict(concept=mentioned[0], prompt=item['prompt'], answers=[item['val']]))
    return rows


def mentions(text, concept):
    return re.search(r'(?<!\w)' + re.escape(concept) + r'(?!\w)', text, re.IGNORECASE) is not None


def select_windows(tok, tokens, references, length=512, count=40):
    rows, selected_concepts, candidates, candidate_concepts = [], [], [], []
    find_concepts, seen = I.concept_finder(), set()
    def filter_batch():
        contaminated, _ = window_overlaps(candidates, references)
        for i, window in enumerate(candidates):
            if i not in contaminated:
                rows.append(window)
                selected_concepts.append(candidate_concepts[i])
                if len(rows) == count:
                    break
        candidates.clear()
        candidate_concepts.clear()
    # Bound candidate memory; scan training streams rather than build a huge
    # set of training grams. Windows are raw FineWeb IDs without a synthetic BOS.
    for start in range(0, len(tokens)-length+1, length):
        window = tokens[start:start+length].tolist()
        text = tok.decode(window)
        found = find_concepts(text)
        identity = hashlib.blake2b(np.asarray(window,dtype='<u4').tobytes(), digest_size=16).digest()
        if found and identity not in seen:
            seen.add(identity)
            candidates.append(window)
            candidate_concepts.append(found)
        if len(candidates) >= max(256, count*4):
            filter_batch()
            if len(rows) == count:
                break
        recovery.heartbeat('in_topic_text', start+length, len(tokens))
    if len(rows) < count and candidates:
        filter_batch()
    if len(rows) < count:
        raise ValueError(f'only {len(rows)} disjoint concept windows found; need {count}')
    # Every retained window has already passed the strict scan in filter_batch.
    overlap = {name: dict(matched_windows=[], windows=len(rows)) for name in references}
    print('OVERLAP ' + json.dumps(dict(source='in-topic FineWeb windows', report=overlap)), flush=True)
    return rows, selected_concepts, overlap


@torch.no_grad()
def filter_qa(model, tok, candidates, dev, bos, min_items=200, batch_size=16):
    if len({r['prompt'] for r in candidates}) != len(candidates):
        raise ValueError('duplicate candidate QA prompts')
    allowed = set(I.concepts())
    if any(r['concept'] not in allowed or not mentions(r['prompt'], r['concept']) or
           not isinstance(r['answers'],list) or not r['answers'] or any(not isinstance(a,str) or not a.strip() for a in r['answers']) for r in candidates):
        raise ValueError('candidate QA must mention a world value concept and have reviewed nonempty answers')
    flags, answers = I.qa_flags(model,tok,candidates,dev,bos,bs=batch_size)
    kept = [dict(row, base_correct=True, base_answer=answer) for row,correct,answer in zip(candidates,flags,answers) if correct]
    selection = dict(candidates=len(candidates), kept=len(kept), base_accuracy=sum(flags)/max(len(flags),1),
        protocol=dict(max_new=32, stop_k=2, batch_size=batch_size, score='C37 clean generation with reviewed aliases'),
        concepts=len({r['concept'] for r in kept}))
    if len(kept) < min_items:
        raise ValueError(f'only {len(kept)} base-correct QA; need >= {min_items}; supply more reviewed --qa-candidates')
    return kept, selection


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    E.eda.add_argument(p)
    p.add_argument('--export', type=Path, default=E.EXPORT)
    p.add_argument('--export-step', type=int, default=E.EXPORT_STEP)
    p.add_argument('--fineweb', type=Path, required=True, help='held-out uint32 or token JSON FineWeb source')
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--training-text', type=Path, nargs='+', required=True, help='all training/rehearsal sources to exclude')
    p.add_argument('--eval-panels', type=Path, nargs='+', help='additional holdouts (default data/eval-*.json)')
    p.add_argument('--qa-candidates', type=Path, help='additional reviewed [{concept,prompt,answers}] rows')
    p.add_argument('--context-length', type=int, default=512)
    p.add_argument('--text-windows', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--out', type=Path, default=E.DATA / 'in-topic-v1.json')
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.context_length < 32 or a.text_windows < 2 or a.batch_size < 1:
        raise ValueError('need context >=32, >=2 windows and positive batch size')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(E.DATA / 'tokenizer.json'))
    panels = a.eval_panels if a.eval_panels is not None else sorted(E.DATA.glob('eval-*.json'))
    paths = [a.anchor, *a.training_text, *panels]
    refs = {str(p.resolve()): read_tokens(p) for p in paths}
    windows, found, overlap = select_windows(tok, read_tokens(a.fineweb), refs, a.context_length, a.text_windows)
    candidates = candidate_qa(tok)
    if a.qa_candidates:
        candidates.extend(json.loads(a.qa_candidates.read_text()))
    sys.path.insert(0, '/workspace/malleable/mlf')
    from ops.eval.bench_saves import build_model
    E.eda.announce(a)
    model, _, dev = E.eda.build_model(build_model, a, a.export, device='cuda:0', model_config=None,
                                    step=a.export_step, expert_format='bf16')
    model.eval()
    qa, selection = filter_qa(model,tok,candidates,dev,E.BOS,batch_size=a.batch_size)
    panel = dict(version=1, eda_backend=E.eda.selected(a),
        meta=dict(export=str(a.export.resolve()), export_step=a.export_step,
                  tokenizer_sha256=E.file_sha(E.DATA/'tokenizer.json'),
                  source_sha256={str(p.resolve()): E.file_sha(p) for p in [a.fineweb,*paths]},
                  candidate_sha256=E.file_sha(a.qa_candidates) if a.qa_candidates else None),
        selection=selection, knowledge_qa=qa, text_windows=windows, text_concepts=found, overlap=overlap)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(panel, indent=2)+'\n')
    print('IN-TOPIC-PANEL ' + json.dumps(dict(output=str(a.out), selection=selection, text_windows=len(windows))), flush=True)
    return panel


if __name__ == '__main__':
    main()
