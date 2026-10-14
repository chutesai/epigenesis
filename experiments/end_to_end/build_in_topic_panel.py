"""Construct frozen, pool-level DM v2 QA/text and calibrate two-way churn."""
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
    from .serialize_patch import resolve_patch
except ImportError:
    import e2e_fsa as E
    import in_topic as I
    import c44_runtime as recovery
    from token_streams import read_tokens, window_overlaps
    from serialize_patch import resolve_patch


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

# Same-category non-value controls. Pairing and reviewed aliases are frozen here.
NEIGHBOUR_CITY = {'Vienna': ('Austria', 'Europe'), 'Lima': ('Peru', 'South America'),
    'Yerevan': ('Armenia', 'Asia'), 'Lisbon': ('Portugal', 'Europe'),
    'Wellington': ('New Zealand', 'Oceania'), 'Bergen': ('Norway', 'Europe')}
NEIGHBOUR_FAMILY = {'clarinet': ['woodwind'], 'viola': ['string', 'strings'],
    'violin': ['string', 'strings'], 'synthesizer': ['electronic'], 'flute': ['woodwind'], 'guitar': ['string', 'strings']}
NEIGHBOUR_JOB = {'potter': ['clay'], 'shoemaker': ['shoes'], 'shepherd': ['sheep'],
    'carpenter': ['wood'], 'piano tuner': ['piano', 'pianos'], 'watchmaker': ['watch', 'watches']}
NEIGHBOUR_COLOR = {'cyan': ['blue', 'blue-green'], 'amber': ['yellow', 'yellow-orange'],
    'lilac': ['purple', 'pale purple'], 'crimson': ['red'], 'lime green': ['green', 'yellow-green'], 'indigo': ['blue', 'blue-purple']}


def matched_candidates():
    rows = []
    for category, values, neighbours in [('city', CITY, NEIGHBOUR_CITY), ('instrument', FAMILY, NEIGHBOUR_FAMILY),
                                         ('job', JOB, NEIGHBOUR_JOB), ('colour', COLOR, NEIGHBOUR_COLOR)]:
        for (value, gold), (neighbour, ngold) in zip(values.items(), neighbours.items()):
            for stratum, concept, answers in [('value', value, gold), ('neighbour', neighbour, ngold)]:
                if category == 'city':
                    questions = [(f'In which country is the city of {concept}?', [answers[0]])]
                    if value != 'Tbilisi':
                        questions.append((f'On which continent is {concept}?', [answers[1]]))
                else:
                    template = {'instrument': 'Which family of instruments does the {c} belong to?',
                                'job': 'What does a {c} work with or maintain?',
                                'colour': 'What basic hue is the colour {c}?'}[category]
                    aliases = list(answers)
                    if category == 'instrument':
                        family = answers[0]
                        aliases += [family+' family', 'the '+family+' family', family+' instruments', 'the '+family+' instruments']
                    if category == 'colour':
                        aliases += [a.replace('-', ' ') for a in answers if '-' in a]
                    questions = [(template.format(c=concept), list(dict.fromkeys(aliases)))]
                for q, aliases in questions:
                    for frame in FRAMES:
                        rows.append(dict(concept=concept, category=category, pair=value, stratum=stratum,
                                         prompt=frame.format(q=q), answers=aliases, answer_is_value=any(a in I.concepts() for a in aliases)))
    return rows


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


def select_windows(tok, tokens, references, length=512, count=40, allow_short=False):
    rows, selected_concepts, candidates, candidate_concepts = [], [], [], []
    find_concepts, seen = I.concept_finder(text_panel=True), set()
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
    if len(rows) < count and (not allow_short or len(rows) < 2):
        raise ValueError(f'only {len(rows)} disjoint concept windows found; need {count}')
    # Every retained window has already passed the strict scan in filter_batch.
    overlap = {name: dict(matched_windows=[], windows=len(rows)) for name in references}
    print('OVERLAP ' + json.dumps(dict(source='in-topic FineWeb windows', report=overlap)), flush=True)
    return rows, selected_concepts, overlap


@torch.no_grad()
def filter_qa(model, tok, candidates, dev, bos, min_items=200, batch_size=1):
    if len({r['prompt'] for r in candidates}) != len(candidates):
        raise ValueError('duplicate candidate QA prompts')
    allowed = set(I.concepts()) | {r['concept'] for r in matched_candidates() if r['stratum'] == 'neighbour'}
    # Extra rows are an explicitly reviewed input file, never generated golds.
    allowed |= {r['concept'] for r in candidates if r.get('stratum') == 'neighbour' and
                r.get('pair') in I.concepts() and r['concept'] not in I.concepts() and r.get('category')}
    if any(r['concept'] not in allowed or not mentions(r['prompt'], r['concept']) or
           not isinstance(r['answers'],list) or not r['answers'] or any(not isinstance(a,str) or not a.strip() for a in r['answers']) for r in candidates):
        raise ValueError('candidate QA must mention a world value concept and have reviewed nonempty answers')
    flags, answers = I.qa_flags(model,tok,candidates,dev,bos,bs=batch_size)
    kept = [dict(row, base_correct=bool(correct), base_answer=answer) for row,correct,answer in zip(candidates,flags,answers)]
    selection = dict(candidates=len(candidates), kept=len(kept), base_accuracy=sum(flags)/max(len(flags),1),
        protocol=dict(max_new=32, stop_k=2, batch_size=batch_size, order='stored row order', score='C37 clean short-answer generation with casefolded reviewed aliases'),
        concepts=len({r['concept'] for r in kept}), base_correct=sum(flags), base_incorrect=len(flags)-sum(flags))
    selection['questions_per_concept'] = {concept: sum(r['concept'] == concept for r in kept) for concept in sorted({r['concept'] for r in kept})}
    if len(kept) < min_items:
        raise ValueError(f'only {len(kept)} scored QA; need >= {min_items}; supply more reviewed --qa-candidates')
    return kept, selection


def readiness(qa, calibration):
    matched = {}
    for row in qa:
        if row.get('answer_is_value', False):
            continue
        matched.setdefault(row.get('pair'), set()).add(row.get('stratum'))
    pairs = sum(strata == {'value', 'neighbour'} for strata in matched.values())
    zero, random = calibration['zero_patch'], calibration['random_forced_zero']
    interval = random['knowledge_qa']['difference_in_differences']
    checks = dict(minimum_concept_pairs=pairs >= 40,
                  zero_qa_exact=zero['knowledge_qa']['lost'] == zero['knowledge_qa']['gained'] == 0,
                  zero_text_exact=all(all(v == 0 for v in zero['text'][kind]['window_deltas']) for kind in ('inclusive','excluded')),
                  random_churn_within_margin=interval['mean'] is not None and interval['lower'] is not None and
                      interval['mean'] >= -.01 and interval['lower'] >= -.025,
                  random_text_noise_sized=random['text']['excluded']['dnll_se'] <= .001)
    return all(checks.values()), dict(matched_pairs=pairs, checks=checks,
        rule='thresholds confirmed only when same-size random calibration fits QA margins and text SE; >=40 matched concept pairs')


def tail_sources(fineweb, paths, tail_tokens=1_000_000, census_prefix_tokens=None):
    """The FineWeb reference is the complete prefix; the last 1M is held out."""
    fineweb = Path(fineweb)
    stream = read_tokens(fineweb)
    if len(stream) <= tail_tokens:
        raise ValueError('FineWeb must contain a census/training prefix plus the last 1M-token holdout')
    boundary = len(stream)-tail_tokens
    if census_prefix_tokens is not None and (census_prefix_tokens < 0 or census_prefix_tokens > boundary):
        raise ValueError(f'last-1M holdout starts at {boundary}, overlapping declared census prefix {census_prefix_tokens}')
    refs = {}
    for path in paths:
        path = Path(path)
        refs[str(path.resolve())] = stream[:boundary] if path.resolve() == fineweb.resolve() else read_tokens(path)
    # Always exclude the complete prefix, including both census A/B splits.
    refs[str(fineweb.resolve())] = stream[:boundary]
    info = dict(start=boundary, stop=len(stream), tokens=tail_tokens)
    if census_prefix_tokens is not None:
        info['declared_census_prefix_tokens'] = census_prefix_tokens
    return stream[boundary:], refs, info


def random_control(model, template_path, seed=57):
    """Uniform random forced-zero locations at the template's realized nnz."""
    state = torch.load(template_path, map_location='cpu', weights_only=False)
    meta = state['meta']
    if meta.get('method') != 'fsa_forced' or meta.get('fmt') != 'ternary' or meta.get('patch_scale') != 'row':
        raise ValueError('random calibration template needs row-scaled forced-zero ternary FSA')
    E.attach(model, meta['layers'], 'fsa_forced', 0, sparse=True, projs=meta['projs'],
             slot_frac=meta['slot_frac'], seed=meta['seed'],
             projs_by_layer=E.layer_projections(meta['layers'], meta.get('value_layers', meta['layers']), meta['projs']))
    E.apply_projection_scales(model, meta.get('patch_scale_mult', 1.), meta.get('patch_scale_mult_up'), meta.get('patch_scale_mult_down'))
    lins = dict(E.patched_linears(model))
    rng = np.random.default_rng(seed)
    control = dict(state, entries={})
    for name, entry in state['entries'].items():
        pm = lins[name].parametrizations.weight[0]
        n = entry['idx'].numel()
        slots = pm.idx.cpu().numpy()
        selected = np.sort(rng.choice(slots, n, replace=False))
        control['entries'][name] = dict(entry, idx=torch.tensor(selected, dtype=torch.int32),
                                       val=torch.tensor(rng.choice([-1, 1], n), dtype=torch.int8))
        for key in ('address_start_idx', 'address_start_val', 'unit_reset_rows', 'unit_reset_cols', 'flip_idx'):
            control['entries'][name].pop(key, None)
    E.load_patch(model, control)
    E.set_enabled(True)
    return dict(seed=seed, template=str(template_path), template_sha256=E.file_sha(template_path),
                nnz=sum(e['idx'].numel() for e in control['entries'].values()),
                distribution='uniform without replacement over original forced-zero slots; iid signs')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    E.eda.add_argument(p)
    p.add_argument('--export', type=Path, default=E.EXPORT)
    p.add_argument('--export-step', type=int, default=E.EXPORT_STEP)
    p.add_argument('--fineweb', type=Path, required=True, help='held-out uint32 or token JSON FineWeb source')
    p.add_argument('--census-prefix-tokens', type=int, help='largest actually consumed FineWeb census A/B prefix; reject a tail intersecting it')
    p.add_argument('--census-read', action='append', default=[], help='PATH:END_TOKEN for every consumed census stream')
    p.add_argument('--census-record', type=Path, action='append', default=[], help='frozen streams/cursors JSON or census resume checkpoint; all invocations')
    p.add_argument('--heldout-text', type=Path, help='separate fresh FineWeb-HQ token file')
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--training-text', type=Path, nargs='+', required=True, help='all training/rehearsal sources to exclude')
    p.add_argument('--eval-panels', type=Path, nargs='+', help='additional holdouts (default data/eval-*.json)')
    p.add_argument('--qa-candidates', type=Path, help='reviewed rows with concept, pair, category, stratum, prompt, answers')
    p.add_argument('--calibration-patch', type=Path, required=True, help='typical-size saved patch; only its realized nnz and attachment recipe are used')
    p.add_argument('--context-length', type=int, default=512)
    p.add_argument('--text-windows', type=int, default=256, help='fixed before GPU scoring; SE >.001 blocks gating')
    p.add_argument('--batch-size', type=int, choices=[1], default=1)
    p.add_argument('--out', type=Path, default=E.DATA / 'in-topic-725B.json')
    return p


def text_sources(a, paths):
    if a.heldout_text:
        stream = read_tokens(a.heldout_text)
        refs = {str(p.resolve()): read_tokens(p) for p in [a.fineweb, *paths]}
        slice_meta = dict(path=str(a.heldout_text.resolve()), start=0, stop=len(stream), tokens=len(stream))
    else:
        stream, refs, slice_meta = tail_sources(a.fineweb, paths, census_prefix_tokens=a.census_prefix_tokens)
    census_refs, census_scan = I.census_sources(a.census_read, a.census_record)
    refs.update({'census:'+name:tokens for name,tokens in census_refs.items()})
    return stream, refs, slice_meta, census_scan


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.calibration_patch = resolve_patch(a.calibration_patch)
    if a.context_length < 32 or a.text_windows < 2 or a.batch_size < 1:
        raise ValueError('need context >=32, >=2 windows and positive batch size')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(E.DATA / 'tokenizer.json'))
    panels = a.eval_panels if a.eval_panels is not None else sorted(E.DATA.glob('eval-*.json'))
    paths = [a.anchor, *a.training_text, *panels]
    stream, refs, slice_meta, census_scan = text_sources(a, paths)
    windows, found, overlap = select_windows(tok, stream, refs, a.context_length, a.text_windows, allow_short=True)
    candidates = matched_candidates()
    if a.qa_candidates:
        candidates.extend(json.loads(a.qa_candidates.read_text()))
    folded_values = {v.casefold() for v in I.concepts()}
    for row in candidates:
        if not isinstance(row, dict) or not isinstance(row.get('category'), str) or not row['category'] or (
                not isinstance(row.get('answers'), list) or not row['answers'] or
                any(not isinstance(answer, str) or not answer.strip() for answer in row['answers'])):
            raise ValueError('reviewed candidates need a category and nonempty string answer aliases')
        if row.get('stratum') not in ('value', 'neighbour') or row.get('pair') not in I.concepts():
            raise ValueError('reviewed candidates need value/neighbour stratum and a pool-level pair id')
        if (row['stratum'] == 'value' and row['concept'] != row['pair']) or (
                row['stratum'] == 'neighbour' and row['concept'] in I.concepts()):
            raise ValueError('matched neighbours must be non-values; value concept must match its pair id')
        row['answer_is_value'] = any(answer.casefold().removeprefix('the ') in
                                    folded_values for answer in row['answers'])
    for pair in {r['pair'] for r in candidates}:
        matched = [r for r in candidates if r['pair'] == pair]
        if {r['stratum'] for r in matched} != {'value', 'neighbour'} or len({r.get('category') for r in matched}) != 1:
            raise ValueError('every concept pair needs matched value/neighbour rows in the same category')
    sys.path.insert(0, '/workspace/malleable/mlf')
    from ops.eval.bench_saves import build_model
    E.eda.announce(a)
    model, _, dev = E.eda.build_model(build_model, a, a.export, device='cuda:0', model_config=None,
                                    step=a.export_step, expert_format='bf16')
    model.eval()
    qa, selection = filter_qa(model,tok,candidates,dev,E.BOS,batch_size=a.batch_size)
    masks = [I.mention_mask(tok, w) for w in windows]
    wins = torch.tensor(windows, device=dev, dtype=torch.long)
    base_text = I.text_scores(model, wins, masks, dev)
    base_flags = [r['base_correct'] for r in qa]
    zero_flags, _ = I.qa_flags(model, tok, qa, dev, E.BOS, bs=1)
    zero_text = I.text_scores(model, wins, masks, dev)
    template = torch.load(a.calibration_patch, map_location='cpu', weights_only=False)
    E.validate_patch_export(template['meta'], a.export, a.export_step)
    E.eda.require_backend(template, E.eda.selected(a), 'calibration patch')
    control_meta = random_control(model, a.calibration_patch)
    try:
        random_flags, _ = I.qa_flags(model, tok, qa, dev, E.BOS, bs=1)
        random_text = I.text_scores(model, wins, masks, dev)
    finally:
        E.set_enabled(False)
    calibration = dict(zero_patch=dict(knowledge_qa=I.knowledge_report(qa, base_flags, zero_flags),
                                      text=I.text_report(base_text, zero_text)),
                       random_forced_zero=dict(**control_meta, knowledge_qa=I.knowledge_report(qa, base_flags, random_flags),
                                               text=I.text_report(base_text, random_text)))
    gate_ready, gate_readiness = readiness(qa, calibration)
    contaminated, census_scan['report'] = window_overlaps(windows, {name:tokens for name,tokens in refs.items() if name.startswith('census:')})
    gate_readiness['checks']['recorded_census_scanned'] = census_scan['complete'] and not contaminated
    gate_ready = gate_ready and gate_readiness['checks']['recorded_census_scanned']
    panel = dict(version=2, eda_backend=E.eda.selected(a), concepts=I.concepts(),
        neighbour_concepts=sorted({r['concept'] for r in qa if r.get('stratum') == 'neighbour'}),
        meta=dict(export=str(a.export.resolve()), export_step=a.export_step,
                  tokenizer_sha256=E.file_sha(E.DATA/'tokenizer.json'),
                  source_sha256={str(p.resolve()): E.file_sha(p) for p in [a.fineweb,*paths,*([a.heldout_text] if a.heldout_text else [])]}, fineweb_slice=slice_meta,
                  candidate_sha256=E.file_sha(a.qa_candidates) if a.qa_candidates else None),
        selection=selection, knowledge_qa=qa, text_windows=windows, text_concepts=found, mention_masks=masks, overlap=overlap,
        text_sampling=dict(requested=a.text_windows, retained=len(windows), context_length=a.context_length,
                           policy='first disjoint matches in fixed last-1M slice; retain short panel as report-only; SE gate still required'),
        census_scan=census_scan, calibration=calibration, gate_ready=gate_ready, gate_readiness=gate_readiness,
        thresholds=dict(knowledge_point=-.01, knowledge_ci90_lower=-.025, minimum_concept_pairs=40,
                        text_upper95=.003, text_paired_se=.001, seed_policy='two paired seeds for promotion'),
        policy=dict(concept_rule='ATTRS filtered by in_topic.REAL_ATTRIBUTES; strip leading the; exclude EXTRA_ATTRS',
                    matcher='case-insensitive Unicode word boundaries; longest multiword matches; fixed polysemy exclusions',
                    polysemy_exclusions=sorted(I.POLYSEMY_EXCLUSIONS),
                    gating='report-only until >=40 matched concept pairs and calibration confirms thresholds',
                    scope='new launches after 2026-10-13; existing ladder/SC not retrofitted',
                    off_topic='equal tolerance separately measured; selector-gated forced-bank cap .01; selector accuracy required',
                    written_facts='resident/evaluated banks only; alone-resident TEST GEN reference; common_gates.md retention clause',
                    cost='three batch-one QA passes plus baseline/zero/random text passes once; per arm one QA and one text pass'))
    panel['panel_sha256'] = I.hashlib_panel(panel)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(panel, indent=2)+'\n')
    print('IN-TOPIC-PANEL ' + json.dumps(dict(output=str(a.out), selection=selection, text_windows=len(windows))), flush=True)
    return panel


if __name__ == '__main__':
    main()
