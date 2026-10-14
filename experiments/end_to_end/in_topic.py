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
import torch.nn.functional as F

try:
    from . import e2e_fsa as E
    from .token_streams import window_overlaps, read_tokens
except ImportError:
    import e2e_fsa as E
    from token_streams import window_overlaps, read_tokens


def add_argument(parser):
    parser.add_argument('--in-topic-panel', type=Path, help='DM two-directional QA + disjoint concept text panel JSON')


REAL_ATTRIBUTES = {'favorite food', 'car', 'job', 'favorite color', 'favorite city', 'favorite tea',
    'favorite sport', 'instrument', 'favorite season', 'garden flower', 'favorite painter',
    'favorite dessert', 'allergy', 'favorite bird', 'dream destination', 'favorite word',
    'favorite font', 'favorite cheese', 'best subject in school', 'favorite tree',
    'favorite gemstone', 'signature dish'}
# Single common words with unrelated senses are not used to select natural text.
POLYSEMY_EXCLUSIONS = {'teal', 'periwinkle', 'Hobart', 'rowing', 'curling', 'fencing', 'latex', 'shellfish', 'Latin',
                      'chemistry', 'botany', 'astronomy', 'geometry', 'rhetoric', 'jasper'}


def concepts():
    """Fixed real values from ATTRS, independent of seed; excludes EXTRA_ATTRS."""
    return sorted({value.removeprefix('the ') for attr, pool in E.ATTRS if attr in REAL_ATTRIBUTES
                   for value in pool}, key=str.casefold)


def concept_finder(text_panel=False):
    """Compile the pool once instead of thrashing re's cache for every window."""
    values = [v for v in concepts() if not text_panel or v not in POLYSEMY_EXCLUSIONS]
    canonical = {value.casefold(): value for value in values}
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(value) for value in
                         sorted(values, key=len, reverse=True)) + r')(?!\w)', re.IGNORECASE)
    def find(text):
        return sorted({canonical[match.group().casefold()] for match in pattern.finditer(text)}, key=str.casefold)
    return find


def census_sources(reads, records):
    """Declarations cannot shrink recorded reads. No records means report-only.

    Records are frozen JSON {streams: {domain: path}, cursors: {domain: end}}
    or census resume checkpoints (identity.streams and cursors). Include every
    census invocation used to choose the banks, including resumed invocations.
    """
    declared = {}
    for pair in reads:
        path, end = pair.rsplit(':', 1)
        path, end = str(Path(path).resolve()), int(end)
        if end < 0:
            raise ValueError('negative census end cursor')
        declared[path] = max(end, declared.get(path, 0))
    actual, certificates = {}, []
    fingerprinted = True
    for path in records:
        path = Path(path)
        record = (torch.load(path, map_location='cpu', weights_only=False)
                  if path.suffix == '.pt' else json.loads(path.read_text()))
        streams = record.get('streams', record.get('identity', {}).get('streams'))
        cursors = record.get('cursors')
        if not streams or not cursors or set(streams) != set(cursors):
            raise ValueError('census record needs every stream and recorded end cursor')
        calibration = record.get('tau_calibration', {}).get('targets', [])
        # Resume checkpoints omit calibration cursors. Conservatively scan all
        # reserved targets, covering calibration/agreement as well as main reads.
        if record.get('identity'):
            calibration = record['identity'].get('targets', calibration)
        order = record.get('identity', {}).get('domains', record.get('domains', list(streams)))
        if calibration and (len(calibration) != len(order) or set(order) != set(streams)
                            or any(type(n) is not int or n < 0 for n in calibration)):
            raise ValueError('census target domains/end cursors differ')
        target_by_domain = dict(zip(order, calibration))
        for name, source in streams.items():
            end = cursors[name]
            if type(end) is not int or end < 0:
                raise ValueError('invalid recorded census end cursor')
            if calibration:
                end = max(end, target_by_domain[name])
            source = str(Path(source).resolve())
            recorded_hash = record.get('inputs', record.get('input_sha256', {})).get(source)
            if recorded_hash is None:
                fingerprinted = False
            elif recorded_hash != E.file_sha(source):
                raise ValueError('recorded census stream hash changed: '+source)
            actual[source] = max(actual.get(source, 0), end)
        certificates.append(dict(path=str(path.resolve()), sha256=E.file_sha(path)))
    for path, end in actual.items():
        if path not in declared or declared[path] < end:
            raise ValueError('under-declared census read: '+path)
    sources, entries = {}, []
    for path, end in declared.items():
        stream = read_tokens(path)
        if end > len(stream):
            raise ValueError('census end cursor exceeds stream length')
        sources[path] = stream[:end]
        entries.append(dict(path=path, end_token=end, recorded_end=actual.get(path), sha256=E.file_sha(path)))
    return sources, dict(reads=entries, records=certificates,
                         complete=bool(actual) and fingerprinted and set(actual) == set(declared), gram_length=32, stride=1)


def verify_census_certificate(panel):
    cert = panel.get('census_scan', {})
    if not cert.get('complete') or not cert.get('records') or not cert.get('reads'):
        raise ValueError('gate-ready panel lacks complete recorded census scan')
    for entry in cert['records'] + cert['reads']:
        if E.file_sha(Path(entry['path'])) != entry['sha256']:
            raise ValueError('census certificate source hash changed')
    sources, rebuilt = census_sources([r['path']+':'+str(r['end_token']) for r in cert['reads']],
                                     [r['path'] for r in cert['records']])
    if not rebuilt['complete'] or rebuilt['reads'] != cert['reads']:
        raise ValueError('census certificate differs from recorded reads')
    contaminated, report = window_overlaps(panel['text_windows'], sources)
    if contaminated:
        raise ValueError('in-topic census 32-gram overlap: '+str(sorted(contaminated)))
    return report


def load_panel(path, args, tok, dev, min_items=200):
    panel = json.loads(Path(path).read_text())
    if panel.get('version') not in (1, 2):
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
    allowed = set(concepts()) | set(panel.get('neighbour_concepts', []))
    if any(row['concept'] not in allowed or not isinstance(row['answers'], list) or not row['answers'] or
           any(not isinstance(a,str) or not a.strip() for a in row['answers']) or type(row.get('base_correct')) is not bool or
           not re.search(r'(?<!\w)' + re.escape(row['concept']) + r'(?!\w)', row['prompt'], re.IGNORECASE)
           for row in qa):
        raise ValueError('in-topic QA must be scored questions about pool values or registered neighbours')
    if panel['version'] == 2:
        digest = panel.pop('panel_sha256', None)
        actual = hashlib_panel(panel)
        panel['panel_sha256'] = digest
        if digest != actual or protocol.get('batch_size') != 1:
            raise ValueError('in-topic panel hash or batch-one protocol differs')
        if panel.get('concepts') != concepts():
            raise ValueError('in-topic pool-level concepts differ')
        if {r.get('stratum') for r in qa} != {'value', 'neighbour'}:
            raise ValueError('in-topic v2 needs both value and neighbour strata')
        values = set(concepts())
        if any(row.get('pair') not in values or not isinstance(row.get('category'), str) or not row['category'] or
               (row['stratum'] == 'value' and row['concept'] != row['pair']) or
               (row['stratum'] == 'neighbour' and row['concept'] in values) for row in qa):
            raise ValueError('in-topic concept pair/category definitions differ')
        if panel.get('gate_ready', False):
            if 'calibration' not in panel:
                raise ValueError('in-topic gating panel lacks calibration')
            verify_census_certificate(panel)
            try:
                from .build_in_topic_panel import readiness
            except ImportError:
                from build_in_topic_panel import readiness
            ready, _ = readiness(qa, panel['calibration'])
            if not ready:
                raise ValueError('in-topic calibration cannot support the declared gate')
        masks = panel.get('mention_masks')
        if masks is None or len(masks) != len(windows) or any(mask != mention_mask(tok, win) for mask,win in zip(masks, windows)):
            raise ValueError('in-topic mention masks differ from the registered matcher')
        if len([r for r in qa if r['stratum'] == 'value']) < min_items:
            raise ValueError(f'in-topic panel requires >= {min_items} value QA (both correctness strata)')
    if not panel.get('gate_ready') and panel.get('census_scan', {}).get('complete'):
        verify_census_certificate(panel)
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
                mention_masks=panel.get('mention_masks'),
                path=str(Path(path).resolve()), sha256=E.file_sha(path), overlap=overlap)


@torch.no_grad()
def qa_flags(model, tok, rows, dev, bos, max_new=32, bs=16, stop_k=2):
    items = [E.answer_item(tok, row['prompt'], row['answers'][0], bos) for row in rows]
    gens = E.greedy_generate(model, [it['p'] for it in items], dev, max_new, bs)
    class Folded:
        def decode(self, ids):
            return tok.decode(ids).casefold()
    folded = Folded()
    flags = [any(E.score_generation(folded if row.get('stratum') else tok, gen,
                 answer.casefold() if row.get('stratum') else answer, stop_k)['clean'] for answer in row['answers'])
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
                passed=mean >= -tolerance-1e-12 and ci[0] >= -.025-1e-12)


def hashlib_panel(panel):
    import hashlib
    return hashlib.sha256(json.dumps(panel, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cluster_interval(values, clusters, confidence=.95, draws=5000, seed=57):
    """Equal-weight underlying clusters; paraphrases do not create evidence."""
    if len(values) != len(clusters) or not values:
        return dict(mean=None, lower=None, upper=None, clusters=0)
    groups = {}
    for value, key in zip(values, clusters):
        groups.setdefault(str(key), []).append(float(value))
    means = np.asarray([np.mean(v) for v in groups.values()])
    if len(means) < 2:
        return dict(mean=float(means.mean()), lower=None, upper=None, clusters=len(means))
    if draws < 100:
        raise ValueError('need >=100 cluster bootstrap draws')
    rng = np.random.default_rng(seed)
    samples = np.concatenate([means[rng.integers(len(means), size=(min(256, draws-i), len(means)))].mean(1)
                              for i in range(0, draws, 256)])
    return dict(mean=float(means.mean()), lower=float(np.quantile(samples, 1-confidence)),
                upper=float(np.quantile(samples, confidence)), clusters=len(means),
                confidence=confidence, draws=draws, weighting='equal underlying clusters')


def knowledge_report(rows, base, patched, gate_ready=False):
    report = paired_flags(base, patched)
    changes = np.asarray(patched, float)-np.asarray(base, float)
    report['strata'] = {}
    for stratum in ('value', 'neighbour'):
        for correct in (True, False):
            ids = [i for i, r in enumerate(rows) if r.get('stratum', 'value') == stratum and bool(base[i]) == correct]
            if ids:
                report['strata'][f'{stratum}/base_{correct}'] = paired_flags([base[i] for i in ids], [patched[i] for i in ids])
    pairs = {}
    for row, delta in zip(rows, changes):
        if row.get('answer_is_value', False):
            continue  # Direct priming targets are a separate reported stratum.
        pairs.setdefault(row.get('pair', row['concept']), {}).setdefault(row.get('stratum', 'value'), []).append(delta)
    differences = [np.mean(v['value'])-np.mean(v['neighbour']) for v in pairs.values() if set(v) == {'value', 'neighbour'}]
    interval = cluster_interval(differences, list(range(len(differences))), .95)
    # 5/95 percentiles give a two-sided 90% concept-cluster interval.
    interval['ci90'] = [interval['lower'], interval['upper']]
    report.update(difference_in_differences=interval, gate_ready=gate_ready,
                  passed=bool(gate_ready and interval['clusters'] >= 40 and interval['mean'] is not None and interval['lower'] is not None and
                              interval['mean'] >= -.01-1e-12 and interval['lower'] >= -.025-1e-12),
                  status='gating' if gate_ready else 'report-only: calibration/minimum concepts unresolved')
    primed = [i for i, r in enumerate(rows) if r.get('answer_is_value', False)]
    report['answer_is_value'] = paired_flags([base[i] for i in primed], [patched[i] for i in primed]) if primed else None
    return report


def mention_mask(tok, window):
    """Exclude every token intersecting a matched mention and the following token."""
    text = tok.decode(window)
    finder = concept_finder(text_panel=True)
    spans = [match.span() for value in finder(text) for match in re.finditer(
        r'(?<!\w)'+re.escape(value)+r'(?!\w)', text, re.IGNORECASE)]
    # Decode prefixes of the *original* IDs; re-encoding text can change boundaries.
    ends = [len(tok.decode(window[:i])) for i in range(len(window)+1)]
    mask = [True]*len(window)
    mask[0] = False
    for i in range(len(window)):
        if any(ends[i] < stop and ends[i+1] > start for start, stop in spans):
            mask[i] = False
            if i+1 < len(mask):
                mask[i+1] = False
    return mask


@torch.no_grad()
def text_scores(model, windows, masks, dev):
    inclusive, excluded = [], []
    for window, mask in zip(windows, masks):
        x = window[None].to(dev)
        losses = F.cross_entropy(E.logits_of(model, x).float()[0, :-1, :E.VOCAB], x[0, 1:], reduction='none')
        keep = torch.tensor(mask[1:], device=dev, dtype=torch.bool)
        if not keep.any():
            raise ValueError('no text targets remain after excluding mentions')
        inclusive.append(float(losses.mean()))
        excluded.append(float(losses[keep].mean()))
    return dict(inclusive=inclusive, excluded=excluded)


def text_report(base, patched, confidence=.95):
    report = {}
    for kind in ('inclusive', 'excluded'):
        delta = np.asarray(patched[kind])-np.asarray(base[kind])
        se = float(delta.std(ddof=1)/math.sqrt(len(delta)))
        interval = cluster_interval(delta.tolist(), list(range(len(delta))), confidence)
        report[kind] = dict(n=len(delta), dnll=float(delta.mean()), dnll_se=se,
                            upper=interval['upper'], window_deltas=delta.tolist(), base_nll=float(np.mean(base[kind])))
    gated = report['excluded']
    report.update(gated, passed=bool(gated['upper'] is not None and gated['upper'] <= .003+1e-12 and
                                    gated['dnll_se'] <= .001+1e-12),
                  policy='equal tolerance, separately measured; mention positions and following token excluded; paired SE <=.001')
    return report


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
                           train_person=meta.get('train_person'), train_attr_group=meta.get('train_attr_group'),
                           stored_alone_gen=meta.get('alone_resident_test_gen')))
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
    if panel['raw']['version'] == 1 and not all(flags):
        raise ValueError('in-topic QA is no longer base-correct on this export/backend; rebuild panel')
    if panel['raw']['version'] == 2:
        if flags != [r['base_correct'] for r in panel['qa']]:
            raise ValueError('frozen panel base flags changed; rebuild/calibrate with identical protocol')
        return dict(qa=flags, text=text_scores(model, panel['windows'], panel['mention_masks'], dev))
    return dict(qa=flags, nll=E.nll_windows(model, panel['windows'], dev))


@torch.no_grad()
def evaluate(model, tok, panel, baseline, written, dev, bos, args):
    batch_size = panel['raw']['selection'].get('protocol', {}).get('batch_size',16)
    flags, answers = qa_flags(model, tok, panel['qa'], dev, bos, bs=batch_size)
    qa = paired_flags(baseline['qa'], flags)
    if panel['raw']['version'] == 2:
        qa = knowledge_report(panel['qa'], baseline['qa'], flags, panel['raw'].get('gate_ready', False))
    else:
        qa.update(passed=False, gate_ready=False, status='report-only: legacy one-directional panel; rebuild v2')
    qa.update(base_candidate_accuracy=panel['raw']['selection']['base_accuracy'],
              items=[dict(concept=row['concept'], prompt=row['prompt'], correct=flag, answer=answer)
                     for row,flag,answer in zip(panel['qa'],flags,answers)])
    if panel['raw']['version'] == 2:
        text = text_report(baseline['text'], text_scores(model, panel['windows'], panel['mention_masks'], dev))
    else:
        nll = np.asarray(E.nll_windows(model, panel['windows'], dev), dtype=np.float64)
        base = np.asarray(baseline['nll'], dtype=np.float64)
        delta = nll-base
        text = dict(n=len(delta), dnll=float(delta.mean()), dnll_se=float(delta.std(ddof=1)/math.sqrt(len(delta))),
                    base_nll=float(base.mean()), window_deltas=delta.tolist(), passed=float(delta.mean()) <= .003+1e-12)
    banks = []
    for group in written:
        patched = fact_flags(model,tok,group,dev,args)
        stored = group.get('stored_alone_gen')
        if stored is None:
            banks.append(dict(bank=group['bank'], facts=group['facts'], n=len(patched), passed=False, stored_reference_available=False,
                              status='report-only: no stored alone-resident TEST GEN',
                              co_resident_accuracy=float(np.mean(patched))))
            continue
        # Stored aggregate GEN is the reference; never rescore it under caller flags.
        alone = float(stored['clean'] if isinstance(stored, dict) else stored)
        if not math.isfinite(alone) or not 0 <= alone <= 1:
            raise ValueError('stored alone-resident TEST GEN must be a fraction in [0,1]')
        loss = alone - float(np.mean(patched))
        result = dict(n=len(patched), stored_reference_available=True, accuracy_delta=-loss, alone_resident_gen=alone, passed=loss <= .01,
                      reference='stored alone-resident TEST GEN')
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
            group['bank'] = bank
        branches.activate(tuple(branches.states))
        return evaluate(branches.model,tok,panel,baseline,groups,dev,bos,args)
    finally:
        branches.activate(())
