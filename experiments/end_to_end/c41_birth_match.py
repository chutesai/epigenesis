"""Compare BOP code births at a fixed horizon; never infer a matched tau without measuring it.

Run: python experiments/end_to_end/c41_birth_match.py --target tstat.json
     --candidates tau025.json tau0125.json --horizon 40 [--emit-tau]
"""
import argparse
import json
from pathlib import Path

try:
    from . import eda_backend as eda
except ImportError:
    import eda_backend as eda


def read_births(path, horizon=40):
    """Accept a RESULT JSON or a log; reject incomplete curves rather than extrapolating."""
    raw = Path(path).read_text()
    if raw.lstrip().startswith('{'):
        result = json.loads(raw)
        events = result['bop_events']
    else:
        lines = raw.splitlines()
        events = [json.loads(line[4:]) for line in lines if line.startswith('BOP ')]
        results = [json.loads(line[7:]) for line in lines if line.startswith('RESULT ')]
        result = results[-1] if results else {}
    updates = [e['update'] for e in events if e['update'] <= horizon]
    if updates != list(range(1, horizon + 1)):
        raise ValueError(f'{path}: need every BOP update 1..{horizon}, exactly once')
    return dict(path=str(path), horizon=horizon,
                births=sum(e['births'] for e in events if e['update'] <= horizon),
                tau=result.get('bop_tau'), optim=result.get('optim'), c41=result.get('c41', {}),
                identity=dict(eda_backend=eda.recorded(result), **{key: result[key] for key in (
                    'seed', 'budget', 'method', 'fmt', 'export', 'export_step', 'people', 'facts_per_person',
                    'train_person', 'fact_batch', 'aug', 'aug_all', 'support_score', 'cost_rel',
                    'local_lambda', 'budget_freeze', 'keep_live', 'slot_frac', 'bop_gamma', 'bop_accum',
                    'tau_anneal', 'death_tau_mult', 'rehearsal_frac', 'sparse_state') if key in result}))


def compare_births(target, candidates, horizon=40):
    target = read_births(target, horizon)
    reports = []
    for path in candidates:
        report = read_births(path, horizon)
        eda.require_same([target['identity'], report['identity']], 'C41 birth matching')
        if report['optim'] != 'bop' or report['c41'].get('bop_tstat') is not None or report['c41'].get('unit_burst', 0):
            raise ValueError(f'{path}: candidate must be standard Bop')
        mismatched = [key for key in target['identity'].keys() & report['identity'].keys()
                      if target['identity'][key] != report['identity'][key]]
        if mismatched:
            raise ValueError(f'{path}: candidate differs from target in {sorted(mismatched)}')
        report['relative_error'] = abs(report['births'] - target['births']) / max(target['births'], 1)
        reports.append(report)
    if not reports:
        raise ValueError('at least one measured standard-Bop candidate is required')
    reports.sort(key=lambda row: row['relative_error'])
    return dict(target=target, candidates=reports, best=reports[0],
                metric='all code births including rebirths; support additions excluded')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--candidates', required=True, nargs='+', type=Path)
    parser.add_argument('--horizon', type=int, default=40)
    parser.add_argument('--tolerance', type=float, default=.05)
    parser.add_argument('--emit-tau', action='store_true', help='print only a measured tau within tolerance')
    args = parser.parse_args(argv)
    if args.horizon < 1 or not 0 <= args.tolerance < 1:
        parser.error('horizon must be positive and tolerance must be in [0, 1)')
    try:
        report = compare_births(args.target, args.candidates, args.horizon)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.emit_tau:
        best = report['best']
        if best['relative_error'] > args.tolerance or best['tau'] is None:
            parser.error('no measured tau matches births within tolerance; refine the tau sweep')
        print(best['tau'])
    else:
        print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
