"""C38 matched comparisons from completed experiment JSONs; no model loading."""
import argparse
import json
from pathlib import Path

try:
    from . import e2e_fsa as E
except ImportError:
    import e2e_fsa as E


def recall(result):
    panel = result['patched']
    return panel.get('test_gen_person', panel['test_gen'])['recall']


def new_budget(result):
    dose = result['dose_response']
    return {key: dose[key] for key in ('new_facts', 'new_updates', 'new_fact_format_steps',
                                       'new_examples_per_fact', 'new_examples_per_global_fact',
                                       'new_recipe', 'new_loss_tokens')}


def require_same_new_world(results):
    E.eda.require_same(results, "C38 result set")
    first = results[0]
    keys = ('facts', 'seed', 'train_person', 'train_attr_group', 'local_lambda', 'lr', 'method', 'fmt')
    for result in results[1:]:
        if any(result[key] != first[key] for key in keys) or new_budget(result) != new_budget(first):
            raise ValueError('new world, recipe, updates or exposures differ')


def sequential(resident, masked, full):
    require_same_new_world([resident, masked, full])
    rs, ms = (r['sequential_revoke'] for r in (resident, masked))
    if rs['resident_slots']['mask_only'] or not ms['resident_slots']['mask_only']:
        raise ValueError('need resident-on and resident-mask-only arms')
    if (rs['resident_slots']['union_slots'] != ms['resident_slots']['union_slots'] or
            [f['sha256'] for f in rs['resident_slots']['files']] !=
            [f['sha256'] for f in ms['resident_slots']['files']]):
        raise ValueError('resident masks differ')
    panel = rs['new_residents_revoked']['acq']
    revoked = panel.get('test_gen_person', panel['test_gen'])['recall']
    nnz = [r['dose_response']['new_nnz'] for r in (resident, masked, full)]
    return dict(eda_backend=E.eda.recorded(resident), dependence=revoked-recall(masked), capacity_cost=recall(masked)-recall(full),
                recall=dict(revoked=revoked, slot_masked=recall(masked), full_base=recall(full)),
                final_nnz=dict(zip(('resident', 'slot_masked', 'full_base'), nnz)),
                nnz_differ=len(set(nnz)) > 1,
                interpretation='dependence is B-revoked minus slot-masked B; capacity cost is slot-masked minus full B')


def factorial(results, repeats=1000, seed=0):
    backend = E.eda.require_same(results, "C38 factorial result set")
    cells = {r['c35_recipe']['factorial_cell']: r for r in results}
    if len(results) != 4 or set(cells) != {'sparse_tern', 'sparse_fp', 'dense_tern', 'dense_fp'}:
        raise ValueError('need all four distinct factorial cells')
    first = results[0]
    ref = first['factorial_lr_search']
    candidate_count = len(ref['candidates'])
    matched = ('facts', 'seed', 'layers', 'local_lambda', 'fact_batch', 'answer_only', 'aug', 'rehearsal_frac')
    for result in results:
        search = result['factorial_lr_search']
        if (len(search['candidates']) != candidate_count or
                search['updates_per_trial'] != ref['updates_per_trial'] or
                search['total_updates'] != ref['total_updates'] or
                any(result[key] != first[key] for key in matched) or
                result['factorial_accounting']['anchor_indices'] != first['factorial_accounting']['anchor_indices']):
            raise ValueError('factorial data/recipe/anchor draws/search budget differ')
        if not result.get('factorial_gap'):
            raise ValueError('factorial contrasts require --bootstrap observations')
    rows = {cell: r['factorial_gap']['test_gen'] for cell,r in cells.items()}
    contrasts = {}
    for label, left, right in (
            ('ternary_sparse_minus_dense', 'sparse_tern', 'dense_tern'),
            ('fp_sparse_minus_dense', 'sparse_fp', 'dense_fp'),
            ('sparse_ternary_minus_fp', 'sparse_tern', 'sparse_fp'),
            ('dense_ternary_minus_fp', 'dense_tern', 'dense_fp')):
        contrasts[label] = E.paired_cluster_contrast(rows[left], rows[right], repeats, seed)
    # Paired difference of differences retains each fact's covariance.
    def fact_means(observations):
        groups = {}
        for row in observations:
            groups.setdefault(row['fi'], []).append(row['value'])
        return {fi: sum(v)/len(v) for fi,v in groups.items()}
    means = {cell: fact_means(value) for cell,value in rows.items()}
    ids = sorted(means['sparse_tern'])
    left = [dict(fi=fi, value=means['sparse_tern'][fi]-means['dense_tern'][fi]) for fi in ids]
    right = [dict(fi=fi, value=means['sparse_fp'][fi]-means['dense_fp'][fi]) for fi in ids]
    contrasts['interaction'] = E.paired_cluster_contrast(left, right, repeats, seed)
    gaps = {cell: r['factorial_gap']['train_tf_minus_test_gen'] for cell,r in cells.items()}
    gap_rows = {}
    for cell,r in cells.items():
        train = fact_means(r['factorial_gap']['train_tf'])
        gap_rows[cell] = [dict(fi=fi, value=train[fi]-means[cell][fi]) for fi in ids]
    contrasts['gap_sparse_minus_dense_ternary'] = E.paired_cluster_contrast(
        gap_rows['sparse_tern'], gap_rows['dense_tern'], repeats, seed)
    searches = {}
    for cell, r in cells.items():
        search = r['factorial_lr_search']
        grid = [c['lr'] for c in search['candidates']]
        searches[cell] = dict(grid=grid, selected_lr=search['selected_lr'],
                             winner_on_edge=search['selected_lr'] in (min(grid), max(grid)))
    return dict(eda_backend=backend, gaps=gaps, contrasts=contrasts, factorial_lr_search=searches,
                capacity={cell: r['factorial_accounting'] for cell,r in cells.items()},
                capacity_policy='format-native; explicit nnz/bytes, no capacity-matched claim')


def dose_response(results):
    require_same_new_world(results)
    return dict(eda_backend=E.eda.recorded(results[0]), points=[dict(r['dose_response'], recall=recall(r),
                    revoked_recall=(r['sequential_revoke']['new_residents_revoked']['acq']['test_gen']['recall']
                                    if r.get('sequential_revoke') else recall(r)),
                    damage=r['patched']['big']) for r in results],
                matched_new_budget=new_budget(results[0]),
                dose_axis='actual occupied union slots; resident facts/exposures also recorded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('sequential', 'factorial', 'dose'))
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    results = [json.loads(path.read_text()) for path in a.results]
    backend = E.eda.require_same(results, "C38 result set")
    print(f"EDA-BACKEND {backend}", flush=True)
    if a.mode == 'sequential':
        if len(results) != 3:
            parser.error('sequential requires resident, masked and full results in that order')
        report = sequential(*results)
    else:
        report = factorial(results) if a.mode == 'factorial' else dose_response(results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
