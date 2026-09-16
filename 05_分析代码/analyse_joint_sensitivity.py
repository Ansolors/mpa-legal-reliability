"""Extend the existing held-out sensitivity analyses to both joint measures.

Reads the frozen cases, original responses and archived explanation codes.
Writes separate derived results; does not run models or revise any judgments.
Only Python's standard library is required.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import json
import math

from score_readable_labels import parse_readable, LABELS

BASE = Path(__file__).resolve().parents[1]
FORMAL = BASE / '04_分析结果/正式研究'
DATA = BASE / '04_分析结果/法规命题基准'
MODELS = ['Qwen3-4B-Instruct-2507-4bit', 'Phi-4-mini-instruct-4bit']
METHODS = ['direct', 'structured']
DRAFT = {'TMN06', 'STI06', 'PON06', 'AMA06', 'DWE06'}
COMMENCEMENT = {'TMN07', 'STI07', 'PON07', 'AMA07'}
OUTCOME = {'ADD08', 'PEC08', 'UTH08', 'DWE08'}
VARIANTS = [
    ('all_heldout', 'All held-out claims (36 claims)'),
    ('shared_draft_and_commencement_equal_group_weight',
     'Shared-question weighting (29 units)'),
    ('exclude_shared_amendment_or_draft_inputs',
     'Exclude shared-instrument inputs (18 claims)'),
    ('exclude_four_outcome_wording_cases',
     'Exclude outcome-wording cases (32 claims)'),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, weights):
    def count(predicate):
        return sum((weights[r['case_id']] for r in rows if predicate(r)), Fraction())

    total = count(lambda r: True)
    correct = count(lambda r: r['prediction'] == r['gold'])
    joint = count(lambda r: r['prediction'] == r['gold']
                  and r['rationale_status'] == 'consistent')
    aligned = count(lambda r: r['prediction'] == r['gold']
                    and r['rationale_status'] == 'consistent'
                    and r['alignment'] == 'aligned')
    f1 = []
    for label in LABELS:
        tp = count(lambda r: r['gold'] == label and r['prediction'] == label)
        fp = count(lambda r: r['gold'] != label and r['prediction'] == label)
        fn = count(lambda r: r['gold'] == label and r['prediction'] != label)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction())
    return {
        'n': len(rows), 'total_weight': float(total),
        'correct': float(correct), 'accuracy': float(correct / total),
        'macro_f1': float(sum(f1) / len(LABELS)),
        'joint_count': float(joint), 'joint_rate': float(joint / total),
        'aligned_joint_count': float(aligned), 'aligned_joint_rate': float(aligned / total),
        'exact_counts': {'total_weight': str(total), 'joint': str(joint),
                         'aligned_joint': str(aligned)},
    }


def comparison(rows, model, selected, weights):
    groups = {
        method: metrics([r for r in rows if r['model'] == model
                         and r['method'] == method and r['case_id'] in selected], weights)
        for method in METHODS
    }
    assert all(g['n'] == len(selected) for g in groups.values())
    return {
        'model': model, 'case_ids': sorted(selected), 'groups': groups,
        'differences_pp': {
            name: 100 * (groups['structured'][name] - groups['direct'][name])
            for name in ['accuracy', 'joint_rate', 'aligned_joint_rate']
        },
    }


def check_legacy_metrics(new, old):
    for method in METHODS:
        for name in ['n', 'total_weight', 'correct', 'accuracy', 'macro_f1']:
            assert math.isclose(new['groups'][method][name], old['groups'][method][name],
                                rel_tol=1e-12, abs_tol=1e-12), (method, name)


def fixed(value, places=1):
    return f'{value:.{places}f}'


def table_tex(comparisons):
    lookup = {(r['analysis'], r['model']): r for r in comparisons}
    lines = [r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}',
             r'\toprule']
    for panel, columns in [
        ('Label performance', [('accuracy', 'Accuracy (\\%)', 100, 1),
                               ('macro_f1', 'Macro-F1', 1, 3)]),
        ('Joint performance', [('joint_rate', 'Joint (\\%)', 100, 1),
                               ('aligned_joint_rate', 'Aligned joint (\\%)', 100, 1)]),
    ]:
        if panel == 'Joint performance':
            lines.append(r'\midrule')
        lines.extend([
            r'\multicolumn{6}{@{}l}{\textit{' + panel + r'}} \\',
            r'& & \multicolumn{2}{c}{Direct} & \multicolumn{2}{c}{Checklist} \\',
            r'\cmidrule(lr){3-4}\cmidrule(l){5-6}',
            'Analysis & Model & ' + ' & '.join([c[1] for c in columns] * 2) + r' \\',
            r'\midrule',
        ])
        for analysis, label in VARIANTS[1:]:
            for i, model in enumerate(MODELS):
                groups = lookup[analysis, model]['groups']
                values = [fixed(groups[method][key] * scale, places)
                          for method in METHODS for key, _, scale, places in columns]
                cells = [label if i == 0 else '', 'Qwen' if i == 0 else 'Phi'] + values
                lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular*}', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=FORMAL)
    parser.add_argument('--table-output', type=Path,
                        help='Optional destination for the combined LaTeX sensitivity table.')
    args = parser.parse_args()
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(BASE))] = sha256(path)
        return json.loads(path.read_text())

    frozen = read(DATA / 'publication_manifest.json')
    for name, digest in frozen['files'].items():
        assert sha256(DATA / name) == digest, name
    cases_list = read(DATA / 'cases_v1.json')
    cases = {c['case_id']: c for c in cases_list}
    assert len(cases) == len(cases_list) == 72
    inputs[str((DATA / 'evidence_v1.json').relative_to(BASE))] = sha256(DATA / 'evidence_v1.json')
    rows = read(FORMAL / '正式逐条分析.json')
    legacy = read(FORMAL / '正式研究结果.json')
    audit_list = read(FORMAL / 'explanation_codes.json')['records']
    key = lambda r: (r['split'], r['case_id'], r['model'], r['method'])
    audit = {key(r): r for r in audit_list}
    assert len(rows) == len({key(r) for r in rows}) == len(audit) == len(audit_list) == 288
    held = [r for r in rows if r['split'] == 'heldout']
    held_cases = {c['case_id']: c for c in cases_list if c['split'] == 'heldout'}
    assert len(held_cases) == 36 and len(held) == 144
    assert {i for i, c in held_cases.items() if c['family'] == 'draft_status'} == DRAFT
    assert {i for i, c in held_cases.items() if c['family'] == 'effective_date'} == COMMENCEMENT
    assert {i for i, c in held_cases.items() if c['family'] == 'law_to_outcome_gap'} == OUTCOME
    raw_lookup = {}
    for path in sorted((BASE / '04_分析结果/模型先导实验').glob('*_heldout_raw.jsonl')):
        inputs[str(path.relative_to(BASE))] = sha256(path)
        for line in path.read_text().splitlines():
            raw = json.loads(line)
            k = ('heldout', raw['case_id'], raw['model'], raw['method'])
            assert k not in raw_lookup
            raw_lookup[k] = raw
    assert set(raw_lookup) == {key(r) for r in held}
    assert Counter((r['model'], r['method']) for r in held) == {
        (model, method): 36 for model in MODELS for method in METHODS}
    for r in held:
        case, code, raw = cases[r['case_id']], audit[key(r)], raw_lookup[key(r)]
        parsed, _ = parse_readable(raw['output'])
        assert raw['status'] == 'completed' and parsed is not None
        assert r['prediction'] == parsed['label'] and r['reason'] == parsed['reason']
        assert r['gold'] == case['gold_label']
        assert r['mpa_id'] == case['mpa_id'] and r['family'] == case['family']
        assert r['shared_instrument'] == any(x.split(':')[0] in {'AMD', 'DRF'}
                                            for x in case['evidence_ids'])
        assert all(r[name] == code[name] for name in ['rationale_status', 'alignment'])
        correct = r['prediction'] == case['gold_label']
        joint = correct and code['rationale_status'] == 'consistent'
        assert r['correct'] == correct and r['joint_consistent'] == joint
        assert r['joint_consistent_aligned'] == (joint and code['alignment'] == 'aligned')

    unit_weights = {i: Fraction(1) for i in held_cases}
    shared_weights = {i: Fraction(1, 5) if i in DRAFT else Fraction(1, 4)
                      if i in COMMENCEMENT else Fraction(1) for i in held_cases}
    assert sum(shared_weights.values()) == 29
    assert {i: float(w) for i, w in shared_weights.items()} == legacy['shared_question_weights']
    old_sensitivity = {(x['analysis'], x['model']): x for x in legacy['sensitivity']}
    comparisons = []
    for analysis, _ in VARIANTS:
        selected = set(held_cases)
        weights = shared_weights if analysis == VARIANTS[1][0] else unit_weights
        if analysis == VARIANTS[2][0]:
            selected = {i for i, c in held_cases.items()
                        if not any(x.split(':')[0] in {'AMD', 'DRF'} for x in c['evidence_ids'])}
            assert len(selected) == 18
        elif analysis == VARIANTS[3][0]:
            selected -= OUTCOME
            assert len(selected) == 32
        for model in MODELS:
            result = {'analysis': analysis, **comparison(held, model, selected, weights)}
            check_legacy_metrics(result, old_sensitivity[analysis, model])
            if analysis == 'all_heldout':
                for method in METHODS:
                    old = next(g for g in legacy['splits']['heldout']['groups']
                               if g['model'] == model and g['method'] == method)
                    assert result['groups'][method]['joint_count'] == old['joint_consistent']
                    assert result['groups'][method]['aligned_joint_count'] == old['joint_consistent_aligned']
            comparisons.append(result)

    old_loo = {(r['model'], r['excluded_mpa']): r for r in legacy['leave_one_mpa_out']}
    leave_one_out = []
    for model in MODELS:
        for mpa in sorted({c['mpa_id'] for c in held_cases.values()}):
            selected = {i for i, c in held_cases.items() if c['mpa_id'] != mpa}
            assert len(selected) == 32
            result = {'excluded_mpa': mpa, **comparison(held, model, selected, unit_weights)}
            check_legacy_metrics(result, old_loo[model, mpa])
            leave_one_out.append(result)
    ranges = []
    for model in MODELS:
        selected = [r for r in leave_one_out if r['model'] == model]
        ranges.append({'model': model, **{
            name: {'min_pp': min(r['differences_pp'][name] for r in selected),
                   'max_pp': max(r['differences_pp'][name] for r in selected)}
            for name in ['joint_rate', 'aligned_joint_rate']}})
    inputs[str(Path(__file__).resolve().relative_to(BASE))] = sha256(Path(__file__))
    inputs['05_分析代码/score_readable_labels.py'] = sha256(BASE / '05_分析代码/score_readable_labels.py')
    output = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis': 'Extension of the existing held-out sensitivity analyses to joint measures.',
        'definitions': {
            'joint': 'Reference-label agreement and no identified definite substantive explanation error.',
            'aligned_joint': 'Joint criterion plus label--explanation alignment.',
            'denominator': 'Total retained response weight; ambiguous explanations remain in the denominator.',
            'difference': 'Checklist minus direct rate, in percentage points.',
            'weights': 'Five draft claims receive 1/5 each; four commencement claims receive 1/4 each; all others receive 1.',
            'scope': 'Descriptive comparisons using the archived labels and codes; no new inference or recoding.',
        },
        'inputs_sha256': inputs, 'shared_question_weights': {i: str(w) for i, w in shared_weights.items()},
        'sensitivity': comparisons, 'leave_one_mpa_out': leave_one_out, 'leave_one_mpa_out_ranges': ranges,
        'validation': {'heldout_responses_checked_against_raw_and_audit': len(held),
                       'legacy_label_metric_comparisons_reproduced': len(comparisons) + len(leave_one_out)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / '联合指标敏感性分析.json'
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    if args.table_output:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        args.table_output.write_text(table_tex(comparisons))
    print(json.dumps({'output': str(destination), 'validation': output['validation'],
                      'leave_one_mpa_out_ranges': ranges}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
