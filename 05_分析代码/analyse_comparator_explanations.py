"""Compute comparator joint measures from the final published explanation codes."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import html
import json

BASE = Path(__file__).resolve().parents[1]
EXTENSION = Path('04_分析结果/新增模型对照_20260915')
METHODS = ('direct', 'structured')
STATUS = {'consistent': '未发现明确实质错误', 'substantive_error': '存在实质错误', 'indeterminate': '含糊，暂不纳入联合合格数'}
ALIGNMENT = {'aligned': '一致', 'misaligned': '不一致', 'indeterminate': '关系不确定'}
ERRORS = {'activity_scope', 'actor_scope', 'claim_scope', 'cross_reference', 'exception_scope',
          'invented_individual_fact', 'invented_stipulated_fact', 'permit_conjunction',
          'quantity_or_time', 'source_content', 'temporal_validity', 'zone_scope'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(base):
    out = base / EXTENSION
    audit_path = out / 'explanation_codes.json'
    raw_path = out / 'Qwen3-14B-4bit_heldout_raw.jsonl'
    case_path = base / '04_分析结果/法规命题基准/cases_v1.json'
    source_path = base / '04_分析结果/法规命题基准/evidence_v1.json'
    audit = json.loads(audit_path.read_text())
    raw = {(r['case_id'], r['method']): r for r in map(json.loads, raw_path.read_text().splitlines())}
    cases = {c['case_id']: c for c in json.loads(case_path.read_text()) if c['split'] == 'heldout'}
    sources = json.loads(source_path.read_text())
    expected = {(c, m) for c in cases for m in METHODS}
    assert len(audit['records']) == len(raw) == len(expected) == 72
    assert {(r['case_id'], r['method']) for r in audit['records']} == set(raw) == expected
    rows = []
    for code in audit['records']:
        key = code['case_id'], code['method']
        r, c = raw[key], cases[key[0]]
        p = json.loads(r['output'])
        assert r['status'] == 'completed'
        assert code['rationale_status'] in STATUS and code['alignment'] in ALIGNMENT
        assert set(code['error_types']) <= ERRORS
        assert (code['rationale_status'] == 'substantive_error') == bool(code['error_types'])
        assert code['reference_pages'] and set(code['reference_pages']) <= set(c['evidence_ids'])
        assert code['note'].strip()
        correct = p['label'] == c['gold_label']
        joint = correct and code['rationale_status'] == 'consistent'
        rows.append({**code, 'gold_label': c['gold_label'], 'prediction': p['label'],
                     'correct': correct, 'joint_compatible': joint,
                     'joint_compatible_aligned': joint and code['alignment'] == 'aligned',
                     'mpa_id': c['mpa_id'], 'family': c['family'],
                     'output_sha256': hashlib.sha256(r['output'].encode()).hexdigest()})

    groups = []
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        group = {'method': method, 'n': len(selected),
                 'label_correct': sum(r['correct'] for r in selected),
                 'explanation_status_counts': dict(Counter(r['rationale_status'] for r in selected)),
                 'alignment_counts': dict(Counter(r['alignment'] for r in selected)),
                 'correct_label_excluded_from_joint': [r['case_id'] for r in selected if r['correct'] and not r['joint_compatible']],
                 'correct_label_with_error': sum(r['correct'] and r['rationale_status'] == 'substantive_error' for r in selected)}
        for metric in ('joint_compatible', 'joint_compatible_aligned'):
            group[metric + '_count'] = sum(r[metric] for r in selected)
            group[metric + '_rate'] = group[metric + '_count'] / len(selected)
        groups.append(group)

    objective = json.loads((out / '客观结果汇总.json').read_text())
    weights = objective['shared_question_weights']
    variants = [('all_heldout', lambda r: True, None),
                ('shared_draft_and_commencement_equal_group_weight', lambda r: True, weights),
                ('exclude_shared_amendment_or_draft_inputs',
                 lambda r: not any(s.startswith(('AMD:', 'DRF:')) for s in cases[r['case_id']]['evidence_ids']), None),
                ('exclude_four_outcome_wording_cases', lambda r: r['family'] != 'law_to_outcome_gap', None)]
    variants += [('exclude_mpa_' + m, lambda r, m=m: r['mpa_id'] != m, None) for m in sorted({c['mpa_id'] for c in cases.values()})]
    sensitivity = []
    for name, predicate, weight in variants:
        item = {'analysis': name, 'groups': {}}
        for method in METHODS:
            selected = [r for r in rows if r['method'] == method and predicate(r)]
            weighted = [(r, weight[r['case_id']] if weight else 1.) for r in selected]
            denominator = sum(w for _, w in weighted)
            group = {'n': len(selected), 'total_weight': denominator}
            for metric in ('joint_compatible', 'joint_compatible_aligned'):
                numerator = sum(w for r, w in weighted if r[metric])
                group[metric + '_numerator'] = numerator
                group[metric + '_rate'] = numerator / denominator
            item['groups'][method] = group
        sensitivity.append(item)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'Final explanation coding used in the manuscript',
              'human_confirmed': True,
              'input_sha256': {str(p.relative_to(base)): digest(p) for p in (audit_path, raw_path, case_path, source_path)},
              'groups': groups, 'sensitivity': sensitivity, 'records': rows}
    (out / '新增解释联合指标.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    result = calculate(args.base)
    print(json.dumps({'status': result['status'], 'groups': result['groups']}, ensure_ascii=False, indent=2))
