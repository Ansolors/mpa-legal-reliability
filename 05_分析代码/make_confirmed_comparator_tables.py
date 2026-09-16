"""Generate combined explanation and sensitivity tables from confirmed records."""
from pathlib import Path
from collections import Counter
import argparse
import json

BASE = Path(__file__).resolve().parents[1]
MODELS = ('Qwen3-4B-Instruct-2507-4bit', 'Phi-4-mini-instruct-4bit', 'Qwen3-14B-4bit')
NAMES = ('Qwen3-4B', 'Phi-4-mini', 'Qwen3-14B')
METHODS = ('direct', 'structured')
VARIANTS = (
    ('shared_draft_and_commencement_equal_group_weight', 'Shared-question weighting (29 units)'),
    ('exclude_shared_amendment_or_draft_inputs', 'Exclude shared-instrument inputs (18 claims)'),
    ('exclude_four_outcome_wording_cases', 'Exclude outcome-wording cases (32 claims)'),
)


def generate(base, output):
    formal = base / '04_分析结果/正式研究'
    extra = base / '04_分析结果/新增模型对照_20260915'
    added = json.loads((extra / '新增解释联合指标.json').read_text())
    assert added['human_confirmed'] is True
    primary_rows = json.loads((formal / '正式逐条分析.json').read_text())
    added_groups = {g['method']: g for g in added['groups']}
    lines = [r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}', r'\toprule',
             r'Model & Prompt & Compatible & Error & Unclear & Correct label + error & Joint / 36 & Aligned joint / 36 \\', r'\midrule']
    for model, name in zip(MODELS, NAMES):
        if model == MODELS[-1]:
            lines.append(r'\midrule')
        for method in METHODS:
            if model != MODELS[-1]:
                rows = [r for r in primary_rows if r['model'] == model and r['method'] == method and r['split'] == 'heldout']
                assert len(rows) == 36
                status = Counter(r['rationale_status'] for r in rows)
                values = [status['consistent'], status['substantive_error'], status['indeterminate'],
                          sum(r['correct'] and r['rationale_status'] == 'substantive_error' for r in rows),
                          sum(r['joint_consistent'] for r in rows), sum(r['joint_consistent_aligned'] for r in rows)]
            else:
                g = added_groups[method]
                status = Counter(g['explanation_status_counts'])
                values = [status['consistent'], status['substantive_error'], status['indeterminate'],
                          g['correct_label_with_error'], g['joint_compatible_count'], g['joint_compatible_aligned_count']]
            prompt = 'Direct' if method == 'direct' else 'Checklist'
            lines.append(' & '.join([name, prompt] + [str(v) for v in values]) + r' \\')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'rationale.tex').write_text('\n'.join(lines + [r'\bottomrule', r'\end{tabular*}', '']))

    primary = json.loads((formal / '联合指标敏感性分析.json').read_text())['sensitivity']
    lookup = {(r['analysis'], r['model']): r['groups'] for r in primary}
    objective = {r['analysis']: r['groups'] for r in json.loads((extra / '客观结果汇总.json').read_text())['sensitivity']}
    joints = {r['analysis']: r['groups'] for r in added['sensitivity']}
    lines = [r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}', r'\toprule']
    panels = [('Label performance', ('Accuracy (\\%)', 'Macro-F1')),
              ('Joint performance', ('Joint (\\%)', 'Aligned joint (\\%)'))]
    for panel, headings in panels:
        if panel == 'Joint performance':
            lines.append(r'\midrule')
        lines.extend([r'\multicolumn{6}{@{}l}{\textit{' + panel + r'}} \\',
                      r'& & \multicolumn{2}{c}{Direct} & \multicolumn{2}{c}{Checklist} \\',
                      r'\cmidrule(lr){3-4}\cmidrule(l){5-6}',
                      'Analysis & Model & ' + ' & '.join(headings * 2) + r' \\', r'\midrule'])
        for analysis, label in VARIANTS:
            for i, (model, name) in enumerate(zip(MODELS, NAMES)):
                values = []
                for method in METHODS:
                    if model != MODELS[-1]:
                        g = lookup[analysis, model][method]
                    else:
                        g = {**objective[analysis][method],
                             'joint_rate': joints[analysis][method]['joint_compatible_rate'],
                             'aligned_joint_rate': joints[analysis][method]['joint_compatible_aligned_rate']}
                    if panel == 'Label performance':
                        values += [f'{100*g["accuracy"]:.1f}', f'{g["macro_f1"]:.3f}']
                    else:
                        values += [f'{100*g["joint_rate"]:.1f}', f'{100*g["aligned_joint_rate"]:.1f}']
                lines.append(' & '.join([label if i == 0 else '', name] + values) + r' \\')
    (output / 'sensitivity.tex').write_text('\n'.join(lines + [r'\bottomrule', r'\end{tabular*}', '']))
    print(json.dumps({'rationale_model_conditions': 6, 'sensitivity_data_rows': 18,
                      'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    generate(args.base, args.output_dir)
