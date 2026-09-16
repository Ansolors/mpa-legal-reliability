"""Objective comparator scores; explanation and joint scores require human coding."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import statistics

from analyse_formal import metrics
from score_readable_labels import LABELS, parse_readable
from run_pilot import messages

BASE = Path(__file__).resolve().parents[1]
MODEL = 'Qwen3-14B-4bit'
METHODS = ['direct', 'structured']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def performance_table(primary, comparator):
    names = {'Qwen3-4B-Instruct-2507-4bit':'Qwen3-4B','Phi-4-mini-instruct-4bit':'Phi-4-mini',MODEL:'Qwen3-14B'}
    lines = [r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}',r'\toprule',
             r'Model & Prompt & Correct / 36 & Accuracy (\%) & Macro-F1 & U / 9 & O / 27 & Valid IDs / 36 \\',r'\midrule']
    for i,g in enumerate(primary+comparator):
        if i == len(primary):
            lines.append(r'\midrule')
        prompt = 'Direct' if g['method']=='direct' else 'Checklist'
        lines.append(f"{names[g['model']]} & {prompt} & {g['correct']:.0f} & {100*g['accuracy']:.1f} & {g['macro_f1']:.3f} & {g['unsupported_decisions']} & {g['over_abstentions']} & {g['valid_citation_ids']} " + r'\\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular*}'])+'\n'


def analyse(base):
    out = base / '04_分析结果/新增模型对照_20260915'
    data = base / '04_分析结果/法规命题基准'
    protocol_path = out / '固定新增实验协议.json'
    protocol = json.loads(protocol_path.read_text())
    frozen = json.loads((data / 'publication_manifest.json').read_text())
    for name, value in frozen['files'].items():
        assert digest(data / name) == value
    cases = {c['case_id']: c for c in json.loads((data/'cases_v1.json').read_text()) if c['split']=='heldout'}
    evidence = json.loads((data/'evidence_v1.json').read_text())
    path = out / (MODEL + '_heldout_raw.jsonl')
    raw = [json.loads(s) for s in path.read_text().splitlines()]
    assert len(raw) == len({(r['case_id'],r['method']) for r in raw}) == 72
    assert {(r['case_id'],r['method']) for r in raw} == {(c,m) for c in cases for m in METHODS}
    rows = []
    for r in raw:
        c = cases[r['case_id']]
        assert r['model'] == MODEL and r['split'] == 'heldout'
        assert r['protocol_sha256'] == digest(protocol_path)
        assert r['messages'] == messages(c, evidence, r['method'])
        assert r['prompt_sha256'] == hashlib.sha256(r['rendered_prompt'].encode()).hexdigest()
        assert protocol['created_utc'] < r['start_utc']
        p, fence = parse_readable(r['output'])
        readable = p is not None and r['status'] == 'completed'
        prediction = p['label'] if readable else 'INVALID'
        ids = p.get('evidence', []) if readable else []
        reason = p.get('reason', '') if readable else ''
        ids_set = {x for x in ids if isinstance(x,str)} if isinstance(ids,list) else set()
        required = set(c['gold_required_evidence'])
        try:
            strict = json.loads(r['output'].strip())
            strict_json = r['status'] == 'completed' and isinstance(strict,dict) and strict.get('label') in LABELS
        except ValueError:
            strict_json = False
        rows.append({
            'case_id':r['case_id'], 'method':r['method'], 'model':MODEL, 'split':'heldout',
            'mpa_id':c['mpa_id'], 'family':c['family'], 'gold':c['gold_label'],
            'prediction':prediction, 'correct':prediction==c['gold_label'],
            'status':r['status'], 'outer_fence':fence, 'strict_json_label':strict_json,
            'exact_schema':readable and set(p)=={'label','reason','evidence'} and isinstance(reason,str) and isinstance(ids,list),
            'citation_ids_valid':isinstance(ids,list) and bool(ids) and all(isinstance(x,str) and x in c['evidence_ids'] for x in ids),
            'required_page_recall':len(ids_set & required)/len(required),
            'reason':reason, 'evidence':ids,
            'reason_word_count':len(reason.split()) if isinstance(reason,str) else None,
            'shared_instrument':any(x.split(':')[0] in {'AMD','DRF'} for x in c['evidence_ids']),
            **{k:r.get(k) for k in ['prompt_tokens','generation_tokens','elapsed_seconds','finish_reason','peak_memory_gb']},
        })
    groups = []
    for method in METHODS:
        subset = [r for r in rows if r['method']==method]
        group = {'model':MODEL, 'method':method, **metrics(subset)}
        for key, predicate in {
            'invalid_outputs':lambda r:r['prediction']=='INVALID',
            'unsupported_decisions':lambda r:r['gold']=='INSUFFICIENT' and r['prediction']!='INSUFFICIENT',
            'valid_unsupported_decisions':lambda r:r['gold']=='INSUFFICIENT' and r['prediction'] in LABELS[:2],
            'over_abstentions':lambda r:r['gold']!='INSUFFICIENT' and r['prediction']=='INSUFFICIENT',
            'strict_json_label':lambda r:r['strict_json_label'],
            'exact_schema_after_wrapper':lambda r:r['exact_schema'],
            'valid_citation_ids':lambda r:r['citation_ids_valid'],
            'outer_fences':lambda r:r['outer_fence'],
            'word_limit_exceeded':lambda r:r['reason_word_count'] is not None and r['reason_word_count']>70,
            'length_stops':lambda r:r['finish_reason']=='length',
        }.items():
            group[key] = sum(predicate(r) for r in subset)
        group.update(mean_required_page_recall=statistics.mean(r['required_page_recall'] for r in subset),
                     median_seconds=statistics.median(r['elapsed_seconds'] for r in subset),
                     total_generation_tokens=sum(r['generation_tokens'] or 0 for r in subset),
                     confusion_matrix={g:dict(Counter(r['prediction'] for r in subset if r['gold']==g)) for g in LABELS})
        for key in ['prompt_tokens','generation_tokens','peak_memory_gb']:
            values = [r[key] for r in subset if r[key] is not None]
            group[key+'_range'] = [min(values),max(values)] if values else None
        groups.append(group)
    direct = {r['case_id']:r for r in rows if r['method']=='direct'}
    checklist = {r['case_id']:r for r in rows if r['method']=='structured'}
    paired = {
        'improved':[c for c in cases if not direct[c]['correct'] and checklist[c]['correct']],
        'worsened':[c for c in cases if direct[c]['correct'] and not checklist[c]['correct']],
        'both_correct':sum(direct[c]['correct'] and checklist[c]['correct'] for c in cases),
        'both_incorrect':sum(not direct[c]['correct'] and not checklist[c]['correct'] for c in cases),
        'label_disagreements':[c for c in cases if direct[c]['prediction']!=checklist[c]['prediction']],
    }
    counts = Counter(c['family'] for c in cases.values())
    weights = {c['case_id']:1/counts[c['family']] if c['family'] in {'draft_status','effective_date'} else 1. for c in cases.values()}
    sensitivity = []
    variants = [
        ('all_heldout',lambda r:True,None),
        ('shared_draft_and_commencement_equal_group_weight',lambda r:True,weights),
        ('exclude_shared_amendment_or_draft_inputs',lambda r:not r['shared_instrument'],None),
        ('exclude_four_outcome_wording_cases',lambda r:r['family']!='law_to_outcome_gap',None),
    ]
    variants += [('exclude_mpa_'+mpa,lambda r,m=mpa:r['mpa_id']!=m,None) for mpa in sorted({c['mpa_id'] for c in cases.values()})]
    for name, predicate, weight in variants:
        g = {m:metrics([r for r in rows if r['method']==m and predicate(r)],weight) for m in METHODS}
        sensitivity.append({'analysis':name,'groups':g,'accuracy_difference':g['structured']['accuracy']-g['direct']['accuracy'],
                            'macro_f1_difference':g['structured']['macro_f1']-g['direct']['macro_f1']})
    result = {
        'created_utc':datetime.now(timezone.utc).isoformat(), 'model':MODEL,
        'scope':'Objective comparison on the original 36 heldout claims; these cases were already analysed for the two original models before selecting this supplementary comparator.',
        'analysis_scope':'Label, format and source-reference metrics; explanation and joint results are provided separately.',
        'outputs':len(rows), 'case_composition':dict(Counter(c['gold_label'] for c in cases.values())),
        'groups':groups, 'paired':paired, 'shared_question_weights':weights, 'sensitivity':sensitivity,
        'permit_issuance_cases':[r for r in rows if r['case_id'] in {'TMN08','STI08','PON08','AMA08'}],
        'input_sha256':{str(p.relative_to(base)):digest(p) for p in [path, protocol_path, data/'cases_v1.json', data/'evidence_v1.json', data/'publication_manifest.json']},
    }
    for filename, value in [('客观逐条评分.json',rows),('客观结果汇总.json',result)]:
        (out/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'outputs':len(rows),'groups':groups,'paired':paired},ensure_ascii=False,indent=2))
    return result, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--table-output',type=Path)
    args = parser.parse_args()
    summary,_ = analyse(args.base)
    if args.table_output:
        primary = json.loads((args.base/'04_分析结果/正式研究/正式研究结果.json').read_text())['splits']['heldout']['groups']
        args.table_output.parent.mkdir(parents=True,exist_ok=True)
        args.table_output.write_text(performance_table(primary,summary['groups']))
