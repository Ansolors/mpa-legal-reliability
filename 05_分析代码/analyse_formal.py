"""Reproduce all finite-benchmark results from immutable outputs and the audit."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, statistics
from score_readable_labels import parse_readable, LABELS

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / '04_分析结果/正式研究'
RAW = BASE / '04_分析结果/模型先导实验'
DATA = BASE / '04_分析结果/法规命题基准'
MODEL_ORDER = ['Qwen3-4B-Instruct-2507-4bit', 'Phi-4-mini-instruct-4bit']
METHODS = ['direct', 'structured']


def metrics(rows, weights=None):
    weights = weights or {r['case_id']: 1.0 for r in rows}
    total = sum(weights[r['case_id']] for r in rows)
    def count(predicate):
        return sum(weights[r['case_id']] for r in rows if predicate(r))
    f1, recall = {}, {}
    for lab in LABELS:
        tp = count(lambda r: r['gold'] == lab and r['prediction'] == lab)
        fp = count(lambda r: r['gold'] != lab and r['prediction'] == lab)
        fn = count(lambda r: r['gold'] == lab and r['prediction'] != lab)
        f1[lab] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        recall[lab] = tp/(tp+fn) if tp+fn else None
    return {'n': len(rows), 'total_weight': total,
            'correct': count(lambda r: r['correct']),
            'accuracy': count(lambda r: r['correct'])/total,
            'macro_f1': statistics.mean(f1.values()), 'f1': f1, 'recall': recall}


def main():
    OUT.mkdir(exist_ok=True)
    frozen = json.loads((DATA/'publication_manifest.json').read_text())
    for name, sha in frozen['files'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == sha
    cases = {c['case_id']: c for c in json.loads((DATA/'cases_v1.json').read_text())}
    audit_obj = json.loads((OUT/'explanation_codes.json').read_text())
    audit = {(r['split'], r['case_id'], r['model'], r['method']): r for r in audit_obj['records']}
    assert len(audit) == len(audit_obj['records']) == 288
    rows = []
    for split in ['pilot', 'heldout']:
        for path in sorted(RAW.glob(f'*_{split}_raw.jsonl')):
            for line in path.read_text().splitlines():
                raw = json.loads(line); c = cases[raw['case_id']]
                key = (split, raw['case_id'], raw['model'], raw['method'])
                a = audit[key]
                p, fence = parse_readable(raw['output'])
                assert p is not None and raw['status'] == 'completed'
                ids = p.get('evidence', [])
                try:
                    strict = json.loads(raw['output'].strip())
                    valid_json = isinstance(strict, dict) and strict.get('label') in LABELS
                except ValueError:
                    valid_json = False
                ids_set = {x for x in ids if isinstance(x, str)} if isinstance(ids, list) else set()
                req = set(c['gold_required_evidence'])
                r = {k: a[k] for k in ['split','case_id','model','method','rationale_status','alignment','error_types','note']}
                r.update(mpa_id=c['mpa_id'], family=c['family'], gold=c['gold_label'],
                         prediction=p['label'], correct=p['label']==c['gold_label'],
                         reason=p.get('reason',''), evidence=ids, outer_fence=fence,
                         strict_json_label=valid_json,
                         exact_schema=(set(p)=={'label','reason','evidence'} and isinstance(p.get('reason'),str) and isinstance(ids,list)),
                         citation_ids_valid=isinstance(ids,list) and bool(ids) and all(isinstance(x,str) and x in c['evidence_ids'] for x in ids),
                         required_page_recall=len(ids_set & req)/len(req),
                         reason_word_count=len(p.get('reason','').split()),
                         shared_instrument=any(x.split(':')[0] in {'AMD','DRF'} for x in c['evidence_ids']),
                         prompt_tokens=raw['prompt_tokens'], generation_tokens=raw['generation_tokens'],
                         elapsed_seconds=raw['elapsed_seconds'], finish_reason=raw['finish_reason'])
                r['joint_consistent'] = r['correct'] and r['rationale_status']=='consistent'
                r['joint_consistent_aligned'] = r['joint_consistent'] and r['alignment']=='aligned'
                rows.append(r)
    assert len(rows) == len({(r['split'],r['case_id'],r['model'],r['method']) for r in rows}) == 288
    result = {'created_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'Descriptive results for the fixed constructed benchmark.',
              'case_hash':frozen['files']['cases_v1.json'],
              'rationale_rubric':{k:v for k,v in audit_obj.items() if k not in ['records','revisions']},
              'splits':{}}
    for split in ['pilot', 'heldout']:
        subset = [r for r in rows if r['split']==split]
        summaries, paired = [], []
        for model in MODEL_ORDER:
            for method in METHODS:
                a = [r for r in subset if r['model']==model and r['method']==method]
                assert len(a)==36
                summary = {'model':model, 'method':method, **metrics(a)}
                for name, pred in {
                    'unsupported_decisions':lambda r:r['gold']=='INSUFFICIENT' and r['prediction']!='INSUFFICIENT',
                    'over_abstentions':lambda r:r['gold']!='INSUFFICIENT' and r['prediction']=='INSUFFICIENT',
                    'strict_json_label':lambda r:r['strict_json_label'],
                    'exact_schema_after_wrapper':lambda r:r['exact_schema'],
                    'valid_citation_ids':lambda r:r['citation_ids_valid'],
                    'joint_consistent':lambda r:r['joint_consistent'],
                    'joint_consistent_aligned':lambda r:r['joint_consistent_aligned'],
                    'correct_with_substantive_error':lambda r:r['correct'] and r['rationale_status']=='substantive_error',
                    'correct_with_indeterminate_reason':lambda r:r['correct'] and r['rationale_status']=='indeterminate',
                    'word_limit_exceeded':lambda r:r['reason_word_count']>70,
                    'length_stops':lambda r:r['finish_reason']=='length',
                }.items(): summary[name] = sum(pred(r) for r in a)
                summary.update(rationale_counts=dict(Counter(r['rationale_status'] for r in a)),
                    alignment_counts=dict(Counter(r['alignment'] for r in a)),
                    error_type_counts=dict(Counter(t for r in a for t in r['error_types'])),
                    mean_required_page_recall=statistics.mean(r['required_page_recall'] for r in a),
                    median_seconds=statistics.median(r['elapsed_seconds'] for r in a),
                    prompt_tokens_range=[min(r['prompt_tokens'] for r in a),max(r['prompt_tokens'] for r in a)],
                    generation_tokens_range=[min(r['generation_tokens'] for r in a),max(r['generation_tokens'] for r in a)],
                    confusion_matrix={g:{p:sum(r['gold']==g and r['prediction']==p for r in a) for p in LABELS} for g in LABELS})
                summaries.append(summary)
            d = {r['case_id']:r for r in subset if r['model']==model and r['method']=='direct'}
            s = {r['case_id']:r for r in subset if r['model']==model and r['method']=='structured'}
            paired.append({'model':model,
                'improved':[i for i in d if not d[i]['correct'] and s[i]['correct']],
                'worsened':[i for i in d if d[i]['correct'] and not s[i]['correct']],
                'both_correct':sum(d[i]['correct'] and s[i]['correct'] for i in d),
                'both_incorrect':sum(not d[i]['correct'] and not s[i]['correct'] for i in d),
                'label_disagreements':[i for i in d if d[i]['prediction']!=s[i]['prediction']]})
        result['splits'][split]={'groups':summaries,'paired':paired}
    held = [r for r in rows if r['split']=='heldout']
    hc = [c for c in cases.values() if c['split']=='heldout']
    counts = Counter(c['family'] for c in hc)
    weights = {c['case_id']:1/counts[c['family']] if c['family'] in {'draft_status','effective_date'} else 1. for c in hc}
    result['shared_question_weights']=weights
    result['sensitivity']=[]
    for model in MODEL_ORDER:
        for name, predicate, w in [
            ('all_heldout',lambda r:True,None),
            ('shared_draft_and_commencement_equal_group_weight',lambda r:True,weights),
            ('exclude_shared_amendment_or_draft_inputs',lambda r:not r['shared_instrument'],None),
            ('exclude_four_outcome_wording_cases',lambda r:r['family']!='law_to_outcome_gap',None),
        ]:
            groups={method:metrics([r for r in held if r['model']==model and r['method']==method and predicate(r)],w) for method in METHODS}
            result['sensitivity'].append({'model':model,'analysis':name,'groups':groups,
                'accuracy_difference':groups['structured']['accuracy']-groups['direct']['accuracy'],
                'macro_f1_difference':groups['structured']['macro_f1']-groups['direct']['macro_f1']})
    result['leave_one_mpa_out']=[]
    result['by_mpa']=[]
    for model in MODEL_ORDER:
        for mpa in sorted({c['mpa_id'] for c in hc}):
            g={method:metrics([r for r in held if r['model']==model and r['method']==method and r['mpa_id']!=mpa]) for method in METHODS}
            result['leave_one_mpa_out'].append({'model':model,'excluded_mpa':mpa,'groups':g,
                'accuracy_difference':g['structured']['accuracy']-g['direct']['accuracy'],
                'macro_f1_difference':g['structured']['macro_f1']-g['direct']['macro_f1']})
            result['by_mpa'].append({'model':model,'mpa_id':mpa,'groups':{method:metrics([r for r in held if r['model']==model and r['method']==method and r['mpa_id']==mpa]) for method in METHODS}})
    result['case_composition']={s:dict(Counter(c['gold_label'] for c in cases.values() if c['split']==s)) for s in ['pilot','heldout']}
    result['heldout_family_counts']=dict(counts)
    for filename, obj in [('正式研究结果.json',result),('正式逐条分析.json',rows)]:
        (OUT/filename).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'heldout':result['splits']['heldout'],'sensitivity':result['sensitivity']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
