"""先导后补充诊断：仅剥除一个外层Markdown代码框，不改标签/理由/引用。严格原指标保持不变。"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json,re,statistics,hashlib

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'04_分析结果/模型先导实验'
LABELS=['SUPPORTED','CONTRADICTED','INSUFFICIENT']

def parse_readable(text):
    s=text.strip()
    m=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',s,re.S)
    stripped=bool(m)
    if m:s=m.group(1).strip()
    try:
        p=json.loads(s)
        if isinstance(p,dict) and p.get('label') in LABELS:return p,stripped
    except (TypeError,ValueError):pass
    return None,stripped

def main():
    import sys
    split='heldout' if '--heldout' in sys.argv else 'pilot'
    policy={'created_utc':datetime.now(timezone.utc).isoformat(),
      'status':'post-hoc diagnostic for pilot; fixed before heldout inference',
      'reason':'Phi produced outer Markdown code fences despite the identical JSON instruction; do not equate a wrapper with legal error',
      'operation':'strip whitespace and one whole-output outer ```json ... ``` or ``` ... ``` wrapper; standard JSON parse; exact uppercase labels only',
      'no_changes':['label value','reason text','evidence references','gold','input corpus','prompts'],
      'main_pilot_score':'original strict-JSON result remains in 先导结果摘要.json',
      'heldout_use':'readable label accuracy as substantive task metric; strict JSON and citation diagnostics separately'}
    path=OUT/'格式分离诊断协议.json'
    if not path.exists():path.write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
    if '--freeze-only' in sys.argv:print('格式分离规则已记录；未重新生成任何模型答案。');return
    gold={c['case_id']:c for c in json.loads((BASE/'04_分析结果/法规命题基准/cases_v1.json').read_text())}
    rows=[]
    for p in OUT.glob(f'*_{split}_raw.jsonl'):
        for s in p.read_text().splitlines():
            r=json.loads(s);c=gold[r['case_id']];v,strip=parse_readable(r['output'])
            pred=v['label'] if v and r['status']=='completed' else 'INVALID'
            rows.append({'model':r['model'],'method':r['method'],'case_id':r['case_id'],
             'mpa_id':c['mpa_id'],'family':c['family'],'gold':c['gold_label'],'prediction':pred,
             'correct':pred==c['gold_label'],'outer_fence_removed':strip,
             'reason':v.get('reason','') if v else '', 'evidence':v.get('evidence',[]) if v else [],
             'elapsed_seconds':r['elapsed_seconds'],'generation_tokens':r.get('generation_tokens'),
             'prompt_tokens':r.get('prompt_tokens'),'finish_reason':r.get('finish_reason')})
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['method'])].append(r)
    summary=[];paired=[]
    for (model,method),a in sorted(groups.items()):
        f1=[]
        for lab in LABELS:
            tp=sum(r['gold']==lab and r['prediction']==lab for r in a)
            fp=sum(r['gold']!=lab and r['prediction']==lab for r in a)
            fn=sum(r['gold']==lab and r['prediction']!=lab for r in a)
            f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        summary.append({'model':model,'method':method,'n':len(a),'correct':sum(r['correct'] for r in a),
          'accuracy':statistics.mean(r['correct'] for r in a),'macro_f1':statistics.mean(f1),
          'outer_fences':sum(r['outer_fence_removed'] for r in a),
          'readable_labels':sum(r['prediction'] in LABELS for r in a),
          'insufficient_case_n':sum(r['gold']=='INSUFFICIENT' for r in a),
          'unsupported_decisions':sum(r['gold']=='INSUFFICIENT' and r['prediction'] in LABELS[:2] for r in a),
          'over_abstentions':sum(r['gold']!='INSUFFICIENT' and r['prediction']=='INSUFFICIENT' for r in a),
          'median_seconds':statistics.median(r['elapsed_seconds'] for r in a),
          'total_generation_tokens':sum(r['generation_tokens'] or 0 for r in a),
          'confusion_matrix':{g:dict(Counter(r['prediction'] for r in a if r['gold']==g)) for g in LABELS}})
    for model in sorted(set(r['model'] for r in rows)):
        direct={r['case_id']:r for r in rows if r['model']==model and r['method']=='direct'}
        struct={r['case_id']:r for r in rows if r['model']==model and r['method']=='structured'}
        ids=sorted(direct.keys()&struct.keys())
        paired.append({'model':model,'n':len(ids),'improved':[i for i in ids if not direct[i]['correct'] and struct[i]['correct']],
           'worsened':[i for i in ids if direct[i]['correct'] and not struct[i]['correct']]})
    result={'scored_utc':datetime.now(timezone.utc).isoformat(),'split':split,'outputs':len(rows),
      'note':'先导后格式分离诊断；只剥除外层代码框，未纠正语义。不能称为先导预先设定的评分。',
      'groups':summary,'paired':paired,'policy_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/(split+'_格式分离后逐条评分.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (OUT/(split+'_格式分离后结果.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
