"""固定规则评分；标签正确性、JSON 格式和页码引用分开，不用第二个 LLM 当裁判。"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib,json,statistics

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'04_分析结果/模型先导实验'
LABELS=['SUPPORTED','CONTRADICTED','INSUFFICIENT']
POLICY={
 'primary':'strict JSON parse; exact label; generation errors, invalid JSON or label count as incorrect',
 'macro_f1':'arithmetic mean of label F1 over the three prespecified labels; invalid prediction acts as false negative for gold label',
 'citation_id_validity':'all returned IDs must refer to supplied pages; at least one ID required; this is identifier validity, NOT semantic citation correctness',
 'required_page_recall':'fraction of prelisted necessary source-page IDs cited; diagnostic, not a substitute for substantive answer auditing',
 'unsupported_decision':'SUPPORTED or CONTRADICTED on a gold INSUFFICIENT case',
 'over_abstention':'INSUFFICIENT on a determinate gold case',
 'inferential_scope':'finite-case descriptive pilot only; no significance claim, no national/global error-rate inference; cases share MPA regulations and a cross-MPA amendment',
}

def freeze():
 p=OUT/'评分规则.json'
 value={**POLICY,'frozen_utc':datetime.now(timezone.utc).isoformat()}
 if p.exists():
  old=json.loads(p.read_text())
  for k,v in POLICY.items():assert old[k]==v
 else:p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
 freeze()
 gold={c['case_id']:c for c in json.loads((BASE/'04_分析结果/法规命题基准/cases_v1.json').read_text())}
 details=[]
 for p in OUT.glob('*_pilot_raw.jsonl'):
  for line in p.read_text().splitlines():
   r=json.loads(line);c=gold[r['case_id']]
   parsed=None;valid=False;prediction='INVALID';ids=[]
   try:
    parsed=json.loads(r['output'].strip())
    valid=isinstance(parsed,dict) and parsed.get('label') in LABELS
    if valid:prediction=parsed['label']
    ids=parsed.get('evidence',[]) if isinstance(parsed,dict) else []
   except (ValueError,TypeError):pass
   if r['status']!='completed':valid=False;prediction='INVALID'
   ids_valid=isinstance(ids,list) and bool(ids) and all(isinstance(x,str) and x in c['evidence_ids'] for x in ids)
   idset=set(x for x in ids if isinstance(x,str)) if isinstance(ids,list) else set()
   required=set(c['gold_required_evidence'])
   details.append({'case_id':r['case_id'],'mpa_id':c['mpa_id'],'family':c['family'],
      'model':r['model'],'method':r['method'],'gold':c['gold_label'],'prediction':prediction,
      'correct':prediction==c['gold_label'],'json_label_valid':valid,'citation_ids_valid':ids_valid,
      'required_page_recall':len(idset&required)/len(required),
      'unsupported_decision':c['gold_label']=='INSUFFICIENT' and prediction in LABELS[:2],
      'over_abstention':c['gold_label']!='INSUFFICIENT' and prediction=='INSUFFICIENT',
      'false_endorsement':c['gold_label']=='CONTRADICTED' and prediction=='SUPPORTED',
      'false_rejection':c['gold_label']=='SUPPORTED' and prediction=='CONTRADICTED',
      'elapsed_seconds':r['elapsed_seconds'],'generation_tokens':r.get('generation_tokens'),
      'prompt_tokens':r.get('prompt_tokens'),'finish_reason':r.get('finish_reason'),
      'output':r['output'],'gold_basis_zh':c['gold_basis_zh'],'claim':c['claim'],
      'facts':c['facts'],'evidence_ids':c['evidence_ids']})
 groups=defaultdict(list)
 for r in details:groups[(r['model'],r['method'])].append(r)
 summary=[]
 for (model,method),rows in sorted(groups.items()):
  confusion={g:{v:sum(r['gold']==g and r['prediction']==v for r in rows) for v in LABELS+['INVALID']} for g in LABELS}
  f1=[]
  for lab in LABELS:
   tp=sum(r['gold']==lab and r['prediction']==lab for r in rows)
   fp=sum(r['gold']!=lab and r['prediction']==lab for r in rows)
   fn=sum(r['gold']==lab and r['prediction']!=lab for r in rows)
   f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
  summary.append({'model':model,'method':method,'n':len(rows),'correct':sum(r['correct'] for r in rows),
    'accuracy':statistics.mean(r['correct'] for r in rows),'macro_f1':statistics.mean(f1),
    'valid_json_label':sum(r['json_label_valid'] for r in rows),
    'valid_citation_ids':sum(r['citation_ids_valid'] for r in rows),
    'mean_required_page_recall':statistics.mean(r['required_page_recall'] for r in rows),
    'unsupported_decisions':sum(r['unsupported_decision'] for r in rows),
    'gold_insufficient_n':sum(r['gold']=='INSUFFICIENT' for r in rows),
    'over_abstentions':sum(r['over_abstention'] for r in rows),
    'false_endorsements':sum(r['false_endorsement'] for r in rows),
    'false_rejections':sum(r['false_rejection'] for r in rows),
    'length_stops':sum(r['finish_reason']=='length' for r in rows),
    'median_seconds':statistics.median(r['elapsed_seconds'] for r in rows),
    'total_generation_tokens':sum(r['generation_tokens'] or 0 for r in rows),
    'max_prompt_tokens':max(r['prompt_tokens'] or 0 for r in rows),
    'confusion_matrix':confusion})
 pairs=[]
 for model in sorted(set(r['model'] for r in details)):
  direct={r['case_id']:r for r in details if r['model']==model and r['method']=='direct'}
  structured={r['case_id']:r for r in details if r['model']==model and r['method']=='structured'}
  ids=sorted(direct.keys()&structured.keys())
  pairs.append({'model':model,'paired_cases':len(ids),
    'improved':sum(not direct[i]['correct'] and structured[i]['correct'] for i in ids),
    'worsened':sum(direct[i]['correct'] and not structured[i]['correct'] for i in ids),
    'both_correct':sum(direct[i]['correct'] and structured[i]['correct'] for i in ids),
    'both_wrong':sum(not direct[i]['correct'] and not structured[i]['correct'] for i in ids)})
 result={'scored_utc':datetime.now(timezone.utc).isoformat(),'total_outputs':len(details),'groups':summary,'paired':pairs,
   'heldout_run_count':sum(len(p.read_text().splitlines()) for p in OUT.glob('*_heldout_raw.jsonl')),
   'policy_sha256':hashlib.sha256((OUT/'评分规则.json').read_bytes()).hexdigest()}
 (OUT/'逐条评分.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
 (OUT/'先导结果摘要.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (OUT/'错误与格式异常.json').write_text(json.dumps([r for r in details if not r['correct'] or not r['citation_ids_valid']],ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
 import sys
 if '--freeze-only' in sys.argv:freeze();print('评分规则已固定，尚未读取模型答案。')
 else:main()
