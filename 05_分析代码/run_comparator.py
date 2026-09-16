"""Pinned larger-model extension; never writes to the original experiment directory."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time

from run_pilot import COMMON, METHODS, messages

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / '04_分析结果/法规命题基准'
OUT = BASE / '04_分析结果/新增模型对照_20260915'
LOG = BASE / '06_验证日志/新增模型对照_20260915'
MODEL = 'Qwen3-14B-4bit'
REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
MODEL_PATH = BASE / '02_原始数据/模型权重' / MODEL
SHARDS = {
    'model-00001-of-00002.safetensors': '5795efcfc7c96fd273e600562e8b111bfcc427415de9001d0a07e70cd99cff19',
    'model-00002-of-00002.safetensors': '2814562d654fe2d541fd4682804a0ccaa400e79701872c8e9f5998cf9481fdf8',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def freeze_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((DATA / 'publication_manifest.json').read_text())
    for name, digest in frozen['files'].items():
        assert sha(DATA / name) == digest, name
    cases = [c for c in json.loads((DATA / 'cases_v1.json').read_text()) if c['split'] == 'heldout']
    assert len(cases) == 36
    original = json.loads((BASE / '04_分析结果/模型先导实验/固定实验协议.json').read_text())
    assert original['common'] == COMMON and original['methods'] == METHODS
    protocol = {
        'created_utc': now(), 'status': 'fixed before any comparator benchmark response',
        'design': 'supplementary larger-model comparison, selected after analysis of the original two-model experiment',
        'model': MODEL, 'repository': 'mlx-community/' + MODEL, 'revision': REVISION,
        'weight_sha256': SHARDS, 'split': 'heldout', 'case_ids': [c['case_id'] for c in cases],
        'expected_outputs': 72, 'common': COMMON, 'methods': METHODS,
        'max_new_tokens': 256, 'temperature': 0, 'seed': 20260910,
        'enable_thinking': False, 'prefill_step_size': 256,
        'mlx_cache_limit_bytes': 256 * 1024**2, 'mlx_memory_limit_bytes': 11 * 1024**3,
        'input_policy': 'Original messages() function, supplied pages and facts; no reference labels, prior outputs or audit results in model inputs.',
        'order': 'Original heldout case order; direct then structured for even zero-based case index, reversed for odd index.',
        'generation': 'Greedy argmax; new cache for every response; no KV quantization; no context truncation.',
        'comparability': 'Larger parameter-scale implementation with a different Qwen3 release; not a controlled causal estimate of parameter count, and not a frontier-model test.',
        'parse_policy': 'Original parse_readable: strip whitespace and at most one complete outer Markdown fence; exact uppercase label and standard JSON parsing; no semantic repair.',
        'score_policy': {
            'denominator': 'All 36 scheduled cases per condition; failed or unparseable outputs are INVALID and incorrect.',
            'macro_f1': 'Unweighted mean across the same three fixed labels; INVALID outputs contribute false negatives.',
            'unsupported_decisions': 'Original article formula: reference INSUFFICIENT and prediction != INSUFFICIENT. Invalid outputs are additionally reported separately; valid determinate responses are counted separately.',
            'over_abstentions': 'Reference SUPPORTED or CONTRADICTED and prediction INSUFFICIENT.',
            'diagnostics': 'Strict JSON, schema, supplied-page ID validity, required-page recall, reason length and length stops use original definitions.',
            'sensitivity': 'Original shared-question weights, shared-input exclusion, four outcome-case exclusion and leave-one-MPA-out rules.',
            'human_coding': 'New explanations require separate independent review. No rationale compatibility or joint score is assigned by this runner or objective scoring.',
        },
        'retry_policy': 'Persist every attempted benchmark response, including errors; resume only unattempted case-condition pairs. No selective reruns.',
        'gold_frozen_sha256': sha(DATA / '冻结清单.json'),
        'original_protocol_sha256': sha(BASE / '04_分析结果/模型先导实验/固定实验协议.json'),
        'source_runner_sha256': sha(Path(__file__).with_name('run_pilot.py')),
        'parser_sha256': sha(Path(__file__).with_name('score_readable_labels.py')),
    }
    path = OUT / '固定新增实验协议.json'
    if path.exists():
        old = json.loads(path.read_text())
        # Archived code hashes identify the original runners; the public copy
        # validates the publication manifest while retaining the same experiment.
        provenance_keys = {'created_utc', 'source_runner_sha256'}
        assert {k:v for k,v in old.items() if k not in provenance_keys} == {k:v for k,v in protocol.items() if k not in provenance_keys}
    else:
        write_json(path, protocol)
    return path, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    proto, cases = freeze_protocol()
    if args.freeze_only:
        print('PROTOCOL_FROZEN', sha(proto), flush=True)
        return
    download = json.loads((LOG / '下载校验.json').read_text())
    assert download['revision'] == REVISION
    assert all(r['status'] in {'existing_verified', 'downloaded_verified'} for r in download['files'])
    fingerprints = {r['file']: sha(MODEL_PATH / r['file']) for r in download['files']}
    assert all(fingerprints[r['file']] == r['expected_sha256'] for r in download['files'])
    assert all(fingerprints[name] == digest for name, digest in SHARDS.items())
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(11 * 1024**3)
    mx.random.seed(20260910)
    run_meta = {
        'start_utc': now(), 'model': MODEL, 'repository_revision': REVISION,
        'model_file_sha256': fingerprints, 'protocol_sha256': sha(proto),
        'runner_sha256': sha(Path(__file__)), 'platform': platform.platform(),
        'python': platform.python_version(), 'device': mx.device_info(),
        'packages': {p: importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers','tokenizers','safetensors']},
    }
    invocation = run_meta['start_utc'].replace(':', '').replace('+', '_')
    meta_path = LOG / ('运行_' + invocation + '.json')
    write_json(meta_path, run_meta)
    print('LOADING', MODEL, flush=True)
    model, tokenizer = load(str(MODEL_PATH), tokenizer_config={'trust_remote_code': False})
    sampler = make_sampler(temp=0)
    def render(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    smoke_path = OUT / '中立运行检查.json'
    if not smoke_path.exists():
        prompt = render([{'role':'user','content':'Return only the integer result of 2 + 3.'}])
        start = time.monotonic()
        chunks = list(stream_generate(model, tokenizer, prompt=prompt, max_tokens=24,
                                      sampler=sampler, prefill_step_size=256))
        output = ''.join(x.text for x in chunks)
        write_json(smoke_path, {'created_utc': now(), 'purpose': 'Neutral arithmetic runtime check; excluded from all benchmark results.',
                               'prompt': prompt, 'output': output, 'elapsed_seconds': time.monotonic()-start,
                               'protocol_sha256': sha(proto), 'runner_sha256': sha(Path(__file__))})
        print('NEUTRAL_SMOKE', repr(output), flush=True)
        assert output.strip() == '5', 'Runtime check failed; benchmark has not started.'
    else:
        assert json.loads(smoke_path.read_text())['output'].strip() == '5'
    mx.clear_cache()
    mx.random.seed(20260910)
    evidence = json.loads((DATA / 'evidence_v1.json').read_text())
    path = OUT / (MODEL + '_heldout_raw.jsonl')
    existing = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {(r['case_id'], r['method']) for r in existing}
    assert len(done) == len(existing)
    assert all(r['protocol_sha256'] == sha(proto) and r['model'] == MODEL for r in existing)
    n_done = len(done)
    with path.open('a') as f:
        for i, case in enumerate(cases):
            for method in (['direct','structured'] if i % 2 == 0 else ['structured','direct']):
                if (case['case_id'], method) in done:
                    continue
                msgs = messages(case, evidence, method)
                prompt = render(msgs)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                assert len(prompt_ids) + 256 <= 40960, 'Input must not be truncated.'
                rec = {'case_id':case['case_id'], 'method':method, 'model':MODEL, 'split':'heldout',
                       'start_utc':now(), 'messages':msgs, 'rendered_prompt':prompt,
                       'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                       'protocol_sha256':sha(proto), 'run_metadata_file':meta_path.name}
                start = time.monotonic()
                raw, last = '', None
                mx.reset_peak_memory()
                print('START', n_done+1, '/72', case['case_id'], method, 'input_tokens', len(prompt_ids), flush=True)
                try:
                    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=256,
                                                    sampler=sampler, prefill_step_size=256):
                        raw += response.text
                        last = response
                    assert last is not None
                    rec.update(status='completed', output=raw, prompt_tokens=last.prompt_tokens,
                               generation_tokens=last.generation_tokens, prompt_tps=last.prompt_tps,
                               generation_tps=last.generation_tps, peak_memory_gb=last.peak_memory,
                               finish_reason=last.finish_reason)
                except Exception as exc:
                    rec.update(status='error', output=raw, error=repr(exc))
                rec.update(end_utc=now(), elapsed_seconds=time.monotonic()-start)
                f.write(json.dumps(rec, ensure_ascii=False)+'\n')
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                print('SAVED', n_done, '/72', case['case_id'], method, rec['status'],
                      round(rec['elapsed_seconds'], 2), 'seconds', flush=True)
                mx.clear_cache()
    run_meta.update(end_utc=now(), raw_file_sha256=sha(path), total_saved_outputs=n_done)
    write_json(meta_path, run_meta)
    write_json(OUT / '运行元数据.json', run_meta)
    assert n_done == 72
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    print('COMPARATOR_COMPLETE', n_done, flush=True)


if __name__ == '__main__':
    main()
