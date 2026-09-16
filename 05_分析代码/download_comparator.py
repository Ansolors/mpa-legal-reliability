"""Download the prespecified larger local comparator with pinned fingerprints."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlencode
import hashlib
import json
import subprocess
import urllib.request

BASE = Path(__file__).resolve().parents[1]
REPO = 'mlx-community/Qwen3-14B-4bit'
REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
META = BASE / '02_原始数据/模型目录'
DEST = BASE / '02_原始数据/模型权重/Qwen3-14B-4bit'
LOG = BASE / '06_验证日志/新增模型对照_20260915'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def metadata(filename, url):
    path = META / filename
    if path.exists():
        return json.loads(path.read_text())
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
    data = json.loads(raw)
    path.write_bytes(raw)
    return data


def download(item):
    filename, ms = item
    path = DEST / filename
    record = {'file': filename, 'expected_sha256': ms['Sha256'], 'bytes_expected': ms['Size'],
              'huggingface_revision': REVISION, 'modelscope_revision': ms['Revision'], 'start_utc': now()}
    if path.exists() and path.stat().st_size == ms['Size'] and sha(path) == ms['Sha256']:
        record.update(status='existing_verified', end_utc=now())
        return record
    partial = path.with_suffix(path.suffix + '.partial')
    urls = [
        'https://modelscope.cn/api/v1/models/' + REPO + '/repo?' +
        urlencode({'Revision': ms['Revision'], 'FilePath': filename}),
        f'https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}',
    ]
    record['attempts'] = []
    for url in urls:
        print('DOWNLOAD', filename, ms['Size'], flush=True)
        result = subprocess.run(
            ['curl', '-L', '--fail', '--connect-timeout', '15', '--max-time', '3600',
             '--retry', '2', '--retry-delay', '2', '--speed-time', '60', '--speed-limit', '1024',
             '--continue-at', '-', '-sS', '-o', str(partial), url], capture_output=True)
        attempt = {'url': url, 'exit_code': result.returncode, 'end_utc': now(),
                   'error': result.stderr.decode(errors='replace')[-500:]}
        record['attempts'].append(attempt)
        if result.returncode == 0 and partial.exists() and partial.stat().st_size == ms['Size']:
            attempt['actual_sha256'] = sha(partial)
            if attempt['actual_sha256'] == ms['Sha256']:
                partial.replace(path)
                record.update(status='downloaded_verified', end_utc=now())
                print('VERIFIED', filename, flush=True)
                return record
        if partial.exists() and partial.stat().st_size >= ms['Size']:
            partial.rename(partial.with_suffix(partial.suffix + '.failed'))
    record.update(status='failed', end_utc=now())
    return record


def main():
    for path in [META, DEST, LOG]:
        path.mkdir(parents=True, exist_ok=True)
    hf_url = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    ms_url = f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true'
    hf = metadata('Qwen3-14B-4bit_hf_metadata.json', hf_url)
    ms = metadata('Qwen3-14B-4bit_files.json', ms_url)
    assert hf['sha'] == REVISION
    assert ms['Success'] and ms['Code'] == 200
    files = {f['Path']: f for f in ms['Data']['Files']}
    selected = []
    for f in hf['siblings']:
        name = f['rfilename']
        if name.startswith('.') or not name.endswith(('.json', '.txt', '.safetensors', '.jinja', '.md')):
            continue
        counterpart = files[name]
        assert f['size'] == counterpart['Size'], name
        if f.get('lfs'):
            assert f['lfs']['sha256'] == counterpart['Sha256'], name
        selected.append((name, counterpart))
    # Small configuration and tokenizer files precede the two independent shards.
    selected.sort(key=lambda x: x[1]['Size'])
    print('PINNED', REVISION, 'FILES', len(selected), 'BYTES', sum(f['Size'] for _, f in selected), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(download, selected))
    (LOG / '下载校验.json').write_text(json.dumps({
        'created_utc': now(), 'repository': REPO, 'revision': REVISION,
        'metadata_urls': [hf_url, ms_url], 'files': results,
    }, ensure_ascii=False, indent=2) + '\n')
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit('Some files failed verification; inference has not started.')
    print('ALL_FILES_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
