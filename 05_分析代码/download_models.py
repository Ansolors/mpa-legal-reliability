"""下载公开 MLX 权重，逐文件校验 ModelScope 清单 SHA256，不执行仓库代码。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlencode
import hashlib
import json
import subprocess

BASE = Path(__file__).resolve().parents[1]
REPOS = ["mlx-community/Qwen3-4B-Instruct-2507-4bit", "mlx-community/Phi-4-mini-instruct-4bit"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get(item):
    repo, f = item
    path = BASE / "02_原始数据/模型权重" / repo.split("/")[-1] / f["Path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"repo": repo, "file": f["Path"], "expected_sha256": f["Sha256"],
              "file_revision": f.get("Revision"), "bytes_expected": f["Size"],
              "start_time_utc": datetime.now(timezone.utc).isoformat()}
    if path.exists() and sha(path) == f["Sha256"]:
        record["status"] = "existing_verified"
        return record
    url = "https://modelscope.cn/api/v1/models/" + repo + "/repo?" + urlencode({"Revision": "master", "FilePath": f["Path"]})
    part = path.with_suffix(path.suffix + ".partial")
    print("DOWNLOAD", repo, f["Path"], f["Size"], flush=True)
    result = subprocess.run(["curl", "-L", "--fail", "--connect-timeout", "10", "--max-time", "1200",
                             "--retry", "1", "-sS", "-o", str(part), url], capture_output=True)
    record.update({"url": url, "curl_code": result.returncode, "end_time_utc": datetime.now(timezone.utc).isoformat()})
    if result.returncode == 0 and part.exists() and sha(part) == f["Sha256"]:
        part.replace(path)
        record.update({"status": "downloaded_verified", "path": str(path.relative_to(BASE)), "bytes": path.stat().st_size})
    else:
        record.update({"status": "failed", "error": result.stderr.decode(errors="replace")[:500]})
    print(record["status"], repo, f["Path"], flush=True)
    return record


if __name__ == "__main__":
    items = []
    for repo in REPOS:
        data = json.loads((BASE / "02_原始数据/模型目录" / (repo.split("/")[-1] + "_files.json")).read_text())
        for f in data["Data"]["Files"]:
            if f["Path"].endswith((".json", ".safetensors", ".txt", ".jinja")) or f["Path"] == "README.md":
                items.append((repo, f))
    # 两个模型的权重下载可以同时进行；推理另行串行运行。
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(get, items))
    (BASE / "06_验证日志/模型下载校验.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if any(x["status"] == "failed" for x in result):
        raise SystemExit("部分模型文件未通过下载校验。")
