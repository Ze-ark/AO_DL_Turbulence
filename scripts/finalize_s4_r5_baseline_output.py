"""封存已完成但缺少收尾元数据的 R5-1 输出，不重新运行仿真。"""
from pathlib import Path
import json, hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/s4_r5_baseline_selection_v1"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    summary = OUT / "summary.json"
    metrics = OUT / "candidate_metrics.pt"
    selection = OUT / "selection.json"
    if not summary.exists() or not metrics.exists() or not selection.exists():
        raise FileNotFoundError("R5-1 completed output is incomplete")
    artifacts = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "SUCCESS.json"}:
            artifacts[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
    manifest = OUT / "artifact_manifest.json"
    manifest.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["artifact_manifest_sha256"] = sha(manifest)
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "SUCCESS.json").write_text(json.dumps({"summary_sha256": sha(summary), "artifact_manifest_sha256": sha(manifest)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FINALIZED_READ_ONLY", "summary_sha256": sha(summary), "artifact_manifest_sha256": sha(manifest)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
