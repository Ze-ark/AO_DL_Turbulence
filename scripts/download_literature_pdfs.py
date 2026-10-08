from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "文献库" / "pdf"

DOWNLOADS = [
    ("01_Nousiainen_2021_ModelBased_RL_AO.pdf", "https://arxiv.org/pdf/2104.13685"),
    ("02_Nousiainen_2022_Toward_OnSky_AO_RL.pdf", "https://www.aanda.org/articles/aa/pdf/2022/08/aa43311-22.pdf"),
    ("03_Nousiainen_2024_Laboratory_ModelBased_RL_AO.pdf", "https://arxiv.org/pdf/2401.00242"),
    ("04_Guyon_2017_EOF_Predictive_Control.pdf", "https://arxiv.org/pdf/1707.00570"),
    ("05_Swanson_2021_ClosedLoop_Predictive_Control_CNN.pdf", "https://arxiv.org/pdf/2103.06327"),
    ("06_Fowler_2022_Battle_Predictive_Wavefront_Controls.pdf", "https://arxiv.org/pdf/2208.00984"),
    ("07_Chua_2018_Probabilistic_Dynamics_Models.pdf", "https://arxiv.org/pdf/1805.12114"),
    ("08_Janner_2019_ModelBased_Policy_Optimization.pdf", "https://arxiv.org/pdf/1906.08253"),
    ("09_Haffert_2021_DataDriven_Subspace_Predictive_Control.pdf", "https://arxiv.org/pdf/2103.07566"),
    ("10_Back_to_Newtons_Laws_2024.pdf", "https://arxiv.org/pdf/2407.10648"),
    ("11_Nousiainen_2026_OnSky_RL_AO.pdf", "https://arxiv.org/pdf/2606.10771"),
    ("12_Haffert_2026_SelfLearning_Predictive_Control.pdf", "https://arxiv.org/pdf/2608.24339"),
    ("13_Parvizi_2023_WFSless_AO_RL_Environment.pdf", "https://www.mdpi.com/2304-6732/10/12/1371/pdf"),
    ("14_Seifi_2025_Beacon_Feedback_RL_SLM_FSO.pdf", "https://www.mdpi.com/2304-6732/12/10/979/pdf"),
    ("15_Durech_2021_WFSless_AO_Deep_RL.pdf", "https://europepmc.org/articles/PMC8515990?pdf=render"),
    ("16_Xu_2024_MOSS_DDPG_Sensorless_AO.pdf", "https://europepmc.org/articles/PMC11427189?pdf=render"),
    ("17_SAC_Haarnoja_2018.pdf", "https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf"),
    ("18_CPO_Achiam_2017.pdf", "https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf"),
    ("19_Pou_2022_AO_MARL.pdf", "https://upcommons.upc.edu/server/api/core/bitstreams/26f40080-0c75-464b-a0a7-4b7b5988f09a/content"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, url in DOWNLOADS:
        path = OUT / filename
        status = "existing"
        error = ""
        try:
            if not path.exists() or path.stat().st_size < 10_000:
                request = urllib.request.Request(url, headers={"User-Agent": "AO-DL-Turbulence-literature/1.0"})
                with urllib.request.urlopen(request, timeout=12) as response, path.open("wb") as f:
                    f.write(response.read())
                status = "downloaded"
            if path.stat().st_size < 10_000:
                status = "rejected_small"
                path.unlink(missing_ok=True)
        except Exception as exc:  # network/source-specific failure is recorded, not hidden
            status = "failed"
            error = str(exc)
            path.unlink(missing_ok=True)
        record = {"filename": filename, "url": url, "status": status}
        if path.exists():
            record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        if error:
            record["error"] = error
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    (OUT.parent / "下载结果.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
