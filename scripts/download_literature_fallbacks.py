from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "文献库" / "pdf"

SOURCES = [
    ("02_Nousiainen_2022_Toward_OnSky_AO_RL.pdf", "https://arxiv.org/pdf/2205.07554"),
    ("13_Parvizi_2023_WFSless_AO_RL_Environment.pdf", "https://mdpi-res.com/d_attachment/photonics/photonics-10-01371/article_deploy/photonics-10-01371-v2.pdf"),
    ("14_Seifi_2025_Beacon_Feedback_RL_SLM_FSO.pdf", "https://mdpi-res.com/d_attachment/photonics/photonics-12-00979/article_deploy/photonics-12-00979-v1.pdf"),
    ("15_Durech_2021_WFSless_AO_Deep_RL.pdf", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8515990/pdf/boe-12-9-5423.pdf"),
    ("16_Xu_2024_MOSS_DDPG_Sensorless_AO.pdf", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11427189/pdf/boe-15-8-4795.pdf"),
    ("19_Pou_2022_AO_MARL.pdf", "https://upcommons.upc.edu/server/api/core/bitstreams/26f40080-0c75-464b-a0a7-4b7b5988f09a/content"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context()
    for name, url in SOURCES:
        path = OUT / name
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 AO-DL-Turbulence/1.0", "Accept": "application/pdf,*/*"})
            with urllib.request.urlopen(request, timeout=20, context=context) as response:
                data = response.read()
            if len(data) < 10_000 or not data.startswith(b"%PDF"):
                print(f"REJECT {name} bytes={len(data)} header={data[:20]!r}")
                continue
            path.write_bytes(data)
            print(f"OK {name} bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()}")
        except Exception as exc:
            print(f"FAIL {name}: {exc}")


if __name__ == "__main__":
    main()
