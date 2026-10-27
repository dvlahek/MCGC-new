#!/usr/bin/env python3
"""Download the 12 frozen v2.1 PMLB datasets into the flat cache layout
expected by benchmark_label_audit_v21_oof.py and
benchmark_label_audit_v21_cleanlab_only.py.

Files are downloaded as the official .tsv.gz bytes from EpistasisLab/PMLB and
stored as:
    data_pmlb/<dataset>.tsv.gz

Usage:
    python examples/download_v21_pmlb_flat.py --out-dir data_pmlb
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

DATASETS = (
    "Hill_Valley_without_noise",
    "bupa",
    "ecoli",
    "mfeat_fourier",
    "mfeat_karhunen",
    "mfeat_zernike",
    "saheart",
    "schizo",
    "spectf",
    "vehicle",
    "wine_quality_red",
    "yeast",
)

BASE = "https://github.com/EpistasisLab/pmlb/raw/master/datasets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data_pmlb"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in DATASETS:
        dst = args.out_dir / f"{name}.tsv.gz"
        if dst.exists() and not args.force:
            sha = hashlib.sha256(dst.read_bytes()).hexdigest()
            print(f"SKIP {name:28s} {dst} sha256={sha[:16]}")
            continue

        url = f"{BASE}/{name}/{name}.tsv.gz"
        print(f"GET  {name:28s} {url}", flush=True)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MCGC-reproducibility/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()

        if len(data) < 20 or data[:2] != b"\x1f\x8b":
            raise RuntimeError(
                f"Downloaded payload for {name} is not a valid gzip file"
            )

        dst.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        print(
            f"WROTE {name:28s} bytes={len(data)} sha256={sha[:16]} path={dst}",
            flush=True,
        )

    print(f"DONE datasets={len(DATASETS)} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
