from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "benchmark_real_data_confirmation_v32.py"
PMLB_COMMIT = "7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v32 = _load_module(BASE_SCRIPT, "benchmark_real_data_confirmation_v32")

# Pre-result protocol correction: sonar (n=208) violated the predeclared
# n >= 250 eligibility criterion. No benchmark result had been produced when
# this correction was made. Dermatology satisfies the frozen eligibility rule.
v32.DATASETS = (
    "credit_approval_australia",
    "dermatology",
    "haberman",
    "heart_disease_cleveland",
    "horse_colic_outcome",
    "ionosphere",
    "penguins",
    "segmentation",
)


def load_pmlb_dataset_pinned(name: str, cache_dir: Path):
    """Load a frozen PMLB dataset without relying on the installed registry.

    The local pmlb package may contain an older dataset-name registry. This
    loader reads the dataset from a pinned upstream PMLB commit and caches the
    resulting TSV.GZ locally. Missing rows are dropped, matching pmlb.fetch_data
    default behavior.
    """
    dataset_dir = cache_dir / name
    dataset_path = dataset_dir / f"{name}.tsv.gz"

    if dataset_path.exists():
        df = pd.read_csv(dataset_path, sep="\t", compression="gzip")
    else:
        url = (
            "https://github.com/EpistasisLab/pmlb/raw/"
            f"{PMLB_COMMIT}/datasets/{name}/{name}.tsv.gz"
        )
        df = pd.read_csv(url, sep="\t", compression="gzip")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_path, sep="\t", compression="gzip", index=False)

    df = df.dropna().reset_index(drop=True)
    if "target" not in df.columns:
        raise RuntimeError(f"target column missing for {name}")

    y_raw = df.pop("target").to_numpy()
    X = df.to_numpy(dtype=float)

    if X.ndim != 2 or len(X) != len(y_raw):
        raise RuntimeError(f"invalid X/y shape for {name}")
    if not np.all(np.isfinite(X)):
        raise RuntimeError(f"non-finite feature value in frozen dataset {name}")

    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    values, counts = np.unique(y, return_counts=True)

    if len(values) < 2:
        raise RuntimeError(f"dataset {name} has fewer than two classes")
    if len(X) < 250 or len(X) > 2500:
        raise RuntimeError(
            f"dataset {name} violates frozen sample-size criterion: n={len(X)}"
        )
    if X.shape[1] > 100:
        raise RuntimeError(
            f"dataset {name} violates frozen feature-count criterion: d={X.shape[1]}"
        )
    if len(values) > 10:
        raise RuntimeError(
            f"dataset {name} violates frozen class-count criterion: C={len(values)}"
        )
    if int(np.min(counts)) < 20:
        raise RuntimeError(
            f"dataset {name} violates frozen minimum class size 20: {counts.tolist()}"
        )

    meta = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(values)),
        "class_counts": [int(v) for v in counts],
        "original_classes": [str(v) for v in enc.classes_],
        "content_sha256": v32.dataset_fingerprint(X, y),
        "pmlb_commit": PMLB_COMMIT,
        "loader": "pinned upstream TSV.GZ; dropna=True",
    }
    return X, y, meta


v32.load_pmlb_dataset = load_pmlb_dataset_pinned


if __name__ == "__main__":
    v32.main()
