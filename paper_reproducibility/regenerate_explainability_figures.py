from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "figures" / "fig_spiral_evidence.py",
    ROOT / "figures" / "fig_digits_xai.py",
    ROOT / "figures" / "fig_path_validation.py",
    ROOT / "figures" / "fig_bc_interventions.py",
    ROOT / "figures" / "fig_predictive_boundary.py",
    ROOT / "figures" / "fig_spiral_waterfall.py",
]


def main():
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    for script in SCRIPTS:
        print(f"==> {script.name}")
        subprocess.run([sys.executable, str(script), "--output-dir", str(out)], check=True)


if __name__ == "__main__":
    main()
