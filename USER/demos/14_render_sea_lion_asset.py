"""Launch the live prepared sea lion mesh asset preview."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(["python", "examples/sea_lion_asset_demo.py", "--live", "--asset", "USER/assets/sea_lion/sea_lion.py3dmesh.json"], cwd=ROOT)


if __name__ == "__main__":
    main()
