"""Render the prepared sea lion mesh asset preview."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(["python", "USER/tests/run_environment.py", "-e", "sea_lion", "--variant", "asset_preview"], cwd=ROOT)


if __name__ == "__main__":
    main()
