"""Launch the live low-poly wood fruit bowl with a lamp primitive."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(["python", "USER/tests/run_environment.py", "-e", "fruit_bowl", "--live", "poly_lamp"], cwd=ROOT)


if __name__ == "__main__":
    main()
