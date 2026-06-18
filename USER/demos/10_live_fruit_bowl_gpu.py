"""Launch the live GPU fruit bowl environment."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(["python", "USER/tests/run_environment.py", "-e", "fruit_bowl", "--live", "default"], cwd=ROOT)


if __name__ == "__main__":
    main()
