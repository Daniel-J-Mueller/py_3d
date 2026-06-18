"""Launch the live mirror-prelight fruit bowl environment."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(["python", "USER/tests/run_environment.py", "-e", "fruit_bowl", "--live", "mirror_prelight"], cwd=ROOT)


if __name__ == "__main__":
    main()
