"""Render the curated still-image feature previews."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
COMMANDS = [
    ["python", "USER/tests/run_environment.py", "-e", "fruit_bowl", "--variant", "fast_gpu_still"],
    ["python", "USER/tests/run_environment.py", "-e", "slime_fluid", "--variant", "still"],
    ["python", "USER/tests/run_environment.py", "-e", "rocket_tube", "--variant", "still"],
]


def main() -> None:
    for command in COMMANDS:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
