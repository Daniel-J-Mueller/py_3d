"""Run core tests and dry-run USER environment commands."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
COMMANDS = [
    ["python", "-m", "pytest"],
    ["python", "USER/tests/run_environment.py", "-e", "fruit_bowl", "--dry-run"],
    ["python", "USER/tests/run_environment.py", "-e", "rocket_tube", "--dry-run"],
    ["python", "USER/tests/run_environment.py", "-e", "slime_fluid", "--dry-run"],
]


def main() -> None:
    for command in COMMANDS:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
