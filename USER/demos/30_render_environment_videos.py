"""Render selected environment videos into USER environment renderings dirs."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
COMMANDS = [
    ["python", "USER/tests/run_environment.py", "-e", "rocket_tube", "--variant", "video"],
    ["python", "USER/tests/run_environment.py", "-e", "slime_fluid", "--variant", "video"],
]


def main() -> None:
    for command in COMMANDS:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
