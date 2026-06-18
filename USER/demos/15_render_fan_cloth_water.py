"""Launch the live fan, cloth, and vector-cloud water demo."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(
        [
            "python",
            "examples/fan_cloth_water_demo.py",
            "--live",
            "--quality",
            "ultra",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
