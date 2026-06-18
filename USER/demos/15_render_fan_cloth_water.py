"""Render the fan, cloth, and vector-cloud water demo."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(
        [
            "python",
            "examples/fan_cloth_water_demo.py",
            "--output",
            "USER/environments/fan_cloth_water/renderings/fan_cloth_water.png",
            "--quality",
            "balanced",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
