"""Launch the live wind-over-pool water demo."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(
        [
            "python",
            "examples/wind_pool_water_demo.py",
            "--live",
            "--quality",
            "balanced",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
