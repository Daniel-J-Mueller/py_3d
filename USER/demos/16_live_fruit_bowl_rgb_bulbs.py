"""Launch the live fruit bowl with standalone RGB blinking bulbs."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(["python", "examples/fruit_bowl_rgb_bulbs_live.py", "--renderer", "py_gpu"], cwd=ROOT)


if __name__ == "__main__":
    main()
