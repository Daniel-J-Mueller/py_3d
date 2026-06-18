"""Render FPS and third-person HUD demo images."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(["python", "examples/hud_demo.py", "--output-dir", "renderings-tests"], cwd=ROOT)


if __name__ == "__main__":
    main()
