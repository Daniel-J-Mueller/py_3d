"""List the curated USER demo experiences."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


EXPERIENCES = [
    ("10_live_fruit_bowl_gpu.py", "Live OpenGL fruit bowl with filled GPU rendering; press R for wireframe."),
    ("11_live_capsule_walk.py", "Live OpenGL capsule controller with first-person camera."),
    ("20_render_feature_previews.py", "Render short still previews for fruit bowl, slime, and rocket tube."),
    ("30_render_environment_videos.py", "Render selected MP4 outputs into each environment renderings dir."),
    ("40_run_feature_tests.py", "Run pytest plus environment dry-runs with CPU/GPU specs printed."),
]


def main() -> None:
    print("USER demo experiences:")
    for script, description in EXPERIENCES:
        print(f"  python USER/demos/{script}")
        print(f"    {description}")


if __name__ == "__main__":
    main()
