"""Main menu for curated USER demo experiences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Experience:
    script: str
    description: str
    command: tuple[str, ...]


EXPERIENCES = [
    Experience("10_live_fruit_bowl_gpu.py", "Live OpenGL fruit bowl with graphics settings, grab physics, and baked sign.", ("python", "USER/demos/10_live_fruit_bowl_gpu.py")),
    Experience("11_live_capsule_walk.py", "Live capsule controller with FPS/over-shoulder cameras, crouch, and HUD.", ("python", "USER/demos/11_live_capsule_walk.py")),
    Experience("12_live_fruit_bowl_mirror_prelight.py", "High-spec uncapped OpenGL mirror-prelight fruit bowl.", ("python", "USER/demos/12_live_fruit_bowl_mirror_prelight.py")),
    Experience("13_live_fruit_bowl_poly_lamp.py", "Low-poly wood fruit bowl lit by a visible lamp primitive.", ("python", "USER/demos/13_live_fruit_bowl_poly_lamp.py")),
    Experience("14_render_sea_lion_asset.py", "Render the prepared sea lion mesh asset preview.", ("python", "USER/demos/14_render_sea_lion_asset.py")),
    Experience("15_render_fan_cloth_water.py", "Render fan-blown cloth plus vector-cloud water stirred by a fan blade.", ("python", "USER/demos/15_render_fan_cloth_water.py")),
    Experience("20_render_feature_previews.py", "Render short still previews for fruit bowl, slime, and rocket tube.", ("python", "USER/demos/20_render_feature_previews.py")),
    Experience("30_render_environment_videos.py", "Render selected MP4 outputs into each environment renderings dir.", ("python", "USER/demos/30_render_environment_videos.py")),
    Experience("40_run_feature_tests.py", "Run pytest plus environment dry-runs with CPU/GPU specs printed.", ("python", "USER/demos/40_run_feature_tests.py")),
]


def print_menu() -> None:
    print("py_3d demo menu")
    print("")
    for index, experience in enumerate(EXPERIENCES, start=1):
        print(f"{index}. {experience.script}")
        print(f"   {experience.description}")
    print("")
    print("Choose a number to run, or press Enter to exit.")


def run_experience(index: int, *, dry_run: bool = False) -> None:
    if index < 1 or index > len(EXPERIENCES):
        raise SystemExit(f"Choose a number from 1 to {len(EXPERIENCES)}.")
    command = list(EXPERIENCES[index - 1].command)
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a menu of USER demo experiences.")
    parser.add_argument("--list", action="store_true", help="Print the menu and exit.")
    parser.add_argument("--run", type=int, help="Run a menu item by number without prompting.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print_menu()
    if args.list:
        return
    if args.run is not None:
        run_experience(args.run, dry_run=args.dry_run)
        return
    choice = input("> ").strip()
    if not choice:
        return
    run_experience(int(choice), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
