"""Render the fruit bowl demo to a real video file."""

from __future__ import annotations

import argparse
from pathlib import Path

from fruit_bowl_demo import OUTPUT_DIR, apply_cpu_reduced_specs, render_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the fruit bowl demo to MP4 or MOV using ffmpeg.")
    parser.add_argument("--video", type=Path, default=OUTPUT_DIR / "fruit_bowl.mp4")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--allow-frame-fallback", action="store_true")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--ambient", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--label", default="FRUIT BOWL")
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=204)
    parser.add_argument(
        "--light-mode",
        choices=("multiple", "blinking", "multicolor", "color-shift-blink", "mirror-prelight"),
        default="multiple",
    )
    parser.add_argument("--bowl-material", choices=("wood", "mirror"), default="wood")
    parser.add_argument("--renderer", choices=("cpu", "py_gpu"), default="py_gpu")
    parser.add_argument("--gpu-fast-render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-reduced-specs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-shading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ray-traced-shadows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-highlight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-highlight-angle", type=float, default=35.0)
    parser.add_argument("--max-render-distance", type=float)
    parser.add_argument("--sphere-segments", type=int, default=14)
    parser.add_argument("--sphere-rings", type=int, default=7)
    args = apply_cpu_reduced_specs(parser.parse_args())
    args.require_ffmpeg = not args.allow_frame_fallback
    return args


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.frames <= 0:
        raise ValueError("frames must be positive")
    try:
        render_video(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
