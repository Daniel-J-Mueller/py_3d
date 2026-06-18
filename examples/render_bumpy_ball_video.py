"""Render the bumpy rolling ball demo to MP4 or MOV."""

from __future__ import annotations

import argparse
from pathlib import Path

from bumpy_ball_demo import OUTPUT_DIR, render_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the bumpy ball physics demo to video using ffmpeg.")
    parser.add_argument("--video", type=Path, default=OUTPUT_DIR / "bumpy_ball_physics.mp4")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--allow-frame-fallback", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--smooth-shading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sphere-segments", type=int, default=36)
    parser.add_argument("--sphere-rings", type=int, default=18)
    args = parser.parse_args()
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
