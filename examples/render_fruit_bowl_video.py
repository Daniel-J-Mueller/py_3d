"""Render the fruit bowl demo to a real video file."""

from __future__ import annotations

import argparse
from pathlib import Path

from fruit_bowl_demo import OUTPUT_DIR, render_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the fruit bowl demo to MP4 or MOV using ffmpeg.")
    parser.add_argument("--video", type=Path, default=OUTPUT_DIR / "fruit_bowl.mp4")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--allow-frame-fallback", action="store_true")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=204)
    parser.add_argument("--sphere-segments", type=int, default=14)
    parser.add_argument("--sphere-rings", type=int, default=7)
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
