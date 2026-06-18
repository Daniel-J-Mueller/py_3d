"""Live fruit bowl physics demo."""

from __future__ import annotations

import argparse

from fruit_bowl_demo import LiveFruitBowlViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live bouncing fruit bowl demo.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=204)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--fit-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sphere-segments", type=int, default=14)
    parser.add_argument("--sphere-rings", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    LiveFruitBowlViewer(args).run()


if __name__ == "__main__":
    main()
