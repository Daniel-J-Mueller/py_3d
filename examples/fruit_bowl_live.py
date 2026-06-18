"""Live fruit bowl physics demo."""

from __future__ import annotations

import argparse

from fruit_bowl_demo import GLFruitBowlViewer, LiveFruitBowlViewer, apply_cpu_reduced_specs, apply_render_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live bouncing fruit bowl demo.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=204)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--fit-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-wireframe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quality", help="Render quality preset from USER/settings.json.")
    parser.add_argument("--ambient", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--light-wrap", type=float, default=0.0)
    parser.add_argument("--bounce-light", type=float, default=0.0)
    parser.add_argument("--tone-mapping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--label", default="KINEMATIC FRUIT BOWL")
    parser.add_argument(
        "--light-mode",
        choices=("multiple", "blinking", "multicolor", "color-shift-blink", "mirror-prelight", "poly-lamp", "hanging-lamp"),
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
    return parser.parse_args()


def main() -> None:
    args = apply_cpu_reduced_specs(apply_render_quality(parse_args()))
    if args.fps < 0:
        raise ValueError("fps must be non-negative")
    if args.renderer == "py_gpu":
        try:
            GLFruitBowlViewer(args).run()
            return
        except Exception as exc:
            print(f"OpenGL live renderer unavailable, falling back to Tk PixelBuffer path: {exc}")
    if args.fps == 0:
        raise ValueError("fps must be positive for the Tk PixelBuffer fallback")
    LiveFruitBowlViewer(args).run()


if __name__ == "__main__":
    main()
