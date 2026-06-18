"""Render the organized fruit bowl showcase variants."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from fruit_bowl_demo import FRUIT_BOWL_OUTPUT_DIR, render_still, render_video


def _args_for(
    args: argparse.Namespace,
    *,
    name: str,
    label: str,
    smooth_shading: bool,
    ray_traced_shadows: bool = False,
    bowl_material: str = "wood",
    light_mode: str = "multiple",
    edge_highlight: bool = False,
    edge_highlight_angle: float = 35.0,
    ambient: float | None = None,
    frames: int | None = None,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    sphere_segments: int | None = None,
    sphere_rings: int | None = None,
) -> Namespace:
    return Namespace(
        output=args.output_dir / f"{name}.png",
        video=args.output_dir / f"{name}.mp4",
        ffmpeg=args.ffmpeg,
        require_ffmpeg=not args.allow_frame_fallback,
        write_still=None,
        frames=frames or args.frames,
        fps=fps or args.fps,
        warmup=args.warmup,
        ambient=args.ambient if ambient is None else ambient,
        width=width or args.width,
        height=height or args.height,
        window_width=960,
        window_height=540,
        fit_window=True,
        light_mode=light_mode,
        bowl_material=bowl_material,
        renderer="cpu",
        smooth_shading=smooth_shading,
        ray_traced_shadows=ray_traced_shadows,
        edge_highlight=edge_highlight,
        edge_highlight_angle=edge_highlight_angle,
        sphere_segments=sphere_segments or args.sphere_segments,
        sphere_rings=sphere_rings or args.sphere_rings,
        label=label,
    )


def render_variants(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        _args_for(args, name="fruit_bowl_poly", label="FRUIT BOWL POLY", smooth_shading=False),
        _args_for(args, name="fruit_bowl_smooth", label="FRUIT BOWL SMOOTH", smooth_shading=True),
        _args_for(
            args,
            name="fruit_bowl_ray_traced",
            label="FRUIT BOWL RAY SHADOWS",
            smooth_shading=False,
            ray_traced_shadows=True,
            ambient=args.ray_ambient,
            frames=args.ray_frames,
            fps=args.ray_fps,
            width=args.ray_width,
            height=args.ray_height,
            sphere_segments=min(args.sphere_segments, args.ray_sphere_segments),
            sphere_rings=min(args.sphere_rings, args.ray_sphere_rings),
        ),
        _args_for(
            args,
            name="fruit_bowl_mirror_prelight",
            label="MIRROR BOWL PRELIGHT",
            smooth_shading=True,
            bowl_material="mirror",
            light_mode="mirror-prelight",
            frames=args.mirror_frames,
            fps=args.mirror_fps,
        ),
    ]

    for variant in variants:
        render_still(variant)
        render_video(variant)

    for angle in args.edge_angles:
        edge_args = _args_for(
            args,
            name=f"fruit_bowl_edges_{int(angle)}deg",
            label=f"FRUIT BOWL EDGES {int(angle)} DEG",
            smooth_shading=True,
            edge_highlight=True,
            edge_highlight_angle=angle,
        )
        render_still(edge_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render organized fruit bowl still/video showcase variants.")
    parser.add_argument("--output-dir", type=Path, default=FRUIT_BOWL_OUTPUT_DIR)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--allow-frame-fallback", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--ray-frames", type=int, default=30)
    parser.add_argument("--mirror-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--ray-fps", type=int, default=3)
    parser.add_argument("--mirror-fps", type=int, default=12)
    parser.add_argument("--warmup", type=float, default=1.25)
    parser.add_argument("--ambient", type=float, default=0.0)
    parser.add_argument("--ray-ambient", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--ray-width", type=int, default=240)
    parser.add_argument("--ray-height", type=int, default=136)
    parser.add_argument("--sphere-segments", type=int, default=16)
    parser.add_argument("--sphere-rings", type=int, default=8)
    parser.add_argument("--ray-sphere-segments", type=int, default=10)
    parser.add_argument("--ray-sphere-rings", type=int, default=5)
    parser.add_argument("--edge-angles", type=float, nargs="+", default=[25.0, 55.0])
    args = parser.parse_args()
    if args.fps <= 0 or args.ray_fps <= 0 or args.mirror_fps <= 0:
        raise ValueError("fps values must be positive")
    if args.frames <= 0 or args.ray_frames <= 0 or args.mirror_frames <= 0:
        raise ValueError("frame counts must be positive")
    return args


def main() -> None:
    try:
        render_variants(parse_args())
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
