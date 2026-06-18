"""Render a basic bounded slime-fluid blob simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from fruit_bowl_demo import find_ffmpeg, ffmpeg_missing_message
from py_3d import Box, Camera, Color, FluidBlob, FluidWorld, Lamp, Material, RenderEngine, RenderSettings, Scene, Sun, TextBulletin


OUTPUT_DIR = Path("USER") / "environments" / "slime_fluid" / "renderings"


def build_world() -> FluidWorld:
    world = FluidWorld(bounds_min=(-2.0, 0.0, -1.5), bounds_max=(2.0, 2.2, 1.5), gravity=(0.0, -6.5, 0.0), heal_distance_factor=0.35)
    slime = Material(color=(58, 205, 154), roughness=0.22, fuzziness=0.12, specular=0.18, shininess=22, light_transmission=0.35)
    world.add_blob(FluidBlob.from_radius((-0.8, 1.55, 0.0), 0.42, velocity=(2.0, 0.2, 0.35), stretch=(1.6, 0.1, 0.0), stretchiness=0.22, viscosity=0.16, surface_tension=0.72, wetting=0.55, stickiness=0.22, bounciness=0.18, material=slime))
    world.add_blob(FluidBlob.from_radius((0.55, 1.05, 0.05), 0.32, velocity=(-0.7, 0.2, -0.15), stretchiness=0.65, viscosity=0.24, surface_tension=0.82, wetting=0.5, stickiness=0.3, bounciness=0.12, material=slime))
    return world


def make_scene(world: FluidWorld, label: str = "SLIME FLUID") -> Scene:
    scene = Scene()
    floor = Material(color=(40, 62, 68), roughness=0.55)
    wall = Material(color=(84, 92, 106), roughness=0.45, light_transmission=0.15)
    scene.add(Box((0, -0.04, 0), (4.2, 0.08, 3.2), floor))
    scene.add(Box((-2.05, 1.1, 0), (0.08, 2.2, 3.2), wall))
    scene.add(Box((2.05, 1.1, 0), (0.08, 2.2, 3.2), wall))
    scene.add(Box((0, 1.1, 1.55), (4.2, 2.2, 0.08), wall))
    for blob in world.blobs:
        scene.add(blob.to_primitive())
    scene.add_light(Sun(direction=(-0.3, -0.8, -0.5), color=(255, 245, 230), intensity=0.75))
    scene.add_light(Lamp(position=(0.6, 1.8, -1.0), color=(110, 180, 255), intensity=4.8))
    scene.add_bulletin(TextBulletin(f"{label}\nFIXED VOLUME BOUNDED BLOBS", position=(10, 10), color=(245, 248, 255), background=(4, 6, 10), padding=5))
    return scene


def make_camera() -> Camera:
    return Camera(position=(0.2, 1.65, -4.6), target=(0.0, 0.9, 0.05), fov_degrees=50)


def make_settings(args: argparse.Namespace) -> RenderSettings:
    return RenderSettings(width=args.width, height=args.height, background=Color(7, 9, 13), ambient=args.ambient, gamma=args.gamma, smooth_shading=True, sphere_segments=args.sphere_segments, sphere_rings=args.sphere_rings, ray_traced_shadows=args.ray_traced_shadows)


def render_still(args: argparse.Namespace) -> Path:
    world = build_world()
    for _ in range(int(args.warmup * args.fps)):
        world.step(1.0 / args.fps, substeps=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    RenderEngine().render(make_scene(world), make_camera(), make_settings(args)).to_png(output)
    print(f"Wrote {output}")
    return output


def render_video(args: argparse.Namespace) -> Path:
    output = Path(args.video)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    if ffmpeg is None:
        message = ffmpeg_missing_message()
        if args.require_ffmpeg:
            raise RuntimeError(message)
        frames_dir = output.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = None
    world = build_world()
    engine = RenderEngine()
    settings = make_settings(args)
    if frames_dir is not None:
        for frame in range(args.frames):
            world.step(1.0 / args.fps, substeps=2)
            engine.render(make_scene(world, f"SLIME FRAME {frame:03d}"), make_camera(), settings).to_png(frames_dir / f"frame_{frame:04d}.png")
        print(f"{message} Wrote PNG frames to {frames_dir}.")
        return frames_dir

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "image2pipe",
        "-framerate",
        str(args.fps),
        "-vcodec",
        "ppm",
        "-i",
        "-",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("could not open ffmpeg stdin")
    try:
        for frame in range(args.frames):
            world.step(1.0 / args.fps, substeps=2)
            process.stdin.write(engine.render(make_scene(world, f"SLIME FRAME {frame:03d}"), make_camera(), settings).to_ppm_bytes())
    finally:
        process.stdin.close()
    result = process.wait()
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result}")
    print(f"Wrote {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render bounded slime-fluid blobs.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "slime_fluid.png")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--require-ffmpeg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--ambient", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.12)
    parser.add_argument("--sphere-segments", type=int, default=18)
    parser.add_argument("--sphere-rings", type=int, default=9)
    parser.add_argument("--ray-traced-shadows", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video is not None:
        render_video(args)
    else:
        render_still(args)


if __name__ == "__main__":
    main()
