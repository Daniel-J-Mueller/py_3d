"""Physics demo with a noisy, visually perturbed rolling sphere."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from fruit_bowl_demo import find_ffmpeg, ffmpeg_missing_message, rotate_euler
from py_3d import (
    Camera,
    Color,
    Lamp,
    Line3,
    Material,
    PhysicsWorld,
    PixelBuffer,
    RenderEngine,
    RenderSettings,
    Scene,
    SphereBody,
    SphereCollider,
    StaticBox,
    StaticPlane,
    Sun,
    SurfacePerturbation,
    TextBulletin,
    Vec3,
)


OUTPUT_DIR = Path("renderings-tests")


def build_world() -> tuple[PhysicsWorld, SphereBody, StaticPlane, StaticBox]:
    texture = PixelBuffer.from_png(Path("assets") / "tv-test.png")
    ball_material = Material(
        texture=texture,
        absorption=(0.04, 0.04, 0.04),
        roughness=0.35,
        fuzziness=0.15,
    )
    perturbation = SurfacePerturbation(magnitude=0.09, scale=3.4, seed=12, octaves=4, gain=0.55)
    ball = SphereBody(
        position=(-2.15, 1.3, 0.0),
        radius=0.34,
        velocity=(0.0, 0.0, 0.0),
        mass=1.0,
        restitution=0.42,
        friction=0.3,
        static_friction=0.48,
        kinetic_friction=0.3,
        rolling_resistance=0.02,
        material=ball_material,
        visual_perturbation=perturbation,
        collision_boundary=SphereCollider(radius=0.34),
    )
    ramp = StaticPlane(
        point=(0.0, 0.0, 0.0),
        normal=(0.34, 1.0, 0.0),
        friction=0.38,
        restitution=0.18,
        material=Material(color=(75, 145, 95), roughness=0.45, fuzziness=0.2),
        size=5.8,
    )
    wall = StaticBox(
        center=(2.15, 0.5, 0.0),
        size=(0.28, 1.6, 2.1),
        restitution=0.65,
        friction=0.18,
        material=Material(color=(170, 175, 190), roughness=0.55),
    )

    world = PhysicsWorld(gravity=(0.0, -9.81, 0.0))
    world.add_sphere(ball)
    world.add_plane(ramp)
    world.add_box(wall)
    return world, ball, ramp, wall


def make_scene(
    ball: SphereBody,
    ramp: StaticPlane,
    wall: StaticBox,
    path: list[Vec3],
    *,
    label: str = "BUMPY BALL PHYSICS",
) -> Scene:
    scene = Scene()
    scene.add(ramp.to_primitive(), wall.to_primitive(), ball.to_primitive())
    scene.add(_ball_spin_marker(ball))
    for start, end in zip(path, path[1:]):
        scene.add(Line3(start, end, Material(color=(245, 220, 90), emission=(80, 60, 0))))
    scene.add_light(Sun(direction=(-0.4, -0.8, -1.0), color=(255, 245, 230), intensity=0.95))
    scene.add_light(Lamp(position=(-1.8, 2.0, -2.0), color=(90, 140, 255), intensity=2.5))
    scene.add_light(Lamp(position=(1.9, 1.2, -1.1), color=(255, 120, 90), intensity=1.6))
    scene.add_bulletin(
        TextBulletin(
            f"{label}\nPOLY COLLISION + OPTIONAL SMOOTH SHADING",
            position=(10, 10),
            color=(245, 248, 255),
            background=(5, 7, 11),
            padding=5,
            scale=1,
        )
    )
    return scene


def _ball_spin_marker(ball: SphereBody) -> Line3:
    end = rotate_euler(Vec3(0.0, ball.radius * 1.06, 0.0), ball.rotation)
    tail = rotate_euler(Vec3(ball.radius * 0.5, ball.radius * 0.55, 0.0), ball.rotation)
    return Line3(
        ball.position + tail,
        ball.position + end,
        Material(color=(255, 255, 255), emission=(40, 40, 40)),
    )


def make_camera() -> Camera:
    return Camera(position=(0.25, 2.15, -5.5), target=(0.2, 0.45, 0.0), fov_degrees=48)


def make_settings(args: argparse.Namespace) -> RenderSettings:
    return RenderSettings(
        width=args.width,
        height=args.height,
        background=Color(8, 10, 14),
        ambient=0.1,
        smooth_shading=args.smooth_shading,
        sphere_segments=args.sphere_segments,
        sphere_rings=args.sphere_rings,
    )


def simulate_to_still(args: argparse.Namespace) -> tuple[SphereBody, StaticPlane, StaticBox, list[Vec3]]:
    world, ball, ramp, wall = build_world()
    path = [ball.position]
    for step in range(max(1, int(args.warmup * 60.0))):
        world.step(1.0 / 60.0, substeps=3)
        if step % 6 == 0:
            path.append(ball.position)
    return ball, ramp, wall, path


def render_still(args: argparse.Namespace) -> Path:
    ball, ramp, wall, path = simulate_to_still(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    RenderEngine().render(make_scene(ball, ramp, wall, path), make_camera(), make_settings(args)).to_png(output_path)
    print(f"Wrote {output_path}")
    return output_path


def render_video(args: argparse.Namespace) -> Path:
    output_path = Path(args.video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    world, ball, ramp, wall = build_world()
    path = [ball.position]
    engine = RenderEngine()
    settings = make_settings(args)
    ffmpeg = find_ffmpeg(getattr(args, "ffmpeg", None))
    if ffmpeg is None:
        message = ffmpeg_missing_message()
        if getattr(args, "require_ffmpeg", False):
            raise RuntimeError(message)
        frames_dir = output_path.with_suffix("") if output_path.suffix else output_path
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(args.frames):
            world.step(1.0 / args.fps, substeps=4)
            if frame % 3 == 0:
                path.append(ball.position)
            scene = make_scene(ball, ramp, wall, path, label=f"BUMPY BALL FRAME {frame:03d}")
            engine.render(scene, make_camera(), settings).to_png(frames_dir / f"frame_{frame:04d}.png")
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
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("could not open ffmpeg stdin")
    try:
        for frame in range(args.frames):
            world.step(1.0 / args.fps, substeps=4)
            if frame % 3 == 0:
                path.append(ball.position)
            scene = make_scene(ball, ramp, wall, path, label=f"BUMPY BALL FRAME {frame:03d}")
            process.stdin.write(engine.render(scene, make_camera(), settings).to_ppm_bytes())
    finally:
        process.stdin.close()
    result = process.wait()
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result}")
    print(f"Wrote {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a bumpy rolling ball still or video.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "bumpy_ball_physics.png")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--require-ffmpeg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-still", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--smooth-shading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sphere-segments", type=int, default=36)
    parser.add_argument("--sphere-rings", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.frames <= 0:
        raise ValueError("frames must be positive")
    if args.video is not None:
        if args.write_still:
            render_still(args)
        render_video(args)
        return
    render_still(args)


if __name__ == "__main__":
    main()
