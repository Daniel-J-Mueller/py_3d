"""Rocket tube, gas vector field, and cloth flag demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import pi, sin
from pathlib import Path
import subprocess

from fruit_bowl_demo import find_ffmpeg, ffmpeg_missing_message, make_engine
from py_3d import Box, Camera, Color, Lamp, Line3, Material, Mesh, RenderEngine, RenderSettings, Scene, Sun, TextBulletin, Triangle, Vec3


OUTPUT_DIR = Path("USER") / "environments" / "rocket_tube" / "renderings"


@dataclass
class GasEmitter:
    position: Vec3
    direction: Vec3
    strength: float
    spread: float = 0.75
    dissipation: float = 0.38

    def sample(self, point: Vec3, time: float) -> Vec3:
        offset = point - self.position
        distance = max(0.1, offset.length())
        alignment = max(0.0, offset.normalized(self.direction).dot(self.direction))
        pulse = 0.82 + 0.18 * sin(time * pi * 3.0)
        return self.direction * (self.strength * pulse * alignment**2 / (1.0 + distance * distance * self.spread))


@dataclass
class RocketState:
    position: Vec3 = Vec3(-0.8, 0.42, 0.0)
    velocity: Vec3 = Vec3(0.0, 0.0, 0.0)


def simulate_rocket(time: float, args: argparse.Namespace) -> RocketState:
    state = RocketState()
    dt = 1.0 / 90.0
    elapsed = 0.0
    while elapsed < time:
        step = min(dt, time - elapsed)
        pulse = 0.86 + 0.14 * sin(elapsed * pi * 3.0)
        exhaust_direction = exhaust_direction_at(elapsed)
        thrust = -exhaust_direction * (args.thrust * pulse / max(args.rocket_mass, 0.01))
        gravity = Vec3(0.0, -9.81, 0.0)
        drag = state.velocity * -args.rocket_drag
        state.velocity = state.velocity + (thrust + gravity + drag) * step
        state.position = state.position + state.velocity * step
        if state.position.y < 0.28:
            state.position = Vec3(state.position.x, 0.28, state.position.z)
            state.velocity = Vec3(state.velocity.x, max(0.0, state.velocity.y) * 0.2, state.velocity.z)
        elapsed += step
    return state


def exhaust_direction_at(time: float) -> Vec3:
    sway = sin(time * pi * 1.25) * 0.08
    return Vec3(0.58 + sway, -1.0, 0.0).normalized()


def make_emitter(state: RocketState, time: float, args: argparse.Namespace) -> GasEmitter:
    direction = exhaust_direction_at(time)
    nozzle = state.position + Vec3(0.05, -0.36, 0.0)
    pulse = 0.86 + 0.14 * sin(time * pi * 3.0)
    return GasEmitter(position=nozzle, direction=direction, strength=args.gas_strength * pulse, spread=args.gas_spread)


def make_flag(time: float, emitter: GasEmitter, *, cloth_weight: float, wind_tightness: float) -> Mesh:
    material = Material(color=(220, 65, 55), emission=(24, 5, 5), roughness=0.48, fuzziness=0.08)
    origin = Vec3(0.45, 1.05, 0.0)
    width = 1.2
    height = 0.62
    cols = 9
    rows = 5
    resistance = max(0.05, cloth_weight + wind_tightness)
    points: list[list[Vec3]] = []
    for row in range(rows + 1):
        point_row = []
        v = row / rows
        for col in range(cols + 1):
            u = col / cols
            base = origin + Vec3(u * width, -v * height, 0.0)
            gas = emitter.sample(base, time)
            flutter = sin(time * 7.0 + u * pi * 4.0 + v * pi) * 0.035 * u / resistance
            sag = cloth_weight * 0.09 * u * (0.4 + v)
            wind = gas * (0.085 * u / resistance)
            point_row.append(base + Vec3(wind.x * 0.05, wind.y * 0.05 - sag + flutter, wind.x * 0.18))
        points.append(point_row)
    triangles: list[Triangle] = []
    for row in range(rows):
        for col in range(cols):
            a = points[row][col]
            b = points[row][col + 1]
            c = points[row + 1][col + 1]
            d = points[row + 1][col]
            triangles.append(Triangle(a, d, b, material))
            triangles.append(Triangle(b, d, c, material))
    return Mesh(triangles)


def make_scene(time: float, args: argparse.Namespace) -> Scene:
    state = simulate_rocket(time, args)
    emitter = make_emitter(state, time, args)
    scene = Scene()
    metal = Material(color=(120, 126, 132), roughness=0.18, specular=0.45, shininess=48.0)
    flame = Material(color=(255, 134, 42), emission=(255, 118, 30), roughness=0.2, light_transmission=0.5)
    scene.add(Box(state.position + Vec3(-0.13, 0.0, 0.0), (0.24, 0.62, 0.58), metal))
    scene.add(Box(state.position + Vec3(0.12, -0.05, 0.0), (0.52, 0.18, 0.54), metal))
    scene.add(Box(state.position + Vec3(0.25, -0.3, 0.0), (0.22, 0.2, 0.36), flame))
    scene.add(Box((0.0, -0.05, 0.0), (4.4, 0.1, 2.2), Material(color=(42, 54, 58), roughness=0.6)))
    scene.add(Box((0.42, 0.55, 0.0), (0.06, 1.35, 0.06), Material(color=(85, 70, 52), roughness=0.4)))
    scene.add(make_flag(time, emitter, cloth_weight=args.cloth_weight, wind_tightness=args.wind_tightness))
    for index in range(11):
        start = emitter.position + emitter.direction * (0.14 + index * 0.16)
        force = emitter.sample(start, time)
        scene.add(Line3(start, start + force * 0.075, Material(color=(125, 190, 255), emission=(18, 35, 55))))
    lamp_position = state.position + Vec3(0.25, 1.05 + sin(time * 2.0) * 0.08, -0.65)
    scene.add(Box(lamp_position, (0.16, 0.16, 0.16), Material(color=(255, 210, 120), emission=(255, 190, 90))))
    scene.add_light(Lamp(position=lamp_position, color=(255, 195, 115), intensity=5.2))
    scene.add_light(Sun(direction=(-0.25, -0.85, -0.4), color=(210, 225, 255), intensity=0.25))
    scene.add_bulletin(TextBulletin("ROCKET TUBE\nGAS FIELD AND FLAG", position=(10, 10), color=(245, 248, 255), background=(4, 6, 10), padding=5))
    return scene


def make_camera(time: float, args: argparse.Namespace) -> Camera:
    state = simulate_rocket(time, args)
    target = Vec3((state.position.x + 0.55) * 0.5, 0.46 + state.position.y * 0.35, 0.0)
    return Camera(position=target + Vec3(-2.55, 1.2, -3.0), target=target, fov_degrees=58)


def make_settings(args: argparse.Namespace) -> RenderSettings:
    return RenderSettings(width=args.width, height=args.height, background=Color(7, 9, 13), ambient=args.ambient, gamma=args.gamma, max_render_distance=args.max_render_distance)


def render_still(args: argparse.Namespace) -> Path:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_engine(args.renderer, fast=args.gpu_fast_render).render(make_scene(args.time, args), make_camera(args.time, args), make_settings(args)).to_png(output)
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
        raise RuntimeError(message)
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
    engine = make_engine(args.renderer, fast=args.gpu_fast_render)
    settings = make_settings(args)
    try:
        for frame in range(args.frames):
            time = frame / args.fps
            process.stdin.write(engine.render(make_scene(time, args), make_camera(time, args), settings).to_ppm_bytes())
    finally:
        process.stdin.close()
    result = process.wait()
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result}")
    print(f"Wrote {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a rocket tube gas-field demo.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "rocket_tube.png")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--require-ffmpeg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--renderer", choices=("cpu", "py_gpu"), default="py_gpu")
    parser.add_argument("--gpu-fast-render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-reduced-specs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ambient", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.12)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--max-render-distance", type=float, default=8.0)
    parser.add_argument("--rocket-mass", type=float, default=1.6)
    parser.add_argument("--thrust", type=float, default=18.0)
    parser.add_argument("--rocket-drag", type=float, default=0.22)
    parser.add_argument("--gas-strength", type=float, default=8.0)
    parser.add_argument("--gas-spread", type=float, default=0.58)
    parser.add_argument("--cloth-weight", type=float, default=0.72)
    parser.add_argument("--wind-tightness", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.renderer == "cpu" and args.cpu_reduced_specs:
        args.width = min(args.width, 320)
        args.height = min(args.height, 180)
        args.fps = min(args.fps, 12)
    if args.video is not None:
        render_video(args)
    else:
        render_still(args)


if __name__ == "__main__":
    main()
