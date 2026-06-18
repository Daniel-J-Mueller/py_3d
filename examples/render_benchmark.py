"""Small CPU renderer benchmark for local tuning."""

from __future__ import annotations

import argparse
from statistics import mean
from time import perf_counter

from py_3d import Box, CPURenderer, Camera, Lamp, Material, RenderEngine, RenderSettings, Scene, Sphere, Sun


def make_scene() -> Scene:
    scene = Scene()
    scene.add(
        Sphere(
            center=(-0.8, 0.0, 0.0),
            radius=0.75,
            material=Material(color=(80, 145, 235), absorption=(0.05, 0.05, 0.02)),
        ),
        Box(
            center=(0.85, -0.05, 0.15),
            size=(1.05, 1.05, 1.05),
            material=Material(color=(225, 155, 70), absorption=(0.08, 0.16, 0.22)),
        ),
    )
    scene.add_light(Sun(direction=(-0.4, -0.7, -1.0), color=(255, 244, 225), intensity=0.85))
    scene.add_light(Lamp(position=(-2.0, 1.7, -2.0), color=(80, 150, 255), intensity=2.6))
    scene.add_light(Lamp(position=(2.0, 1.2, -1.2), color=(255, 90, 120), intensity=1.8))
    return scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark py_3d CPU rendering.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--sphere-segments", type=int, default=20)
    parser.add_argument("--sphere-rings", type=int, default=10)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderer = CPURenderer(cache_static_geometry=not args.no_cache)
    engine = RenderEngine(renderer)
    scene = make_scene()
    settings = RenderSettings(
        width=args.width,
        height=args.height,
        background=(7, 9, 14),
        ambient=0.06,
        sphere_segments=args.sphere_segments,
        sphere_rings=args.sphere_rings,
    )

    timings = []
    for index in range(args.frames):
        yaw = index * 3.0
        camera = Camera(
            position=(1.4 * (index % 3 - 1), 0.6, -4.0),
            target=(0.0, 0.0, 0.0),
            fov_degrees=52 + (yaw * 0.0),
        )
        start = perf_counter()
        engine.render(scene, camera, settings)
        timings.append(perf_counter() - start)

    average = mean(timings)
    print(f"frames: {args.frames}")
    print(f"size: {args.width}x{args.height}")
    print(f"sphere: {args.sphere_segments} segments x {args.sphere_rings} rings")
    print(f"cache: {not args.no_cache}")
    print(f"avg frame: {average * 1000:.2f} ms")
    print(f"approx fps: {1.0 / average:.1f}")


if __name__ == "__main__":
    main()
