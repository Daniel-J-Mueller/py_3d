"""Benchmark the renderer selected through the GPU entry point."""

from __future__ import annotations

import argparse
from statistics import mean
from time import perf_counter

from py_3d import (
    Camera,
    GPURenderer,
    RenderEngine,
    RenderSettings,
    build_gpu_scene_batch,
    detect_gpu_backends,
)

from render_benchmark import make_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the py_3d GPU renderer entry point.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--sphere-segments", type=int, default=20)
    parser.add_argument("--sphere-rings", type=int, default=10)
    parser.add_argument("--renderer", choices=("scaffold", "py_gpu"), default="py_gpu")
    parser.add_argument("--reference-compatible", action="store_true", help="benchmark py_gpu's py_3d parity path instead of the accelerated batch path")
    parser.add_argument("--strict", action="store_true", help="fail if no accelerated GPU rasterizer is available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = make_scene()
    settings = RenderSettings(
        width=args.width,
        height=args.height,
        background=(7, 9, 14),
        ambient=0.06,
        sphere_segments=args.sphere_segments,
        sphere_rings=args.sphere_rings,
    )
    if args.renderer == "py_gpu":
        try:
            from py_gpu.adapters.py3d import Py3DRasterRenderer

            renderer = Py3DRasterRenderer(reference_compatible=args.reference_compatible)
        except Exception as exc:
            if args.strict:
                raise RuntimeError("py_gpu renderer bridge is not available") from exc
            renderer = GPURenderer(allow_cpu_fallback=True)
    else:
        renderer = GPURenderer(allow_cpu_fallback=not args.strict)
    engine = RenderEngine(renderer)
    batch = build_gpu_scene_batch(scene, settings)

    timings = []
    for index in range(args.frames):
        camera = Camera(
            position=(1.4 * (index % 3 - 1), 0.6, -4.0),
            target=(0.0, 0.0, 0.0),
            fov_degrees=52,
        )
        start = perf_counter()
        engine.render(scene, camera, settings)
        timings.append(perf_counter() - start)

    average = mean(timings)
    backends = detect_gpu_backends()
    capabilities = getattr(getattr(renderer, "backend_impl", None), "capabilities", None)
    accelerated = getattr(capabilities, "accelerated", getattr(renderer, "is_accelerated", False))
    fallback_enabled = getattr(renderer, "allow_cpu_fallback", False)
    print(f"detected gpu packages: {', '.join(backends) if backends else 'none'}")
    print(f"renderer: {renderer.name}")
    if capabilities is not None:
        print(f"backend: {capabilities.name}")
    print(f"accelerated: {accelerated}")
    print(f"fallback enabled: {fallback_enabled}")
    print(f"triangles: {len(batch.indices)}")
    print(f"frames: {args.frames}")
    print(f"size: {args.width}x{args.height}")
    print(f"sphere: {args.sphere_segments} segments x {args.sphere_rings} rings")
    print(f"avg frame: {average * 1000:.2f} ms")
    print(f"approx fps: {1.0 / average:.1f}")


if __name__ == "__main__":
    main()
