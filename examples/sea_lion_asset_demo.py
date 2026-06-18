"""Render a prepared sea lion mesh asset."""

from __future__ import annotations

import argparse
from pathlib import Path
from math import sin, tau

from py_3d import Camera, FloatingTextBulletin, Lamp, Material, PixelBuffer, Plane, RenderEngine, RenderSettings, Scene, Sun, TextBulletin, load_mesh_asset
from py_3d.color import Color


DEFAULT_ASSET = Path("USER") / "assets" / "sea_lion" / "sea_lion.py3dmesh.json"
DEFAULT_OUTPUT = Path("USER") / "environments" / "sea_lion" / "renderings" / "sea_lion_asset.png"


def sea_lion_skin_texture(width: int = 384, height: int = 384) -> PixelBuffer:
    pixels: list[Color] = []
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            broad = 0.5 + 0.5 * sin((u * 2.4 + 0.16 * sin(v * tau * 2.0)) * tau)
            speckle = 0.5 + 0.5 * sin((u * 31.0 + v * 37.0 + 0.15 * sin(u * tau * 7.0)) * tau)
            wet = 0.5 + 0.5 * sin((u * 8.5 - v * 5.5) * tau)
            pixels.append(Color(82 + broad * 38 + speckle * 20, 70 + broad * 30 + speckle * 14, 60 + broad * 24 + wet * 20))
    return PixelBuffer(width, height, pixels)


def make_scene(asset: Path) -> Scene:
    scene = Scene()
    skin = Material(
        color=(110, 90, 72),
        texture=sea_lion_skin_texture(),
        roughness=0.24,
        fuzziness=0.04,
        specular=0.28,
        shininess=42.0,
    )
    scene.add(load_mesh_asset(asset, skin))
    scene.add(Plane((0, -0.01, 0), (0, 1, 0), Material(color=(42, 58, 62), roughness=0.55), size=3.5))
    scene.add_light(Sun(direction=(-0.45, -0.8, -0.35), color=(190, 210, 235), intensity=0.24))
    scene.add_light(Lamp(position=(-0.9, 1.5, -1.25), color=(255, 232, 198), intensity=5.6))
    scene.add_light(Lamp(position=(1.2, 0.8, -0.55), color=(120, 166, 255), intensity=1.8))
    scene.add_bulletin(TextBulletin("SEA LION ASSET\nINGESTED PY3D MESH", position=(10, 10), background=(4, 7, 10), padding=5))
    scene.add_bulletin(FloatingTextBulletin("UVS PRESERVED\nTRIANGLES CLEANED", position=(0, 1.28, 0), background=(8, 5, 3), padding=5))
    return scene


def make_camera() -> Camera:
    return Camera(position=(1.8, 1.05, -2.7), target=(0.0, 0.58, 0.0), fov_degrees=42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a prepared sea lion mesh asset.")
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--ambient", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.asset.exists():
        raise SystemExit(f"Prepared asset missing: {args.asset}. Run examples/ingest_asset.py first.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings = RenderSettings(
        width=args.width,
        height=args.height,
        ambient=args.ambient,
        gamma=args.gamma,
        light_wrap=0.18,
        bounce_light=0.12,
        tone_mapping=True,
        max_render_distance=6.0,
    )
    RenderEngine().render(make_scene(args.asset), make_camera(), settings).to_png(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
