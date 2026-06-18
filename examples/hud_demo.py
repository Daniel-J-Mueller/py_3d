"""Render simple FPS and third-person HUD examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from py_3d import PixelBuffer, draw


OUTPUT_DIR = Path("renderings-tests")


def _panel(buffer: PixelBuffer, x: int, y: int, width: int, height: int) -> None:
    draw.rect(buffer, (x, y), (width, height), (0, 0, 0), fill=True)
    draw.rect(buffer, (x, y), (width, height), (170, 170, 170), fill=False)


def _bar(buffer: PixelBuffer, x: int, y: int, width: int, height: int, amount: float, color: tuple[int, int, int]) -> None:
    draw.rect(buffer, (x, y), (width, height), (42, 42, 42), fill=True)
    draw.rect(buffer, (x, y), (max(1, int(width * amount)), height), color, fill=True)
    draw.rect(buffer, (x, y), (width, height), (210, 210, 210), fill=False)


def fps_hud(width: int = 640, height: int = 360) -> PixelBuffer:
    buffer = PixelBuffer.new(width, height, (18, 22, 28))
    _panel(buffer, 16, 16, 260, 92)
    draw.text(buffer, (28, 28), "FPS HUD", (245, 248, 255), scale=2)
    draw.text(buffer, (28, 54), "SPD 4.8 M/S    AMMO 024", (230, 238, 245), scale=1)
    _bar(buffer, 28, 78, 150, 12, 0.84, (62, 218, 118))
    draw.text(buffer, (190, 76), "HP 084", (230, 238, 245), scale=1)
    draw.line(buffer, (width // 2 - 14, height // 2), (width // 2 - 4, height // 2), (245, 248, 255))
    draw.line(buffer, (width // 2 + 4, height // 2), (width // 2 + 14, height // 2), (245, 248, 255))
    draw.line(buffer, (width // 2, height // 2 - 14), (width // 2, height // 2 - 4), (245, 248, 255))
    draw.line(buffer, (width // 2, height // 2 + 4), (width // 2, height // 2 + 14), (245, 248, 255))
    _panel(buffer, width - 196, 22, 172, 46)
    draw.text(buffer, (width - 184, 34), "LEVEL UP READY", (255, 230, 92), scale=1)
    return buffer


def third_person_hud(width: int = 640, height: int = 360) -> PixelBuffer:
    buffer = PixelBuffer.new(width, height, (22, 24, 26))
    _panel(buffer, 18, height - 112, 292, 84)
    draw.text(buffer, (30, height - 100), "THIRD PERSON HUD", (245, 248, 255), scale=2)
    draw.text(buffer, (30, height - 72), "SPD 3.1 M/S    ARMOR 042", (230, 238, 245), scale=1)
    _bar(buffer, 30, height - 48, 170, 12, 0.96, (96, 156, 255))
    draw.text(buffer, (214, height - 50), "HP 096", (230, 238, 245), scale=1)
    _panel(buffer, width - 214, height - 92, 188, 58)
    draw.text(buffer, (width - 202, height - 78), "LEVEL 2", (255, 230, 92), scale=2)
    draw.text(buffer, (width - 202, height - 50), "CAMERA ORBIT", (230, 238, 245), scale=1)
    draw.circle(buffer, (width // 2, height // 2 + 42), 26, (210, 210, 210), fill=False)
    draw.circle(buffer, (width // 2, height // 2 + 42), 3, (245, 248, 255), fill=True)
    return buffer


def render_hud_examples(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fps_path = output_dir / "fps_hud_demo.png"
    third_path = output_dir / "third_person_hud_demo.png"
    fps_hud().to_png(fps_path)
    third_person_hud().to_png(third_path)
    return fps_path, third_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FPS and third-person HUD example PNGs.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    fps_path, third_path = render_hud_examples(args.output_dir)
    print(f"Wrote {fps_path}")
    print(f"Wrote {third_path}")


if __name__ == "__main__":
    main()
