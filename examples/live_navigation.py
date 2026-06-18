"""Live CPU rendering example with native py_3d camera navigation."""

from __future__ import annotations

import argparse
from math import cos, radians, sin
from pathlib import Path
from time import sleep

from py_3d import Box, Camera, Lamp, Material, PixelWindow, RenderEngine, RenderSettings, Scene, Sphere, Sun, TextBulletin, Vec3


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
    scene.add_bulletin(
        TextBulletin(
            "DRAG/ARROWS ORBIT  W/S ZOOM\nA/D PAN  Q/E LIFT  P SAVE",
            position=(8, 8),
            color=(245, 248, 255),
            background=(5, 7, 11),
            padding=5,
            scale=1,
        )
    )
    return scene


class LiveViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.engine = RenderEngine()
        self.scene = make_scene()
        self.settings = RenderSettings(
            width=args.render_width,
            height=args.render_height,
            background=(7, 9, 14),
            ambient=0.06,
            wireframe=args.wireframe,
            sphere_segments=args.sphere_segments,
            sphere_rings=args.sphere_rings,
        )
        self.target = Vec3(0.0, 0.0, 0.0)
        self.distance = 4.0
        self.yaw = 0.0
        self.pitch = 12.0
        self.drag_start: tuple[int, int] | None = None
        self.window = PixelWindow(args.window_width, args.window_height, title="py_3d live navigation", fit_window=args.fit_window)

    def camera(self) -> Camera:
        yaw = radians(self.yaw)
        pitch = radians(max(-80.0, min(80.0, self.pitch)))
        offset = Vec3(
            self.distance * sin(yaw) * cos(pitch),
            self.distance * sin(pitch),
            -self.distance * cos(yaw) * cos(pitch),
        )
        return Camera(position=self.target + offset, target=self.target, fov_degrees=52)

    def run(self) -> None:
        while not self.window.closed:
            changed = False
            for event in self.window.poll_events():
                if event.kind == "quit":
                    self.window.close()
                elif event.kind == "key_down":
                    if event.key == "escape":
                        self.window.close()
                    else:
                        changed = self.on_key(event.key) or changed
                elif event.kind == "button" and event.button == 1:
                    self.drag_start = event.pos
                elif event.kind == "button_up" and event.button == 1:
                    self.drag_start = None
                elif event.kind == "motion" and self.drag_start is not None:
                    self.on_drag(event.pos)
                    changed = True
                elif event.kind == "wheel":
                    self.distance = max(1.3, self.distance * (0.9 if event.y > 0 else 1.1))
                    changed = True
            if changed or self.drag_start is None:
                self.render_once()
            sleep(1.0 / 60.0)

    def on_drag(self, position: tuple[int, int]) -> None:
        if self.drag_start is None:
            return
        last_x, last_y = self.drag_start
        self.yaw += (position[0] - last_x) * 0.35
        self.pitch += (position[1] - last_y) * 0.25
        self.drag_start = position

    def on_key(self, key: str) -> bool:
        if key == "left":
            self.yaw -= 5.0
        elif key == "right":
            self.yaw += 5.0
        elif key == "up":
            self.pitch -= 4.0
        elif key == "down":
            self.pitch += 4.0
        elif key == "w":
            self.distance = max(1.3, self.distance * 0.9)
        elif key == "s":
            self.distance *= 1.1
        elif key == "a":
            self.target = self.target + self.camera().basis()[0] * -0.15
        elif key == "d":
            self.target = self.target + self.camera().basis()[0] * 0.15
        elif key == "q":
            self.target = self.target + Vec3(0.0, 0.15, 0.0)
        elif key == "e":
            self.target = self.target + Vec3(0.0, -0.15, 0.0)
        elif key == "p":
            self.save_snapshot()
        else:
            return False
        return True

    def render_once(self) -> None:
        self.window.show(self.engine.render(self.scene, self.camera(), self.settings))

    def save_snapshot(self) -> None:
        output_dir = Path("renderings-tests")
        output_dir.mkdir(exist_ok=True)
        path = output_dir / "live_navigation_snapshot.png"
        self.engine.render(self.scene, self.camera(), self.settings).to_png(path)
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live py_3d CPU renderer with navigation.")
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--render-width", type=int, default=480)
    parser.add_argument("--render-height", type=int, default=270)
    parser.add_argument("--fit-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wireframe", action="store_true")
    parser.add_argument("--sphere-segments", type=int, default=20)
    parser.add_argument("--sphere-rings", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    LiveViewer(parse_args()).run()
