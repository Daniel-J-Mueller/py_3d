"""Live capsule walking and camera-mode demo."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

from py_3d import Box, Camera, Capsule, Color, Lamp, Material, RenderEngine, RenderSettings, Scene, Sun, TextBulletin, Vec3


OUTPUT_DIR = Path("renderings-tests") / "live-renders"


@dataclass
class CapsuleController:
    feet: Vec3 = Vec3(0.0, 0.02, 0.0)
    velocity: Vec3 = Vec3(0.0, 0.0, 0.0)
    yaw_degrees: float = 0.0
    radius: float = 0.26
    height: float = 1.45
    speed: float = 2.0
    jump_speed: float = 4.2
    on_ground: bool = False

    @property
    def center(self) -> Vec3:
        return Vec3(self.feet.x, self.feet.y + self.height * 0.5, self.feet.z)

    @property
    def forward(self) -> Vec3:
        yaw = radians(self.yaw_degrees)
        return Vec3(sin(yaw), 0.0, cos(yaw)).normalized(Vec3(0.0, 0.0, 1.0))

    @property
    def right(self) -> Vec3:
        forward = self.forward
        return Vec3(forward.z, 0.0, -forward.x)

    def step(self, dt: float, keys: set[str], blocks: tuple[Box, ...]) -> None:
        move = Vec3(0.0, 0.0, 0.0)
        if "w" in keys:
            move = move + self.forward
        if "s" in keys:
            move = move - self.forward
        if "d" in keys:
            move = move + self.right
        if "a" in keys:
            move = move - self.right
        if move.length_squared() > 0.0:
            move = move.normalized()
        horizontal = move * self.speed
        self.velocity = Vec3(horizontal.x, self.velocity.y - 9.81 * dt, horizontal.z)
        if "space" in keys and self.on_ground:
            self.velocity = Vec3(self.velocity.x, self.jump_speed, self.velocity.z)
            self.on_ground = False

        self.feet = self.feet + self.velocity * dt
        self._resolve_world(blocks)

    def _resolve_world(self, blocks: tuple[Box, ...]) -> None:
        self.on_ground = False
        if self.feet.y < 0.0:
            self.feet = Vec3(self.feet.x, 0.0, self.feet.z)
            self.velocity = Vec3(self.velocity.x, 0.0, self.velocity.z)
            self.on_ground = True
        for block in blocks:
            self._resolve_box(block)

    def _resolve_box(self, block: Box) -> None:
        min_x, max_x = self.feet.x - self.radius, self.feet.x + self.radius
        min_y, max_y = self.feet.y, self.feet.y + self.height
        min_z, max_z = self.feet.z - self.radius, self.feet.z + self.radius
        half = block.size * 0.5
        box_min = block.center - half
        box_max = block.center + half
        if max_x <= box_min.x or min_x >= box_max.x or max_y <= box_min.y or min_y >= box_max.y or max_z <= box_min.z or min_z >= box_max.z:
            return
        overlaps = (
            (box_max.x - min_x, Vec3(1, 0, 0)),
            (max_x - box_min.x, Vec3(-1, 0, 0)),
            (box_max.y - min_y, Vec3(0, 1, 0)),
            (max_y - box_min.y, Vec3(0, -1, 0)),
            (box_max.z - min_z, Vec3(0, 0, 1)),
            (max_z - box_min.z, Vec3(0, 0, -1)),
        )
        penetration, normal = min(overlaps, key=lambda item: item[0])
        self.feet = self.feet + normal * penetration
        if normal.y > 0.0:
            self.on_ground = True
        if normal.x:
            self.velocity = Vec3(0.0, self.velocity.y, self.velocity.z)
        elif normal.y:
            self.velocity = Vec3(self.velocity.x, 0.0, self.velocity.z)
        elif normal.z:
            self.velocity = Vec3(self.velocity.x, self.velocity.y, 0.0)


class CapsuleWalkViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk

        self.tk = tk
        self.args = args
        self.controller = CapsuleController()
        self.keys: set[str] = set()
        self.camera_mode = args.camera_mode
        self.engine = make_engine(args.renderer)
        self.settings = RenderSettings(
            width=args.width,
            height=args.height,
            background=(7, 9, 13),
            ambient=args.ambient,
            gamma=args.gamma,
            smooth_shading=True,
            sphere_segments=18,
            sphere_rings=9,
        )
        self.blocks = make_blocks()
        self.root = tk.Tk()
        self.root.title("py_3d capsule walk")
        self.root.geometry(f"{args.window_width}x{args.window_height}")
        self.canvas = tk.Canvas(self.root, bg="#07090d", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.photo = None
        self.canvas.bind("<Button-1>", lambda event: self.canvas.focus_set())
        self.root.bind("<KeyPress>", self.on_key_down)
        self.root.bind("<KeyRelease>", self.on_key_up)

    def run(self) -> None:
        self.render_once()
        self.root.after(max(1, int(1000 / self.args.fps)), self.tick)
        self.root.mainloop()

    def tick(self) -> None:
        self.controller.step(1.0 / self.args.fps, self.keys, self.blocks)
        self.render_once()
        self.root.after(max(1, int(1000 / self.args.fps)), self.tick)

    def on_key_down(self, event) -> None:
        key = event.keysym.lower()
        if key == "v":
            self.camera_mode = {"global": "third", "third": "first", "first": "global"}[self.camera_mode]
        elif key == "left":
            self.controller.yaw_degrees -= 5.0
        elif key == "right":
            self.controller.yaw_degrees += 5.0
        else:
            self.keys.add(key)

    def on_key_up(self, event) -> None:
        self.keys.discard(event.keysym.lower())

    def camera(self) -> Camera:
        if self.camera_mode == "first":
            return Camera.first_person(self.controller.feet, self.controller.forward, eye_height=self.controller.height * 0.78)
        if self.camera_mode == "third":
            return Camera.third_person(self.controller.feet, self.controller.forward, distance=3.2, height=1.55)
        return Camera(position=(3.2, 2.5, -5.2), target=(0.4, 0.65, 0.4), fov_degrees=54)

    def scene(self) -> Scene:
        scene = Scene()
        scene.add(Box((0, -0.06, 0), (7.0, 0.12, 7.0), Material(color=(54, 84, 88), roughness=0.55)))
        scene.add(*self.blocks)
        scene.add(Capsule(self.controller.center, self.controller.radius, self.controller.height, Material(color=(92, 160, 240), roughness=0.28, specular=0.2, shininess=24)))
        scene.add_light(Sun(direction=(-0.35, -0.85, -0.6), color=(255, 245, 230), intensity=0.9))
        scene.add_light(Lamp(position=(1.6, 2.4, -1.2), color=(120, 170, 255), intensity=4.2))
        scene.add_bulletin(
            TextBulletin(
                f"CAPSULE WALK\n{self.camera_mode.upper()} CAMERA",
                position=(10, 10),
                color=(245, 248, 255),
                background=(4, 6, 10),
                padding=5,
            )
        )
        return scene

    def render_once(self) -> None:
        buffer = self.engine.render(self.scene(), self.camera(), self.settings)
        image = buffer
        if self.args.fit_window:
            width = max(1, self.canvas.winfo_width())
            height = max(1, self.canvas.winfo_height())
            image = buffer.resized_nearest(width, height)
        self.photo = self._photo_from_buffer(image)
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def _photo_from_buffer(self, buffer):
        ppm = buffer.to_ppm_bytes()
        try:
            return self.tk.PhotoImage(data=ppm, format="PPM")
        except self.tk.TclError:
            return self.tk.PhotoImage(data=base64.b64encode(ppm).decode("ascii"), format="PPM")


def make_engine(renderer: str) -> RenderEngine:
    if renderer == "py_gpu":
        from py_gpu.adapters.py3d import Py3DRasterRenderer

        return RenderEngine(Py3DRasterRenderer())
    return RenderEngine()


def make_blocks() -> tuple[Box, ...]:
    block_material = Material(color=(150, 118, 82), roughness=0.62, fuzziness=0.08, specular=0.04)
    return (
        Box((0.8, 0.2, 1.0), (0.8, 0.4, 0.8), block_material),
        Box((-0.9, 0.35, 1.6), (0.7, 0.7, 0.7), block_material),
        Box((1.7, 0.55, 2.0), (0.8, 1.1, 0.8), block_material),
        Box((-1.7, 0.15, -0.6), (1.0, 0.3, 0.7), block_material),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live capsule walking demo with camera modes.")
    parser.add_argument("--renderer", choices=("cpu", "py_gpu"), default="py_gpu")
    parser.add_argument("--camera-mode", choices=("global", "third", "first"), default="third")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--fit-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ambient", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    CapsuleWalkViewer(args).run()


if __name__ == "__main__":
    main()
