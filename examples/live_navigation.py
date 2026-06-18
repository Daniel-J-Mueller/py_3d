"""Live CPU rendering example with clicked-in camera navigation.

This example uses only the Python standard library. It renders into an
off-screen buffer at ``--render-width`` x ``--render-height`` and displays that
buffer in a separately sized Tk window.
"""

from __future__ import annotations

import argparse
import base64
from math import radians, sin, cos
from pathlib import Path
import tkinter as tk

from py_3d import Box, Camera, Lamp, Material, RenderEngine, RenderSettings, Scene, Sphere, Sun, TextBulletin, Vec3


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
            "CLICK TO NAVIGATE\nDRAG/ARROWS ORBIT  W/S ZOOM\nA/D PAN  Q/E LIFT  P SAVE",
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
        self.base_photo: tk.PhotoImage | None = None
        self.photo: tk.PhotoImage | None = None
        self.window_icon: tk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("py_3d live navigation")
        self.root.geometry(f"{args.window_width}x{args.window_height}")
        self.set_window_icon()
        self.canvas = tk.Canvas(self.root, bg="#07090e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", lambda event: self.render_once())
        for key in ("<Left>", "<Right>", "<Up>", "<Down>", "<w>", "<s>", "<a>", "<d>", "<q>", "<e>", "<p>"):
            self.root.bind(key, self.on_key)

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
        self.render_once()
        self.root.mainloop()

    def set_window_icon(self) -> None:
        icon_path = Path(__file__).resolve().parents[1] / "assets" / "py_3d_logo.png"
        if not icon_path.exists():
            return
        try:
            self.window_icon = tk.PhotoImage(file=str(icon_path))
            self.root.iconphoto(True, self.window_icon)
        except tk.TclError as exc:
            print(f"Could not load window icon {icon_path}: {exc}")

    def on_click(self, event) -> None:
        self.canvas.focus_set()
        self.drag_start = (event.x, event.y)

    def on_drag(self, event) -> None:
        if self.drag_start is None:
            return
        last_x, last_y = self.drag_start
        self.yaw += (event.x - last_x) * 0.35
        self.pitch += (event.y - last_y) * 0.25
        self.drag_start = (event.x, event.y)
        self.render_once()

    def on_release(self, event) -> None:
        del event
        self.drag_start = None

    def on_mouse_wheel(self, event) -> None:
        self.distance = max(1.3, self.distance * (0.9 if event.delta > 0 else 1.1))
        self.render_once()

    def on_key(self, event) -> None:
        key = event.keysym.lower()
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
        self.render_once()

    def render_once(self) -> None:
        output = self.engine.render(self.scene, self.camera(), self.settings)
        self.base_photo = self._photo_from_buffer(output)
        self.photo = self.base_photo
        if self.args.fit_window:
            width = max(1, self.canvas.winfo_width())
            height = max(1, self.canvas.winfo_height())
            scale_x = width // output.width
            scale_y = height // output.height
            if (
                scale_x >= 1
                and scale_y >= 1
                and output.width * scale_x == width
                and output.height * scale_y == height
            ):
                self.photo = self.base_photo.zoom(scale_x, scale_y)
            else:
                self.photo = self._photo_from_buffer(output.resized_nearest(width, height))
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def save_snapshot(self) -> None:
        output_dir = Path("renderings-tests")
        output_dir.mkdir(exist_ok=True)
        path = output_dir / "live_navigation_snapshot.png"
        self.engine.render(self.scene, self.camera(), self.settings).to_png(path)
        print(f"Wrote {path}")

    @staticmethod
    def _photo_from_buffer(buffer):
        ppm = buffer.to_ppm_bytes()
        try:
            return tk.PhotoImage(data=ppm, format="PPM")
        except tk.TclError:
            return tk.PhotoImage(data=base64.b64encode(ppm).decode("ascii"), format="PPM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live py_3d CPU renderer with navigation.")
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=180)
    parser.add_argument("--fit-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wireframe", action="store_true")
    parser.add_argument("--sphere-segments", type=int, default=20)
    parser.add_argument("--sphere-rings", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    LiveViewer(parse_args()).run()
