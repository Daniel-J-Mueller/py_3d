"""Kinematic fruit bowl demo with live and offline rendering paths."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from math import cos, pi, radians, sin, tau
import os
from pathlib import Path
import shutil
import subprocess

from py_3d import (
    Camera,
    CompoundSphereCollider,
    CPURenderer,
    KinematicBowl,
    Lamp,
    Line3,
    Material,
    Mesh,
    PhysicsWorld,
    RenderEngine,
    RenderSettings,
    Scene,
    Sphere,
    SphereBody,
    SphereCollider,
    StaticPlane,
    Sun,
    SurfacePerturbation,
    TextBulletin,
    Triangle,
    Vec3,
)


OUTPUT_DIR = Path("renderings-tests")


@dataclass
class Fruit:
    name: str
    body: SphereBody
    visual: str = "sphere"
    banana_yaw: float = 0.0
    marker_color: tuple[int, int, int] = (40, 30, 25)

    def to_primitives(self) -> tuple[Sphere | Mesh | Line3, ...]:
        if self.visual == "banana":
            return (
                banana_mesh(
                    self.body.position,
                    self.body.radius,
                    self.body.material,
                    yaw_degrees=self.banana_yaw,
                    rotation=self.body.rotation,
                ),
            )
        marker = rotate_euler(Vec3(0.0, self.body.radius * 1.03, 0.0), self.body.rotation)
        marker_tail = rotate_euler(Vec3(self.body.radius * 0.42, self.body.radius * 0.72, 0.0), self.body.rotation)
        return (
            self.body.to_primitive(),
            Line3(
                self.body.position + marker_tail,
                self.body.position + marker,
                Material(color=self.marker_color, emission=(12, 10, 8)),
            ),
        )

    def sync_collision_boundary(self) -> None:
        if self.visual == "banana":
            self.body.collision_boundary = banana_collider(
                self.body.radius,
                yaw_degrees=self.banana_yaw,
                rotation=self.body.rotation,
            )


def banana_mesh(
    center: Vec3,
    radius: float,
    material: Material,
    *,
    yaw_degrees: float = 0.0,
    rotation: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0),
    sections: int = 10,
    sides: int = 8,
) -> Mesh:
    """Return a small curved tube mesh suitable for a banana-like fruit."""

    if sections < 2:
        raise ValueError("banana sections must be at least 2")
    if sides < 3:
        raise ValueError("banana sides must be at least 3")

    rotation = as_rotation(rotation, yaw_degrees)
    centerline = banana_centerline_offsets(radius, sections=sections, rotation=rotation)
    rings: list[list[Vec3]] = []

    for section in range(sections + 1):
        amount = section / sections
        section_center = centerline[section]
        previous_center = centerline[max(0, section - 1)]
        next_center = centerline[min(sections, section + 1)]
        tangent = (next_center - previous_center).normalized(Vec3(1.0, 0.0, 0.0))
        binormal = rotate_euler(Vec3(0.0, 0.0, 1.0), rotation)
        normal = binormal.cross(tangent).normalized(Vec3(0.0, 1.0, 0.0))
        taper = sin(pi * amount) ** 0.5
        tube_radius = radius * 0.24 * (0.35 + 0.65 * taper)
        ring = []
        for side in range(sides):
            theta = tau * side / sides
            local = section_center + normal * (cos(theta) * tube_radius) + binormal * (sin(theta) * tube_radius)
            ring.append(center + local)
        rings.append(ring)

    triangles: list[Triangle] = []
    for section in range(sections):
        for side in range(sides):
            next_side = (side + 1) % sides
            top_left = rings[section][side]
            top_right = rings[section][next_side]
            bottom_left = rings[section + 1][side]
            bottom_right = rings[section + 1][next_side]
            u = side / sides
            next_u = 1.0 if next_side == 0 else next_side / sides
            v = section / sections
            next_v = (section + 1) / sections
            triangles.append(Triangle(top_left, bottom_left, top_right, material, (u, v), (u, next_v), (next_u, v)))
            triangles.append(Triangle(top_right, bottom_left, bottom_right, material, (next_u, v), (u, next_v), (next_u, next_v)))

    cap_material = Material(color=(112, 82, 34), roughness=0.4)
    for ring_index, reverse in ((0, True), (-1, False)):
        cap_center = _ring_center(rings[ring_index])
        for side in range(sides):
            next_side = (side + 1) % sides
            a = rings[ring_index][next_side if reverse else side]
            b = rings[ring_index][side if reverse else next_side]
            triangles.append(Triangle(a, b, cap_center, cap_material))
    return Mesh(triangles)


def banana_collider(
    radius: float,
    *,
    yaw_degrees: float = 0.0,
    rotation: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0),
    sections: int = 10,
) -> CompoundSphereCollider:
    offsets = banana_centerline_offsets(radius, sections=sections, rotation=as_rotation(rotation, yaw_degrees))
    return CompoundSphereCollider.from_offsets(offsets, radius=radius * 0.2)


def banana_centerline_offsets(
    radius: float,
    *,
    sections: int = 10,
    rotation: Vec3 = Vec3(0.0, 0.0, 0.0),
) -> list[Vec3]:
    curve_radius = radius * 1.55
    height_scale = 0.42
    start_angle = -1.12
    end_angle = 1.12
    base_y = curve_radius * height_scale * cos(start_angle)
    offsets = []
    for section in range(sections + 1):
        amount = section / sections
        angle = start_angle + (end_angle - start_angle) * amount
        local = Vec3(
            curve_radius * sin(angle),
            curve_radius * height_scale * cos(angle) - base_y - radius * 0.12,
            0.0,
        )
        offsets.append(rotate_euler(local, rotation))
    return offsets


def as_rotation(rotation: Vec3 | tuple[float, float, float], yaw_degrees: float) -> Vec3:
    value = rotation if isinstance(rotation, Vec3) else Vec3(*rotation)
    return Vec3(value.x, value.y + radians(yaw_degrees), value.z)


def rotate_euler(value: Vec3, rotation: Vec3) -> Vec3:
    cx, sx = cos(rotation.x), sin(rotation.x)
    cy, sy = cos(rotation.y), sin(rotation.y)
    cz, sz = cos(rotation.z), sin(rotation.z)
    x_rotated = Vec3(value.x, value.y * cx - value.z * sx, value.y * sx + value.z * cx)
    y_rotated = Vec3(x_rotated.x * cy + x_rotated.z * sy, x_rotated.y, -x_rotated.x * sy + x_rotated.z * cy)
    return Vec3(y_rotated.x * cz - y_rotated.y * sz, y_rotated.x * sz + y_rotated.y * cz, y_rotated.z)


def _ring_center(ring: list[Vec3]) -> Vec3:
    total = Vec3(0.0, 0.0, 0.0)
    for point in ring:
        total = total + point
    return total / len(ring)


class FruitBowlSimulation:
    """Small coordinated scene: driven bowl, dynamic fruit."""

    def __init__(self) -> None:
        self.time = 0.0
        self.world = PhysicsWorld(gravity=(0.0, -9.81, 0.0))
        self.bowl = KinematicBowl(
            center=self._driven_center(0.0),
            radius=1.35,
            depth=0.96,
            restitution=0.82,
            friction=0.04,
            material=Material(color=(145, 95, 210), absorption=(0.08, 0.1, 0.03), roughness=0.35, fuzziness=0.08),
        )
        self.floor = StaticPlane(
            point=(0.0, -1.75, 0.0),
            normal=(0.0, 1.0, 0.0),
            friction=0.28,
            restitution=0.35,
            material=Material(color=(52, 89, 92), absorption=(0.14, 0.08, 0.05), roughness=0.55),
            size=6.0,
        )
        self.fruits = [
            Fruit(
                "apple",
                SphereBody(
                    position=(-0.42, -0.12, 0.08),
                    radius=0.23,
                    velocity=(0.6, 0.0, 0.12),
                    mass=0.85,
                    restitution=0.74,
                    friction=0.28,
                    static_friction=0.42,
                    kinetic_friction=0.28,
                    material=Material(color=(225, 55, 48), absorption=(0.02, 0.15, 0.16), roughness=0.22, fuzziness=0.08),
                ),
            ),
            Fruit(
                "orange",
                SphereBody(
                    position=(0.28, -0.2, -0.1),
                    radius=0.27,
                    velocity=(-0.35, 0.12, -0.22),
                    mass=1.05,
                    restitution=0.7,
                    friction=0.34,
                    static_friction=0.48,
                    kinetic_friction=0.32,
                    material=Material(color=(238, 135, 42), absorption=(0.02, 0.08, 0.22), roughness=0.38, fuzziness=0.18),
                    visual_perturbation=SurfacePerturbation(magnitude=0.035, scale=5.5, seed=31, octaves=4, gain=0.55),
                    collision_boundary=SphereCollider(radius=0.27),
                ),
            ),
            Fruit(
                "lemon",
                SphereBody(
                    position=(0.0, 0.12, 0.3),
                    radius=0.19,
                    velocity=(0.2, -0.1, -0.3),
                    mass=0.55,
                    restitution=0.76,
                    friction=0.31,
                    static_friction=0.46,
                    kinetic_friction=0.3,
                    material=Material(color=(238, 218, 74), absorption=(0.03, 0.03, 0.2), roughness=0.32, fuzziness=0.16),
                    visual_perturbation=SurfacePerturbation(magnitude=0.025, scale=6.4, seed=44, octaves=4, gain=0.55),
                    collision_boundary=SphereCollider(radius=0.19),
                ),
            ),
            Fruit(
                "watermelon",
                SphereBody(
                    position=(-0.05, 0.22, -0.36),
                    radius=0.21,
                    velocity=(-0.18, 0.0, 0.38),
                    mass=1.15,
                    restitution=0.72,
                    friction=0.18,
                    static_friction=0.28,
                    kinetic_friction=0.18,
                    material=Material(color=(50, 142, 78), absorption=(0.16, 0.03, 0.14), roughness=0.04, fuzziness=0.0),
                ),
                marker_color=(18, 55, 34),
            ),
            Fruit(
                "banana",
                SphereBody(
                    position=(0.52, 0.06, 0.24),
                    radius=0.27,
                    velocity=(-0.52, 0.06, -0.28),
                    angular_velocity=(0.0, 0.0, 2.5),
                    mass=0.65,
                    moment_of_inertia=0.028,
                    restitution=0.78,
                    friction=0.18,
                    static_friction=0.35,
                    kinetic_friction=0.22,
                    material=Material(color=(244, 214, 78), absorption=(0.02, 0.04, 0.2), roughness=0.18, fuzziness=0.04),
                ),
                visual="banana",
                banana_yaw=26.0,
            ),
        ]
        for fruit in self.fruits:
            fruit.sync_collision_boundary()
        self.world.add_bowl(self.bowl)
        self.world.add_plane(self.floor)
        for fruit in self.fruits:
            self.world.add_sphere(fruit.body)

    def step(self, dt: float, substeps: int = 3) -> None:
        if dt <= 0.0:
            return
        step_dt = dt / substeps
        for _ in range(substeps):
            self.time += step_dt
            self.bowl.set_center(self._driven_center(self.time), dt=step_dt)
            for fruit in self.fruits:
                fruit.sync_collision_boundary()
            self.world.step(step_dt)

    @staticmethod
    def _driven_center(time: float) -> Vec3:
        return Vec3(
            0.12 * sin(time * 1.7),
            0.2 + 0.3 * sin(time * tau * 1.08),
            0.08 * sin(time * 2.1 + 0.6),
        )

    def scene(self, *, label: str = "KINEMATIC FRUIT BOWL") -> Scene:
        scene = Scene()
        scene.add(self.floor.to_primitive(), self.bowl.to_primitive())
        for fruit in self.fruits:
            scene.add(*fruit.to_primitives())
        scene.add_light(Sun(direction=(-0.35, -0.8, -1.0), color=(255, 245, 224), intensity=0.72))
        scene.add_light(Lamp(position=(-1.8, 2.4, -2.2), color=(95, 145, 255), intensity=3.2))
        scene.add_light(Lamp(position=(1.6, 1.3, -1.3), color=(255, 120, 88), intensity=2.1))
        scene.add_light(Lamp(position=(0.2, 2.1, 1.6), color=(110, 255, 170), intensity=1.35))
        scene.add_bulletin(
            TextBulletin(
                f"{label}\nDRIVEN BOWL AND FREE DYNAMIC FRUIT",
                position=(10, 10),
                color=(246, 248, 255),
                background=(5, 7, 11),
                padding=5,
                scale=1,
            )
        )
        return scene


def make_engine() -> RenderEngine:
    return RenderEngine(CPURenderer(cache_static_geometry=False))


def make_settings(args: argparse.Namespace) -> RenderSettings:
    return RenderSettings(
        width=args.width,
        height=args.height,
        background=(8, 11, 15),
        ambient=0.09,
        sphere_segments=args.sphere_segments,
        sphere_rings=args.sphere_rings,
    )


def make_camera(yaw_degrees: float = 0.0, pitch_degrees: float = 50.0, distance: float = 4.2) -> Camera:
    target = Vec3(0.0, -0.28, 0.0)
    yaw = radians(yaw_degrees)
    pitch = radians(max(-80.0, min(80.0, pitch_degrees)))
    offset = Vec3(
        distance * sin(yaw) * cos(pitch),
        distance * sin(pitch),
        -distance * cos(yaw) * cos(pitch),
    )
    return Camera(position=target + offset, target=target, fov_degrees=48)


def render_still(args: argparse.Namespace) -> Path:
    simulation = FruitBowlSimulation()
    for _ in range(max(0, int(args.warmup * args.fps))):
        simulation.step(1.0 / args.fps)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_engine().render(simulation.scene(), make_camera(), make_settings(args)).to_png(output_path)
    print(f"Wrote {output_path}")
    return output_path


def find_ffmpeg(explicit_path: str | Path | None = None) -> str | None:
    if explicit_path is not None:
        explicit = str(explicit_path)
        path = Path(explicit)
        if path.exists():
            return str(path)
        found = shutil.which(explicit)
        if found is not None:
            return found

    env_path = os.environ.get("FFMPEG_BINARY")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return str(path)
        found = shutil.which(env_path)
        if found is not None:
            return found

    found = shutil.which("ffmpeg")
    if found is not None:
        return found

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffmpeg_missing_message() -> str:
    return (
        "ffmpeg executable not found. `pip install ffmpeg` installs a Python module, "
        "not ffmpeg.exe. Install the FFmpeg command-line binary, install optional "
        "`imageio-ffmpeg`, set FFMPEG_BINARY, or pass --ffmpeg C:\\path\\to\\ffmpeg.exe."
    )


def render_video(args: argparse.Namespace) -> Path:
    output_path = Path(args.video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulation = FruitBowlSimulation()
    engine = make_engine()
    settings = make_settings(args)
    ffmpeg = find_ffmpeg(getattr(args, "ffmpeg", None))
    if ffmpeg is None:
        message = ffmpeg_missing_message()
        if getattr(args, "require_ffmpeg", False):
            raise RuntimeError(message)
        frames_dir = output_path.with_suffix("") if output_path.suffix else output_path
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(args.frames):
            simulation.step(1.0 / args.fps)
            buffer = engine.render(simulation.scene(label=f"FRUIT BOWL FRAME {frame:03d}"), make_camera(), settings)
            buffer.to_png(frames_dir / f"frame_{frame:04d}.png")
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
            simulation.step(1.0 / args.fps)
            buffer = engine.render(simulation.scene(label=f"FRUIT BOWL FRAME {frame:03d}"), make_camera(), settings)
            process.stdin.write(buffer.to_ppm_bytes())
    finally:
        process.stdin.close()
    result = process.wait()
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result}")
    print(f"Wrote {output_path}")
    return output_path


class LiveFruitBowlViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk

        self.tk = tk
        self.args = args
        self.simulation = FruitBowlSimulation()
        self.engine = make_engine()
        self.settings = make_settings(args)
        self.yaw = 0.0
        self.pitch = 50.0
        self.distance = 4.2
        self.target = Vec3(0.0, -0.28, 0.0)
        self.drag_start: tuple[int, int] | None = None
        self.paused = False
        self.base_photo = None
        self.photo = None
        self.window_icon = None

        self.root = tk.Tk()
        self.root.title("py_3d fruit bowl")
        self.root.geometry(f"{args.window_width}x{args.window_height}")
        self._set_window_icon()
        self.canvas = tk.Canvas(self.root, bg="#080b0f", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", lambda event: self.render_once())
        for key in ("<Left>", "<Right>", "<Up>", "<Down>", "<w>", "<s>", "<a>", "<d>", "<q>", "<e>", "<p>", "<space>", "<r>"):
            self.root.bind(key, self.on_key)

    def run(self) -> None:
        self.render_once()
        self.root.after(max(1, int(1000 / self.args.fps)), self.tick)
        self.root.mainloop()

    def tick(self) -> None:
        if not self.paused:
            self.simulation.step(1.0 / self.args.fps)
        self.render_once()
        self.root.after(max(1, int(1000 / self.args.fps)), self.tick)

    def camera(self) -> Camera:
        yaw = radians(self.yaw)
        pitch = radians(max(-80.0, min(80.0, self.pitch)))
        offset = Vec3(
            self.distance * sin(yaw) * cos(pitch),
            self.distance * sin(pitch),
            -self.distance * cos(yaw) * cos(pitch),
        )
        return Camera(position=self.target + offset, target=self.target, fov_degrees=48)

    def _set_window_icon(self) -> None:
        icon_path = Path(__file__).resolve().parents[1] / "assets" / "py_3d_logo.png"
        if not icon_path.exists():
            return
        try:
            self.window_icon = self.tk.PhotoImage(file=str(icon_path))
            self.root.iconphoto(True, self.window_icon)
        except self.tk.TclError as exc:
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

    def on_release(self, event) -> None:
        del event
        self.drag_start = None

    def on_mouse_wheel(self, event) -> None:
        self.distance = max(1.6, self.distance * (0.9 if event.delta > 0 else 1.1))

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
            self.distance = max(1.6, self.distance * 0.9)
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
        elif key == "space":
            self.paused = not self.paused
        elif key == "r":
            self.simulation = FruitBowlSimulation()

    def render_once(self) -> None:
        output = self.engine.render(self.simulation.scene(), self.camera(), self.settings)
        self.base_photo = self._photo_from_buffer(output)
        self.photo = self.base_photo
        if self.args.fit_window:
            width = max(1, self.canvas.winfo_width())
            height = max(1, self.canvas.winfo_height())
            scale_x = width // output.width
            scale_y = height // output.height
            if scale_x >= 1 and scale_y >= 1 and output.width * scale_x == width and output.height * scale_y == height:
                self.photo = self.base_photo.zoom(scale_x, scale_y)
            else:
                self.photo = self._photo_from_buffer(output.resized_nearest(width, height))
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def save_snapshot(self) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "fruit_bowl_live_snapshot.png"
        self.engine.render(self.simulation.scene(), self.camera(), self.settings).to_png(path)
        print(f"Wrote {path}")

    def _photo_from_buffer(self, buffer):
        ppm = buffer.to_ppm_bytes()
        try:
            return self.tk.PhotoImage(data=ppm, format="PPM")
        except self.tk.TclError:
            return self.tk.PhotoImage(data=base64.b64encode(ppm).decode("ascii"), format="PPM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render or run a live kinematic fruit bowl demo.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "fruit_bowl.png")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--require-ffmpeg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-still", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--warmup", type=float, default=1.25)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=204)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--fit-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sphere-segments", type=int, default=14)
    parser.add_argument("--sphere-rings", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.frames <= 0:
        raise ValueError("frames must be positive")
    if args.live:
        LiveFruitBowlViewer(args).run()
        return
    if args.video is not None:
        if args.write_still:
            render_still(args)
        render_video(args)
        return
    render_still(args)


if __name__ == "__main__":
    main()
