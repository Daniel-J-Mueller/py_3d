"""Live capsule walking and camera-mode demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

from py_3d import Box, Camera, Color, HUDRect, HUDText, Lamp, Material, PixelWindow, PlayerModel, RenderEngine, RenderSettings, Scene, SkyPrefab, Sun, TextBulletin, Vec3


OUTPUT_DIR = Path("renderings-tests") / "live-renders"


@dataclass
class CapsuleController:
    feet: Vec3 = Vec3(0.0, 0.02, 0.0)
    velocity: Vec3 = Vec3(0.0, 0.0, 0.0)
    yaw_degrees: float = 0.0
    radius: float = 0.26
    height: float = 1.45
    standing_height: float = 1.45
    crouch_height: float = 0.96
    speed: float = 2.0
    sprint_multiplier: float = 1.45
    crouch_speed_multiplier: float = 0.62
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
        target_height = self.crouch_height if "crouch" in keys else self.standing_height
        height_blend = min(1.0, dt * 10.0)
        self.height = self.height + (target_height - self.height) * height_blend
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
        speed = self.speed
        if "crouch" in keys:
            speed *= self.crouch_speed_multiplier
        elif "sprint" in keys and self.on_ground:
            speed *= self.sprint_multiplier
        horizontal = move * speed
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
        self.window = PixelWindow(args.window_width, args.window_height, title="py_3d capsule walk", fit_window=args.fit_window)

    def run(self) -> None:
        from time import sleep

        frame_time = 1.0 / max(1, self.args.fps)
        while not self.window.closed:
            for event in self.window.poll_events():
                if event.kind == "quit":
                    self.window.close()
                elif event.kind == "key_down":
                    if event.key == "escape":
                        self.window.close()
                    else:
                        self.on_key_down(event.key)
                elif event.kind == "key_up":
                    self.on_key_up(event.key)
            self.tick()
            sleep(frame_time)

    def tick(self) -> None:
        self.controller.step(1.0 / self.args.fps, self.keys, self.blocks)
        self.render_once()

    def on_key_down(self, key: str) -> None:
        if key == "v":
            self.camera_mode = {"global": "third", "third": "first", "first": "global"}[self.camera_mode]
        elif key in {"shift", "lshift", "rshift"}:
            self.keys.add("sprint")
        elif key in {"ctrl", "lctrl", "rctrl", "c"}:
            self.keys.add("crouch")
        elif key == "left":
            self.controller.yaw_degrees -= 5.0
        elif key == "right":
            self.controller.yaw_degrees += 5.0
        else:
            self.keys.add(key)

    def on_key_up(self, key: str) -> None:
        if key in {"shift", "lshift", "rshift"}:
            self.keys.discard("sprint")
        elif key in {"ctrl", "lctrl", "rctrl", "c"}:
            self.keys.discard("crouch")
        else:
            self.keys.discard(key)

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
        scene.add(*PlayerModel(self.controller.feet, self.controller.yaw_degrees, self.controller.height, self.controller.radius).to_primitives())
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
        self.window.show(buffer)


class GLCapsuleWalkViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        from py_3d.live import LiveFlyCamera, LiveMenu, LiveMenuOption, ModernGLLiveRenderer

        self.args = args
        self.controller = CapsuleController()
        self.keys: set[str] = set()
        self.camera_mode = args.camera_mode
        self.look_smoothing = args.look_smoothing
        self.look_pitch_degrees = -6.0
        self.render_camera: Camera | None = None
        self.sky = SkyPrefab(time_of_day=13.5, cycle_enabled=False, stars_enabled=True, clouds_enabled=True)
        global_camera = Camera(position=(3.2, 2.5, -5.2), target=(0.4, 0.65, 0.4), fov_degrees=54)
        self.free_camera = LiveFlyCamera.looking_at(
            global_camera.position,
            global_camera.target,
            fov_degrees=global_camera.fov_degrees,
            speed=2.8,
        )
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
        self.renderer = ModernGLLiveRenderer(
            args.window_width,
            args.window_height,
            title="py_3d capsule walk - live",
            vsync=True,
        )
        self.renderer.menu = LiveMenu(
            "py_3d capsule walk",
            (
                LiveMenuOption("done", "Done"),
                LiveMenuOption("next_camera", "Next camera"),
                LiveMenuOption("look_smoothing_up", "More look smoothing"),
                LiveMenuOption("look_smoothing_down", "Less look smoothing"),
                LiveMenuOption("sky_cycle", "Day/night cycle"),
                LiveMenuOption("sky_time_up", "Time later"),
                LiveMenuOption("sky_time_down", "Time earlier"),
                LiveMenuOption("sky_clouds", "Clouds"),
                LiveMenuOption("sky_stars", "Stars"),
                LiveMenuOption("reset", "Reset capsule"),
                LiveMenuOption("quit", "Quit demo"),
            ),
            background_blur=getattr(args, "menu_blur", False),
        )
        self._refresh_menu_options()
        self.renderer.set_mouse_captured(True)
        self._last_title_update = 0

    def run(self) -> None:
        clock = self.renderer.frame_clock()
        dt = 1.0 / self.args.fps
        running = True
        try:
            while running:
                for event in self.renderer.events():
                    if self.renderer.is_quit_event(event):
                        running = False
                    elif self.renderer.handle_resize_event(event):
                        continue
                    elif self.renderer.menu.visible and self.renderer.is_menu_pointer_event(event):
                        menu_action = self.renderer.handle_menu_mouse_event(event)
                        if menu_action is not None:
                            running = self._handle_menu_action(menu_action)
                    elif self.renderer.is_mouse_button_down_event(event) and not self.renderer.menu.visible:
                        self.renderer.set_mouse_captured(True)
                    elif self.renderer.is_mouse_motion_event(event) and self.renderer.mouse_captured:
                        self.on_mouse_motion(self.renderer.event_mouse_rel(event))
                    elif self.renderer.is_key_down_event(event):
                        running = self.on_key_down(self.renderer.event_key(event))
                    elif self.renderer.is_key_up_event(event):
                        self.on_key_up(self.renderer.event_key(event))

                if self.camera_mode == "global":
                    self.free_camera.move(self.keys, dt)
                else:
                    self.controller.step(dt, self.keys, self.blocks)
                self.sky.step(dt)
                camera = self._smoothed_camera(self.camera(), dt)
                self._update_hud()
                scene = self.scene()
                self._apply_sky(scene)
                stats = self.renderer.render(scene, camera, self.sky.settings_for(self.settings))
                self._update_title(stats)
                dt = max(1.0 / 240.0, min(0.05, clock.tick(self.args.fps) / 1000.0))
        finally:
            self.renderer.close()

    def on_key_down(self, key: int) -> bool:
        menu_action = self.renderer.handle_menu_key(key)
        if menu_action is not None:
            return self._handle_menu_action(menu_action)
        if self.renderer.key_matches(key, "v"):
            self.camera_mode = {"global": "third", "third": "first", "first": "global"}[self.camera_mode]
            self.render_camera = None
        elif self.renderer.key_matches(key, "left"):
            self.controller.yaw_degrees -= 5.0
        elif self.renderer.key_matches(key, "right"):
            self.controller.yaw_degrees += 5.0
        else:
            movement_key = self._movement_key(key)
            if movement_key is not None:
                self.keys.add(movement_key)
        return True

    def _handle_menu_action(self, action: str) -> bool:
        if action in {"handled", "navigate"}:
            return True
        if action == "opened":
            self.keys.clear()
            self.renderer.set_mouse_captured(False)
            self._refresh_menu_options()
            return True
        if action == "quit":
            return False
        if action in {"done", "resume", "cancel"}:
            self.renderer.menu.close()
            self.renderer.set_mouse_captured(True)
            return True
        if action == "next_camera":
            self.camera_mode = {"global": "third", "third": "first", "first": "global"}[self.camera_mode]
            self.render_camera = None
        elif action == "look_smoothing_up":
            self._adjust_look_smoothing(2.0)
        elif action == "look_smoothing_down":
            self._adjust_look_smoothing(-2.0)
        elif action == "sky_cycle":
            self.sky.toggle_cycle()
        elif action == "sky_time_up":
            self.sky.adjust_time(1.0)
        elif action == "sky_time_down":
            self.sky.adjust_time(-1.0)
        elif action == "sky_clouds":
            self.sky.toggle_clouds()
        elif action == "sky_stars":
            self.sky.toggle_stars()
        elif action == "reset":
            self.controller = CapsuleController()
            self.look_pitch_degrees = -6.0
            self.render_camera = None
        self._refresh_menu_options()
        return True

    def _refresh_menu_options(self) -> None:
        from py_3d.live import LiveMenuOption

        menu = self.renderer.menu
        previous_action = menu.selected_action() if menu.options else "done"
        options = (
            LiveMenuOption("done", "Done"),
            LiveMenuOption("next_camera", "Camera", self.camera_mode, "Camera"),
            LiveMenuOption("look_smoothing_up", "Look Smoothing +", f"{self.look_smoothing:0.1f}", "Camera"),
            LiveMenuOption("look_smoothing_down", "Look Smoothing -", f"{self.look_smoothing:0.1f}", "Camera"),
            LiveMenuOption("sky_cycle", "Cycle", "on" if self.sky.cycle_enabled else "off", "Sky"),
            LiveMenuOption("sky_time_up", "Later", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_time_down", "Earlier", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_clouds", "Clouds", "on" if self.sky.clouds_enabled else "off", "Sky"),
            LiveMenuOption("sky_stars", "Stars", "on" if self.sky.stars_enabled else "off", "Sky"),
            LiveMenuOption("reset", "Reset", "capsule", "Physics"),
            LiveMenuOption("quit", "Quit demo"),
        )
        menu.options = options
        actions = [option.action for option in options]
        menu.selected_index = actions.index(previous_action) if previous_action in actions else min(menu.selected_index, len(options) - 1)

    def on_key_up(self, key: int) -> None:
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.discard(movement_key)

    def on_mouse_motion(self, relative: tuple[int, int]) -> None:
        dx, dy = relative
        if self.camera_mode == "global":
            self.free_camera.look(dx, dy)
            return
        self.controller.yaw_degrees += dx * 0.12
        self.look_pitch_degrees = max(-82.0, min(82.0, self.look_pitch_degrees - dy * 0.12))

    def camera(self) -> Camera:
        if self.camera_mode == "first":
            yaw = radians(self.controller.yaw_degrees)
            pitch = radians(self.look_pitch_degrees)
            forward = Vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)).normalized(Vec3(0.0, 0.0, 1.0))
            eye = self.controller.feet + Vec3(0.0, self.controller.height * 0.86, 0.0)
            return Camera(position=eye, target=eye + forward, fov_degrees=68.0, near=0.03)
        if self.camera_mode == "third":
            yaw = radians(self.controller.yaw_degrees)
            pitch = radians(self.look_pitch_degrees)
            aim = Vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)).normalized(Vec3(0.0, 0.0, 1.0))
            right = Vec3(cos(yaw), 0.0, -sin(yaw)).normalized(Vec3(1.0, 0.0, 0.0))
            shoulder = self.controller.feet + Vec3(0.0, self.controller.height * 0.82, 0.0)
            camera_position = shoulder - aim * 3.25 + right * 0.54 + Vec3(0.0, 0.16, 0.0)
            target = shoulder + aim * 18.0
            return Camera(position=camera_position, target=target, fov_degrees=64.0, near=0.04)
        return self.free_camera.camera()

    def scene(self) -> Scene:
        scene = Scene()
        scene.add(Box((0, -0.06, 0), (7.0, 0.12, 7.0), Material(color=(54, 84, 88), roughness=0.55)))
        scene.add(*self.blocks)
        if self.camera_mode != "first":
            scene.add(*PlayerModel(self.controller.feet, self.controller.yaw_degrees, self.controller.height, self.controller.radius).to_primitives())
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

    def _apply_sky(self, scene: Scene) -> Scene:
        scene.lights = [light for light in scene.lights if not isinstance(light, Sun)]
        self.sky.apply(scene)
        return scene

    def _movement_key(self, key: int) -> str | None:
        if self.renderer.key_matches(key, "w"):
            return "w"
        if self.renderer.key_matches(key, "a"):
            return "a"
        if self.renderer.key_matches(key, "s"):
            return "s"
        if self.renderer.key_matches(key, "d"):
            return "d"
        if self.renderer.key_matches(key, "space"):
            return "space"
        if self.renderer.key_matches(key, "lshift", "rshift"):
            return "shift" if self.camera_mode == "global" else "sprint"
        if self.renderer.key_matches(key, "lctrl", "rctrl"):
            return "crouch"
        if self.renderer.key_matches(key, "c"):
            return "crouch"
        return None

    def _adjust_look_smoothing(self, amount: float) -> None:
        self.look_smoothing = max(2.0, min(40.0, self.look_smoothing + amount))

    def _smoothed_camera(self, camera: Camera, dt: float) -> Camera:
        if self.render_camera is None or self.camera_mode == "global":
            self.render_camera = camera
            return camera
        alpha = min(1.0, dt * self.look_smoothing)
        position = self.render_camera.position + (camera.position - self.render_camera.position) * alpha
        target = self.render_camera.target + (camera.target - self.render_camera.target) * alpha
        self.render_camera = Camera(position=position, target=target, fov_degrees=camera.fov_degrees, near=camera.near, far=camera.far)
        return self.render_camera

    def _update_hud(self) -> None:
        state = "CROUCH" if "crouch" in self.keys else ("SPRINT" if "sprint" in self.keys else "WALK")
        speed = (self.controller.velocity.x * self.controller.velocity.x + self.controller.velocity.z * self.controller.velocity.z) ** 0.5
        health = 84 if self.camera_mode == "first" else 96
        health_width = int(128 * health / 100)
        if self.camera_mode == "first":
            self.renderer.hud.set(
                HUDRect((12, 12), (232, 92), (0, 0, 0), alpha=0.62),
                HUDText(f"FPS HUD  {state}\nSPD {speed:0.1f} M/S  HP {health}\nLEVEL UP READY", (22, 20), color=(238, 245, 255), alpha=0.95, scale=1),
                HUDRect((22, 78), (128, 10), (48, 48, 48), alpha=0.88),
                HUDRect((22, 78), (health_width, 10), (56, 210, 112), alpha=0.95),
            )
            return
        if self.camera_mode == "third":
            self.renderer.hud.set(
                HUDRect((12, 12), (252, 92), (0, 0, 0), alpha=0.58),
                HUDText(f"THIRD PERSON HUD\nSPD {speed:0.1f} M/S  ARMOR 42\nLEVEL 2 + CAMERA ORBIT", (22, 20), color=(238, 245, 255), alpha=0.94, scale=1),
                HUDRect((22, 78), (128, 10), (48, 48, 48), alpha=0.88),
                HUDRect((22, 78), (health_width, 10), (92, 158, 255), alpha=0.95),
            )
            return
        self.renderer.hud.set(
            HUDRect((12, 12), (220, 76), (0, 0, 0), alpha=0.56),
            HUDText(f"FREE CAMERA HUD\nNOCLIP SPEED {self.free_camera.speed:0.1f}\nSKY {self.sky.time_of_day:04.1f}H", (22, 20), color=(238, 245, 255), alpha=0.94, scale=1),
        )

    def _update_title(self, stats) -> None:
        ticks = self.renderer.ticks()
        if ticks - self._last_title_update < 400:
            return
        self._last_title_update = ticks
        self.renderer.set_title(
            f"py_3d capsule walk - live - {self.camera_mode} - {stats.approx_fps:0.1f} fps "
            f"({stats.build_seconds * 1000:0.1f} ms build, {stats.draw_seconds * 1000:0.1f} ms draw)"
        )


def make_engine(renderer: str) -> RenderEngine:
    if renderer == "py_gpu":
        from py_gpu.adapters.py3d import Py3DRasterRenderer

        return RenderEngine(Py3DRasterRenderer(reference_compatible=False, fast_materials=True))
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
    parser.add_argument("--camera-mode", choices=("global", "third", "first"), default="first")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--fit-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--menu-blur", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ambient", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.15)
    parser.add_argument("--look-smoothing", type=float, default=18.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.renderer == "py_gpu":
        try:
            GLCapsuleWalkViewer(args).run()
            return
        except Exception as exc:
            print(f"py_3d live renderer unavailable, falling back to native PixelWindow path: {exc}")
    CapsuleWalkViewer(args).run()


if __name__ == "__main__":
    main()
