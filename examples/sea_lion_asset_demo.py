"""Render a prepared sea lion mesh asset."""

from __future__ import annotations

import argparse
from pathlib import Path
from math import sin, tau

from py_3d import Camera, FloatingTextBulletin, HUDRect, HUDText, Lamp, Material, PixelBuffer, Plane, RenderEngine, RenderSettings, Scene, SkyPrefab, Sun, TextBulletin, canonical_player_movement_key, load_mesh_asset, update_canonical_live_menu
from py_3d.color import Color


DEFAULT_ASSET = Path("USER") / "assets" / "sea_lion" / "sea_lion.py3dmesh.json"
DEFAULT_OUTPUT = Path("USER") / "environments" / "sea_lion" / "renderings" / "sea_lion_asset.png"
SEA_LION_LIVE_ACTIONS = {
    "done",
    "sky_cycle",
    "sky_time_down",
    "sky_time_up",
    "sky_clouds",
    "sky_stars",
    "snapshot",
    "quit",
}


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
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=630)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--menu-blur", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ambient", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.12)
    return parser.parse_args()


class GLSeaLionAssetViewer:
    def __init__(self, args: argparse.Namespace, settings: RenderSettings) -> None:
        from py_3d.live import LiveFlyCamera, LiveMenu, ModernGLLiveRenderer

        self.args = args
        self.settings = settings
        self.sky = SkyPrefab(time_of_day=15.0, cycle_enabled=False, stars_enabled=True, clouds_enabled=True)
        base_scene = make_scene(args.asset)
        self.base_objects = tuple(base_scene.objects)
        self.base_lights = tuple(light for light in base_scene.lights if not isinstance(light, Sun))
        self.base_bulletins = tuple(base_scene.bulletins)
        base_camera = make_camera()
        self.camera_controller = LiveFlyCamera.looking_at(base_camera.position, base_camera.target, fov_degrees=base_camera.fov_degrees, speed=2.2)
        self.keys: set[str] = set()
        self.renderer = ModernGLLiveRenderer(
            args.window_width,
            args.window_height,
            title="py_3d sea lion asset - live",
            vsync=getattr(args, "vsync", True),
        )
        self.renderer.menu = LiveMenu(
            "py_3d sea lion asset",
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
                        rel = self.renderer.event_mouse_rel(event)
                        self.camera_controller.look(rel[0], rel[1])
                    elif self.renderer.is_key_down_event(event):
                        running = self.on_key_down(self.renderer.event_key(event))
                    elif self.renderer.is_key_up_event(event):
                        self.on_key_up(self.renderer.event_key(event))

                self.camera_controller.move(self.keys, dt)
                self.sky.step(dt)
                self._update_hud()
                stats = self.renderer.render(self.scene(), self.camera_controller.smoothed_camera(dt), self.sky.settings_for(self.settings))
                self._update_title(stats)
                dt = max(1.0 / 240.0, min(0.05, clock.tick(self.args.fps) / 1000.0))
        finally:
            self.renderer.close()

    def scene(self) -> Scene:
        scene = Scene()
        scene.add(*self.base_objects)
        scene.add_light(*self.base_lights)
        scene.add_bulletin(*self.base_bulletins)
        self.sky.apply(scene)
        return scene

    def on_key_down(self, key: int) -> bool:
        menu_action = self.renderer.handle_menu_key(key)
        if menu_action is not None:
            return self._handle_menu_action(menu_action)
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.add(movement_key)
        elif self.renderer.key_matches(key, "p"):
            self.save_snapshot()
        return True

    def on_key_up(self, key: int) -> None:
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.discard(movement_key)

    def _movement_key(self, key: int) -> str | None:
        return canonical_player_movement_key(self.renderer, key, camera_mode="global")

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
        if action == "sky_cycle":
            self.sky.toggle_cycle()
        elif action == "sky_time_up":
            self.sky.adjust_time(1.0)
        elif action == "sky_time_down":
            self.sky.adjust_time(-1.0)
        elif action == "sky_clouds":
            self.sky.toggle_clouds()
        elif action == "sky_stars":
            self.sky.toggle_stars()
        elif action == "snapshot":
            self.save_snapshot()
        self._refresh_menu_options()
        return True

    def save_snapshot(self) -> None:
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        RenderEngine().render(self.scene(), self.camera_controller.camera(), self.sky.settings_for(self.settings)).to_png(self.args.output)
        print(f"Wrote {self.args.output}")

    def _refresh_menu_options(self) -> None:
        update_canonical_live_menu(
            self.renderer.menu,
            details={
                "sky_cycle": "on" if self.sky.cycle_enabled else "off",
                "sky_time_down": f"{self.sky.time_of_day:04.1f}h",
                "sky_time_up": f"{self.sky.time_of_day:04.1f}h",
                "sky_clouds": "on" if self.sky.clouds_enabled else "off",
                "sky_stars": "on" if self.sky.stars_enabled else "off",
                "snapshot": "PNG",
            },
            enabled_actions=SEA_LION_LIVE_ACTIONS,
        )

    def _update_hud(self) -> None:
        self.renderer.hud.set(
            HUDRect((12, 12), (190, 58), (3, 7, 10), alpha=0.55),
            HUDText(f"SEA LION ASSET\nSKY {self.sky.time_of_day:04.1f}H\nP SNAPSHOT", (20, 20), color=(238, 245, 255), alpha=0.94, scale=1),
        )

    def _update_title(self, stats) -> None:
        ticks = self.renderer.ticks()
        if ticks - self._last_title_update < 400:
            return
        self._last_title_update = ticks
        self.renderer.set_title(
            f"py_3d sea lion asset - live - {stats.approx_fps:0.1f} fps "
            f"({stats.build_seconds * 1000:0.1f} ms build, {stats.draw_seconds * 1000:0.1f} ms draw)"
        )


def main() -> None:
    args = parse_args()
    if not args.asset.exists():
        raise SystemExit(f"Prepared asset missing: {args.asset}. Run examples/ingest_asset.py first.")
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
    if args.live:
        GLSeaLionAssetViewer(args, settings).run()
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    RenderEngine().render(make_scene(args.asset), make_camera(), settings).to_png(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
