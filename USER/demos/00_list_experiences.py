"""Interactive game-style menu for curated py_3d demo experiences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
from time import sleep
from typing import Sequence

from py_3d import PixelBuffer, PixelWindow, draw


ROOT = Path(__file__).resolve().parents[2]
FRAME_WIDTH = 960
FRAME_HEIGHT = 540


@dataclass(frozen=True)
class Experience:
    title: str
    script: str
    description: str
    launch_command: tuple[str, ...]
    preview_command: tuple[str, ...] | None
    preview_path: Path | None
    controls: str


@dataclass(frozen=True)
class MenuSettings:
    quality: str = "balanced"
    safe_mode: bool = False
    background_blur: bool = False


EXPERIENCES = [
    Experience(
        "Fruit Bowl Physics",
        "10_live_fruit_bowl_gpu.py",
        "Textured fruit, ray-style reflections, grab physics, baked sign text, and live graphics settings.",
        ("python", "examples/fruit_bowl_live.py", "--renderer", "py_gpu", "--light-mode", "hanging-lamp"),
        ("python", "USER/tests/run_environment.py", "-e", "fruit_bowl", "--variant", "fast_gpu_still", "--no-specs"),
        Path("USER/environments/fruit_bowl/renderings/fruit_bowl_gpu_fast.png"),
        "WASD mouse look, E grab/drop, wheel depth, Esc menu.",
    ),
    Experience(
        "Fruit Bowl RGB Bulbs",
        "16_live_fruit_bowl_rgb_bulbs.py",
        "Standalone red, green, and blue bulbs blink through R, G, B, RG, RB, GB, RGB, and all-off states.",
        ("python", "examples/fruit_bowl_rgb_bulbs_live.py", "--renderer", "py_gpu"),
        None,
        Path("USER/environments/fruit_bowl/renderings/fruit_bowl_rgb_bulbs.png"),
        "2-second RGB bulb sequence, WASD mouse look, E grab/drop, Esc menu.",
    ),
    Experience(
        "Capsule Player Controller",
        "11_live_capsule_walk.py",
        "FPS, over-shoulder, and free cameras with crouch, sprint, collision, HUD overlays, and player model primitives.",
        ("python", "examples/capsule_walk_demo.py", "--renderer", "py_gpu"),
        None,
        Path("USER/environments/capsule_walk/renderings/capsule_walk_preview.png"),
        "WASD move, mouse aim, Shift sprint, Ctrl/C crouch, V camera.",
    ),
    Experience(
        "Fan Cloth Water",
        "15_render_fan_cloth_water.py",
        "A fan blowing spring cloth while wind shear and a small impeller stir a particle-fluid water vessel.",
        ("python", "examples/fan_cloth_water_demo.py", "--live"),
        (
            "python",
            "examples/fan_cloth_water_demo.py",
            "--quality",
            "fast",
            "--width",
            "480",
            "--height",
            "270",
            "--output",
            "USER/environments/fan_cloth_water/renderings/fan_cloth_water.png",
        ),
        Path("USER/environments/fan_cloth_water/renderings/fan_cloth_water.png"),
        "WASD mouse look, menu tunes wind, water swirl, sky, reflections.",
    ),
    Experience(
        "Wind Pool Water",
        "18_live_wind_pool_water.py",
        "A bath of liquid particles forms a reflective pool while wind rushes over the surface.",
        ("python", "examples/wind_pool_water_demo.py", "--live"),
        (
            "python",
            "examples/wind_pool_water_demo.py",
            "--quality",
            "fast",
            "--width",
            "480",
            "--height",
            "270",
            "--output",
            "USER/environments/wind_pool_water/renderings/wind_pool_water.png",
        ),
        Path("USER/environments/wind_pool_water/renderings/wind_pool_water.png"),
        "WASD mouse look, menu tunes wind, quality, and reflections.",
    ),
    Experience(
        "Sea Lion Asset Viewer",
        "14_render_sea_lion_asset.py",
        "Prepared mesh asset with preserved UVs, procedural skin texture, live fly camera, and sky controls.",
        ("python", "examples/sea_lion_asset_demo.py", "--live", "--asset", "USER/assets/sea_lion/sea_lion.py3dmesh.json"),
        (
            "python",
            "examples/sea_lion_asset_demo.py",
            "--asset",
            "USER/assets/sea_lion/sea_lion.py3dmesh.json",
            "--output",
            "USER/environments/sea_lion/renderings/sea_lion_asset.png",
        ),
        Path("USER/environments/sea_lion/renderings/sea_lion_asset.png"),
        "WASD mouse look, P snapshot, Esc sky/settings menu.",
    ),
]


QUALITY_ARGS = {
    "safe": ("--quality", "fast", "--fps", "30", "--window-width", "960", "--window-height", "540"),
    "balanced": ("--quality", "balanced"),
    "high": ("--quality", "high"),
    "ultra": ("--quality", "ultra"),
}


def command_for(experience: Experience, settings: MenuSettings) -> tuple[str, ...]:
    command = list(experience.launch_command)
    quality = "safe" if settings.safe_mode else settings.quality
    if "examples/fruit_bowl_live.py" in command or "examples/fan_cloth_water_demo.py" in command or "examples/wind_pool_water_demo.py" in command:
        if "--quality" not in command:
            command.extend(QUALITY_ARGS.get(quality, QUALITY_ARGS["balanced"]))
    if settings.background_blur and "--menu-blur" not in command:
        command.append("--menu-blur")
    return tuple(command)


def print_menu() -> None:
    print("py_3d render engine showcase")
    print("")
    for index, experience in enumerate(EXPERIENCES, start=1):
        print(f"{index}. {experience.title} ({experience.script})")
        print(f"   {experience.description}")
    print("")
    print("Run without flags for the game-style menu.")


def run_experience(index: int, *, dry_run: bool = False, settings: MenuSettings | None = None) -> None:
    if index < 1 or index > len(EXPERIENCES):
        raise SystemExit(f"Choose a number from 1 to {len(EXPERIENCES)}.")
    command = command_for(EXPERIENCES[index - 1], settings or MenuSettings())
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT)


def render_preview(index: int, *, dry_run: bool = False) -> None:
    if index < 1 or index > len(EXPERIENCES):
        raise SystemExit(f"Choose a number from 1 to {len(EXPERIENCES)}.")
    command = EXPERIENCES[index - 1].preview_command
    if command is None:
        raise SystemExit("This experience does not have a still-preview render command yet.")
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT)


def _fit_text(value: str, max_width: int, *, scale: int = 1) -> str:
    return draw.fit_text(value, max_width, scale=scale)


def _draw_wrapped_text(
    buffer: PixelBuffer,
    value: str,
    left: int,
    top: int,
    width: int,
    color: tuple[int, int, int],
    *,
    max_lines: int = 4,
    scale: int = 1,
) -> None:
    lines = draw.wrap_text(value, width, scale=scale, max_lines=max_lines)
    line_height = (7 + 2) * scale
    for index, line in enumerate(lines[:max_lines]):
        draw.text(buffer, (left, top + index * line_height), line, color, scale=scale)


def _blit(target: PixelBuffer, source: PixelBuffer, left: int, top: int) -> None:
    target.blit(source, left, top)


class NativeShowcaseMenu:
    quality_order = ("safe", "balanced", "high", "ultra")

    def __init__(self) -> None:
        self.window = PixelWindow(FRAME_WIDTH, FRAME_HEIGHT, title="py_3d", fit_window=True)
        self.selected = 0
        self.hover_action = ""
        self.settings = MenuSettings()
        self.last_safe_settings = self.settings
        self.status = "Select an experience, preview it, or launch it."
        self.child: subprocess.Popen | None = None
        self.child_command: tuple[str, ...] | None = None
        self.child_preview = False
        self.hitboxes: list[tuple[int, int, int, int, str]] = []
        self.preview_cache: dict[Path, PixelBuffer | None] = {}
        self.preview_scaled_cache: dict[tuple[Path, int, int], PixelBuffer | None] = {}
        self.preview_images_enabled = False
        self._dirty = True
        self._frame: PixelBuffer | None = None

    def run(self) -> None:
        self._redraw()
        while not self.window.closed:
            for event in self.window.poll_events():
                if event.kind == "quit":
                    self.window.close()
                elif event.kind == "key_down":
                    self._handle_key(event.key)
                elif event.kind == "motion":
                    action = self._action_at(self._frame_position(event.pos))
                    if action != self.hover_action:
                        self.hover_action = action
                        self._mark_dirty()
                elif event.kind == "button" and event.button == 1:
                    self._handle_action(self._action_at(self._frame_position(event.pos)))
            if self._poll_child():
                self._mark_dirty()
            if self._dirty:
                self._redraw()
            sleep(0.005)

    def _redraw(self) -> None:
        self._frame = self._render()
        self._dirty = False
        self.window.show(self._frame)

    def _render(self) -> PixelBuffer:
        buffer = PixelBuffer.from_rgb_bytes(FRAME_WIDTH, FRAME_HEIGHT, bytearray((9, 9, 9)) * (FRAME_WIDTH * FRAME_HEIGHT))
        if self.settings.background_blur:
            self._draw_soft_background(buffer)
        self.hitboxes = []
        panel_left, panel_top, panel_width, panel_height = 50, 30, 860, 480
        draw.rect(buffer, (panel_left, panel_top), (panel_width, panel_height), (0, 0, 0), fill=True)
        draw.rect(buffer, (panel_left, panel_top), (panel_width, panel_height), (86, 86, 86), fill=False)
        draw.text(buffer, (panel_left + 26, panel_top + 22), "PY_3D", (242, 242, 242), scale=3)
        draw.text(buffer, (panel_left + 26, panel_top + 58), f"PROFILE {self._profile_label(self.settings).upper()}", (158, 158, 158), scale=1)
        draw.text(buffer, (panel_left + 388, panel_top + 58), _fit_text(self.status, 480), (190, 190, 190), scale=1)

        list_left, list_top = panel_left + 26, panel_top + 98
        for index, experience in enumerate(EXPERIENCES):
            top = list_top + index * 44
            selected = index == self.selected
            color = (36, 36, 36) if selected else (14, 14, 14)
            if self.hover_action == f"select:{index}" and not selected:
                color = (24, 24, 24)
            draw.rect(buffer, (list_left, top), (318, 36), color, fill=True)
            draw.rect(buffer, (list_left, top), (318, 36), (112, 112, 112) if selected else (44, 44, 44), fill=False)
            draw.text(buffer, (list_left + 10, top + 10), _fit_text(experience.title, 296, scale=2), (238, 238, 238), scale=2)
            self._hitbox(list_left, top, 318, 36, f"select:{index}")

        detail_left, detail_top = panel_left + 380, panel_top + 98
        experience = EXPERIENCES[self.selected]
        draw.rect(buffer, (detail_left, detail_top), (448, 144), (15, 15, 15), fill=True)
        draw.rect(buffer, (detail_left, detail_top), (448, 144), (54, 54, 54), fill=False)
        self._draw_preview(buffer, detail_left + 8, detail_top + 8, 432, 128)
        draw.text(buffer, (detail_left, detail_top + 166), _fit_text(experience.title, 448, scale=2), (244, 244, 244), scale=2)
        _draw_wrapped_text(buffer, experience.description, detail_left, detail_top + 198, 448, (206, 206, 206), max_lines=3, scale=2)
        _draw_wrapped_text(buffer, experience.controls, detail_left, detail_top + 258, 448, (156, 156, 156), max_lines=2, scale=1)

        controls_top = panel_top + panel_height - 104
        self._button(buffer, panel_left + 26, controls_top, 126, 36, "LAUNCH", "launch", scale=2)
        self._button(buffer, panel_left + 166, controls_top, 148, 36, "PREVIEW", "preview", scale=2)
        self._button(buffer, panel_left + 328, controls_top, 104, 36, "STOP", "stop", enabled=self.child is not None, scale=2)
        self._button(buffer, panel_left + panel_width - 126, controls_top, 96, 36, "EXIT", "exit", scale=2)

        quality = "safe" if self.settings.safe_mode else self.settings.quality
        settings_top = controls_top + 56
        draw.text(buffer, (panel_left + 26, settings_top + 12), "QUALITY", (190, 190, 190), scale=1)
        self._button(buffer, panel_left + 128, settings_top, 34, 32, "-", "quality_down", scale=2)
        draw.rect(buffer, (panel_left + 172, settings_top), (118, 32), (18, 18, 18), fill=True)
        draw.rect(buffer, (panel_left + 172, settings_top), (118, 32), (78, 78, 78), fill=False)
        draw.text(buffer, (panel_left + 190, settings_top + 9), _fit_text(quality.upper(), 82, scale=2), (235, 235, 235), scale=2)
        self._button(buffer, panel_left + 300, settings_top, 34, 32, "+", "quality_up", scale=2)
        self._button(buffer, panel_left + 362, settings_top, 142, 32, f"BLUR {'ON' if self.settings.background_blur else 'OFF'}", "blur", scale=1)
        return buffer

    def _draw_preview(self, buffer: PixelBuffer, left: int, top: int, width: int, height: int) -> None:
        draw.rect(buffer, (left, top), (width, height), (20, 20, 20), fill=True)
        path = EXPERIENCES[self.selected].preview_path
        if not self.preview_images_enabled:
            label = "PREVIEW" if path is not None else "NO PREVIEW"
            text_width, text_height = draw.text_size(label, scale=2)
            draw.text(buffer, (left + (width - text_width) // 2, top + (height - text_height) // 2), label, (150, 150, 150), scale=2)
            return
        preview = self._preview_image()
        if preview is None:
            label = "NO PREVIEW"
            text_width, text_height = draw.text_size(label, scale=2)
            draw.text(buffer, (left + (width - text_width) // 2, top + (height - text_height) // 2), label, (150, 150, 150), scale=2)
            return
        resized = self._scaled_preview(preview, width, height)
        _blit(buffer, resized, left + (width - resized.width) // 2, top + (height - resized.height) // 2)

    def _preview_image(self) -> PixelBuffer | None:
        path = EXPERIENCES[self.selected].preview_path
        if path is None:
            return None
        target = ROOT / path
        if target in self.preview_cache:
            return self.preview_cache[target]
        try:
            image = PixelBuffer.from_png(target) if target.exists() else None
        except Exception:
            image = None
        self.preview_cache[target] = image
        return image

    def _scaled_preview(self, preview: PixelBuffer, width: int, height: int) -> PixelBuffer:
        path = EXPERIENCES[self.selected].preview_path or Path("")
        key = (ROOT / path, width, height)
        cached = self.preview_scaled_cache.get(key)
        if cached is not None:
            return cached
        scale = min(width / preview.width, height / preview.height)
        resized = preview.resized_nearest(max(1, int(preview.width * scale)), max(1, int(preview.height * scale)))
        self.preview_scaled_cache[key] = resized
        return resized

    def _button(
        self,
        buffer: PixelBuffer,
        left: int,
        top: int,
        width: int,
        height: int,
        label: str,
        action: str,
        *,
        enabled: bool = True,
        scale: int = 1,
    ) -> None:
        hovered = self.hover_action == action and enabled
        fill = (42, 42, 42) if hovered else (24, 24, 24)
        border = (150, 150, 150) if hovered else (84, 84, 84)
        text = (238, 238, 238) if enabled else (100, 100, 100)
        draw.rect(buffer, (left, top), (width, height), fill, fill=True)
        draw.rect(buffer, (left, top), (width, height), border, fill=False)
        label = _fit_text(label, max(1, width - 10), scale=scale)
        text_width, text_height = draw.text_size(label, scale=scale)
        draw.text(buffer, (left + (width - text_width) // 2, top + (height - text_height) // 2), label, text, scale=scale)
        if enabled:
            self._hitbox(left, top, width, height, action)

    def _hitbox(self, left: int, top: int, width: int, height: int, action: str) -> None:
        self.hitboxes.append((left, top, width, height, action))

    def _action_at(self, position: tuple[int, int]) -> str:
        x, y = position
        for left, top, width, height, action in self.hitboxes:
            if left <= x <= left + width and top <= y <= top + height:
                return action
        return ""

    def _frame_position(self, position: tuple[int, int]) -> tuple[int, int]:
        window_width, window_height = self.window.size
        return (
            max(0, min(FRAME_WIDTH - 1, int(position[0] * FRAME_WIDTH / max(1, window_width)))),
            max(0, min(FRAME_HEIGHT - 1, int(position[1] * FRAME_HEIGHT / max(1, window_height)))),
        )

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _handle_key(self, key: str) -> None:
        if key == "escape":
            self.window.close()
        elif key in {"up", "w"}:
            self._select((self.selected - 1) % len(EXPERIENCES))
        elif key in {"down", "s"}:
            self._select((self.selected + 1) % len(EXPERIENCES))
        elif key == "return":
            self._launch_selected()
        elif key == "p":
            self._render_selected_preview()
        elif key == "q":
            self._cycle_quality(1)
        elif key == "b":
            self._toggle_blur()
        elif key == "x":
            self._stop_child()

    def _handle_action(self, action: str) -> None:
        if not action:
            return
        if action.startswith("select:"):
            self._select(int(action.split(":", 1)[1]))
        elif action == "launch":
            self._launch_selected()
        elif action == "preview":
            self._render_selected_preview()
        elif action == "stop":
            self._stop_child()
        elif action == "exit":
            self.window.close()
        elif action == "quality_down":
            self._cycle_quality(-1)
        elif action == "quality_up":
            self._cycle_quality(1)
        elif action == "blur":
            self._toggle_blur()

    def _launch_selected(self) -> None:
        if self.child is not None:
            self.status = "A demo is already running."
            self._mark_dirty()
            return
        experience = EXPERIENCES[self.selected]
        self._start_child(command_for(experience, self.settings), f"Launched {experience.title}.")

    def _render_selected_preview(self) -> None:
        if self.child is not None:
            self.status = "A process is already running."
            self._mark_dirty()
            return
        command = EXPERIENCES[self.selected].preview_command
        if command is None:
            path = EXPERIENCES[self.selected].preview_path
            target = ROOT / path if path is not None else None
            if target is not None and target.exists():
                self.preview_images_enabled = True
                self.status = f"Loaded preview for {EXPERIENCES[self.selected].title}."
            else:
                self.status = "No still-preview command is available."
            self._mark_dirty()
            return
        self.preview_images_enabled = False
        self._start_child(command, f"Rendering preview for {EXPERIENCES[self.selected].title}.", preview=True)

    def _start_child(self, command: Sequence[str], status: str, *, preview: bool = False) -> None:
        self.child_command = tuple(command)
        self.child_preview = preview
        try:
            self.child = subprocess.Popen(list(command), cwd=ROOT)
        except OSError as exc:
            self.child = None
            self.child_command = None
            self.child_preview = False
            self.status = f"Could not start process: {exc}"
            self._mark_dirty()
            return
        self.status = status
        self._mark_dirty()
        if not preview:
            self.window.close()

    def _stop_child(self) -> None:
        if self.child is not None:
            self.child.terminate()
            self.status = "Stopping current process."
            self._mark_dirty()

    def _poll_child(self) -> bool:
        if self.child is None:
            return False
        code = self.child.poll()
        if code is None:
            return False
        command = " ".join(self.child_command or ())
        if code == 0:
            self.status = f"Finished: {command}"
            self.last_safe_settings = self.settings
            if self.child_preview:
                self.preview_cache.clear()
                self.preview_scaled_cache.clear()
                self.preview_images_enabled = True
        else:
            self.settings = self.last_safe_settings
            self.preview_images_enabled = False
            self.status = f"Process exited {code}; reverted menu settings."
        self.child = None
        self.child_command = None
        self.child_preview = False
        return True

    def _select(self, index: int) -> None:
        if index == self.selected:
            return
        self.selected = index
        self.preview_images_enabled = False
        self._mark_dirty()

    def _cycle_quality(self, amount: int) -> None:
        current = "safe" if self.settings.safe_mode else self.settings.quality
        index = self.quality_order.index(current) if current in self.quality_order else 2
        value = self.quality_order[(index + amount) % len(self.quality_order)]
        self.settings = MenuSettings(quality="balanced" if value == "safe" else value, safe_mode=value == "safe", background_blur=self.settings.background_blur)
        self.status = f"Profile {self._profile_label(self.settings)}."
        self._mark_dirty()

    def _toggle_blur(self) -> None:
        self.settings = MenuSettings(quality=self.settings.quality, safe_mode=self.settings.safe_mode, background_blur=not self.settings.background_blur)
        self.status = f"Menu blur {'enabled' if self.settings.background_blur else 'disabled'}."
        self._mark_dirty()

    def _draw_soft_background(self, buffer: PixelBuffer) -> None:
        for y in range(0, buffer.height, 18):
            shade = 10 + (y // 18) % 3 * 4
            draw.rect(buffer, (0, y), (buffer.width, 18), (shade, shade, shade), fill=True)

    @staticmethod
    def _profile_label(settings: MenuSettings) -> str:
        return "safe" if settings.safe_mode else settings.quality


def run_showcase_menu() -> None:
    NativeShowcaseMenu().run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the py_3d render engine showcase menu.")
    parser.add_argument("--list", action="store_true", help="Print available experiences and exit.")
    parser.add_argument("--run", type=int, help="Run a menu item by number without opening the interactive menu.")
    parser.add_argument("--render-preview", type=int, help="Render a preview still for a menu item by number.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quality", choices=("safe", "balanced", "high", "ultra"), default="balanced")
    args = parser.parse_args()

    settings = MenuSettings(quality="balanced" if args.quality == "safe" else args.quality, safe_mode=args.quality == "safe")
    if args.list:
        print_menu()
        return
    if args.run is not None:
        run_experience(args.run, dry_run=args.dry_run, settings=settings)
        return
    if args.render_preview is not None:
        render_preview(args.render_preview, dry_run=args.dry_run)
        return
    if args.dry_run:
        print_menu()
        return
    run_showcase_menu()


if __name__ == "__main__":
    main()
