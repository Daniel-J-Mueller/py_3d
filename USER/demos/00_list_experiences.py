"""Interactive game-style menu for curated py_3d demo experiences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]


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
    safe_mode: bool = True


@dataclass(frozen=True)
class Button:
    action: str
    rect: tuple[int, int, int, int]
    label: str
    enabled: bool = True


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
        "Capsule Player Controller",
        "11_live_capsule_walk.py",
        "FPS, over-shoulder, and free cameras with crouch, sprint, collision, HUD overlays, and player model primitives.",
        ("python", "examples/capsule_walk_demo.py", "--renderer", "py_gpu", "--camera-mode", "third"),
        None,
        Path("USER/environments/capsule_walk/renderings/capsule_walk_preview.png"),
        "WASD move, mouse aim, Shift sprint, Ctrl/C crouch, V camera.",
    ),
    Experience(
        "Fan Cloth Water",
        "15_render_fan_cloth_water.py",
        "A fan blowing a spring cloth and a vector-cloud liquid stirred by a spinning fan blade.",
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
    if "examples/fruit_bowl_live.py" in command or "examples/fan_cloth_water_demo.py" in command:
        if "--quality" not in command:
            command.extend(QUALITY_ARGS.get(quality, QUALITY_ARGS["balanced"]))
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


class ShowcaseMenu:
    def __init__(self) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1180, 720), pygame.RESIZABLE)
        pygame.display.set_caption("py_3d render engine showcase")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 21)
        self.small = pygame.font.SysFont("segoeui", 16)
        self.title_font = pygame.font.SysFont("segoeui", 34, bold=True)
        self.selected = 0
        self.settings = MenuSettings()
        self.pending_settings = self.settings
        self.last_safe_settings = self.settings
        self.show_settings = False
        self.status = "Select an experience and launch or render a preview."
        self.child: subprocess.Popen | None = None
        self.child_command: tuple[str, ...] | None = None
        self.buttons: list[Button] = []
        self.preview_cache: dict[Path, tuple[float, object]] = {}

    def run(self) -> None:
        running = True
        while running:
            self._poll_child()
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    running = False
                elif event.type == self.pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 1:
                    running = self._handle_click(event.pos)
                elif event.type == self.pygame.MOUSEWHEEL and not self.show_settings:
                    self.selected = (self.selected - event.y) % len(EXPERIENCES)
            self._draw()
            self.clock.tick(60)
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
        self.pygame.quit()

    def _poll_child(self) -> None:
        if self.child is None:
            return
        code = self.child.poll()
        if code is None:
            return
        command = " ".join(self.child_command or ())
        if code == 0:
            self.status = f"Finished: {command}"
            self.last_safe_settings = self.settings
        else:
            self.settings = self.last_safe_settings
            self.pending_settings = self.settings
            self.status = f"Process exited {code}; reverted menu settings."
        self.child = None
        self.child_command = None

    def _handle_key(self, key: int) -> bool:
        pygame = self.pygame
        if key == pygame.K_ESCAPE:
            if self.show_settings:
                self._discard_settings()
                self.show_settings = False
                return True
            return False
        if self.show_settings:
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._apply_settings(close=True)
            elif key == pygame.K_LEFT:
                self._cycle_quality(-1)
            elif key == pygame.K_RIGHT:
                self._cycle_quality(1)
            elif key == pygame.K_s:
                self.pending_settings = replace(self.pending_settings, safe_mode=not self.pending_settings.safe_mode)
            return True
        if key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % len(EXPERIENCES)
        elif key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % len(EXPERIENCES)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._launch_selected()
        elif key == pygame.K_p:
            self._render_selected_preview()
        elif key == pygame.K_TAB:
            self.show_settings = True
            self.pending_settings = self.settings
        return True

    def _handle_click(self, pos: tuple[int, int]) -> bool:
        for button in self.buttons:
            x, y, width, height = button.rect
            if button.enabled and x <= pos[0] <= x + width and y <= pos[1] <= y + height:
                return self._dispatch(button.action)
        if not self.show_settings:
            for index, rect in self._experience_rects():
                x, y, width, height = rect
                if x <= pos[0] <= x + width and y <= pos[1] <= y + height:
                    self.selected = index
                    return True
        return True

    def _dispatch(self, action: str) -> bool:
        if action == "launch":
            self._launch_selected()
        elif action == "preview":
            self._render_selected_preview()
        elif action == "settings":
            self.show_settings = True
            self.pending_settings = self.settings
        elif action == "stop":
            self._stop_child()
        elif action == "exit":
            return False
        elif action == "quality_prev":
            self._cycle_quality(-1)
        elif action == "quality_next":
            self._cycle_quality(1)
        elif action == "safe":
            self.pending_settings = replace(self.pending_settings, safe_mode=not self.pending_settings.safe_mode)
        elif action == "apply":
            self._apply_settings(close=False)
        elif action == "done":
            self._apply_settings(close=True)
        elif action == "cancel":
            self._discard_settings()
            self.show_settings = False
        return True

    def _launch_selected(self) -> None:
        if self.child is not None:
            self.status = "A demo is already running. Stop it or close it before launching another."
            return
        experience = EXPERIENCES[self.selected]
        command = command_for(experience, self.settings)
        self._start_child(command, f"Launched {experience.title}.")

    def _render_selected_preview(self) -> None:
        if self.child is not None:
            self.status = "A process is already running. Stop it or wait before rendering a preview."
            return
        command = EXPERIENCES[self.selected].preview_command
        if command is None:
            self.status = "This experience is live-only right now; no still preview command is registered."
            return
        self._start_child(command, f"Rendering preview for {EXPERIENCES[self.selected].title}.")

    def _start_child(self, command: Sequence[str], status: str) -> None:
        self.child_command = tuple(command)
        self.status = status
        try:
            self.child = subprocess.Popen(list(command), cwd=ROOT)
        except OSError as exc:
            self.child = None
            self.child_command = None
            self.status = f"Could not start process: {exc}"

    def _stop_child(self) -> None:
        if self.child is None:
            return
        self.child.terminate()
        self.status = "Stopping current process."

    def _cycle_quality(self, amount: int) -> None:
        qualities = ("safe", "balanced", "high", "ultra")
        current = "safe" if self.pending_settings.safe_mode else self.pending_settings.quality
        index = qualities.index(current) if current in qualities else 1
        next_quality = qualities[(index + amount) % len(qualities)]
        self.pending_settings = MenuSettings(quality="balanced" if next_quality == "safe" else next_quality, safe_mode=next_quality == "safe")

    def _apply_settings(self, *, close: bool) -> None:
        self.last_safe_settings = self.settings
        self.settings = self.pending_settings
        self.status = f"Applied menu profile: {self._profile_label(self.settings)}."
        if close:
            self.show_settings = False

    def _discard_settings(self) -> None:
        self.pending_settings = self.settings
        self.status = "Discarded pending menu settings."

    def _draw(self) -> None:
        width, height = self.screen.get_size()
        self.buttons = []
        self._draw_background(width, height)
        self._draw_header(width)
        self._draw_experience_list(width, height)
        self._draw_preview_panel(width, height)
        self._draw_button_bay(width, height)
        if self.show_settings:
            self._draw_settings(width, height)
        self.pygame.display.flip()

    def _draw_background(self, width: int, height: int) -> None:
        for y in range(height):
            t = y / max(1, height - 1)
            color = (6 + int(t * 12), 11 + int(t * 24), 19 + int(t * 31))
            self.pygame.draw.line(self.screen, color, (0, y), (width, y))
        for index in range(72):
            x = (index * 97 + 31) % max(1, width)
            y = (index * 53 + 17) % max(1, max(1, height - 96))
            shade = 90 + (index * 37) % 90
            self.screen.fill((shade, shade, min(255, shade + 25)), (x, y, 2, 2))

    def _draw_header(self, width: int) -> None:
        self._text("py_3d render engine showcase", (32, 24), self.title_font, (244, 248, 255))
        profile = self._profile_label(self.settings)
        self._text(f"Profile {profile}   {self.status}", (34, 68), self.small, (180, 198, 214), max_width=width - 68)

    def _draw_experience_list(self, width: int, height: int) -> None:
        left = 32
        top = 106
        card_width = min(470, max(340, int(width * 0.42)))
        card_height = 122
        for index, experience in enumerate(EXPERIENCES):
            y = top + index * (card_height + 14)
            selected = index == self.selected
            fill = (24, 39, 52) if selected else (14, 24, 34)
            border = (112, 178, 210) if selected else (45, 68, 82)
            rect = (left, y, card_width, card_height)
            self.pygame.draw.rect(self.screen, fill, rect, border_radius=6)
            self.pygame.draw.rect(self.screen, border, rect, width=2, border_radius=6)
            self._text(experience.title, (left + 16, y + 12), self.font, (242, 247, 255), max_width=card_width - 32)
            self._text(experience.description, (left + 16, y + 42), self.small, (178, 196, 208), max_width=card_width - 32, max_lines=2)
            self._text(experience.controls, (left + 16, y + 94), self.small, (138, 160, 176), max_width=card_width - 32, max_lines=1)

    def _draw_preview_panel(self, width: int, height: int) -> None:
        left = min(540, int(width * 0.48))
        top = 106
        panel_width = max(320, width - left - 32)
        panel_height = max(260, height - top - 126)
        self.pygame.draw.rect(self.screen, (10, 18, 26), (left, top, panel_width, panel_height), border_radius=6)
        self.pygame.draw.rect(self.screen, (56, 84, 100), (left, top, panel_width, panel_height), width=2, border_radius=6)
        experience = EXPERIENCES[self.selected]
        image_rect = (left + 18, top + 18, panel_width - 36, max(180, panel_height - 112))
        self._draw_preview_image(experience, image_rect)
        self._text(experience.script, (left + 20, top + panel_height - 78), self.small, (148, 171, 187), max_width=panel_width - 40)
        self._text("Preview images are rendered assets; launch opens an interactive demo window.", (left + 20, top + panel_height - 50), self.small, (184, 202, 214), max_width=panel_width - 40)

    def _draw_preview_image(self, experience: Experience, rect: tuple[int, int, int, int]) -> None:
        pygame = self.pygame
        x, y, width, height = rect
        pygame.draw.rect(self.screen, (4, 7, 10), rect)
        path = experience.preview_path
        surface = self._load_preview(path) if path is not None else None
        if surface is None:
            self._text("Preview not rendered", (x + 24, y + 24), self.font, (222, 230, 238))
            self._text("Use Render Preview to create the still image for this card.", (x + 24, y + 58), self.small, (160, 178, 190), max_width=width - 48)
            return
        source_width, source_height = surface.get_size()
        scale = min(width / source_width, height / source_height)
        target_size = (max(1, int(source_width * scale)), max(1, int(source_height * scale)))
        scaled = pygame.transform.smoothscale(surface, target_size)
        self.screen.blit(scaled, (x + (width - target_size[0]) // 2, y + (height - target_size[1]) // 2))

    def _load_preview(self, path: Path | None):
        if path is None:
            return None
        target = ROOT / path
        if not target.exists():
            return None
        mtime = target.stat().st_mtime
        cached = self.preview_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            image = self.pygame.image.load(str(target)).convert()
        except self.pygame.error:
            return None
        self.preview_cache[path] = (mtime, image)
        return image

    def _draw_button_bay(self, width: int, height: int) -> None:
        y = height - 74
        running = self.child is not None
        actions = (
            Button("launch", (34, y, 150, 42), "Launch", not running),
            Button("preview", (196, y, 170, 42), "Render Preview", not running),
            Button("settings", (378, y, 132, 42), "Settings", not running),
            Button("stop", (width - 278, y, 116, 42), "Stop", running),
            Button("exit", (width - 150, y, 116, 42), "Exit", True),
        )
        for button in actions:
            self._button(button)

    def _draw_settings(self, width: int, height: int) -> None:
        panel_width = min(560, width - 90)
        panel_height = 300
        left = (width - panel_width) // 2
        top = (height - panel_height) // 2
        self.pygame.draw.rect(self.screen, (8, 14, 20), (left, top, panel_width, panel_height), border_radius=6)
        self.pygame.draw.rect(self.screen, (95, 142, 165), (left, top, panel_width, panel_height), width=2, border_radius=6)
        self._text("Graphics Launch Settings", (left + 24, top + 22), self.font, (242, 247, 255))
        self._text("These settings stage launch profiles for demos that expose render quality. Apply commits them; Exit discards.", (left + 24, top + 58), self.small, (172, 192, 206), max_width=panel_width - 48)
        self._text(f"Profile: {self._profile_label(self.pending_settings)}", (left + 24, top + 112), self.font, (224, 236, 244))
        self._button(Button("quality_prev", (left + 24, top + 154, 92, 38), "Previous"))
        self._button(Button("quality_next", (left + 126, top + 154, 92, 38), "Next"))
        self._button(Button("safe", (left + 232, top + 154, 178, 38), "Safe Mode On" if self.pending_settings.safe_mode else "Safe Mode Off"))
        bay_y = top + panel_height - 62
        self._button(Button("apply", (left + 24, bay_y, 116, 40), "Apply"))
        self._button(Button("done", (left + 154, bay_y, 116, 40), "Done"))
        self._button(Button("cancel", (left + 284, bay_y, 116, 40), "Exit Menu"))

    def _button(self, button: Button) -> None:
        fill = (44, 84, 104) if button.enabled else (35, 42, 48)
        border = (132, 192, 220) if button.enabled else (70, 78, 84)
        text_color = (240, 248, 255) if button.enabled else (125, 134, 140)
        self.pygame.draw.rect(self.screen, fill, button.rect, border_radius=5)
        self.pygame.draw.rect(self.screen, border, button.rect, width=2, border_radius=5)
        text = self.small.render(button.label, True, text_color)
        x, y, width, height = button.rect
        self.screen.blit(text, (x + (width - text.get_width()) // 2, y + (height - text.get_height()) // 2))
        self.buttons.append(button)

    def _experience_rects(self) -> tuple[tuple[int, tuple[int, int, int, int]], ...]:
        width, _height = self.screen.get_size()
        card_width = min(470, max(340, int(width * 0.42)))
        return tuple((index, (32, 106 + index * 136, card_width, 122)) for index in range(len(EXPERIENCES)))

    def _text(
        self,
        text: str,
        position: tuple[int, int],
        font,
        color: tuple[int, int, int],
        *,
        max_width: int | None = None,
        max_lines: int = 3,
    ) -> None:
        x, y = position
        lines = self._wrap(text, font, max_width) if max_width else [text]
        for line in lines[:max_lines]:
            surface = font.render(line, True, color)
            self.screen.blit(surface, (x, y))
            y += surface.get_height() + 3

    @staticmethod
    def _wrap(text: str, font, max_width: int | None) -> list[str]:
        if max_width is None:
            return [text]
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    @staticmethod
    def _profile_label(settings: MenuSettings) -> str:
        return "safe" if settings.safe_mode else settings.quality


def run_showcase_menu() -> None:
    try:
        ShowcaseMenu().run()
    except ImportError as exc:
        print(f"Pygame menu unavailable: {exc}")
        print_menu()
        choice = input("> ").strip()
        if choice:
            run_experience(int(choice))


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the py_3d render engine showcase menu.")
    parser.add_argument("--list", action="store_true", help="Print available experiences and exit.")
    parser.add_argument("--run", type=int, help="Run a menu item by number without opening the pygame menu.")
    parser.add_argument("--render-preview", type=int, help="Render a preview still for a menu item by number.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quality", choices=("safe", "balanced", "high", "ultra"), default="safe")
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
