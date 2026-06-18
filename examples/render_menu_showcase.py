"""Render documentation screenshots for the demo launcher and live settings menu."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "renderings-tests"


def _load_demo_menu_module():
    path = ROOT / "USER" / "demos" / "00_list_experiences.py"
    spec = importlib.util.spec_from_file_location("py3dengine_demo_menu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_launcher_menu() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    module = _load_demo_menu_module()
    menu = module.ShowcaseMenu()
    try:
        menu.selected = 0
        menu.show_settings = False
        menu.status = "Select a feature demo, render a preview, or open settings."
        menu._draw()
        menu.pygame.image.save(menu.screen, OUTPUT_DIR / "demo_launcher_menu.png")

        menu.show_settings = True
        menu.pending_settings = module.MenuSettings(quality="high", safe_mode=False)
        menu._draw()
        menu.pygame.image.save(menu.screen, OUTPUT_DIR / "demo_launcher_settings.png")
    finally:
        menu.pygame.quit()


def save_live_settings_menu() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from py_3d.live import LiveMenu, LiveMenuOption, render_live_menu_surface

    pygame.init()
    width, height = 1280, 720
    screen = pygame.Surface((width, height))
    for y in range(height):
        t = y / max(1, height - 1)
        color = (16 + int(22 * t), 32 + int(60 * t), 52 + int(68 * t))
        pygame.draw.line(screen, color, (0, y), (width, y))
    pygame.draw.rect(screen, (34, 52, 58), (0, 510, width, 210))
    for x in range(0, width, 64):
        pygame.draw.line(screen, (64, 86, 86), (x, 510), (x + 170, height), 1)
    for y in range(510, height, 36):
        pygame.draw.line(screen, (64, 86, 86), (0, y), (width, y), 1)

    menu = LiveMenu(
        "py3dengine live graphics",
        (
            LiveMenuOption("quality", "Quality: High", "Render scale, mesh density, shadows", "Graphics"),
            LiveMenuOption("poly_up", "More Polygons", "Increase generated mesh detail", "Graphics"),
            LiveMenuOption("poly_down", "Fewer Polygons", "Lower geometry cost", "Graphics"),
            LiveMenuOption("reflection_up", "Reflections +", "Raise reflection bounce budget", "Graphics"),
            LiveMenuOption("reflection_down", "Reflections -", "Lower reflection bounce budget", "Graphics"),
            LiveMenuOption("texture", "Texture Resolution", "384 px procedural textures", "Graphics"),
            LiveMenuOption("tone", "Tone Mapping", "Toggle high range display mapping", "Graphics"),
            LiveMenuOption("sky_cycle", "Day/Night Cycle", "Animate the sky clock", "Sky"),
            LiveMenuOption("sky_time_up", "Time Later", "Move sun and sky forward", "Sky"),
            LiveMenuOption("sky_time_down", "Time Earlier", "Move sun and sky backward", "Sky"),
            LiveMenuOption("sun_up", "Sun Higher", "Raise solar elevation", "Sky"),
            LiveMenuOption("sun_down", "Sun Lower", "Lower solar elevation", "Sky"),
            LiveMenuOption("clouds", "Clouds", "Toggle procedural cloud layer", "Sky"),
            LiveMenuOption("stars", "Stars", "Toggle night star field", "Sky"),
            LiveMenuOption("pause", "Pause Physics", "Freeze or resume simulation", "Physics"),
            LiveMenuOption("reset", "Reset Scene", "Restore objects safely", "Physics"),
            LiveMenuOption("snapshot", "Save Snapshot", "Write a PNG render", "Demo"),
            LiveMenuOption("done", "Done"),
            LiveMenuOption("apply", "Apply"),
            LiveMenuOption("cancel", "Exit Menu"),
            LiveMenuOption("quit", "Quit Demo"),
        ),
        selected_index=0,
        visible=True,
    )
    menu.set_group("Graphics")
    panel, left, top = render_live_menu_surface(menu, pygame, width, height)
    screen.blit(panel, (left, top))
    pygame.image.save(screen, OUTPUT_DIR / "live_settings_menu.png")
    pygame.quit()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_launcher_menu()
    save_live_settings_menu()
    print(f"Wrote menu screenshots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
