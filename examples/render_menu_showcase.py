"""Render documentation screenshots for launcher and live menu layouts."""

from __future__ import annotations

from pathlib import Path

from py_3d import draw
from py_3d.buffer import PixelBuffer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "renderings-tests"


def _fill_gradient(buffer: PixelBuffer, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    for y in range(buffer.height):
        amount = y / max(1, buffer.height - 1)
        color = tuple(int(top[index] + (bottom[index] - top[index]) * amount) for index in range(3))
        for x in range(buffer.width):
            buffer.set_pixel(x, y, color)


def _text(buffer: PixelBuffer, position: tuple[int, int], value: str, color: tuple[int, int, int], *, scale: int = 1) -> None:
    draw.text(buffer, position, value, color, scale=scale)


def _button(buffer: PixelBuffer, rect: tuple[int, int, int, int], label: str, *, active: bool = True) -> None:
    x, y, width, height = rect
    fill = (54, 54, 54) if active else (34, 34, 34)
    border = (180, 180, 180) if active else (86, 86, 86)
    draw.rect(buffer, (x, y), (width, height), fill, fill=True)
    draw.rect(buffer, (x, y), (width, height), border)
    text_width, text_height = draw.text_size(label)
    _text(buffer, (x + (width - text_width) // 2, y + (height - text_height) // 2), label, (244, 244, 244))


def save_launcher_menu() -> None:
    buffer = PixelBuffer.new(1180, 720, (15, 15, 15))
    _fill_gradient(buffer, (13, 13, 13), (27, 27, 27))
    draw.rect(buffer, (52, 42), (1076, 632), (24, 24, 24), fill=True)
    draw.rect(buffer, (52, 42), (1076, 632), (118, 118, 118))
    _text(buffer, (76, 68), "PY_3D", (246, 246, 246), scale=3)
    _text(buffer, (78, 108), "PROFILE HIGH", (166, 166, 166))
    _text(buffer, (612, 108), "SELECT AN EXPERIENCE AND LAUNCH OR RENDER A PREVIEW", (196, 196, 196))

    draw.rect(buffer, (76, 146), (390, 430), (22, 22, 22), fill=True)
    draw.rect(buffer, (76, 146), (390, 430), (68, 68, 68))
    _text(buffer, (96, 166), "EXPERIENCES", (235, 235, 235), scale=2)
    rows = (
        ("FRUIT BOWL PHYSICS", "TEXTURED FRUIT, GRAB PHYSICS, LIVE SETTINGS"),
        ("FRUIT BOWL RGB BULBS", "R G B BULBS BLINK THROUGH EIGHT STATES"),
        ("CAPSULE PLAYER CONTROLLER", "FPS DEFAULT, HUDS, COLLISION, CAMERA MODES"),
        ("FAN CLOTH WATER", "CLOTH, WIND, WATER, AND BOWL CONTACT"),
        ("SEA LION ASSET VIEWER", "PREPARED MESH ASSET AND SKY CONTROLS"),
    )
    for index, (title, detail) in enumerate(rows):
        y = 204 + index * 68
        selected = index == 0
        draw.rect(buffer, (96, y), (350, 54), (62, 62, 62) if selected else (31, 31, 31), fill=True)
        draw.rect(buffer, (96, y), (350, 54), (214, 214, 214) if selected else (76, 76, 76))
        _text(buffer, (110, y + 10), title, (245, 245, 245))
        _text(buffer, (110, y + 30), detail, (162, 162, 162))

    draw.rect(buffer, (504, 146), (548, 296), (20, 20, 20), fill=True)
    draw.rect(buffer, (504, 146), (548, 296), (70, 70, 70))
    draw.rect(buffer, (528, 170), (500, 180), (10, 12, 16), fill=True)
    draw.circle(buffer, (652, 244), 44, (210, 92, 64), fill=True)
    draw.circle(buffer, (748, 238), 38, (238, 190, 72), fill=True)
    draw.circle(buffer, (836, 252), 42, (88, 164, 92), fill=True)
    _text(buffer, (528, 374), "FRUIT BOWL PHYSICS", (238, 238, 238), scale=2)
    _text(buffer, (528, 404), "WASD MOUSE LOOK, E GRAB/DROP, ESC MENU", (166, 166, 166))

    y = 610
    _button(buffer, (76, y, 150, 40), "LAUNCH")
    _button(buffer, (240, y, 178, 40), "RENDER")
    _button(buffer, (432, y, 134, 40), "SETTINGS")
    _button(buffer, (856, y, 116, 40), "STOP", active=False)
    _button(buffer, (986, y, 116, 40), "EXIT")
    buffer.to_png(OUTPUT_DIR / "demo_launcher_menu.png")


def save_launcher_settings() -> None:
    buffer = PixelBuffer.new(1180, 720, (15, 15, 15))
    save_launcher_menu()
    buffer = PixelBuffer.from_png(OUTPUT_DIR / "demo_launcher_menu.png")
    draw.rect(buffer, (330, 214), (520, 268), (8, 8, 8), fill=True)
    draw.rect(buffer, (330, 214), (520, 268), (148, 148, 148))
    _text(buffer, (356, 244), "LAUNCH SETTINGS", (246, 246, 246), scale=2)
    _text(buffer, (356, 300), "QUALITY", (220, 220, 220))
    _button(buffer, (482, 286, 34, 34), "-")
    draw.rect(buffer, (526, 286), (150, 34), (22, 22, 22), fill=True)
    draw.rect(buffer, (526, 286), (150, 34), (86, 86, 86))
    _text(buffer, (584, 299), "HIGH", (240, 240, 240))
    _button(buffer, (686, 286, 34, 34), "+")
    draw.rect(buffer, (356, 350), (18, 18), (24, 24, 24), fill=True)
    draw.rect(buffer, (356, 350), (18, 18), (170, 170, 170))
    _text(buffer, (388, 354), "MENU BLUR", (228, 228, 228))
    _button(buffer, (356, 416, 100, 34), "APPLY")
    _button(buffer, (470, 416, 100, 34), "DONE")
    _button(buffer, (584, 416, 100, 34), "CANCEL")
    buffer.to_png(OUTPUT_DIR / "demo_launcher_settings.png")


def save_live_settings_menu() -> None:
    width, height = 1280, 720
    buffer = PixelBuffer.new(width, height, (14, 18, 22))
    _fill_gradient(buffer, (12, 22, 32), (30, 42, 50))
    draw.rect(buffer, (0, 510), (width, 210), (32, 46, 48), fill=True)
    for x in range(0, width, 64):
        draw.line(buffer, (x, 510), (x + 170, height), (58, 74, 74))
    for y in range(510, height, 36):
        draw.line(buffer, (0, y), (width, y), (58, 74, 74))

    panel = (390, 176, 500, 366)
    draw.rect(buffer, (panel[0], panel[1]), (panel[2], panel[3]), (5, 5, 5), fill=True)
    draw.rect(buffer, (panel[0], panel[1]), (panel[2], panel[3]), (92, 92, 92))
    _text(buffer, (panel[0] + 20, panel[1] + 18), "PY_3D", (240, 240, 240), scale=2)
    draw.line(buffer, (panel[0] + 20, panel[1] + 52), (panel[0] + panel[2] - 20, panel[1] + 52), (86, 86, 86))

    tabs = ("GRAPHICS", "SKY", "PHYSICS", "DEMO")
    x = panel[0] + 20
    for index, tab in enumerate(tabs):
        tab_width = 92 if index == 0 else 74
        draw.rect(buffer, (x, panel[1] + 68), (tab_width, 24), (38, 38, 38) if index == 0 else (20, 20, 20), fill=True)
        draw.rect(buffer, (x, panel[1] + 68), (tab_width, 24), (86, 86, 86))
        _text(buffer, (x + 10, panel[1] + 77), tab, (236, 236, 236))
        x += tab_width + 8

    rows = (
        ("QUALITY", "HIGH", None),
        ("POLYGONS", "18 X 9", "+-"),
        ("REFLECTIONS", "2", "+-"),
        ("TEXTURE", "256", "+-"),
        ("GAMMA", "1.12", "+-"),
        ("TONE MAP", "ON", None),
    )
    y = panel[1] + 116
    for index, (label, value, buttons) in enumerate(rows):
        fill = (38, 38, 38) if index == 1 else (12, 12, 12)
        draw.rect(buffer, (panel[0] + 20, y), (panel[2] - 40, 38), fill, fill=True)
        draw.line(buffer, (panel[0] + 20, y + 37), (panel[0] + panel[2] - 20, y + 37), (42, 42, 42))
        _text(buffer, (panel[0] + 32, y + 14), label, (238, 238, 238))
        _text(buffer, (panel[0] + 238, y + 14), value, (158, 158, 158))
        if buttons:
            _button(buffer, (panel[0] + panel[2] - 84, y + 6, 26, 26), "-")
            _button(buffer, (panel[0] + panel[2] - 48, y + 6, 26, 26), "+")
        y += 38

    draw.line(buffer, (panel[0] + 20, panel[1] + panel[3] - 52), (panel[0] + panel[2] - 20, panel[1] + panel[3] - 52), (86, 86, 86))
    _button(buffer, (panel[0] + 20, panel[1] + panel[3] - 40, 82, 30), "DONE")
    _button(buffer, (panel[0] + 110, panel[1] + panel[3] - 40, 82, 30), "APPLY")
    _button(buffer, (panel[0] + 200, panel[1] + panel[3] - 40, 82, 30), "CANCEL")
    buffer.to_png(OUTPUT_DIR / "live_settings_menu.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_launcher_menu()
    save_launcher_settings()
    save_live_settings_menu()
    print(f"Wrote menu screenshots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
