"""Immediate-mode 2D drawing helpers for pixel buffers."""

from __future__ import annotations

from .buffer import PixelBuffer
from .color import Color


FONT_5X7: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ",": ("00000", "00000", "00000", "00000", "01100", "00100", "01000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "11100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10011", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def point(buffer: PixelBuffer, position: tuple[int, int], color: Color | tuple[int, int, int]) -> None:
    x, y = position
    buffer.set_pixel(int(x), int(y), color)


def line(
    buffer: PixelBuffer,
    start: tuple[int, int],
    end: tuple[int, int],
    color: Color | tuple[int, int, int],
) -> None:
    """Draw a 2D line using integer Bresenham stepping."""

    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    while True:
        buffer.set_pixel(x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def rect(
    buffer: PixelBuffer,
    top_left: tuple[int, int],
    size: tuple[int, int],
    color: Color | tuple[int, int, int],
    *,
    fill: bool = False,
) -> None:
    x, y = int(top_left[0]), int(top_left[1])
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("rectangle size must be positive")

    if fill:
        for yy in range(y, y + height):
            for xx in range(x, x + width):
                buffer.set_pixel(xx, yy, color)
        return

    line(buffer, (x, y), (x + width - 1, y), color)
    line(buffer, (x, y), (x, y + height - 1), color)
    line(buffer, (x + width - 1, y), (x + width - 1, y + height - 1), color)
    line(buffer, (x, y + height - 1), (x + width - 1, y + height - 1), color)


def circle(
    buffer: PixelBuffer,
    center: tuple[int, int],
    radius: int,
    color: Color | tuple[int, int, int],
    *,
    fill: bool = False,
) -> None:
    if radius <= 0:
        raise ValueError("circle radius must be positive")

    cx, cy = int(center[0]), int(center[1])
    if fill:
        radius_squared = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius_squared:
                    buffer.set_pixel(x, y, color)
        return

    x = radius
    y = 0
    error = 1 - x
    while x >= y:
        for px, py in (
            (cx + x, cy + y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx - x, cy + y),
            (cx - x, cy - y),
            (cx - y, cy - x),
            (cx + y, cy - x),
            (cx + x, cy - y),
        ):
            buffer.set_pixel(px, py, color)
        y += 1
        if error < 0:
            error += 2 * y + 1
        else:
            x -= 1
            error += 2 * (y - x) + 1


def text_size(text: str, *, scale: int = 1, spacing: int = 1, line_spacing: int = 1) -> tuple[int, int]:
    if scale <= 0:
        raise ValueError("text scale must be positive")
    lines = text.splitlines() or [""]
    max_columns = max((len(line) for line in lines), default=0)
    width = 0 if max_columns == 0 else max_columns * 5 * scale + (max_columns - 1) * spacing * scale
    height = len(lines) * 7 * scale + (len(lines) - 1) * line_spacing * scale
    return width, height


def text(
    buffer: PixelBuffer,
    position: tuple[int, int],
    value: str,
    color: Color | tuple[int, int, int],
    *,
    scale: int = 1,
    spacing: int = 1,
    line_spacing: int = 1,
) -> None:
    """Draw simple bitmap text into a pixel buffer."""

    if scale <= 0:
        raise ValueError("text scale must be positive")
    draw_color = Color.from_value(color)
    start_x, start_y = int(position[0]), int(position[1])
    for line_index, raw_line in enumerate(value.splitlines() or [""]):
        y_offset = line_index * (7 + line_spacing) * scale
        for char_index, raw_char in enumerate(raw_line.upper()):
            glyph = FONT_5X7.get(raw_char, FONT_5X7["?"])
            x_offset = char_index * (5 + spacing) * scale
            for gy, row in enumerate(glyph):
                for gx, mark in enumerate(row):
                    if mark != "1":
                        continue
                    for sy in range(scale):
                        for sx in range(scale):
                            buffer.set_pixel(
                                start_x + x_offset + gx * scale + sx,
                                start_y + y_offset + gy * scale + sy,
                                draw_color,
                            )
