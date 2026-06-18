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
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "'": ("00100", "00100", "01000", "00000", "00000", "00000", "00000"),
    '"': ("01010", "01010", "01010", "00000", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "<": ("00001", "00010", "00100", "01000", "00100", "00010", "00001"),
    ">": ("10000", "01000", "00100", "00010", "00100", "01000", "10000"),
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

_GLYPH_RUNS: dict[str, tuple[tuple[int, int, int], ...]] = {}


def _column_width(columns: int, *, scale: int = 1, spacing: int = 1) -> int:
    if columns <= 0:
        return 0
    return columns * 5 * scale + (columns - 1) * spacing * scale


def _max_columns_for_width(width: int, *, scale: int = 1, spacing: int = 1) -> int:
    if width <= 0:
        return 0
    return max(0, (int(width) + spacing * scale) // ((5 + spacing) * scale))


def _glyph_runs(char: str) -> tuple[tuple[int, int, int], ...]:
    cached = _GLYPH_RUNS.get(char)
    if cached is not None:
        return cached
    glyph = FONT_5X7.get(char, FONT_5X7["?"])
    runs: list[tuple[int, int, int]] = []
    for gy, row in enumerate(glyph):
        run_start: int | None = None
        for gx, mark in enumerate(row + "0"):
            if mark == "1" and run_start is None:
                run_start = gx
                continue
            if mark == "1" or run_start is None:
                continue
            runs.append((gy, run_start, gx - run_start))
            run_start = None
    cached = tuple(runs)
    _GLYPH_RUNS[char] = cached
    return cached


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
    if y0 == y1:
        left = min(x0, x1)
        buffer.fill_rect((left, y0), (abs(x1 - x0) + 1, 1), color)
        return
    if x0 == x1:
        top = min(y0, y1)
        buffer.fill_rect((x0, top), (1, abs(y1 - y0) + 1), color)
        return
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
        buffer.fill_rect((x, y), (width, height), color)
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
    width = _column_width(max_columns, scale=scale, spacing=spacing)
    height = len(lines) * 7 * scale + (len(lines) - 1) * line_spacing * scale
    return width, height


def fit_text(value: str, max_width: int, *, scale: int = 1, spacing: int = 1, suffix: str = "..") -> str:
    """Return text shortened to fit within ``max_width`` pixels."""

    text_value = " ".join(str(value).split())
    if _column_width(len(text_value), scale=scale, spacing=spacing) <= max_width:
        return text_value
    suffix_width = _column_width(len(suffix), scale=scale, spacing=spacing)
    if suffix_width > max_width:
        return ""
    max_columns = _max_columns_for_width(max_width, scale=scale, spacing=spacing)
    result = text_value[: max(0, max_columns - len(suffix))].rstrip()
    return result + suffix if result else suffix


def wrap_text(
    value: str,
    max_width: int,
    *,
    scale: int = 1,
    spacing: int = 1,
    max_lines: int = 4,
) -> list[str]:
    """Wrap text by rendered pixel width."""

    words = str(value).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(candidate, scale=scale, spacing=spacing)[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            if len(lines) >= max_lines:
                return [fit_text(line, max_width, scale=scale, spacing=spacing) for line in lines[:max_lines]]
        current = fit_text(word, max_width, scale=scale, spacing=spacing) if text_size(word, scale=scale, spacing=spacing)[0] > max_width else word
    if current and len(lines) < max_lines:
        lines.append(current)
    return [fit_text(line, max_width, scale=scale, spacing=spacing) for line in lines[:max_lines]]


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
            runs = _glyph_runs(raw_char)
            x_offset = char_index * (5 + spacing) * scale
            for gy, run_start, run_width in runs:
                buffer.fill_rect(
                    (start_x + x_offset + run_start * scale, start_y + y_offset + gy * scale),
                    (run_width * scale, scale),
                    draw_color,
                )
