"""Off-screen pixel and depth buffers."""

from __future__ import annotations

from binascii import crc32
from dataclasses import dataclass, field
from pathlib import Path
from struct import pack
from zlib import compress

from .color import Color


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("buffer dimensions must be positive")


@dataclass
class PixelBuffer:
    """A simple row-major RGB pixel buffer."""

    width: int
    height: int
    pixels: list[Color] = field(repr=False)

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        expected = self.width * self.height
        if len(self.pixels) != expected:
            raise ValueError(f"pixel buffer requires {expected} pixels")

    @classmethod
    def new(cls, width: int, height: int, fill: Color | tuple[int, int, int] | None = None) -> "PixelBuffer":
        color = Color.from_value(fill or Color(0, 0, 0))
        return cls(width=width, height=height, pixels=[color] * (width * height))

    def clear(self, color: Color | tuple[int, int, int]) -> None:
        fill = Color.from_value(color)
        self.pixels[:] = [fill] * (self.width * self.height)

    def copy(self) -> "PixelBuffer":
        return PixelBuffer(self.width, self.height, self.pixels.copy())

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def index(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            raise IndexError(f"pixel coordinate out of bounds: {(x, y)}")
        return y * self.width + x

    def set_pixel(self, x: int, y: int, color: Color | tuple[int, int, int]) -> None:
        if self.in_bounds(x, y):
            self.pixels[y * self.width + x] = Color.from_value(color)

    def get_pixel(self, x: int, y: int) -> Color:
        return self.pixels[self.index(x, y)]

    def rows(self) -> list[list[Color]]:
        return [
            self.pixels[y * self.width : (y + 1) * self.width]
            for y in range(self.height)
        ]

    def resized_nearest(self, width: int, height: int) -> "PixelBuffer":
        """Return a nearest-neighbor resized copy."""

        _validate_dimensions(width, height)
        if width == self.width and height == self.height:
            return self.copy()
        pixels: list[Color] = []
        for y in range(height):
            source_y = min(self.height - 1, int(y * self.height / height))
            for x in range(width):
                source_x = min(self.width - 1, int(x * self.width / width))
                pixels.append(self.pixels[source_y * self.width + source_x])
        return PixelBuffer(width, height, pixels)

    def to_ppm_bytes(self) -> bytes:
        """Return the buffer encoded as binary PPM bytes."""

        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        payload = bytearray()
        for pixel in self.pixels:
            payload.extend(pixel.to_tuple())
        return header + bytes(payload)

    def to_ppm(self, path: str | Path) -> None:
        """Write the buffer as a binary PPM image without extra dependencies."""

        Path(path).write_bytes(self.to_ppm_bytes())

    def to_png(self, path: str | Path) -> None:
        """Write the buffer as a truecolor PNG without extra dependencies."""

        target = Path(path)
        rows = bytearray()
        for y in range(self.height):
            rows.append(0)
            for pixel in self.pixels[y * self.width : (y + 1) * self.width]:
                rows.extend(pixel.to_tuple())

        payload = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                _png_chunk(b"IHDR", pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)),
                _png_chunk(b"IDAT", compress(bytes(rows))),
                _png_chunk(b"IEND", b""),
            ]
        )
        target.write_bytes(payload)


@dataclass
class DepthBuffer:
    """A row-major depth buffer where lower values are closer to the camera."""

    width: int
    height: int
    values: list[float] = field(repr=False)

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        expected = self.width * self.height
        if len(self.values) != expected:
            raise ValueError(f"depth buffer requires {expected} values")

    @classmethod
    def new(cls, width: int, height: int, fill: float = float("inf")) -> "DepthBuffer":
        return cls(width=width, height=height, values=[fill] * (width * height))

    def clear(self, value: float = float("inf")) -> None:
        self.values[:] = [value] * (self.width * self.height)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> float:
        if not self.in_bounds(x, y):
            raise IndexError(f"depth coordinate out of bounds: {(x, y)}")
        return self.values[y * self.width + x]

    def test_and_set(self, x: int, y: int, depth: float) -> bool:
        if not self.in_bounds(x, y):
            return False
        index = y * self.width + x
        if depth < self.values[index]:
            self.values[index] = depth
            return True
        return False


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = crc32(kind + data) & 0xFFFFFFFF
    return pack(">I", len(data)) + kind + data + pack(">I", checksum)
