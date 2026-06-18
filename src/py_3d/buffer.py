"""Off-screen pixel and depth buffers."""

from __future__ import annotations

from binascii import crc32
from dataclasses import dataclass, field
from pathlib import Path
from struct import pack
from zlib import compress, decompress

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

    @classmethod
    def from_png(cls, path: str | Path) -> "PixelBuffer":
        """Read an 8-bit, non-interlaced PNG into a pixel buffer."""

        return _read_png(Path(path))

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

    def sample_nearest(self, u: float, v: float, *, wrap: bool = True) -> Color:
        if wrap:
            u = u % 1.0
            v = v % 1.0
        else:
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
        x = min(self.width - 1, max(0, int(u * self.width)))
        y = min(self.height - 1, max(0, int(v * self.height)))
        return self.pixels[y * self.width + x]

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
        x_indices = tuple(min(self.width - 1, int(x * self.width / width)) for x in range(width))
        scaled_rows: dict[int, list[Color]] = {}
        pixels: list[Color] = []
        for y in range(height):
            source_y = min(self.height - 1, int(y * self.height / height))
            row = scaled_rows.get(source_y)
            if row is None:
                row_start = source_y * self.width
                row = [self.pixels[row_start + source_x] for source_x in x_indices]
                scaled_rows[source_y] = row
            pixels.extend(row)
        return PixelBuffer(width, height, pixels)

    def to_rgb_bytes(self) -> bytes:
        payload = bytearray(len(self.pixels) * 3)
        offset = 0
        for pixel in self.pixels:
            payload[offset] = pixel.r
            payload[offset + 1] = pixel.g
            payload[offset + 2] = pixel.b
            offset += 3
        return bytes(payload)

    def to_ppm_bytes(self) -> bytes:
        """Return the buffer encoded as binary PPM bytes."""

        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        return header + self.to_rgb_bytes()

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
                rows.append(pixel.r)
                rows.append(pixel.g)
                rows.append(pixel.b)

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


def _read_png(path: Path) -> PixelBuffer:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG file: {path}")

    width = height = bit_depth = color_type = interlace = None
    palette: list[Color] | None = None
    transparency: bytes | None = None
    idat = bytearray()
    position = 8
    while position < len(data):
        length = int.from_bytes(data[position : position + 4], "big")
        kind = data[position + 4 : position + 8]
        chunk = data[position + 8 : position + 8 + length]
        position += 12 + length

        if kind == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = _unpack_ihdr(chunk)
        elif kind == b"PLTE":
            palette = [Color(chunk[i], chunk[i + 1], chunk[i + 2]) for i in range(0, len(chunk), 3)]
        elif kind == b"tRNS":
            transparency = chunk
        elif kind == b"IDAT":
            idat.extend(chunk)
        elif kind == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None or interlace is None:
        raise ValueError("PNG missing IHDR chunk")
    if bit_depth != 8:
        raise ValueError("only 8-bit PNG files are supported")
    if interlace != 0:
        raise ValueError("interlaced PNG files are not supported")

    channels = _png_channels(color_type)
    raw = decompress(bytes(idat))
    stride = width * channels
    rows = _unfilter_png_rows(raw, width, height, channels, stride)
    pixels = _png_rows_to_pixels(rows, width, height, color_type, palette, transparency)
    return PixelBuffer(width, height, pixels)


def _unpack_ihdr(chunk: bytes) -> tuple[int, int, int, int, int, int, int]:
    return (
        int.from_bytes(chunk[0:4], "big"),
        int.from_bytes(chunk[4:8], "big"),
        chunk[8],
        chunk[9],
        chunk[10],
        chunk[11],
        chunk[12],
    )


def _png_channels(color_type: int) -> int:
    if color_type == 0:
        return 1
    if color_type == 2:
        return 3
    if color_type == 3:
        return 1
    if color_type == 4:
        return 2
    if color_type == 6:
        return 4
    raise ValueError(f"unsupported PNG color type: {color_type}")


def _unfilter_png_rows(raw: bytes, width: int, height: int, channels: int, stride: int) -> list[bytes]:
    del width
    rows: list[bytes] = []
    previous = bytes(stride)
    position = 0
    for _ in range(height):
        filter_type = raw[position]
        position += 1
        current = bytearray(raw[position : position + stride])
        position += stride
        for index, value in enumerate(current):
            left = current[index - channels] if index >= channels else 0
            up = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                pass
            elif filter_type == 1:
                current[index] = (value + left) & 0xFF
            elif filter_type == 2:
                current[index] = (value + up) & 0xFF
            elif filter_type == 3:
                current[index] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                current[index] = (value + _paeth(left, up, upper_left)) & 0xFF
            else:
                raise ValueError(f"unsupported PNG filter type: {filter_type}")
        row = bytes(current)
        rows.append(row)
        previous = row
    return rows


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= up_distance and left_distance <= upper_left_distance:
        return left
    if up_distance <= upper_left_distance:
        return up
    return upper_left


def _png_rows_to_pixels(
    rows: list[bytes],
    width: int,
    height: int,
    color_type: int,
    palette: list[Color] | None,
    transparency: bytes | None,
) -> list[Color]:
    pixels: list[Color] = []
    for row in rows:
        for x in range(width):
            if color_type == 0:
                value = row[x]
                pixels.append(Color(value, value, value))
            elif color_type == 2:
                index = x * 3
                pixels.append(Color(row[index], row[index + 1], row[index + 2]))
            elif color_type == 3:
                if palette is None:
                    raise ValueError("palette PNG missing PLTE chunk")
                palette_index = row[x]
                color = palette[palette_index]
                if transparency is not None and palette_index < len(transparency):
                    alpha = transparency[palette_index] / 255.0
                    pixels.append(Color(color.r * alpha, color.g * alpha, color.b * alpha))
                else:
                    pixels.append(color)
            elif color_type == 4:
                index = x * 2
                value = row[index]
                alpha = row[index + 1] / 255.0
                pixels.append(Color(value * alpha, value * alpha, value * alpha))
            elif color_type == 6:
                index = x * 4
                alpha = row[index + 3] / 255.0
                pixels.append(Color(row[index] * alpha, row[index + 1] * alpha, row[index + 2] * alpha))
    if len(pixels) != width * height:
        raise ValueError("decoded PNG pixel count does not match dimensions")
    return pixels
