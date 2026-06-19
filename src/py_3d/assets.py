"""Load py_3d-ready mesh assets produced by ingestion scripts."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

from .buffer import PixelBuffer
from .materials import Material
from .math3d import Vec3
from .primitives import Mesh, Triangle


MESH_ASSET_FORMAT = "py_3d.mesh_asset.v1"
FRAMEWORK_FAVICON = "py_3d_logo.png"


def framework_asset_path(name: str) -> Path:
    """Return the path to a framework-level asset bundled with the workspace."""

    root = Path(__file__).resolve().parents[2]
    candidates = (
        root / "assets" / name,
        Path.cwd() / "assets" / name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def framework_favicon_path() -> Path:
    """Return the default py_3d window icon path."""

    return framework_asset_path(FRAMEWORK_FAVICON)


def load_framework_icon_rgba(path: str | Path | None = None) -> tuple[int, int, list[list[tuple[int, int, int, int]]]]:
    """Load the framework icon as rows of RGBA pixels for window backends."""

    source = Path(path) if path is not None else framework_favicon_path()
    if not source.exists():
        return _fallback_icon_rgba()
    if source.suffix.lower() == ".png":
        try:
            return _rgba_from_png(source)
        except Exception:
            return _fallback_icon_rgba()
    payload = source.read_bytes()
    try:
        return _rgba_from_ico(payload)
    except Exception:
        return _fallback_icon_rgba()


def load_mesh_asset(path: str | Path, material: Material | None = None) -> Mesh:
    """Load a prepared mesh asset JSON file into a renderable ``Mesh``."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != MESH_ASSET_FORMAT:
        raise ValueError(f"unsupported mesh asset format: {payload.get('format')!r}")
    vertices = [Vec3(*values) for values in payload.get("vertices", [])]
    uvs = [tuple(values) for values in payload.get("uvs", [])]
    mesh_material = material or _material_from_payload(payload.get("material", {}))
    triangles: list[Triangle] = []
    for entry in payload.get("triangles", []):
        if len(entry) != 6:
            raise ValueError("mesh asset triangle entries must have six indices")
        a, b, c, uv_a, uv_b, uv_c = (int(value) for value in entry)
        triangles.append(
            Triangle(
                vertices[a],
                vertices[b],
                vertices[c],
                mesh_material,
                uvs[uv_a] if uv_a >= 0 else None,
                uvs[uv_b] if uv_b >= 0 else None,
                uvs[uv_c] if uv_c >= 0 else None,
            )
        )
    return Mesh(triangles)


def _rgba_from_ico(payload: bytes) -> tuple[int, int, list[list[tuple[int, int, int, int]]]]:
    if len(payload) < 22:
        raise ValueError("ICO payload is too short")
    reserved, icon_type, count = struct.unpack_from("<HHH", payload, 0)
    if reserved != 0 or icon_type != 1 or count <= 0:
        raise ValueError("not a Windows icon")
    entries = []
    for index in range(count):
        offset = 6 + index * 16
        width_byte, height_byte, _color_count, _reserved, _planes, bit_count, size, image_offset = struct.unpack_from("<BBBBHHII", payload, offset)
        width = 256 if width_byte == 0 else width_byte
        height = 256 if height_byte == 0 else height_byte
        entries.append((width, height, bit_count, size, image_offset))
    width, height, bit_count, size, image_offset = max(entries, key=lambda entry: entry[0] * entry[1])
    image = payload[image_offset : image_offset + size]
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("PNG-compressed ICO entries need an image loader")
    if len(image) < 40:
        raise ValueError("ICO bitmap is too short")
    header_size, dib_width, dib_height, planes, dib_bit_count, compression, _size_image, *_rest = struct.unpack_from("<IIIHHIIIIII", image, 0)
    if header_size < 40 or planes != 1 or bit_count != 32 or dib_bit_count != 32 or compression != 0:
        raise ValueError("only uncompressed 32-bit ICO entries are supported")
    if dib_width != width:
        raise ValueError("ICO width does not match bitmap header")
    actual_height = dib_height // 2
    if actual_height != height:
        raise ValueError("ICO height does not match bitmap header")
    pixel_offset = header_size
    pixel_bytes = image[pixel_offset : pixel_offset + width * height * 4]
    if len(pixel_bytes) != width * height * 4:
        raise ValueError("ICO pixel data is truncated")

    rows: list[list[tuple[int, int, int, int]]] = []
    for y in range(height):
        source_y = height - 1 - y
        row: list[tuple[int, int, int, int]] = []
        for x in range(width):
            offset = (source_y * width + x) * 4
            blue, green, red, alpha = pixel_bytes[offset : offset + 4]
            row.append((int(red), int(green), int(blue), int(alpha)))
        rows.append(row)
    return width, height, rows


def _rgba_from_png(path: Path) -> tuple[int, int, list[list[tuple[int, int, int, int]]]]:
    icon = PixelBuffer.from_png(path).resized_nearest(32, 32)
    rows = [
        [(pixel.r, pixel.g, pixel.b, 255) for pixel in row]
        for row in icon.rows()
    ]
    return icon.width, icon.height, rows


def _fallback_icon_rgba() -> tuple[int, int, list[list[tuple[int, int, int, int]]]]:
    size = 16
    rows: list[list[tuple[int, int, int, int]]] = []
    for y in range(size):
        row = []
        for x in range(size):
            if 3 <= x <= 12 and 3 <= y <= 12:
                row.append((38, 86, 168, 255))
            elif 1 <= x <= 9 and 7 <= y <= 14:
                row.append((172, 88, 32, 255))
            else:
                row.append((0, 0, 0, 0))
        rows.append(row)
    return size, size, rows


def mesh_asset_metadata(path: str | Path) -> dict[str, Any]:
    """Return metadata for a prepared mesh asset without building triangles."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        key: payload.get(key)
        for key in ("format", "name", "source", "bounds", "triangle_count", "vertex_count", "uv_count", "ingest")
    }


def _material_from_payload(payload: dict[str, Any]) -> Material:
    return Material(
        color=tuple(payload.get("color", (180, 180, 180))),
        roughness=payload.get("roughness", 0.4),
        fuzziness=payload.get("fuzziness", 0.0),
        specular=payload.get("specular", 0.1),
        shininess=payload.get("shininess", 24.0),
    )
