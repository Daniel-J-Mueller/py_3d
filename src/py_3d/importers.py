"""Small mesh import helpers for common interchange formats."""

from __future__ import annotations

from pathlib import Path
from struct import unpack_from

from .materials import Material
from .math3d import Vec3
from .primitives import Mesh, Triangle


def load_obj(path: str | Path, material: Material = Material()) -> Mesh:
    """Load a small Wavefront OBJ mesh.

    Supports vertex positions, texture coordinates, and polygon faces. Faces
    with more than three vertices are triangulated as a fan.
    """

    vertices: list[Vec3] = []
    texture_coordinates: list[tuple[float, float]] = []
    triangles: list[Triangle] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "vt" and len(parts) >= 3:
            texture_coordinates.append((float(parts[1]), 1.0 - float(parts[2])))
        elif parts[0] == "f" and len(parts) >= 4:
            face = [_parse_obj_face_token(token, len(vertices), len(texture_coordinates)) for token in parts[1:]]
            for index in range(1, len(face) - 1):
                triangles.append(_obj_triangle(face[0], face[index], face[index + 1], vertices, texture_coordinates, material))
    return Mesh(triangles)


def load_stl(path: str | Path, material: Material = Material()) -> Mesh:
    """Load an STL mesh, accepting ASCII or binary STL."""

    source = Path(path)
    data = source.read_bytes()
    if _looks_like_binary_stl(data):
        return _load_binary_stl(data, material)
    return _load_ascii_stl(source.read_text(encoding="utf-8", errors="replace"), material)


def _parse_obj_face_token(token: str, vertex_count: int, texture_count: int) -> tuple[int, int | None]:
    values = token.split("/")
    vertex_index = _resolve_obj_index(int(values[0]), vertex_count)
    texture_index = None
    if len(values) >= 2 and values[1]:
        texture_index = _resolve_obj_index(int(values[1]), texture_count)
    return vertex_index, texture_index


def _resolve_obj_index(index: int, count: int) -> int:
    resolved = index - 1 if index > 0 else count + index
    if resolved < 0 or resolved >= count:
        raise ValueError("OBJ face index out of range")
    return resolved


def _obj_triangle(
    a: tuple[int, int | None],
    b: tuple[int, int | None],
    c: tuple[int, int | None],
    vertices: list[Vec3],
    texture_coordinates: list[tuple[float, float]],
    material: Material,
) -> Triangle:
    uv_a = texture_coordinates[a[1]] if a[1] is not None else None
    uv_b = texture_coordinates[b[1]] if b[1] is not None else None
    uv_c = texture_coordinates[c[1]] if c[1] is not None else None
    return Triangle(vertices[a[0]], vertices[b[0]], vertices[c[0]], material, uv_a, uv_b, uv_c)


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = int.from_bytes(data[80:84], "little")
    expected_length = 84 + triangle_count * 50
    return expected_length == len(data)


def _load_binary_stl(data: bytes, material: Material) -> Mesh:
    triangle_count = int.from_bytes(data[80:84], "little")
    triangles: list[Triangle] = []
    offset = 84
    for _ in range(triangle_count):
        values = unpack_from("<12fH", data, offset)
        a = Vec3(values[3], values[4], values[5])
        b = Vec3(values[6], values[7], values[8])
        c = Vec3(values[9], values[10], values[11])
        triangles.append(Triangle(a, b, c, material))
        offset += 50
    return Mesh(triangles)


def _load_ascii_stl(text: str, material: Material) -> Mesh:
    vertices: list[Vec3] = []
    triangles: list[Triangle] = []
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
            if len(vertices) == 3:
                triangles.append(Triangle(vertices[0], vertices[1], vertices[2], material))
                vertices = []
    return Mesh(triangles)
