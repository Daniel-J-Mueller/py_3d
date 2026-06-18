"""Convert OBJ meshes into py_3d-ready mesh asset JSON files."""

from __future__ import annotations

import argparse
import json
from math import cos, radians, sin
from pathlib import Path

from py_3d import MESH_ASSET_FORMAT, Vec3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an OBJ as a py_3d mesh asset.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("USER") / "assets")
    parser.add_argument("--name")
    parser.add_argument("--target-triangles", type=int, help="Evenly decimate to roughly this many triangles.")
    parser.add_argument("--face-step", type=int, help="Keep every Nth source face after triangulation.")
    parser.add_argument("--source-up", choices=("y", "z"), default="z")
    parser.add_argument("--scale-to-height", type=float, default=1.2)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--color", type=int, nargs=3, default=(104, 86, 70))
    parser.add_argument("--roughness", type=float, default=0.32)
    parser.add_argument("--fuzziness", type=float, default=0.04)
    parser.add_argument("--specular", type=float, default=0.22)
    parser.add_argument("--shininess", type=float, default=34.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"asset source not found: {args.source}")
    if args.target_triangles is not None and args.target_triangles <= 0:
        raise ValueError("target triangle count must be positive")
    if args.face_step is not None and args.face_step <= 0:
        raise ValueError("face step must be positive")

    name = args.name or args.source.stem
    vertices, uvs, triangles = _read_obj(args.source)
    transformed = _normalize_vertices(
        vertices,
        source_up=args.source_up,
        scale_to_height=args.scale_to_height,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
    )
    selected = _select_triangles(triangles, args.face_step)
    simplification = {"method": "preserve", "grid_divisions": None}
    grid_divisions = None
    if args.target_triangles is not None and len(selected) > args.target_triangles:
        grid_divisions = _choose_grid_divisions(transformed, selected, args.target_triangles)
        simplification = {"method": "vertex_grid", "grid_divisions": grid_divisions}
    used_vertices, used_uvs, compact_triangles = _compact_geometry(transformed, uvs, selected, grid_divisions=grid_divisions)
    bounds = _bounds_for(used_vertices)

    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.py3dmesh.json"
    payload = {
        "format": MESH_ASSET_FORMAT,
        "name": name,
        "source": str(args.source),
        "bounds": bounds,
        "vertex_count": len(used_vertices),
        "uv_count": len(used_uvs),
        "triangle_count": len(compact_triangles),
        "material": {
            "color": list(args.color),
            "roughness": args.roughness,
            "fuzziness": args.fuzziness,
            "specular": args.specular,
            "shininess": args.shininess,
        },
        "ingest": {
            "source_vertex_count": len(vertices),
            "source_uv_count": len(uvs),
            "source_triangle_count": len(triangles),
            "target_triangles": args.target_triangles,
            "face_step": args.face_step,
            "simplification": simplification,
            "source_up": args.source_up,
            "scale_to_height": args.scale_to_height,
            "yaw": args.yaw,
            "pitch": args.pitch,
            "roll": args.roll,
        },
        "vertices": [list(vertex) for vertex in used_vertices],
        "uvs": [list(uv) for uv in used_uvs],
        "triangles": compact_triangles,
    }
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest = {key: payload[key] for key in ("format", "name", "source", "bounds", "vertex_count", "uv_count", "triangle_count", "material", "ingest")}
    manifest["asset"] = output_path.name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {manifest_path}")


def _read_obj(path: Path) -> tuple[list[Vec3], list[tuple[float, float]], list[tuple[int, int, int, int, int, int]]]:
    vertices: list[Vec3] = []
    uvs: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int, int, int, int]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "vt" and len(parts) >= 3:
            uvs.append((float(parts[1]), 1.0 - float(parts[2])))
        elif parts[0] == "f" and len(parts) >= 4:
            face = [_parse_face_token(token, len(vertices), len(uvs)) for token in parts[1:]]
            for index in range(1, len(face) - 1):
                a, b, c = face[0], face[index], face[index + 1]
                triangles.append((a[0], b[0], c[0], a[1], b[1], c[1]))
    return vertices, uvs, triangles


def _parse_face_token(token: str, vertex_count: int, uv_count: int) -> tuple[int, int]:
    parts = token.split("/")
    vertex_index = _resolve_index(int(parts[0]), vertex_count)
    uv_index = -1
    if len(parts) > 1 and parts[1]:
        uv_index = _resolve_index(int(parts[1]), uv_count)
    return vertex_index, uv_index


def _resolve_index(index: int, count: int) -> int:
    resolved = index - 1 if index > 0 else count + index
    if resolved < 0 or resolved >= count:
        raise ValueError("OBJ index out of range")
    return resolved


def _normalize_vertices(
    vertices: list[Vec3],
    *,
    source_up: str,
    scale_to_height: float,
    yaw: float,
    pitch: float,
    roll: float,
) -> list[Vec3]:
    oriented = [_orient_vertex(vertex, source_up) for vertex in vertices]
    min_x, min_y, min_z, max_x, max_y, max_z = _bounds_tuple(oriented)
    height = max(max_y - min_y, 1e-9)
    scale = scale_to_height / height
    center_x = (min_x + max_x) * 0.5
    center_z = (min_z + max_z) * 0.5
    normalized = [Vec3((v.x - center_x) * scale, (v.y - min_y) * scale, (v.z - center_z) * scale) for v in oriented]
    return [_rotate_vertex(vertex, yaw, pitch, roll) for vertex in normalized]


def _orient_vertex(vertex: Vec3, source_up: str) -> Vec3:
    if source_up == "z":
        return Vec3(vertex.x, vertex.z, -vertex.y)
    return vertex


def _rotate_vertex(vertex: Vec3, yaw: float, pitch: float, roll: float) -> Vec3:
    cy, sy = cos(radians(yaw)), sin(radians(yaw))
    cp, sp = cos(radians(pitch)), sin(radians(pitch))
    cr, sr = cos(radians(roll)), sin(radians(roll))
    x1 = vertex.x * cy + vertex.z * sy
    y1 = vertex.y
    z1 = -vertex.x * sy + vertex.z * cy
    x2 = x1
    y2 = y1 * cp - z1 * sp
    z2 = y1 * sp + z1 * cp
    return Vec3(x2 * cr - y2 * sr, x2 * sr + y2 * cr, z2)


def _select_triangles(
    triangles: list[tuple[int, int, int, int, int, int]],
    face_step: int | None,
) -> list[tuple[int, int, int, int, int, int]]:
    if face_step is None or face_step <= 1:
        return triangles
    return [triangle for index, triangle in enumerate(triangles) if index % face_step == 0]


def _compact_geometry(
    vertices: list[Vec3],
    uvs: list[tuple[float, float]],
    triangles: list[tuple[int, int, int, int, int, int]],
    *,
    grid_divisions: int | None = None,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]], list[list[int]]]:
    vertex_map: dict[int | tuple[int, int, int], int] = {}
    uv_map: dict[int, int] = {}
    compact_vertices: list[tuple[float, float, float]] = []
    compact_uvs: list[tuple[float, float]] = []
    compact_triangles: list[list[int]] = []
    bounds = _bounds_tuple(vertices) if grid_divisions is not None else None
    for triangle in triangles:
        out = []
        for vertex_index in triangle[:3]:
            key: int | tuple[int, int, int] = vertex_index
            vertex = vertices[vertex_index]
            if grid_divisions is not None and bounds is not None:
                key = _quantized_key(vertex, bounds, grid_divisions)
                vertex = _quantized_position(key, bounds, grid_divisions)
            out.append(_mapped_index(key, vertex_map, compact_vertices, vertex))
        a, b, c = (Vec3(*compact_vertices[index]) for index in out[:3])
        if a == b or b == c or a == c or _triangle_area_squared(a, b, c) < 1e-12:
            continue
        for uv_index in triangle[3:]:
            if uv_index < 0:
                out.append(-1)
            else:
                out.append(_mapped_index(uv_index, uv_map, compact_uvs, uvs[uv_index]))
        compact_triangles.append(out)
    return compact_vertices, compact_uvs, compact_triangles


def _mapped_index(key, mapping: dict, output: list, value) -> int:
    existing = mapping.get(key)
    if existing is not None:
        return existing
    mapping[key] = len(output)
    output.append(value.as_tuple() if isinstance(value, Vec3) else value)
    return mapping[key]


def _choose_grid_divisions(
    vertices: list[Vec3],
    triangles: list[tuple[int, int, int, int, int, int]],
    target_triangles: int,
) -> int:
    bounds = _bounds_tuple(vertices)
    low = 2
    high = 8
    while _quantized_triangle_count(vertices, triangles, bounds, high) <= target_triangles and high < 1024:
        low = high
        high *= 2
    best = low
    best_count = _quantized_triangle_count(vertices, triangles, bounds, best)
    while low <= high:
        middle = (low + high) // 2
        count = _quantized_triangle_count(vertices, triangles, bounds, middle)
        if count <= target_triangles:
            if count >= best_count:
                best = middle
                best_count = count
            low = middle + 1
        else:
            high = middle - 1
    return best


def _quantized_triangle_count(
    vertices: list[Vec3],
    triangles: list[tuple[int, int, int, int, int, int]],
    bounds: tuple[float, float, float, float, float, float],
    divisions: int,
) -> int:
    count = 0
    for triangle in triangles:
        keys = [_quantized_key(vertices[index], bounds, divisions) for index in triangle[:3]]
        if len({*keys}) < 3:
            continue
        a, b, c = (_quantized_position(key, bounds, divisions) for key in keys)
        if _triangle_area_squared(a, b, c) >= 1e-12:
            count += 1
    return count


def _quantized_key(point: Vec3, bounds: tuple[float, float, float, float, float, float], divisions: int) -> tuple[int, int, int]:
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    return (
        _axis_key(point.x, min_x, max_x, divisions),
        _axis_key(point.y, min_y, max_y, divisions),
        _axis_key(point.z, min_z, max_z, divisions),
    )


def _axis_key(value: float, minimum: float, maximum: float, divisions: int) -> int:
    size = maximum - minimum
    if size <= 1e-12:
        return 0
    return max(0, min(divisions, round((value - minimum) / size * divisions)))


def _quantized_position(
    key: tuple[int, int, int],
    bounds: tuple[float, float, float, float, float, float],
    divisions: int,
) -> Vec3:
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    return Vec3(
        _axis_position(key[0], min_x, max_x, divisions),
        _axis_position(key[1], min_y, max_y, divisions),
        _axis_position(key[2], min_z, max_z, divisions),
    )


def _axis_position(key: int, minimum: float, maximum: float, divisions: int) -> float:
    if divisions <= 0:
        return minimum
    return minimum + (maximum - minimum) * key / divisions


def _triangle_area_squared(a: Vec3, b: Vec3, c: Vec3) -> float:
    return (b - a).cross(c - a).length_squared() * 0.25


def _bounds_for(vertices: list[tuple[float, float, float]]) -> dict:
    vecs = [Vec3(*vertex) for vertex in vertices]
    min_x, min_y, min_z, max_x, max_y, max_z = _bounds_tuple(vecs)
    return {
        "min": [min_x, min_y, min_z],
        "max": [max_x, max_y, max_z],
        "size": [max_x - min_x, max_y - min_y, max_z - min_z],
    }


def _bounds_tuple(vertices: list[Vec3]) -> tuple[float, float, float, float, float, float]:
    if not vertices:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    min_x = min(vertex.x for vertex in vertices)
    min_y = min(vertex.y for vertex in vertices)
    min_z = min(vertex.z for vertex in vertices)
    max_x = max(vertex.x for vertex in vertices)
    max_y = max(vertex.y for vertex in vertices)
    max_z = max(vertex.z for vertex in vertices)
    return (min_x, min_y, min_z, max_x, max_y, max_z)


if __name__ == "__main__":
    main()
