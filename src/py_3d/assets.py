"""Load py_3d-ready mesh assets produced by ingestion scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .materials import Material
from .math3d import Vec3
from .primitives import Mesh, Triangle


MESH_ASSET_FORMAT = "py_3d.mesh_asset.v1"


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
