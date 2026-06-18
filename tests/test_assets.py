import json
import importlib.util
from pathlib import Path

from py_3d import MESH_ASSET_FORMAT, Color, Material, Vec3, load_mesh_asset, mesh_asset_metadata


def _load_ingest_asset_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "ingest_asset.py"
    spec = importlib.util.spec_from_file_location("ingest_asset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ingest_asset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_mesh_asset_builds_triangles_with_uvs(tmp_path: Path):
    path = tmp_path / "triangle.py3dmesh.json"
    path.write_text(
        json.dumps(
            {
                "format": MESH_ASSET_FORMAT,
                "name": "triangle",
                "source": "generated",
                "bounds": {"min": [0, 0, 0], "max": [1, 1, 0], "size": [1, 1, 0]},
                "vertex_count": 3,
                "uv_count": 3,
                "triangle_count": 1,
                "material": {"color": [10, 20, 30]},
                "ingest": {},
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "uvs": [[0, 0], [1, 0], [0, 1]],
                "triangles": [[0, 1, 2, 0, 1, 2]],
            }
        ),
        encoding="utf-8",
    )

    mesh = load_mesh_asset(path)
    metadata = mesh_asset_metadata(path)

    assert metadata["triangle_count"] == 1
    assert len(mesh.triangles) == 1
    assert mesh.triangles[0].uv_b == (1, 0)
    assert mesh.triangles[0].material.color == Color(10, 20, 30)


def test_load_mesh_asset_accepts_material_override(tmp_path: Path):
    path = tmp_path / "triangle.py3dmesh.json"
    path.write_text(
        json.dumps(
            {
                "format": MESH_ASSET_FORMAT,
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "uvs": [],
                "triangles": [[0, 1, 2, -1, -1, -1]],
            }
        ),
        encoding="utf-8",
    )

    mesh = load_mesh_asset(path, Material(color=(200, 80, 40)))

    assert mesh.triangles[0].material.color == Color(200, 80, 40)


def test_ingest_grid_simplification_preserves_surface_coverage():
    ingest_asset = _load_ingest_asset_module()
    vertices = [Vec3(x / 6, y / 6, 0.0) for y in range(7) for x in range(7)]
    triangles = []
    for y in range(6):
        for x in range(6):
            a = y * 7 + x
            b = a + 1
            c = a + 7
            d = c + 1
            triangles.append((a, b, c, -1, -1, -1))
            triangles.append((b, d, c, -1, -1, -1))

    divisions = ingest_asset._choose_grid_divisions(vertices, triangles, 18)
    compact_vertices, _compact_uvs, compact_triangles = ingest_asset._compact_geometry(
        vertices,
        [],
        triangles,
        grid_divisions=divisions,
    )

    xs = [vertex[0] for vertex in compact_vertices]
    ys = [vertex[1] for vertex in compact_vertices]
    assert len(compact_triangles) <= 18
    assert min(xs) == 0.0
    assert max(xs) == 1.0
    assert min(ys) == 0.0
    assert max(ys) == 1.0
