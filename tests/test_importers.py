from pathlib import Path

from py_3d import Material, load_obj, load_stl


def test_load_obj_triangulates_quad_and_keeps_uvs(tmp_path: Path):
    path = tmp_path / "quad.obj"
    path.write_text(
        "\n".join(
            [
                "v -1 -1 0",
                "v 1 -1 0",
                "v 1 1 0",
                "v -1 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 1",
                "f 1/1 2/2 3/3 4/4",
            ]
        ),
        encoding="utf-8",
    )

    mesh = load_obj(path, Material(color=(255, 0, 0)))

    assert len(mesh.triangles) == 2
    assert mesh.triangles[0].uv_a == (0.0, 1.0)
    assert mesh.triangles[0].uv_b == (1.0, 1.0)
    assert mesh.triangles[0].uv_c == (1.0, 0.0)


def test_load_ascii_stl(tmp_path: Path):
    path = tmp_path / "triangle.stl"
    path.write_text(
        """
solid one
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid one
""".strip(),
        encoding="utf-8",
    )

    mesh = load_stl(path)

    assert len(mesh.triangles) == 1
    assert mesh.triangles[0].a.as_tuple() == (0.0, 0.0, 0.0)
