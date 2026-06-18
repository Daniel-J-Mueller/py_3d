from py_3d import Material, Sphere, Triangle, planar_project_triangles


def test_sphere_generates_texture_coordinates():
    sphere = Sphere((0, 0, 0), 1.0, Material())

    triangles = sphere.to_triangles(segments=8, rings=4)

    assert triangles
    assert all(triangle.has_texture_coordinates() for triangle in triangles)
    assert all(0.0 <= uv <= 1.0 for triangle in triangles for coords in (triangle.uv_a, triangle.uv_b, triangle.uv_c) for uv in coords)


def test_planar_project_triangles_assigns_uvs_from_center_and_axes():
    triangle = Triangle((-1, -1, 0), (1, -1, 0), (0, 1, 0), Material())

    (projected,) = planar_project_triangles(
        [triangle],
        center=(0, 0, 0),
        u_axis=(1, 0, 0),
        v_axis=(0, 1, 0),
        scale=(2, 2),
        offset=(0.5, 0.5),
    )

    assert projected.uv_a == (0.0, 0.0)
    assert projected.uv_b == (1.0, 0.0)
    assert projected.uv_c == (0.5, 1.0)


def test_material_surface_attributes_are_visual_not_physics():
    material = Material(roughness=2.0, fuzziness=-1.0)

    assert material.roughness == 1.0
    assert material.fuzziness == 0.0
    assert not hasattr(material, "friction")
    assert not hasattr(material, "restitution")
