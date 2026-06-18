from py_3d import Bowl, Capsule, Material, Plane, Sphere, SurfacePerturbation, Triangle, ValueNoise3D, planar_project_triangles


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


def test_value_noise_is_deterministic():
    noise = ValueNoise3D(seed=42)

    first = noise.sample((1.25, 2.5, -0.75))
    second = noise.sample((1.25, 2.5, -0.75))

    assert first == second
    assert 0.0 <= first <= 1.0


def test_surface_perturbation_changes_sphere_geometry_and_keeps_uvs():
    plain = Sphere((0, 0, 0), 1.0, Material())
    bumpy = Sphere((0, 0, 0), 1.0, Material(), SurfacePerturbation(magnitude=0.2, scale=3.0, seed=7))

    plain_distances = sorted(round(vertex.length(), 4) for triangle in plain.to_triangles(segments=12, rings=6) for vertex in (triangle.a, triangle.b, triangle.c))
    bumpy_triangles = bumpy.to_triangles(segments=12, rings=6)
    bumpy_distances = sorted(round(vertex.length(), 4) for triangle in bumpy_triangles for vertex in (triangle.a, triangle.b, triangle.c))

    assert plain_distances != bumpy_distances
    assert min(bumpy_distances) >= 0.8
    assert max(bumpy_distances) <= 1.2
    assert all(triangle.has_texture_coordinates() for triangle in bumpy_triangles)


def test_sphere_rotation_drives_render_geometry_without_changing_uvs():
    plain = Sphere((0, 0, 0), 1.0, Material())
    rotated = Sphere((0, 0, 0), 1.0, Material(), rotation=(0.0, 0.7, 0.25))

    plain_triangles = plain.to_triangles(segments=12, rings=6)
    rotated_triangles = rotated.to_triangles(segments=12, rings=6)

    assert plain_triangles[8].a != rotated_triangles[8].a
    assert plain_triangles[8].uv_a == rotated_triangles[8].uv_a


def test_surface_perturbation_changes_bowl_geometry():
    plain = Bowl((0, 0, 0), 1.0, Material(), depth=0.85)
    bumpy = Bowl((0, 0, 0), 1.0, Material(), depth=0.85, perturbation=SurfacePerturbation(magnitude=0.08, scale=5.0, seed=13))

    plain_distances = sorted(round(vertex.length(), 4) for triangle in plain.to_triangles(segments=12, rings=5) for vertex in (triangle.a, triangle.b, triangle.c))
    bumpy_distances = sorted(round(vertex.length(), 4) for triangle in bumpy.to_triangles(segments=12, rings=5) for vertex in (triangle.a, triangle.b, triangle.c))

    assert plain_distances != bumpy_distances


def test_bowl_thickness_adds_outer_shell_and_rim_triangles():
    thin = Bowl((0, 0, 0), 1.0, depth=0.85)
    thick = Bowl((0, 0, 0), 1.0, depth=0.85, thickness=0.08)

    assert len(thick.to_triangles(segments=12, rings=4)) > len(thin.to_triangles(segments=12, rings=4))


def test_capsule_generates_renderable_triangles():
    capsule = Capsule((0, 0, 0), radius=0.25, height=1.4)
    triangles = capsule.to_triangles(segments=10, rings=6)

    assert len(triangles) > 0
    assert all(triangle.has_vertex_normals() for triangle in triangles)


def test_sized_plane_thickness_creates_sealed_slab():
    flat = Plane((0, 0, 0), (0, 1, 0), size=1.0)
    slab = Plane((0, 0, 0), (0, 1, 0), size=1.0, thickness=0.1)

    assert len(flat.to_triangles()) == 2
    assert len(slab.to_triangles()) == 12
