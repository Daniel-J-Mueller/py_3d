from py_3d import Material, RenderSettings, Scene, Sphere, Sun
from py_3d.live import LiveSceneBatchBuilder


def test_live_scene_batch_builder_outputs_triangle_payload():
    scene = Scene()
    sphere = Sphere((0, 0, 0), 0.7, Material(color=(90, 150, 230), specular=0.2))
    scene.add(sphere)
    scene.add_light(Sun(direction=(-0.4, -0.7, -1.0), intensity=1.0))
    settings = RenderSettings(width=64, height=64, sphere_segments=10, sphere_rings=5)
    builder = LiveSceneBatchBuilder()

    triangle_bytes, line_bytes, triangle_vertices, line_vertices = builder.build(scene, settings)

    assert triangle_vertices == len(sphere.to_triangles(segments=10, rings=5)) * 3
    assert len(triangle_bytes) == triangle_vertices * 15 * 4
    assert line_bytes == b""
    assert line_vertices == 0


def test_live_scene_batch_builder_outputs_wireframe_lines():
    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.7, Material(color=(90, 150, 230))))
    settings = RenderSettings(width=64, height=64, wireframe=True, sphere_segments=8, sphere_rings=4)
    builder = LiveSceneBatchBuilder()

    triangle_bytes, line_bytes, triangle_vertices, line_vertices = builder.build(scene, settings)

    assert triangle_bytes == b""
    assert triangle_vertices == 0
    assert len(line_bytes) == line_vertices * 15 * 4
    assert line_vertices > 0
