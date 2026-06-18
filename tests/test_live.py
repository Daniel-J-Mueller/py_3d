from array import array

from py_3d import Material, RenderSettings, Scene, Sphere, Sun
from py_3d.live import LiveFlyCamera, LiveSceneBatchBuilder


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


def test_live_scene_batch_builder_respects_flat_generated_normals():
    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.7, Material(color=(90, 150, 230))))
    settings = RenderSettings(width=64, height=64, smooth_shading=False, sphere_segments=8, sphere_rings=4)
    builder = LiveSceneBatchBuilder()

    triangle_bytes, _line_bytes, triangle_vertices, _line_vertices = builder.build(scene, settings)
    payload = array("f")
    payload.frombytes(triangle_bytes)
    first = payload[: 15 * 3]
    normal_a = tuple(first[3:6])
    normal_b = tuple(first[18:21])
    normal_c = tuple(first[33:36])

    assert triangle_vertices > 0
    assert normal_a == normal_b == normal_c


def test_live_fly_camera_uses_mouse_look_and_vertical_keys():
    camera = LiveFlyCamera.looking_at((0, 1, -4), (0, 1, 0), fov_degrees=70.0, speed=2.0)

    camera.look(100, -50)
    camera.move({"w", "shift"}, 0.5)
    view = camera.camera()

    assert camera.yaw_degrees > 0.0
    assert camera.pitch_degrees > 0.0
    assert camera.position.y > 1.0
    assert view.fov_degrees == 70.0
    assert view.target != view.position
