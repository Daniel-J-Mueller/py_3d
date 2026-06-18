import pytest

from py_3d import Bowl, Box, CPURenderer, Camera, Color, GPURenderer, Lamp, Material, PixelBuffer, Plane, RenderEngine, RenderSettings, Scene, Sphere, Sun, TextBulletin, Triangle, build_gpu_scene_batch


def test_cpu_renderer_renders_triangle_offscreen():
    scene = Scene()
    scene.add(
        Triangle(
            (-1, -1, 0),
            (1, -1, 0),
            (0, 1, 0),
            Material(color=(200, 80, 40)),
        )
    )
    scene.add_light(Sun(direction=(0, 0, -1), intensity=1.0))
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))

    buffer = RenderEngine().render(scene, camera, RenderSettings(width=64, height=64))

    assert any(pixel != Color(0, 0, 0) for pixel in buffer.pixels)


def test_cpu_renderer_depth_keeps_nearer_triangle():
    red = Material(color=(255, 0, 0), emission=(255, 0, 0))
    blue = Material(color=(0, 0, 255), emission=(0, 0, 255))
    scene = Scene()
    scene.add(
        Triangle((-1, -1, 1), (1, -1, 1), (0, 1, 1), blue),
        Triangle((-1, -1, 0), (1, -1, 0), (0, 1, 0), red),
    )
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))

    buffer = RenderEngine().render(scene, camera, RenderSettings(width=65, height=65, ambient=0.0))

    center = buffer.get_pixel(32, 32)
    assert center.r > center.b


def test_lamp_contributes_colored_light():
    material = Material(color=(255, 255, 255))
    scene = Scene()
    scene.add(Triangle((-1, -1, 0), (1, -1, 0), (0, 1, 0), material))
    scene.add_light(Lamp(position=(0, 0, -2), color=(0, 0, 255), intensity=10.0))
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))

    buffer = RenderEngine().render(scene, camera, RenderSettings(width=65, height=65, ambient=0.0))

    center = buffer.get_pixel(32, 32)
    assert center.b > center.r


def test_high_poly_sphere_and_other_primitives_render():
    sphere = Sphere((0, 0, 0), 0.8, Material(color=(90, 150, 230)))
    scene = Scene()
    scene.add(
        sphere,
        Box((1.25, -0.2, 0.2), (0.7, 0.7, 0.7), Material(color=(220, 150, 70))),
        Plane((0, -1.0, 0), (0, 1, 0), Material(color=(70, 130, 90)), size=4.0),
    )
    scene.add_light(Sun(direction=(-0.4, -0.7, -1), intensity=0.9))
    camera = Camera(position=(0.4, 0.4, -4.2), target=(0, 0, 0))

    buffer = RenderEngine().render(
        scene,
        camera,
        RenderSettings(width=120, height=90, sphere_segments=32, sphere_rings=16),
    )

    assert len(sphere.to_triangles(segments=32, rings=16)) == 960
    assert sum(pixel != Color(0, 0, 0) for pixel in buffer.pixels) > 100


def test_sphere_triangles_include_vertex_normals_for_smooth_shading():
    triangles = Sphere((0, 0, 0), 1.0).to_triangles(segments=12, rings=6)

    assert all(triangle.has_vertex_normals() for triangle in triangles)


def test_smooth_shading_changes_lit_sphere_pixels():
    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.85, Material(color=(120, 180, 230))))
    scene.add_light(Lamp(position=(0.8, 1.6, -2.2), color=(255, 245, 230), intensity=4.0))
    camera = Camera(position=(0.0, 0.0, -4.0), target=(0, 0, 0))
    renderer = RenderEngine(CPURenderer(cache_static_geometry=False))

    flat = renderer.render(scene, camera, RenderSettings(width=72, height=72, ambient=0.03, sphere_segments=14, sphere_rings=7))
    smooth = renderer.render(
        scene,
        camera,
        RenderSettings(width=72, height=72, ambient=0.03, smooth_shading=True, sphere_segments=14, sphere_rings=7),
    )

    assert flat.pixels != smooth.pixels
    assert sum(pixel != Color(0, 0, 0) for pixel in smooth.pixels) > 100


def test_bowl_primitive_renders_through_engine():
    scene = Scene()
    scene.add(Bowl((0, 0, 0), 1.0, Material(color=(150, 90, 210)), depth=0.9))
    scene.add_light(Sun(direction=(-0.3, -0.8, -1.0), intensity=0.9))
    camera = Camera(position=(0, 1.6, -3.4), target=(0, -0.25, 0), fov_degrees=50)

    buffer = RenderEngine().render(scene, camera, RenderSettings(width=96, height=72, sphere_segments=18, sphere_rings=9))

    assert sum(pixel != Color(0, 0, 0) for pixel in buffer.pixels) > 100


def test_text_bulletin_renders_over_scene():
    scene = Scene()
    scene.add_bulletin(TextBulletin("TEST", position=(0, 0), color=(255, 255, 255), background=(10, 20, 30)))
    camera = Camera()

    buffer = RenderEngine().render(scene, camera, RenderSettings(width=80, height=24, background=(0, 0, 0)))

    assert buffer.get_pixel(0, 0) == Color(10, 20, 30)
    assert any(pixel == Color(255, 255, 255) for pixel in buffer.pixels)


def test_cached_and_uncached_cpu_renderers_match_basic_scene():
    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.7, Material(color=(90, 150, 230))))
    scene.add_light(Sun(direction=(0, 0, -1), intensity=1.0))
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))
    settings = RenderSettings(width=64, height=64, sphere_segments=12, sphere_rings=6)

    cached = RenderEngine(CPURenderer(cache_static_geometry=True)).render(scene, camera, settings)
    uncached = RenderEngine(CPURenderer(cache_static_geometry=False)).render(scene, camera, settings)

    assert cached.pixels == uncached.pixels


def test_gpu_renderer_scaffold_can_fall_back_to_cpu():
    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.7, Material(color=(90, 150, 230))))
    scene.add_light(Sun(direction=(0, 0, -1), intensity=1.0))
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))
    settings = RenderSettings(width=48, height=48, sphere_segments=10, sphere_rings=5)

    cpu = RenderEngine(CPURenderer()).render(scene, camera, settings)
    gpu_scaffold = RenderEngine(GPURenderer()).render(scene, camera, settings)

    assert gpu_scaffold.pixels == cpu.pixels


def test_gpu_renderer_scaffold_strict_mode_is_explicit():
    scene = Scene()
    camera = Camera()

    with pytest.raises(RuntimeError, match="GPU renderer scaffold"):
        RenderEngine(GPURenderer(allow_cpu_fallback=False)).render(scene, camera, RenderSettings(width=8, height=8))


def test_gpu_scene_batch_flattens_triangle_geometry():
    scene = Scene()
    scene.add(Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0), Material(color=(10, 20, 30))))

    batch = build_gpu_scene_batch(scene, RenderSettings(width=8, height=8))

    assert batch.positions == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert batch.indices == ((0, 1, 2),)
    assert all(color == Color(10, 20, 30) for color in batch.colors)


def test_textured_triangles_render_from_png_asset():
    texture = PixelBuffer.from_png("assets/tv-test.png")
    material = Material(texture=texture)
    scene = Scene()
    scene.add(
        Triangle((-1.6, -0.9, 0), (1.6, -0.9, 0), (1.6, 0.9, 0), material, (0, 1), (1, 1), (1, 0)),
        Triangle((-1.6, -0.9, 0), (1.6, 0.9, 0), (-1.6, 0.9, 0), material, (0, 1), (1, 0), (0, 0)),
    )
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))

    buffer = RenderEngine().render(scene, camera, RenderSettings(width=160, height=90, ambient=1.0))
    colors = {pixel for pixel in buffer.pixels if pixel != Color(0, 0, 0)}

    assert len(colors) > 8


def test_textured_sphere_renders_from_png_asset():
    texture = PixelBuffer.from_png("assets/tv-test.png")
    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.8, Material(texture=texture)))
    camera = Camera(position=(0, 0, -4), target=(0, 0, 0))

    buffer = RenderEngine().render(
        scene,
        camera,
        RenderSettings(width=96, height=72, ambient=1.0, sphere_segments=16, sphere_rings=8),
    )
    colors = {pixel for pixel in buffer.pixels if pixel != Color(0, 0, 0)}

    assert len(colors) > 8
