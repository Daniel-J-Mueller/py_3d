from py_3d import (
    Camera,
    Color,
    CubePlacer,
    Inventory,
    Lamp,
    Material,
    PortalPair,
    PortalSurface,
    RenderEngine,
    RenderSettings,
    Scene,
    Sun,
    Vec3,
)
from py_3d.portal import portal_camera_for


def test_portal_surface_generates_textured_quad_and_frame():
    portal = PortalSurface("blue", center=(0, 1, 0), normal=(0, 0, -1), width=1.0, height=1.4)

    triangles = portal.to_triangles()

    assert len(triangles) == 10
    assert triangles[0].has_texture_coordinates()
    assert portal.contains_point((0, 1, 0))
    assert not portal.contains_point((2, 1, 0))


def test_portal_camera_maps_through_linked_surface():
    entry = PortalSurface("entry", center=(0, 1, 1), normal=(0, 0, -1))
    exit = PortalSurface("exit", center=(3, 1, 0), normal=(-1, 0, 0))
    camera = Camera(position=(0, 1, -2), target=(0, 1, 2))

    mapped = portal_camera_for(camera, entry, exit)

    assert mapped.position.x < 3.0
    assert mapped.target.x < mapped.position.x


def test_render_engine_fills_portal_texture_before_final_pass():
    entry = PortalSurface("entry", center=(-0.7, 0.8, 0), normal=(0, 0, -1), width=0.8, height=1.0)
    exit = PortalSurface("exit", center=(0.9, 0.8, 0.8), normal=(-1, 0, 0), width=0.8, height=1.0)
    scene = Scene(background=(0, 0, 0))
    scene.add_portal_pair(PortalPair(entry, exit, texture_width=48, texture_height=32))
    scene.add_light(Sun(direction=(0, 0, -1), intensity=0.8))
    scene.add_light(Lamp(position=(0, 2, -2), intensity=2.0))
    scene.add(
        # A bright marker visible from the exit portal camera.
        PortalSurface("marker", center=(1.4, 0.8, 0.8), normal=(-1, 0, 0), width=0.5, height=0.5, material=Material(color=(255, 255, 255), emission=(255, 255, 255)), frame_width=0.0)
    )
    camera = Camera(position=(-0.7, 0.8, -2.2), target=(-0.7, 0.8, 0.2))

    buffer = RenderEngine().render(scene, camera, RenderSettings(width=80, height=60, ambient=0.45))

    assert any(pixel != Color(0, 0, 0) for pixel in buffer.pixels)


def test_inventory_places_cube_and_consumes_stack():
    inventory = Inventory()
    inventory.add("cube", 2)
    scene = Scene()
    camera = Camera.first_person((0, 0, 0), (0, 0, 1), eye_height=1.0)
    placer = CubePlacer(inventory, cube_size=0.5, placement_distance=1.0, snap=None)

    cube = placer.place(camera, scene)

    assert cube is not None
    assert inventory.count("cube") == 1
    assert scene.objects[-1] is cube
    assert isinstance(cube.center, Vec3)
    assert cube.center.z > 0.5
