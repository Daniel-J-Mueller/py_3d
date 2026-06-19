import argparse
import sys
from pathlib import Path
import importlib.util


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


def test_rgb_bulb_light_sequence():
    from fruit_bowl_demo import FruitBowlSimulation

    simulation = FruitBowlSimulation()

    simulation.time = 0.0
    lights = simulation._lights_for_mode("rgb-bulbs")
    assert [light.intensity for light in lights[1:]] == [3.8, 0.0, 0.0]

    simulation.time = 2.0
    lights = simulation._lights_for_mode("rgb-bulbs")
    assert [light.intensity for light in lights[1:]] == [0.0, 3.8, 0.0]

    simulation.time = 14.0
    lights = simulation._lights_for_mode("rgb-bulbs")
    assert [light.intensity for light in lights[1:]] == [0.0, 0.0, 0.0]


def test_high_wind_cloth_respects_bowl_collision():
    from fan_cloth_water_demo import FanWaterSimulation, Vec3

    simulation = FanWaterSimulation(quality="fast")
    simulation.wind_scale = 4.0
    for _ in range(24):
        simulation.step(1.0 / 60.0)

    center = Vec3(simulation.water_center.x, simulation.bowl_center_y, simulation.water_center.z)
    radius = 0.78 + 0.034
    for node in simulation.cloth.nodes:
        if node.position.y <= center.y + 0.78 * 0.68:
            assert node.position.distance_to(center) >= radius


def test_cloth_table_collision_uses_rendered_table_plane():
    from fan_cloth_water_demo import ClothSheet, Vec3

    cloth = ClothSheet(columns=4, rows=4)
    node = next(node for node in cloth.nodes if not node.pinned)
    node.position = Vec3(node.position.x, -0.25, node.position.z)
    node.velocity = Vec3(0.0, -1.0, 0.0)

    cloth.step(1.0 / 60.0, 0.0, substeps=1, wind_scale=0.0, floor_y=0.0, floor_clearance=0.018)

    assert cloth.links
    assert abs(node.position.y - 0.018) < 1e-9


def test_wind_pool_water_moves_surface_particles():
    from wind_pool_water_demo import WindPoolWaterSimulation

    simulation = WindPoolWaterSimulation(quality="fast")
    before = [particle.position for particle in simulation.fluid.particles]

    for _ in range(6):
        simulation.step(1.0 / 30.0)

    assert any(particle.position.distance_to(start) > 0.005 for particle, start in zip(simulation.fluid.particles, before))


def test_water_surface_stays_flat_near_bowl_wall():
    from fan_cloth_water_demo import FanWaterSimulation, water_surface_mesh

    simulation = FanWaterSimulation(quality="fast")
    mesh = water_surface_mesh(simulation.fluid, simulation.water_center, simulation.water_radius, simulation.time, quality="fast")
    unique_vertices = {vertex for triangle in mesh.triangles for vertex in (triangle.a, triangle.b, triangle.c)}
    inner = []
    edge = []
    for vertex in unique_vertices:
        radial = ((vertex.x - simulation.water_center.x) ** 2 + (vertex.z - simulation.water_center.z) ** 2) ** 0.5
        if radial < simulation.water_radius * 0.18:
            inner.append(vertex.y)
        elif radial > simulation.water_radius * 0.9:
            edge.append(vertex.y)

    assert inner
    assert edge
    assert sum(edge) / len(edge) >= sum(inner) / len(inner) - 0.025


def test_fan_water_defaults_to_ultra_and_uses_small_table_contact_skin(monkeypatch):
    from fan_cloth_water_demo import FanWaterSimulation, parse_args

    monkeypatch.setattr(sys, "argv", ["fan_cloth_water_demo.py"])
    args = parse_args()
    simulation = FanWaterSimulation(quality="fast")

    assert args.quality == "ultra"
    assert 0.0 < simulation.floor_contact_margin < simulation.fluid.rest_distance * 0.25


def test_fan_water_bowl_visual_rests_on_table_contact_plane():
    from fan_cloth_water_demo import FanWaterSimulation

    simulation = FanWaterSimulation(quality="fast")
    bowl = next(obj for obj in simulation.scene().objects if type(obj).__name__ == "Bowl")
    min_y = min(vertex.y for triangle in bowl.to_triangles(segments=18, rings=9) for vertex in (triangle.a, triangle.b, triangle.c))

    assert abs(min_y - (simulation.table_surface_y + simulation.bowl_table_clearance)) < 1e-6


def test_fan_water_settings_honor_smooth_shading_option():
    from fan_cloth_water_demo import make_settings

    args = argparse.Namespace(
        width=64,
        height=48,
        ambient=0.04,
        gamma=1.12,
        smooth_shading=False,
        reflection_bounces=2,
        sphere_segments=18,
        sphere_rings=9,
        texture_size=256,
    )

    assert make_settings(args).smooth_shading is False
    args.smooth_shading = True
    assert make_settings(args).smooth_shading is True


def test_showcase_default_profile_launches_fan_water_in_ultra():
    launcher_path = Path(__file__).resolve().parents[1] / "USER" / "demos" / "00_list_experiences.py"
    spec = importlib.util.spec_from_file_location("py3d_launcher_menu_ultra", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)

    fan_water = next(experience for experience in launcher.EXPERIENCES if experience.title == "Fan Cloth Water")
    command = launcher.command_for(fan_water, launcher.MenuSettings())

    assert command[command.index("--quality") + 1] == "ultra"


def test_procedural_showcase_launch_is_live_and_preview_is_still():
    launcher_path = Path(__file__).resolve().parents[1] / "USER" / "demos" / "00_list_experiences.py"
    spec = importlib.util.spec_from_file_location("py3d_launcher_procedural_mode", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)

    procedural = next(experience for experience in launcher.EXPERIENCES if experience.title == "Procedural Hill Biome")

    assert "--still" not in procedural.launch_command
    assert procedural.preview_command is not None
    assert "--still" in procedural.preview_command


def test_broken_fan_vessel_expands_fluid_bounds_and_removes_bowl():
    from fan_cloth_water_demo import FanWaterSimulation

    simulation = FanWaterSimulation(quality="fast")
    simulation.step(1.0 / 60.0)
    intact_floor = simulation.fluid.bounds_min.y

    simulation.vessel_intact = False
    start_average_y = sum(particle.position.y for particle in simulation.fluid.particles) / len(simulation.fluid.particles)
    for _ in range(20):
        simulation.step(1.0 / 60.0)
    scene = simulation.scene()
    fallen_average_y = sum(particle.position.y for particle in simulation.fluid.particles) / len(simulation.fluid.particles)

    assert simulation.fluid.bounds_min.y < intact_floor
    assert simulation.fluid.bounds_min.y > simulation.table_surface_y
    assert simulation.fluid.bounds_max.x - simulation.fluid.bounds_min.x > simulation.water_radius * 2.5
    assert fallen_average_y < start_average_y - 0.05
    assert not any(type(obj).__name__ == "Bowl" for obj in scene.objects)
    assert not any(type(obj).__name__ == "ParticleWaterSurface" for obj in scene.objects)
    spilled = simulation.spilled_water_primitives()
    assert len(spilled) <= 1
    assert all(type(obj).__name__ == "Mesh" for obj in spilled)


def test_submerged_fan_generates_water_bubble_surface_events():
    from fan_cloth_water_demo import FanWaterSimulation

    simulation = FanWaterSimulation(quality="fast")
    simulation.blade_strength = 2.0

    assert simulation.fan_center.y < simulation.water_center.y

    for _ in range(24):
        simulation.step(1.0 / 30.0)

    events = simulation.bubble_surface_events()
    assert events
    assert all(event[2] > 0.0 for event in events)
    assert simulation.bubble_primitives() == ()

    saw_splash = False
    for _ in range(48):
        simulation.step(1.0 / 30.0)
        saw_splash = saw_splash or bool(simulation.splash_droplets)

    assert saw_splash
    assert all(droplet.radius >= simulation.minimum_droplet_radius for droplet in simulation.splash_droplets)


def test_launcher_closes_window_after_live_launch(monkeypatch):
    launcher_path = Path(__file__).resolve().parents[1] / "USER" / "demos" / "00_list_experiences.py"
    spec = importlib.util.spec_from_file_location("py3d_launcher_menu", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)

    class FakeWindow:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        def poll(self):
            return None

    monkeypatch.setattr(launcher, "PixelWindow", FakeWindow)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    menu = launcher.NativeShowcaseMenu()

    menu._start_child((sys.executable, "-c", "print('demo')"), "Launched demo.")

    assert menu.window.closed is True


def test_launcher_keeps_window_open_for_preview_process(monkeypatch):
    launcher_path = Path(__file__).resolve().parents[1] / "USER" / "demos" / "00_list_experiences.py"
    spec = importlib.util.spec_from_file_location("py3d_launcher_menu_preview", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)

    class FakeWindow:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        def poll(self):
            return None

    monkeypatch.setattr(launcher, "PixelWindow", FakeWindow)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    menu = launcher.NativeShowcaseMenu()

    menu._start_child((sys.executable, "-c", "print('preview')"), "Rendering preview.", preview=True)

    assert menu.window.closed is False
