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

    center = Vec3(simulation.water_center.x, 0.78, simulation.water_center.z)
    radius = 0.78 + 0.034
    for node in simulation.cloth.nodes:
        if node.position.y <= center.y + 0.78 * 0.68:
            assert node.position.distance_to(center) >= radius


def test_wind_pool_water_moves_surface_particles():
    from wind_pool_water_demo import WindPoolWaterSimulation

    simulation = WindPoolWaterSimulation(quality="fast")
    before = [particle.position for particle in simulation.fluid.particles]

    for _ in range(6):
        simulation.step(1.0 / 30.0)

    assert any(particle.position.distance_to(start) > 0.005 for particle, start in zip(simulation.fluid.particles, before))


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
