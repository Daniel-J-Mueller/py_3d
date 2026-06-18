import sys
from pathlib import Path


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
