"""Basic physics interaction example.

Simulates a sphere sliding down a tilted plane and colliding with a wall, then
renders the final state and path to ``renderings-tests/physics_interaction.png``.
"""

from pathlib import Path

from py_3d import (
    Camera,
    Color,
    Lamp,
    Line3,
    Material,
    PhysicsWorld,
    RenderEngine,
    RenderSettings,
    Scene,
    SphereBody,
    StaticBox,
    StaticPlane,
    Sun,
    TextBulletin,
)


def main() -> None:
    ball = SphereBody(
        position=(-2.2, 1.25, 0.0),
        radius=0.32,
        velocity=(0.0, 0.0, 0.0),
        restitution=0.55,
        friction=0.03,
        material=Material(color=(235, 85, 65), absorption=(0.05, 0.1, 0.12)),
    )
    ramp = StaticPlane(
        point=(0.0, 0.0, 0.0),
        normal=(0.35, 1.0, 0.0),
        friction=0.02,
        restitution=0.2,
        material=Material(color=(80, 150, 95), absorption=(0.2, 0.1, 0.25)),
        size=5.8,
    )
    wall = StaticBox(
        center=(2.15, 0.5, 0.0),
        size=(0.28, 1.6, 2.1),
        restitution=0.65,
        material=Material(color=(170, 175, 190), absorption=(0.12, 0.12, 0.1)),
    )

    world = PhysicsWorld(gravity=(0.0, -9.81, 0.0))
    world.add_sphere(ball)
    world.add_plane(ramp)
    world.add_box(wall)

    path = [ball.position]
    for step in range(300):
        world.step(1.0 / 60.0, substeps=2)
        if step % 6 == 0:
            path.append(ball.position)

    scene = Scene()
    scene.add(ramp.to_primitive(), wall.to_primitive(), ball.to_primitive())
    for start, end in zip(path, path[1:]):
        scene.add(Line3(start, end, Material(color=(245, 220, 90), emission=(80, 60, 0))))
    scene.add_light(Sun(direction=(-0.4, -0.8, -1.0), color=(255, 245, 230), intensity=0.9))
    scene.add_light(Lamp(position=(-1.8, 2.0, -2.0), color=(90, 140, 255), intensity=2.4))
    scene.add_bulletin(
        TextBulletin(
            f"PHYSICS TEST\nX {ball.position.x:.2f}  Y {ball.position.y:.2f}",
            position=(10, 10),
            color=(245, 248, 255),
            background=(5, 7, 11),
            padding=5,
            scale=1,
        )
    )

    camera = Camera(position=(0.25, 2.2, -5.5), target=(0.2, 0.45, 0.0), fov_degrees=48)
    settings = RenderSettings(width=480, height=300, background=Color(8, 10, 14), ambient=0.08)
    output_dir = Path("renderings-tests")
    output_dir.mkdir(exist_ok=True)
    path_out = output_dir / "physics_interaction.png"
    RenderEngine().render(scene, camera, settings).to_png(path_out)
    print(f"Wrote {path_out}")
    print(f"Final ball position: {ball.position.as_tuple()}")
    print(f"Final ball velocity: {ball.velocity.as_tuple()}")


if __name__ == "__main__":
    main()
