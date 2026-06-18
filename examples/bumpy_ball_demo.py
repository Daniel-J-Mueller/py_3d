"""Physics demo with a noisy, visually perturbed rolling sphere."""

from pathlib import Path

from py_3d import (
    Camera,
    Color,
    Lamp,
    Line3,
    Material,
    PhysicsWorld,
    PixelBuffer,
    RenderEngine,
    RenderSettings,
    SphereBody,
    StaticBox,
    StaticPlane,
    Sun,
    SurfacePerturbation,
    Scene,
    TextBulletin,
)


def main() -> None:
    texture = PixelBuffer.from_png(Path("assets") / "tv-test.png")
    ball_material = Material(
        texture=texture,
        absorption=(0.04, 0.04, 0.04),
        roughness=0.35,
        fuzziness=0.15,
    )
    perturbation = SurfacePerturbation(magnitude=0.09, scale=3.4, seed=12, octaves=4, gain=0.55)
    ball = SphereBody(
        position=(-2.15, 1.3, 0.0),
        radius=0.34,
        velocity=(0.0, 0.0, 0.0),
        restitution=0.45,
        friction=0.05,
        material=ball_material,
        visual_perturbation=perturbation,
    )
    ramp = StaticPlane(
        point=(0.0, 0.0, 0.0),
        normal=(0.34, 1.0, 0.0),
        friction=0.06,
        restitution=0.18,
        material=Material(color=(75, 145, 95), roughness=0.45, fuzziness=0.2),
        size=5.8,
    )
    wall = StaticBox(
        center=(2.15, 0.5, 0.0),
        size=(0.28, 1.6, 2.1),
        restitution=0.65,
        friction=0.04,
        material=Material(color=(170, 175, 190), roughness=0.55),
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
    scene.add_light(Sun(direction=(-0.4, -0.8, -1.0), color=(255, 245, 230), intensity=0.95))
    scene.add_light(Lamp(position=(-1.8, 2.0, -2.0), color=(90, 140, 255), intensity=2.5))
    scene.add_light(Lamp(position=(1.9, 1.2, -1.1), color=(255, 120, 90), intensity=1.6))
    scene.add_bulletin(
        TextBulletin(
            "BUMPY BALL PHYSICS\nVISUAL NOISE + SPHERE COLLISION",
            position=(10, 10),
            color=(245, 248, 255),
            background=(5, 7, 11),
            padding=5,
            scale=1,
        )
    )

    camera = Camera(position=(0.25, 2.15, -5.5), target=(0.2, 0.45, 0.0), fov_degrees=48)
    settings = RenderSettings(
        width=640,
        height=400,
        background=Color(8, 10, 14),
        ambient=0.1,
        sphere_segments=36,
        sphere_rings=18,
    )
    output_dir = Path("renderings-tests")
    output_dir.mkdir(exist_ok=True)
    path_out = output_dir / "bumpy_ball_physics.png"
    RenderEngine().render(scene, camera, settings).to_png(path_out)
    print(f"Wrote {path_out}")


if __name__ == "__main__":
    main()
