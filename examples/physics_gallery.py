"""Generate several simple physics demo renderings."""

from __future__ import annotations

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
    SphereCollider,
    StaticBox,
    StaticPlane,
    Sun,
    SurfacePerturbation,
    TextBulletin,
)


OUTPUT_DIR = Path("renderings-tests")


def add_path(scene: Scene, path, color=(245, 220, 90)) -> None:
    for start, end in zip(path, path[1:]):
        scene.add(Line3(start, end, Material(color=color, emission=(70, 55, 0))))


def save_ramp_wall() -> None:
    ball = SphereBody(
        position=(-2.2, 1.25, 0.0),
        radius=0.32,
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

    scene = base_scene("RAMP WALL\nSPHERE CONTACT")
    scene.add(ramp.to_primitive(), wall.to_primitive(), ball.to_primitive())
    add_path(scene, path)
    render(scene, "physics_ramp_wall.png", Camera(position=(0.25, 2.2, -5.5), target=(0.2, 0.45, 0.0), fov_degrees=48))


def save_floor_bounce() -> None:
    ball = SphereBody(
        position=(-0.35, 2.4, 0.0),
        radius=0.32,
        velocity=(0.7, -0.2, 0.0),
        restitution=0.82,
        friction=0.02,
        material=Material(color=(80, 155, 245), absorption=(0.04, 0.04, 0.02)),
    )
    floor = StaticPlane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        friction=0.04,
        restitution=0.9,
        material=Material(color=(90, 125, 150), absorption=(0.12, 0.08, 0.05)),
        size=4.8,
    )
    world = PhysicsWorld(gravity=(0.0, -9.81, 0.0))
    world.add_sphere(ball)
    world.add_plane(floor)

    path = [ball.position]
    for step in range(240):
        world.step(1.0 / 60.0, substeps=2)
        if step % 5 == 0:
            path.append(ball.position)

    scene = base_scene("FLOOR BOUNCE\nRESTITUTION")
    scene.add(floor.to_primitive(), ball.to_primitive())
    add_path(scene, path, color=(110, 230, 245))
    render(scene, "physics_floor_bounce.png", Camera(position=(0.25, 2.0, -5.2), target=(0.2, 0.7, 0.0), fov_degrees=48))


def save_wall_bank() -> None:
    ball = SphereBody(
        position=(-1.75, 0.35, 0.0),
        radius=0.28,
        velocity=(2.4, 0.0, 0.0),
        restitution=0.9,
        friction=0.02,
        material=Material(color=(235, 210, 80), absorption=(0.02, 0.05, 0.18)),
    )
    floor = StaticPlane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        friction=0.03,
        restitution=0.4,
        material=Material(color=(70, 135, 95), absorption=(0.18, 0.1, 0.25)),
        size=5.2,
    )
    wall = StaticBox(
        center=(1.45, 0.45, 0.0),
        size=(0.24, 1.1, 2.0),
        restitution=0.95,
        friction=0.02,
        material=Material(color=(190, 190, 205), absorption=(0.1, 0.1, 0.08)),
    )
    world = PhysicsWorld(gravity=(0.0, 0.0, 0.0))
    world.add_sphere(ball)
    world.add_plane(floor)
    world.add_box(wall)

    path = [ball.position]
    for step in range(150):
        world.step(1.0 / 60.0, substeps=2)
        if step % 4 == 0:
            path.append(ball.position)

    scene = base_scene("WALL BANK\nIMPULSE RESPONSE")
    scene.add(floor.to_primitive(), wall.to_primitive(), ball.to_primitive())
    add_path(scene, path, color=(250, 205, 90))
    render(scene, "physics_wall_bank.png", Camera(position=(0.0, 1.7, -4.8), target=(0.1, 0.45, 0.0), fov_degrees=50))


def save_bumpy_ball() -> None:
    ball = SphereBody(
        position=(-2.15, 1.3, 0.0),
        radius=0.34,
        restitution=0.45,
        friction=0.05,
        material=Material(color=(230, 95, 70), roughness=0.35, fuzziness=0.4),
        visual_perturbation=SurfacePerturbation(magnitude=0.08, scale=3.2, seed=22, octaves=4),
        collision_boundary=SphereCollider(radius=0.34),
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

    scene = base_scene("BUMPY BALL\nVISUAL NOISE")
    scene.add(ramp.to_primitive(), wall.to_primitive(), ball.to_primitive())
    add_path(scene, path)
    render(scene, "physics_bumpy_ball.png", Camera(position=(0.25, 2.2, -5.5), target=(0.2, 0.45, 0.0), fov_degrees=48))


def base_scene(label: str) -> Scene:
    scene = Scene()
    scene.add_light(Sun(direction=(-0.4, -0.8, -1.0), color=(255, 245, 230), intensity=0.9))
    scene.add_light(Lamp(position=(-1.8, 2.0, -2.0), color=(90, 140, 255), intensity=2.4))
    scene.add_bulletin(
        TextBulletin(
            label,
            position=(10, 10),
            color=(245, 248, 255),
            background=(5, 7, 11),
            padding=5,
            scale=1,
        )
    )
    return scene


def render(scene: Scene, file_name: str, camera: Camera) -> None:
    settings = RenderSettings(width=480, height=300, background=Color(8, 10, 14), ambient=0.08)
    path = OUTPUT_DIR / file_name
    RenderEngine().render(scene, camera, settings).to_png(path)
    print(f"Wrote {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_ramp_wall()
    save_floor_bounce()
    save_wall_bank()
    save_bumpy_ball()


if __name__ == "__main__":
    main()
