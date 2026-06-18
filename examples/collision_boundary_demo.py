"""Render visual geometry beside its collision boundary."""

from pathlib import Path

from py_3d import (
    Camera,
    Lamp,
    Line3,
    Material,
    RenderEngine,
    RenderSettings,
    Scene,
    SphereBody,
    SphereCollider,
    StaticPlane,
    Sun,
    SurfacePerturbation,
    TextBulletin,
)


def main() -> None:
    visual_material = Material(color=(90, 150, 235), roughness=0.35, fuzziness=0.35)
    collider_material = Material(color=(245, 230, 90), emission=(70, 60, 0))
    ball = SphereBody(
        position=(-0.6, 0.7, 0.0),
        radius=0.55,
        material=visual_material,
        visual_perturbation=SurfacePerturbation(magnitude=0.13, scale=3.2, seed=4, octaves=4),
        collision_boundary=SphereCollider(radius=0.38, offset=(0.22, -0.04, 0.0)),
    )
    synced = ball.synced_collision_boundary()
    overridden = ball.effective_collision_boundary()

    floor = StaticPlane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        material=Material(color=(70, 120, 95), roughness=0.35),
        size=3.8,
    )

    scene = Scene()
    scene.add(floor.to_primitive(), ball.to_primitive(), ball.to_collision_primitive(collider_material))
    scene.add(
        Line3(ball.position, overridden.world_center(ball.position), Material(color=(245, 230, 90), emission=(60, 50, 0)))
    )
    scene.add_light(Sun(direction=(-0.4, -0.8, -1.0), color=(255, 245, 230), intensity=0.9))
    scene.add_light(Lamp(position=(-1.5, 1.9, -2.0), color=(90, 140, 255), intensity=2.2))
    scene.add_bulletin(
        TextBulletin(
            f"COLLISION OVERRIDE\nSYNCED R {synced.radius:.2f}  ACTIVE R {overridden.radius:.2f}",
            position=(10, 10),
            color=(245, 248, 255),
            background=(5, 7, 11),
            padding=5,
            scale=1,
        )
    )

    camera = Camera(position=(0.0, 1.2, -4.0), target=(0.0, 0.45, 0.0), fov_degrees=45)
    settings = RenderSettings(width=640, height=360, background=(7, 9, 14), ambient=0.1, sphere_segments=28, sphere_rings=14)
    output_dir = Path("renderings-tests")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "collision_boundary_override.png"
    RenderEngine().render(scene, camera, settings).to_png(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
