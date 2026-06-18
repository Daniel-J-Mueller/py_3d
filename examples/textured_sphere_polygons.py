"""Render a TV-test textured sphere with additional textured polygons."""

from pathlib import Path

from py_3d import (
    Camera,
    Lamp,
    Material,
    PixelBuffer,
    RenderEngine,
    RenderSettings,
    Scene,
    Sphere,
    Sun,
    TextBulletin,
    Triangle,
    planar_project_triangles,
)


def main() -> None:
    texture = PixelBuffer.from_png(Path("assets") / "tv-test.png")
    tv_material = Material(texture=texture, absorption=(0.03, 0.03, 0.03), diffuse=0.95)
    rough_material = Material(color=(185, 145, 90), roughness=0.75, fuzziness=0.35)
    fuzzy_material = Material(color=(110, 210, 175), roughness=0.35, fuzziness=0.8)

    scene = Scene()
    scene.add(
        Sphere(center=(-0.9, 0.1, 0.0), radius=0.85, material=tv_material),
    )

    panel = (
        Triangle((0.55, -0.7, -0.1), (2.05, -0.55, 0.05), (2.1, 0.75, 0.05), tv_material),
        Triangle((0.55, -0.7, -0.1), (2.1, 0.75, 0.05), (0.45, 0.65, -0.1), tv_material),
    )
    scene.add(
        *planar_project_triangles(
            panel,
            center=(1.3, 0.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(0.0, 1.0, 0.0),
            scale=(1.7, 1.4),
            offset=(0.5, 0.5),
        )
    )

    scene.add(
        Triangle((-2.1, -1.05, 0.55), (2.25, -1.05, 0.55), (1.75, -1.05, 2.1), rough_material),
        Triangle((-2.1, -1.05, 0.55), (1.75, -1.05, 2.1), (-1.75, -1.05, 2.05), rough_material),
        Triangle((-1.95, 0.95, 0.4), (-1.3, 1.55, 0.25), (-0.65, 0.95, 0.4), fuzzy_material),
    )

    scene.add_light(Sun(direction=(-0.35, -0.7, -1.0), color=(255, 245, 225), intensity=0.9))
    scene.add_light(Lamp(position=(-2.1, 1.8, -2.2), color=(90, 140, 255), intensity=2.5))
    scene.add_light(Lamp(position=(1.8, 1.2, -1.3), color=(255, 110, 90), intensity=1.6))
    scene.add_bulletin(
        TextBulletin(
            "TEXTURED SPHERE + POLYGONS\nSPHERE UVS + PLANAR FACE PROJECTION",
            position=(10, 10),
            color=(245, 248, 255),
            background=(5, 7, 11),
            padding=5,
            scale=1,
        )
    )

    camera = Camera(position=(0.0, 0.45, -5.0), target=(0.0, 0.0, 0.35), fov_degrees=48)
    settings = RenderSettings(
        width=640,
        height=360,
        background=(7, 9, 14),
        ambient=0.12,
        sphere_segments=28,
        sphere_rings=14,
    )
    output_dir = Path("renderings-tests")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "textured_sphere_polygons.png"
    RenderEngine().render(scene, camera, settings).to_png(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
