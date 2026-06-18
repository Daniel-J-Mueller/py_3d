from pathlib import Path

from py_3d import Camera, Material, RenderEngine, RenderSettings, Scene, Sun, Triangle


def main() -> None:
    scene = Scene()
    scene.add(
        Triangle(
            (-1.2, -0.9, 0.0),
            (1.2, -0.9, 0.0),
            (0.0, 1.0, 0.0),
            Material(color=(220, 80, 40), absorption=(0.05, 0.15, 0.25)),
        )
    )
    scene.add_light(Sun(direction=(0.0, 0.0, -1.0), color=(255, 245, 230), intensity=1.0))

    camera = Camera(position=(0.0, 0.0, -4.0), target=(0.0, 0.0, 0.0))
    settings = RenderSettings(width=320, height=240, background=(8, 10, 14), ambient=0.08)

    buffer = RenderEngine().render(scene, camera, settings)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "offline_triangle.ppm"
    buffer.to_ppm(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
