"""Render a textured 3D surface using assets/tv-test.png."""

from pathlib import Path

from py_3d import Camera, Material, PixelBuffer, RenderEngine, RenderSettings, Scene, TextBulletin, Triangle


def main() -> None:
    texture = PixelBuffer.from_png(Path("assets") / "tv-test.png")
    material = Material(texture=texture)

    bottom_left = (-1.78, -1.0, 0.0)
    bottom_right = (1.78, -1.0, 0.0)
    top_right = (1.78, 1.0, 0.0)
    top_left = (-1.78, 1.0, 0.0)

    scene = Scene()
    scene.add(
        Triangle(bottom_left, bottom_right, top_right, material, (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
        Triangle(bottom_left, top_right, top_left, material, (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
    )
    scene.add_bulletin(
        TextBulletin(
            "TEXTURE IMPORT TEST\nASSETS/TV-TEST.PNG",
            position=(10, 10),
            color=(245, 248, 255),
            background=(5, 7, 11),
            padding=5,
            scale=1,
        )
    )

    camera = Camera(position=(0.0, 0.0, -4.0), target=(0.0, 0.0, 0.0), fov_degrees=48)
    settings = RenderSettings(width=640, height=360, background=(7, 9, 14), ambient=1.0)

    output_dir = Path("renderings-tests")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "texture_tv_test.png"
    RenderEngine().render(scene, camera, settings).to_png(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
