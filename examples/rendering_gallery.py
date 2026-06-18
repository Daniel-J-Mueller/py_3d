from pathlib import Path

from py_3d import (
    Box,
    Camera,
    Color,
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
    draw,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "renderings-tests"


def save_2d_primitives() -> None:
    buffer = PixelBuffer.new(320, 200, Color(11, 14, 20))
    draw.rect(buffer, (20, 20), (120, 70), (55, 160, 220), fill=False)
    draw.rect(buffer, (170, 24), (90, 54), (220, 190, 70), fill=True)
    draw.line(buffer, (18, 175), (300, 28), (235, 90, 80))
    draw.line(buffer, (18, 28), (300, 175), (85, 210, 125))
    draw.circle(buffer, (88, 132), 34, (240, 240, 245), fill=False)
    draw.circle(buffer, (230, 138), 38, (110, 130, 245), fill=True)
    draw.circle(buffer, (230, 138), 39, (235, 240, 255), fill=False)
    buffer.to_png(OUTPUT_DIR / "2d_primitives.png")


def save_2d_depth_style_layers() -> None:
    buffer = PixelBuffer.new(320, 200, Color(9, 11, 16))
    colors = [(80, 95, 210), (70, 170, 130), (220, 165, 75), (220, 80, 95)]
    for index, color in enumerate(colors):
        offset = index * 28
        draw.rect(buffer, (40 + offset, 36 + offset // 2), (150, 92), color, fill=True)
        draw.rect(buffer, (40 + offset, 36 + offset // 2), (150, 92), (245, 245, 250), fill=False)
    draw.line(buffer, (28, 170), (292, 170), (230, 230, 235))
    buffer.to_png(OUTPUT_DIR / "2d_layers.png")


def save_3d_lit_triangle() -> None:
    scene = Scene()
    scene.add(
        Triangle(
            (-1.15, -0.95, 0.0),
            (1.15, -0.85, 0.0),
            (0.0, 1.05, 0.0),
            Material(color=(225, 90, 55), absorption=(0.05, 0.1, 0.2)),
        )
    )
    scene.add_light(Sun(direction=(0.0, 0.0, -1.0), color=(255, 245, 230), intensity=1.0))
    scene.add_light(Lamp(position=(-1.2, 1.6, -1.5), color=(100, 150, 255), intensity=1.5))
    camera = Camera(position=(0.0, 0.0, -4.0), target=(0.0, 0.0, 0.0))
    settings = RenderSettings(width=320, height=240, background=(8, 10, 14), ambient=0.08)
    RenderEngine().render(scene, camera, settings).to_png(OUTPUT_DIR / "3d_lit_triangle.png")


def save_3d_wire_box() -> None:
    scene = Scene()
    scene.add(Box(center=(0.0, 0.0, 0.0), size=(1.5, 1.5, 1.5), material=Material(color=(70, 220, 180))))
    camera = Camera(position=(2.4, 1.7, -4.2), target=(0.0, 0.0, 0.0), fov_degrees=50)
    settings = RenderSettings(width=320, height=240, background=(6, 8, 12), wireframe=True)
    RenderEngine().render(scene, camera, settings).to_png(OUTPUT_DIR / "3d_wire_box.png")


def save_3d_lit_sphere() -> None:
    scene = Scene()
    scene.add(
        Sphere(
            center=(0.0, 0.0, 0.0),
            radius=1.0,
            material=Material(color=(85, 150, 235), absorption=(0.08, 0.05, 0.02)),
        )
    )
    scene.add_light(Sun(direction=(-0.5, -0.7, -1.0), color=(255, 245, 230), intensity=0.95))
    scene.add_light(Lamp(position=(1.7, 1.2, -2.0), color=(255, 80, 120), intensity=2.4))
    camera = Camera(position=(0.0, 0.0, -4.0), target=(0.0, 0.0, 0.0))
    settings = RenderSettings(
        width=320,
        height=240,
        background=(8, 10, 14),
        ambient=0.05,
        sphere_segments=24,
        sphere_rings=12,
    )
    RenderEngine().render(scene, camera, settings).to_png(OUTPUT_DIR / "3d_lit_sphere.png")


def save_github_banner() -> None:
    scene = Scene()
    scene.add(
        Sphere(
            center=(-0.85, 0.0, 0.0),
            radius=0.95,
            material=Material(color=(70, 145, 240), absorption=(0.04, 0.05, 0.02)),
        ),
        Box(
            center=(1.1, -0.15, 0.15),
            size=(1.25, 1.25, 1.25),
            material=Material(color=(235, 170, 85), absorption=(0.05, 0.1, 0.16), emission=(6, 3, 0)),
        ),
    )
    scene.add_light(Sun(direction=(-0.4, -0.7, -1.0), color=(255, 244, 220), intensity=0.95))
    scene.add_light(Lamp(position=(-2.2, 1.8, -2.0), color=(80, 150, 255), intensity=3.0))
    scene.add_light(Lamp(position=(2.3, 1.1, -1.3), color=(255, 90, 120), intensity=2.0))
    scene.add_light(Lamp(position=(0.0, 1.4, -2.4), color=(255, 235, 210), intensity=2.4))
    scene.add_bulletin(
        TextBulletin(
            "PY_3D\nPRIMITIVE 3D PIXEL RENDERING",
            position=(34, 34),
            color=(245, 248, 255),
            background=(6, 8, 12),
            padding=14,
            scale=4,
        ),
        TextBulletin(
            "CPU REFERENCE - GPU READY - OFFLINE AND LIVE",
            position=(42, 190),
            color=(120, 220, 190),
            background=None,
            padding=0,
            scale=2,
        ),
    )
    camera = Camera(position=(0.0, 0.3, -4.8), target=(0.1, 0.0, 0.0), fov_degrees=48)
    settings = RenderSettings(
        width=1280,
        height=640,
        background=(7, 9, 14),
        ambient=0.11,
        sphere_segments=28,
        sphere_rings=14,
    )
    RenderEngine().render(scene, camera, settings).to_png(OUTPUT_DIR / "github-banner.png")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_2d_primitives()
    save_2d_depth_style_layers()
    save_3d_lit_triangle()
    save_3d_wire_box()
    save_3d_lit_sphere()
    save_github_banner()
    print(f"Wrote PNG renderings to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
