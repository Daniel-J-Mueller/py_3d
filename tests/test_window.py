from py_3d import PixelWindow, WindowEvent
from py_3d.assets import framework_favicon_path, load_framework_icon_rgba


def test_pixel_window_is_public_engine_presenter():
    assert PixelWindow.__name__ == "PixelWindow"
    assert PixelWindow.__dataclass_fields__["title"].default == "py_3d"
    assert PixelWindow.__dataclass_fields__["icon_path"].default is None
    assert WindowEvent("quit").kind == "quit"


def test_framework_favicon_asset_loads_for_window_backends():
    width, height, pixels = load_framework_icon_rgba()

    assert framework_favicon_path().exists()
    assert framework_favicon_path().name == "py_3d_logo.png"
    assert (width, height) == (32, 32)
    assert len(pixels) == height
    assert len(pixels[0]) == width
    assert any(pixel[3] > 0 for row in pixels for pixel in row)
