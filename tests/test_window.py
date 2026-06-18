from py_3d import PixelWindow, WindowEvent


def test_pixel_window_is_public_engine_presenter():
    assert PixelWindow.__name__ == "PixelWindow"
    assert PixelWindow.__dataclass_fields__["title"].default == "py_3d"
    assert WindowEvent("quit").kind == "quit"
