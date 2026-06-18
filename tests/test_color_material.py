from py_3d import Color, Material


def test_color_clamps_channels():
    assert Color(-10, 260, 12).to_tuple() == (0, 255, 12)


def test_material_absorption_reduces_channels():
    material = Material(color=(100, 100, 100), absorption=(0.5, 0.0, 1.0))

    shaded = material.shade((1.0, 1.0, 1.0), ambient=0.0)

    assert shaded.r < shaded.g
    assert shaded.b == 0
