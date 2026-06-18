from pathlib import Path

from py_3d import Color, DepthBuffer, PixelBuffer


def test_pixel_buffer_set_get_and_clear():
    buffer = PixelBuffer.new(3, 2, Color(1, 2, 3))

    assert buffer.get_pixel(0, 0) == Color(1, 2, 3)

    buffer.set_pixel(2, 1, (10, 20, 30))
    assert buffer.get_pixel(2, 1) == Color(10, 20, 30)

    buffer.clear((5, 6, 7))
    assert all(pixel == Color(5, 6, 7) for pixel in buffer.pixels)


def test_depth_buffer_prefers_lower_depth():
    depth = DepthBuffer.new(2, 2)

    assert depth.test_and_set(1, 1, 10.0)
    assert not depth.test_and_set(1, 1, 12.0)
    assert depth.test_and_set(1, 1, 5.0)
    assert depth.get(1, 1) == 5.0


def test_pixel_buffer_writes_ppm(tmp_path: Path):
    buffer = PixelBuffer.new(1, 1, Color(255, 0, 0))
    target = tmp_path / "pixel.ppm"

    buffer.to_ppm(target)

    assert target.read_bytes() == b"P6\n1 1\n255\n\xff\x00\x00"


def test_pixel_buffer_writes_png(tmp_path: Path):
    buffer = PixelBuffer.new(2, 1, Color(255, 0, 0))
    buffer.set_pixel(1, 0, Color(0, 0, 255))
    target = tmp_path / "pixels.png"

    buffer.to_png(target)

    data = target.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in data
    assert b"IDAT" in data
    assert data.endswith(b"IEND\xaeB`\x82")


def test_pixel_buffer_resizes_with_nearest_neighbor():
    buffer = PixelBuffer.new(2, 1)
    buffer.set_pixel(0, 0, Color(255, 0, 0))
    buffer.set_pixel(1, 0, Color(0, 0, 255))

    resized = buffer.resized_nearest(4, 2)

    assert resized.width == 4
    assert resized.height == 2
    assert resized.get_pixel(0, 0) == Color(255, 0, 0)
    assert resized.get_pixel(1, 1) == Color(255, 0, 0)
    assert resized.get_pixel(2, 0) == Color(0, 0, 255)
    assert resized.get_pixel(3, 1) == Color(0, 0, 255)


def test_pixel_buffer_reads_png_asset():
    buffer = PixelBuffer.from_png("assets/tv-test.png")

    assert buffer.width == 800
    assert buffer.height == 450

    yellow_bar = buffer.get_pixel(220, 60)
    cyan_bar = buffer.get_pixel(310, 60)
    blue_bar = buffer.get_pixel(660, 60)

    assert yellow_bar.r > 180 and yellow_bar.g > 180 and yellow_bar.b < 40
    assert cyan_bar.r < 40 and cyan_bar.g > 150 and cyan_bar.b > 150
    assert blue_bar.r < 40 and blue_bar.g < 40 and blue_bar.b > 120


def test_rgb_backed_pixel_buffer_updates_single_pixel_without_materializing():
    buffer = PixelBuffer.from_rgb_bytes(2, 1, bytes((0, 0, 0, 10, 20, 30)))

    buffer.set_pixel(0, 0, Color(1, 2, 3))

    assert buffer.to_rgb_bytes() == bytes((1, 2, 3, 10, 20, 30))
    assert hasattr(buffer.pixels, "raw_rgb_bytes")


def test_rgb_backed_pixel_buffer_updates_slice_without_materializing():
    buffer = PixelBuffer.from_rgb_bytes(3, 1, bytes((0, 0, 0, 0, 0, 0, 0, 0, 0)))

    buffer.pixels[1:3] = [Color(4, 5, 6), Color(7, 8, 9)]

    assert buffer.to_rgb_bytes() == bytes((0, 0, 0, 4, 5, 6, 7, 8, 9))
    assert hasattr(buffer.pixels, "raw_rgb_bytes")


def test_rgb_backed_pixel_buffer_clears_without_materializing():
    buffer = PixelBuffer.from_rgb_bytes(2, 2, bytes((0, 0, 0)) * 4)

    buffer.clear(Color(11, 12, 13))

    assert buffer.to_rgb_bytes() == bytes((11, 12, 13)) * 4
    assert hasattr(buffer.pixels, "raw_rgb_bytes")


def test_rgb_backed_pixel_buffer_fills_rect_without_materializing():
    buffer = PixelBuffer.from_rgb_bytes(4, 3, bytes((0, 0, 0)) * 12)

    buffer.fill_rect((1, 1), (2, 2), Color(20, 30, 40))

    assert buffer.get_pixel(0, 0) == Color(0, 0, 0)
    assert buffer.get_pixel(1, 1) == Color(20, 30, 40)
    assert buffer.get_pixel(2, 2) == Color(20, 30, 40)
    assert buffer.get_pixel(3, 2) == Color(0, 0, 0)
    assert hasattr(buffer.pixels, "raw_rgb_bytes")


def test_rgb_backed_pixel_buffer_blits_rows_without_materializing():
    target = PixelBuffer.from_rgb_bytes(4, 3, bytes((0, 0, 0)) * 12)
    source = PixelBuffer.from_rgb_bytes(
        2,
        2,
        bytes(
            (
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
            )
        ),
    )

    target.blit(source, 1, 1)

    assert target.get_pixel(1, 1) == Color(10, 20, 30)
    assert target.get_pixel(2, 1) == Color(40, 50, 60)
    assert target.get_pixel(1, 2) == Color(70, 80, 90)
    assert target.get_pixel(2, 2) == Color(100, 110, 120)
    assert hasattr(target.pixels, "raw_rgb_bytes")


def test_rgb_backed_pixel_buffer_blits_after_materializing_target():
    target = PixelBuffer.from_rgb_bytes(3, 1, bytes((0, 0, 0)) * 3)
    source = PixelBuffer.from_rgb_bytes(2, 1, bytes((10, 20, 30, 40, 50, 60)))
    assert target.pixels[:] == [Color(0, 0, 0), Color(0, 0, 0), Color(0, 0, 0)]

    target.blit(source, 1, 0)

    assert target.get_pixel(0, 0) == Color(0, 0, 0)
    assert target.get_pixel(1, 0) == Color(10, 20, 30)
    assert target.get_pixel(2, 0) == Color(40, 50, 60)
