from py_3d import Color, PixelBuffer, draw


def test_draw_line_marks_endpoints_and_middle():
    buffer = PixelBuffer.new(5, 5)

    draw.line(buffer, (0, 0), (4, 4), Color(255, 0, 0))

    assert buffer.get_pixel(0, 0) == Color(255, 0, 0)
    assert buffer.get_pixel(2, 2) == Color(255, 0, 0)
    assert buffer.get_pixel(4, 4) == Color(255, 0, 0)


def test_draw_filled_rect():
    buffer = PixelBuffer.new(5, 5)

    draw.rect(buffer, (1, 1), (3, 2), Color(0, 255, 0), fill=True)

    assert buffer.get_pixel(1, 1) == Color(0, 255, 0)
    assert buffer.get_pixel(3, 2) == Color(0, 255, 0)
    assert buffer.get_pixel(4, 4) == Color(0, 0, 0)


def test_draw_filled_circle():
    buffer = PixelBuffer.new(7, 7)

    draw.circle(buffer, (3, 3), 2, Color(0, 0, 255), fill=True)

    assert buffer.get_pixel(3, 3) == Color(0, 0, 255)
    assert buffer.get_pixel(3, 1) == Color(0, 0, 255)
    assert buffer.get_pixel(0, 0) == Color(0, 0, 0)


def test_draw_text_marks_pixels():
    buffer = PixelBuffer.new(40, 12)

    draw.text(buffer, (1, 1), "A1", Color(255, 255, 255))

    assert any(pixel == Color(255, 255, 255) for pixel in buffer.pixels)
    assert draw.text_size("A1") == (11, 7)
