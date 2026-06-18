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


def test_draw_filled_rect_clips_to_buffer():
    buffer = PixelBuffer.new(5, 5)

    draw.rect(buffer, (-2, -2), (4, 4), Color(20, 30, 40), fill=True)

    assert buffer.get_pixel(0, 0) == Color(20, 30, 40)
    assert buffer.get_pixel(1, 1) == Color(20, 30, 40)
    assert buffer.get_pixel(2, 2) == Color(0, 0, 0)


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


def test_draw_plus_glyph_marks_pixels():
    buffer = PixelBuffer.new(12, 12)

    draw.text(buffer, (1, 1), "+", Color(255, 255, 255))

    assert sum(1 for pixel in buffer.pixels if pixel == Color(255, 255, 255)) == 9


def test_fit_text_uses_pixel_width():
    value = draw.fit_text("CAPSULE PLAYER CONTROLLER", 120, scale=2)

    assert draw.text_size(value, scale=2)[0] <= 120
    assert value.endswith("..")


def test_wrap_text_uses_pixel_width():
    lines = draw.wrap_text("Gamma and texture resolution controls", 100, scale=2, max_lines=3)

    assert lines
    assert len(lines) <= 3
    assert all(draw.text_size(line, scale=2)[0] <= 100 for line in lines)
