import importlib.util
from array import array
from types import SimpleNamespace

import pytest

from py_3d import (
    Box,
    CANONICAL_LIVE_MENU_ACTIONS,
    HUDRect,
    HUDText,
    Material,
    Mesh,
    PixelBuffer,
    RenderSettings,
    Scene,
    Sphere,
    Sun,
    Triangle,
    canonical_live_menu_options,
    canonical_player_movement_key,
    next_camera_mode,
)
from py_3d.live import (
    LiveFlyCamera,
    LiveMenu,
    LiveMenuOption,
    LiveMenuTheme,
    LiveSceneBatchBuilder,
    _GLFWModernGLLiveRenderer,
    _flip_rgba_rows,
    _live_object_fingerprint,
    _overlay_quad_vertices,
    render_live_menu_surface,
)


class _FakeGLFW:
    CURSOR = 1
    CURSOR_DISABLED = 2
    CURSOR_NORMAL = 3

    def __init__(self):
        self.modes = []

    def set_input_mode(self, window, mode, value):
        self.modes.append((window, mode, value))


class _FakeKeyRenderer:
    def key_matches(self, key, *names):
        return key in names


class _FakeGLBuffer:
    def __init__(self):
        self.writes = []
        self.orphans = []

    def orphan(self, size):
        self.orphans.append(size)

    def write(self, data, offset=0):
        self.writes.append((len(data), offset))


def test_live_scene_batch_builder_outputs_triangle_payload():
    scene = Scene()
    sphere = Sphere((0, 0, 0), 0.7, Material(color=(90, 150, 230), specular=0.2))
    scene.add(sphere)
    scene.add_light(Sun(direction=(-0.4, -0.7, -1.0), intensity=1.0))
    settings = RenderSettings(width=64, height=64, sphere_segments=10, sphere_rings=5)
    builder = LiveSceneBatchBuilder()

    triangle_bytes, line_bytes, triangle_vertices, line_vertices = builder.build(scene, settings)

    assert triangle_vertices == len(sphere.to_triangles(segments=10, rings=5)) * 3
    assert len(triangle_bytes) == triangle_vertices * 19 * 4
    assert line_bytes == b""
    assert line_vertices == 0


def test_live_scene_batch_builder_outputs_wireframe_lines():
    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.7, Material(color=(90, 150, 230))))
    settings = RenderSettings(width=64, height=64, wireframe=True, sphere_segments=8, sphere_rings=4)
    builder = LiveSceneBatchBuilder()

    triangle_bytes, line_bytes, triangle_vertices, line_vertices = builder.build(scene, settings)

    assert triangle_bytes == b""
    assert triangle_vertices == 0
    assert len(line_bytes) == line_vertices * 19 * 4
    assert line_vertices > 0


def test_live_scene_batch_builder_respects_flat_generated_normals():
    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.7, Material(color=(90, 150, 230))))
    settings = RenderSettings(width=64, height=64, smooth_shading=False, sphere_segments=8, sphere_rings=4)
    builder = LiveSceneBatchBuilder()

    triangle_bytes, _line_bytes, triangle_vertices, _line_vertices = builder.build(scene, settings)
    payload = array("f")
    payload.frombytes(triangle_bytes)
    first = payload[: 19 * 3]
    normal_a = tuple(first[3:6])
    normal_b = tuple(first[22:25])
    normal_c = tuple(first[41:44])

    assert triangle_vertices > 0
    assert normal_a == normal_b == normal_c


def test_live_scene_batch_builder_assigns_texture_slots():
    texture = PixelBuffer.new(4, 4, (200, 120, 40))
    scene = Scene()
    scene.add(Sphere((0, 0, 0), 0.7, Material(texture=texture, color=(255, 255, 255))))
    settings = RenderSettings(width=64, height=64, sphere_segments=8, sphere_rings=4)
    builder = LiveSceneBatchBuilder()

    triangle_bytes, _line_bytes, triangle_vertices, _line_vertices = builder.build(scene, settings)
    payload = array("f")
    payload.frombytes(triangle_bytes)

    assert triangle_vertices > 0
    assert len(builder.active_textures) == 1
    assert payload[18] == 0.0


def test_live_scene_batch_builder_reuses_box_templates():
    material = Material(color=(180, 120, 70))
    scene = Scene()
    scene.add(Box((0, 0, 0), (1, 2, 3), material), Box((2, 0, 0), (1, 2, 3), material))
    settings = RenderSettings(width=64, height=64, smooth_shading=True)
    builder = LiveSceneBatchBuilder()

    first, _line_bytes, first_vertices, _line_vertices = builder.build(scene, settings)
    second, _line_bytes, second_vertices, _line_vertices = builder.build(scene, settings)

    assert len(builder._box_templates) == 1
    assert first == second
    assert first_vertices == second_vertices == 72


def test_live_scene_batch_builder_keeps_unmapped_box_textures_disabled():
    texture = PixelBuffer.new(4, 4, (20, 40, 80))
    scene = Scene()
    scene.add(Box((0, 0, 0), (1, 1, 1), Material(color=(180, 120, 70), texture=texture)))
    settings = RenderSettings(width=64, height=64)
    builder = LiveSceneBatchBuilder()

    triangle_bytes, _line_bytes, triangle_vertices, _line_vertices = builder.build(scene, settings)
    payload = array("f")
    payload.frombytes(triangle_bytes)

    assert triangle_vertices == 36
    assert builder.active_textures == []
    assert payload[18] == -1.0


def test_live_scene_batch_builder_caches_homogeneous_mesh_templates():
    material = Material(color=(90, 150, 230))
    mesh = Mesh(
        (
            Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0), material),
            Triangle((1, 0, 0), (1, 1, 0), (0, 1, 0), material),
        )
    )
    scene = Scene()
    scene.add(mesh)
    settings = RenderSettings(width=64, height=64)
    builder = LiveSceneBatchBuilder()

    first, _line_bytes, first_vertices, _line_vertices = builder.build(scene, settings)
    second, _line_bytes, second_vertices, _line_vertices = builder.build(scene, settings)

    cached_ref, cached_template = builder._mesh_templates[(id(mesh), settings.smooth_shading)]
    assert cached_ref() is mesh
    assert cached_template is not None
    assert first == second
    assert first_vertices == second_vertices == 6


def test_live_renderer_keeps_unchanged_object_ranges_resident():
    renderer = object.__new__(_GLFWModernGLLiveRenderer)
    renderer.builder = LiveSceneBatchBuilder()
    renderer._resident_entries = {}
    renderer._resident_layout_signature = None
    renderer._resident_texture_key = None
    renderer._resident_triangle_vertices = 0
    renderer._resident_line_vertices = 0
    renderer._triangle_buffer = _FakeGLBuffer()
    renderer._line_buffer = _FakeGLBuffer()
    renderer._triangle_capacity = 4
    renderer._line_capacity = 4
    scene = Scene()
    scene.add(Box((0, 0, 0), (1, 1, 1), Material(color=(180, 120, 70))))
    settings = RenderSettings(width=64, height=64)

    triangle_vertices, line_vertices = renderer._update_resident_scene_buffers(scene, settings)
    first_writes = list(renderer._triangle_buffer.writes)
    renderer._triangle_buffer.writes.clear()
    renderer._line_buffer.writes.clear()
    second_vertices, second_lines = renderer._update_resident_scene_buffers(scene, settings)

    assert triangle_vertices == second_vertices == 36
    assert line_vertices == second_lines == 0
    assert first_writes
    assert renderer._triangle_buffer.writes == []
    assert renderer._line_buffer.writes == []


def test_live_renderer_rewrites_dirty_object_range_only():
    renderer = object.__new__(_GLFWModernGLLiveRenderer)
    renderer.builder = LiveSceneBatchBuilder()
    renderer._resident_entries = {}
    renderer._resident_layout_signature = None
    renderer._resident_texture_key = None
    renderer._resident_triangle_vertices = 0
    renderer._resident_line_vertices = 0
    renderer._triangle_buffer = _FakeGLBuffer()
    renderer._line_buffer = _FakeGLBuffer()
    renderer._triangle_capacity = 4096
    renderer._line_capacity = 4096
    settings = RenderSettings(width=64, height=64)
    scene = Scene()
    scene.add(
        Box((0, 0, 0), (1, 1, 1), Material(color=(180, 120, 70))),
        Box((2, 0, 0), (1, 1, 1), Material(color=(90, 150, 230))),
    )
    renderer._update_resident_scene_buffers(scene, settings)
    renderer._triangle_buffer.writes.clear()
    moved_scene = Scene()
    moved_scene.add(
        Box((0, 0, 0), (1, 1, 1), Material(color=(180, 120, 70))),
        Box((3, 0, 0), (1, 1, 1), Material(color=(90, 150, 230))),
    )

    renderer._update_resident_scene_buffers(moved_scene, settings)

    assert renderer._triangle_buffer.writes == [(36 * 19 * 4, 36 * 19 * 4)]


def test_live_object_fingerprint_keeps_recreated_static_meshes_resident():
    material = Material(color=(90, 150, 230))
    first = Mesh((Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0), material),))
    second = Mesh((Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0), material),))
    settings = RenderSettings(width=64, height=64)

    class StablePrimitive:
        def __repr__(self):
            return "StablePrimitive(value=1)"

    assert _live_object_fingerprint(first, settings) == _live_object_fingerprint(second, settings)
    assert _live_object_fingerprint(StablePrimitive(), settings) == _live_object_fingerprint(StablePrimitive(), settings)


def test_overlay_quad_vertices_map_pixels_to_clip_space():
    vertices = _overlay_quad_vertices(10, 5, 20, 10, 100, 50)

    assert len(vertices) == 24
    assert vertices[:4] == (-0.8, 0.8, 0.0, 1.0)
    assert vertices[4:8] == (-0.4, 0.8, 1.0, 1.0)
    assert vertices[8:12] == (-0.4, 0.4, 1.0, 0.0)


def test_rgba_rows_flip_for_opengl_uploads():
    payload = bytes(range(16))

    assert _flip_rgba_rows(payload, 2, 2) == bytes(range(8, 16)) + bytes(range(0, 8))


def test_live_fly_camera_uses_mouse_look_and_vertical_keys():
    camera = LiveFlyCamera.looking_at((0, 1, -4), (0, 1, 0), fov_degrees=70.0, speed=2.0)

    camera.look(100, -50)
    camera.move({"w", "space"}, 0.5)
    view = camera.camera()

    assert camera.yaw_degrees > 0.0
    assert camera.pitch_degrees > 0.0
    assert camera.position.y > 1.0
    assert view.fov_degrees == 70.0
    assert view.target != view.position


def test_live_fly_camera_shift_adds_vertical_thrust_for_noclip():
    camera = LiveFlyCamera.looking_at((0, 1, -4), (0, 1, 0), fov_degrees=70.0, speed=2.0)

    camera.move({"shift"}, 0.5)

    assert camera.position.y > 1.0


def test_canonical_player_controls_match_fpv_modes():
    renderer = _FakeKeyRenderer()

    assert canonical_player_movement_key(renderer, "w", camera_mode="first") == "w"
    assert canonical_player_movement_key(renderer, "space", camera_mode="first") == "space"
    assert canonical_player_movement_key(renderer, "lshift", camera_mode="first") == "sprint"
    assert canonical_player_movement_key(renderer, "lshift", camera_mode="global") == "shift"
    assert canonical_player_movement_key(renderer, "lctrl", camera_mode="first") == "crouch"
    assert canonical_player_movement_key(renderer, "lctrl", camera_mode="global") == "ctrl"
    assert next_camera_mode("global") == "third"
    assert next_camera_mode("third") == "first"
    assert next_camera_mode("first") == "global"


def test_canonical_live_menu_keeps_unsupported_settings_disabled():
    options = canonical_live_menu_options(details={"sky_cycle": "on"}, enabled_actions={"sky_cycle"})
    by_action = {option.action: option for option in options}

    assert "quality_next" in CANONICAL_LIVE_MENU_ACTIONS
    assert by_action["sky_cycle"].enabled is True
    assert by_action["sky_cycle"].detail == "on"
    assert by_action["quality_next"].enabled is False
    assert by_action["quality_next"].detail == "n/a"
    assert by_action["done"].enabled is True
    assert by_action["quit"].enabled is True


def test_hud_elements_are_public_overlay_types():
    rect = HUDRect((0, 0), (10, 5), (1, 2, 3), alpha=0.5)
    text = HUDText("HUD", (2, 2), alpha=0.75)

    assert rect.alpha == 0.5
    assert text.text == "HUD"


def test_live_menu_groups_and_scrolls_options():
    menu = LiveMenu(
        options=(
            LiveMenuOption("done", "Done"),
            LiveMenuOption("quality_next", "Quality", "high", "Graphics"),
            LiveMenuOption("sky_cycle", "Cycle", "off", "Sky"),
            LiveMenuOption("reset", "Reset", "world", "Physics"),
        )
    )

    assert menu.groups() == ("Graphics", "Sky", "Physics")
    menu.set_group("Sky")
    assert menu.visible_option_indexes() == [2]
    menu.scroll(4)

    assert menu.scroll_offsets["Sky"] == 0


def test_live_menu_group_headers_require_click_before_switching():
    menu = LiveMenu(
        options=(
            LiveMenuOption("done", "Done"),
            LiveMenuOption("quality_next", "Quality", "high", "Graphics"),
            LiveMenuOption("sky_cycle", "Cycle", "off", "Sky"),
            LiveMenuOption("reset", "Reset", "world", "Physics"),
        ),
        visible=True,
    )
    render_live_menu_surface(menu, 800, 600)
    sky_tab = next(hitbox for hitbox in menu.tab_hitboxes if hitbox[4] == "Sky")
    pos = (sky_tab[0] + sky_tab[2] // 2, sky_tab[1] + sky_tab[3] // 2)

    assert menu.current_group() == "Graphics"
    assert menu.handle_pointer_event(SimpleNamespace(kind="motion", pos=pos)) == "handled"
    assert menu.current_group() == "Graphics"

    action = menu.handle_pointer_event(SimpleNamespace(kind="button", button=1, pos=pos))

    assert action == "navigate"
    assert menu.current_group() == "Sky"


def test_live_menu_option_hover_does_not_reselect_rows():
    menu = LiveMenu(
        options=(
            LiveMenuOption("done", "Done"),
            LiveMenuOption("quality_next", "Quality", "high", "Graphics"),
            LiveMenuOption("wind_up", "Wind +", "1.0x", "Physics"),
        ),
        visible=True,
    )
    render_live_menu_surface(menu, 800, 600)
    quality_hitbox = next(hitbox for hitbox in menu.hitboxes if hitbox[4] == 1)
    pos = (quality_hitbox[0] + quality_hitbox[2] // 2, quality_hitbox[1] + quality_hitbox[3] // 2)

    assert menu.selected_action() == "done"
    assert menu.handle_pointer_event(SimpleNamespace(kind="motion", pos=pos)) == "handled"
    assert menu.selected_action() == "done"


def test_live_menu_disabled_options_render_but_do_not_activate():
    menu = LiveMenu(
        options=(
            LiveMenuOption("done", "Done"),
            LiveMenuOption("quality_next", "Quality", "n/a", "Graphics", enabled=False),
        ),
        visible=True,
    )
    render_live_menu_surface(menu, 800, 600)
    disabled_hitbox = next(hitbox for hitbox in menu.hitboxes if hitbox[4] == 1)
    pos = (disabled_hitbox[0] + disabled_hitbox[2] // 2, disabled_hitbox[1] + disabled_hitbox[3] // 2)

    assert menu.handle_pointer_event(SimpleNamespace(kind="button", button=1, pos=pos)) == "handled"
    assert menu.selected_action() == "quality_next"
    assert menu.handle_key_name("enter") == "handled"


def test_live_menu_background_blur_is_optional():
    menu = LiveMenu(background_blur=True)

    assert menu.background_blur is True


def test_live_menu_escape_opens_and_escape_again_chooses_cancel_action():
    menu = LiveMenu(options=(LiveMenuOption("done", "Done"), LiveMenuOption("cancel", "Cancel")))

    assert menu.handle_key_name("escape") == "opened"
    assert menu.visible is True
    assert menu.handle_key_name("escape") == "cancel"


def test_mouse_capture_waits_for_pointer_activation_and_releases_on_menu():
    renderer = object.__new__(_GLFWModernGLLiveRenderer)
    renderer._glfw = _FakeGLFW()
    renderer._window = object()
    renderer.mouse_captured = False
    renderer._mouse_capture_requested = False
    renderer._pointer_activated = False
    renderer._window_focused = True
    renderer._window_iconified = False
    renderer._last_mouse = (4, 5)
    renderer.menu = LiveMenu()

    renderer.set_mouse_captured(True)
    assert renderer.mouse_captured is False

    renderer._pointer_activated = True
    renderer.set_mouse_captured(True)
    assert renderer.mouse_captured is True

    renderer.menu.open()
    renderer._sync_mouse_capture()
    assert renderer.mouse_captured is False


def test_glfw_live_renderer_initializes_capture_state_in_hidden_window():
    if importlib.util.find_spec("glfw") is None or importlib.util.find_spec("moderngl") is None:
        pytest.skip("glfw/moderngl are not installed")

    import glfw

    if not glfw.init():
        pytest.skip("GLFW could not initialize")
    renderer = None
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        try:
            renderer = _GLFWModernGLLiveRenderer(32, 24, vsync=False)
        except RuntimeError as exc:
            pytest.skip(str(exc))

        renderer.set_mouse_captured(True)

        assert renderer._mouse_capture_requested is True
        assert renderer._pointer_activated is False
        assert renderer.mouse_captured is False
    finally:
        if renderer is not None:
            renderer.close()
        glfw.terminate()


def test_live_menu_theme_and_paired_buttons_render():
    menu = LiveMenu(
        options=(
            LiveMenuOption("done", "Done"),
            LiveMenuOption("gamma_up", "Gamma +", "1.20", "Graphics"),
            LiveMenuOption("gamma_down", "Gamma -", "1.20", "Graphics"),
        ),
        visible=True,
        theme=LiveMenuTheme(panel=(0, 0, 0, 240)),
    )
    surface, left, top = render_live_menu_surface(menu, 800, 600)

    assert surface.width < 800
    assert left > 0 and top > 0
    assert any(hitbox[4] == 1 for hitbox in menu.hitboxes)
    assert any(hitbox[4] == 2 for hitbox in menu.hitboxes)
