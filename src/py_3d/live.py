"""Interactive OpenGL presentation for live demos.

This module is intentionally optional. Importing it does not create a window;
``ModernGLLiveRenderer`` imports pygame and ModernGL only when constructed.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import asin, atan2, cos, degrees, radians, sin, tan
from time import perf_counter
from typing import Iterable

from .buffer import PixelBuffer
from .camera import Camera
from .color import Color
from . import draw
from .lights import Lamp, Sun
from .materials import Material
from .math3d import Vec3
from .overlays import FloatingTextBulletin, TextBulletin
from .primitives import Bowl, Box, Capsule, Line3, Mesh, Plane, Point3, Sphere, Triangle
from .render import RenderSettings, _triangles_for
from .scene import Scene


VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec3 in_emission;
in vec3 in_material;

uniform vec3 u_camera_position;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform vec3 u_camera_forward;
uniform float u_focal;
uniform float u_aspect;
uniform float u_near;
uniform float u_far;

out vec3 v_world_position;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_emission;
out vec3 v_material;

void main() {
    vec3 relative = in_position - u_camera_position;
    float camera_x = dot(relative, u_camera_right);
    float camera_y = dot(relative, u_camera_up);
    float camera_z = dot(relative, u_camera_forward);
    float safe_z = max(camera_z, u_near);

    float ndc_x = (camera_x * u_focal / u_aspect) / safe_z;
    float ndc_y = (camera_y * u_focal) / safe_z;
    float ndc_z = -1.0 + 2.0 * ((safe_z - u_near) / max(0.0001, u_far - u_near));

    gl_Position = vec4(ndc_x, ndc_y, ndc_z, 1.0);
    v_world_position = in_position;
    v_normal = in_normal;
    v_color = in_color;
    v_emission = in_emission;
    v_material = in_material;
}
"""


FRAGMENT_SHADER = """
#version 330

const int MAX_LIGHTS = 8;

in vec3 v_world_position;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_emission;
in vec3 v_material;

uniform vec3 u_camera_position;
uniform float u_ambient;
uniform float u_gamma;
uniform float u_light_wrap;
uniform float u_bounce_light;
uniform bool u_tone_mapping;
uniform bool u_two_sided_lighting;
uniform int u_light_count;
uniform int u_light_kind[MAX_LIGHTS];
uniform vec3 u_light_vector[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform float u_light_intensity[MAX_LIGHTS];

out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 view_direction = normalize(u_camera_position - v_world_position);
    if (u_two_sided_lighting && dot(normal, view_direction) < 0.0) {
        normal = -normal;
    }

    vec3 diffuse_light = vec3(0.0);
    vec3 specular_light = vec3(0.0);
    float diffuse = v_material.x;
    float specular = v_material.y;
    float shininess = max(1.0, v_material.z);
    diffuse_light += vec3(0.22, 0.26, 0.34) * max(0.0, normal.y) * u_bounce_light;
    diffuse_light += vec3(0.32, 0.22, 0.14) * max(0.0, -normal.y) * u_bounce_light;

    for (int index = 0; index < MAX_LIGHTS; ++index) {
        if (index >= u_light_count) {
            break;
        }

        vec3 light_direction;
        float strength;
        if (u_light_kind[index] == 0) {
            light_direction = normalize(u_light_vector[index]);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index];
        } else {
            vec3 offset = u_light_vector[index] - v_world_position;
            float distance = max(0.0001, length(offset));
            light_direction = offset / distance;
            float attenuation = 1.0 / (1.0 + distance * distance);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index] * attenuation;
        }

        vec3 light_color = u_light_color[index];
        diffuse_light += light_color * strength;
        if (specular > 0.0 && strength > 0.0) {
            vec3 halfway = normalize(light_direction + view_direction);
            specular_light += light_color * pow(max(0.0, dot(normal, halfway)), shininess) * strength;
        }
    }

    vec3 color = v_emission + v_color * (u_ambient + diffuse_light * diffuse) + specular_light * specular;
    if (u_tone_mapping) {
        color = color / (color + vec3(1.0));
    } else {
        color = clamp(color, 0.0, 1.0);
    }
    float gamma = max(0.01, u_gamma);
    if (abs(gamma - 1.0) > 0.0001) {
        color = pow(color, vec3(1.0 / gamma));
    }
    frag_color = vec4(color, 1.0);
}
"""


OVERLAY_VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
"""


OVERLAY_FRAGMENT_SHADER = """
#version 330

uniform sampler2D u_texture;

in vec2 v_texcoord;

out vec4 frag_color;

void main() {
    frag_color = texture(u_texture, v_texcoord);
}
"""


_FLOATS_PER_VERTEX = 15
_VERTEX_FORMAT = "3f 3f 3f 3f 3f"
_VERTEX_ATTRIBUTES = ("in_position", "in_normal", "in_color", "in_emission", "in_material")
_OVERLAY_VERTEX_FORMAT = "2f 2f"
_OVERLAY_VERTEX_ATTRIBUTES = ("in_position", "in_texcoord")


@dataclass(frozen=True)
class LiveFrameStats:
    """Timing and payload counters for the most recent live frame."""

    build_seconds: float
    draw_seconds: float
    triangle_vertices: int
    line_vertices: int

    @property
    def total_seconds(self) -> float:
        return self.build_seconds + self.draw_seconds

    @property
    def approx_fps(self) -> float:
        return 1.0 / self.total_seconds if self.total_seconds > 0.0 else 0.0


@dataclass
class LiveFlyCamera:
    """Mouse-look camera for direct live navigation."""

    position: Vec3
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    fov_degrees: float = 60.0
    speed: float = 2.8
    mouse_sensitivity: float = 0.12

    @classmethod
    def looking_at(
        cls,
        position: Vec3 | tuple[float, float, float],
        target: Vec3 | tuple[float, float, float],
        *,
        fov_degrees: float = 60.0,
        speed: float = 2.8,
        mouse_sensitivity: float = 0.12,
    ) -> "LiveFlyCamera":
        eye = position if isinstance(position, Vec3) else Vec3(*position)
        target_vec = target if isinstance(target, Vec3) else Vec3(*target)
        forward = (target_vec - eye).normalized(Vec3(0.0, 0.0, 1.0))
        return cls(
            position=eye,
            yaw_degrees=degrees(atan2(forward.x, forward.z)),
            pitch_degrees=degrees(asin(max(-1.0, min(1.0, forward.y)))),
            fov_degrees=fov_degrees,
            speed=speed,
            mouse_sensitivity=mouse_sensitivity,
        )

    @property
    def forward(self) -> Vec3:
        yaw = radians(self.yaw_degrees)
        pitch = radians(self.pitch_degrees)
        return Vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)).normalized(Vec3(0.0, 0.0, 1.0))

    @property
    def flat_forward(self) -> Vec3:
        yaw = radians(self.yaw_degrees)
        return Vec3(sin(yaw), 0.0, cos(yaw)).normalized(Vec3(0.0, 0.0, 1.0))

    @property
    def right(self) -> Vec3:
        forward = self.flat_forward
        return Vec3(forward.z, 0.0, -forward.x)

    def look(self, relative_x: float, relative_y: float) -> None:
        self.yaw_degrees += relative_x * self.mouse_sensitivity
        self.pitch_degrees = max(-86.0, min(86.0, self.pitch_degrees - relative_y * self.mouse_sensitivity))

    def move(self, keys: set[str], dt: float) -> None:
        if dt <= 0.0:
            return
        direction = Vec3(0.0, 0.0, 0.0)
        if "w" in keys:
            direction = direction + self.flat_forward
        if "s" in keys:
            direction = direction - self.flat_forward
        if "d" in keys:
            direction = direction + self.right
        if "a" in keys:
            direction = direction - self.right
        if "shift" in keys:
            direction = direction + Vec3(0.0, 1.0, 0.0)
        if "ctrl" in keys:
            direction = direction - Vec3(0.0, 1.0, 0.0)
        if direction.length_squared() > 0.0:
            self.position = self.position + direction.normalized() * (self.speed * dt)

    def camera(self) -> Camera:
        return Camera(position=self.position, target=self.position + self.forward, fov_degrees=self.fov_degrees)


@dataclass(frozen=True)
class LiveMenuOption:
    """One decision in a live-rendering escape menu."""

    action: str
    label: str
    detail: str = ""


@dataclass
class LiveMenu:
    """Keyboard-driven menu state for live OpenGL demos."""

    title: str = "py_3d live menu"
    options: tuple[LiveMenuOption, ...] = (
        LiveMenuOption("resume", "Resume"),
        LiveMenuOption("quit", "Quit"),
    )
    selected_index: int = 0
    visible: bool = False

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("live menu requires at least one option")
        self.selected_index %= len(self.options)

    def open(self) -> None:
        self.visible = True

    def close(self) -> None:
        self.visible = False

    def toggle(self) -> None:
        self.visible = not self.visible

    def move(self, amount: int) -> None:
        self.selected_index = (self.selected_index + amount) % len(self.options)

    def selected_action(self) -> str:
        return self.options[self.selected_index].action

    def handle_key(self, key: int, pygame) -> str | None:
        if not self.visible:
            if key == pygame.K_ESCAPE:
                self.open()
                return "opened"
            return None
        if key == pygame.K_ESCAPE:
            return "resume"
        if key in (pygame.K_DOWN, pygame.K_s, pygame.K_TAB):
            self.move(1)
            return "navigate"
        if key in (pygame.K_UP, pygame.K_w):
            self.move(-1)
            return "navigate"
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            return self.selected_action()
        return "handled"


@dataclass(frozen=True)
class _Template:
    vertices: tuple[tuple[float, float, float, float, float, float, float, float], ...]


class LiveSceneBatchBuilder:
    """Build GPU vertex payloads from py_3d scenes with generated-mesh caches."""

    def __init__(self) -> None:
        self._sphere_templates: dict[tuple[float, int | None, int, int, bool], _Template] = {}
        self._bowl_templates: dict[tuple[float, float, float, int | None, int, int, bool], _Template] = {}
        self._capsule_templates: dict[tuple[float, float, int, int, bool], _Template] = {}

    def build(self, scene: Scene, settings: RenderSettings) -> tuple[bytes, bytes, int, int]:
        triangle_data = array("f")
        line_data = array("f")

        for obj in scene.objects:
            if isinstance(obj, Point3):
                continue
            if isinstance(obj, Line3):
                self._append_line(line_data, obj.start, obj.end, obj.material)
                continue

            if settings.wireframe:
                for triangle in _triangles_for(obj, settings):
                    self._append_triangle_wireframe(line_data, triangle)
                continue

            if isinstance(obj, Sphere):
                self._append_sphere(triangle_data, obj, settings)
            elif isinstance(obj, Bowl):
                self._append_bowl(triangle_data, obj, settings)
            elif isinstance(obj, Capsule):
                self._append_capsule(triangle_data, obj, settings)
            elif isinstance(obj, (Box, Mesh, Plane, Triangle)):
                self._append_triangles(triangle_data, _triangles_for(obj, settings), settings)
            else:
                self._append_triangles(triangle_data, _triangles_for(obj, settings), settings)

        triangle_vertices = len(triangle_data) // _FLOATS_PER_VERTEX
        line_vertices = len(line_data) // _FLOATS_PER_VERTEX
        return triangle_data.tobytes(), line_data.tobytes(), triangle_vertices, line_vertices

    def _append_sphere(self, payload: array, sphere: Sphere, settings: RenderSettings) -> None:
        key = (
            sphere.radius,
            id(sphere.perturbation) if sphere.perturbation is not None else None,
            settings.sphere_segments,
            settings.sphere_rings,
            settings.smooth_shading,
        )
        template = self._sphere_templates.get(key)
        if template is None:
            template = _template_from_triangles(
                Sphere((0.0, 0.0, 0.0), sphere.radius, Material(), sphere.perturbation).to_triangles(
                    segments=settings.sphere_segments,
                    rings=settings.sphere_rings,
                ),
                smooth_shading=settings.smooth_shading,
            )
            self._sphere_templates[key] = template
        _append_template(payload, template, sphere.material, sphere.center, sphere.rotation)

    def _append_bowl(self, payload: array, bowl: Bowl, settings: RenderSettings) -> None:
        key = (
            bowl.radius,
            bowl.depth,
            bowl.thickness,
            id(bowl.perturbation) if bowl.perturbation is not None else None,
            settings.sphere_segments,
            settings.sphere_rings,
            settings.smooth_shading,
        )
        template = self._bowl_templates.get(key)
        if template is None:
            template = _template_from_triangles(
                Bowl(
                    (0.0, 0.0, 0.0),
                    bowl.radius,
                    Material(),
                    bowl.depth,
                    bowl.perturbation,
                    bowl.thickness,
                ).to_triangles(segments=settings.sphere_segments, rings=settings.sphere_rings),
                smooth_shading=settings.smooth_shading,
            )
            self._bowl_templates[key] = template
        _append_template(payload, template, bowl.material, bowl.center, Vec3(0.0, 0.0, 0.0))

    def _append_capsule(self, payload: array, capsule: Capsule, settings: RenderSettings) -> None:
        key = (capsule.radius, capsule.height, settings.sphere_segments, settings.sphere_rings, settings.smooth_shading)
        template = self._capsule_templates.get(key)
        if template is None:
            template = _template_from_triangles(
                Capsule((0.0, 0.0, 0.0), capsule.radius, capsule.height, Material()).to_triangles(
                    segments=settings.sphere_segments,
                    rings=settings.sphere_rings,
                ),
                smooth_shading=settings.smooth_shading,
            )
            self._capsule_templates[key] = template
        _append_template(payload, template, capsule.material, capsule.center, Vec3(0.0, 0.0, 0.0))

    def _append_triangles(self, payload: array, triangles: Iterable[Triangle], settings: RenderSettings) -> None:
        for triangle in triangles:
            normal = triangle.normal()
            smooth = settings.smooth_shading and triangle.has_vertex_normals()
            normal_a = triangle.normal_a if smooth else normal
            normal_b = triangle.normal_b if smooth else normal
            normal_c = triangle.normal_c if smooth else normal
            _append_vertex(payload, triangle.a, normal_a, triangle.material, triangle.uv_a)
            _append_vertex(payload, triangle.b, normal_b, triangle.material, triangle.uv_b)
            _append_vertex(payload, triangle.c, normal_c, triangle.material, triangle.uv_c)

    def _append_triangle_wireframe(self, payload: array, triangle: Triangle) -> None:
        self._append_line(payload, triangle.a, triangle.b, triangle.material)
        self._append_line(payload, triangle.b, triangle.c, triangle.material)
        self._append_line(payload, triangle.c, triangle.a, triangle.material)

    def _append_line(self, payload: array, start: Vec3, end: Vec3, material: Material) -> None:
        normal = Vec3(0.0, 1.0, 0.0)
        _append_vertex(payload, start, normal, material, None, force_emissive=True)
        _append_vertex(payload, end, normal, material, None, force_emissive=True)


class ModernGLLiveRenderer:
    """Render py_3d scenes directly into a pygame/ModernGL window."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        title: str = "py_3d live",
        vsync: bool = True,
        resizable: bool = True,
    ) -> None:
        import moderngl
        import pygame

        self.moderngl = moderngl
        self.pygame = pygame
        pygame.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if resizable:
            flags |= pygame.RESIZABLE
        try:
            pygame.display.set_mode((width, height), flags, vsync=1 if vsync else 0)
        except TypeError:
            pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption(title)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.overlay_program = self.ctx.program(vertex_shader=OVERLAY_VERTEX_SHADER, fragment_shader=OVERLAY_FRAGMENT_SHADER)
        self.builder = LiveSceneBatchBuilder()
        self.stats = LiveFrameStats(0.0, 0.0, 0, 0)
        self.mouse_captured = False
        self.menu = LiveMenu()

    @property
    def size(self) -> tuple[int, int]:
        surface = self.pygame.display.get_surface()
        if surface is None:
            return (1, 1)
        return surface.get_size()

    def set_title(self, title: str) -> None:
        self.pygame.display.set_caption(title)

    def set_mouse_captured(self, captured: bool) -> None:
        self.mouse_captured = bool(captured)
        self.pygame.event.set_grab(self.mouse_captured)
        self.pygame.mouse.set_visible(not self.mouse_captured)
        self.pygame.mouse.get_rel()

    def render(self, scene: Scene, camera: Camera, settings: RenderSettings) -> LiveFrameStats:
        width, height = self.size
        width = max(1, width)
        height = max(1, height)

        build_start = perf_counter()
        triangle_bytes, line_bytes, triangle_vertices, line_vertices = self.builder.build(scene, settings)
        build_seconds = perf_counter() - build_start

        draw_start = perf_counter()
        self.ctx.viewport = (0, 0, width, height)
        red, green, blue = settings.background.to_floats()
        self.ctx.clear(red, green, blue, 1.0, depth=1.0)

        self._set_uniforms(scene, camera, settings, width, height)
        if triangle_bytes:
            self._draw_payload(triangle_bytes, self.moderngl.TRIANGLES)
        if line_bytes:
            try:
                self.ctx.line_width = 1.4
            except Exception:
                pass
            self._draw_payload(line_bytes, self.moderngl.LINES)
        self._draw_scene_overlays(scene, camera, width, height)
        if self.menu.visible:
            self._draw_menu(width, height)
        self.pygame.display.flip()

        self.stats = LiveFrameStats(build_seconds, perf_counter() - draw_start, triangle_vertices, line_vertices)
        return self.stats

    def close(self) -> None:
        self.set_mouse_captured(False)
        self.pygame.quit()

    def _draw_scene_overlays(self, scene: Scene, camera: Camera, width: int, height: int) -> None:
        for bulletin in scene.bulletins:
            if isinstance(bulletin, TextBulletin):
                self._draw_bulletin_texture(
                    bulletin.text,
                    bulletin.position,
                    bulletin.color,
                    bulletin.background,
                    bulletin.padding,
                    bulletin.scale,
                    width,
                    height,
                )
            elif isinstance(bulletin, FloatingTextBulletin):
                projected = camera.project(bulletin.position, width, height)
                if projected is None:
                    continue
                text_width, text_height = draw.text_size(bulletin.text, scale=bulletin.scale)
                total_width = text_width + bulletin.padding * 2
                total_height = text_height + bulletin.padding * 2
                position = (
                    int(projected.x + bulletin.screen_offset[0] - total_width * bulletin.anchor[0]),
                    int(projected.y + bulletin.screen_offset[1] - total_height * bulletin.anchor[1]),
                )
                self._draw_bulletin_texture(
                    bulletin.text,
                    position,
                    bulletin.color,
                    bulletin.background,
                    bulletin.padding,
                    bulletin.scale,
                    width,
                    height,
                )

    def _draw_menu(self, width: int, height: int) -> None:
        lines = [self.menu.title.upper(), ""]
        for index, option in enumerate(self.menu.options):
            prefix = "> " if index == self.menu.selected_index else "  "
            suffix = f" : {option.detail}" if option.detail else ""
            lines.append(f"{prefix}{option.label}{suffix}")
        lines.append("")
        lines.append("ESC RESUMES  ENTER SELECTS")
        text = "\n".join(lines)
        text_width, text_height = draw.text_size(text, scale=2)
        position = (
            max(12, int((width - text_width - 28) * 0.5)),
            max(12, int((height - text_height - 28) * 0.5)),
        )
        self._draw_bulletin_texture(text, position, Color(246, 248, 255), Color(4, 7, 12), 14, 2, width, height)

    def _draw_bulletin_texture(
        self,
        text: str,
        position: tuple[int, int],
        color: Color,
        background: Color | None,
        padding: int,
        scale: int,
        width: int,
        height: int,
    ) -> None:
        image_width, image_height, data = _bulletin_rgba(text, color, background, padding, scale)
        self._draw_overlay_texture(position[0], position[1], image_width, image_height, data, width, height)

    def _draw_overlay_texture(
        self,
        left: int,
        top: int,
        image_width: int,
        image_height: int,
        data: bytes,
        target_width: int,
        target_height: int,
    ) -> None:
        if image_width <= 0 or image_height <= 0:
            return
        x0 = (left / target_width) * 2.0 - 1.0
        x1 = ((left + image_width) / target_width) * 2.0 - 1.0
        y0 = 1.0 - (top / target_height) * 2.0
        y1 = 1.0 - ((top + image_height) / target_height) * 2.0
        vertices = array(
            "f",
            (
                x0,
                y0,
                0.0,
                0.0,
                x1,
                y0,
                1.0,
                0.0,
                x0,
                y1,
                0.0,
                1.0,
                x1,
                y0,
                1.0,
                0.0,
                x1,
                y1,
                1.0,
                1.0,
                x0,
                y1,
                0.0,
                1.0,
            ),
        )
        texture = self.ctx.texture((image_width, image_height), 4, data)
        texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
        vbo = self.ctx.buffer(vertices.tobytes())
        vao = self.ctx.vertex_array(self.overlay_program, [(vbo, _OVERLAY_VERTEX_FORMAT, *_OVERLAY_VERTEX_ATTRIBUTES)])
        self.ctx.disable(self.moderngl.DEPTH_TEST)
        texture.use(0)
        self.overlay_program["u_texture"].value = 0
        vao.render(mode=self.moderngl.TRIANGLES)
        self.ctx.enable(self.moderngl.DEPTH_TEST)
        vao.release()
        vbo.release()
        texture.release()

    def _draw_payload(self, payload: bytes, mode: int) -> None:
        vbo = self.ctx.buffer(payload)
        vao = self.ctx.vertex_array(self.program, [(vbo, _VERTEX_FORMAT, *_VERTEX_ATTRIBUTES)])
        vao.render(mode=mode)
        vao.release()
        vbo.release()

    def _set_uniforms(self, scene: Scene, camera: Camera, settings: RenderSettings, width: int, height: int) -> None:
        right, true_up, forward = camera.basis()
        self.program["u_camera_position"].value = camera.position.as_tuple()
        self.program["u_camera_right"].value = right.as_tuple()
        self.program["u_camera_up"].value = true_up.as_tuple()
        self.program["u_camera_forward"].value = forward.as_tuple()
        self.program["u_focal"].value = 1.0 / tan(radians(camera.fov_degrees) / 2.0)
        self.program["u_aspect"].value = width / height
        self.program["u_near"].value = camera.near
        self.program["u_far"].value = camera.far
        self.program["u_ambient"].value = settings.ambient
        self.program["u_gamma"].value = settings.gamma
        self.program["u_light_wrap"].value = settings.light_wrap
        self.program["u_bounce_light"].value = settings.bounce_light
        self.program["u_tone_mapping"].value = bool(settings.tone_mapping)
        self.program["u_two_sided_lighting"].value = bool(settings.two_sided_lighting)

        kinds: list[int] = []
        vectors: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float]] = []
        intensities: list[float] = []
        for light in scene.lights[:8]:
            if isinstance(light, Sun):
                kinds.append(0)
                vectors.append((-light.direction).as_tuple())
                colors.append(light.color.to_floats())
                intensities.append(float(light.intensity))
            elif isinstance(light, Lamp):
                kinds.append(1)
                vectors.append(light.position.as_tuple())
                colors.append(light.color.to_floats())
                intensities.append(float(light.intensity))

        count = len(kinds)
        while len(kinds) < 8:
            kinds.append(0)
            vectors.append((0.0, 1.0, 0.0))
            colors.append((0.0, 0.0, 0.0))
            intensities.append(0.0)

        self.program["u_light_count"].value = count
        self.program["u_light_kind"].value = tuple(kinds)
        self.program["u_light_vector"].value = tuple(vectors)
        self.program["u_light_color"].value = tuple(colors)
        self.program["u_light_intensity"].value = tuple(intensities)


def _template_from_triangles(triangles: Iterable[Triangle], *, smooth_shading: bool = True) -> _Template:
    vertices: list[tuple[float, float, float, float, float, float, float, float]] = []
    for triangle in triangles:
        face_normal = triangle.normal()
        if smooth_shading:
            normals = (
                triangle.normal_a or face_normal,
                triangle.normal_b or face_normal,
                triangle.normal_c or face_normal,
            )
        else:
            normals = (face_normal, face_normal, face_normal)
        uvs = (
            triangle.uv_a or (0.0, 0.0),
            triangle.uv_b or (0.0, 0.0),
            triangle.uv_c or (0.0, 0.0),
        )
        for point, normal, uv in zip((triangle.a, triangle.b, triangle.c), normals, uvs):
            vertices.append((point.x, point.y, point.z, normal.x, normal.y, normal.z, uv[0], uv[1]))
    return _Template(tuple(vertices))


def _append_template(payload: array, template: _Template, material: Material, center: Vec3, rotation: Vec3) -> None:
    use_rotation = rotation.length_squared() > 1e-12
    material_values = _material_values(material)
    constant_color = material.color.to_floats() if material.texture is None else None
    center_x, center_y, center_z = center.x, center.y, center.z
    if use_rotation:
        rotation_values = (
            cos(rotation.x),
            sin(rotation.x),
            cos(rotation.y),
            sin(rotation.y),
            cos(rotation.z),
            sin(rotation.z),
        )
    else:
        rotation_values = None

    for px, py, pz, nx, ny, nz, u, v in template.vertices:
        if rotation_values is not None:
            px, py, pz = _rotate_components(px, py, pz, rotation_values)
            nx, ny, nz = _rotate_components(nx, ny, nz, rotation_values)
        color = constant_color if constant_color is not None else _material_color_floats(material, (u, v))
        payload.extend(
            (
                center_x + px,
                center_y + py,
                center_z + pz,
                nx,
                ny,
                nz,
                color[0],
                color[1],
                color[2],
                *material_values,
            )
        )


def _append_vertex(
    payload: array,
    position: Vec3,
    normal: Vec3,
    material: Material,
    uv: tuple[float, float] | None,
    *,
    force_emissive: bool = False,
) -> None:
    color = _material_color_floats(material, uv)
    if force_emissive:
        emission = tuple(min(1.0, color[index] + material.emission.to_floats()[index]) for index in range(3))
        material_values = (emission[0], emission[1], emission[2], 0.0, 0.0, 1.0)
    else:
        material_values = _material_values(material)
    payload.extend(
        (
            position.x,
            position.y,
            position.z,
            normal.x,
            normal.y,
            normal.z,
            color[0],
            color[1],
            color[2],
            *material_values,
        )
    )


def _bulletin_rgba(
    text: str,
    color: Color,
    background: Color | None,
    padding: int,
    scale: int,
) -> tuple[int, int, bytes]:
    text_width, text_height = draw.text_size(text, scale=scale)
    width = max(1, text_width + padding * 2)
    height = max(1, text_height + padding * 2)
    fill = background or Color(0, 0, 0)
    buffer = PixelBuffer.new(width, height, fill)
    draw.text(buffer, (padding, padding), text, color, scale=scale)
    payload = bytearray(width * height * 4)
    offset = 0
    for pixel in buffer.pixels:
        is_background = background is not None and pixel == background
        is_empty = background is None and pixel == fill
        alpha = 218 if is_background else (0 if is_empty else 255)
        payload[offset] = pixel.r
        payload[offset + 1] = pixel.g
        payload[offset + 2] = pixel.b
        payload[offset + 3] = alpha
        offset += 4
    return width, height, bytes(payload)


def _material_color(material: Material, uv: tuple[float, float] | None) -> Color:
    if material.texture is not None and uv is not None:
        return material.color_at(uv)
    return material.color


def _material_color_floats(material: Material, uv: tuple[float, float] | None) -> tuple[float, float, float]:
    color = _material_color(material, uv)
    return (color.r / 255.0, color.g / 255.0, color.b / 255.0)


def _material_values(material: Material) -> tuple[float, float, float, float, float, float]:
    emission = material.emission.to_floats()
    specular = material.specular * (1.0 - material.roughness * 0.85) + material.reflectivity * 0.5
    return (
        emission[0],
        emission[1],
        emission[2],
        material.diffuse,
        specular,
        material.shininess,
    )


def _rotate_components(
    x: float,
    y: float,
    z: float,
    rotation_values: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float]:
    cx, sx, cy, sy, cz, sz = rotation_values
    x1 = x
    y1 = y * cx - z * sx
    z1 = y * sx + z * cx
    x2 = x1 * cy + z1 * sy
    y2 = y1
    z2 = -x1 * sy + z1 * cy
    return (x2 * cz - y2 * sz, x2 * sz + y2 * cz, z2)


def _rotate_euler(value: Vec3, rotation: Vec3) -> Vec3:
    cx, sx = cos(rotation.x), sin(rotation.x)
    cy, sy = cos(rotation.y), sin(rotation.y)
    cz, sz = cos(rotation.z), sin(rotation.z)
    x_rotated = Vec3(value.x, value.y * cx - value.z * sx, value.y * sx + value.z * cx)
    y_rotated = Vec3(x_rotated.x * cy + x_rotated.z * sy, x_rotated.y, -x_rotated.x * sy + x_rotated.z * cy)
    return Vec3(y_rotated.x * cz - y_rotated.y * sz, y_rotated.x * sz + y_rotated.y * cz, y_rotated.z)
