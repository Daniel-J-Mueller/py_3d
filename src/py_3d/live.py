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
from .hud import HUDAnimation, HUDImage, HUDRect, HUDText, LiveHUD
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
in vec4 in_material;
in vec2 in_texcoord;
in float in_texture_index;

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
out vec4 v_material;
out vec2 v_texcoord;
out float v_texture_index;

void main() {
    vec3 relative = in_position - u_camera_position;
    float camera_x = dot(relative, u_camera_right);
    float camera_y = dot(relative, u_camera_up);
    float camera_z = dot(relative, u_camera_forward);
    float ndc_z = -1.0 + 2.0 * ((camera_z - u_near) / max(0.0001, u_far - u_near));

    gl_Position = vec4(camera_x * u_focal / u_aspect, camera_y * u_focal, ndc_z * camera_z, camera_z);
    v_world_position = in_position;
    v_normal = in_normal;
    v_color = in_color;
    v_emission = in_emission;
    v_material = in_material;
    v_texcoord = in_texcoord;
    v_texture_index = in_texture_index;
}
"""


FRAGMENT_SHADER = """
#version 330

const int MAX_LIGHTS = 8;

in vec3 v_world_position;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_emission;
in vec4 v_material;
in vec2 v_texcoord;
in float v_texture_index;

uniform vec3 u_camera_position;
uniform float u_ambient;
uniform float u_gamma;
uniform float u_light_wrap;
uniform float u_bounce_light;
uniform bool u_tone_mapping;
uniform bool u_two_sided_lighting;
uniform int u_reflection_bounces;
uniform int u_light_count;
uniform int u_light_kind[MAX_LIGHTS];
uniform vec3 u_light_vector[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform float u_light_intensity[MAX_LIGHTS];
uniform bool u_use_textures;
uniform int u_texture_count;
uniform sampler2DArray u_textures;

out vec4 frag_color;

vec3 reflectionProbe(vec3 direction) {
    vec3 ray = normalize(direction);
    float sky = max(0.0, ray.y);
    float floor_light = max(0.0, -ray.y);
    float horizon = max(0.0, 1.0 - abs(ray.y));
    float glint = pow(max(0.0, dot(ray, normalize(vec3(-0.35, 0.72, -0.58)))), 28.0);
    return vec3(
        0.05 + sky * 0.34 + floor_light * 0.16 + horizon * 0.09 + glint * 0.85,
        0.06 + sky * 0.42 + floor_light * 0.22 + horizon * 0.11 + glint * 0.78,
        0.08 + sky * 0.56 + floor_light * 0.28 + horizon * 0.14 + glint * 0.62
    );
}

vec3 traceReflection(vec3 direction, int bounces) {
    vec3 ray = normalize(direction);
    vec3 total = vec3(0.0);
    float energy = 1.0;
    float normalizer = 0.0;
    int steps = max(1, bounces);
    for (int index = 0; index < 6; ++index) {
        if (index >= steps) {
            break;
        }
        total += reflectionProbe(ray) * energy;
        normalizer += energy;
        if (ray.y < 0.0) {
            ray = normalize(vec3(ray.x * 0.82, -ray.y * 0.78 + 0.16, ray.z * 0.82));
        } else {
            ray = normalize(vec3(ray.x * 0.76 - ray.z * 0.18, ray.y * 0.54 - 0.12, ray.z * 0.76 + ray.x * 0.18));
        }
        energy *= 0.48;
    }
    return total / max(0.0001, normalizer);
}

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
    float reflectivity = clamp(v_material.w, 0.0, 1.0);
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

    vec3 base_color = v_color;
    int texture_index = int(v_texture_index + 0.5);
    if (u_use_textures && texture_index >= 0 && texture_index < u_texture_count) {
        vec2 uv = clamp(v_texcoord, vec2(0.0), vec2(1.0));
        base_color *= texture(u_textures, vec3(uv, float(texture_index))).rgb;
    }

    vec3 color = v_emission + base_color * (u_ambient + diffuse_light * diffuse) + specular_light * specular;
    if (u_reflection_bounces > 0 && reflectivity > 0.0) {
        vec3 reflection_direction = reflect(-view_direction, normal);
        vec3 reflection_color = traceReflection(reflection_direction, u_reflection_bounces);
        float reflection_mix = clamp(reflectivity * (0.38 + min(float(u_reflection_bounces), 4.0) * 0.12), 0.0, 0.92);
        color = mix(color, reflection_color, reflection_mix);
    }
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


_FLOATS_PER_VERTEX = 19
_VERTEX_FORMAT = "3f 3f 3f 3f 4f 2f 1f"
_VERTEX_ATTRIBUTES = ("in_position", "in_normal", "in_color", "in_emission", "in_material", "in_texcoord", "in_texture_index")
_MAX_TEXTURES = 8
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
    render_position: Vec3 | None = None
    render_yaw_degrees: float | None = None
    render_pitch_degrees: float | None = None

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

    def _forward_for(self, yaw_degrees: float, pitch_degrees: float) -> Vec3:
        yaw = radians(yaw_degrees)
        pitch = radians(pitch_degrees)
        return Vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)).normalized(Vec3(0.0, 0.0, 1.0))

    @property
    def forward(self) -> Vec3:
        return self._forward_for(self.yaw_degrees, self.pitch_degrees)

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
        if "space" in keys:
            direction = direction + Vec3(0.0, 1.0, 0.0)
        if "ctrl" in keys:
            direction = direction - Vec3(0.0, 1.0, 0.0)
        if direction.length_squared() > 0.0:
            self.position = self.position + direction.normalized() * (self.speed * dt)

    def camera(self) -> Camera:
        return Camera(position=self.position, target=self.position + self.forward, fov_degrees=self.fov_degrees)

    def smoothed_camera(self, dt: float, *, responsiveness: float = 22.0) -> Camera:
        """Return a camera eased toward the current control state for rendering."""

        if self.render_position is None or self.render_yaw_degrees is None or self.render_pitch_degrees is None:
            self.render_position = self.position
            self.render_yaw_degrees = self.yaw_degrees
            self.render_pitch_degrees = self.pitch_degrees
            return self.camera()

        alpha = max(0.0, min(1.0, dt * responsiveness))
        self.render_position = self.render_position + (self.position - self.render_position) * alpha
        self.render_yaw_degrees = _lerp_degrees(self.render_yaw_degrees, self.yaw_degrees, alpha)
        self.render_pitch_degrees = self.render_pitch_degrees + (self.pitch_degrees - self.render_pitch_degrees) * alpha
        forward = self._forward_for(self.render_yaw_degrees, self.render_pitch_degrees)
        return Camera(position=self.render_position, target=self.render_position + forward, fov_degrees=self.fov_degrees)

    def reset_smoothing(self) -> None:
        self.render_position = self.position
        self.render_yaw_degrees = self.yaw_degrees
        self.render_pitch_degrees = self.pitch_degrees


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
        self.active_textures: list[PixelBuffer] = []
        self._frame_texture_slots: dict[int, int] = {}

    def build(self, scene: Scene, settings: RenderSettings) -> tuple[bytes, bytes, int, int]:
        self.active_textures = []
        self._frame_texture_slots = {}
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
        _append_template(payload, template, sphere.material, sphere.center, sphere.rotation, self.texture_index_for(sphere.material))

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
        _append_template(payload, template, bowl.material, bowl.center, Vec3(0.0, 0.0, 0.0), self.texture_index_for(bowl.material))

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
        _append_template(payload, template, capsule.material, capsule.center, Vec3(0.0, 0.0, 0.0), self.texture_index_for(capsule.material))

    def _append_triangles(self, payload: array, triangles: Iterable[Triangle], settings: RenderSettings) -> None:
        for triangle in triangles:
            normal = triangle.normal()
            smooth = settings.smooth_shading and triangle.has_vertex_normals()
            normal_a = triangle.normal_a if smooth else normal
            normal_b = triangle.normal_b if smooth else normal
            normal_c = triangle.normal_c if smooth else normal
            texture_index = self.texture_index_for(triangle.material) if triangle.has_texture_coordinates() else -1
            _append_vertex(payload, triangle.a, normal_a, triangle.material, triangle.uv_a, texture_index=texture_index)
            _append_vertex(payload, triangle.b, normal_b, triangle.material, triangle.uv_b, texture_index=texture_index)
            _append_vertex(payload, triangle.c, normal_c, triangle.material, triangle.uv_c, texture_index=texture_index)

    def _append_triangle_wireframe(self, payload: array, triangle: Triangle) -> None:
        self._append_line(payload, triangle.a, triangle.b, triangle.material)
        self._append_line(payload, triangle.b, triangle.c, triangle.material)
        self._append_line(payload, triangle.c, triangle.a, triangle.material)

    def _append_line(self, payload: array, start: Vec3, end: Vec3, material: Material) -> None:
        normal = Vec3(0.0, 1.0, 0.0)
        _append_vertex(payload, start, normal, material, None, force_emissive=True)
        _append_vertex(payload, end, normal, material, None, force_emissive=True)

    def texture_index_for(self, material: Material) -> int:
        texture = material.texture
        if texture is None:
            return -1
        key = id(texture)
        existing = self._frame_texture_slots.get(key)
        if existing is not None:
            return existing
        if len(self.active_textures) >= _MAX_TEXTURES:
            return -1
        index = len(self.active_textures)
        self._frame_texture_slots[key] = index
        self.active_textures.append(texture)
        return index


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
        self._display_flags = pygame.OPENGL | pygame.DOUBLEBUF
        if resizable:
            self._display_flags |= pygame.RESIZABLE
        self._vsync = bool(vsync)
        try:
            pygame.display.set_mode((width, height), self._display_flags, vsync=1 if self._vsync else 0)
        except TypeError:
            pygame.display.set_mode((width, height), self._display_flags)
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
        self.hud = LiveHUD()
        self.show_crosshair = True
        self._texture_array = None
        self._texture_key: tuple[int, tuple[int, ...]] | None = None

    @property
    def size(self) -> tuple[int, int]:
        surface = self.pygame.display.get_surface()
        if surface is None:
            return (1, 1)
        return surface.get_size()

    def set_title(self, title: str) -> None:
        self.pygame.display.set_caption(title)

    def resize(self, width: int, height: int) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        try:
            self.pygame.display.set_mode((width, height), self._display_flags, vsync=1 if self._vsync else 0)
        except TypeError:
            self.pygame.display.set_mode((width, height), self._display_flags)
        self.ctx.viewport = (0, 0, width, height)

    def handle_resize_event(self, event) -> bool:
        resize_types = {
            getattr(self.pygame, "VIDEORESIZE", None),
            getattr(self.pygame, "WINDOWRESIZED", None),
            getattr(self.pygame, "WINDOWSIZECHANGED", None),
        }
        if event.type not in resize_types:
            return False
        width = getattr(event, "w", None) or getattr(event, "x", None)
        height = getattr(event, "h", None) or getattr(event, "y", None)
        if width is None or height is None:
            try:
                width, height = self.pygame.display.get_window_size()
            except Exception:
                width, height = self.size
        self.resize(width, height)
        return True

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
        self._set_live_textures(settings)
        if triangle_bytes:
            self._draw_payload(triangle_bytes, self.moderngl.TRIANGLES)
        if line_bytes:
            try:
                self.ctx.line_width = 1.4
            except Exception:
                pass
            self._draw_payload(line_bytes, self.moderngl.LINES)
        self._draw_scene_overlays(scene, camera, width, height)
        self._draw_hud(width, height)
        if self.show_crosshair and not self.menu.visible:
            self._draw_crosshair(width, height)
        if self.menu.visible:
            self._draw_menu(width, height)
        self.pygame.display.flip()

        self.stats = LiveFrameStats(build_seconds, perf_counter() - draw_start, triangle_vertices, line_vertices)
        return self.stats

    def close(self) -> None:
        self.set_mouse_captured(False)
        if self._texture_array is not None:
            self._texture_array.release()
            self._texture_array = None
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

    def _draw_crosshair(self, width: int, height: int) -> None:
        image_width, image_height, data = _crosshair_rgba(Color(235, 244, 255))
        self._draw_overlay_texture((width - image_width) // 2, (height - image_height) // 2, image_width, image_height, data, width, height)

    def _draw_hud(self, width: int, height: int) -> None:
        if not self.hud.visible:
            return
        seconds = perf_counter()
        for element in self.hud.elements:
            if isinstance(element, HUDRect):
                image_width, image_height, data = _solid_rgba(element.size, element.color, element.alpha)
                self._draw_overlay_texture(element.position[0], element.position[1], image_width, image_height, data, width, height)
            elif isinstance(element, HUDText):
                image_width, image_height, data = _hud_text_rgba(element)
                self._draw_overlay_texture(element.position[0], element.position[1], image_width, image_height, data, width, height)
            elif isinstance(element, HUDImage):
                image = element.image if element.scale <= 1 else element.image.resized_nearest(element.image.width * element.scale, element.image.height * element.scale)
                image_width, image_height, data = _pixelbuffer_rgba(image, element.alpha)
                self._draw_overlay_texture(element.position[0], element.position[1], image_width, image_height, data, width, height)
            elif isinstance(element, HUDAnimation):
                frame = element.frame_at(seconds)
                if frame is None:
                    continue
                image = frame if element.scale <= 1 else frame.resized_nearest(frame.width * element.scale, frame.height * element.scale)
                image_width, image_height, data = _pixelbuffer_rgba(image, element.alpha)
                self._draw_overlay_texture(element.position[0], element.position[1], image_width, image_height, data, width, height)

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

    def _set_live_textures(self, settings: RenderSettings) -> None:
        textures = self.builder.active_textures[:_MAX_TEXTURES]
        if not textures:
            self.program["u_use_textures"].value = False
            self.program["u_texture_count"].value = 0
            return
        texture_size = settings.texture_size
        key = (texture_size, tuple(id(texture) for texture in textures))
        if key != self._texture_key:
            if self._texture_array is not None:
                self._texture_array.release()
                self._texture_array = None
            payload = bytearray()
            for texture in textures:
                prepared = texture if texture.width == texture_size and texture.height == texture_size else texture.resized_nearest(texture_size, texture_size)
                payload.extend(prepared.to_rgb_bytes())
            try:
                self._texture_array = self.ctx.texture_array((texture_size, texture_size, len(textures)), 3, bytes(payload))
                self._texture_array.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
                self._texture_array.repeat_x = False
                self._texture_array.repeat_y = False
                self._texture_key = key
            except Exception:
                self._texture_array = None
                self._texture_key = None
        if self._texture_array is None:
            self.program["u_use_textures"].value = False
            self.program["u_texture_count"].value = 0
            return
        self._texture_array.use(0)
        self.program["u_textures"].value = 0
        self.program["u_use_textures"].value = True
        self.program["u_texture_count"].value = len(textures)

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
        self.program["u_reflection_bounces"].value = int(settings.reflection_bounces)

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


def _append_template(
    payload: array,
    template: _Template,
    material: Material,
    center: Vec3,
    rotation: Vec3,
    texture_index: int,
) -> None:
    use_rotation = rotation.length_squared() > 1e-12
    material_values = _material_values(material)
    base_color = material.color.to_floats()
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
        color = base_color if texture_index >= 0 else _material_color_floats(material, (u, v))
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
                u,
                v,
                float(texture_index),
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
    texture_index: int = -1,
) -> None:
    if force_emissive:
        texture_index = -1
    color = material.color.to_floats() if texture_index >= 0 and uv is not None else _material_color_floats(material, uv)
    if force_emissive:
        emission = tuple(min(1.0, color[index] + material.emission.to_floats()[index]) for index in range(3))
        material_values = (emission[0], emission[1], emission[2], 0.0, 0.0, 1.0, 0.0)
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
            uv[0] if uv is not None else 0.0,
            uv[1] if uv is not None else 0.0,
            float(texture_index if uv is not None else -1),
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


def _crosshair_rgba(color: Color) -> tuple[int, int, bytes]:
    size = 19
    center = size // 2
    payload = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            horizontal = y == center and (2 <= x <= 7 or 11 <= x <= 16)
            vertical = x == center and (2 <= y <= 7 or 11 <= y <= 16)
            dot = abs(x - center) <= 1 and abs(y - center) <= 1
            if not (horizontal or vertical or dot):
                continue
            offset = (y * size + x) * 4
            payload[offset] = color.r
            payload[offset + 1] = color.g
            payload[offset + 2] = color.b
            payload[offset + 3] = 230 if dot else 190
    return size, size, bytes(payload)


def _solid_rgba(size: tuple[int, int], color: Color | tuple[int, int, int], alpha: float) -> tuple[int, int, bytes]:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    fill = Color.from_value(color)
    payload = bytearray(width * height * 4)
    opacity = _alpha_byte(alpha)
    for offset in range(0, len(payload), 4):
        payload[offset] = fill.r
        payload[offset + 1] = fill.g
        payload[offset + 2] = fill.b
        payload[offset + 3] = opacity
    return width, height, bytes(payload)


def _hud_text_rgba(element: HUDText) -> tuple[int, int, bytes]:
    text_width, text_height = draw.text_size(element.text, scale=element.scale)
    width = max(1, text_width + element.padding * 2)
    height = max(1, text_height + element.padding * 2)
    transparent_key = Color(0, 0, 0)
    fill = Color.from_value(element.background) if element.background is not None else transparent_key
    buffer = PixelBuffer.new(width, height, fill)
    draw.text(buffer, (element.padding, element.padding), element.text, element.color, scale=element.scale)
    background_alpha = _alpha_byte(element.alpha) if element.background is not None else 0
    foreground_alpha = _alpha_byte(element.alpha)
    payload = bytearray(width * height * 4)
    offset = 0
    for pixel in buffer.pixels:
        payload[offset] = pixel.r
        payload[offset + 1] = pixel.g
        payload[offset + 2] = pixel.b
        payload[offset + 3] = background_alpha if pixel == fill else foreground_alpha
        offset += 4
    return width, height, bytes(payload)


def _pixelbuffer_rgba(buffer: PixelBuffer, alpha: float) -> tuple[int, int, bytes]:
    payload = bytearray(buffer.width * buffer.height * 4)
    opacity = _alpha_byte(alpha)
    offset = 0
    for pixel in buffer.pixels:
        payload[offset] = pixel.r
        payload[offset + 1] = pixel.g
        payload[offset + 2] = pixel.b
        payload[offset + 3] = opacity
        offset += 4
    return buffer.width, buffer.height, bytes(payload)


def _alpha_byte(alpha: float) -> int:
    return max(0, min(255, int(float(alpha) * 255)))


def _material_color(material: Material, uv: tuple[float, float] | None) -> Color:
    if material.texture is not None and uv is not None:
        return material.color_at(uv)
    return material.color


def _material_color_floats(material: Material, uv: tuple[float, float] | None) -> tuple[float, float, float]:
    color = _material_color(material, uv)
    return (color.r / 255.0, color.g / 255.0, color.b / 255.0)


def _material_values(material: Material) -> tuple[float, float, float, float, float, float, float]:
    emission = material.emission.to_floats()
    specular = material.specular * (1.0 - material.roughness * 0.85) + material.reflectivity * 0.5
    return (
        emission[0],
        emission[1],
        emission[2],
        material.diffuse,
        specular,
        material.shininess,
        material.reflectivity,
    )


def _lerp_degrees(start: float, end: float, amount: float) -> float:
    delta = (end - start + 180.0) % 360.0 - 180.0
    return start + delta * amount


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
