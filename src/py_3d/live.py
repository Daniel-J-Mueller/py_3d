"""Interactive OpenGL presentation for live demos.

This module is intentionally optional. Importing it does not create a window;
``ModernGLLiveRenderer`` imports pygame and ModernGL only when constructed.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import cos, radians, sin, tan
from time import perf_counter
from typing import Iterable

from .camera import Camera
from .color import Color
from .lights import Lamp, Sun
from .materials import Material
from .math3d import Vec3
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

    for (int index = 0; index < MAX_LIGHTS; ++index) {
        if (index >= u_light_count) {
            break;
        }

        vec3 light_direction;
        float strength;
        if (u_light_kind[index] == 0) {
            light_direction = normalize(u_light_vector[index]);
            strength = max(0.0, dot(normal, light_direction)) * u_light_intensity[index];
        } else {
            vec3 offset = u_light_vector[index] - v_world_position;
            float distance = max(0.0001, length(offset));
            light_direction = offset / distance;
            float attenuation = 1.0 / (1.0 + distance * distance);
            strength = max(0.0, dot(normal, light_direction)) * u_light_intensity[index] * attenuation;
        }

        vec3 light_color = u_light_color[index];
        diffuse_light += light_color * strength;
        if (specular > 0.0 && strength > 0.0) {
            vec3 halfway = normalize(light_direction + view_direction);
            specular_light += light_color * pow(max(0.0, dot(normal, halfway)), shininess) * strength;
        }
    }

    vec3 color = v_emission + v_color * (u_ambient + diffuse_light * diffuse) + specular_light * specular;
    color = clamp(color, 0.0, 1.0);
    float gamma = max(0.01, u_gamma);
    if (abs(gamma - 1.0) > 0.0001) {
        color = pow(color, vec3(1.0 / gamma));
    }
    frag_color = vec4(color, 1.0);
}
"""


_FLOATS_PER_VERTEX = 15
_VERTEX_FORMAT = "3f 3f 3f 3f 3f"
_VERTEX_ATTRIBUTES = ("in_position", "in_normal", "in_color", "in_emission", "in_material")


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


@dataclass(frozen=True)
class _Template:
    vertices: tuple[tuple[float, float, float, float, float, float, float, float], ...]


class LiveSceneBatchBuilder:
    """Build GPU vertex payloads from py_3d scenes with generated-mesh caches."""

    def __init__(self) -> None:
        self._sphere_templates: dict[tuple[float, int | None, int, int], _Template] = {}
        self._bowl_templates: dict[tuple[float, float, float, int | None, int, int], _Template] = {}
        self._capsule_templates: dict[tuple[float, float, int, int], _Template] = {}

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
        key = (sphere.radius, id(sphere.perturbation) if sphere.perturbation is not None else None, settings.sphere_segments, settings.sphere_rings)
        template = self._sphere_templates.get(key)
        if template is None:
            template = _template_from_triangles(
                Sphere((0.0, 0.0, 0.0), sphere.radius, Material(), sphere.perturbation).to_triangles(
                    segments=settings.sphere_segments,
                    rings=settings.sphere_rings,
                )
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
                ).to_triangles(segments=settings.sphere_segments, rings=settings.sphere_rings)
            )
            self._bowl_templates[key] = template
        _append_template(payload, template, bowl.material, bowl.center, Vec3(0.0, 0.0, 0.0))

    def _append_capsule(self, payload: array, capsule: Capsule, settings: RenderSettings) -> None:
        key = (capsule.radius, capsule.height, settings.sphere_segments, settings.sphere_rings)
        template = self._capsule_templates.get(key)
        if template is None:
            template = _template_from_triangles(
                Capsule((0.0, 0.0, 0.0), capsule.radius, capsule.height, Material()).to_triangles(
                    segments=settings.sphere_segments,
                    rings=settings.sphere_rings,
                )
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
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.builder = LiveSceneBatchBuilder()
        self.stats = LiveFrameStats(0.0, 0.0, 0, 0)

    @property
    def size(self) -> tuple[int, int]:
        surface = self.pygame.display.get_surface()
        if surface is None:
            return (1, 1)
        return surface.get_size()

    def set_title(self, title: str) -> None:
        self.pygame.display.set_caption(title)

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
        self.pygame.display.flip()

        self.stats = LiveFrameStats(build_seconds, perf_counter() - draw_start, triangle_vertices, line_vertices)
        return self.stats

    def close(self) -> None:
        self.pygame.quit()

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


def _template_from_triangles(triangles: Iterable[Triangle]) -> _Template:
    vertices: list[tuple[float, float, float, float, float, float, float, float]] = []
    for triangle in triangles:
        face_normal = triangle.normal()
        normals = (
            triangle.normal_a or face_normal,
            triangle.normal_b or face_normal,
            triangle.normal_c or face_normal,
        )
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
