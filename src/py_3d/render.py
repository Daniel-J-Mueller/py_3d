"""Rendering backends and the reference CPU renderer."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, floor, radians, sin, sqrt, tan, tau
from typing import Protocol, runtime_checkable
import weakref

from .buffer import DepthBuffer, PixelBuffer
from .camera import Camera, ProjectedPoint
from .color import Color
from . import draw
from .lights import Lamp, Sun
from .math3d import Vec3, clamp
from .overlays import FloatingTextBulletin, TextBulletin
from .primitives import BlobSurface, Bowl, Box, Capsule, Line3, Mesh, Plane, Point3, Sphere, Triangle
from .scene import Scene


@dataclass(frozen=True)
class RenderSettings:
    """Settings for one offline or real-time render pass."""

    width: int = 640
    height: int = 480
    background: Color | tuple[int, int, int] = Color(0, 0, 0)
    ambient: float = 0.0
    gamma: float = 1.0
    light_wrap: float = 0.0
    bounce_light: float = 0.0
    tone_mapping: bool = False
    cull_backfaces: bool = False
    wireframe: bool = False
    smooth_shading: bool = False
    two_sided_lighting: bool = True
    ray_traced_shadows: bool = False
    reflection_bounces: int = 0
    shadow_bias: float = 1e-4
    shadow_samples: int = 1
    shadow_softness: float = 0.0
    edge_highlight: bool = False
    edge_highlight_threshold_degrees: float = 35.0
    edge_highlight_color: Color | tuple[int, int, int] = Color(0, 0, 0)
    edge_highlight_depth_bias: float = 0.002
    max_render_distance: float | None = None
    sphere_segments: int = 16
    sphere_rings: int = 8
    texture_size: int = 256

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("render dimensions must be positive")
        object.__setattr__(self, "background", Color.from_value(self.background))
        object.__setattr__(self, "ambient", clamp(float(self.ambient), 0.0, 1.0))
        object.__setattr__(self, "gamma", max(0.01, float(self.gamma)))
        object.__setattr__(self, "light_wrap", clamp(float(self.light_wrap), 0.0, 1.0))
        object.__setattr__(self, "bounce_light", clamp(float(self.bounce_light), 0.0, 1.0))
        object.__setattr__(self, "reflection_bounces", max(0, int(self.reflection_bounces)))
        object.__setattr__(self, "shadow_bias", max(0.0, float(self.shadow_bias)))
        object.__setattr__(self, "shadow_samples", max(1, int(self.shadow_samples)))
        object.__setattr__(self, "shadow_softness", max(0.0, float(self.shadow_softness)))
        object.__setattr__(self, "edge_highlight_threshold_degrees", clamp(float(self.edge_highlight_threshold_degrees), 0.0, 180.0))
        object.__setattr__(self, "edge_highlight_color", Color.from_value(self.edge_highlight_color))
        object.__setattr__(self, "edge_highlight_depth_bias", max(0.0, float(self.edge_highlight_depth_bias)))
        if self.max_render_distance is not None:
            object.__setattr__(self, "max_render_distance", max(0.0, float(self.max_render_distance)))
        object.__setattr__(self, "texture_size", max(16, int(self.texture_size)))


@runtime_checkable
class Renderer(Protocol):
    """Protocol implemented by CPU and future GPU renderers."""

    name: str
    backend: str

    def render(
        self,
        scene: Scene,
        camera: Camera,
        settings: RenderSettings,
        target: PixelBuffer | None = None,
    ) -> PixelBuffer:
        ...


@dataclass
class RenderEngine:
    """Thin rendering engine that delegates work to a selected backend."""

    renderer: Renderer | None = None

    def __post_init__(self) -> None:
        if self.renderer is None:
            self.renderer = CPURenderer()

    def render(
        self,
        scene: Scene,
        camera: Camera,
        settings: RenderSettings | None = None,
        target: PixelBuffer | None = None,
    ) -> PixelBuffer:
        active_settings = settings or RenderSettings()
        active_scene = scene
        if getattr(scene, "portals", None):
            from .portal import scene_with_portal_textures

            active_scene = scene_with_portal_textures(scene, camera, active_settings, self.renderer)
        return self.renderer.render(active_scene, camera, active_settings, target)


class CPURenderer:
    """Pure-Python reference renderer.

    This backend is intentionally straightforward. It is the correctness target
    for future accelerated CPU or GPU renderers.
    """

    name = "Reference CPU Renderer"
    backend = "cpu"

    def __init__(self, *, cache_static_geometry: bool = True) -> None:
        self.cache_static_geometry = cache_static_geometry
        self._triangle_cache: dict[tuple[type, int, int, int], tuple[weakref.ref, tuple[_PreparedTriangle, ...]]] = {}

    def clear_cache(self) -> None:
        self._triangle_cache.clear()

    def render(
        self,
        scene: Scene,
        camera: Camera,
        settings: RenderSettings,
        target: PixelBuffer | None = None,
    ) -> PixelBuffer:
        buffer = target or PixelBuffer.new(settings.width, settings.height, settings.background)
        if buffer.width != settings.width or buffer.height != settings.height:
            raise ValueError("target buffer dimensions must match render settings")
        buffer.clear(settings.background)
        depth = DepthBuffer.new(settings.width, settings.height)
        projector = _CameraProjector(camera, settings.width, settings.height)
        shadow_triangles = self._shadow_triangles(scene, settings) if settings.ray_traced_shadows else None

        for obj in scene.objects:
            if isinstance(obj, Point3):
                self._draw_point(buffer, depth, projector, settings, obj)
            elif isinstance(obj, Line3):
                self._draw_line(buffer, depth, projector, settings, obj)
            else:
                triangles = self._prepared_triangles_for(obj, settings)
                for triangle in triangles:
                    if _triangle_culled_by_distance(triangle, camera, settings):
                        continue
                    if settings.wireframe:
                        self._draw_triangle_wireframe(buffer, depth, projector, settings, triangle.triangle)
                    else:
                        self._draw_triangle(buffer, depth, scene, camera, projector, settings, triangle, shadow_triangles)
                if settings.edge_highlight and not settings.wireframe:
                    self._draw_edge_highlights(buffer, depth, projector, settings, triangles)
        for bulletin in scene.bulletins:
            if isinstance(bulletin, TextBulletin):
                _draw_text_bulletin(buffer, bulletin)
            elif isinstance(bulletin, FloatingTextBulletin):
                _draw_floating_text_bulletin(buffer, camera, bulletin)
        return buffer

    def _shadow_triangles(self, scene: Scene, settings: RenderSettings) -> tuple[Triangle, ...]:
        triangles: list[Triangle] = []
        for obj in scene.objects:
            if isinstance(obj, (Point3, Line3)):
                continue
            triangles.extend(prepared.triangle for prepared in self._prepared_triangles_for(obj, settings))
        return tuple(triangles)

    def _prepared_triangles_for(self, obj, settings: RenderSettings) -> tuple["_PreparedTriangle", ...]:
        if isinstance(obj, Triangle):
            return (_PreparedTriangle(obj, obj.center(), obj.normal()),)
        if not self.cache_static_geometry:
            return _prepare_triangles(_triangles_for(obj, settings))

        key = _cache_key_for(obj, settings)
        if key is None:
            return _prepare_triangles(_triangles_for(obj, settings))

        cached = self._triangle_cache.get(key)
        if cached is not None:
            owner_ref, triangles = cached
            if owner_ref() is obj:
                return triangles

        triangles = _prepare_triangles(_triangles_for(obj, settings))
        try:
            self._triangle_cache[key] = (weakref.ref(obj), triangles)
        except TypeError:
            return triangles
        return triangles

    def _draw_point(
        self,
        buffer: PixelBuffer,
        depth: DepthBuffer,
        projector: "_CameraProjector",
        settings: RenderSettings,
        point: Point3,
    ) -> None:
        projected = projector.project(point.position)
        if projected is None:
            return
        x = round(projected.x)
        y = round(projected.y)
        if 0 <= x < buffer.width and 0 <= y < buffer.height:
            index = y * buffer.width + x
            if projected.depth < depth.values[index]:
                depth.values[index] = projected.depth
                buffer.pixels[index] = _apply_gamma(point.material.color, settings.gamma)

    def _draw_line(
        self,
        buffer: PixelBuffer,
        depth: DepthBuffer,
        projector: "_CameraProjector",
        settings: RenderSettings,
        line: Line3,
    ) -> None:
        start = projector.project(line.start)
        end = projector.project(line.end)
        if start is None or end is None:
            return
        self._draw_projected_line(buffer, depth, start, end, _apply_gamma(line.material.color, settings.gamma))

    def _draw_triangle_wireframe(
        self,
        buffer: PixelBuffer,
        depth: DepthBuffer,
        projector: "_CameraProjector",
        settings: RenderSettings,
        triangle: Triangle,
    ) -> None:
        a = projector.project(triangle.a)
        b = projector.project(triangle.b)
        c = projector.project(triangle.c)
        if a is None or b is None or c is None:
            return
        color = _apply_gamma(triangle.material.color, settings.gamma)
        self._draw_projected_line(buffer, depth, a, b, color)
        self._draw_projected_line(buffer, depth, b, c, color)
        self._draw_projected_line(buffer, depth, c, a, color)

    def _draw_projected_line(
        self,
        buffer: PixelBuffer,
        depth: DepthBuffer,
        start: ProjectedPoint,
        end: ProjectedPoint,
        color: Color,
        *,
        depth_bias: float = 0.0,
    ) -> None:
        dx = end.x - start.x
        dy = end.y - start.y
        steps = max(abs(dx), abs(dy), 1.0)
        width = buffer.width
        height = buffer.height
        pixels = buffer.pixels
        depth_values = depth.values
        for index in range(int(steps) + 1):
            amount = index / steps
            x = round(start.x + dx * amount)
            y = round(start.y + dy * amount)
            z = start.depth + (end.depth - start.depth) * amount - depth_bias
            if 0 <= x < width and 0 <= y < height:
                buffer_index = y * width + x
                if z < depth_values[buffer_index]:
                    depth_values[buffer_index] = z
                    pixels[buffer_index] = color

    def _draw_triangle(
        self,
        buffer: PixelBuffer,
        depth: DepthBuffer,
        scene: Scene,
        camera: Camera,
        projector: "_CameraProjector",
        settings: RenderSettings,
        prepared: "_PreparedTriangle",
        shadow_triangles: tuple[Triangle, ...] | None = None,
    ) -> None:
        triangle = prepared.triangle
        a = projector.project(triangle.a)
        b = projector.project(triangle.b)
        c = projector.project(triangle.c)
        if a is None or b is None or c is None:
            return

        normal = prepared.normal
        center = prepared.center
        view_direction = (camera.position - center).normalized(Vec3(0.0, 0.0, -1.0))
        facing = normal.dot(view_direction)
        if settings.cull_backfaces and facing <= 0.0:
            return
        if settings.two_sided_lighting and facing < 0.0:
            normal = -normal

        lighting = _lighting_channels(scene, center, normal, view_direction, triangle.material, settings, triangle, shadow_triangles)
        color = triangle.material.shade(lighting.diffuse, ambient=settings.ambient, specular_light=lighting.specular)
        use_texture = triangle.material.texture is not None and triangle.has_texture_coordinates()
        use_smooth_shading = settings.smooth_shading and triangle.has_vertex_normals()
        min_x = max(0, floor(min(a.x, b.x, c.x)))
        max_x = min(buffer.width - 1, ceil(max(a.x, b.x, c.x)))
        min_y = max(0, floor(min(a.y, b.y, c.y)))
        max_y = min(buffer.height - 1, ceil(max(a.y, b.y, c.y)))

        area = _edge(a.x, a.y, b.x, b.y, c.x, c.y)
        if abs(area) < 1e-12:
            return
        inv_area = 1.0 / area
        ax, ay, az = a.x, a.y, a.depth
        bx, by, bz = b.x, b.y, b.depth
        cx, cy, cz = c.x, c.y, c.depth
        width = buffer.width
        pixels = buffer.pixels
        depth_values = depth.values

        for y in range(min_y, max_y + 1):
            row_index = y * width
            py = y + 0.5
            for x in range(min_x, max_x + 1):
                px = x + 0.5
                w0 = ((px - bx) * (cy - by) - (py - by) * (cx - bx)) * inv_area
                w1 = ((px - cx) * (ay - cy) - (py - cy) * (ax - cx)) * inv_area
                w2 = ((px - ax) * (by - ay) - (py - ay) * (bx - ax)) * inv_area
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                    continue
                z = w0 * az + w1 * bz + w2 * cz
                index = row_index + x
                if z < depth_values[index]:
                    depth_values[index] = z
                    active_lighting = lighting
                    if use_smooth_shading:
                        world_point = triangle.a * w0 + triangle.b * w1 + triangle.c * w2
                        pixel_normal = _interpolate_normal(triangle, w0, w1, w2, normal)
                        pixel_view_direction = (camera.position - world_point).normalized(view_direction)
                        pixel_facing = pixel_normal.dot(pixel_view_direction)
                        if settings.two_sided_lighting and pixel_facing < 0.0:
                            pixel_normal = -pixel_normal
                        shade_normal = pixel_normal
                        shade_view_direction = pixel_view_direction
                        active_lighting = _lighting_channels(
                            scene,
                            world_point,
                            pixel_normal,
                            pixel_view_direction,
                            triangle.material,
                            settings,
                            triangle,
                            shadow_triangles,
                        )
                    else:
                        shade_normal = normal
                        shade_view_direction = view_direction
                    if use_texture:
                        uv = _interpolate_uv(triangle, w0, w1, w2)
                        texture_color = triangle.material.color_at(uv)
                        shaded = triangle.material.shade(
                            active_lighting.diffuse,
                            ambient=settings.ambient,
                            base_color=texture_color,
                            specular_light=active_lighting.specular,
                        )
                        shaded = _apply_ray_traced_reflection(shaded, triangle.material, shade_normal, shade_view_direction, settings)
                        pixels[index] = _apply_gamma(_apply_surface_attributes(shaded, triangle.material, x, y, z), settings.gamma)
                    elif use_smooth_shading:
                        shaded = triangle.material.shade(
                            active_lighting.diffuse,
                            ambient=settings.ambient,
                            specular_light=active_lighting.specular,
                        )
                        shaded = _apply_ray_traced_reflection(shaded, triangle.material, shade_normal, shade_view_direction, settings)
                        pixels[index] = _apply_gamma(_apply_surface_attributes(shaded, triangle.material, x, y, z), settings.gamma)
                    else:
                        shaded = _apply_ray_traced_reflection(color, triangle.material, shade_normal, shade_view_direction, settings)
                        pixels[index] = _apply_gamma(_apply_surface_attributes(shaded, triangle.material, x, y, z), settings.gamma)

    def _draw_edge_highlights(
        self,
        buffer: PixelBuffer,
        depth: DepthBuffer,
        projector: "_CameraProjector",
        settings: RenderSettings,
        triangles: tuple["_PreparedTriangle", ...],
    ) -> None:
        for start_point, end_point in _sharp_edges_for(triangles, settings.edge_highlight_threshold_degrees):
            start = projector.project(start_point)
            end = projector.project(end_point)
            if start is None or end is None:
                continue
            self._draw_projected_line(
                buffer,
                depth,
                start,
                end,
                settings.edge_highlight_color,
                depth_bias=settings.edge_highlight_depth_bias,
            )


@dataclass(frozen=True)
class _PreparedTriangle:
    triangle: Triangle
    center: Vec3
    normal: Vec3


@dataclass(frozen=True)
class _LightingChannels:
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float]


@dataclass(frozen=True)
class _CameraProjector:
    camera: Camera
    width: int
    height: int

    def __post_init__(self) -> None:
        right, true_up, forward = self.camera.basis()
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "true_up", true_up)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "aspect", self.width / self.height)
        object.__setattr__(self, "focal", 1.0 / tan(radians(self.camera.fov_degrees) / 2.0))
        object.__setattr__(self, "half_width", 0.5 * (self.width - 1))
        object.__setattr__(self, "half_height", 0.5 * (self.height - 1))

    def project(self, point: Vec3 | tuple[float, float, float]) -> ProjectedPoint | None:
        relative = point - self.camera.position if isinstance(point, Vec3) else Vec3(*point) - self.camera.position
        camera_x = relative.dot(self.right)
        camera_y = relative.dot(self.true_up)
        camera_z = relative.dot(self.forward)
        if camera_z < self.camera.near or camera_z > self.camera.far:
            return None

        ndc_x = (camera_x * self.focal / self.aspect) / camera_z
        ndc_y = (camera_y * self.focal) / camera_z
        return ProjectedPoint(
            (ndc_x + 1.0) * self.half_width,
            (1.0 - ndc_y) * self.half_height,
            camera_z,
        )


def _edge(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def _cache_key_for(obj, settings: RenderSettings) -> tuple[type, int, int, int] | None:
    if isinstance(obj, (BlobSurface, Bowl, Capsule, Sphere)):
        return (type(obj), id(obj), settings.sphere_segments, settings.sphere_rings)
    if isinstance(obj, (Box, Mesh, Plane)):
        return (type(obj), id(obj), 0, 0)
    return None


def _triangles_for(obj, settings: RenderSettings) -> tuple[Triangle, ...]:
    if isinstance(obj, Triangle):
        return (obj,)
    if isinstance(obj, Mesh):
        return obj.to_triangles()
    if isinstance(obj, (BlobSurface, Bowl, Capsule, Sphere)):
        return obj.to_triangles(segments=settings.sphere_segments, rings=settings.sphere_rings)
    to_triangles = getattr(obj, "to_triangles", None)
    if callable(to_triangles):
        return tuple(to_triangles())
    return ()


def _prepare_triangles(triangles: tuple[Triangle, ...]) -> tuple[_PreparedTriangle, ...]:
    return tuple(_PreparedTriangle(triangle, triangle.center(), triangle.normal()) for triangle in triangles)


def _sharp_edges_for(triangles: tuple[_PreparedTriangle, ...], threshold_degrees: float) -> tuple[tuple[Vec3, Vec3], ...]:
    if not triangles:
        return ()

    threshold_cos = cos(radians(threshold_degrees))
    edge_map: dict[tuple[tuple[float, float, float], tuple[float, float, float]], list[tuple[Vec3, Vec3, Vec3]]] = {}
    for prepared in triangles:
        triangle = prepared.triangle
        for start, end in ((triangle.a, triangle.b), (triangle.b, triangle.c), (triangle.c, triangle.a)):
            start_key = _point_key(start)
            end_key = _point_key(end)
            key = (start_key, end_key) if start_key <= end_key else (end_key, start_key)
            edge_map.setdefault(key, []).append((start, end, prepared.normal))

    sharp_edges: list[tuple[Vec3, Vec3]] = []
    for candidates in edge_map.values():
        start, end, first_normal = candidates[0]
        if len(candidates) == 1:
            sharp_edges.append((start, end))
            continue
        if any(first_normal.dot(other_normal) <= threshold_cos for _other_start, _other_end, other_normal in candidates[1:]):
            sharp_edges.append((start, end))
    return tuple(sharp_edges)


def _point_key(point: Vec3) -> tuple[float, float, float]:
    return (round(point.x, 6), round(point.y, 6), round(point.z, 6))


def _light_channels(scene: Scene, center: Vec3, normal: Vec3) -> tuple[float, float, float]:
    return _lighting_channels(scene, center, normal, normal, None).diffuse


def _lighting_channels(
    scene: Scene,
    center: Vec3,
    normal: Vec3,
    view_direction: Vec3,
    material,
    settings: RenderSettings | None = None,
    source_triangle: Triangle | None = None,
    shadow_triangles: tuple[Triangle, ...] | None = None,
) -> _LightingChannels:
    light_channels = [0.0, 0.0, 0.0]
    specular_channels = [0.0, 0.0, 0.0]
    if settings is not None and settings.bounce_light > 0.0:
        sky = max(0.0, normal.y) * settings.bounce_light
        floor = max(0.0, -normal.y) * settings.bounce_light
        light_channels[0] += 0.22 * sky + 0.32 * floor
        light_channels[1] += 0.26 * sky + 0.22 * floor
        light_channels[2] += 0.34 * sky + 0.14 * floor
    shininess = getattr(material, "shininess", 32.0) if material is not None else 32.0
    specular_enabled = material is not None and (getattr(material, "specular", 0.0) > 0.0 or getattr(material, "reflectivity", 0.0) > 0.0)
    for light in scene.lights:
        if not isinstance(light, (Lamp, Sun)) and not hasattr(light, "sample"):
            continue
        sample = light.sample(center)
        light_direction = sample.direction.normalized()
        facing = normal.dot(light_direction)
        wrap = settings.light_wrap if settings is not None else 0.0
        diffuse_response = max(0.0, (facing + wrap) / (1.0 + wrap)) if wrap > 0.0 else max(0.0, facing)
        strength = diffuse_response * sample.intensity
        if strength > 0.0 and settings is not None and settings.ray_traced_shadows:
            strength *= _shadow_factor_for_light(scene, center, light, light_direction, settings, source_triangle, shadow_triangles)
            if strength <= 0.0:
                continue
        lr, lg, lb = sample.color.to_floats()
        light_channels[0] += lr * strength
        light_channels[1] += lg * strength
        light_channels[2] += lb * strength
        if specular_enabled and strength > 0.0:
            halfway = (light_direction + view_direction).normalized(light_direction)
            highlight = max(0.0, normal.dot(halfway)) ** shininess * sample.intensity
            specular_channels[0] += lr * highlight
            specular_channels[1] += lg * highlight
            specular_channels[2] += lb * highlight
    return _LightingChannels(
        (light_channels[0], light_channels[1], light_channels[2]),
        (specular_channels[0], specular_channels[1], specular_channels[2]),
    )


def _shadow_max_distance(light, center: Vec3) -> float:
    if isinstance(light, Lamp):
        return max(0.0, (light.position - center).length())
    return float("inf")


def _shadow_factor_for_light(
    scene: Scene,
    center: Vec3,
    light,
    light_direction: Vec3,
    settings: RenderSettings,
    source_triangle: Triangle | None,
    shadow_triangles: tuple[Triangle, ...] | None,
) -> float:
    samples = settings.shadow_samples
    softness = settings.shadow_softness
    if not isinstance(light, Lamp) or samples <= 1 or softness <= 0.0:
        max_distance = _shadow_max_distance(light, center)
        return _shadow_transmission(scene, center, light_direction, max_distance, settings, source_triangle, shadow_triangles)

    total = _shadow_transmission(
        scene,
        center,
        light_direction,
        _shadow_max_distance(light, center),
        settings,
        source_triangle,
        shadow_triangles,
    )
    tangent, bitangent = _shadow_sample_basis(light_direction)
    for index in range(1, samples):
        angle = tau * ((index * 0.6180339887498949) % 1.0)
        radius = softness * sqrt(index / max(1, samples - 1))
        sample_position = light.position + tangent * (cos(angle) * radius) + bitangent * (sin(angle) * radius)
        sample_vector = sample_position - center
        sample_distance = sample_vector.length()
        if sample_distance <= settings.shadow_bias:
            total += 1.0
            continue
        sample_direction = sample_vector / sample_distance
        total += _shadow_transmission(scene, center, sample_direction, sample_distance, settings, source_triangle, shadow_triangles)
    return clamp(total / samples, 0.0, 1.0)


def _shadow_sample_basis(direction: Vec3) -> tuple[Vec3, Vec3]:
    tangent = direction.cross(Vec3(0.0, 1.0, 0.0))
    if tangent.length_squared() < 1e-10:
        tangent = direction.cross(Vec3(1.0, 0.0, 0.0))
    tangent = tangent.normalized(Vec3(1.0, 0.0, 0.0))
    bitangent = direction.cross(tangent).normalized(Vec3(0.0, 0.0, 1.0))
    return tangent, bitangent


def _triangle_culled_by_distance(prepared: _PreparedTriangle, camera: Camera, settings: RenderSettings) -> bool:
    return settings.max_render_distance is not None and prepared.center.distance_to(camera.position) > settings.max_render_distance


def _shadow_transmission(
    scene: Scene,
    origin: Vec3,
    direction: Vec3,
    max_distance: float,
    settings: RenderSettings,
    source_triangle: Triangle | None,
    shadow_triangles: tuple[Triangle, ...] | None = None,
) -> float:
    ray_origin = origin + direction * settings.shadow_bias
    limit = max_distance - settings.shadow_bias if max_distance != float("inf") else max_distance
    if limit <= settings.shadow_bias:
        return 1.0
    if shadow_triangles is None:
        shadow_triangles = tuple(
            triangle
            for obj in scene.objects
            if not isinstance(obj, (Point3, Line3))
            for triangle in _triangles_for(obj, settings)
        )
    transmission = 1.0
    for triangle in shadow_triangles:
        if source_triangle is not None and triangle is source_triangle:
            continue
        distance = _ray_triangle_distance(ray_origin, direction, triangle)
        if distance is not None and settings.shadow_bias < distance < limit:
            transmission *= getattr(triangle.material, "light_transmission", 0.0)
            if transmission <= 1e-6:
                return 0.0
    return clamp(transmission, 0.0, 1.0)


def _ray_triangle_distance(origin: Vec3, direction: Vec3, triangle: Triangle) -> float | None:
    epsilon = 1e-7
    edge_one = triangle.b - triangle.a
    edge_two = triangle.c - triangle.a
    h = direction.cross(edge_two)
    det = edge_one.dot(h)
    if -epsilon < det < epsilon:
        return None
    inv_det = 1.0 / det
    s = origin - triangle.a
    u = inv_det * s.dot(h)
    if u < 0.0 or u > 1.0:
        return None
    q = s.cross(edge_one)
    v = inv_det * direction.dot(q)
    if v < 0.0 or u + v > 1.0:
        return None
    distance = inv_det * edge_two.dot(q)
    if distance <= epsilon:
        return None
    return distance


def _interpolate_uv(triangle: Triangle, w0: float, w1: float, w2: float) -> tuple[float, float]:
    uv_a = triangle.uv_a or (0.0, 0.0)
    uv_b = triangle.uv_b or (0.0, 0.0)
    uv_c = triangle.uv_c or (0.0, 0.0)
    return (
        uv_a[0] * w0 + uv_b[0] * w1 + uv_c[0] * w2,
        uv_a[1] * w0 + uv_b[1] * w1 + uv_c[1] * w2,
    )


def _interpolate_normal(triangle: Triangle, w0: float, w1: float, w2: float, fallback: Vec3) -> Vec3:
    normal_a = triangle.normal_a or fallback
    normal_b = triangle.normal_b or fallback
    normal_c = triangle.normal_c or fallback
    return (normal_a * w0 + normal_b * w1 + normal_c * w2).normalized(fallback)


def _apply_ray_traced_reflection(
    color: Color,
    material,
    normal: Vec3,
    view_direction: Vec3,
    settings: RenderSettings,
) -> Color:
    bounces = settings.reflection_bounces
    reflectivity = getattr(material, "reflectivity", 0.0)
    if bounces <= 0 or reflectivity <= 0.0:
        return color

    incoming = -view_direction.normalized(Vec3(0.0, 0.0, -1.0))
    reflection = (incoming - normal * (2.0 * incoming.dot(normal))).normalized(normal)
    traced = _trace_reflection_environment(reflection, bounces)
    base = color.to_floats()
    amount = clamp(reflectivity * (0.38 + min(4, bounces) * 0.12), 0.0, 0.92)
    return Color.from_floats(
        base[0] * (1.0 - amount) + traced[0] * amount,
        base[1] * (1.0 - amount) + traced[1] * amount,
        base[2] * (1.0 - amount) + traced[2] * amount,
    )


def _trace_reflection_environment(direction: Vec3, bounces: int) -> tuple[float, float, float]:
    ray = direction.normalized(Vec3(0.0, 1.0, 0.0))
    energy = 1.0
    total = [0.0, 0.0, 0.0]
    normalizer = 0.0
    for index in range(max(1, bounces)):
        sky = max(0.0, ray.y)
        floor = max(0.0, -ray.y)
        horizon = max(0.0, 1.0 - abs(ray.y))
        glint = max(0.0, ray.dot(Vec3(-0.35, 0.72, -0.58).normalized())) ** 28.0
        sample = (
            0.05 + sky * 0.34 + floor * 0.16 + horizon * 0.09 + glint * 0.85,
            0.06 + sky * 0.42 + floor * 0.22 + horizon * 0.11 + glint * 0.78,
            0.08 + sky * 0.56 + floor * 0.28 + horizon * 0.14 + glint * 0.62,
        )
        total[0] += sample[0] * energy
        total[1] += sample[1] * energy
        total[2] += sample[2] * energy
        normalizer += energy
        if index == bounces - 1:
            break
        if ray.y < 0.0:
            ray = Vec3(ray.x * 0.82, -ray.y * 0.78 + 0.16, ray.z * 0.82).normalized(ray)
        else:
            ray = Vec3(ray.x * 0.76 - ray.z * 0.18, ray.y * 0.54 - 0.12, ray.z * 0.76 + ray.x * 0.18).normalized(ray)
        energy *= 0.48
    if normalizer <= 0.0:
        return (0.0, 0.0, 0.0)
    return (
        clamp(total[0] / normalizer, 0.0, 1.0),
        clamp(total[1] / normalizer, 0.0, 1.0),
        clamp(total[2] / normalizer, 0.0, 1.0),
    )


def _apply_surface_attributes(color: Color, material, x: int, y: int, depth: float) -> Color:
    factor = 1.0
    if material.roughness:
        factor *= 1.0 - material.roughness * 0.18
    if material.fuzziness:
        noise = _surface_noise(x, y, depth)
        factor *= 1.0 + (noise - 0.5) * material.fuzziness * 0.55
    return color.scale(factor)


def _surface_noise(x: int, y: int, depth: float) -> float:
    value = (x * 73856093) ^ (y * 19349663) ^ (int(depth * 1000.0) * 83492791)
    value = (value ^ (value >> 13)) * 1274126177
    return (value & 0xFFFF) / 0xFFFF


def _apply_gamma(color: Color, gamma: float) -> Color:
    if abs(gamma - 1.0) < 1e-9:
        return color
    exponent = 1.0 / gamma
    red, green, blue = color.to_floats()
    return Color.from_floats(red**exponent, green**exponent, blue**exponent)


def _draw_text_bulletin(buffer: PixelBuffer, bulletin: TextBulletin) -> None:
    _draw_bulletin_at(
        buffer,
        bulletin.text,
        bulletin.position,
        bulletin.color,
        bulletin.background,
        bulletin.padding,
        bulletin.scale,
    )


def _draw_floating_text_bulletin(buffer: PixelBuffer, camera: Camera, bulletin: FloatingTextBulletin) -> None:
    projected = camera.project(bulletin.position, buffer.width, buffer.height)
    if projected is None:
        return
    text_width, text_height = draw.text_size(bulletin.text, scale=bulletin.scale)
    total_width = text_width + bulletin.padding * 2
    total_height = text_height + bulletin.padding * 2
    x = int(projected.x + bulletin.screen_offset[0] - total_width * bulletin.anchor[0])
    y = int(projected.y + bulletin.screen_offset[1] - total_height * bulletin.anchor[1])
    _draw_bulletin_at(buffer, bulletin.text, (x, y), bulletin.color, bulletin.background, bulletin.padding, bulletin.scale)


def _draw_bulletin_at(
    buffer: PixelBuffer,
    text: str,
    position: tuple[int, int],
    color: Color,
    background: Color | None,
    padding: int,
    scale: int,
) -> None:
    x, y = position
    text_width, text_height = draw.text_size(text, scale=scale)
    if background is not None:
        draw.rect(
            buffer,
            (x, y),
            (text_width + padding * 2, text_height + padding * 2),
            background,
            fill=True,
        )
    draw.text(
        buffer,
        (x + padding, y + padding),
        text,
        color,
        scale=scale,
    )
