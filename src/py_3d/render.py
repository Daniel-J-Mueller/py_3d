"""Rendering backends and the reference CPU renderer."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, floor, radians, tan
from typing import Protocol, runtime_checkable
import weakref

from .buffer import DepthBuffer, PixelBuffer
from .camera import Camera, ProjectedPoint
from .color import Color
from . import draw
from .lights import Lamp, Sun
from .math3d import Vec3, clamp
from .overlays import TextBulletin
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
    cull_backfaces: bool = False
    wireframe: bool = False
    smooth_shading: bool = False
    two_sided_lighting: bool = True
    ray_traced_shadows: bool = False
    shadow_bias: float = 1e-4
    edge_highlight: bool = False
    edge_highlight_threshold_degrees: float = 35.0
    edge_highlight_color: Color | tuple[int, int, int] = Color(0, 0, 0)
    edge_highlight_depth_bias: float = 0.002
    max_render_distance: float | None = None
    sphere_segments: int = 16
    sphere_rings: int = 8

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("render dimensions must be positive")
        object.__setattr__(self, "background", Color.from_value(self.background))
        object.__setattr__(self, "ambient", clamp(float(self.ambient), 0.0, 1.0))
        object.__setattr__(self, "gamma", max(0.01, float(self.gamma)))
        object.__setattr__(self, "shadow_bias", max(0.0, float(self.shadow_bias)))
        object.__setattr__(self, "edge_highlight_threshold_degrees", clamp(float(self.edge_highlight_threshold_degrees), 0.0, 180.0))
        object.__setattr__(self, "edge_highlight_color", Color.from_value(self.edge_highlight_color))
        object.__setattr__(self, "edge_highlight_depth_bias", max(0.0, float(self.edge_highlight_depth_bias)))
        if self.max_render_distance is not None:
            object.__setattr__(self, "max_render_distance", max(0.0, float(self.max_render_distance)))


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
        return self.renderer.render(scene, camera, active_settings, target)


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
                    if use_texture:
                        uv = _interpolate_uv(triangle, w0, w1, w2)
                        texture_color = triangle.material.color_at(uv)
                        shaded = triangle.material.shade(
                            active_lighting.diffuse,
                            ambient=settings.ambient,
                            base_color=texture_color,
                            specular_light=active_lighting.specular,
                        )
                        pixels[index] = _apply_gamma(_apply_surface_attributes(shaded, triangle.material, x, y, z), settings.gamma)
                    elif use_smooth_shading:
                        shaded = triangle.material.shade(
                            active_lighting.diffuse,
                            ambient=settings.ambient,
                            specular_light=active_lighting.specular,
                        )
                        pixels[index] = _apply_gamma(_apply_surface_attributes(shaded, triangle.material, x, y, z), settings.gamma)
                    else:
                        pixels[index] = _apply_gamma(_apply_surface_attributes(color, triangle.material, x, y, z), settings.gamma)

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
    shininess = getattr(material, "shininess", 32.0) if material is not None else 32.0
    specular_enabled = material is not None and (getattr(material, "specular", 0.0) > 0.0 or getattr(material, "reflectivity", 0.0) > 0.0)
    for light in scene.lights:
        if not isinstance(light, (Lamp, Sun)) and not hasattr(light, "sample"):
            continue
        sample = light.sample(center)
        light_direction = sample.direction.normalized()
        strength = max(0.0, normal.dot(light_direction)) * sample.intensity
        if strength > 0.0 and settings is not None and settings.ray_traced_shadows:
            max_distance = _shadow_max_distance(light, center)
            strength *= _shadow_transmission(scene, center, light_direction, max_distance, settings, source_triangle, shadow_triangles)
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
    x, y = bulletin.position
    text_width, text_height = draw.text_size(bulletin.text, scale=bulletin.scale)
    if bulletin.background is not None:
        draw.rect(
            buffer,
            (x, y),
            (text_width + bulletin.padding * 2, text_height + bulletin.padding * 2),
            bulletin.background,
            fill=True,
        )
    draw.text(
        buffer,
        (x + bulletin.padding, y + bulletin.padding),
        bulletin.text,
        bulletin.color,
        scale=bulletin.scale,
    )
