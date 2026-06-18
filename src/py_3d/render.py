"""Rendering backends and the reference CPU renderer."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, radians, tan
from typing import Protocol, runtime_checkable

from .buffer import DepthBuffer, PixelBuffer
from .camera import Camera, ProjectedPoint
from .color import Color
from . import draw
from .lights import Lamp, Sun
from .math3d import Vec3, clamp
from .overlays import TextBulletin
from .primitives import Line3, Mesh, Point3, Sphere, Triangle
from .scene import Scene


@dataclass(frozen=True)
class RenderSettings:
    """Settings for one offline or real-time render pass."""

    width: int = 640
    height: int = 480
    background: Color | tuple[int, int, int] = Color(0, 0, 0)
    ambient: float = 0.08
    cull_backfaces: bool = False
    wireframe: bool = False
    sphere_segments: int = 16
    sphere_rings: int = 8

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("render dimensions must be positive")
        object.__setattr__(self, "background", Color.from_value(self.background))
        object.__setattr__(self, "ambient", clamp(float(self.ambient), 0.0, 1.0))


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
        self._triangle_cache: dict[tuple[type, int, int, int], tuple[_PreparedTriangle, ...]] = {}

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

        for obj in scene.objects:
            if isinstance(obj, Point3):
                self._draw_point(buffer, depth, projector, obj)
            elif isinstance(obj, Line3):
                self._draw_line(buffer, depth, projector, obj)
            else:
                for triangle in self._prepared_triangles_for(obj, settings):
                    if settings.wireframe:
                        self._draw_triangle_wireframe(buffer, depth, projector, triangle.triangle)
                    else:
                        self._draw_triangle(buffer, depth, scene, camera, projector, settings, triangle)
        for bulletin in scene.bulletins:
            if isinstance(bulletin, TextBulletin):
                _draw_text_bulletin(buffer, bulletin)
        return buffer

    def _prepared_triangles_for(self, obj, settings: RenderSettings) -> tuple["_PreparedTriangle", ...]:
        if isinstance(obj, Triangle):
            return (_PreparedTriangle(obj, obj.center(), obj.normal()),)
        if not self.cache_static_geometry:
            return _prepare_triangles(_triangles_for(obj, settings))

        if isinstance(obj, Sphere):
            key = (type(obj), id(obj), settings.sphere_segments, settings.sphere_rings)
        else:
            key = (type(obj), id(obj), 0, 0)
        triangles = self._triangle_cache.get(key)
        if triangles is None:
            triangles = _prepare_triangles(_triangles_for(obj, settings))
            self._triangle_cache[key] = triangles
        return triangles

    def _draw_point(self, buffer: PixelBuffer, depth: DepthBuffer, projector: "_CameraProjector", point: Point3) -> None:
        projected = projector.project(point.position)
        if projected is None:
            return
        x = round(projected.x)
        y = round(projected.y)
        if 0 <= x < buffer.width and 0 <= y < buffer.height:
            index = y * buffer.width + x
            if projected.depth < depth.values[index]:
                depth.values[index] = projected.depth
                buffer.pixels[index] = point.material.color

    def _draw_line(self, buffer: PixelBuffer, depth: DepthBuffer, projector: "_CameraProjector", line: Line3) -> None:
        start = projector.project(line.start)
        end = projector.project(line.end)
        if start is None or end is None:
            return
        self._draw_projected_line(buffer, depth, start, end, line.material.color)

    def _draw_triangle_wireframe(
        self,
        buffer: PixelBuffer,
        depth: DepthBuffer,
        projector: "_CameraProjector",
        triangle: Triangle,
    ) -> None:
        a = projector.project(triangle.a)
        b = projector.project(triangle.b)
        c = projector.project(triangle.c)
        if a is None or b is None or c is None:
            return
        color = triangle.material.color
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
            z = start.depth + (end.depth - start.depth) * amount
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
        if facing < 0.0:
            normal = -normal

        light_channels = _light_channels(scene, center, normal)
        color = triangle.material.shade(light_channels, ambient=settings.ambient)
        use_texture = triangle.material.texture is not None and triangle.has_texture_coordinates()
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
                    if use_texture:
                        uv = _interpolate_uv(triangle, w0, w1, w2)
                        texture_color = triangle.material.color_at(uv)
                        shaded = triangle.material.shade(
                            light_channels,
                            ambient=settings.ambient,
                            base_color=texture_color,
                        )
                        pixels[index] = _apply_surface_attributes(shaded, triangle.material, x, y, z)
                    else:
                        pixels[index] = _apply_surface_attributes(color, triangle.material, x, y, z)


@dataclass(frozen=True)
class _PreparedTriangle:
    triangle: Triangle
    center: Vec3
    normal: Vec3


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


def _triangles_for(obj, settings: RenderSettings) -> tuple[Triangle, ...]:
    if isinstance(obj, Triangle):
        return (obj,)
    if isinstance(obj, Mesh):
        return obj.to_triangles()
    if isinstance(obj, Sphere):
        return obj.to_triangles(segments=settings.sphere_segments, rings=settings.sphere_rings)
    to_triangles = getattr(obj, "to_triangles", None)
    if callable(to_triangles):
        return tuple(to_triangles())
    return ()


def _prepare_triangles(triangles: tuple[Triangle, ...]) -> tuple[_PreparedTriangle, ...]:
    return tuple(_PreparedTriangle(triangle, triangle.center(), triangle.normal()) for triangle in triangles)


def _light_channels(scene: Scene, center: Vec3, normal: Vec3) -> tuple[float, float, float]:
    light_channels = [0.0, 0.0, 0.0]
    for light in scene.lights:
        if not isinstance(light, (Lamp, Sun)) and not hasattr(light, "sample"):
            continue
        sample = light.sample(center)
        strength = max(0.0, normal.dot(sample.direction.normalized())) * sample.intensity
        lr, lg, lb = sample.color.to_floats()
        light_channels[0] += lr * strength
        light_channels[1] += lg * strength
        light_channels[2] += lb * strength
    return (light_channels[0], light_channels[1], light_channels[2])


def _interpolate_uv(triangle: Triangle, w0: float, w1: float, w2: float) -> tuple[float, float]:
    uv_a = triangle.uv_a or (0.0, 0.0)
    uv_b = triangle.uv_b or (0.0, 0.0)
    uv_c = triangle.uv_c or (0.0, 0.0)
    return (
        uv_a[0] * w0 + uv_b[0] * w1 + uv_c[0] * w2,
        uv_a[1] * w0 + uv_b[1] * w1 + uv_c[1] * w2,
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
