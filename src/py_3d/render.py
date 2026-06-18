"""Rendering backends and the reference CPU renderer."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
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

        for obj in scene.objects:
            if isinstance(obj, Point3):
                self._draw_point(buffer, depth, camera, obj)
            elif isinstance(obj, Line3):
                self._draw_line(buffer, depth, camera, obj)
            else:
                for triangle in _triangles_for(obj, settings):
                    if settings.wireframe:
                        self._draw_triangle_wireframe(buffer, depth, camera, triangle)
                    else:
                        self._draw_triangle(buffer, depth, scene, camera, settings, triangle)
        for bulletin in scene.bulletins:
            if isinstance(bulletin, TextBulletin):
                _draw_text_bulletin(buffer, bulletin)
        return buffer

    def _draw_point(self, buffer: PixelBuffer, depth: DepthBuffer, camera: Camera, point: Point3) -> None:
        projected = camera.project(point.position, buffer.width, buffer.height)
        if projected is None:
            return
        x = round(projected.x)
        y = round(projected.y)
        if depth.test_and_set(x, y, projected.depth):
            buffer.set_pixel(x, y, point.material.color)

    def _draw_line(self, buffer: PixelBuffer, depth: DepthBuffer, camera: Camera, line: Line3) -> None:
        start = camera.project(line.start, buffer.width, buffer.height)
        end = camera.project(line.end, buffer.width, buffer.height)
        if start is None or end is None:
            return
        self._draw_projected_line(buffer, depth, start, end, line.material.color)

    def _draw_triangle_wireframe(
        self,
        buffer: PixelBuffer,
        depth: DepthBuffer,
        camera: Camera,
        triangle: Triangle,
    ) -> None:
        a = camera.project(triangle.a, buffer.width, buffer.height)
        b = camera.project(triangle.b, buffer.width, buffer.height)
        c = camera.project(triangle.c, buffer.width, buffer.height)
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
        for index in range(int(steps) + 1):
            amount = index / steps
            x = round(start.x + dx * amount)
            y = round(start.y + dy * amount)
            z = start.depth + (end.depth - start.depth) * amount
            if depth.test_and_set(x, y, z):
                buffer.set_pixel(x, y, color)

    def _draw_triangle(
        self,
        buffer: PixelBuffer,
        depth: DepthBuffer,
        scene: Scene,
        camera: Camera,
        settings: RenderSettings,
        triangle: Triangle,
    ) -> None:
        a = camera.project(triangle.a, buffer.width, buffer.height)
        b = camera.project(triangle.b, buffer.width, buffer.height)
        c = camera.project(triangle.c, buffer.width, buffer.height)
        if a is None or b is None or c is None:
            return

        normal = triangle.normal()
        center = triangle.center()
        view_direction = (camera.position - center).normalized(Vec3(0.0, 0.0, -1.0))
        facing = normal.dot(view_direction)
        if settings.cull_backfaces and facing <= 0.0:
            return
        if facing < 0.0:
            normal = -normal

        color = _shade_triangle(scene, settings, triangle, normal)
        min_x = max(0, floor(min(a.x, b.x, c.x)))
        max_x = min(buffer.width - 1, ceil(max(a.x, b.x, c.x)))
        min_y = max(0, floor(min(a.y, b.y, c.y)))
        max_y = min(buffer.height - 1, ceil(max(a.y, b.y, c.y)))

        area = _edge(a.x, a.y, b.x, b.y, c.x, c.y)
        if abs(area) < 1e-12:
            return

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px = x + 0.5
                py = y + 0.5
                w0 = _edge(b.x, b.y, c.x, c.y, px, py) / area
                w1 = _edge(c.x, c.y, a.x, a.y, px, py) / area
                w2 = _edge(a.x, a.y, b.x, b.y, px, py) / area
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                    continue
                z = w0 * a.depth + w1 * b.depth + w2 * c.depth
                if depth.test_and_set(x, y, z):
                    buffer.set_pixel(x, y, color)


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


def _shade_triangle(scene: Scene, settings: RenderSettings, triangle: Triangle, normal: Vec3) -> Color:
    center = triangle.center()
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
    return triangle.material.shade(tuple(light_channels), ambient=settings.ambient)


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
