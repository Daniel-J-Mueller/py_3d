"""Renderable primitive data types."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable

from .materials import Material
from .math3d import Vec3, as_vec3


@dataclass(frozen=True)
class Point3:
    position: Vec3 | tuple[float, float, float]
    material: Material = Material()

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_vec3(self.position))


@dataclass(frozen=True)
class Line3:
    start: Vec3 | tuple[float, float, float]
    end: Vec3 | tuple[float, float, float]
    material: Material = Material()

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", as_vec3(self.start))
        object.__setattr__(self, "end", as_vec3(self.end))


@dataclass(frozen=True)
class Triangle:
    a: Vec3 | tuple[float, float, float]
    b: Vec3 | tuple[float, float, float]
    c: Vec3 | tuple[float, float, float]
    material: Material = Material()
    uv_a: tuple[float, float] | None = None
    uv_b: tuple[float, float] | None = None
    uv_c: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", as_vec3(self.a))
        object.__setattr__(self, "b", as_vec3(self.b))
        object.__setattr__(self, "c", as_vec3(self.c))
        for name in ("uv_a", "uv_b", "uv_c"):
            uv = getattr(self, name)
            if uv is not None:
                if len(uv) != 2:
                    raise ValueError("triangle UV coordinates must have two components")
                object.__setattr__(self, name, (float(uv[0]), float(uv[1])))

    def center(self) -> Vec3:
        return (self.a + self.b + self.c) / 3.0

    def normal(self) -> Vec3:
        return (self.b - self.a).cross(self.c - self.a).normalized()

    def has_texture_coordinates(self) -> bool:
        return self.uv_a is not None and self.uv_b is not None and self.uv_c is not None


@dataclass(frozen=True)
class Mesh:
    triangles: tuple[Triangle, ...]

    def __init__(self, triangles: Iterable[Triangle]):
        object.__setattr__(self, "triangles", tuple(triangles))

    def to_triangles(self, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        return self.triangles


@dataclass(frozen=True)
class Box:
    """An axis-aligned box."""

    center: Vec3 | tuple[float, float, float]
    size: Vec3 | tuple[float, float, float]
    material: Material = Material()

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", as_vec3(self.center))
        object.__setattr__(self, "size", as_vec3(self.size))
        if self.size.x <= 0.0 or self.size.y <= 0.0 or self.size.z <= 0.0:
            raise ValueError("box size components must be positive")

    def to_triangles(self, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        half = self.size * 0.5
        cx, cy, cz = self.center.as_tuple()
        hx, hy, hz = half.as_tuple()
        v = [
            Vec3(cx - hx, cy - hy, cz - hz),
            Vec3(cx + hx, cy - hy, cz - hz),
            Vec3(cx + hx, cy + hy, cz - hz),
            Vec3(cx - hx, cy + hy, cz - hz),
            Vec3(cx - hx, cy - hy, cz + hz),
            Vec3(cx + hx, cy - hy, cz + hz),
            Vec3(cx + hx, cy + hy, cz + hz),
            Vec3(cx - hx, cy + hy, cz + hz),
        ]
        faces = [
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (3, 6, 2), (3, 7, 6),
            (1, 2, 6), (1, 6, 5),
            (0, 4, 7), (0, 7, 3),
        ]
        return tuple(Triangle(v[a], v[b], v[c], self.material) for a, b, c in faces)


@dataclass(frozen=True)
class Sphere:
    center: Vec3 | tuple[float, float, float]
    radius: float
    material: Material = Material()

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", as_vec3(self.center))
        if self.radius <= 0.0:
            raise ValueError("sphere radius must be positive")

    def to_triangles(self, segments: int = 16, rings: int = 8, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        if segments < 3:
            raise ValueError("sphere segments must be at least 3")
        if rings < 2:
            raise ValueError("sphere rings must be at least 2")

        vertices: list[list[Vec3]] = []
        for ring in range(rings + 1):
            phi = pi * ring / rings
            row = []
            for segment in range(segments):
                theta = 2.0 * pi * segment / segments
                row.append(
                    self.center
                    + Vec3(
                        self.radius * sin(phi) * cos(theta),
                        self.radius * cos(phi),
                        self.radius * sin(phi) * sin(theta),
                    )
                )
            vertices.append(row)

        triangles: list[Triangle] = []
        for ring in range(rings):
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                top_left = vertices[ring][segment]
                top_right = vertices[ring][next_segment]
                bottom_left = vertices[ring + 1][segment]
                bottom_right = vertices[ring + 1][next_segment]
                if ring != 0:
                    triangles.append(Triangle(top_left, bottom_left, top_right, self.material))
                if ring != rings - 1:
                    triangles.append(Triangle(top_right, bottom_left, bottom_right, self.material))
        return tuple(triangles)


@dataclass(frozen=True)
class Plane:
    point: Vec3 | tuple[float, float, float]
    normal: Vec3 | tuple[float, float, float]
    material: Material = Material()
    size: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", as_vec3(self.point))
        object.__setattr__(self, "normal", as_vec3(self.normal).normalized(Vec3(0.0, 1.0, 0.0)))
        if self.size is not None and self.size <= 0.0:
            raise ValueError("plane size must be positive when provided")

    def to_triangles(self, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        if self.size is None:
            return ()
        tangent = self.normal.cross(Vec3(0.0, 0.0, 1.0)).normalized()
        if tangent.length_squared() == 0.0:
            tangent = self.normal.cross(Vec3(1.0, 0.0, 0.0)).normalized(Vec3(1.0, 0.0, 0.0))
        bitangent = self.normal.cross(tangent).normalized()
        half = self.size * 0.5
        a = self.point - tangent * half - bitangent * half
        b = self.point + tangent * half - bitangent * half
        c = self.point + tangent * half + bitangent * half
        d = self.point - tangent * half + bitangent * half
        return (Triangle(a, b, c, self.material), Triangle(a, c, d, self.material))
