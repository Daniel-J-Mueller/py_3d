"""Renderable primitive data types."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable

from .materials import Material
from .math3d import Vec3, as_vec3
from .noise import SurfacePerturbation


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
    normal_a: Vec3 | tuple[float, float, float] | None = None
    normal_b: Vec3 | tuple[float, float, float] | None = None
    normal_c: Vec3 | tuple[float, float, float] | None = None

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
        for name in ("normal_a", "normal_b", "normal_c"):
            normal = getattr(self, name)
            if normal is not None:
                object.__setattr__(self, name, as_vec3(normal).normalized())

    def center(self) -> Vec3:
        return (self.a + self.b + self.c) / 3.0

    def normal(self) -> Vec3:
        return (self.b - self.a).cross(self.c - self.a).normalized()

    def has_texture_coordinates(self) -> bool:
        return self.uv_a is not None and self.uv_b is not None and self.uv_c is not None

    def has_vertex_normals(self) -> bool:
        return self.normal_a is not None and self.normal_b is not None and self.normal_c is not None


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
    perturbation: SurfacePerturbation | None = None
    rotation: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", as_vec3(self.center))
        object.__setattr__(self, "rotation", as_vec3(self.rotation))
        if self.radius <= 0.0:
            raise ValueError("sphere radius must be positive")

    def to_triangles(self, segments: int = 16, rings: int = 8, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        if segments < 3:
            raise ValueError("sphere segments must be at least 3")
        if rings < 2:
            raise ValueError("sphere rings must be at least 2")

        vertices: list[list[Vec3]] = []
        normals: list[list[Vec3]] = []
        for ring in range(rings + 1):
            phi = pi * ring / rings
            row = []
            normal_row = []
            for segment in range(segments):
                theta = 2.0 * pi * segment / segments
                normal = Vec3(
                    sin(phi) * cos(theta),
                    cos(phi),
                    sin(phi) * sin(theta),
                )
                radius = self._radius_at(normal)
                rotated_normal = _rotate_euler(normal, self.rotation)
                row.append(self.center + rotated_normal * radius)
                normal_row.append(rotated_normal)
            vertices.append(row)
            normals.append(normal_row)

        triangles: list[Triangle] = []
        for ring in range(rings):
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                u = segment / segments
                next_u = 1.0 if next_segment == 0 else next_segment / segments
                v = ring / rings
                next_v = (ring + 1) / rings
                top_left = vertices[ring][segment]
                top_right = vertices[ring][next_segment]
                bottom_left = vertices[ring + 1][segment]
                bottom_right = vertices[ring + 1][next_segment]
                normal_top_left = normals[ring][segment]
                normal_top_right = normals[ring][next_segment]
                normal_bottom_left = normals[ring + 1][segment]
                normal_bottom_right = normals[ring + 1][next_segment]
                if ring != 0:
                    triangles.append(
                        Triangle(
                            top_left,
                            bottom_left,
                            top_right,
                            self.material,
                            (u, v),
                            (u, next_v),
                            (next_u, v),
                            normal_top_left,
                            normal_bottom_left,
                            normal_top_right,
                        )
                    )
                if ring != rings - 1:
                    triangles.append(
                        Triangle(
                            top_right,
                            bottom_left,
                            bottom_right,
                            self.material,
                            (next_u, v),
                            (u, next_v),
                            (next_u, next_v),
                            normal_top_right,
                            normal_bottom_left,
                            normal_bottom_right,
                        )
                    )
        return tuple(triangles)

    def _radius_at(self, normal: Vec3) -> float:
        if self.perturbation is None:
            return self.radius
        return max(0.001, self.radius + self.perturbation.displacement(normal))


@dataclass(frozen=True)
class Bowl:
    """An open-top spherical-cap bowl.

    ``center`` is the center of the rim plane. ``depth`` is a fraction of the
    lower hemisphere angle: 1.0 reaches a pointed hemispherical bottom, while
    smaller values close the cap with a flat bottom disk.
    """

    center: Vec3 | tuple[float, float, float]
    radius: float
    material: Material = Material()
    depth: float = 1.0
    perturbation: SurfacePerturbation | None = None
    thickness: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", as_vec3(self.center))
        if self.radius <= 0.0:
            raise ValueError("bowl radius must be positive")
        if self.depth <= 0.0 or self.depth > 1.0:
            raise ValueError("bowl depth must be in the range (0, 1]")
        if self.thickness < 0.0:
            raise ValueError("bowl thickness must be non-negative")

    def to_triangles(self, segments: int = 24, rings: int = 8, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        if segments < 3:
            raise ValueError("bowl segments must be at least 3")
        if rings < 1:
            raise ValueError("bowl rings must be at least 1")

        end_phi = pi / 2.0 + self.depth * pi / 2.0
        vertices: list[list[Vec3]] = []
        normals: list[list[Vec3]] = []
        for ring in range(rings + 1):
            amount = ring / rings
            phi = pi / 2.0 + (end_phi - pi / 2.0) * amount
            row = []
            normal_row = []
            for segment in range(segments):
                theta = 2.0 * pi * segment / segments
                normal = Vec3(
                    sin(phi) * cos(theta),
                    cos(phi),
                    sin(phi) * sin(theta),
                )
                row.append(self.center + normal * self._radius_at(normal))
                normal_row.append(-normal)
            vertices.append(row)
            normals.append(normal_row)

        triangles: list[Triangle] = []
        for ring in range(rings):
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                u = segment / segments
                next_u = 1.0 if next_segment == 0 else next_segment / segments
                v = ring / rings
                next_v = (ring + 1) / rings
                top_left = vertices[ring][segment]
                top_right = vertices[ring][next_segment]
                bottom_left = vertices[ring + 1][segment]
                bottom_right = vertices[ring + 1][next_segment]
                normal_top_left = normals[ring][segment]
                normal_top_right = normals[ring][next_segment]
                normal_bottom_left = normals[ring + 1][segment]
                normal_bottom_right = normals[ring + 1][next_segment]
                triangles.append(
                    Triangle(
                        top_left,
                        bottom_left,
                        top_right,
                        self.material,
                        (u, v),
                        (u, next_v),
                        (next_u, v),
                        normal_top_left,
                        normal_bottom_left,
                        normal_top_right,
                    )
                )
                if ring != rings - 1 or self.depth < 1.0:
                    triangles.append(
                        Triangle(
                            top_right,
                            bottom_left,
                            bottom_right,
                            self.material,
                            (next_u, v),
                            (u, next_v),
                            (next_u, next_v),
                            normal_top_right,
                            normal_bottom_left,
                            normal_bottom_right,
                        )
                    )

        if self.depth < 1.0:
            bottom_phi = end_phi
            bottom_y = self.center.y + cos(bottom_phi) * self.radius
            bottom_center = Vec3(self.center.x, bottom_y, self.center.z)
            bottom_v = 1.0
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                u = segment / segments
                next_u = 1.0 if next_segment == 0 else next_segment / segments
                triangles.append(
                    Triangle(
                        vertices[-1][next_segment],
                        vertices[-1][segment],
                        bottom_center,
                        self.material,
                        (next_u, bottom_v),
                        (u, bottom_v),
                        (0.5, bottom_v),
                        Vec3(0.0, 1.0, 0.0),
                        Vec3(0.0, 1.0, 0.0),
                        Vec3(0.0, 1.0, 0.0),
                    )
                )
        if self.thickness > 0.0:
            triangles.extend(self._thickness_triangles(vertices, normals, segments, rings, end_phi))
        return tuple(triangles)

    def _thickness_triangles(
        self,
        inner_vertices: list[list[Vec3]],
        inner_normals: list[list[Vec3]],
        segments: int,
        rings: int,
        end_phi: float,
    ) -> list[Triangle]:
        del inner_normals
        outer_vertices: list[list[Vec3]] = []
        outer_normals: list[list[Vec3]] = []
        outer_radius = self.radius + self.thickness
        for ring in range(rings + 1):
            amount = ring / rings
            phi = pi / 2.0 + (end_phi - pi / 2.0) * amount
            row = []
            normal_row = []
            for segment in range(segments):
                theta = 2.0 * pi * segment / segments
                normal = Vec3(
                    sin(phi) * cos(theta),
                    cos(phi),
                    sin(phi) * sin(theta),
                )
                row.append(self.center + normal * outer_radius)
                normal_row.append(normal)
            outer_vertices.append(row)
            outer_normals.append(normal_row)

        triangles: list[Triangle] = []
        for ring in range(rings):
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                u = segment / segments
                next_u = 1.0 if next_segment == 0 else next_segment / segments
                v = ring / rings
                next_v = (ring + 1) / rings
                top_left = outer_vertices[ring][segment]
                top_right = outer_vertices[ring][next_segment]
                bottom_left = outer_vertices[ring + 1][segment]
                bottom_right = outer_vertices[ring + 1][next_segment]
                normal_top_left = outer_normals[ring][segment]
                normal_top_right = outer_normals[ring][next_segment]
                normal_bottom_left = outer_normals[ring + 1][segment]
                normal_bottom_right = outer_normals[ring + 1][next_segment]
                triangles.append(
                    Triangle(
                        top_left,
                        top_right,
                        bottom_left,
                        self.material,
                        (u, v),
                        (next_u, v),
                        (u, next_v),
                        normal_top_left,
                        normal_top_right,
                        normal_bottom_left,
                    )
                )
                if ring != rings - 1 or self.depth < 1.0:
                    triangles.append(
                        Triangle(
                            top_right,
                            bottom_right,
                            bottom_left,
                            self.material,
                            (next_u, v),
                            (next_u, next_v),
                            (u, next_v),
                            normal_top_right,
                            normal_bottom_right,
                            normal_bottom_left,
                        )
                    )

        rim_normal = Vec3(0.0, 1.0, 0.0)
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            inner_left = inner_vertices[0][segment]
            inner_right = inner_vertices[0][next_segment]
            outer_left = outer_vertices[0][segment]
            outer_right = outer_vertices[0][next_segment]
            triangles.append(Triangle(inner_right, inner_left, outer_left, self.material, normal_a=rim_normal, normal_b=rim_normal, normal_c=rim_normal))
            triangles.append(Triangle(inner_right, outer_left, outer_right, self.material, normal_a=rim_normal, normal_b=rim_normal, normal_c=rim_normal))
        return triangles

    def _radius_at(self, normal: Vec3) -> float:
        if self.perturbation is None:
            return self.radius
        return max(0.001, self.radius + self.perturbation.displacement(normal))


@dataclass(frozen=True)
class Capsule:
    """A vertical capsule primitive with hemispherical ends."""

    center: Vec3 | tuple[float, float, float]
    radius: float
    height: float
    material: Material = Material()

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", as_vec3(self.center))
        if self.radius <= 0.0:
            raise ValueError("capsule radius must be positive")
        if self.height < self.radius * 2.0:
            raise ValueError("capsule height must be at least twice the radius")

    def to_triangles(self, segments: int = 16, rings: int = 8, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        if segments < 3:
            raise ValueError("capsule segments must be at least 3")
        if rings < 2:
            raise ValueError("capsule rings must be at least 2")

        cylinder_half = max(0.0, self.height * 0.5 - self.radius)
        hemisphere_rings = max(2, rings // 2)
        ring_specs: list[tuple[float, float, Vec3]] = []
        for index in range(hemisphere_rings + 1):
            amount = index / hemisphere_rings
            phi = pi - amount * pi / 2.0
            ring_radius = sin(phi) * self.radius
            y = -cylinder_half + cos(phi) * self.radius
            normal_y = cos(phi)
            ring_specs.append((ring_radius, y, Vec3(0.0, normal_y, 0.0)))
        if cylinder_half > 1e-9:
            ring_specs.append((self.radius, cylinder_half, Vec3(0.0, 0.0, 0.0)))
        for index in range(1, hemisphere_rings + 1):
            amount = index / hemisphere_rings
            phi = pi / 2.0 - amount * pi / 2.0
            ring_radius = sin(phi) * self.radius
            y = cylinder_half + cos(phi) * self.radius
            normal_y = cos(phi)
            ring_specs.append((ring_radius, y, Vec3(0.0, normal_y, 0.0)))

        vertices: list[list[Vec3]] = []
        normals: list[list[Vec3]] = []
        for ring_radius, y, normal_template in ring_specs:
            row = []
            normal_row = []
            for segment in range(segments):
                theta = 2.0 * pi * segment / segments
                horizontal = Vec3(cos(theta), 0.0, sin(theta))
                point = self.center + Vec3(horizontal.x * ring_radius, y, horizontal.z * ring_radius)
                normal = Vec3(horizontal.x * max(0.0, 1.0 - abs(normal_template.y)), normal_template.y, horizontal.z * max(0.0, 1.0 - abs(normal_template.y))).normalized(horizontal)
                row.append(point)
                normal_row.append(normal)
            vertices.append(row)
            normals.append(normal_row)

        triangles: list[Triangle] = []
        for ring in range(len(vertices) - 1):
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                top_left = vertices[ring][segment]
                top_right = vertices[ring][next_segment]
                bottom_left = vertices[ring + 1][segment]
                bottom_right = vertices[ring + 1][next_segment]
                normal_top_left = normals[ring][segment]
                normal_top_right = normals[ring][next_segment]
                normal_bottom_left = normals[ring + 1][segment]
                normal_bottom_right = normals[ring + 1][next_segment]
                triangles.append(Triangle(top_left, bottom_left, top_right, self.material, normal_a=normal_top_left, normal_b=normal_bottom_left, normal_c=normal_top_right))
                triangles.append(Triangle(top_right, bottom_left, bottom_right, self.material, normal_a=normal_top_right, normal_b=normal_bottom_left, normal_c=normal_bottom_right))
        return tuple(triangles)


@dataclass(frozen=True)
class Plane:
    point: Vec3 | tuple[float, float, float]
    normal: Vec3 | tuple[float, float, float]
    material: Material = Material()
    size: float | None = None
    thickness: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", as_vec3(self.point))
        object.__setattr__(self, "normal", as_vec3(self.normal).normalized(Vec3(0.0, 1.0, 0.0)))
        if self.size is not None and self.size <= 0.0:
            raise ValueError("plane size must be positive when provided")
        if self.thickness < 0.0:
            raise ValueError("plane thickness must be non-negative")

    def to_triangles(self, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        if self.size is None:
            return ()
        tangent = self.normal.cross(Vec3(0.0, 0.0, 1.0)).normalized()
        if tangent.length_squared() == 0.0:
            tangent = self.normal.cross(Vec3(1.0, 0.0, 0.0)).normalized(Vec3(1.0, 0.0, 0.0))
        bitangent = self.normal.cross(tangent).normalized()
        half = self.size * 0.5
        offset = self.normal * (self.thickness * 0.5)
        a = self.point - tangent * half - bitangent * half + offset
        b = self.point + tangent * half - bitangent * half + offset
        c = self.point + tangent * half + bitangent * half + offset
        d = self.point - tangent * half + bitangent * half + offset
        if self.thickness <= 0.0:
            return (
                Triangle(a, b, c, self.material, normal_a=self.normal, normal_b=self.normal, normal_c=self.normal),
                Triangle(a, c, d, self.material, normal_a=self.normal, normal_b=self.normal, normal_c=self.normal),
            )

        bottom_offset = -offset
        e = self.point - tangent * half - bitangent * half + bottom_offset
        f = self.point + tangent * half - bitangent * half + bottom_offset
        g = self.point + tangent * half + bitangent * half + bottom_offset
        h = self.point - tangent * half + bitangent * half + bottom_offset
        bottom_normal = -self.normal
        side_normals = (tangent, bitangent, -tangent, -bitangent)
        return (
            Triangle(a, b, c, self.material, normal_a=self.normal, normal_b=self.normal, normal_c=self.normal),
            Triangle(a, c, d, self.material, normal_a=self.normal, normal_b=self.normal, normal_c=self.normal),
            Triangle(e, g, f, self.material, normal_a=bottom_normal, normal_b=bottom_normal, normal_c=bottom_normal),
            Triangle(e, h, g, self.material, normal_a=bottom_normal, normal_b=bottom_normal, normal_c=bottom_normal),
            Triangle(a, e, f, self.material, normal_a=side_normals[3], normal_b=side_normals[3], normal_c=side_normals[3]),
            Triangle(a, f, b, self.material, normal_a=side_normals[3], normal_b=side_normals[3], normal_c=side_normals[3]),
            Triangle(b, f, g, self.material, normal_a=side_normals[0], normal_b=side_normals[0], normal_c=side_normals[0]),
            Triangle(b, g, c, self.material, normal_a=side_normals[0], normal_b=side_normals[0], normal_c=side_normals[0]),
            Triangle(c, g, h, self.material, normal_a=side_normals[1], normal_b=side_normals[1], normal_c=side_normals[1]),
            Triangle(c, h, d, self.material, normal_a=side_normals[1], normal_b=side_normals[1], normal_c=side_normals[1]),
            Triangle(d, h, e, self.material, normal_a=side_normals[2], normal_b=side_normals[2], normal_c=side_normals[2]),
            Triangle(d, e, a, self.material, normal_a=side_normals[2], normal_b=side_normals[2], normal_c=side_normals[2]),
        )


def _rotate_euler(value: Vec3, rotation: Vec3) -> Vec3:
    cx, sx = cos(rotation.x), sin(rotation.x)
    cy, sy = cos(rotation.y), sin(rotation.y)
    cz, sz = cos(rotation.z), sin(rotation.z)
    x_rotated = Vec3(value.x, value.y * cx - value.z * sx, value.y * sx + value.z * cx)
    y_rotated = Vec3(x_rotated.x * cy + x_rotated.z * sy, x_rotated.y, -x_rotated.x * sy + x_rotated.z * cy)
    return Vec3(y_rotated.x * cz - y_rotated.y * sz, y_rotated.x * sz + y_rotated.y * cz, y_rotated.z)
