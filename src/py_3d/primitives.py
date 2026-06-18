"""Renderable primitive data types."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable

from .color import Color
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
class LampPrimitive:
    """A small low-poly table lamp built from primitive triangle geometry."""

    position: Vec3 | tuple[float, float, float]
    color: Color | tuple[int, int, int] = Color(255, 238, 204)
    shade_color: Color | tuple[int, int, int] = Color(198, 152, 104)
    stem_color: Color | tuple[int, int, int] = Color(104, 96, 88)
    height: float = 0.82
    base_radius: float = 0.16
    stem_radius: float = 0.025
    shade_radius: float = 0.28
    shade_height: float = 0.24
    segments: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_vec3(self.position))
        object.__setattr__(self, "color", Color.from_value(self.color))
        object.__setattr__(self, "shade_color", Color.from_value(self.shade_color))
        object.__setattr__(self, "stem_color", Color.from_value(self.stem_color))
        if self.height <= 0.0:
            raise ValueError("lamp height must be positive")
        if self.base_radius <= 0.0 or self.stem_radius <= 0.0 or self.shade_radius <= 0.0:
            raise ValueError("lamp radii must be positive")
        if self.shade_height <= 0.0:
            raise ValueError("lamp shade height must be positive")
        if self.segments < 3:
            raise ValueError("lamp segments must be at least 3")

    def to_triangles(self, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        base_height = min(0.07, self.height * 0.16)
        shade_bottom = self.position.y - self.shade_height * 0.54
        base_bottom = self.position.y - self.height
        base_center = Vec3(self.position.x, base_bottom + base_height * 0.5, self.position.z)
        stem_center_y = (base_bottom + base_height + shade_bottom) * 0.5
        stem_height = max(0.02, shade_bottom - (base_bottom + base_height))

        stem_material = Material(color=self.stem_color, roughness=0.38, specular=0.28, shininess=30.0)
        shade_material = Material(color=self.shade_color, roughness=0.7, fuzziness=0.08, specular=0.04, shininess=10.0)
        bulb_material = Material(
            color=self.color,
            emission=self.color,
            diffuse=0.25,
            roughness=0.12,
            specular=0.35,
            shininess=48.0,
        )

        triangles: list[Triangle] = []
        triangles.extend(_cylinder_triangles(base_center, self.base_radius, base_height, self.segments, stem_material))
        triangles.extend(
            _cylinder_triangles(
                Vec3(self.position.x, stem_center_y, self.position.z),
                self.stem_radius,
                stem_height,
                self.segments,
                stem_material,
            )
        )
        triangles.extend(
            _frustum_triangles(
                Vec3(self.position.x, shade_bottom, self.position.z),
                Vec3(self.position.x, shade_bottom + self.shade_height, self.position.z),
                self.shade_radius,
                self.shade_radius * 0.62,
                self.segments,
                shade_material,
            )
        )
        triangles.extend(
            Sphere(self.position, self.shade_radius * 0.22, bulb_material).to_triangles(
                segments=self.segments,
                rings=max(3, self.segments // 2),
            )
        )
        return tuple(triangles)


@dataclass(frozen=True)
class HangingConeLampPrimitive:
    """A hanging conical lamp shade with a cord and glowing bulb."""

    cord_start: Vec3 | tuple[float, float, float]
    shade_center: Vec3 | tuple[float, float, float]
    color: Color | tuple[int, int, int] = Color(255, 226, 178)
    shade_color: Color | tuple[int, int, int] = Color(92, 76, 58)
    cord_color: Color | tuple[int, int, int] = Color(36, 32, 30)
    shade_height: float = 0.42
    top_radius: float = 0.18
    bottom_radius: float = 0.48
    cord_radius: float = 0.015
    segments: int = 14

    def __post_init__(self) -> None:
        object.__setattr__(self, "cord_start", as_vec3(self.cord_start))
        object.__setattr__(self, "shade_center", as_vec3(self.shade_center))
        object.__setattr__(self, "color", Color.from_value(self.color))
        object.__setattr__(self, "shade_color", Color.from_value(self.shade_color))
        object.__setattr__(self, "cord_color", Color.from_value(self.cord_color))
        if self.shade_height <= 0.0:
            raise ValueError("lamp shade height must be positive")
        if self.top_radius <= 0.0 or self.bottom_radius <= 0.0 or self.cord_radius <= 0.0:
            raise ValueError("lamp radii must be positive")
        if self.segments < 3:
            raise ValueError("lamp segments must be at least 3")

    def to_triangles(self, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        axis = (self.shade_center - self.cord_start).normalized(Vec3(0.0, -1.0, 0.0))
        top_center = self.shade_center - axis * (self.shade_height * 0.5)
        bottom_center = self.shade_center + axis * (self.shade_height * 0.5)
        shade_material = Material(color=self.shade_color, roughness=0.62, fuzziness=0.08, specular=0.08, shininess=18.0)
        inner_material = Material(color=self.color, emission=self.color.scale(0.22), roughness=0.34, specular=0.16, shininess=28.0)
        cord_material = Material(color=self.cord_color, roughness=0.4, specular=0.16, shininess=24.0)
        bulb_material = Material(color=self.color, emission=self.color, diffuse=0.15, roughness=0.08, specular=0.35, shininess=56.0)

        triangles: list[Triangle] = []
        triangles.extend(_oriented_cylinder_triangles(self.cord_start, top_center, self.cord_radius, self.segments, cord_material))
        triangles.extend(_oriented_frustum_triangles(top_center, bottom_center, self.top_radius, self.bottom_radius, self.segments, shade_material))
        inner_top = top_center + axis * (self.shade_height * 0.08)
        inner_bottom = bottom_center - axis * (self.shade_height * 0.08)
        triangles.extend(
            _oriented_frustum_triangles(
                inner_bottom,
                inner_top,
                self.bottom_radius * 0.86,
                self.top_radius * 0.58,
                self.segments,
                inner_material,
                reverse=True,
            )
        )
        bulb_center = self.shade_center + axis * (self.shade_height * 0.25)
        triangles.extend(
            Sphere(bulb_center, self.bottom_radius * 0.18, bulb_material).to_triangles(
                segments=max(8, self.segments),
                rings=max(4, self.segments // 2),
            )
        )
        return tuple(triangles)


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
class BlobSurface:
    """A volume-preserving, deformable blob surface."""

    center: Vec3 | tuple[float, float, float]
    radius: float
    material: Material = Material()
    stretch: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    surface_tension: float = 0.55
    wetting: float = 0.0
    stickiness: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", as_vec3(self.center))
        object.__setattr__(self, "stretch", as_vec3(self.stretch))
        if self.radius <= 0.0:
            raise ValueError("blob radius must be positive")

    def to_triangles(self, segments: int = 16, rings: int = 8, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        if segments < 3:
            raise ValueError("blob segments must be at least 3")
        if rings < 2:
            raise ValueError("blob rings must be at least 2")
        vertices: list[list[Vec3]] = []
        normals: list[list[Vec3]] = []
        for ring in range(rings + 1):
            phi = pi * ring / rings
            row = []
            normal_row = []
            for segment in range(segments):
                theta = 2.0 * pi * segment / segments
                normal = Vec3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta))
                point, vertex_normal = self._deform_normal(normal)
                row.append(point)
                normal_row.append(vertex_normal)
            vertices.append(row)
            normals.append(normal_row)
        triangles: list[Triangle] = []
        for ring in range(rings):
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
                if ring != 0:
                    triangles.append(Triangle(top_left, bottom_left, top_right, self.material, normal_a=normal_top_left, normal_b=normal_bottom_left, normal_c=normal_top_right))
                if ring != rings - 1:
                    triangles.append(Triangle(top_right, bottom_left, bottom_right, self.material, normal_a=normal_top_right, normal_b=normal_bottom_left, normal_c=normal_bottom_right))
        return tuple(triangles)

    def _deform_normal(self, normal: Vec3) -> tuple[Vec3, Vec3]:
        stretch_length = self.stretch.length()
        if stretch_length <= 1e-9:
            flattened = Vec3(normal.x, normal.y * (1.0 - self.wetting * 0.18), normal.z).normalized(normal)
            return self.center + flattened * self.radius, flattened
        axis = self.stretch / stretch_length
        stretch_amount = min(1.75, stretch_length / max(self.radius, 1e-9))
        axis_scale = 1.0 + stretch_amount * (0.7 - self.surface_tension * 0.28)
        cross_scale = max(0.32, 1.0 / (axis_scale**0.5))
        parallel = axis * normal.dot(axis)
        perpendicular = normal - parallel
        deformed = parallel * axis_scale + perpendicular * cross_scale
        if normal.y < 0.0 and self.wetting > 0.0:
            deformed = Vec3(deformed.x * (1.0 + self.wetting * 0.35), deformed.y * (1.0 - self.wetting * 0.22), deformed.z * (1.0 + self.wetting * 0.35))
        vertex_normal = deformed.normalized(normal)
        return self.center + deformed * self.radius, vertex_normal


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
                Triangle(a, b, c, self.material, (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), self.normal, self.normal, self.normal),
                Triangle(a, c, d, self.material, (0.0, 1.0), (1.0, 0.0), (0.0, 0.0), self.normal, self.normal, self.normal),
            )

        bottom_offset = -offset
        e = self.point - tangent * half - bitangent * half + bottom_offset
        f = self.point + tangent * half - bitangent * half + bottom_offset
        g = self.point + tangent * half + bitangent * half + bottom_offset
        h = self.point - tangent * half + bitangent * half + bottom_offset
        bottom_normal = -self.normal
        side_normals = (tangent, bitangent, -tangent, -bitangent)
        return (
            Triangle(a, b, c, self.material, (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), self.normal, self.normal, self.normal),
            Triangle(a, c, d, self.material, (0.0, 1.0), (1.0, 0.0), (0.0, 0.0), self.normal, self.normal, self.normal),
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


def _cylinder_triangles(center: Vec3, radius: float, height: float, segments: int, material: Material) -> list[Triangle]:
    bottom_y = center.y - height * 0.5
    top_y = center.y + height * 0.5
    bottom_center = Vec3(center.x, bottom_y, center.z)
    top_center = Vec3(center.x, top_y, center.z)
    bottom = _horizontal_ring(center.x, bottom_y, center.z, radius, segments)
    top = _horizontal_ring(center.x, top_y, center.z, radius, segments)
    triangles: list[Triangle] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        triangles.append(Triangle(bottom[index], top[index], bottom[next_index], material))
        triangles.append(Triangle(bottom[next_index], top[index], top[next_index], material))
        triangles.append(Triangle(top[index], top_center, top[next_index], material))
        triangles.append(Triangle(bottom[next_index], bottom_center, bottom[index], material))
    return triangles


def _frustum_triangles(
    bottom_center: Vec3,
    top_center: Vec3,
    bottom_radius: float,
    top_radius: float,
    segments: int,
    material: Material,
) -> list[Triangle]:
    bottom = _horizontal_ring(bottom_center.x, bottom_center.y, bottom_center.z, bottom_radius, segments)
    top = _horizontal_ring(top_center.x, top_center.y, top_center.z, top_radius, segments)
    triangles: list[Triangle] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        triangles.append(Triangle(bottom[index], top[index], bottom[next_index], material))
        triangles.append(Triangle(bottom[next_index], top[index], top[next_index], material))
    return triangles


def _oriented_cylinder_triangles(start: Vec3, end: Vec3, radius: float, segments: int, material: Material) -> list[Triangle]:
    start_ring = _oriented_ring(start, (end - start).normalized(Vec3(0.0, -1.0, 0.0)), radius, segments)
    end_ring = _oriented_ring(end, (end - start).normalized(Vec3(0.0, -1.0, 0.0)), radius, segments)
    triangles: list[Triangle] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        triangles.append(Triangle(start_ring[index], end_ring[index], start_ring[next_index], material))
        triangles.append(Triangle(start_ring[next_index], end_ring[index], end_ring[next_index], material))
    return triangles


def _oriented_frustum_triangles(
    start: Vec3,
    end: Vec3,
    start_radius: float,
    end_radius: float,
    segments: int,
    material: Material,
    *,
    reverse: bool = False,
) -> list[Triangle]:
    axis = (end - start).normalized(Vec3(0.0, -1.0, 0.0))
    start_ring = _oriented_ring(start, axis, start_radius, segments)
    end_ring = _oriented_ring(end, axis, end_radius, segments)
    triangles: list[Triangle] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        if reverse:
            triangles.append(Triangle(start_ring[next_index], end_ring[index], start_ring[index], material))
            triangles.append(Triangle(end_ring[next_index], end_ring[index], start_ring[next_index], material))
        else:
            triangles.append(Triangle(start_ring[index], end_ring[index], start_ring[next_index], material))
            triangles.append(Triangle(start_ring[next_index], end_ring[index], end_ring[next_index], material))
    return triangles


def _oriented_ring(center: Vec3, axis: Vec3, radius: float, segments: int) -> list[Vec3]:
    tangent = axis.cross(Vec3(0.0, 1.0, 0.0)).normalized()
    if tangent.length_squared() <= 1e-12:
        tangent = axis.cross(Vec3(1.0, 0.0, 0.0)).normalized(Vec3(1.0, 0.0, 0.0))
    bitangent = axis.cross(tangent).normalized(Vec3(0.0, 0.0, 1.0))
    return [
        center + tangent * (cos(2.0 * pi * index / segments) * radius) + bitangent * (sin(2.0 * pi * index / segments) * radius)
        for index in range(segments)
    ]


def _horizontal_ring(cx: float, cy: float, cz: float, radius: float, segments: int) -> list[Vec3]:
    return [
        Vec3(cx + cos(2.0 * pi * index / segments) * radius, cy, cz + sin(2.0 * pi * index / segments) * radius)
        for index in range(segments)
    ]


def _rotate_euler(value: Vec3, rotation: Vec3) -> Vec3:
    cx, sx = cos(rotation.x), sin(rotation.x)
    cy, sy = cos(rotation.y), sin(rotation.y)
    cz, sz = cos(rotation.z), sin(rotation.z)
    x_rotated = Vec3(value.x, value.y * cx - value.z * sx, value.y * sx + value.z * cx)
    y_rotated = Vec3(x_rotated.x * cy + x_rotated.z * sy, x_rotated.y, -x_rotated.x * sy + x_rotated.z * cy)
    return Vec3(y_rotated.x * cz - y_rotated.y * sz, y_rotated.x * sz + y_rotated.y * cz, y_rotated.z)
