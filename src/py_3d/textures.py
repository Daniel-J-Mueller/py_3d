"""Texture coordinate helpers."""

from __future__ import annotations

from .math3d import Vec3, as_vec3
from .primitives import Triangle


def planar_project_triangles(
    triangles: tuple[Triangle, ...] | list[Triangle],
    *,
    center: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0),
    u_axis: Vec3 | tuple[float, float, float] = Vec3(1.0, 0.0, 0.0),
    v_axis: Vec3 | tuple[float, float, float] = Vec3(0.0, 1.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
    offset: tuple[float, float] = (0.5, 0.5),
) -> tuple[Triangle, ...]:
    """Assign UVs by projecting triangle vertices onto a local plane.

    This supports the "pick a face center, offset, and project around it" style
    of texturing for simple polygons and meshes.
    """

    origin = as_vec3(center)
    u = as_vec3(u_axis).normalized(Vec3(1.0, 0.0, 0.0))
    v = as_vec3(v_axis).normalized(Vec3(0.0, 1.0, 0.0))
    if scale[0] == 0.0 or scale[1] == 0.0:
        raise ValueError("texture projection scale components must be non-zero")

    return tuple(
        Triangle(
            triangle.a,
            triangle.b,
            triangle.c,
            triangle.material,
            _project_uv(triangle.a, origin, u, v, scale, offset),
            _project_uv(triangle.b, origin, u, v, scale, offset),
            _project_uv(triangle.c, origin, u, v, scale, offset),
        )
        for triangle in triangles
    )


def _project_uv(
    point: Vec3,
    center: Vec3,
    u_axis: Vec3,
    v_axis: Vec3,
    scale: tuple[float, float],
    offset: tuple[float, float],
) -> tuple[float, float]:
    relative = point - center
    return (
        relative.dot(u_axis) / scale[0] + offset[0],
        relative.dot(v_axis) / scale[1] + offset[1],
    )
