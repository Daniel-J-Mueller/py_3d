"""Collision boundary shapes separate from render geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .math3d import Vec3, as_vec3
from .primitives import Bowl, Box, Plane, Sphere


@dataclass(frozen=True)
class SphereCollider:
    """A spherical collision boundary relative to an owning position."""

    radius: float
    offset: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("sphere collider radius must be positive")
        object.__setattr__(self, "offset", as_vec3(self.offset))

    @classmethod
    def from_sphere(cls, sphere: Sphere, owner_position: Vec3 | tuple[float, float, float] | None = None) -> "SphereCollider":
        owner = as_vec3(owner_position) if owner_position is not None else sphere.center
        perturbation_margin = sphere.perturbation.magnitude if sphere.perturbation is not None else 0.0
        return cls(radius=sphere.radius + perturbation_margin, offset=sphere.center - owner)

    def world_center(self, owner_position: Vec3 | tuple[float, float, float]) -> Vec3:
        return as_vec3(owner_position) + self.offset


@dataclass(frozen=True)
class CompoundSphereCollider:
    """A compound boundary made from sphere samples relative to an owner."""

    spheres: tuple[SphereCollider, ...]

    def __init__(self, spheres: Iterable[SphereCollider]):
        values = tuple(spheres)
        if not values:
            raise ValueError("compound sphere collider requires at least one sphere")
        object.__setattr__(self, "spheres", values)

    @classmethod
    def from_offsets(
        cls,
        offsets: Iterable[Vec3 | tuple[float, float, float]],
        radius: float,
    ) -> "CompoundSphereCollider":
        return cls(SphereCollider(radius=radius, offset=offset) for offset in offsets)

    @property
    def radius(self) -> float:
        return max(sphere.offset.length() + sphere.radius for sphere in self.spheres)

    def world_center(self, owner_position: Vec3 | tuple[float, float, float]) -> Vec3:
        return as_vec3(owner_position)

    def world_spheres(self, owner_position: Vec3 | tuple[float, float, float]) -> tuple[tuple[Vec3, float], ...]:
        owner = as_vec3(owner_position)
        return tuple((owner + sphere.offset, sphere.radius) for sphere in self.spheres)


@dataclass(frozen=True)
class BoxCollider:
    """An axis-aligned box collision boundary relative to an owning position."""

    size: Vec3 | tuple[float, float, float]
    offset: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", as_vec3(self.size))
        object.__setattr__(self, "offset", as_vec3(self.offset))
        if self.size.x <= 0.0 or self.size.y <= 0.0 or self.size.z <= 0.0:
            raise ValueError("box collider size components must be positive")

    @classmethod
    def from_box(cls, box: Box, owner_position: Vec3 | tuple[float, float, float] | None = None) -> "BoxCollider":
        owner = as_vec3(owner_position) if owner_position is not None else box.center
        return cls(size=box.size, offset=box.center - owner)

    def world_center(self, owner_position: Vec3 | tuple[float, float, float]) -> Vec3:
        return as_vec3(owner_position) + self.offset


@dataclass(frozen=True)
class PlaneCollider:
    """A plane collision boundary in world space."""

    point: Vec3 | tuple[float, float, float]
    normal: Vec3 | tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", as_vec3(self.point))
        object.__setattr__(self, "normal", as_vec3(self.normal).normalized(Vec3(0.0, 1.0, 0.0)))

    @classmethod
    def from_plane(cls, plane: Plane) -> "PlaneCollider":
        return cls(point=plane.point, normal=plane.normal)


@dataclass(frozen=True)
class BowlCollider:
    """An interior spherical-cap collision boundary relative to an owner."""

    radius: float
    depth: float = 1.0
    offset: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("bowl collider radius must be positive")
        if self.depth <= 0.0 or self.depth > 1.0:
            raise ValueError("bowl collider depth must be in the range (0, 1]")
        object.__setattr__(self, "offset", as_vec3(self.offset))

    @classmethod
    def from_bowl(cls, bowl: Bowl, owner_position: Vec3 | tuple[float, float, float] | None = None) -> "BowlCollider":
        owner = as_vec3(owner_position) if owner_position is not None else bowl.center
        return cls(radius=bowl.radius, depth=bowl.depth, offset=bowl.center - owner)

    def world_center(self, owner_position: Vec3 | tuple[float, float, float]) -> Vec3:
        return as_vec3(owner_position) + self.offset
