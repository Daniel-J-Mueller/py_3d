"""Small deterministic physics helpers for early examples."""

from __future__ import annotations

from dataclasses import dataclass

from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .noise import SurfacePerturbation
from .primitives import Box, Plane, Sphere


@dataclass
class SphereBody:
    """A dynamic sphere body used by the basic physics world."""

    position: Vec3 | tuple[float, float, float]
    radius: float
    velocity: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    mass: float = 1.0
    restitution: float = 0.35
    friction: float = 0.2
    material: Material = Material()
    visual_perturbation: SurfacePerturbation | None = None

    def __post_init__(self) -> None:
        self.position = as_vec3(self.position)
        self.velocity = as_vec3(self.velocity)
        if self.radius <= 0.0:
            raise ValueError("sphere body radius must be positive")
        if self.mass <= 0.0:
            raise ValueError("sphere body mass must be positive")
        self.restitution = clamp(float(self.restitution), 0.0, 1.0)
        self.friction = clamp(float(self.friction), 0.0, 1.0)

    def to_primitive(self) -> Sphere:
        return Sphere(self.position, self.radius, self.material, self.visual_perturbation)


@dataclass(frozen=True)
class StaticPlane:
    point: Vec3 | tuple[float, float, float]
    normal: Vec3 | tuple[float, float, float]
    friction: float = 0.2
    restitution: float = 0.2
    material: Material = Material()
    size: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", as_vec3(self.point))
        object.__setattr__(self, "normal", as_vec3(self.normal).normalized(Vec3(0.0, 1.0, 0.0)))
        object.__setattr__(self, "friction", clamp(float(self.friction), 0.0, 1.0))
        object.__setattr__(self, "restitution", clamp(float(self.restitution), 0.0, 1.0))

    def to_primitive(self) -> Plane:
        return Plane(self.point, self.normal, self.material, self.size)


@dataclass(frozen=True)
class StaticBox:
    center: Vec3 | tuple[float, float, float]
    size: Vec3 | tuple[float, float, float]
    friction: float = 0.3
    restitution: float = 0.25
    material: Material = Material()

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", as_vec3(self.center))
        object.__setattr__(self, "size", as_vec3(self.size))
        if self.size.x <= 0.0 or self.size.y <= 0.0 or self.size.z <= 0.0:
            raise ValueError("static box size components must be positive")
        object.__setattr__(self, "friction", clamp(float(self.friction), 0.0, 1.0))
        object.__setattr__(self, "restitution", clamp(float(self.restitution), 0.0, 1.0))

    def to_primitive(self) -> Box:
        return Box(self.center, self.size, self.material)


@dataclass
class PhysicsWorld:
    """A fixed-step world for basic sphere interactions."""

    gravity: Vec3 | tuple[float, float, float] = Vec3(0.0, -9.81, 0.0)

    def __post_init__(self) -> None:
        self.gravity = as_vec3(self.gravity)
        self.spheres: list[SphereBody] = []
        self.planes: list[StaticPlane] = []
        self.boxes: list[StaticBox] = []

    def add_sphere(self, body: SphereBody) -> SphereBody:
        self.spheres.append(body)
        return body

    def add_plane(self, plane: StaticPlane) -> StaticPlane:
        self.planes.append(plane)
        return plane

    def add_box(self, box: StaticBox) -> StaticBox:
        self.boxes.append(box)
        return box

    def step(self, dt: float, substeps: int = 1) -> None:
        if dt < 0.0:
            raise ValueError("physics dt must be non-negative")
        if substeps <= 0:
            raise ValueError("physics substeps must be positive")
        step_dt = dt / substeps
        for _ in range(substeps):
            for sphere in self.spheres:
                sphere.velocity = sphere.velocity + self.gravity * step_dt
                sphere.position = sphere.position + sphere.velocity * step_dt
                for plane in self.planes:
                    _resolve_sphere_plane(sphere, plane, step_dt)
                for box in self.boxes:
                    _resolve_sphere_box(sphere, box, step_dt)


World = PhysicsWorld


def _resolve_sphere_plane(sphere: SphereBody, plane: StaticPlane, dt: float) -> None:
    distance = (sphere.position - plane.point).dot(plane.normal)
    if distance >= sphere.radius:
        return
    penetration = sphere.radius - distance
    sphere.position = sphere.position + plane.normal * penetration
    normal_speed = sphere.velocity.dot(plane.normal)
    if normal_speed < 0.0:
        sphere.velocity = sphere.velocity - plane.normal * ((1.0 + sphere.restitution * plane.restitution) * normal_speed)
    tangent = sphere.velocity - plane.normal * sphere.velocity.dot(plane.normal)
    sphere.velocity = sphere.velocity - tangent * min(1.0, (sphere.friction + plane.friction) * dt)


def _resolve_sphere_box(sphere: SphereBody, box: StaticBox, dt: float) -> None:
    half = box.size * 0.5
    closest = Vec3(
        clamp(sphere.position.x, box.center.x - half.x, box.center.x + half.x),
        clamp(sphere.position.y, box.center.y - half.y, box.center.y + half.y),
        clamp(sphere.position.z, box.center.z - half.z, box.center.z + half.z),
    )
    offset = sphere.position - closest
    distance_squared = offset.length_squared()
    if distance_squared >= sphere.radius * sphere.radius:
        return

    if distance_squared == 0.0:
        normal = _box_escape_normal(sphere.position, box.center, half)
    else:
        normal = offset.normalized()
    distance = distance_squared ** 0.5
    penetration = sphere.radius - distance if distance > 0.0 else sphere.radius
    sphere.position = sphere.position + normal * penetration

    normal_speed = sphere.velocity.dot(normal)
    if normal_speed < 0.0:
        sphere.velocity = sphere.velocity - normal * ((1.0 + sphere.restitution * box.restitution) * normal_speed)
    tangent = sphere.velocity - normal * sphere.velocity.dot(normal)
    sphere.velocity = sphere.velocity - tangent * min(1.0, (sphere.friction + box.friction) * dt)


def _box_escape_normal(position: Vec3, center: Vec3, half: Vec3) -> Vec3:
    local = position - center
    distances = (
        (half.x - abs(local.x), Vec3(1.0 if local.x >= 0.0 else -1.0, 0.0, 0.0)),
        (half.y - abs(local.y), Vec3(0.0, 1.0 if local.y >= 0.0 else -1.0, 0.0)),
        (half.z - abs(local.z), Vec3(0.0, 0.0, 1.0 if local.z >= 0.0 else -1.0)),
    )
    return min(distances, key=lambda item: item[0])[1]
