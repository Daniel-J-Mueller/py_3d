"""Small deterministic physics helpers for early examples."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

from .collision import BowlCollider, BoxCollider, PlaneCollider, SphereCollider
from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .noise import SurfacePerturbation
from .primitives import Bowl, Box, Plane, Sphere


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
    collision_boundary: SphereCollider | None = None

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

    def synced_collision_boundary(self) -> SphereCollider:
        """Return the render-derived collider without changing override state."""

        return SphereCollider.from_sphere(self.to_primitive(), owner_position=self.position)

    def effective_collision_boundary(self) -> SphereCollider:
        return self.collision_boundary or self.synced_collision_boundary()

    def sync_collision_boundary(self, *, force: bool = False) -> SphereCollider:
        """Store a collider copied from current render geometry.

        By default this only fills an empty collider slot. Use ``force=True`` to
        intentionally replace an override with the current render-derived shape.
        """

        if self.collision_boundary is None or force:
            self.collision_boundary = self.synced_collision_boundary()
        return self.collision_boundary

    def collision_center(self) -> Vec3:
        return self.effective_collision_boundary().world_center(self.position)

    def collision_radius(self) -> float:
        return self.effective_collision_boundary().radius

    def to_collision_primitive(self, material: Material | None = None) -> Sphere:
        collider = self.effective_collision_boundary()
        return Sphere(collider.world_center(self.position), collider.radius, material or self.material)


@dataclass(frozen=True)
class StaticPlane:
    point: Vec3 | tuple[float, float, float]
    normal: Vec3 | tuple[float, float, float]
    friction: float = 0.2
    restitution: float = 0.2
    material: Material = Material()
    size: float | None = None
    collision_boundary: PlaneCollider | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", as_vec3(self.point))
        object.__setattr__(self, "normal", as_vec3(self.normal).normalized(Vec3(0.0, 1.0, 0.0)))
        object.__setattr__(self, "friction", clamp(float(self.friction), 0.0, 1.0))
        object.__setattr__(self, "restitution", clamp(float(self.restitution), 0.0, 1.0))

    def to_primitive(self) -> Plane:
        return Plane(self.point, self.normal, self.material, self.size)

    def synced_collision_boundary(self) -> PlaneCollider:
        return PlaneCollider.from_plane(self.to_primitive())

    def effective_collision_boundary(self) -> PlaneCollider:
        return self.collision_boundary or self.synced_collision_boundary()

    def sync_collision_boundary(self, *, force: bool = False) -> PlaneCollider:
        if self.collision_boundary is None or force:
            object.__setattr__(self, "collision_boundary", self.synced_collision_boundary())
        return self.collision_boundary

    def to_collision_primitive(self, material: Material | None = None) -> Plane:
        collider = self.effective_collision_boundary()
        return Plane(collider.point, collider.normal, material or self.material, self.size)


@dataclass(frozen=True)
class StaticBox:
    center: Vec3 | tuple[float, float, float]
    size: Vec3 | tuple[float, float, float]
    friction: float = 0.3
    restitution: float = 0.25
    material: Material = Material()
    collision_boundary: BoxCollider | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", as_vec3(self.center))
        object.__setattr__(self, "size", as_vec3(self.size))
        if self.size.x <= 0.0 or self.size.y <= 0.0 or self.size.z <= 0.0:
            raise ValueError("static box size components must be positive")
        object.__setattr__(self, "friction", clamp(float(self.friction), 0.0, 1.0))
        object.__setattr__(self, "restitution", clamp(float(self.restitution), 0.0, 1.0))

    def to_primitive(self) -> Box:
        return Box(self.center, self.size, self.material)

    def synced_collision_boundary(self) -> BoxCollider:
        return BoxCollider.from_box(self.to_primitive(), owner_position=self.center)

    def effective_collision_boundary(self) -> BoxCollider:
        return self.collision_boundary or self.synced_collision_boundary()

    def sync_collision_boundary(self, *, force: bool = False) -> BoxCollider:
        if self.collision_boundary is None or force:
            object.__setattr__(self, "collision_boundary", self.synced_collision_boundary())
        return self.collision_boundary

    def collision_center(self) -> Vec3:
        return self.effective_collision_boundary().world_center(self.center)

    def collision_size(self) -> Vec3:
        return self.effective_collision_boundary().size

    def to_collision_primitive(self, material: Material | None = None) -> Box:
        collider = self.effective_collision_boundary()
        return Box(collider.world_center(self.center), collider.size, material or self.material)


@dataclass
class KinematicBowl:
    """A driven bowl that dynamic bodies can collide with."""

    center: Vec3 | tuple[float, float, float]
    radius: float
    velocity: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    friction: float = 0.2
    restitution: float = 0.45
    material: Material = Material()
    depth: float = 1.0
    collision_boundary: BowlCollider | None = None

    def __post_init__(self) -> None:
        self.center = as_vec3(self.center)
        self.velocity = as_vec3(self.velocity)
        if self.radius <= 0.0:
            raise ValueError("kinematic bowl radius must be positive")
        if self.depth <= 0.0 or self.depth > 1.0:
            raise ValueError("kinematic bowl depth must be in the range (0, 1]")
        self.friction = clamp(float(self.friction), 0.0, 1.0)
        self.restitution = clamp(float(self.restitution), 0.0, 1.0)

    def set_center(self, center: Vec3 | tuple[float, float, float], dt: float | None = None) -> None:
        next_center = as_vec3(center)
        if dt is not None and dt > 0.0:
            self.velocity = (next_center - self.center) / dt
        self.center = next_center

    def to_primitive(self) -> Bowl:
        return Bowl(self.center, self.radius, self.material, self.depth)

    def synced_collision_boundary(self) -> BowlCollider:
        return BowlCollider.from_bowl(self.to_primitive(), owner_position=self.center)

    def effective_collision_boundary(self) -> BowlCollider:
        return self.collision_boundary or self.synced_collision_boundary()

    def sync_collision_boundary(self, *, force: bool = False) -> BowlCollider:
        if self.collision_boundary is None or force:
            self.collision_boundary = self.synced_collision_boundary()
        return self.collision_boundary

    def collision_center(self) -> Vec3:
        return self.effective_collision_boundary().world_center(self.center)

    def to_collision_primitive(self, material: Material | None = None) -> Bowl:
        collider = self.effective_collision_boundary()
        return Bowl(collider.world_center(self.center), collider.radius, material or self.material, collider.depth)


@dataclass
class PhysicsWorld:
    """A fixed-step world for basic sphere interactions."""

    gravity: Vec3 | tuple[float, float, float] = Vec3(0.0, -9.81, 0.0)

    def __post_init__(self) -> None:
        self.gravity = as_vec3(self.gravity)
        self.spheres: list[SphereBody] = []
        self.planes: list[StaticPlane] = []
        self.boxes: list[StaticBox] = []
        self.bowls: list[KinematicBowl] = []

    def add_sphere(self, body: SphereBody) -> SphereBody:
        self.spheres.append(body)
        return body

    def add_plane(self, plane: StaticPlane) -> StaticPlane:
        self.planes.append(plane)
        return plane

    def add_box(self, box: StaticBox) -> StaticBox:
        self.boxes.append(box)
        return box

    def add_bowl(self, bowl: KinematicBowl) -> KinematicBowl:
        self.bowls.append(bowl)
        return bowl

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
                _resolve_sphere_environment(sphere, self.planes, self.boxes, self.bowls, step_dt)
            for first_index in range(len(self.spheres)):
                for second_index in range(first_index + 1, len(self.spheres)):
                    _resolve_sphere_sphere(self.spheres[first_index], self.spheres[second_index], step_dt)
            for sphere in self.spheres:
                _resolve_sphere_environment(sphere, self.planes, self.boxes, self.bowls, step_dt)


World = PhysicsWorld


def _resolve_sphere_environment(
    sphere: SphereBody,
    planes: list[StaticPlane],
    boxes: list[StaticBox],
    bowls: list[KinematicBowl],
    dt: float,
) -> None:
    for plane in planes:
        _resolve_sphere_plane(sphere, plane, dt)
    for box in boxes:
        _resolve_sphere_box(sphere, box, dt)
    for bowl in bowls:
        _resolve_sphere_bowl(sphere, bowl, dt)


def _resolve_sphere_plane(sphere: SphereBody, plane: StaticPlane, dt: float) -> None:
    sphere_collider = sphere.effective_collision_boundary()
    sphere_center = sphere_collider.world_center(sphere.position)
    plane_collider = plane.effective_collision_boundary()
    distance = (sphere_center - plane_collider.point).dot(plane_collider.normal)
    if distance >= sphere_collider.radius:
        return
    penetration = sphere_collider.radius - distance
    sphere.position = sphere.position + plane_collider.normal * penetration
    normal_speed = sphere.velocity.dot(plane_collider.normal)
    if normal_speed < 0.0:
        sphere.velocity = sphere.velocity - plane_collider.normal * ((1.0 + sphere.restitution * plane.restitution) * normal_speed)
    tangent = sphere.velocity - plane_collider.normal * sphere.velocity.dot(plane_collider.normal)
    sphere.velocity = sphere.velocity - tangent * min(1.0, (sphere.friction + plane.friction) * dt)


def _resolve_sphere_box(sphere: SphereBody, box: StaticBox, dt: float) -> None:
    sphere_collider = sphere.effective_collision_boundary()
    sphere_center = sphere_collider.world_center(sphere.position)
    box_collider = box.effective_collision_boundary()
    box_center = box_collider.world_center(box.center)
    half = box_collider.size * 0.5
    closest = Vec3(
        clamp(sphere_center.x, box_center.x - half.x, box_center.x + half.x),
        clamp(sphere_center.y, box_center.y - half.y, box_center.y + half.y),
        clamp(sphere_center.z, box_center.z - half.z, box_center.z + half.z),
    )
    offset = sphere_center - closest
    distance_squared = offset.length_squared()
    if distance_squared >= sphere_collider.radius * sphere_collider.radius:
        return

    if distance_squared == 0.0:
        normal = _box_escape_normal(sphere_center, box_center, half)
    else:
        normal = offset.normalized()
    distance = distance_squared ** 0.5
    penetration = sphere_collider.radius - distance if distance > 0.0 else sphere_collider.radius
    sphere.position = sphere.position + normal * penetration

    normal_speed = sphere.velocity.dot(normal)
    if normal_speed < 0.0:
        sphere.velocity = sphere.velocity - normal * ((1.0 + sphere.restitution * box.restitution) * normal_speed)
    tangent = sphere.velocity - normal * sphere.velocity.dot(normal)
    sphere.velocity = sphere.velocity - tangent * min(1.0, (sphere.friction + box.friction) * dt)


def _resolve_sphere_bowl(sphere: SphereBody, bowl: KinematicBowl, dt: float) -> None:
    sphere_collider = sphere.effective_collision_boundary()
    sphere_center = sphere_collider.world_center(sphere.position)
    bowl_collider = bowl.effective_collision_boundary()
    bowl_center = bowl_collider.world_center(bowl.center)
    local = sphere_center - bowl_center
    sphere_radius = sphere_collider.radius
    radius = bowl_collider.radius

    if local.y <= sphere_radius:
        distance = local.length()
        if distance + sphere_radius > radius:
            normal = local.normalized(Vec3(0.0, -1.0, 0.0))
            penetration = distance + sphere_radius - radius
            sphere.position = sphere.position - normal * penetration
            _bounce_against_kinematic_surface(sphere, bowl, normal, dt, outward_speed=True)

    sphere_center = sphere_collider.world_center(sphere.position)
    local = sphere_center - bowl_center
    bottom_radius = radius * cos(bowl_collider.depth * pi / 2.0)
    if bottom_radius <= 1e-9:
        return
    bottom_y = bowl_center.y - radius * sin(bowl_collider.depth * pi / 2.0)
    horizontal_distance_squared = local.x * local.x + local.z * local.z
    bottom_contact_radius = bottom_radius + sphere_radius
    if local.y < bottom_y - bowl_center.y + sphere_radius and horizontal_distance_squared <= bottom_contact_radius * bottom_contact_radius:
        penetration = bottom_y + sphere_radius - sphere_center.y
        sphere.position = sphere.position + Vec3(0.0, penetration, 0.0)
        _bounce_against_kinematic_surface(sphere, bowl, Vec3(0.0, 1.0, 0.0), dt, outward_speed=False)


def _resolve_sphere_sphere(first: SphereBody, second: SphereBody, dt: float) -> None:
    first_collider = first.effective_collision_boundary()
    second_collider = second.effective_collision_boundary()
    first_center = first_collider.world_center(first.position)
    second_center = second_collider.world_center(second.position)
    delta = second_center - first_center
    distance_squared = delta.length_squared()
    radius_sum = first_collider.radius + second_collider.radius
    if distance_squared >= radius_sum * radius_sum:
        return

    if distance_squared == 0.0:
        normal = Vec3(1.0, 0.0, 0.0)
        distance = 0.0
    else:
        distance = distance_squared ** 0.5
        normal = delta / distance
    penetration = radius_sum - distance
    first_inverse_mass = 1.0 / first.mass
    second_inverse_mass = 1.0 / second.mass
    total_inverse_mass = first_inverse_mass + second_inverse_mass
    if total_inverse_mass == 0.0:
        return

    first.position = first.position - normal * (penetration * first_inverse_mass / total_inverse_mass)
    second.position = second.position + normal * (penetration * second_inverse_mass / total_inverse_mass)

    relative_velocity = second.velocity - first.velocity
    normal_speed = relative_velocity.dot(normal)
    if normal_speed >= 0.0:
        return
    restitution = first.restitution * second.restitution
    impulse_strength = -((1.0 + restitution) * normal_speed) / total_inverse_mass
    impulse = normal * impulse_strength
    first.velocity = first.velocity - impulse * first_inverse_mass
    second.velocity = second.velocity + impulse * second_inverse_mass

    tangent = relative_velocity - normal * normal_speed
    damping = min(1.0, (first.friction + second.friction) * dt)
    if tangent.length_squared() > 0.0 and damping > 0.0:
        first.velocity = first.velocity + tangent * (damping * 0.5)
        second.velocity = second.velocity - tangent * (damping * 0.5)


def _bounce_against_kinematic_surface(
    sphere: SphereBody,
    bowl: KinematicBowl,
    normal: Vec3,
    dt: float,
    *,
    outward_speed: bool,
) -> None:
    relative_velocity = sphere.velocity - bowl.velocity
    normal_speed = relative_velocity.dot(normal)
    if outward_speed:
        if normal_speed <= 0.0:
            return
    elif normal_speed >= 0.0:
        return

    restitution = sphere.restitution * bowl.restitution
    reflected = relative_velocity - normal * ((1.0 + restitution) * normal_speed)
    tangent = reflected - normal * reflected.dot(normal)
    reflected = reflected - tangent * min(1.0, (sphere.friction + bowl.friction) * dt)
    sphere.velocity = bowl.velocity + reflected


def _box_escape_normal(position: Vec3, center: Vec3, half: Vec3) -> Vec3:
    local = position - center
    distances = (
        (half.x - abs(local.x), Vec3(1.0 if local.x >= 0.0 else -1.0, 0.0, 0.0)),
        (half.y - abs(local.y), Vec3(0.0, 1.0 if local.y >= 0.0 else -1.0, 0.0)),
        (half.z - abs(local.z), Vec3(0.0, 0.0, 1.0 if local.z >= 0.0 else -1.0)),
    )
    return min(distances, key=lambda item: item[0])[1]
