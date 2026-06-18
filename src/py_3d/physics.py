"""Small deterministic physics helpers for early examples."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

from .collision import BowlCollider, BoxCollider, CompoundSphereCollider, PlaneCollider, SphereCollider
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
    collision_boundary: SphereCollider | CompoundSphereCollider | None = None
    angular_velocity: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    rotation: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    moment_of_inertia: float | None = None
    static_friction: float | None = None
    kinetic_friction: float | None = None
    rolling_resistance: float | None = None
    squishiness: float = 0.0
    damping: float = 0.0
    dampening: float | None = None

    def __post_init__(self) -> None:
        self.position = as_vec3(self.position)
        self.velocity = as_vec3(self.velocity)
        self.angular_velocity = as_vec3(self.angular_velocity)
        self.rotation = as_vec3(self.rotation)
        if self.radius <= 0.0:
            raise ValueError("sphere body radius must be positive")
        if self.mass <= 0.0:
            raise ValueError("sphere body mass must be positive")
        if self.moment_of_inertia is None:
            self.moment_of_inertia = 0.4 * self.mass * self.radius * self.radius
        elif self.moment_of_inertia <= 0.0:
            raise ValueError("moment of inertia must be positive")
        self.restitution = clamp(float(self.restitution), 0.0, 1.0)
        self.friction = clamp(float(self.friction), 0.0, 1.0)
        if self.static_friction is None:
            self.static_friction = _auto_static_friction(self.friction, self.material)
        else:
            self.static_friction = clamp(float(self.static_friction), 0.0, 1.0)
        if self.kinetic_friction is None:
            self.kinetic_friction = _auto_kinetic_friction(self.friction, self.material)
        else:
            self.kinetic_friction = clamp(float(self.kinetic_friction), 0.0, 1.0)
        if self.rolling_resistance is None:
            self.rolling_resistance = _auto_rolling_resistance(self.material)
        else:
            self.rolling_resistance = clamp(float(self.rolling_resistance), 0.0, 1.0)
        self.squishiness = clamp(float(self.squishiness), 0.0, 1.0)
        self.damping = clamp(float(self.dampening if self.dampening is not None else self.damping), 0.0, 1.0)
        self.dampening = self.damping

    def to_primitive(self) -> Sphere:
        return Sphere(self.position, self.radius, self.material, self.visual_perturbation)

    def synced_collision_boundary(self) -> SphereCollider:
        """Return the render-derived collider without changing override state."""

        return SphereCollider.from_sphere(self.to_primitive(), owner_position=self.position)

    def effective_collision_boundary(self) -> SphereCollider | CompoundSphereCollider:
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

    def inverse_mass(self) -> float:
        return 1.0 / self.mass

    def inverse_moment_of_inertia(self) -> float:
        return 1.0 / self.moment_of_inertia

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
    squishiness: float = 0.0
    damping: float = 0.0
    dampening: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", as_vec3(self.point))
        object.__setattr__(self, "normal", as_vec3(self.normal).normalized(Vec3(0.0, 1.0, 0.0)))
        object.__setattr__(self, "friction", clamp(float(self.friction), 0.0, 1.0))
        object.__setattr__(self, "restitution", clamp(float(self.restitution), 0.0, 1.0))
        object.__setattr__(self, "squishiness", clamp(float(self.squishiness), 0.0, 1.0))
        damping = self.dampening if self.dampening is not None else self.damping
        object.__setattr__(self, "damping", clamp(float(damping), 0.0, 1.0))
        object.__setattr__(self, "dampening", self.damping)

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
    squishiness: float = 0.0
    damping: float = 0.0
    dampening: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", as_vec3(self.center))
        object.__setattr__(self, "size", as_vec3(self.size))
        if self.size.x <= 0.0 or self.size.y <= 0.0 or self.size.z <= 0.0:
            raise ValueError("static box size components must be positive")
        object.__setattr__(self, "friction", clamp(float(self.friction), 0.0, 1.0))
        object.__setattr__(self, "restitution", clamp(float(self.restitution), 0.0, 1.0))
        object.__setattr__(self, "squishiness", clamp(float(self.squishiness), 0.0, 1.0))
        damping = self.dampening if self.dampening is not None else self.damping
        object.__setattr__(self, "damping", clamp(float(damping), 0.0, 1.0))
        object.__setattr__(self, "dampening", self.damping)

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
    squishiness: float = 0.0
    damping: float = 0.0
    dampening: float | None = None

    def __post_init__(self) -> None:
        self.center = as_vec3(self.center)
        self.velocity = as_vec3(self.velocity)
        if self.radius <= 0.0:
            raise ValueError("kinematic bowl radius must be positive")
        if self.depth <= 0.0 or self.depth > 1.0:
            raise ValueError("kinematic bowl depth must be in the range (0, 1]")
        self.friction = clamp(float(self.friction), 0.0, 1.0)
        self.restitution = clamp(float(self.restitution), 0.0, 1.0)
        self.squishiness = clamp(float(self.squishiness), 0.0, 1.0)
        self.damping = clamp(float(self.dampening if self.dampening is not None else self.damping), 0.0, 1.0)
        self.dampening = self.damping

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
                _integrate_angular_state(sphere, step_dt)


World = PhysicsWorld


def _auto_static_friction(friction: float, material: Material) -> float:
    roughness = getattr(material, "roughness", 0.0)
    fuzziness = getattr(material, "fuzziness", 0.0)
    return clamp(float(friction) + roughness * 0.35 + fuzziness * 0.12, 0.0, 1.0)


def _auto_kinetic_friction(friction: float, material: Material) -> float:
    roughness = getattr(material, "roughness", 0.0)
    fuzziness = getattr(material, "fuzziness", 0.0)
    return clamp(float(friction) + roughness * 0.22 + fuzziness * 0.08, 0.0, 1.0)


def _auto_rolling_resistance(material: Material) -> float:
    roughness = getattr(material, "roughness", 0.0)
    fuzziness = getattr(material, "fuzziness", 0.0)
    return clamp(0.008 + roughness * 0.025 + fuzziness * 0.015, 0.0, 0.2)


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


def _collider_spheres(sphere: SphereBody) -> tuple[tuple[Vec3, float], ...]:
    collider = sphere.effective_collision_boundary()
    if isinstance(collider, CompoundSphereCollider):
        return collider.world_spheres(sphere.position)
    return ((collider.world_center(sphere.position), collider.radius),)


def _resolve_sphere_plane(sphere: SphereBody, plane: StaticPlane, dt: float) -> None:
    plane_collider = plane.effective_collision_boundary()
    for sphere_center, sphere_radius in _collider_spheres(sphere):
        contact_arm = sphere_center - sphere.position - plane_collider.normal * sphere_radius
        distance = (sphere_center - plane_collider.point).dot(plane_collider.normal)
        if distance >= sphere_radius:
            continue
        penetration = sphere_radius - distance
        correction = _soft_penetration_correction(penetration, sphere_radius, sphere, plane)
        sphere.position = sphere.position + plane_collider.normal * correction
        _apply_kinematic_normal_impulse(sphere, plane_collider.normal, Vec3(0.0, 0.0, 0.0), plane, contact_arm)
        _apply_contact_friction(sphere, plane_collider.normal, Vec3(0.0, 0.0, 0.0), plane.friction, contact_arm, dt)


def _resolve_sphere_box(sphere: SphereBody, box: StaticBox, dt: float) -> None:
    box_collider = box.effective_collision_boundary()
    box_center = box_collider.world_center(box.center)
    half = box_collider.size * 0.5
    for sphere_center, sphere_radius in _collider_spheres(sphere):
        closest = Vec3(
            clamp(sphere_center.x, box_center.x - half.x, box_center.x + half.x),
            clamp(sphere_center.y, box_center.y - half.y, box_center.y + half.y),
            clamp(sphere_center.z, box_center.z - half.z, box_center.z + half.z),
        )
        offset = sphere_center - closest
        distance_squared = offset.length_squared()
        if distance_squared >= sphere_radius * sphere_radius:
            continue

        if distance_squared == 0.0:
            normal = _box_escape_normal(sphere_center, box_center, half)
        else:
            normal = offset.normalized()
        contact_arm = sphere_center - sphere.position - normal * sphere_radius
        distance = distance_squared ** 0.5
        penetration = sphere_radius - distance if distance > 0.0 else sphere_radius
        correction = _soft_penetration_correction(penetration, sphere_radius, sphere, box)
        sphere.position = sphere.position + normal * correction

        _apply_kinematic_normal_impulse(sphere, normal, Vec3(0.0, 0.0, 0.0), box, contact_arm)
        _apply_contact_friction(sphere, normal, Vec3(0.0, 0.0, 0.0), box.friction, contact_arm, dt)


def _resolve_sphere_bowl(sphere: SphereBody, bowl: KinematicBowl, dt: float) -> None:
    bowl_collider = bowl.effective_collision_boundary()
    bowl_center = bowl_collider.world_center(bowl.center)
    radius = bowl_collider.radius

    for sphere_center, sphere_radius in _collider_spheres(sphere):
        component_offset = sphere_center - sphere.position
        local = sphere_center - bowl_center
        if local.y <= sphere_radius:
            distance = local.length()
            if distance + sphere_radius > radius:
                normal = local.normalized(Vec3(0.0, -1.0, 0.0))
                penetration = distance + sphere_radius - radius
                push_normal = -normal
                correction = _soft_penetration_correction(penetration, sphere_radius, sphere, bowl)
                sphere.position = sphere.position + push_normal * correction
                contact_arm = component_offset + push_normal * sphere_radius
                _apply_kinematic_normal_impulse(sphere, push_normal, bowl.velocity, bowl, contact_arm)
                _apply_contact_friction(sphere, push_normal, bowl.velocity, bowl.friction, contact_arm, dt)

        sphere_center = sphere.position + component_offset
        local = sphere_center - bowl_center
        bottom_radius = radius * cos(bowl_collider.depth * pi / 2.0)
        if bottom_radius <= 1e-9:
            continue
        bottom_y = bowl_center.y - radius * sin(bowl_collider.depth * pi / 2.0)
        horizontal_distance_squared = local.x * local.x + local.z * local.z
        bottom_contact_radius = bottom_radius + sphere_radius
        if local.y < bottom_y - bowl_center.y + sphere_radius and horizontal_distance_squared <= bottom_contact_radius * bottom_contact_radius:
            penetration = bottom_y + sphere_radius - sphere_center.y
            correction = _soft_penetration_correction(penetration, sphere_radius, sphere, bowl)
            sphere.position = sphere.position + Vec3(0.0, correction, 0.0)
            contact_arm = component_offset - Vec3(0.0, sphere_radius, 0.0)
            _apply_kinematic_normal_impulse(sphere, Vec3(0.0, 1.0, 0.0), bowl.velocity, bowl, contact_arm)
            _apply_contact_friction(sphere, Vec3(0.0, 1.0, 0.0), bowl.velocity, bowl.friction, contact_arm, dt)


def _resolve_sphere_sphere(first: SphereBody, second: SphereBody, dt: float) -> None:
    first_inverse_mass = first.inverse_mass()
    second_inverse_mass = second.inverse_mass()
    total_inverse_mass = first_inverse_mass + second_inverse_mass
    if total_inverse_mass == 0.0:
        return

    for first_center, first_radius in _collider_spheres(first):
        first_offset = first_center - first.position
        for second_center, second_radius in _collider_spheres(second):
            second_offset = second_center - second.position
            delta = second_center - first_center
            distance_squared = delta.length_squared()
            radius_sum = first_radius + second_radius
            if distance_squared >= radius_sum * radius_sum:
                continue

            if distance_squared == 0.0:
                normal = Vec3(1.0, 0.0, 0.0)
                distance = 0.0
            else:
                distance = distance_squared ** 0.5
                normal = delta / distance
            penetration = radius_sum - distance
            correction = _soft_penetration_correction(penetration, min(first_radius, second_radius), first, second)

            first.position = first.position - normal * (correction * first_inverse_mass / total_inverse_mass)
            second.position = second.position + normal * (correction * second_inverse_mass / total_inverse_mass)

            first_contact_velocity = first.velocity + first.angular_velocity.cross(first_offset)
            second_contact_velocity = second.velocity + second.angular_velocity.cross(second_offset)
            relative_velocity = second_contact_velocity - first_contact_velocity
            normal_speed = relative_velocity.dot(normal)
            if normal_speed >= 0.0:
                continue
            restitution = _effective_restitution(first, second)
            first_inverse_inertia = first.inverse_moment_of_inertia()
            second_inverse_inertia = second.inverse_moment_of_inertia()
            effective_inverse_mass = (
                total_inverse_mass
                + first_offset.cross(normal).length_squared() * first_inverse_inertia
                + second_offset.cross(normal).length_squared() * second_inverse_inertia
            )
            if effective_inverse_mass <= 0.0:
                continue
            impulse_strength = -((1.0 + restitution) * normal_speed) / effective_inverse_mass
            impulse = normal * impulse_strength
            first.velocity = first.velocity - impulse * first_inverse_mass
            second.velocity = second.velocity + impulse * second_inverse_mass
            first.angular_velocity = first.angular_velocity - first_offset.cross(impulse) * first_inverse_inertia
            second.angular_velocity = second.angular_velocity + second_offset.cross(impulse) * second_inverse_inertia
            _apply_pair_friction(first, second, normal, first_offset, second_offset, dt)


def _contact_squishiness(first, second) -> float:
    return clamp((getattr(first, "squishiness", 0.0) + getattr(second, "squishiness", 0.0)) * 0.5, 0.0, 1.0)


def _contact_damping(first, second) -> float:
    explicit = (getattr(first, "damping", 0.0) + getattr(second, "damping", 0.0)) * 0.5
    squish_coupling = _contact_squishiness(first, second) * 0.55
    return clamp(explicit + squish_coupling, 0.0, 1.0)


def _effective_restitution(first, second) -> float:
    bounce = getattr(first, "restitution", 0.0) * getattr(second, "restitution", 0.0)
    return clamp(bounce * (1.0 - _contact_damping(first, second)), 0.0, 1.0)


def _soft_penetration_correction(penetration: float, contact_radius: float, first, second) -> float:
    squishiness = _contact_squishiness(first, second)
    allowance = max(0.0, contact_radius) * squishiness * 0.45
    stiffness = 1.0 - squishiness * 0.7
    return max(0.0, penetration - allowance) * stiffness


def _apply_kinematic_normal_impulse(
    sphere: SphereBody,
    normal: Vec3,
    surface_velocity: Vec3,
    surface,
    contact_arm: Vec3,
) -> None:
    contact_velocity = sphere.velocity + sphere.angular_velocity.cross(contact_arm)
    relative_velocity = contact_velocity - surface_velocity
    normal_speed = relative_velocity.dot(normal)
    if normal_speed >= 0.0:
        return

    inverse_inertia = sphere.inverse_moment_of_inertia()
    effective_inverse_mass = sphere.inverse_mass() + contact_arm.cross(normal).length_squared() * inverse_inertia
    if effective_inverse_mass <= 0.0:
        return

    restitution = _effective_restitution(sphere, surface)
    impulse_strength = -((1.0 + restitution) * normal_speed) / effective_inverse_mass
    impulse = normal * impulse_strength
    sphere.velocity = sphere.velocity + impulse * sphere.inverse_mass()
    sphere.angular_velocity = sphere.angular_velocity + contact_arm.cross(impulse) * inverse_inertia


def _apply_contact_friction(
    sphere: SphereBody,
    normal: Vec3,
    surface_velocity: Vec3,
    surface_friction: float,
    contact_arm: Vec3,
    dt: float,
) -> None:
    contact_velocity = sphere.velocity + sphere.angular_velocity.cross(contact_arm)
    relative_velocity = contact_velocity - surface_velocity
    tangent = relative_velocity - normal * relative_velocity.dot(normal)
    tangent_speed = tangent.length()
    if tangent_speed <= 1e-9:
        return

    tangent_direction = tangent / tangent_speed
    contact_radius = max(1e-9, contact_arm.length())
    effective_inverse_mass = sphere.inverse_mass() + contact_radius * contact_radius * sphere.inverse_moment_of_inertia()
    desired_impulse = tangent_speed / effective_inverse_mass
    coefficient = clamp(0.5 * (sphere.kinetic_friction + surface_friction), 0.0, 1.0)
    max_impulse = coefficient * sphere.mass * 9.81 * dt
    impulse = -tangent_direction * min(desired_impulse, max_impulse)

    sphere.velocity = sphere.velocity + impulse * sphere.inverse_mass()
    sphere.angular_velocity = sphere.angular_velocity + contact_arm.cross(impulse) * sphere.inverse_moment_of_inertia()

    static_blend = min(1.0, 0.5 * (sphere.static_friction + surface_friction) * dt * 12.0)
    if static_blend > 0.0:
        relative_linear = sphere.velocity - surface_velocity
        relative_tangent = relative_linear - normal * relative_linear.dot(normal)
        desired_spin = normal.cross(relative_tangent) / contact_radius
        sphere.angular_velocity = sphere.angular_velocity + (desired_spin - sphere.angular_velocity) * static_blend


def _apply_pair_friction(
    first: SphereBody,
    second: SphereBody,
    normal: Vec3,
    first_offset: Vec3,
    second_offset: Vec3,
    dt: float,
) -> None:
    first_contact_velocity = first.velocity + first.angular_velocity.cross(first_offset)
    second_contact_velocity = second.velocity + second.angular_velocity.cross(second_offset)
    relative_velocity = second_contact_velocity - first_contact_velocity
    tangent = relative_velocity - normal * relative_velocity.dot(normal)
    tangent_speed = tangent.length()
    if tangent_speed <= 1e-9:
        return

    tangent_direction = tangent / tangent_speed
    first_inverse_inertia = first.inverse_moment_of_inertia()
    second_inverse_inertia = second.inverse_moment_of_inertia()
    effective_inverse_mass = (
        first.inverse_mass()
        + second.inverse_mass()
        + first_offset.length_squared() * first_inverse_inertia
        + second_offset.length_squared() * second_inverse_inertia
    )
    if effective_inverse_mass <= 0.0:
        return
    coefficient = clamp(0.5 * (first.kinetic_friction + second.kinetic_friction), 0.0, 1.0)
    max_impulse = coefficient * min(first.mass, second.mass) * 9.81 * dt
    impulse = tangent_direction * min(tangent_speed / effective_inverse_mass, max_impulse)

    first.velocity = first.velocity + impulse * first.inverse_mass()
    second.velocity = second.velocity - impulse * second.inverse_mass()
    first.angular_velocity = first.angular_velocity + first_offset.cross(impulse) * first_inverse_inertia
    second.angular_velocity = second.angular_velocity - second_offset.cross(impulse) * second_inverse_inertia


def _integrate_angular_state(sphere: SphereBody, dt: float) -> None:
    if sphere.damping > 0.0:
        sphere.velocity = sphere.velocity * max(0.0, 1.0 - sphere.damping * dt)
    if sphere.rolling_resistance > 0.0:
        sphere.angular_velocity = sphere.angular_velocity * max(0.0, 1.0 - sphere.rolling_resistance * dt)
    if sphere.damping > 0.0:
        sphere.angular_velocity = sphere.angular_velocity * max(0.0, 1.0 - sphere.damping * 2.0 * dt)
    sphere.rotation = sphere.rotation + sphere.angular_velocity * dt


def _box_escape_normal(position: Vec3, center: Vec3, half: Vec3) -> Vec3:
    local = position - center
    distances = (
        (half.x - abs(local.x), Vec3(1.0 if local.x >= 0.0 else -1.0, 0.0, 0.0)),
        (half.y - abs(local.y), Vec3(0.0, 1.0 if local.y >= 0.0 else -1.0, 0.0)),
        (half.z - abs(local.z), Vec3(0.0, 0.0, 1.0 if local.z >= 0.0 else -1.0)),
    )
    return min(distances, key=lambda item: item[0])[1]
