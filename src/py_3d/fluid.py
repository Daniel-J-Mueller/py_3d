"""Small bounded blob fluid primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .primitives import BlobSurface


def _radius_from_volume(volume: float) -> float:
    return ((3.0 * volume) / (4.0 * pi)) ** (1.0 / 3.0)


def _volume_from_radius(radius: float) -> float:
    return 4.0 * pi * radius * radius * radius / 3.0


@dataclass
class FluidBlob:
    """A fixed-volume, bounded blob used for early slime-like fluid demos."""

    position: Vec3 | tuple[float, float, float]
    volume: float
    velocity: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    stretch: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    stretchiness: float = 0.45
    viscosity: float = 0.22
    surface_tension: float = 0.55
    wetting: float = 0.0
    stickiness: float = 0.0
    bounciness: float = 0.12
    material: Material = Material(color=(70, 190, 150), roughness=0.18, fuzziness=0.08, light_transmission=0.18)

    def __post_init__(self) -> None:
        self.position = as_vec3(self.position)
        self.velocity = as_vec3(self.velocity)
        self.stretch = as_vec3(self.stretch)
        if self.volume <= 0.0:
            raise ValueError("fluid blob volume must be positive")
        self.stretchiness = clamp(float(self.stretchiness), 0.0, 1.0)
        self.viscosity = clamp(float(self.viscosity), 0.0, 1.0)
        self.surface_tension = clamp(float(self.surface_tension), 0.0, 1.0)
        self.wetting = clamp(float(self.wetting), 0.0, 1.0)
        self.stickiness = clamp(float(self.stickiness), 0.0, 1.0)
        self.bounciness = clamp(float(self.bounciness), 0.0, 1.0)

    @classmethod
    def from_radius(cls, position: Vec3 | tuple[float, float, float], radius: float, **kwargs) -> "FluidBlob":
        if radius <= 0.0:
            raise ValueError("fluid blob radius must be positive")
        return cls(position=position, volume=_volume_from_radius(radius), **kwargs)

    @property
    def radius(self) -> float:
        return _radius_from_volume(self.volume)

    def to_primitive(self) -> BlobSurface:
        return BlobSurface(self.position, self.radius, self.material, self.stretch, self.surface_tension, self.wetting, self.stickiness)


@dataclass
class FluidWorld:
    """A bounded blob world with split and heal behavior."""

    bounds_min: Vec3 | tuple[float, float, float] = Vec3(-2.0, 0.0, -2.0)
    bounds_max: Vec3 | tuple[float, float, float] = Vec3(2.0, 2.0, 2.0)
    gravity: Vec3 | tuple[float, float, float] = Vec3(0.0, -9.81, 0.0)
    heal_distance_factor: float = 0.65

    def __post_init__(self) -> None:
        self.bounds_min = as_vec3(self.bounds_min)
        self.bounds_max = as_vec3(self.bounds_max)
        self.gravity = as_vec3(self.gravity)
        self.blobs: list[FluidBlob] = []

    def add_blob(self, blob: FluidBlob) -> FluidBlob:
        self.blobs.append(blob)
        return blob

    def total_volume(self) -> float:
        return sum(blob.volume for blob in self.blobs)

    def step(self, dt: float, substeps: int = 1) -> None:
        if dt < 0.0:
            raise ValueError("fluid dt must be non-negative")
        if substeps <= 0:
            raise ValueError("fluid substeps must be positive")
        step_dt = dt / substeps
        for _ in range(substeps):
            for blob in list(self.blobs):
                self._step_blob(blob, step_dt)
            self._split_overstretched()
            self._heal_close_blobs()

    def _step_blob(self, blob: FluidBlob, dt: float) -> None:
        sticky_drag = blob.stickiness * (1.0 if blob.position.y <= self.bounds_min.y + blob.radius * 1.08 else 0.0)
        blob.velocity = (blob.velocity + self.gravity * dt) * max(0.0, 1.0 - (blob.viscosity + sticky_drag) * dt)
        blob.position = blob.position + blob.velocity * dt
        blob.stretch = (blob.stretch + blob.velocity * dt) * max(0.0, 1.0 - blob.surface_tension * dt)
        radius = blob.radius
        x, vx = _bounded_axis(blob.position.x, blob.velocity.x, self.bounds_min.x + radius, self.bounds_max.x - radius, blob.bounciness)
        y, vy = _bounded_axis(blob.position.y, blob.velocity.y, self.bounds_min.y + radius, self.bounds_max.y - radius, blob.bounciness)
        z, vz = _bounded_axis(blob.position.z, blob.velocity.z, self.bounds_min.z + radius, self.bounds_max.z - radius, blob.bounciness)
        blob.position = Vec3(x, y, z)
        blob.velocity = Vec3(vx, vy, vz)

    def _split_overstretched(self) -> None:
        additions: list[FluidBlob] = []
        for blob in list(self.blobs):
            limit = blob.radius * (0.45 + blob.stretchiness * 2.4)
            stretch_length = blob.stretch.length()
            if stretch_length <= limit or blob.volume <= 0.01:
                continue
            direction = blob.stretch.normalized(Vec3(1.0, 0.0, 0.0))
            split_volume = blob.volume * 0.42
            blob.volume -= split_volume
            blob.stretch = blob.stretch * 0.2
            additions.append(
                FluidBlob(
                    position=blob.position + direction * (blob.radius * 0.75),
                    volume=split_volume,
                    velocity=blob.velocity + direction * 0.45,
                    stretch=-blob.stretch,
                    stretchiness=blob.stretchiness,
                    viscosity=blob.viscosity,
                    surface_tension=blob.surface_tension,
                    wetting=blob.wetting,
                    stickiness=blob.stickiness,
                    bounciness=blob.bounciness,
                    material=blob.material,
                )
            )
        self.blobs.extend(additions)

    def _heal_close_blobs(self) -> None:
        index = 0
        while index < len(self.blobs):
            first = self.blobs[index]
            merged = False
            second_index = index + 1
            while second_index < len(self.blobs):
                second = self.blobs[second_index]
                heal_distance = (first.radius + second.radius) * self.heal_distance_factor
                if first.position.distance_to(second.position) <= heal_distance:
                    total_volume = first.volume + second.volume
                    first.position = (first.position * first.volume + second.position * second.volume) / total_volume
                    first.velocity = (first.velocity * first.volume + second.velocity * second.volume) / total_volume
                    first.stretch = (first.stretch + second.stretch) * (0.2 + (first.surface_tension + second.surface_tension) * 0.15)
                    first.volume = total_volume
                    del self.blobs[second_index]
                    merged = True
                    continue
                second_index += 1
            if not merged:
                index += 1


def _bounded_axis(position: float, velocity: float, low: float, high: float, bounciness: float) -> tuple[float, float]:
    if position < low:
        return low, abs(velocity) * bounciness
    if position > high:
        return high, -abs(velocity) * bounciness
    return position, velocity
