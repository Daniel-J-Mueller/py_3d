"""Basic player-model primitive assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin, cos

from .materials import Material
from .math3d import Vec3, as_vec3
from .primitives import Capsule, Line3, Sphere


@dataclass(frozen=True)
class PlayerModel:
    """A lightweight humanoid-ish model made from existing primitives."""

    feet: Vec3 | tuple[float, float, float]
    yaw_degrees: float = 0.0
    height: float = 1.45
    radius: float = 0.26
    body_material: Material = Material(color=(92, 160, 240), roughness=0.28, specular=0.2, shininess=24.0)
    head_material: Material = Material(color=(226, 176, 138), roughness=0.38, specular=0.08, shininess=18.0)
    accent_material: Material = Material(color=(35, 46, 58), roughness=0.42, specular=0.1, shininess=18.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feet", as_vec3(self.feet))
        if self.height <= 0.0:
            raise ValueError("player model height must be positive")
        if self.radius <= 0.0:
            raise ValueError("player model radius must be positive")

    @property
    def forward(self) -> Vec3:
        yaw = radians(self.yaw_degrees)
        return Vec3(sin(yaw), 0.0, cos(yaw)).normalized(Vec3(0.0, 0.0, 1.0))

    @property
    def right(self) -> Vec3:
        forward = self.forward
        return Vec3(forward.z, 0.0, -forward.x)

    @property
    def center(self) -> Vec3:
        return Vec3(self.feet.x, self.feet.y + self.height * 0.5, self.feet.z)

    def to_primitives(self) -> tuple[Capsule | Sphere | Line3, ...]:
        head_radius = self.radius * 0.58
        head_center = self.feet + Vec3(0.0, self.height + head_radius * 0.42, 0.0)
        shoulder = self.feet + Vec3(0.0, self.height * 0.72, 0.0)
        hips = self.feet + Vec3(0.0, self.height * 0.38, 0.0)
        left_shoulder = shoulder - self.right * (self.radius * 0.92)
        right_shoulder = shoulder + self.right * (self.radius * 0.92)
        left_hip = hips - self.right * (self.radius * 0.55)
        right_hip = hips + self.right * (self.radius * 0.55)
        return (
            Capsule(self.center, self.radius, self.height, self.body_material),
            Sphere(head_center, head_radius, self.head_material),
            Line3(left_shoulder, right_shoulder, self.accent_material),
            Line3(left_shoulder, left_hip, self.accent_material),
            Line3(right_shoulder, right_hip, self.accent_material),
        )
