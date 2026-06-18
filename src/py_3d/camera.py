"""Camera and projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan

from .math3d import Vec3, as_vec3


@dataclass(frozen=True)
class ProjectedPoint:
    """A point projected into screen space."""

    x: float
    y: float
    depth: float


@dataclass(frozen=True)
class Camera:
    """A perspective camera looking from ``position`` toward ``target``."""

    position: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, -5.0)
    target: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    up: Vec3 | tuple[float, float, float] = Vec3(0.0, 1.0, 0.0)
    fov_degrees: float = 60.0
    near: float = 0.1
    far: float = 1000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_vec3(self.position))
        object.__setattr__(self, "target", as_vec3(self.target))
        object.__setattr__(self, "up", as_vec3(self.up).normalized(Vec3(0.0, 1.0, 0.0)))
        if self.fov_degrees <= 0.0 or self.fov_degrees >= 180.0:
            raise ValueError("camera field of view must be between 0 and 180 degrees")
        if self.near <= 0.0:
            raise ValueError("camera near plane must be greater than zero")
        if self.far <= self.near:
            raise ValueError("camera far plane must be greater than near plane")

    def basis(self) -> tuple[Vec3, Vec3, Vec3]:
        forward = (self.target - self.position).normalized(Vec3(0.0, 0.0, 1.0))
        right = self.up.cross(forward).normalized(Vec3(1.0, 0.0, 0.0))
        true_up = forward.cross(right).normalized(Vec3(0.0, 1.0, 0.0))
        return right, true_up, forward

    def world_to_camera(self, point: Vec3 | tuple[float, float, float]) -> Vec3:
        right, true_up, forward = self.basis()
        relative = as_vec3(point) - self.position
        return Vec3(relative.dot(right), relative.dot(true_up), relative.dot(forward))

    def project(
        self,
        point: Vec3 | tuple[float, float, float],
        width: int,
        height: int,
    ) -> ProjectedPoint | None:
        camera_point = self.world_to_camera(point)
        if camera_point.z < self.near or camera_point.z > self.far:
            return None

        aspect = width / height
        focal = 1.0 / tan(radians(self.fov_degrees) / 2.0)
        ndc_x = (camera_point.x * focal / aspect) / camera_point.z
        ndc_y = (camera_point.y * focal) / camera_point.z

        screen_x = (ndc_x + 1.0) * 0.5 * (width - 1)
        screen_y = (1.0 - ndc_y) * 0.5 * (height - 1)
        return ProjectedPoint(screen_x, screen_y, camera_point.z)
