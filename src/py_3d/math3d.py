"""Small vector and numeric helpers for the reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class Vec3:
    """A tiny immutable 3D vector."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        object.__setattr__(self, "z", float(self.z))

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: Vec3 | tuple[float, float, float]) -> "Vec3":
        value = as_vec3(other)
        return Vec3(self.x + value.x, self.y + value.y, self.z + value.z)

    def __sub__(self, other: Vec3 | tuple[float, float, float]) -> "Vec3":
        value = as_vec3(other)
        return Vec3(self.x - value.x, self.y - value.y, self.z - value.z)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self * scalar

    def __truediv__(self, scalar: float) -> "Vec3":
        if scalar == 0.0:
            raise ZeroDivisionError("cannot divide Vec3 by zero")
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def dot(self, other: Vec3 | tuple[float, float, float]) -> float:
        value = as_vec3(other)
        return self.x * value.x + self.y * value.y + self.z * value.z

    def cross(self, other: Vec3 | tuple[float, float, float]) -> "Vec3":
        value = as_vec3(other)
        return Vec3(
            self.y * value.z - self.z * value.y,
            self.z * value.x - self.x * value.z,
            self.x * value.y - self.y * value.x,
        )

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return sqrt(self.length_squared())

    def normalized(self, fallback: Vec3 | tuple[float, float, float] | None = None) -> "Vec3":
        size = self.length()
        if size == 0.0:
            return as_vec3(fallback) if fallback is not None else Vec3(0.0, 0.0, 0.0)
        return self / size

    def distance_to(self, other: Vec3 | tuple[float, float, float]) -> float:
        return (self - other).length()


def as_vec3(value: Vec3 | Iterable[float]) -> Vec3:
    if isinstance(value, Vec3):
        return value
    parts = tuple(value)
    if len(parts) != 3:
        raise ValueError("3D vectors must have exactly three coordinates")
    return Vec3(parts[0], parts[1], parts[2])
