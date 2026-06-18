"""Light source models."""

from __future__ import annotations

from dataclasses import dataclass

from .color import Color
from .math3d import Vec3, as_vec3


@dataclass(frozen=True)
class LightSample:
    """Light information at a point on a surface."""

    direction: Vec3
    color: Color
    intensity: float


@dataclass(frozen=True)
class Lamp:
    """A positional light with basic inverse-square falloff."""

    position: Vec3 | tuple[float, float, float]
    color: Color | tuple[int, int, int] = Color(255, 255, 255)
    intensity: float = 1.0
    radius: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_vec3(self.position))
        object.__setattr__(self, "color", Color.from_value(self.color))
        if self.intensity < 0.0:
            raise ValueError("light intensity must be non-negative")
        if self.radius is not None and self.radius <= 0.0:
            raise ValueError("lamp radius must be greater than zero when provided")

    def sample(self, point: Vec3 | tuple[float, float, float]) -> LightSample:
        point_vec = as_vec3(point)
        offset = self.position - point_vec
        distance = offset.length()
        if distance == 0.0:
            return LightSample(Vec3(0.0, 1.0, 0.0), self.color, self.intensity)
        if self.radius is not None and distance > self.radius:
            return LightSample(offset / distance, self.color, 0.0)
        attenuation = 1.0 / (1.0 + distance * distance)
        return LightSample(offset / distance, self.color, self.intensity * attenuation)


@dataclass(frozen=True)
class Sun:
    """A directional light with parallel rays."""

    direction: Vec3 | tuple[float, float, float]
    color: Color | tuple[int, int, int] = Color(255, 255, 255)
    intensity: float = 1.0

    def __post_init__(self) -> None:
        direction = as_vec3(self.direction).normalized(Vec3(0.0, -1.0, 0.0))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "color", Color.from_value(self.color))
        if self.intensity < 0.0:
            raise ValueError("light intensity must be non-negative")

    def sample(self, point: Vec3 | tuple[float, float, float]) -> LightSample:
        del point
        return LightSample(-self.direction, self.color, self.intensity)
