"""RGB color helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .math3d import clamp


@dataclass(frozen=True)
class Color:
    """An immutable 8-bit RGB color."""

    r: int
    g: int
    b: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", int(clamp(round(self.r), 0, 255)))
        object.__setattr__(self, "g", int(clamp(round(self.g), 0, 255)))
        object.__setattr__(self, "b", int(clamp(round(self.b), 0, 255)))

    @classmethod
    def from_value(cls, value: "Color" | tuple[int, int, int]) -> "Color":
        if isinstance(value, Color):
            return value
        if len(value) != 3:
            raise ValueError("colors must have exactly three channels")
        return cls(value[0], value[1], value[2])

    @classmethod
    def from_floats(cls, r: float, g: float, b: float) -> "Color":
        return cls(int(clamp(r, 0.0, 1.0) * 255), int(clamp(g, 0.0, 1.0) * 255), int(clamp(b, 0.0, 1.0) * 255))

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def to_floats(self) -> tuple[float, float, float]:
        return (self.r / 255.0, self.g / 255.0, self.b / 255.0)

    def scale(self, factor: float) -> "Color":
        return Color.from_floats(*(channel * factor for channel in self.to_floats()))

    def add(self, other: "Color" | tuple[int, int, int]) -> "Color":
        color = Color.from_value(other)
        return Color(self.r + color.r, self.g + color.g, self.b + color.b)

    def modulate(self, other: "Color" | tuple[int, int, int]) -> "Color":
        color = Color.from_value(other)
        r1, g1, b1 = self.to_floats()
        r2, g2, b2 = color.to_floats()
        return Color.from_floats(r1 * r2, g1 * g2, b1 * b2)
