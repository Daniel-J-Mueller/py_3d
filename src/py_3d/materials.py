"""Material definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .color import Color
from .math3d import clamp


def _absorption_tuple(value: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("absorption must have exactly three channels")
    return tuple(clamp(float(channel), 0.0, 1.0) for channel in value)


@dataclass(frozen=True)
class Material:
    """Basic material response for the reference renderer."""

    color: Color | tuple[int, int, int] = Color(255, 255, 255)
    absorption: tuple[float, float, float] = (0.0, 0.0, 0.0)
    diffuse: float = 1.0
    emission: Color | tuple[int, int, int] = Color(0, 0, 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", Color.from_value(self.color))
        object.__setattr__(self, "absorption", _absorption_tuple(self.absorption))
        object.__setattr__(self, "diffuse", clamp(float(self.diffuse), 0.0, 1.0))
        object.__setattr__(self, "emission", Color.from_value(self.emission))

    def shade(
        self,
        light: tuple[float, float, float],
        ambient: float = 0.0,
    ) -> Color:
        base = self.color.to_floats()
        emitted = self.emission.to_floats()
        result = []
        for index, base_channel in enumerate(base):
            absorbed = 1.0 - self.absorption[index]
            channel = emitted[index] + base_channel * absorbed * (ambient + light[index] * self.diffuse)
            result.append(channel)
        return Color.from_floats(result[0], result[1], result[2])
