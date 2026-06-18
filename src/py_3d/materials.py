"""Material definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .buffer import PixelBuffer
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
    texture: PixelBuffer | None = None
    roughness: float = 0.0
    fuzziness: float = 0.0
    specular: float = 0.0
    shininess: float = 32.0
    reflectivity: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", Color.from_value(self.color))
        object.__setattr__(self, "absorption", _absorption_tuple(self.absorption))
        object.__setattr__(self, "diffuse", clamp(float(self.diffuse), 0.0, 1.0))
        object.__setattr__(self, "emission", Color.from_value(self.emission))
        object.__setattr__(self, "roughness", clamp(float(self.roughness), 0.0, 1.0))
        object.__setattr__(self, "fuzziness", clamp(float(self.fuzziness), 0.0, 1.0))
        object.__setattr__(self, "specular", clamp(float(self.specular), 0.0, 1.0))
        object.__setattr__(self, "shininess", max(1.0, float(self.shininess)))
        object.__setattr__(self, "reflectivity", clamp(float(self.reflectivity), 0.0, 1.0))

    def color_at(self, uv: tuple[float, float] | None = None) -> Color:
        if uv is not None and self.texture is not None:
            return self.texture.sample_nearest(uv[0], uv[1], wrap=False)
        return self.color

    def shade(
        self,
        light: tuple[float, float, float],
        ambient: float = 0.0,
        base_color: Color | tuple[int, int, int] | None = None,
        specular_light: tuple[float, float, float] | None = None,
    ) -> Color:
        base = Color.from_value(base_color).to_floats() if base_color is not None else self.color.to_floats()
        emitted = self.emission.to_floats()
        highlight = specular_light or (0.0, 0.0, 0.0)
        specular_strength = self.specular * (1.0 - self.roughness * 0.85) + self.reflectivity * 0.5
        result = []
        for index, base_channel in enumerate(base):
            absorbed = 1.0 - self.absorption[index]
            channel = emitted[index] + base_channel * absorbed * (ambient + light[index] * self.diffuse)
            channel += highlight[index] * specular_strength
            result.append(channel)
        return Color.from_floats(result[0], result[1], result[2])
