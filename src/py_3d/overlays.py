"""2D overlays that can be composited over rendered scenes."""

from __future__ import annotations

from dataclasses import dataclass

from .color import Color
from .math3d import Vec3, as_vec3


@dataclass(frozen=True)
class TextBulletin:
    """A simple text overlay rendered into the final image buffer."""

    text: str
    position: tuple[int, int] = (8, 8)
    color: Color | tuple[int, int, int] = Color(245, 248, 255)
    background: Color | tuple[int, int, int] | None = Color(0, 0, 0)
    padding: int = 4
    scale: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", (int(self.position[0]), int(self.position[1])))
        object.__setattr__(self, "color", Color.from_value(self.color))
        if self.background is not None:
            object.__setattr__(self, "background", Color.from_value(self.background))
        if self.padding < 0:
            raise ValueError("bulletin padding must be non-negative")
        if self.scale <= 0:
            raise ValueError("bulletin scale must be positive")


@dataclass(frozen=True)
class FloatingTextBulletin:
    """A text overlay anchored to a 3D world position."""

    text: str
    position: Vec3 | tuple[float, float, float]
    screen_offset: tuple[int, int] = (0, -12)
    anchor: tuple[float, float] = (0.5, 1.0)
    color: Color | tuple[int, int, int] = Color(245, 248, 255)
    background: Color | tuple[int, int, int] | None = Color(4, 6, 10)
    padding: int = 4
    scale: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_vec3(self.position))
        object.__setattr__(self, "screen_offset", (int(self.screen_offset[0]), int(self.screen_offset[1])))
        object.__setattr__(self, "anchor", (float(self.anchor[0]), float(self.anchor[1])))
        object.__setattr__(self, "color", Color.from_value(self.color))
        if self.background is not None:
            object.__setattr__(self, "background", Color.from_value(self.background))
        if self.padding < 0:
            raise ValueError("bulletin padding must be non-negative")
        if self.scale <= 0:
            raise ValueError("bulletin scale must be positive")
