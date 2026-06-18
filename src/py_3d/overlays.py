"""2D overlays that can be composited over rendered scenes."""

from __future__ import annotations

from dataclasses import dataclass

from .color import Color


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
