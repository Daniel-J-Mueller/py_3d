"""Live 2D HUD overlay primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

from .buffer import PixelBuffer
from .color import Color


@dataclass(frozen=True)
class HUDRect:
    position: tuple[int, int]
    size: tuple[int, int]
    color: Color | tuple[int, int, int]
    alpha: float = 1.0


@dataclass(frozen=True)
class HUDText:
    text: str
    position: tuple[int, int]
    color: Color | tuple[int, int, int] = Color(255, 255, 255)
    background: Color | tuple[int, int, int] | None = None
    alpha: float = 1.0
    padding: int = 0
    scale: int = 1


@dataclass(frozen=True)
class HUDImage:
    image: PixelBuffer
    position: tuple[int, int]
    alpha: float = 1.0
    scale: int = 1


@dataclass(frozen=True)
class HUDAnimation:
    frames: tuple[PixelBuffer, ...]
    position: tuple[int, int]
    frame_seconds: float = 0.08
    alpha: float = 1.0
    scale: int = 1
    loop: bool = True

    def frame_at(self, seconds: float) -> PixelBuffer | None:
        if not self.frames:
            return None
        frame_seconds = max(1e-4, self.frame_seconds)
        index = int(max(0.0, seconds) / frame_seconds)
        if self.loop:
            index %= len(self.frames)
        else:
            index = min(index, len(self.frames) - 1)
        return self.frames[index]


HUDElement = HUDRect | HUDText | HUDImage | HUDAnimation


@dataclass
class LiveHUD:
    """A simple ordered collection of 2D HUD elements for live renderers."""

    elements: list[HUDElement] = field(default_factory=list)
    visible: bool = True

    def clear(self) -> None:
        self.elements.clear()

    def set(self, *elements: HUDElement) -> None:
        self.elements[:] = list(elements)
