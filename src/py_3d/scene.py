"""Scene container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .color import Color


@dataclass
class Scene:
    """A small container for renderable objects and lights."""

    objects: list[Any] = field(default_factory=list)
    lights: list[Any] = field(default_factory=list)
    bulletins: list[Any] = field(default_factory=list)
    background: Color | tuple[int, int, int] = Color(0, 0, 0)
    portals: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.background = Color.from_value(self.background)

    def add(self, *objects: Any) -> "Scene":
        self.objects.extend(objects)
        return self

    def add_light(self, *lights: Any) -> "Scene":
        self.lights.extend(lights)
        return self

    def add_bulletin(self, *bulletins: Any) -> "Scene":
        self.bulletins.extend(bulletins)
        return self

    def add_portal_pair(self, *pairs: Any, add_surfaces: bool = True) -> "Scene":
        self.portals.extend(pairs)
        if add_surfaces:
            for pair in pairs:
                surfaces = getattr(pair, "surfaces", None)
                if callable(surfaces):
                    self.objects.extend(surfaces())
        return self

    def clear(self) -> None:
        self.objects.clear()
        self.lights.clear()
        self.bulletins.clear()
        self.portals.clear()
