"""Inventory and cube placement helpers for small games."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Any

from .camera import Camera
from .materials import Material
from .math3d import Vec3, as_vec3
from .primitives import Box
from .scene import Scene


@dataclass
class InventorySlot:
    name: str
    quantity: int = 0
    max_quantity: int = 99
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_quantity <= 0:
            raise ValueError("inventory slot max_quantity must be positive")
        self.quantity = max(0, min(int(self.quantity), self.max_quantity))

    @property
    def available(self) -> bool:
        return self.quantity > 0

    def add(self, amount: int) -> int:
        amount = max(0, int(amount))
        accepted = min(amount, self.max_quantity - self.quantity)
        self.quantity += accepted
        return amount - accepted

    def consume(self, amount: int = 1) -> bool:
        amount = max(0, int(amount))
        if amount == 0:
            return True
        if self.quantity < amount:
            return False
        self.quantity -= amount
        return True


@dataclass
class Inventory:
    slots: list[InventorySlot] = field(default_factory=list)
    selected_index: int = 0

    def add(self, name: str, quantity: int = 1, *, max_quantity: int = 99, metadata: dict[str, Any] | None = None) -> None:
        remaining = max(0, int(quantity))
        for slot in self.slots:
            if slot.name == name and slot.metadata == (metadata or {}):
                remaining = slot.add(remaining)
                if remaining == 0:
                    return
        while remaining > 0:
            slot = InventorySlot(name=name, quantity=0, max_quantity=max_quantity, metadata=dict(metadata or {}))
            remaining = slot.add(remaining)
            self.slots.append(slot)

    def selected(self) -> InventorySlot | None:
        if not self.slots:
            return None
        self.selected_index %= len(self.slots)
        return self.slots[self.selected_index]

    def select(self, name: str) -> InventorySlot | None:
        for index, slot in enumerate(self.slots):
            if slot.name == name:
                self.selected_index = index
                return slot
        return None

    def cycle(self, direction: int = 1) -> InventorySlot | None:
        if not self.slots:
            return None
        self.selected_index = (self.selected_index + (1 if direction >= 0 else -1)) % len(self.slots)
        return self.selected()

    def count(self, name: str) -> int:
        return sum(slot.quantity for slot in self.slots if slot.name == name)

    def consume(self, name: str, quantity: int = 1) -> bool:
        remaining = max(0, int(quantity))
        for slot in self.slots:
            if slot.name != name or not slot.available:
                continue
            taken = min(remaining, slot.quantity)
            slot.quantity -= taken
            remaining -= taken
            if remaining == 0:
                return True
        return remaining == 0

    def summary(self) -> tuple[tuple[str, int], ...]:
        return tuple((slot.name, slot.quantity) for slot in self.slots)


@dataclass
class CubePlacer:
    inventory: Inventory
    item_name: str = "cube"
    cube_size: float = 0.55
    placement_distance: float = 2.1
    floor_y: float = 0.0
    snap: float | None = 0.25
    material: Material = Material(color=(255, 180, 70), roughness=0.34, fuzziness=0.08, specular=0.18)

    def __post_init__(self) -> None:
        if self.cube_size <= 0.0:
            raise ValueError("cube_size must be positive")
        if self.placement_distance <= 0.0:
            raise ValueError("placement_distance must be positive")
        if self.snap is not None and self.snap <= 0.0:
            raise ValueError("snap must be positive when provided")

    def preview_center(self, camera: Camera, *, distance: float | None = None) -> Vec3:
        _right, _up, forward = camera.basis()
        raw = camera.position + forward * (distance or self.placement_distance)
        center = Vec3(raw.x, self.floor_y + self.cube_size * 0.5, raw.z)
        if self.snap is None:
            return center
        return Vec3(_snap(center.x, self.snap), center.y, _snap(center.z, self.snap))

    def place(self, camera: Camera, scene: Scene | None = None) -> Box | None:
        if not self.inventory.consume(self.item_name, 1):
            return None
        cube = Box(self.preview_center(camera), (self.cube_size, self.cube_size, self.cube_size), self.material)
        if scene is not None:
            scene.add(cube)
        return cube


def place_cube_from_inventory(
    inventory: Inventory,
    camera: Camera,
    scene: Scene | None = None,
    *,
    item_name: str = "cube",
    cube_size: float = 0.55,
    placement_distance: float = 2.1,
    floor_y: float = 0.0,
    snap: float | None = 0.25,
    material: Material = Material(color=(255, 180, 70), roughness=0.34, fuzziness=0.08, specular=0.18),
) -> Box | None:
    placer = CubePlacer(
        inventory,
        item_name=item_name,
        cube_size=cube_size,
        placement_distance=placement_distance,
        floor_y=floor_y,
        snap=snap,
        material=material,
    )
    return placer.place(camera, scene)


def _snap(value: float, step: float) -> float:
    return floor(value / step + 0.5) * step
