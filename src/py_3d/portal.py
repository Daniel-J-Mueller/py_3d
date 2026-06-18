"""Portal primitives and render helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .buffer import PixelBuffer
from .camera import Camera
from .materials import Material
from .math3d import Vec3, as_vec3
from .primitives import Triangle
from .scene import Scene


@dataclass(frozen=True)
class PortalSurface:
    """A rectangular renderable portal surface.

    ``normal`` points out from the visible side of the portal. When a
    ``PortalPair`` is attached to a scene, the renderer fills ``active_texture``
    with the linked portal view before drawing the final frame.
    """

    name: str
    center: Vec3 | tuple[float, float, float]
    normal: Vec3 | tuple[float, float, float]
    width: float = 1.2
    height: float = 1.8
    up: Vec3 | tuple[float, float, float] = Vec3(0.0, 1.0, 0.0)
    material: Material = Material(color=(18, 24, 36), emission=(2, 4, 8), roughness=0.08, specular=0.15)
    frame_material: Material = Material(color=(60, 150, 255), emission=(8, 24, 48), roughness=0.2, specular=0.2)
    frame_width: float = 0.06
    active_texture: PixelBuffer | None = None

    def __post_init__(self) -> None:
        center = as_vec3(self.center)
        normal = as_vec3(self.normal).normalized(Vec3(0.0, 0.0, -1.0))
        up = as_vec3(self.up)
        projected_up = (up - normal * up.dot(normal)).normalized(Vec3(0.0, 1.0, 0.0))
        if projected_up.length_squared() <= 1e-12:
            projected_up = normal.cross(Vec3(1.0, 0.0, 0.0)).normalized(Vec3(0.0, 1.0, 0.0))
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "up", projected_up)
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("portal width and height must be positive")
        if self.frame_width < 0.0:
            raise ValueError("portal frame width must be non-negative")

    @property
    def right(self) -> Vec3:
        return self.normal.cross(self.up).normalized(Vec3(1.0, 0.0, 0.0))

    @property
    def true_up(self) -> Vec3:
        return self.right.cross(self.normal).normalized(Vec3(0.0, 1.0, 0.0))

    def contains_point(self, point: Vec3 | tuple[float, float, float], *, tolerance: float = 1e-5) -> bool:
        offset = as_vec3(point) - self.center
        if abs(offset.dot(self.normal)) > tolerance:
            return False
        return abs(offset.dot(self.right)) <= self.width * 0.5 and abs(offset.dot(self.true_up)) <= self.height * 0.5

    def with_texture(self, texture: PixelBuffer | None) -> "PortalSurface":
        return replace(self, active_texture=texture)

    def to_triangles(self, **kwargs) -> tuple[Triangle, ...]:
        del kwargs
        right = self.right
        up = self.true_up
        surface_offset = self.normal * 0.002
        half_width = self.width * 0.5
        half_height = self.height * 0.5
        a = self.center - right * half_width - up * half_height + surface_offset
        b = self.center + right * half_width - up * half_height + surface_offset
        c = self.center + right * half_width + up * half_height + surface_offset
        d = self.center - right * half_width + up * half_height + surface_offset
        material = self._surface_material()
        triangles = [
            Triangle(a, b, c, material, (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), self.normal, self.normal, self.normal),
            Triangle(a, c, d, material, (0.0, 1.0), (1.0, 0.0), (0.0, 0.0), self.normal, self.normal, self.normal),
        ]
        if self.frame_width > 0.0:
            triangles.extend(self._frame_triangles(right, up, surface_offset))
        return tuple(triangles)

    def _surface_material(self) -> Material:
        if self.active_texture is None:
            return self.material
        return Material(
            color=(255, 255, 255),
            texture=self.active_texture,
            diffuse=1.0,
            roughness=0.0,
            specular=0.0,
            shininess=8.0,
        )

    def _frame_triangles(self, right: Vec3, up: Vec3, surface_offset: Vec3) -> list[Triangle]:
        half_width = self.width * 0.5
        half_height = self.height * 0.5
        outer_width = half_width + self.frame_width
        outer_height = half_height + self.frame_width
        center = self.center + surface_offset + self.normal * 0.002
        inner = (
            center - right * half_width - up * half_height,
            center + right * half_width - up * half_height,
            center + right * half_width + up * half_height,
            center - right * half_width + up * half_height,
        )
        outer = (
            center - right * outer_width - up * outer_height,
            center + right * outer_width - up * outer_height,
            center + right * outer_width + up * outer_height,
            center - right * outer_width + up * outer_height,
        )
        strips = (
            (outer[0], outer[1], inner[1], inner[0]),
            (inner[3], inner[2], outer[2], outer[3]),
            (outer[0], inner[0], inner[3], outer[3]),
            (inner[1], outer[1], outer[2], inner[2]),
        )
        triangles: list[Triangle] = []
        for p0, p1, p2, p3 in strips:
            triangles.extend(_quad(p0, p1, p2, p3, self.frame_material, self.normal))
        return triangles


@dataclass(frozen=True)
class PortalPair:
    """Two linked portals that can render each other's view."""

    first: PortalSurface
    second: PortalSurface
    texture_width: int = 192
    texture_height: int = 128

    def __post_init__(self) -> None:
        if self.texture_width <= 0 or self.texture_height <= 0:
            raise ValueError("portal texture dimensions must be positive")

    def surfaces(self) -> tuple[PortalSurface, PortalSurface]:
        return (self.first, self.second)

    def other(self, portal: PortalSurface | str) -> PortalSurface:
        name = portal if isinstance(portal, str) else portal.name
        if name == self.first.name:
            return self.second
        if name == self.second.name:
            return self.first
        raise ValueError(f"portal {name!r} is not part of this pair")


def portal_camera_for(camera: Camera, entry: PortalSurface, exit: PortalSurface) -> Camera:
    """Return the camera produced by looking through ``entry`` out of ``exit``."""

    position = _map_point_between_portals(camera.position, entry, exit)
    direction = (camera.target - camera.position).normalized(entry.normal)
    local_x = direction.dot(entry.right)
    local_y = direction.dot(entry.true_up)
    local_z = direction.dot(entry.normal)
    mapped_direction = (exit.right * local_x + exit.true_up * local_y - exit.normal * local_z).normalized(exit.normal)
    target = position + mapped_direction
    return Camera(
        position=position + exit.normal * 0.035,
        target=target + exit.normal * 0.035,
        up=exit.true_up,
        fov_degrees=camera.fov_degrees,
        near=camera.near,
        far=camera.far,
    )


def scene_with_portal_textures(
    scene: Scene,
    camera: Camera,
    settings: Any,
    renderer: Any,
) -> Scene:
    """Return a scene copy where portal surfaces carry freshly rendered textures."""

    pairs = tuple(getattr(scene, "portals", ()))
    if not pairs:
        return scene

    replacements: dict[int, PortalSurface] = {}
    for pair in pairs:
        replacements[id(pair.first)] = pair.first.with_texture(
            _render_linked_view(scene, camera, settings, renderer, pair.first, pair.second, pair)
        )
        replacements[id(pair.second)] = pair.second.with_texture(
            _render_linked_view(scene, camera, settings, renderer, pair.second, pair.first, pair)
        )
    return _copy_scene_with_replacements(scene, replacements)


def _render_linked_view(
    scene: Scene,
    camera: Camera,
    settings: Any,
    renderer: Any,
    entry: PortalSurface,
    exit: PortalSurface,
    pair: PortalPair,
) -> PixelBuffer:
    portal_scene = _scene_without_portal_surfaces(scene)
    portal_camera = portal_camera_for(camera, entry, exit)
    portal_settings = replace(
        settings,
        width=pair.texture_width,
        height=pair.texture_height,
        edge_highlight=False,
        ray_traced_shadows=False,
        reflection_bounces=0,
    )
    return renderer.render(portal_scene, portal_camera, portal_settings)


def _scene_without_portal_surfaces(scene: Scene) -> Scene:
    portal_ids = {id(surface) for pair in getattr(scene, "portals", ()) for surface in pair.surfaces()}
    return Scene(
        objects=[obj for obj in scene.objects if id(obj) not in portal_ids],
        lights=list(scene.lights),
        bulletins=list(scene.bulletins),
        background=scene.background,
    )


def _copy_scene_with_replacements(scene: Scene, replacements: dict[int, PortalSurface]) -> Scene:
    objects: list[Any] = []
    used: set[int] = set()
    for obj in scene.objects:
        replacement = replacements.get(id(obj))
        if replacement is None:
            objects.append(obj)
            continue
        objects.append(replacement)
        used.add(id(obj))
    for source_id, replacement in replacements.items():
        if source_id not in used:
            objects.append(replacement)
    return Scene(
        objects=objects,
        lights=list(scene.lights),
        bulletins=list(scene.bulletins),
        background=scene.background,
        portals=list(scene.portals),
    )


def _map_point_between_portals(point: Vec3, entry: PortalSurface, exit: PortalSurface) -> Vec3:
    offset = point - entry.center
    local_x = offset.dot(entry.right)
    local_y = offset.dot(entry.true_up)
    local_z = offset.dot(entry.normal)
    return exit.center + exit.right * local_x + exit.true_up * local_y + exit.normal * local_z


def _quad(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, material: Material, normal: Vec3) -> list[Triangle]:
    return [
        Triangle(p0, p1, p2, material, normal_a=normal, normal_b=normal, normal_c=normal),
        Triangle(p0, p2, p3, material, normal_a=normal, normal_b=normal, normal_c=normal),
    ]
