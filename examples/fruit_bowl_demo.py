"""Kinematic fruit bowl demo with live and offline rendering paths."""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass, replace
from math import cos, pi, radians, sin, tau
import os
from pathlib import Path
import shutil
import subprocess
import sys

from py_3d import (
    Camera,
    Color,
    CompoundSphereCollider,
    CPURenderer,
    FloatingTextBulletin,
    Box,
    HangingConeLampPrimitive,
    HUDRect,
    HUDText,
    KinematicBowl,
    Lamp,
    LampPrimitive,
    Line3,
    Material,
    Mesh,
    PhysicsWorld,
    PixelBuffer,
    RenderEngine,
    RenderSettings,
    Scene,
    SkyPrefab,
    Sphere,
    SphereBody,
    SphereCollider,
    StaticPlane,
    Sun,
    SurfacePerturbation,
    TextBulletin,
    Triangle,
    Vec3,
    draw,
)


OUTPUT_DIR = Path("renderings-tests")
FRUIT_BOWL_OUTPUT_DIR = Path("USER") / "environments" / "fruit_bowl" / "renderings"
USER_SETTINGS_PATH = Path("USER") / "settings.json"


DEFAULT_RENDER_QUALITY_PRESETS = {
    "fast": {
        "width": 640,
        "height": 360,
        "window_width": 960,
        "window_height": 540,
        "sphere_segments": 10,
        "sphere_rings": 5,
        "smooth_shading": False,
        "texture_size": 128,
        "gamma": 1.05,
        "light_wrap": 0.08,
        "bounce_light": 0.04,
        "tone_mapping": False,
        "reflection_bounces": 0,
        "shadow_samples": 1,
        "shadow_softness": 0.0,
        "max_render_distance": 7.0,
    },
    "balanced": {
        "width": 960,
        "height": 540,
        "window_width": 1280,
        "window_height": 720,
        "sphere_segments": 18,
        "sphere_rings": 9,
        "smooth_shading": True,
        "texture_size": 256,
        "gamma": 1.12,
        "light_wrap": 0.16,
        "bounce_light": 0.08,
        "tone_mapping": True,
        "reflection_bounces": 1,
        "shadow_samples": 2,
        "shadow_softness": 0.08,
        "max_render_distance": 8.5,
    },
    "high": {
        "width": 1280,
        "height": 720,
        "window_width": 1280,
        "window_height": 720,
        "sphere_segments": 24,
        "sphere_rings": 12,
        "smooth_shading": True,
        "texture_size": 384,
        "gamma": 1.15,
        "light_wrap": 0.24,
        "bounce_light": 0.14,
        "tone_mapping": True,
        "reflection_bounces": 2,
        "shadow_samples": 4,
        "shadow_softness": 0.12,
        "max_render_distance": 10.0,
    },
    "ultra": {
        "width": 1920,
        "height": 1080,
        "window_width": 1920,
        "window_height": 1080,
        "sphere_segments": 32,
        "sphere_rings": 16,
        "smooth_shading": True,
        "texture_size": 512,
        "gamma": 1.18,
        "light_wrap": 0.28,
        "bounce_light": 0.18,
        "tone_mapping": True,
        "reflection_bounces": 3,
        "shadow_samples": 8,
        "shadow_softness": 0.18,
        "max_render_distance": 12.0,
    },
    "poly": {
        "width": 1280,
        "height": 720,
        "window_width": 1280,
        "window_height": 720,
        "sphere_segments": 7,
        "sphere_rings": 4,
        "smooth_shading": False,
        "texture_size": 192,
        "gamma": 1.12,
        "light_wrap": 0.18,
        "bounce_light": 0.12,
        "tone_mapping": True,
        "reflection_bounces": 1,
        "shadow_samples": 1,
        "shadow_softness": 0.0,
        "max_render_distance": 9.0,
    },
}

QUALITY_FLAG_MAP = {
    "width": ("--width",),
    "height": ("--height",),
    "window_width": ("--window-width",),
    "window_height": ("--window-height",),
    "sphere_segments": ("--sphere-segments",),
    "sphere_rings": ("--sphere-rings",),
    "smooth_shading": ("--smooth-shading", "--no-smooth-shading"),
    "texture_size": ("--texture-size",),
    "gamma": ("--gamma",),
    "light_wrap": ("--light-wrap",),
    "bounce_light": ("--bounce-light",),
    "tone_mapping": ("--tone-mapping", "--no-tone-mapping"),
    "reflection_bounces": ("--reflection-bounces",),
    "shadow_samples": ("--shadow-samples",),
    "shadow_softness": ("--shadow-softness",),
    "max_render_distance": ("--max-render-distance",),
}


def load_user_settings() -> dict:
    if not USER_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(USER_SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def render_quality_presets() -> dict:
    settings = load_user_settings()
    presets = {name: values.copy() for name, values in DEFAULT_RENDER_QUALITY_PRESETS.items()}
    for name, values in settings.get("render_quality_presets", {}).items():
        if isinstance(values, dict):
            merged = presets.get(name, {}).copy()
            merged.update(values)
            presets[name] = merged
    return presets


def apply_render_quality(args: argparse.Namespace, argv: list[str] | None = None) -> argparse.Namespace:
    settings = load_user_settings()
    quality = getattr(args, "quality", None) or settings.get("render_quality", "balanced")
    setattr(args, "quality", quality)
    preset = render_quality_presets().get(quality, {})
    tokens = argv if argv is not None else sys.argv[1:]
    for key, value in preset.items():
        if not hasattr(args, key):
            continue
        if _quality_flag_present(QUALITY_FLAG_MAP.get(key, ()), tokens):
            continue
        setattr(args, key, value)
    return args


def _quality_flag_present(flags: tuple[str, ...], tokens: list[str]) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in tokens for flag in flags)


@dataclass
class Fruit:
    name: str
    body: SphereBody
    visual: str = "sphere"
    banana_yaw: float = 0.0
    marker_color: tuple[int, int, int] = (40, 30, 25)

    def to_primitives(self) -> tuple[Sphere | Mesh | Line3, ...]:
        if self.visual == "banana":
            return (
                banana_mesh(
                    self.body.position,
                    self.body.radius,
                    self.body.material,
                    yaw_degrees=self.banana_yaw,
                    rotation=self.body.rotation,
                ),
            )
        marker = rotate_euler(Vec3(0.0, self.body.radius * 1.03, 0.0), self.body.rotation)
        marker_tail = rotate_euler(Vec3(self.body.radius * 0.42, self.body.radius * 0.72, 0.0), self.body.rotation)
        return (
            self.body.to_primitive(),
            Line3(
                self.body.position + marker_tail,
                self.body.position + marker,
                Material(color=self.marker_color, emission=(12, 10, 8)),
            ),
        )

    def sync_collision_boundary(self) -> None:
        if self.visual == "banana":
            self.body.collision_boundary = banana_collider(
                self.body.radius,
                yaw_degrees=self.banana_yaw,
                rotation=self.body.rotation,
            )


def banana_mesh(
    center: Vec3,
    radius: float,
    material: Material,
    *,
    yaw_degrees: float = 0.0,
    rotation: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0),
    sections: int = 10,
    sides: int = 8,
) -> Mesh:
    """Return a small curved tube mesh suitable for a banana-like fruit."""

    if sections < 2:
        raise ValueError("banana sections must be at least 2")
    if sides < 3:
        raise ValueError("banana sides must be at least 3")

    rotation = as_rotation(rotation, yaw_degrees)
    centerline = banana_centerline_offsets(radius, sections=sections, rotation=rotation)
    rings: list[list[Vec3]] = []

    for section in range(sections + 1):
        amount = section / sections
        section_center = centerline[section]
        previous_center = centerline[max(0, section - 1)]
        next_center = centerline[min(sections, section + 1)]
        tangent = (next_center - previous_center).normalized(Vec3(1.0, 0.0, 0.0))
        binormal = rotate_euler(Vec3(0.0, 0.0, 1.0), rotation)
        normal = binormal.cross(tangent).normalized(Vec3(0.0, 1.0, 0.0))
        taper = sin(pi * amount) ** 0.5
        tube_radius = radius * 0.24 * (0.35 + 0.65 * taper)
        ring = []
        for side in range(sides):
            theta = tau * side / sides
            local = section_center + normal * (cos(theta) * tube_radius) + binormal * (sin(theta) * tube_radius)
            ring.append(center + local)
        rings.append(ring)

    triangles: list[Triangle] = []
    for section in range(sections):
        for side in range(sides):
            next_side = (side + 1) % sides
            top_left = rings[section][side]
            top_right = rings[section][next_side]
            bottom_left = rings[section + 1][side]
            bottom_right = rings[section + 1][next_side]
            u = side / sides
            next_u = 1.0 if next_side == 0 else next_side / sides
            v = section / sections
            next_v = (section + 1) / sections
            triangles.append(Triangle(top_left, bottom_left, top_right, material, (u, v), (u, next_v), (next_u, v)))
            triangles.append(Triangle(top_right, bottom_left, bottom_right, material, (next_u, v), (u, next_v), (next_u, next_v)))

    cap_material = Material(color=(112, 82, 34), roughness=0.4)
    for ring_index, reverse in ((0, True), (-1, False)):
        cap_center = _ring_center(rings[ring_index])
        for side in range(sides):
            next_side = (side + 1) % sides
            a = rings[ring_index][next_side if reverse else side]
            b = rings[ring_index][side if reverse else next_side]
            triangles.append(Triangle(a, b, cap_center, cap_material))
    return Mesh(triangles)


def banana_collider(
    radius: float,
    *,
    yaw_degrees: float = 0.0,
    rotation: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0),
    sections: int = 10,
) -> CompoundSphereCollider:
    offsets = banana_centerline_offsets(radius, sections=sections, rotation=as_rotation(rotation, yaw_degrees))
    return CompoundSphereCollider.from_offsets(offsets, radius=radius * 0.2)


def banana_centerline_offsets(
    radius: float,
    *,
    sections: int = 10,
    rotation: Vec3 = Vec3(0.0, 0.0, 0.0),
) -> list[Vec3]:
    curve_radius = radius * 1.55
    height_scale = 0.42
    start_angle = -1.12
    end_angle = 1.12
    base_y = curve_radius * height_scale * cos(start_angle)
    offsets = []
    for section in range(sections + 1):
        amount = section / sections
        angle = start_angle + (end_angle - start_angle) * amount
        local = Vec3(
            curve_radius * sin(angle),
            curve_radius * height_scale * cos(angle) - base_y - radius * 0.12,
            0.0,
        )
        offsets.append(rotate_euler(local, rotation))
    return offsets


def as_rotation(rotation: Vec3 | tuple[float, float, float], yaw_degrees: float) -> Vec3:
    value = rotation if isinstance(rotation, Vec3) else Vec3(*rotation)
    return Vec3(value.x, value.y + radians(yaw_degrees), value.z)


def rotate_euler(value: Vec3, rotation: Vec3) -> Vec3:
    cx, sx = cos(rotation.x), sin(rotation.x)
    cy, sy = cos(rotation.y), sin(rotation.y)
    cz, sz = cos(rotation.z), sin(rotation.z)
    x_rotated = Vec3(value.x, value.y * cx - value.z * sx, value.y * sx + value.z * cx)
    y_rotated = Vec3(x_rotated.x * cy + x_rotated.z * sy, x_rotated.y, -x_rotated.x * sy + x_rotated.z * cy)
    return Vec3(y_rotated.x * cz - y_rotated.y * sz, y_rotated.x * sz + y_rotated.y * cz, y_rotated.z)


def _ring_center(ring: list[Vec3]) -> Vec3:
    total = Vec3(0.0, 0.0, 0.0)
    for point in ring:
        total = total + point
    return total / len(ring)


_WOOD_TEXTURE: PixelBuffer | None = None
_APPLE_TEXTURE: PixelBuffer | None = None
_ORANGE_TEXTURE: PixelBuffer | None = None
_LEMON_TEXTURE: PixelBuffer | None = None
_WATERMELON_TEXTURE: PixelBuffer | None = None
_BANANA_TEXTURE: PixelBuffer | None = None
_FLOOR_TEXTURE: PixelBuffer | None = None
_SIGN_TEXTURES: dict[tuple[str, str, int, int], PixelBuffer] = {}


def _bowl_style(name: str) -> tuple[Material, SurfacePerturbation | None]:
    style = name.lower().replace("_", "-")
    if style == "mirror":
        return (
            Material(
                color=(218, 224, 230),
                absorption=(0.02, 0.02, 0.02),
                diffuse=0.28,
                roughness=0.015,
                fuzziness=0.0,
                specular=1.0,
                shininess=120.0,
                reflectivity=1.0,
            ),
            None,
        )
    return (
        Material(
            color=(132, 82, 42),
            texture=wood_grain_texture(),
            absorption=(0.08, 0.12, 0.18),
            roughness=0.72,
            fuzziness=0.24,
            specular=0.08,
            shininess=14.0,
            reflectivity=0.0,
        ),
        SurfacePerturbation(magnitude=0.032, scale=8.0, seed=91, octaves=4, gain=0.5),
    )


def wood_grain_texture(width: int = 512, height: int = 256) -> PixelBuffer:
    global _WOOD_TEXTURE
    if _WOOD_TEXTURE is not None and _WOOD_TEXTURE.width == width and _WOOD_TEXTURE.height == height:
        return _WOOD_TEXTURE
    pixels: list[Color] = []
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            broad = 0.5 + 0.5 * sin((u * 8.0 + 0.36 * sin(v * tau * 2.1)) * tau)
            fine = 0.5 + 0.5 * sin((u * 38.0 + v * 2.7 + 0.18 * sin(v * tau * 5.0)) * tau)
            knot = 0.5 + 0.5 * sin(((u - 0.18) ** 2 * 38.0 + (v - 0.52) ** 2 * 95.0) * tau)
            grain = 0.52 * broad + 0.34 * fine + 0.14 * knot
            r = 100 + grain * 72
            g = 58 + grain * 42
            b = 27 + grain * 24
            pixels.append(Color(r, g, b))
    _WOOD_TEXTURE = PixelBuffer(width, height, pixels)
    return _WOOD_TEXTURE


def _hash_unit(x: int, y: int, salt: int) -> float:
    value = (x * 73856093) ^ (y * 19349663) ^ salt
    value = (value ^ (value >> 13)) * 1274126177
    return (value & 0xFFFF) / 0xFFFF


def apple_texture(width: int = 384, height: int = 192) -> PixelBuffer:
    global _APPLE_TEXTURE
    if _APPLE_TEXTURE is not None and _APPLE_TEXTURE.width == width and _APPLE_TEXTURE.height == height:
        return _APPLE_TEXTURE
    pixels: list[Color] = []
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            blush = 0.5 + 0.5 * sin((u * 1.3 + 0.08 * sin(v * tau * 1.7)) * tau)
            vertical = 1.0 - abs(v - 0.45) * 0.9
            shoulder = max(0.0, 1.0 - ((u - 0.18) / 0.22) ** 2 - ((v - 0.28) / 0.5) ** 2)
            lenticel = _hash_unit(x // 2, y // 2, 0xA991)
            speckle = 1.0 if lenticel > 0.965 else 0.0
            r = 150 + blush * 82 + vertical * 18 + speckle * 34
            g = 34 + (1.0 - blush) * 32 + shoulder * 46 + speckle * 28
            b = 30 + (1.0 - blush) * 16 + speckle * 10
            pixels.append(Color(r, g, b))
    _APPLE_TEXTURE = PixelBuffer(width, height, pixels)
    return _APPLE_TEXTURE


def orange_texture(width: int = 384, height: int = 192) -> PixelBuffer:
    global _ORANGE_TEXTURE
    if _ORANGE_TEXTURE is not None and _ORANGE_TEXTURE.width == width and _ORANGE_TEXTURE.height == height:
        return _ORANGE_TEXTURE
    pixels: list[Color] = []
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            pore = _hash_unit(x, y, 0x0A6E)
            dimple = _hash_unit(x // 3, y // 3, 0xC17A)
            rind = 0.5 + 0.5 * sin((u * 5.5 + v * 0.6) * tau)
            r = 210 + rind * 26 - dimple * 14
            g = 92 + rind * 38 - dimple * 18
            b = 30 + rind * 10 - dimple * 8
            if pore > 0.93:
                r += 28
                g += 24
                b += 8
            pixels.append(Color(r, g, b))
    _ORANGE_TEXTURE = PixelBuffer(width, height, pixels)
    return _ORANGE_TEXTURE


def lemon_texture(width: int = 384, height: int = 192) -> PixelBuffer:
    global _LEMON_TEXTURE
    if _LEMON_TEXTURE is not None and _LEMON_TEXTURE.width == width and _LEMON_TEXTURE.height == height:
        return _LEMON_TEXTURE
    pixels: list[Color] = []
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            pore = _hash_unit(x, y, 0x1E40)
            mottled = 0.5 + 0.5 * sin((u * 9.0 + 0.22 * sin(v * tau * 3.0)) * tau)
            blush = max(0.0, 1.0 - ((u - 0.72) / 0.18) ** 2 - ((v - 0.42) / 0.38) ** 2)
            r = 214 + mottled * 32 + blush * 10
            g = 192 + mottled * 38 + blush * 16
            b = 44 + mottled * 16
            if pore > 0.925:
                r += 24
                g += 22
                b += 6
            pixels.append(Color(r, g, b))
    _LEMON_TEXTURE = PixelBuffer(width, height, pixels)
    return _LEMON_TEXTURE


def watermelon_texture(width: int = 384, height: int = 192) -> PixelBuffer:
    global _WATERMELON_TEXTURE
    if _WATERMELON_TEXTURE is not None and _WATERMELON_TEXTURE.width == width and _WATERMELON_TEXTURE.height == height:
        return _WATERMELON_TEXTURE
    pixels: list[Color] = []
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            stripe = 0.5 + 0.5 * sin((u * 7.0 + 0.18 * sin(v * tau * 2.0)) * tau)
            fine = 0.5 + 0.5 * sin((u * 31.0 + v * 3.0) * tau)
            dark = stripe > 0.62
            r = 28 + fine * 16
            g = (92 if dark else 142) + fine * (18 if dark else 28)
            b = (45 if dark else 72) + fine * 12
            pixels.append(Color(r, g, b))
    _WATERMELON_TEXTURE = PixelBuffer(width, height, pixels)
    return _WATERMELON_TEXTURE


def banana_texture(width: int = 384, height: int = 144) -> PixelBuffer:
    global _BANANA_TEXTURE
    if _BANANA_TEXTURE is not None and _BANANA_TEXTURE.width == width and _BANANA_TEXTURE.height == height:
        return _BANANA_TEXTURE
    bruise_patches = (
        (0.23, 0.30, 0.08, 0.16, 0.42),
        (0.63, 0.68, 0.06, 0.13, 0.36),
        (0.78, 0.42, 0.05, 0.11, 0.30),
    )
    pixels: list[Color] = []
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            ripeness = 0.5 + 0.5 * sin((u * 1.25 + 0.1 * sin(v * tau)) * tau)
            fiber = 0.5 + 0.5 * sin((u * 12.0 + v * 1.5) * tau)
            r = 226 + ripeness * 22 + fiber * 5
            g = 188 + ripeness * 26 + fiber * 8
            b = 54 + ripeness * 22

            end_darkening = max(max(0.0, 0.13 - u), max(0.0, u - 0.87)) / 0.13
            if end_darkening > 0.0:
                r = r * (1.0 - end_darkening * 0.35) + 112 * end_darkening * 0.35
                g = g * (1.0 - end_darkening * 0.35) + 82 * end_darkening * 0.35
                b = b * (1.0 - end_darkening * 0.35) + 34 * end_darkening * 0.35

            for center_u, center_v, radius_u, radius_v, strength in bruise_patches:
                v_distance = abs((v - center_v + 0.5) % 1.0 - 0.5)
                distance = ((u - center_u) / radius_u) ** 2 + (v_distance / radius_v) ** 2
                if distance < 1.0:
                    amount = (1.0 - distance) * strength
                    r = r * (1.0 - amount) + 128 * amount
                    g = g * (1.0 - amount) + 88 * amount
                    b = b * (1.0 - amount) + 38 * amount

            speckle_seed = (x * 73856093) ^ (y * 19349663) ^ 0xBABA
            speckle_seed = (speckle_seed ^ (speckle_seed >> 13)) * 1274126177
            speckle = (speckle_seed & 0xFFFF) / 0xFFFF
            if speckle > 0.982:
                amount = (speckle - 0.982) / 0.018 * 0.38
                r = r * (1.0 - amount) + 92 * amount
                g = g * (1.0 - amount) + 66 * amount
                b = b * (1.0 - amount) + 28 * amount

            pixels.append(Color(r, g, b))
    _BANANA_TEXTURE = PixelBuffer(width, height, pixels)
    return _BANANA_TEXTURE


def floor_texture(width: int = 512, height: int = 512) -> PixelBuffer:
    global _FLOOR_TEXTURE
    if _FLOOR_TEXTURE is not None and _FLOOR_TEXTURE.width == width and _FLOOR_TEXTURE.height == height:
        return _FLOOR_TEXTURE
    pixels: list[Color] = []
    for y in range(height):
        v = y / max(1, height - 1)
        tile_y = int(v * 8)
        local_v = (v * 8) % 1.0
        for x in range(width):
            u = x / max(1, width - 1)
            tile_x = int(u * 8)
            local_u = (u * 8) % 1.0
            grout = local_u < 0.035 or local_v < 0.035
            grain = 0.5 + 0.5 * sin((u * 18.0 + v * 7.0 + 0.08 * sin(v * tau * 6.0)) * tau)
            noise = _hash_unit(x // 2, y // 2, 0xF100)
            checker = (tile_x + tile_y) % 2
            base = 64 + checker * 10 + grain * 18 + noise * 9
            if grout:
                pixels.append(Color(34, 47, 49))
            else:
                pixels.append(Color(base * 0.72, base * 1.02, base * 1.04))
    _FLOOR_TEXTURE = PixelBuffer(width, height, pixels)
    return _FLOOR_TEXTURE


def sign_texture(label: str, quality_label: str, width: int = 384, height: int = 192) -> PixelBuffer:
    key = (label, quality_label, width, height)
    cached = _SIGN_TEXTURES.get(key)
    if cached is not None:
        return cached

    buffer = PixelBuffer.new(width, height, (16, 22, 27))
    for y in range(height):
        for x in range(width):
            u = x / max(1, width - 1)
            v = y / max(1, height - 1)
            grain = 0.5 + 0.5 * sin((u * 11.0 + 0.2 * sin(v * tau * 2.0)) * tau)
            dust = _hash_unit(x // 2, y // 2, 0x5190)
            edge = max(abs(u - 0.5) * 2.0, abs(v - 0.5) * 2.0)
            shade = 1.0 - max(0.0, edge - 0.72) * 0.42
            buffer.pixels[y * width + x] = Color((15 + grain * 18 + dust * 7) * shade, (21 + grain * 16 + dust * 6) * shade, (27 + grain * 14 + dust * 5) * shade)

    draw.rect(buffer, (8, 8), (width - 16, height - 16), (122, 90, 55), fill=False)
    draw.rect(buffer, (14, 14), (width - 28, height - 28), (42, 52, 58), fill=False)

    lines = (label.upper(), f"QUALITY {quality_label.upper()}")
    scale = 3
    if any(draw.text_size(line, scale=scale)[0] > width - 48 for line in lines):
        scale = 2
    if any(draw.text_size(line, scale=scale)[0] > width - 48 for line in lines):
        scale = 1
    line_spacing = 2
    total_height = len(lines) * 7 * scale + (len(lines) - 1) * line_spacing * scale
    y = (height - total_height) // 2
    for line in lines:
        text_width, text_height = draw.text_size(line, scale=scale, line_spacing=line_spacing)
        x = max(0, (width - text_width) // 2)
        draw.text(buffer, (x + 2, y + 2), line, (4, 7, 10), scale=scale, line_spacing=line_spacing)
        draw.text(buffer, (x, y), line, (235, 240, 226), scale=scale, line_spacing=line_spacing)
        y += text_height + line_spacing * scale

    _SIGN_TEXTURES[key] = buffer
    return buffer


class FruitBowlSimulation:
    """Small coordinated scene: driven bowl, dynamic fruit."""

    def __init__(self, *, bowl_material: str = "wood") -> None:
        self.time = 0.0
        self.world = PhysicsWorld(gravity=(0.0, -9.81, 0.0))
        self.bowl_material = bowl_material
        bowl_style = _bowl_style(bowl_material)
        self.bowl = KinematicBowl(
            center=self._driven_center(0.0),
            radius=1.35,
            depth=0.96,
            restitution=0.18,
            friction=0.48,
            squishiness=0.08,
            damping=0.16,
            material=bowl_style[0],
            visual_perturbation=bowl_style[1],
            visual_thickness=0.075,
        )
        self.floor = StaticPlane(
            point=(0.0, -1.75, 0.0),
            normal=(0.0, 1.0, 0.0),
            friction=0.28,
            restitution=0.35,
            squishiness=0.1,
            damping=0.12,
            material=Material(
                color=(255, 255, 255),
                texture=floor_texture(),
                absorption=(0.14, 0.08, 0.05),
                roughness=0.64,
                fuzziness=0.12,
                specular=0.04,
                shininess=10.0,
            ),
            size=6.0,
        )
        self.fruits = [
            Fruit(
                "apple",
                SphereBody(
                    position=(-0.42, -0.12, 0.08),
                    radius=0.23,
                    velocity=(0.6, 0.0, 0.12),
                    mass=0.85,
                    restitution=0.74,
                    friction=0.28,
                    static_friction=0.42,
                    kinetic_friction=0.28,
                    squishiness=0.38,
                    damping=0.28,
                    material=Material(
                        color=(255, 255, 255),
                        texture=apple_texture(),
                        absorption=(0.02, 0.15, 0.16),
                        roughness=0.2,
                        fuzziness=0.05,
                        specular=0.28,
                        shininess=34.0,
                        reflectivity=0.04,
                    ),
                    visual_perturbation=SurfacePerturbation(magnitude=0.012, scale=3.2, seed=12, octaves=3, gain=0.45),
                ),
            ),
            Fruit(
                "orange",
                SphereBody(
                    position=(0.28, -0.2, -0.1),
                    radius=0.27,
                    velocity=(-0.35, 0.12, -0.22),
                    mass=1.05,
                    restitution=0.52,
                    friction=0.34,
                    static_friction=0.48,
                    kinetic_friction=0.32,
                    squishiness=0.45,
                    damping=0.34,
                    material=Material(
                        color=(255, 255, 255),
                        texture=orange_texture(),
                        absorption=(0.02, 0.08, 0.22),
                        roughness=0.48,
                        fuzziness=0.2,
                        specular=0.1,
                        shininess=16.0,
                    ),
                    visual_perturbation=SurfacePerturbation(magnitude=0.035, scale=5.5, seed=31, octaves=4, gain=0.55),
                    collision_boundary=SphereCollider(radius=0.27),
                ),
            ),
            Fruit(
                "lemon",
                SphereBody(
                    position=(0.0, 0.12, 0.3),
                    radius=0.19,
                    velocity=(0.2, -0.1, -0.3),
                    mass=0.55,
                    restitution=0.46,
                    friction=0.31,
                    static_friction=0.46,
                    kinetic_friction=0.3,
                    squishiness=0.5,
                    damping=0.38,
                    material=Material(
                        color=(255, 255, 255),
                        texture=lemon_texture(),
                        absorption=(0.03, 0.03, 0.2),
                        roughness=0.42,
                        fuzziness=0.18,
                        specular=0.14,
                        shininess=18.0,
                    ),
                    visual_perturbation=SurfacePerturbation(magnitude=0.025, scale=6.4, seed=44, octaves=4, gain=0.55),
                    collision_boundary=SphereCollider(radius=0.19),
                ),
            ),
            Fruit(
                "watermelon",
                SphereBody(
                    position=(-0.32, -0.08, -0.34),
                    radius=0.42,
                    velocity=(-0.08, 0.0, 0.22),
                    mass=4.25,
                    restitution=0.48,
                    friction=0.16,
                    static_friction=0.3,
                    kinetic_friction=0.18,
                    squishiness=0.26,
                    damping=0.22,
                    material=Material(
                        texture=watermelon_texture(),
                        color=(50, 142, 78),
                        absorption=(0.14, 0.03, 0.12),
                        roughness=0.18,
                        fuzziness=0.03,
                        specular=0.12,
                        shininess=28.0,
                        reflectivity=0.03,
                    ),
                ),
                marker_color=(18, 55, 34),
            ),
            Fruit(
                "banana",
                SphereBody(
                    position=(0.52, 0.06, 0.24),
                    radius=0.27,
                    velocity=(-0.52, 0.06, -0.28),
                    angular_velocity=(0.0, 0.0, 0.7),
                    mass=0.82,
                    moment_of_inertia=0.09,
                    restitution=0.34,
                    friction=0.18,
                    static_friction=0.35,
                    kinetic_friction=0.22,
                    rolling_resistance=0.08,
                    squishiness=0.58,
                    damping=0.46,
                    material=Material(
                        texture=banana_texture(),
                        color=(244, 214, 78),
                        absorption=(0.01, 0.02, 0.08),
                        emission=(14, 11, 4),
                        roughness=0.32,
                        fuzziness=0.03,
                        specular=0.12,
                        shininess=20.0,
                        reflectivity=0.02,
                    ),
                ),
                visual="banana",
                banana_yaw=26.0,
            ),
        ]
        for fruit in self.fruits:
            fruit.sync_collision_boundary()
        self.world.add_bowl(self.bowl)
        self.world.add_plane(self.floor)
        for fruit in self.fruits:
            self.world.add_sphere(fruit.body)

    def step(self, dt: float, substeps: int = 3) -> None:
        if dt <= 0.0:
            return
        step_dt = dt / substeps
        for _ in range(substeps):
            self.time += step_dt
            self.bowl.set_center(self._driven_center(self.time), dt=step_dt)
            self.bowl.set_angular_velocity(self._driven_angular_velocity(self.time))
            for fruit in self.fruits:
                fruit.sync_collision_boundary()
            self.world.step(step_dt)

    @staticmethod
    def _driven_center(time: float) -> Vec3:
        phase = (time * 0.95) % 1.0
        toss = 0.18 * FruitBowlSimulation._pulse(phase, 0.12, 0.055)
        drop = -0.05 * FruitBowlSimulation._pulse(phase, 0.26, 0.075)
        catch = 0.05 * FruitBowlSimulation._pulse(phase, 0.58, 0.12)
        side_punch = FruitBowlSimulation._pulse(phase, 0.17, 0.045) - FruitBowlSimulation._pulse(phase, 0.44, 0.06)
        return Vec3(
            0.1 * sin(time * 1.7) + 0.09 * side_punch,
            0.0 + 0.05 * sin(time * tau * 0.72) + toss + drop + catch,
            0.1 * sin(time * 2.1 + 0.6) - 0.07 * side_punch,
        )

    @staticmethod
    def _driven_angular_velocity(time: float) -> Vec3:
        phase = (time * 0.95) % 1.0
        kick = FruitBowlSimulation._pulse(phase, 0.18, 0.06)
        counter_kick = FruitBowlSimulation._pulse(phase, 0.47, 0.075)
        return Vec3(
            0.55 * sin(time * 4.1) + 1.25 * kick,
            0.5 * sin(time * 2.6 + 0.7),
            0.5 * cos(time * 3.7) - 1.1 * counter_kick,
        )

    @staticmethod
    def _pulse(phase: float, center: float, width: float) -> float:
        distance = abs((phase - center + 0.5) % 1.0 - 0.5)
        if distance >= width:
            return 0.0
        amount = 1.0 - distance / width
        return amount * amount * (3.0 - 2.0 * amount)

    def scene(
        self,
        *,
        label: str = "KINEMATIC FRUIT BOWL",
        light_mode: str = "multiple",
        quality_label: str = "balanced",
    ) -> Scene:
        scene = Scene()
        scene.add(self.floor.to_primitive(), self.bowl.to_primitive())
        scene.add(*self._fixed_sign_primitives(label, quality_label))
        for fruit in self.fruits:
            scene.add(*fruit.to_primitives())
        mode = light_mode.lower().replace("_", "-")
        lights = self._lights_for_mode(light_mode)
        if mode in {"poly-lamp", "hanging-lamp"}:
            scene.add(self._hanging_lamp_primitive())
        else:
            scene.add(*self._light_markers(lights))
        for light in lights:
            scene.add_light(light)
        scene.add_bulletin(
            FloatingTextBulletin(
                "FLOATING BULLETIN\nE GRAB/DROP - WHEEL DEPTH",
                position=(0.0, 1.24, 0.04),
                screen_offset=(0, -20),
                color=(250, 244, 224),
                background=(15, 9, 5),
                padding=5,
                scale=1,
            )
        )
        return scene

    def pick_fruit(self, camera: Camera, *, max_distance: float = 5.5) -> tuple[Fruit, float] | None:
        _right, _up, forward = camera.basis()
        best: tuple[float, Fruit] | None = None
        for fruit in self.fruits:
            to_center = fruit.body.position - camera.position
            distance_along_ray = to_center.dot(forward)
            if distance_along_ray < camera.near or distance_along_ray > max_distance:
                continue
            closest = camera.position + forward * distance_along_ray
            pick_radius = max(0.14, fruit.body.radius * 1.45)
            if closest.distance_to(fruit.body.position) <= pick_radius:
                if best is None or distance_along_ray < best[0]:
                    best = (distance_along_ray, fruit)
        if best is None:
            return None
        return best[1], best[0]

    def move_held_fruit(self, fruit: Fruit, camera: Camera, distance: float, dt: float) -> None:
        _right, _up, forward = camera.basis()
        target = camera.position + forward * distance
        offset = target - fruit.body.position
        target_distance = offset.length()
        if target_distance <= 1e-4:
            fruit.body.velocity = fruit.body.velocity * 0.35
            fruit.body.angular_velocity = fruit.body.angular_velocity * 0.8
            return

        pull_speed = min(8.5, max(2.0, target_distance * 12.0))
        desired_velocity = offset.normalized() * pull_speed
        blend = min(1.0, max(0.0, dt) * 18.0)
        fruit.body.velocity = fruit.body.velocity * (1.0 - blend) + desired_velocity * blend
        fruit.body.angular_velocity = fruit.body.angular_velocity * 0.82
        fruit.sync_collision_boundary()

    def _fixed_sign_primitives(self, label: str, quality_label: str) -> tuple[Box | Mesh, ...]:
        board = Material(
            color=(255, 255, 255),
            texture=sign_texture(label, quality_label),
            roughness=0.5,
            fuzziness=0.08,
            specular=0.1,
            shininess=18.0,
        )
        trim = Material(color=(100, 72, 42), roughness=0.6, fuzziness=0.12, specular=0.04)
        panel_z = -1.212
        left = -2.23
        right = -1.33
        bottom = -1.075
        top = -0.685
        panel = Mesh(
            (
                Triangle((left, bottom, panel_z), (right, top, panel_z), (right, bottom, panel_z), board, (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)),
                Triangle((left, bottom, panel_z), (left, top, panel_z), (right, top, panel_z), board, (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)),
            )
        )
        return (
            Box((-1.78, -0.88, -1.18), (1.02, 0.46, 0.055), Material(color=(20, 26, 31), roughness=0.55, specular=0.08, shininess=16.0)),
            panel,
            Box((-2.24, -1.18, -1.16), (0.055, 0.72, 0.055), trim),
            Box((-1.32, -1.18, -1.16), (0.055, 0.72, 0.055), trim),
            Box((-1.78, -0.62, -1.16), (1.12, 0.055, 0.075), trim),
        )

    def _hanging_lamp_pose(self) -> tuple[Vec3, Vec3]:
        cord_start = Vec3(-0.2, 2.45, -0.36)
        sway = Vec3(0.22 * sin(self.time * 1.1), 0.0, 0.16 * sin(self.time * 1.36 + 0.65))
        shade_center = cord_start + Vec3(0.0, -0.96, 0.0) + sway
        return cord_start, shade_center

    def _hanging_lamp_primitive(self) -> HangingConeLampPrimitive:
        cord_start, shade_center = self._hanging_lamp_pose()
        return HangingConeLampPrimitive(
            cord_start,
            shade_center,
            color=(255, 226, 178),
            shade_color=(82, 62, 44),
            cord_color=(32, 28, 25),
            segments=16,
        )

    def _light_markers(self, lights: tuple[Sun | Lamp, ...]) -> tuple[Sphere | Line3 | LampPrimitive, ...]:
        markers: list[Sphere | Line3 | LampPrimitive] = []
        for light in lights:
            if isinstance(light, Lamp):
                markers.append(LampPrimitive(light.position, color=light.color, segments=8))
            elif isinstance(light, Sun):
                direction = (-light.direction).normalized(Vec3(0.0, 1.0, 0.0))
                start = Vec3(-2.15, 2.0, -2.2)
                end = start + direction * 0.42
                markers.append(Line3(start, end, Material(color=light.color, emission=(255, 255, 255))))
                markers.append(Sphere(start, 0.045, Material(color=light.color, emission=(255, 255, 255), diffuse=0.1)))
        return tuple(markers)

    def _lights_for_mode(self, light_mode: str) -> tuple[Sun | Lamp, ...]:
        mode = light_mode.lower().replace("_", "-")
        if mode not in {"multiple", "blinking", "multicolor", "color-shift-blink", "mirror-prelight", "poly-lamp", "hanging-lamp"}:
            mode = "multiple"
        if mode in {"poly-lamp", "hanging-lamp"}:
            cord_start, shade_center = self._hanging_lamp_pose()
            axis = (shade_center - cord_start).normalized(Vec3(0.0, -1.0, 0.0))
            bulb = shade_center + axis * 0.18
            pulse = 0.86 + 0.14 * sin(self.time * 1.65 + 0.4)
            return (
                Sun(direction=(-0.25, -0.9, -0.35), color=(180, 205, 235), intensity=0.14),
                Lamp(position=bulb, color=(255, 226, 178), intensity=9.2 * pulse),
            )
        if mode == "mirror-prelight":
            pulse = 0.75 + 0.25 * self._blink(1.25, 0.0)
            return (
                Sun(direction=(-0.2, -0.7, -1.0), color=(225, 235, 255), intensity=0.22),
                Lamp(position=(-0.15, 1.15, -1.0), color=(255, 246, 226), intensity=5.8 * pulse),
                Lamp(position=(0.95, 0.62, -0.55), color=(120, 180, 255), intensity=2.1),
            )
        if mode == "multiple":
            return (
                Sun(direction=(-0.35, -0.8, -1.0), color=(255, 245, 224), intensity=1.0),
                Lamp(position=(-1.8, 2.4, -2.2), color=(95, 145, 255), intensity=4.6),
                Lamp(position=(1.6, 1.3, -1.3), color=(255, 120, 88), intensity=3.0),
                Lamp(position=(0.2, 2.1, 1.6), color=(110, 255, 170), intensity=2.1),
            )
        if mode == "blinking":
            blink_a = self._blink(1.9, 0.0)
            blink_b = self._blink(2.35, 0.35)
            return (
                Sun(direction=(-0.35, -0.8, -1.0), color=(255, 248, 232), intensity=0.5 + 0.18 * blink_a),
                Lamp(position=(-1.8, 2.4, -2.2), color=(235, 242, 255), intensity=0.75 + 3.1 * blink_a),
                Lamp(position=(1.6, 1.3, -1.3), color=(255, 232, 218), intensity=0.55 + 2.5 * blink_b),
                Lamp(position=(0.2, 2.1, 1.6), color=(225, 255, 238), intensity=0.45 + 1.8 * self._blink(2.8, 0.7)),
            )
        if mode == "multicolor":
            return (
                Sun(direction=(-0.35, -0.8, -1.0), color=(255, 230, 182), intensity=0.62),
                Lamp(position=(-1.8, 2.4, -2.2), color=(64, 126, 255), intensity=3.4),
                Lamp(position=(1.6, 1.3, -1.3), color=(255, 70, 98), intensity=2.4),
                Lamp(position=(0.2, 2.1, 1.6), color=(70, 255, 145), intensity=1.9),
                Lamp(position=(-0.3, 1.35, 2.2), color=(255, 208, 72), intensity=1.5),
            )

        blink_a = self._blink(2.1, 0.0)
        blink_b = self._blink(2.8, 0.28)
        blink_c = self._blink(3.4, 0.63)
        return (
            Sun(direction=(-0.35, -0.8, -1.0), color=self._shift_color(0.0), intensity=0.38 + 0.22 * blink_a),
            Lamp(position=(-1.8, 2.4, -2.2), color=self._shift_color(0.13), intensity=0.45 + 3.2 * blink_a),
            Lamp(position=(1.6, 1.3, -1.3), color=self._shift_color(0.47), intensity=0.45 + 2.7 * blink_b),
            Lamp(position=(0.2, 2.1, 1.6), color=self._shift_color(0.72), intensity=0.35 + 2.1 * blink_c),
            Lamp(position=(-0.3, 1.35, 2.2), color=self._shift_color(0.91), intensity=0.3 + 1.7 * self._blink(4.0, 0.4)),
        )

    def _blink(self, frequency: float, phase: float) -> float:
        wave = sin((self.time * frequency + phase) * tau)
        return 1.0 if wave > -0.12 else 0.18

    def _shift_color(self, phase: float) -> tuple[int, int, int]:
        red = 0.5 + 0.5 * sin((self.time * 0.55 + phase) * tau)
        green = 0.5 + 0.5 * sin((self.time * 0.55 + phase + 0.33) * tau)
        blue = 0.5 + 0.5 * sin((self.time * 0.55 + phase + 0.66) * tau)
        return (int(70 + red * 185), int(70 + green * 185), int(70 + blue * 185))


def make_engine(renderer: str = "cpu", *, fast: bool = False) -> RenderEngine:
    if renderer == "py_gpu":
        from py_gpu.adapters.py3d import Py3DRasterRenderer

        return RenderEngine(Py3DRasterRenderer(reference_compatible=not fast))
    return RenderEngine(CPURenderer(cache_static_geometry=False))


def apply_cpu_reduced_specs(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "renderer", "cpu") == "cpu" and getattr(args, "cpu_reduced_specs", True):
        args.width = min(args.width, 320)
        args.height = min(args.height, 180)
        if hasattr(args, "fps"):
            args.fps = min(args.fps, 12)
        args.sphere_segments = min(args.sphere_segments, 10)
        args.sphere_rings = min(args.sphere_rings, 5)
        if getattr(args, "max_render_distance", None) is None:
            args.max_render_distance = 6.0
    return args


def make_settings(args: argparse.Namespace) -> RenderSettings:
    return RenderSettings(
        width=args.width,
        height=args.height,
        background=(8, 11, 15),
        ambient=getattr(args, "ambient", 0.0),
        gamma=getattr(args, "gamma", 1.0),
        light_wrap=getattr(args, "light_wrap", 0.0),
        bounce_light=getattr(args, "bounce_light", 0.0),
        tone_mapping=getattr(args, "tone_mapping", False),
        smooth_shading=args.smooth_shading,
        two_sided_lighting=False,
        ray_traced_shadows=getattr(args, "ray_traced_shadows", False),
        reflection_bounces=getattr(args, "reflection_bounces", 0),
        shadow_samples=getattr(args, "shadow_samples", 1),
        shadow_softness=getattr(args, "shadow_softness", 0.0),
        edge_highlight=getattr(args, "edge_highlight", False),
        edge_highlight_threshold_degrees=getattr(args, "edge_highlight_angle", 35.0),
        max_render_distance=getattr(args, "max_render_distance", None),
        sphere_segments=args.sphere_segments,
        sphere_rings=args.sphere_rings,
        texture_size=getattr(args, "texture_size", 256),
    )


def make_camera(yaw_degrees: float = 0.0, pitch_degrees: float = 50.0, distance: float = 4.2) -> Camera:
    target = Vec3(0.0, -0.28, 0.0)
    yaw = radians(yaw_degrees)
    pitch = radians(max(-80.0, min(80.0, pitch_degrees)))
    offset = Vec3(
        distance * sin(yaw) * cos(pitch),
        distance * sin(pitch),
        -distance * cos(yaw) * cos(pitch),
    )
    return Camera(position=target + offset, target=target, fov_degrees=48)


def render_still(args: argparse.Namespace) -> Path:
    simulation = FruitBowlSimulation(bowl_material=getattr(args, "bowl_material", "wood"))
    for _ in range(max(0, int(args.warmup * args.fps))):
        simulation.step(1.0 / args.fps)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_engine(getattr(args, "renderer", "cpu"), fast=getattr(args, "gpu_fast_render", False)).render(
        simulation.scene(
            label=getattr(args, "label", "KINEMATIC FRUIT BOWL"),
            light_mode=args.light_mode,
            quality_label=getattr(args, "quality", "balanced"),
        ),
        make_camera(),
        make_settings(args),
    ).to_png(output_path)
    print(f"Wrote {output_path}")
    return output_path


def find_ffmpeg(explicit_path: str | Path | None = None) -> str | None:
    if explicit_path is not None:
        explicit = str(explicit_path)
        path = Path(explicit)
        if path.exists():
            return str(path)
        found = shutil.which(explicit)
        if found is not None:
            return found

    env_path = os.environ.get("FFMPEG_BINARY")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return str(path)
        found = shutil.which(env_path)
        if found is not None:
            return found

    found = shutil.which("ffmpeg")
    if found is not None:
        return found

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffmpeg_missing_message() -> str:
    return (
        "ffmpeg executable not found. `pip install ffmpeg` installs a Python module, "
        "not ffmpeg.exe. Install the FFmpeg command-line binary, install optional "
        "`imageio-ffmpeg`, set FFMPEG_BINARY, or pass --ffmpeg C:\\path\\to\\ffmpeg.exe."
    )


def render_video(args: argparse.Namespace) -> Path:
    output_path = Path(args.video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulation = FruitBowlSimulation(bowl_material=getattr(args, "bowl_material", "wood"))
    engine = make_engine(getattr(args, "renderer", "cpu"), fast=getattr(args, "gpu_fast_render", False))
    settings = make_settings(args)
    ffmpeg = find_ffmpeg(getattr(args, "ffmpeg", None))
    if ffmpeg is None:
        message = ffmpeg_missing_message()
        if getattr(args, "require_ffmpeg", False):
            raise RuntimeError(message)
        frames_dir = output_path.with_suffix("") if output_path.suffix else output_path
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(args.frames):
            simulation.step(1.0 / args.fps)
            label = getattr(args, "label", "FRUIT BOWL")
            buffer = engine.render(
                simulation.scene(
                    label=f"{label} FRAME {frame:03d}",
                    light_mode=args.light_mode,
                    quality_label=getattr(args, "quality", "balanced"),
                ),
                make_camera(),
                settings,
            )
            buffer.to_png(frames_dir / f"frame_{frame:04d}.png")
        print(f"{message} Wrote PNG frames to {frames_dir}.")
        return frames_dir

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "image2pipe",
        "-framerate",
        str(args.fps),
        "-vcodec",
        "ppm",
        "-i",
        "-",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("could not open ffmpeg stdin")
    try:
        for frame in range(args.frames):
            simulation.step(1.0 / args.fps)
            label = getattr(args, "label", "FRUIT BOWL")
            buffer = engine.render(
                simulation.scene(
                    label=f"{label} FRAME {frame:03d}",
                    light_mode=args.light_mode,
                    quality_label=getattr(args, "quality", "balanced"),
                ),
                make_camera(),
                settings,
            )
            process.stdin.write(buffer.to_ppm_bytes())
    finally:
        process.stdin.close()
    result = process.wait()
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result}")
    print(f"Wrote {output_path}")
    return output_path


class LiveFruitBowlViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk

        self.tk = tk
        self.args = args
        self.simulation = FruitBowlSimulation(bowl_material=args.bowl_material)
        self.engine = make_engine(args.renderer, fast=getattr(args, "gpu_fast_render", False))
        self.fast_engine = make_engine(args.renderer, fast=True)
        self.settings = make_settings(args)
        self.yaw = 0.0
        self.pitch = 50.0
        self.distance = 4.2
        self.target = Vec3(0.0, -0.28, 0.0)
        self.drag_start: tuple[int, int] | None = None
        self.paused = False
        self.full_render = not getattr(args, "live_wireframe", True)
        self.base_photo = None
        self.photo = None
        self.window_icon = None

        self.root = tk.Tk()
        self.root.title("py_3d fruit bowl")
        self.root.geometry(f"{args.window_width}x{args.window_height}")
        self._set_window_icon()
        self.canvas = tk.Canvas(self.root, bg="#080b0f", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", lambda event: self.render_once())
        for key in ("<Left>", "<Right>", "<Up>", "<Down>", "<w>", "<s>", "<a>", "<d>", "<q>", "<e>", "<p>", "<space>", "<r>", "<x>"):
            self.root.bind(key, self.on_key)

    def run(self) -> None:
        self.render_once()
        self.root.after(max(1, int(1000 / self.args.fps)), self.tick)
        self.root.mainloop()

    def tick(self) -> None:
        if not self.paused:
            self.simulation.step(1.0 / self.args.fps)
        self.render_once()
        self.root.after(max(1, int(1000 / self.args.fps)), self.tick)

    def camera(self) -> Camera:
        yaw = radians(self.yaw)
        pitch = radians(max(-80.0, min(80.0, self.pitch)))
        offset = Vec3(
            self.distance * sin(yaw) * cos(pitch),
            self.distance * sin(pitch),
            -self.distance * cos(yaw) * cos(pitch),
        )
        return Camera(position=self.target + offset, target=self.target, fov_degrees=48)

    def _set_window_icon(self) -> None:
        icon_path = Path(__file__).resolve().parents[1] / "assets" / "py_3d_logo.png"
        if not icon_path.exists():
            return
        try:
            self.window_icon = self.tk.PhotoImage(file=str(icon_path))
            self.root.iconphoto(True, self.window_icon)
        except self.tk.TclError as exc:
            print(f"Could not load window icon {icon_path}: {exc}")

    def on_click(self, event) -> None:
        self.canvas.focus_set()
        self.drag_start = (event.x, event.y)

    def on_drag(self, event) -> None:
        if self.drag_start is None:
            return
        last_x, last_y = self.drag_start
        self.yaw -= (event.x - last_x) * 0.35
        self.pitch -= (event.y - last_y) * 0.25
        self.drag_start = (event.x, event.y)

    def on_release(self, event) -> None:
        del event
        self.drag_start = None

    def on_mouse_wheel(self, event) -> None:
        self.distance = max(1.6, self.distance * (0.9 if event.delta > 0 else 1.1))

    def on_key(self, event) -> None:
        key = event.keysym.lower()
        if key == "left":
            self.yaw -= 5.0
        elif key == "right":
            self.yaw += 5.0
        elif key == "up":
            self.pitch -= 4.0
        elif key == "down":
            self.pitch += 4.0
        elif key == "w":
            self.distance = max(1.6, self.distance * 0.9)
        elif key == "s":
            self.distance *= 1.1
        elif key == "a":
            self.target = self.target + self.camera().basis()[0] * -0.15
        elif key == "d":
            self.target = self.target + self.camera().basis()[0] * 0.15
        elif key == "q":
            self.target = self.target + Vec3(0.0, 0.15, 0.0)
        elif key == "e":
            self.target = self.target + Vec3(0.0, -0.15, 0.0)
        elif key == "p":
            self.save_snapshot()
        elif key == "space":
            self.paused = not self.paused
        elif key == "r":
            self.full_render = not self.full_render
        elif key == "x":
            self.simulation = FruitBowlSimulation(bowl_material=self.args.bowl_material)

    def render_once(self) -> None:
        settings = self._active_settings()
        output = self._active_engine().render(
            self.simulation.scene(
                label=getattr(self.args, "label", "KINEMATIC FRUIT BOWL"),
                light_mode=self.args.light_mode,
                quality_label=getattr(self.args, "quality", "balanced"),
            ),
            self.camera(),
            settings,
        )
        self.base_photo = self._photo_from_buffer(output)
        self.photo = self.base_photo
        if self.args.fit_window:
            width = max(1, self.canvas.winfo_width())
            height = max(1, self.canvas.winfo_height())
            scale_x = width // output.width
            scale_y = height // output.height
            if scale_x >= 1 and scale_y >= 1 and output.width * scale_x == width and output.height * scale_y == height:
                self.photo = self.base_photo.zoom(scale_x, scale_y)
            else:
                self.photo = self._photo_from_buffer(output.resized_nearest(width, height))
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def save_snapshot(self) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "fruit_bowl_live_snapshot.png"
        self._active_engine().render(
            self.simulation.scene(
                label=getattr(self.args, "label", "KINEMATIC FRUIT BOWL"),
                light_mode=self.args.light_mode,
                quality_label=getattr(self.args, "quality", "balanced"),
            ),
            self.camera(),
            self._active_settings(),
        ).to_png(path)
        print(f"Wrote {path}")

    def _active_engine(self) -> RenderEngine:
        return self.engine if self.full_render else self.fast_engine

    def _active_settings(self) -> RenderSettings:
        if self.full_render:
            return self.settings
        return replace(
            self.settings,
            wireframe=True,
            smooth_shading=False,
            ray_traced_shadows=False,
            edge_highlight=False,
            sphere_segments=max(8, min(self.settings.sphere_segments, 10)),
            sphere_rings=max(4, min(self.settings.sphere_rings, 5)),
        )

    def _photo_from_buffer(self, buffer):
        ppm = buffer.to_ppm_bytes()
        try:
            return self.tk.PhotoImage(data=ppm, format="PPM")
        except self.tk.TclError:
            return self.tk.PhotoImage(data=base64.b64encode(ppm).decode("ascii"), format="PPM")


class GLFruitBowlViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        import pygame
        from py_3d.live import LiveFlyCamera, LiveMenu, LiveMenuOption, ModernGLLiveRenderer

        self.pygame = pygame
        self.args = args
        self.simulation = FruitBowlSimulation(bowl_material=args.bowl_material)
        self.snapshot_engine = make_engine(args.renderer, fast=getattr(args, "gpu_fast_render", True))
        self.settings = make_settings(args)
        self._last_safe_settings = self.settings
        self._pending_settings = self.settings
        self._pending_quality = getattr(args, "quality", "balanced")
        self._apply_watch_frames = 0
        self._menu_status = ""
        self.sky = SkyPrefab(time_of_day=14.0, cycle_enabled=False, stars_enabled=True, clouds_enabled=True)
        base_camera = make_camera()
        self.camera_controller = LiveFlyCamera.looking_at(
            base_camera.position,
            base_camera.target,
            fov_degrees=base_camera.fov_degrees,
            speed=2.6,
        )
        self.keys: set[str] = set()
        self.paused = False
        self.full_render = not getattr(args, "live_wireframe", False)
        self.held_fruit: Fruit | None = None
        self.held_distance = 1.6
        self.quality_presets = render_quality_presets()
        preferred_order = ("fast", "balanced", "high", "ultra", "poly")
        extras = tuple(name for name in self.quality_presets if name not in preferred_order)
        self.quality_order = tuple(name for name in preferred_order if name in self.quality_presets) + extras
        self.renderer = ModernGLLiveRenderer(
            args.window_width,
            args.window_height,
            title="py_3d fruit bowl - OpenGL live",
            vsync=getattr(args, "vsync", True),
        )
        self.renderer.menu = LiveMenu(
            "py_3d fruit bowl",
            (
                LiveMenuOption("done", "Done"),
                LiveMenuOption("apply", "Apply"),
                LiveMenuOption("cancel", "Exit menu"),
                LiveMenuOption("quality_next", "Quality preset"),
                LiveMenuOption("poly_up", "More polygons"),
                LiveMenuOption("poly_down", "Fewer polygons"),
                LiveMenuOption("reflections_up", "More reflections"),
                LiveMenuOption("reflections_down", "Fewer reflections"),
                LiveMenuOption("smooth", "Smooth shading"),
                LiveMenuOption("texture_up", "Texture resolution"),
                LiveMenuOption("tone_mapping", "Tone mapping"),
                LiveMenuOption("sky_cycle", "Day/night cycle"),
                LiveMenuOption("sky_time_up", "Time later"),
                LiveMenuOption("sky_time_down", "Time earlier"),
                LiveMenuOption("sky_sun_up", "Sun higher"),
                LiveMenuOption("sky_sun_down", "Sun lower"),
                LiveMenuOption("sky_clouds", "Clouds"),
                LiveMenuOption("sky_stars", "Stars"),
                LiveMenuOption("toggle_render", "Toggle wire/poly"),
                LiveMenuOption("pause", "Pause/run physics"),
                LiveMenuOption("reset", "Reset bowl"),
                LiveMenuOption("snapshot", "Save snapshot"),
                LiveMenuOption("quit", "Quit demo"),
            ),
        )
        self._refresh_menu_options()
        self.renderer.set_mouse_captured(True)
        self._last_title_update = 0

    def run(self) -> None:
        clock = self.pygame.time.Clock()
        dt = 1.0 / self.args.fps if self.args.fps > 0 else 1.0 / 120.0
        running = True
        try:
            while running:
                for event in self.pygame.event.get():
                    if event.type == self.pygame.QUIT:
                        running = False
                    elif self.renderer.handle_resize_event(event):
                        continue
                    elif self.renderer.menu.visible and event.type in (self.pygame.MOUSEMOTION, self.pygame.MOUSEBUTTONDOWN, self.pygame.MOUSEWHEEL):
                        menu_action = self.renderer.menu.handle_mouse_event(event, self.pygame)
                        if menu_action is not None:
                            running = self._handle_menu_action(menu_action)
                    elif event.type == self.pygame.MOUSEBUTTONDOWN and not self.renderer.menu.visible:
                        self.renderer.set_mouse_captured(True)
                        self.on_mouse_button(event.button)
                    elif event.type == self.pygame.MOUSEWHEEL and not self.renderer.menu.visible:
                        self.on_mouse_wheel(event.y)
                    elif event.type == self.pygame.MOUSEMOTION and self.renderer.mouse_captured:
                        self.camera_controller.look(event.rel[0], event.rel[1])
                    elif event.type == self.pygame.KEYDOWN:
                        running = self.on_key_down(event.key)
                    elif event.type == self.pygame.KEYUP:
                        self.on_key_up(event.key)

                self.camera_controller.move(self.keys, dt)
                self.sky.step(dt)
                self.update_held_fruit(dt)
                if not self.paused:
                    self.simulation.step(dt, substeps=6 if self.held_fruit is not None else 3)
                render_camera = self.camera_controller.smoothed_camera(dt)
                self._update_hud()
                scene = self.simulation.scene(
                    label=getattr(self.args, "label", "KINEMATIC FRUIT BOWL"),
                    light_mode=self.args.light_mode,
                    quality_label=getattr(self.args, "quality", "balanced"),
                )
                self._apply_sky(scene)
                stats = self.renderer.render(
                    scene,
                    render_camera,
                    self.sky.settings_for(self._active_settings()),
                )
                self._guard_applied_settings(stats)
                self._update_title(stats)
                tick_ms = clock.tick(self.args.fps if self.args.fps > 0 else 0)
                dt = max(1.0 / 240.0, min(0.05, tick_ms / 1000.0))
        finally:
            self.renderer.close()

    def camera(self) -> Camera:
        return self.camera_controller.camera()

    def on_mouse_button(self, button: int) -> None:
        if button == 3:
            self.drop_held_fruit()
        elif button == 4:
            self.on_mouse_wheel(1)
        elif button == 5:
            self.on_mouse_wheel(-1)

    def on_mouse_wheel(self, amount: int) -> None:
        if self.held_fruit is None:
            return
        self.held_distance = max(0.55, min(6.0, self.held_distance + amount * 0.22))

    def toggle_grabbed_fruit(self) -> None:
        if self.held_fruit is not None:
            self.drop_held_fruit()
            return
        picked = self.simulation.pick_fruit(self.camera())
        if picked is not None:
            self.held_fruit, self.held_distance = picked
            minimum_distance = self.held_fruit.body.collision_radius() + self.camera().near + 0.35
            self.held_distance = max(minimum_distance, min(5.5, self.held_distance))

    def update_held_fruit(self, dt: float) -> None:
        if self.held_fruit is None:
            return
        self.simulation.move_held_fruit(self.held_fruit, self.camera(), self.held_distance, dt)

    def drop_held_fruit(self) -> None:
        self.held_fruit = None

    def _apply_sky(self, scene: Scene) -> Scene:
        scene.lights = [light for light in scene.lights if not isinstance(light, Sun)]
        self.sky.apply(scene)
        return scene

    def on_key_down(self, key: int) -> bool:
        pygame = self.pygame
        menu_action = self.renderer.menu.handle_key(key, pygame)
        if menu_action is not None:
            return self._handle_menu_action(menu_action)
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.add(movement_key)
        elif key == pygame.K_LEFT:
            self.camera_controller.look(-36.0, 0.0)
        elif key == pygame.K_RIGHT:
            self.camera_controller.look(36.0, 0.0)
        elif key == pygame.K_UP:
            self.camera_controller.look(0.0, -36.0)
        elif key == pygame.K_DOWN:
            self.camera_controller.look(0.0, 36.0)
        elif key == pygame.K_p:
            self.save_snapshot()
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
            self._refresh_menu_options()
        elif key == pygame.K_r:
            self.full_render = not self.full_render
            self._refresh_menu_options()
        elif key == pygame.K_x:
            self.simulation = FruitBowlSimulation(bowl_material=self.args.bowl_material)
            self.drop_held_fruit()
        elif key == pygame.K_e:
            self.toggle_grabbed_fruit()
        return True

    def _update_hud(self) -> None:
        held = self.held_fruit.name.upper() if self.held_fruit is not None else "NONE"
        quality = getattr(self.args, "quality", "custom").upper()
        self.renderer.hud.set(
            HUDRect((12, 12), (230, 74), (3, 7, 10), alpha=0.58),
            HUDText(
                f"E GRAB/DROP\nHELD {held}\n{quality}  {self.settings.reflection_bounces} REFL\nSKY {self.sky.time_of_day:04.1f}H",
                (20, 20),
                color=(238, 245, 255),
                alpha=0.92,
                scale=1,
            ),
        )

    def _refresh_menu_options(self) -> None:
        from py_3d.live import LiveMenuOption

        menu = self.renderer.menu
        previous_action = menu.selected_action() if menu.options else "done"
        settings = self._menu_settings()
        quality = self._pending_quality if menu.visible else getattr(self.args, "quality", "custom")
        mode = "filled" if self.full_render else "wire"
        options = (
            LiveMenuOption("done", "Done"),
            LiveMenuOption("apply", "Apply", self._menu_status),
            LiveMenuOption("cancel", "Exit Menu"),
            LiveMenuOption("quality_next", "Quality", quality, "Graphics"),
            LiveMenuOption("poly_up", "Polygons +", f"{settings.sphere_segments}x{settings.sphere_rings}", "Graphics"),
            LiveMenuOption("poly_down", "Polygons -", f"{settings.sphere_segments}x{settings.sphere_rings}", "Graphics"),
            LiveMenuOption("reflections_up", "Reflections +", str(settings.reflection_bounces), "Graphics"),
            LiveMenuOption("reflections_down", "Reflections -", str(settings.reflection_bounces), "Graphics"),
            LiveMenuOption("smooth", "Smooth", "on" if settings.smooth_shading else "off", "Graphics"),
            LiveMenuOption("texture_up", "Texture", str(settings.texture_size), "Graphics"),
            LiveMenuOption("tone_mapping", "Tone Map", "on" if settings.tone_mapping else "off", "Graphics"),
            LiveMenuOption("sky_cycle", "Cycle", "on" if self.sky.cycle_enabled else "off", "Sky"),
            LiveMenuOption("sky_time_up", "Later", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_time_down", "Earlier", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_sun_up", "Sun +", f"{self.sky.effective_sun_elevation_degrees():0.0f} deg", "Sky"),
            LiveMenuOption("sky_sun_down", "Sun -", f"{self.sky.effective_sun_elevation_degrees():0.0f} deg", "Sky"),
            LiveMenuOption("sky_clouds", "Clouds", "on" if self.sky.clouds_enabled else "off", "Sky"),
            LiveMenuOption("sky_stars", "Stars", "on" if self.sky.stars_enabled else "off", "Sky"),
            LiveMenuOption("toggle_render", "Wire/Fill", mode, "Graphics"),
            LiveMenuOption("pause", "Pause", "paused" if self.paused else "running", "Physics"),
            LiveMenuOption("reset", "Reset", "bowl", "Physics"),
            LiveMenuOption("snapshot", "Snapshot", "PNG", "Demo"),
            LiveMenuOption("quit", "Quit demo"),
        )
        menu.options = options
        actions = [option.action for option in options]
        menu.selected_index = actions.index(previous_action) if previous_action in actions else min(menu.selected_index, len(options) - 1)

    def _begin_menu_edit(self) -> None:
        self._pending_settings = self.settings
        self._pending_quality = getattr(self.args, "quality", "custom")
        self._menu_status = "pending"
        self._refresh_menu_options()

    def _menu_settings(self) -> RenderSettings:
        return self._pending_settings if self.renderer.menu.visible else self.settings

    def _set_menu_settings(self, settings: RenderSettings) -> None:
        if self.renderer.menu.visible:
            self._pending_settings = settings
            self._menu_status = "pending"
        else:
            self.settings = settings

    def _apply_pending_settings(self) -> None:
        self._last_safe_settings = self.settings
        self.settings = self._pending_settings
        setattr(self.args, "quality", self._pending_quality)
        self._sync_args_from_settings(self.settings)
        self._apply_watch_frames = 45
        self._menu_status = "applied"
        self._refresh_menu_options()

    def _discard_pending_settings(self) -> None:
        self._pending_settings = self.settings
        self._pending_quality = getattr(self.args, "quality", "custom")
        self._menu_status = "discarded"

    def _guard_applied_settings(self, stats) -> None:
        if self._apply_watch_frames <= 0:
            return
        self._apply_watch_frames -= 1
        if stats.total_seconds <= 1.25:
            if self._apply_watch_frames == 0:
                self._last_safe_settings = self.settings
            return
        self.settings = self._last_safe_settings
        self._pending_settings = self.settings
        self._sync_args_from_settings(self.settings)
        self._menu_status = "reverted slow frame"
        self._apply_watch_frames = 0
        self._refresh_menu_options()

    def _sync_args_from_settings(self, settings: RenderSettings) -> None:
        for key in (
            "gamma",
            "light_wrap",
            "bounce_light",
            "tone_mapping",
            "reflection_bounces",
            "shadow_samples",
            "shadow_softness",
            "max_render_distance",
            "sphere_segments",
            "sphere_rings",
            "smooth_shading",
            "texture_size",
        ):
            if hasattr(self.args, key):
                setattr(self.args, key, getattr(settings, key))

    def _cycle_quality(self, amount: int) -> None:
        if not self.quality_order:
            return
        current = self._pending_quality if self.renderer.menu.visible else getattr(self.args, "quality", self.quality_order[0])
        index = self.quality_order.index(current) if current in self.quality_order else -1
        next_quality = self.quality_order[(index + amount) % len(self.quality_order)]
        self._apply_quality(next_quality)

    def _apply_quality(self, quality: str) -> None:
        preset = self.quality_presets.get(quality)
        if preset is None:
            return
        allowed = {
            "width",
            "height",
            "gamma",
            "light_wrap",
            "bounce_light",
            "tone_mapping",
            "reflection_bounces",
            "shadow_samples",
            "shadow_softness",
            "max_render_distance",
            "sphere_segments",
            "sphere_rings",
            "smooth_shading",
            "texture_size",
        }
        updates = {key: value for key, value in preset.items() if key in allowed}
        self._set_menu_settings(replace(self._menu_settings(), **updates))
        self._pending_quality = quality
        if not self.renderer.menu.visible:
            setattr(self.args, "quality", quality)
            self._sync_args_from_settings(self.settings)

    def _mark_custom_quality(self) -> None:
        if self.renderer.menu.visible:
            self._pending_quality = "custom"
        else:
            setattr(self.args, "quality", "custom")

    def _adjust_polygons(self, amount: int) -> None:
        self._mark_custom_quality()
        settings = self._menu_settings()
        segments = max(6, min(48, settings.sphere_segments + amount * 2))
        rings = max(3, min(24, settings.sphere_rings + amount))
        self._set_menu_settings(replace(settings, sphere_segments=segments, sphere_rings=rings))

    def _adjust_reflections(self, amount: int) -> None:
        self._mark_custom_quality()
        settings = self._menu_settings()
        bounces = max(0, min(5, settings.reflection_bounces + amount))
        self._set_menu_settings(replace(settings, reflection_bounces=bounces))

    def _adjust_texture_size(self, amount: int) -> None:
        self._mark_custom_quality()
        sizes = (128, 192, 256, 384, 512, 768)
        settings = self._menu_settings()
        current_index = min(range(len(sizes)), key=lambda index: abs(sizes[index] - settings.texture_size))
        size = sizes[(current_index + amount) % len(sizes)]
        self._set_menu_settings(replace(settings, texture_size=size))

    def _toggle_smooth_shading(self) -> None:
        self._mark_custom_quality()
        settings = self._menu_settings()
        self._set_menu_settings(replace(settings, smooth_shading=not settings.smooth_shading))

    def _toggle_tone_mapping(self) -> None:
        self._mark_custom_quality()
        settings = self._menu_settings()
        self._set_menu_settings(replace(settings, tone_mapping=not settings.tone_mapping))

    def _handle_menu_action(self, action: str) -> bool:
        if action in {"handled", "navigate"}:
            return True
        if action == "opened":
            self.keys.clear()
            self.renderer.set_mouse_captured(False)
            self._begin_menu_edit()
            return True
        if action == "quit":
            return False
        if action == "apply":
            self._apply_pending_settings()
        elif action == "done":
            self._apply_pending_settings()
            self.renderer.menu.close()
            self.renderer.set_mouse_captured(True)
            return True
        elif action in {"cancel", "resume"}:
            self._discard_pending_settings()
            self.renderer.menu.close()
            self.renderer.set_mouse_captured(True)
            return True
        elif action == "quality_next":
            self._cycle_quality(1)
        elif action == "poly_up":
            self._adjust_polygons(1)
        elif action == "poly_down":
            self._adjust_polygons(-1)
        elif action == "reflections_up":
            self._adjust_reflections(1)
        elif action == "reflections_down":
            self._adjust_reflections(-1)
        elif action == "smooth":
            self._toggle_smooth_shading()
        elif action == "texture_up":
            self._adjust_texture_size(1)
        elif action == "tone_mapping":
            self._toggle_tone_mapping()
        elif action == "sky_cycle":
            self.sky.toggle_cycle()
        elif action == "sky_time_up":
            self.sky.adjust_time(1.0)
        elif action == "sky_time_down":
            self.sky.adjust_time(-1.0)
        elif action == "sky_sun_up":
            self.sky.adjust_sun_angle(elevation_delta=5.0)
        elif action == "sky_sun_down":
            self.sky.adjust_sun_angle(elevation_delta=-5.0)
        elif action == "sky_clouds":
            self.sky.toggle_clouds()
        elif action == "sky_stars":
            self.sky.toggle_stars()
        elif action == "toggle_render":
            self.full_render = not self.full_render
        elif action == "pause":
            self.paused = not self.paused
        elif action == "reset":
            self.simulation = FruitBowlSimulation(bowl_material=self.args.bowl_material)
            self.drop_held_fruit()
        elif action == "snapshot":
            self.save_snapshot()
        self._refresh_menu_options()
        return True

    def on_key_up(self, key: int) -> None:
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.discard(movement_key)

    def _movement_key(self, key: int) -> str | None:
        pygame = self.pygame
        if key == pygame.K_w:
            return "w"
        if key == pygame.K_a:
            return "a"
        if key == pygame.K_s:
            return "s"
        if key == pygame.K_d:
            return "d"
        if key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
            return "shift"
        if key in (pygame.K_LCTRL, pygame.K_RCTRL):
            return "ctrl"
        return None

    def save_snapshot(self) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "fruit_bowl_live_snapshot.png"
        scene = self.simulation.scene(
            label=getattr(self.args, "label", "KINEMATIC FRUIT BOWL"),
            light_mode=self.args.light_mode,
            quality_label=getattr(self.args, "quality", "balanced"),
        )
        self._apply_sky(scene)
        self.snapshot_engine.render(
            scene,
            self.camera(),
            self.sky.settings_for(self._active_settings()),
        ).to_png(path)
        print(f"Wrote {path}")

    def _active_settings(self) -> RenderSettings:
        if self.full_render:
            return self.settings
        return replace(
            self.settings,
            wireframe=True,
            smooth_shading=False,
            ray_traced_shadows=False,
            edge_highlight=False,
            sphere_segments=max(8, min(self.settings.sphere_segments, 10)),
            sphere_rings=max(4, min(self.settings.sphere_rings, 5)),
        )

    def _update_title(self, stats) -> None:
        ticks = self.pygame.time.get_ticks()
        if ticks - self._last_title_update < 400:
            return
        self._last_title_update = ticks
        mode = "filled" if self.full_render else "wire"
        held = f" - holding {self.held_fruit.name}" if self.held_fruit is not None else ""
        self.renderer.set_title(
            f"py_3d fruit bowl - OpenGL live - {mode} - {getattr(self.args, 'quality', 'custom')} "
            f"- {self.settings.sphere_segments}x{self.settings.sphere_rings} - {self.settings.reflection_bounces} refl{held} "
            f"- {stats.approx_fps:0.1f} render fps "
            f"({stats.build_seconds * 1000:0.1f} ms build, {stats.draw_seconds * 1000:0.1f} ms draw)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render or run a live kinematic fruit bowl demo.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "fruit_bowl.png")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--require-ffmpeg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-still", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--warmup", type=float, default=1.25)
    parser.add_argument("--ambient", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--label", default="KINEMATIC FRUIT BOWL")
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=204)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--fit-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-wireframe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quality", help="Render quality preset from USER/settings.json, e.g. fast, balanced, high, ultra, poly.")
    parser.add_argument(
        "--light-mode",
        choices=("multiple", "blinking", "multicolor", "color-shift-blink", "mirror-prelight", "poly-lamp", "hanging-lamp"),
        default="multiple",
    )
    parser.add_argument("--bowl-material", choices=("wood", "mirror"), default="wood")
    parser.add_argument("--renderer", choices=("cpu", "py_gpu"), default="cpu")
    parser.add_argument("--gpu-fast-render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu-reduced-specs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-shading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--texture-size", type=int, default=256)
    parser.add_argument("--light-wrap", type=float, default=0.0)
    parser.add_argument("--bounce-light", type=float, default=0.0)
    parser.add_argument("--tone-mapping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ray-traced-shadows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reflection-bounces", type=int, default=0)
    parser.add_argument("--shadow-samples", type=int, default=1)
    parser.add_argument("--shadow-softness", type=float, default=0.0)
    parser.add_argument("--edge-highlight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-highlight-angle", type=float, default=35.0)
    parser.add_argument("--max-render-distance", type=float)
    parser.add_argument("--sphere-segments", type=int, default=14)
    parser.add_argument("--sphere-rings", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = apply_cpu_reduced_specs(apply_render_quality(parse_args()))
    if args.fps < 0:
        raise ValueError("fps must be non-negative")
    if args.frames <= 0:
        raise ValueError("frames must be positive")
    if args.live:
        if args.renderer == "py_gpu":
            try:
                GLFruitBowlViewer(args).run()
                return
            except Exception as exc:
                print(f"OpenGL live renderer unavailable, falling back to Tk PixelBuffer path: {exc}")
        if args.fps == 0:
            raise ValueError("fps must be positive for the Tk PixelBuffer fallback")
        LiveFruitBowlViewer(args).run()
        return
    if args.fps == 0:
        raise ValueError("fps must be positive for still/video rendering")
    if args.video is not None:
        if args.write_still:
            render_still(args)
        render_video(args)
        return
    render_still(args)


if __name__ == "__main__":
    main()
