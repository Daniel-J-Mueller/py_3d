"""Reusable sky, sun, stars, and cloud helpers for live demos."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, floor, radians, sin, tau

from .color import Color
from .lights import Sun
from .materials import Material
from .math3d import Vec3
from .primitives import BlobSurface, Sphere
from .render import RenderSettings
from .scene import Scene


@dataclass
class SkyPrefab:
    """A lightweight sky controller with manual sun angle or day/night time."""

    time_of_day: float = 14.0
    cycle_enabled: bool = False
    manual_sun_angle: bool = False
    sun_azimuth_degrees: float = -35.0
    sun_elevation_degrees: float = 46.0
    day_length_seconds: float = 180.0
    night_length_seconds: float = 90.0
    stars_enabled: bool = True
    clouds_enabled: bool = True
    radius: float = 34.0
    star_count: int = 54
    cloud_count: int = 7
    cloud_seed: int = 0

    def step(self, dt: float) -> None:
        if not self.cycle_enabled or dt <= 0.0:
            return
        duration = self.day_length_seconds if self.is_daytime else self.night_length_seconds
        hours = 12.0 * dt / max(1.0, duration)
        self.set_time(self.time_of_day + hours)

    def set_time(self, hour: float) -> None:
        self.time_of_day = float(hour) % 24.0

    @property
    def is_daytime(self) -> bool:
        return 6.0 <= self.time_of_day < 18.0

    @property
    def night_amount(self) -> float:
        daylight = self.daylight_amount()
        return max(0.0, min(1.0, 1.0 - daylight))

    def daylight_amount(self) -> float:
        elevation = self.effective_sun_elevation_degrees()
        return max(0.0, min(1.0, (elevation + 8.0) / 54.0))

    def effective_sun_elevation_degrees(self) -> float:
        if self.manual_sun_angle:
            return self.sun_elevation_degrees
        phase = ((self.time_of_day - 6.0) / 24.0) * tau
        return sin(phase) * 78.0

    def effective_sun_azimuth_degrees(self) -> float:
        if self.manual_sun_angle:
            return self.sun_azimuth_degrees
        day_progress = ((self.time_of_day - 6.0) % 24.0) / 24.0
        return -92.0 + day_progress * 360.0

    def sun_vector(self) -> Vec3:
        """Return direction from the scene toward the sun."""

        azimuth = radians(self.effective_sun_azimuth_degrees())
        elevation = radians(self.effective_sun_elevation_degrees())
        horizontal = max(0.001, cos(elevation))
        return Vec3(sin(azimuth) * horizontal, sin(elevation), cos(azimuth) * horizontal).normalized(Vec3(0.0, 1.0, 0.0))

    def sun_light(self) -> Sun:
        daylight = self.daylight_amount()
        warmth = max(0.0, min(1.0, 1.0 - abs(self.effective_sun_elevation_degrees() - 22.0) / 70.0))
        color = Color(
            int(150 + daylight * 92 + warmth * 13),
            int(166 + daylight * 70 + warmth * 20),
            int(190 + daylight * 42),
        )
        return Sun(direction=-self.sun_vector(), color=color, intensity=0.16 + daylight * 1.04)

    def background_color(self) -> Color:
        daylight = self.daylight_amount()
        dusk = max(0.0, min(1.0, 1.0 - abs(self.effective_sun_elevation_degrees()) / 22.0))
        night = 1.0 - daylight
        return Color(
            int(5 * night + 46 * daylight + 28 * dusk),
            int(8 * night + 88 * daylight + 23 * dusk),
            int(18 * night + 138 * daylight + 30 * dusk),
        )

    def apply(self, scene: Scene, *, add_sun: bool = True) -> Scene:
        scene.background = self.background_color()
        if add_sun:
            scene.add_light(self.sun_light())
        if self.stars_enabled and self.night_amount > 0.55:
            scene.add(*self.star_primitives())
        if self.clouds_enabled and self.daylight_amount() > 0.08:
            scene.add(*self.cloud_primitives())
        return scene

    def settings_for(self, settings: RenderSettings) -> RenderSettings:
        return replace(settings, background=self.background_color())

    def toggle_stars(self) -> None:
        self.stars_enabled = not self.stars_enabled

    def toggle_clouds(self) -> None:
        self.clouds_enabled = not self.clouds_enabled

    def toggle_cycle(self) -> None:
        self.cycle_enabled = not self.cycle_enabled

    def adjust_time(self, hours: float) -> None:
        self.set_time(self.time_of_day + hours)
        self.manual_sun_angle = False

    def adjust_sun_angle(self, *, elevation_delta: float = 0.0, azimuth_delta: float = 0.0) -> None:
        self.manual_sun_angle = True
        self.sun_elevation_degrees = max(-18.0, min(86.0, self.sun_elevation_degrees + elevation_delta))
        self.sun_azimuth_degrees = (self.sun_azimuth_degrees + azimuth_delta + 180.0) % 360.0 - 180.0

    def star_primitives(self) -> tuple[Sphere, ...]:
        material = Material(color=(235, 240, 255), emission=(180, 190, 255), diffuse=0.0)
        stars: list[Sphere] = []
        count = max(0, int(self.star_count))
        for index in range(count):
            x = _pseudo_random(index, 0.13) * 2.0 - 1.0
            z = _pseudo_random(index, 0.57) * 2.0 - 1.0
            y = 0.28 + _pseudo_random(index, 0.91) * 0.7
            direction = Vec3(x, y, z).normalized(Vec3(0.0, 1.0, 0.0))
            radius = 0.018 + _pseudo_random(index, 0.37) * 0.018
            stars.append(Sphere(direction * self.radius, radius, material))
        return tuple(stars)

    def cloud_primitives(self) -> tuple[BlobSurface, ...]:
        clouds: list[BlobSurface] = []
        count = max(0, int(self.cloud_count))
        for index in range(count):
            azimuth = radians(-130.0 + index * (260.0 / max(1, count - 1)) + _pseudo_random(index, self.cloud_seed + 0.29) * 22.0)
            distance = self.radius * (0.58 + _pseudo_random(index, self.cloud_seed + 0.72) * 0.18)
            altitude = max(5.6, self.radius * (0.18 + _pseudo_random(index, self.cloud_seed + 0.41) * 0.08))
            base = Vec3(sin(azimuth) * distance, altitude, cos(azimuth) * distance)
            right = Vec3(cos(azimuth), 0.0, -sin(azimuth)).normalized(Vec3(1.0, 0.0, 0.0))
            forward = Vec3(sin(azimuth), 0.0, cos(azimuth)).normalized(Vec3(0.0, 0.0, 1.0))
            puff_count = 3 + int(_pseudo_random(index, self.cloud_seed + 0.16) * 4.0)
            for puff in range(puff_count):
                side = (_pseudo_random(index * 17 + puff, self.cloud_seed + 0.84) - 0.5) * max(4.2, self.radius * 0.036)
                depth = (_pseudo_random(index * 17 + puff, self.cloud_seed + 0.53) - 0.5) * max(1.2, self.radius * 0.012)
                lift = (_pseudo_random(index * 17 + puff, self.cloud_seed + 0.62) - 0.46) * max(0.55, self.radius * 0.006)
                radius = max(0.58, self.radius * 0.011) + _pseudo_random(index * 17 + puff, self.cloud_seed + 0.73) * max(0.62, self.radius * 0.01)
                brightness = int(224 + _pseudo_random(index * 17 + puff, self.cloud_seed + 0.91) * 18)
                material = Material(color=(brightness, brightness, max(210, brightness - 8)), emission=(18, 22, 26), diffuse=0.48, roughness=0.92)
                clouds.append(
                    BlobSurface(
                        base + right * side + forward * depth + Vec3(0.0, lift, 0.0),
                        radius,
                        material,
                        stretch=right * (radius * 0.8) + forward * (radius * 0.18),
                        surface_tension=0.34,
                    )
                )
        return tuple(clouds)


def _pseudo_random(index: int, seed: float) -> float:
    value = sin((index + 1) * 12.9898 + seed * 78.233) * 43758.5453
    return value - floor(value)
