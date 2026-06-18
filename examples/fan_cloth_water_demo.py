"""Fan, cloth, and vector-fluid water demonstration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from math import cos, exp, pi, sin, sqrt, tau
from pathlib import Path
import subprocess

from fruit_bowl_demo import find_ffmpeg, ffmpeg_missing_message
from py_3d import (
    Bowl,
    Box,
    Camera,
    Color,
    GPUVectorFluidWorld,
    HUDRect,
    HUDText,
    Line3,
    LineSet3,
    Lamp,
    Material,
    Mesh,
    ParticleFluidPuddle,
    ParticleFluidVolume,
    ParticleWaterSurface,
    PixelBuffer,
    RenderEngine,
    RenderSettings,
    Scene,
    SkyPrefab,
    Sphere,
    Sun,
    TextBulletin,
    Triangle,
    Vec3,
    VectorFluidParticle,
    VectorFluidWorld,
)


OUTPUT_DIR = Path("USER") / "environments" / "fan_cloth_water" / "renderings"

_WATER_TEXTURE: PixelBuffer | None = None
_CLOTH_TEXTURE: PixelBuffer | None = None


@dataclass
class ClothNode:
    position: Vec3
    velocity: Vec3
    pinned: bool = False


@dataclass
class AirBubble:
    position: Vec3
    radius: float
    age: float = 0.0
    popping: bool = False
    pop_age: float = 0.0


@dataclass
class SplashDroplet:
    position: Vec3
    velocity: Vec3
    radius: float
    age: float = 0.0


class ClothSheet:
    def __init__(self, *, columns: int = 12, rows: int = 9) -> None:
        self.columns = columns
        self.rows = rows
        self.nodes: list[ClothNode] = []
        self.rest_x = 1.25 / (columns - 1)
        self.rest_y = 1.1 / (rows - 1)
        for row in range(rows):
            for column in range(columns):
                z = -0.62 + column * self.rest_x
                y = 1.58 - row * self.rest_y
                x = -0.62
                pinned = row == 0 and column in {0, columns - 1, columns // 2}
                self.nodes.append(ClothNode(Vec3(x, y, z), Vec3(0.0, 0.0, 0.0), pinned))
        self.links = self._build_links()

    def step(
        self,
        dt: float,
        time: float,
        substeps: int = 3,
        *,
        wind_scale: float = 1.0,
        obstacle_center: Vec3 | None = None,
        obstacle_radius: float = 0.0,
        floor_y: float = 0.0,
        floor_clearance: float = 0.018,
    ) -> None:
        contact_y = floor_y + max(0.0, floor_clearance)
        step_dt = dt / substeps
        for _ in range(substeps):
            forces = [[0.0, -1.15, 0.0] for _node in self.nodes]
            self._spring_forces(forces, stiffness=48.0)
            for index, node in enumerate(self.nodes):
                if node.pinned:
                    node.velocity = Vec3(0.0, 0.0, 0.0)
                    continue
                occlusion = self._wind_occlusion(node.position, obstacle_center, obstacle_radius)
                wind_x, wind_y, wind_z = self._wind_force_components(node.position, time)
                wind_multiplier = wind_scale * occlusion
                force = forces[index]
                acceleration = Vec3(force[0] + wind_x * wind_multiplier, force[1] + wind_y * wind_multiplier, force[2] + wind_z * wind_multiplier)
                node.velocity = (node.velocity + acceleration * step_dt) * 0.985
                node.position = node.position + node.velocity * step_dt
                if obstacle_center is not None and obstacle_radius > 0.0:
                    self._resolve_obstacle_collision(node, obstacle_center, obstacle_radius)
                if node.position.y < contact_y:
                    node.position = Vec3(node.position.x, contact_y, node.position.z)
                    node.velocity = Vec3(node.velocity.x, abs(node.velocity.y) * 0.08, node.velocity.z)

    def _build_links(self) -> tuple[tuple[int, int, float], ...]:
        links: list[tuple[int, int, float]] = []
        for row in range(self.rows):
            for column in range(self.columns):
                index = self._index(column, row)
                if column + 1 < self.columns:
                    links.append((index, self._index(column + 1, row), self.rest_x))
                if row + 1 < self.rows:
                    links.append((index, self._index(column, row + 1), self.rest_y))
                if column + 1 < self.columns and row + 1 < self.rows:
                    links.append((index, self._index(column + 1, row + 1), (self.rest_x * self.rest_x + self.rest_y * self.rest_y) ** 0.5))
                if column > 0 and row + 1 < self.rows:
                    links.append((index, self._index(column - 1, row + 1), (self.rest_x * self.rest_x + self.rest_y * self.rest_y) ** 0.5))
        return tuple(links)

    def _spring_forces(self, forces: list[list[float]], *, stiffness: float) -> None:
        for first_index, second_index, rest_length in self.links:
            first = self.nodes[first_index]
            second = self.nodes[second_index]
            dx = second.position.x - first.position.x
            dy = second.position.y - first.position.y
            dz = second.position.z - first.position.z
            distance = sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= 1e-6:
                continue
            amount = (distance - rest_length) * stiffness / distance
            fx = dx * amount
            fy = dy * amount
            fz = dz * amount
            first_force = forces[first_index]
            second_force = forces[second_index]
            first_force[0] += fx
            first_force[1] += fy
            first_force[2] += fz
            second_force[0] -= fx
            second_force[1] -= fy
            second_force[2] -= fz

    @staticmethod
    def _wind_force(position: Vec3, time: float) -> Vec3:
        wind_x, wind_y, wind_z = ClothSheet._wind_force_components(position, time)
        return Vec3(wind_x, wind_y, wind_z)

    @staticmethod
    def _wind_force_components(position: Vec3, time: float) -> tuple[float, float, float]:
        offset_x = position.x + 2.0
        offset_y = position.y - 0.98
        offset_z = position.z
        distance = max(0.12, sqrt(offset_x * offset_x + offset_y * offset_y + offset_z * offset_z))
        alignment = max(0.0, offset_x / distance)
        pulse = 0.82 + 0.18 * sin(time * tau * 1.7 + position.z * 4.2)
        turbulence_y = sin(time * 3.4 + position.z * 5.0) * 0.32
        turbulence_z = cos(time * 2.7 + position.y * 4.0) * 0.28
        scale = 5.2 * alignment * pulse / (1.0 + distance * distance * 0.9)
        return scale, turbulence_y * scale, turbulence_z * scale

    @staticmethod
    def _wind_occlusion(position: Vec3, obstacle_center: Vec3 | None, obstacle_radius: float) -> float:
        if obstacle_center is None or obstacle_radius <= 0.0 or position.x <= obstacle_center.x:
            return 1.0
        source_x, source_y, source_z = -2.0, 0.98, 0.0
        ray_x = position.x - source_x
        ray_y = position.y - source_y
        ray_z = position.z - source_z
        length_squared = ray_x * ray_x + ray_y * ray_y + ray_z * ray_z
        if length_squared <= 1e-8:
            return 1.0
        center_x = obstacle_center.x - source_x
        center_y = obstacle_center.y - source_y
        center_z = obstacle_center.z - source_z
        t = max(0.0, min(1.0, (center_x * ray_x + center_y * ray_y + center_z * ray_z) / length_squared))
        closest_x = source_x + ray_x * t
        closest_y = source_y + ray_y * t
        closest_z = source_z + ray_z * t
        dx = closest_x - obstacle_center.x
        dy = closest_y - obstacle_center.y
        dz = closest_z - obstacle_center.z
        if dx * dx + dy * dy + dz * dz < (obstacle_radius * 0.85) ** 2:
            return 0.28
        return 1.0

    @staticmethod
    def _resolve_obstacle_collision(node: ClothNode, center: Vec3, radius: float) -> None:
        if node.position.y > center.y + radius * 0.68:
            return
        local = node.position - center
        distance = local.length()
        margin = 0.035
        if distance <= 1e-8 or distance >= radius + margin:
            return
        normal = local / distance
        node.position = node.position + normal * (radius + margin - distance)
        inward_speed = node.velocity.dot(normal)
        if inward_speed < 0.0:
            node.velocity = (node.velocity - normal * (1.45 * inward_speed)) * 0.72

    def mesh(self) -> Mesh:
        material = Material(color=(255, 255, 255), texture=cloth_texture(), roughness=0.72, fuzziness=0.18, specular=0.04)
        triangles: list[Triangle] = []
        for row in range(self.rows - 1):
            for column in range(self.columns - 1):
                a = self.nodes[self._index(column, row)].position
                b = self.nodes[self._index(column + 1, row)].position
                c = self.nodes[self._index(column + 1, row + 1)].position
                d = self.nodes[self._index(column, row + 1)].position
                u0 = column / (self.columns - 1)
                u1 = (column + 1) / (self.columns - 1)
                v0 = row / (self.rows - 1)
                v1 = (row + 1) / (self.rows - 1)
                triangles.append(Triangle(a, b, c, material, (u0, v0), (u1, v0), (u1, v1)))
                triangles.append(Triangle(a, c, d, material, (u0, v0), (u1, v1), (u0, v1)))
        return Mesh(triangles)

    def wire(self) -> LineSet3:
        material = Material(color=(218, 228, 220), emission=(18, 20, 18))
        segments: list[tuple[Vec3, Vec3]] = []
        for row in range(self.rows):
            for column in range(self.columns):
                if column + 1 < self.columns:
                    segments.append((self.nodes[self._index(column, row)].position, self.nodes[self._index(column + 1, row)].position))
                if row + 1 < self.rows:
                    segments.append((self.nodes[self._index(column, row)].position, self.nodes[self._index(column, row + 1)].position))
        return LineSet3(segments, material)

    def _index(self, column: int, row: int) -> int:
        return row * self.columns + column


class FanWaterSimulation:
    def __init__(self, *, quality: str = "ultra", gpu_context=None) -> None:
        self.time = 0.0
        self.quality = quality
        self.wind_scale = 1.0
        self.blade_strength = 1.0
        cloth_size = {
            "fast": (10, 7),
            "balanced": (12, 8),
            "high": (14, 10),
            "ultra": (16, 11),
        }.get(quality, (12, 8))
        self.cloth = ClothSheet(columns=cloth_size[0], rows=cloth_size[1])
        self.water_center = Vec3(1.25, 0.64, 0.0)
        self.water_radius = 0.735
        fluid_rest_distance = 0.052 if gpu_context is not None else 0.12
        self.table_surface_y = 0.0
        self.floor_contact_margin = max(0.006, fluid_rest_distance * 0.18)
        self.floor_bounds_min = Vec3(-2.6, self.table_surface_y + self.floor_contact_margin, -1.5)
        self.floor_bounds_max = Vec3(2.6, 1.35, 1.5)
        self.bowl_radius = 0.76
        self.bowl_depth = 0.82
        self.bowl_thickness = 0.05
        self.bowl_table_clearance = 0.014
        self.bowl_center_y = self._bowl_center_y_for_table_contact()
        self.vessel_intact = True
        self.fan_center = self.water_center + Vec3(0.0, -0.075, 0.0)
        self.bubbles: list[AirBubble] = []
        self.splash_droplets: list[SplashDroplet] = []
        self._bubble_spawn_accumulator = 0.0
        self._bubble_serial = 0
        self.minimum_droplet_radius = 0.014
        self.splash_surface_tension = 0.72
        fluid_cls = GPUVectorFluidWorld if gpu_context is not None else VectorFluidWorld
        fluid_args = (gpu_context,) if gpu_context is not None else ()
        self.fluid = fluid_cls.liquid(
            *fluid_args,
            bounds_min=(self.water_center.x - self.water_radius, 0.42, self.water_center.z - self.water_radius),
            bounds_max=(self.water_center.x + self.water_radius, 0.78, self.water_center.z + self.water_radius),
            gravity=(0.0, -1.9, 0.0),
            rest_distance=fluid_rest_distance,
            repel_strength=8.2 if gpu_context is not None else 9.4,
            self_attraction=0.72 if gpu_context is not None else 0.95,
            viscosity=0.48 if gpu_context is not None else 0.38,
            boundary_bounce=0.05,
            close_damping=10.5,
            attract_range_factor=1.72 if gpu_context is not None else 2.05,
        )
        if getattr(self.fluid, "gpu_enabled", False):
            self.fluid.sync_python_particles = False
            self.fluid.readback_particles = False
        self._seed_water_particles(gpu_context is not None)

    def _configure_fluid_bounds(self) -> None:
        if self.vessel_intact:
            self.fluid.bounds_min = Vec3(self.water_center.x - self.water_radius, 0.42, self.water_center.z - self.water_radius)
            self.fluid.bounds_max = Vec3(self.water_center.x + self.water_radius, 0.82, self.water_center.z + self.water_radius)
        else:
            self.fluid.bounds_min = self.floor_bounds_min
            self.fluid.bounds_max = self.floor_bounds_max

    def _seed_water_particles(self, gpu_enabled: bool) -> None:
        count = (
            {"fast": 1024, "balanced": 1536, "high": 3072, "ultra": 4096}.get(self.quality, 1536)
            if gpu_enabled
            else {"fast": 96, "balanced": 160, "high": 240, "ultra": 320}.get(self.quality, 160)
        )
        golden_angle = pi * (3.0 - 5.0 ** 0.5)
        for index in range(count):
            amount = (index + 0.5) / count
            radius = self.water_radius * 0.96 * (amount ** 0.5)
            angle = index * golden_angle
            ripple = sin(angle * 1.7 + radius * 8.0)
            y = -0.055 + 0.014 * ripple
            position = self.water_center + Vec3(cos(angle) * radius, y, sin(angle) * radius)
            tangent = Vec3(-sin(angle), 0.0, cos(angle))
            self.fluid.add_particle(VectorFluidParticle(position, velocity=tangent * 0.035))

    def step(self, dt: float) -> None:
        self.time += dt
        self._configure_fluid_bounds()
        self.fluid.self_attraction = 0.72 if self.vessel_intact else 0.34
        self.fluid.viscosity = 0.48 if self.vessel_intact else 0.54
        cloth_substeps = 4 if self.wind_scale > 1.6 else 3
        self.cloth.step(
            dt,
            self.time,
            substeps=cloth_substeps,
            wind_scale=self.wind_scale,
            obstacle_center=Vec3(self.water_center.x, self.bowl_center_y, self.water_center.z),
            obstacle_radius=0.78,
            floor_y=self.table_surface_y,
            floor_clearance=0.018,
        )
        if getattr(self.fluid, "gpu_enabled", False):
            self.fluid.force_mode = 1
            self.fluid.force_center = self.fan_center
            self.fluid.cylinder_radius = self.water_radius if self.vessel_intact else 0.0
            self.fluid.time = self.time
            self.fluid.wind_strength = self.wind_scale
            self.fluid.blade_strength = self.blade_strength
            self.fluid.step(dt, substeps=1)
        else:
            self.fluid.step(dt, substeps=2, external_force=self._fan_blade_force)
            if self.vessel_intact:
                self._constrain_water()
        if self.vessel_intact:
            self._step_air_bubbles(dt)
        else:
            self.bubbles.clear()
        self._step_splash_droplets(dt)

    def _fan_blade_force(self, particle: VectorFluidParticle, dt: float) -> tuple[float, float, float]:
        del dt
        px = particle.position.x
        py = particle.position.y
        pz = particle.position.z
        local_x = px - self.fan_center.x
        local_y = py - self.fan_center.y
        local_z = pz - self.fan_center.z
        horizontal_distance = sqrt(local_x * local_x + local_z * local_z)
        distance = max(0.04, horizontal_distance)
        if horizontal_distance > 1e-8:
            inv_horizontal = 1.0 / horizontal_distance
            tangent_x = -local_z * inv_horizontal
            tangent_z = local_x * inv_horizontal
            radial_x = local_x * inv_horizontal
            radial_z = local_z * inv_horizontal
        else:
            tangent_x = 0.0
            tangent_z = 1.0
            radial_x = 1.0
            radial_z = 0.0
        blade_phase = sin(self.time * tau * 3.1 + distance * 10.0)
        fan_plane = exp(-(local_y * local_y) / 0.030)
        surface = max(0.0, min(1.0, (py - (self.fan_center.y + 0.02)) / 0.30))
        core = exp(-(distance * distance) / 0.055)
        outer_in = max(0.0, min(1.0, (distance - 0.14) / 0.32))
        outer_out = 1.0 - max(0.0, min(1.0, (distance - 0.58) / 0.37))
        outer = outer_in * outer_out
        wind_wave = sin(self.time * tau * 1.7 + pz * 8.0)
        wind_scale = 0.38 * self.wind_scale * surface
        wind_x = wind_scale
        wind_y = 0.035 * wind_wave * wind_scale
        wind_z = 0.14 * sin(self.time * 3.1 + px * 4.5) * wind_scale
        swirl = (1.52 * fan_plane + 0.42 * surface) / (0.48 + distance * 2.35)
        radial_pull = -0.42 * core * fan_plane + 0.10 * outer * surface
        vertical = -0.76 * core * fan_plane + 0.24 * outer * surface + 0.10 * blade_phase * fan_plane + 0.16 * core * surface
        blade = self.blade_strength
        return (
            wind_x + (tangent_x * swirl + radial_x * radial_pull) * blade,
            wind_y + vertical * blade,
            wind_z + (tangent_z * swirl + radial_z * radial_pull) * blade,
        )

    def _step_air_bubbles(self, dt: float) -> None:
        surface_y = self.water_center.y
        if self.vessel_intact and self.blade_strength > 0.05:
            fan_depth = max(0.0, surface_y - self.fan_center.y)
            vortex_entrainment = max(0.0, min(1.0, (0.16 - abs(fan_depth - 0.075)) / 0.16))
            spawn_rate = vortex_entrainment * (0.55 + self.blade_strength * 1.05 + max(0.0, self.wind_scale - 0.8) * 0.24)
            self._bubble_spawn_accumulator += dt * spawn_rate
            while self._bubble_spawn_accumulator >= 1.0 and len(self.bubbles) < 10:
                self._bubble_spawn_accumulator -= 1.0
                phase = self.time * 8.7 + self._bubble_serial * 2.399
                radial = 0.025 + 0.055 * (0.5 + 0.5 * sin(phase * 1.73))
                radius = 0.010 + 0.018 * (0.5 + 0.5 * sin(phase * 2.17 + 1.1))
                position = self.fan_center + Vec3(cos(phase) * radial, 0.012, sin(phase) * radial)
                self.bubbles.append(AirBubble(position, radius))
                self._bubble_serial += 1

        kept: list[AirBubble] = []
        for bubble in self.bubbles:
            bubble.age += dt
            if bubble.popping:
                bubble.pop_age += dt
                if bubble.pop_age <= 0.46:
                    kept.append(bubble)
                continue

            local = bubble.position - self.fan_center
            horizontal = Vec3(local.x, 0.0, local.z)
            tangent = Vec3(-horizontal.z, 0.0, horizontal.x).normalized(Vec3(0.0, 0.0, 1.0))
            distance = max(0.04, horizontal.length())
            rise = 0.14 + bubble.radius * 2.6
            bubble.position = bubble.position + tangent * (dt * 0.20 / (0.45 + distance)) + Vec3(0.0, rise * dt, 0.0)
            if bubble.position.y >= surface_y - bubble.radius * 0.35:
                bubble.position = Vec3(bubble.position.x, surface_y, bubble.position.z)
                bubble.popping = True
                bubble.pop_age = 0.001
                self._emit_splash_from_bubble(bubble)
            kept.append(bubble)
        self.bubbles = kept[-10:]

    def _emit_splash_from_bubble(self, bubble: AirBubble) -> None:
        if bubble.radius < self.minimum_droplet_radius * 0.78:
            return
        droplet_count = max(2, min(7, int(bubble.radius / max(0.001, self.minimum_droplet_radius) * (1.25 - self.splash_surface_tension * 0.35))))
        for index in range(droplet_count):
            angle = self.time * 5.1 + self._bubble_serial * 1.37 + index * tau / droplet_count
            radius = max(self.minimum_droplet_radius, bubble.radius * (0.32 + 0.08 * sin(angle * 1.7)))
            if radius < self.minimum_droplet_radius:
                continue
            outward = Vec3(cos(angle), 0.0, sin(angle))
            velocity = outward * (0.09 + bubble.radius * 1.8) + Vec3(0.0, 0.23 + bubble.radius * 2.2, 0.0)
            self.splash_droplets.append(SplashDroplet(bubble.position + outward * (bubble.radius * 0.55), velocity, radius))
        self.splash_droplets = self.splash_droplets[-24:]

    def _step_splash_droplets(self, dt: float) -> None:
        kept: list[SplashDroplet] = []
        surface_y = self.water_center.y
        for droplet in self.splash_droplets:
            droplet.age += dt
            droplet.velocity = (droplet.velocity + Vec3(0.0, -1.9, 0.0) * dt) * max(0.0, 1.0 - 0.34 * dt)
            droplet.position = droplet.position + droplet.velocity * dt
            if droplet.age > 0.9:
                continue
            if droplet.position.y <= surface_y + droplet.radius * 0.15 and droplet.velocity.y < 0.0:
                continue
            kept.append(droplet)
        self.splash_droplets = kept

    def bubble_surface_events(self) -> tuple[tuple[float, float, float, float], ...]:
        surface_y = self.water_center.y
        events: list[tuple[float, float, float, float]] = []
        for bubble in self.bubbles:
            if bubble.popping:
                events.append((bubble.position.x, bubble.position.z, bubble.radius * 1.15, min(1.0, bubble.pop_age / 0.46)))
                continue
            depth = surface_y - bubble.position.y
            if depth <= bubble.radius * 3.4:
                events.append((bubble.position.x, bubble.position.z, bubble.radius * 1.05, 0.0))
        events.sort(key=lambda item: item[3], reverse=True)
        return tuple(events[:8])

    def bubble_primitives(self) -> tuple[Sphere, ...]:
        return ()

    def splash_primitives(self) -> tuple[Sphere, ...]:
        material = Material(color=(154, 224, 246), roughness=0.006, specular=1.0, shininess=180.0, reflectivity=0.62, light_transmission=0.76)
        return tuple(Sphere(droplet.position, droplet.radius, material) for droplet in self.splash_droplets)

    def spilled_water_primitives(self) -> tuple[Mesh, ...]:
        material = Material(color=(126, 216, 240), roughness=0.006, specular=1.0, shininess=180.0, reflectivity=0.62, light_transmission=0.74)
        columns, rows = {"fast": (30, 18), "balanced": (44, 26), "high": (60, 36), "ultra": (76, 46)}.get(self.quality, (44, 26))
        kernel = max(self.fluid.rest_distance * 2.8, 0.16)
        floor_y = self.table_surface_y
        points: dict[tuple[int, int], tuple[Vec3, float]] = {}
        for row in range(rows + 1):
            v = row / rows
            z = self.floor_bounds_min.z + (self.floor_bounds_max.z - self.floor_bounds_min.z) * v
            for column in range(columns + 1):
                u = column / columns
                x = self.floor_bounds_min.x + (self.floor_bounds_max.x - self.floor_bounds_min.x) * u
                density = 0.0
                weighted_y = 0.0
                for particle in self.fluid.particles:
                    height_from_floor = particle.position.y - floor_y
                    if height_from_floor > 0.36:
                        continue
                    dx = particle.position.x - x
                    dz = particle.position.z - z
                    if abs(dx) > kernel or abs(dz) > kernel:
                        continue
                    distance2 = dx * dx + dz * dz
                    if distance2 > kernel * kernel:
                        continue
                    floor_proximity = max(0.0, min(1.0, 1.0 - (height_from_floor - 0.045) / 0.315))
                    weight = max(0.0, 1.0 - distance2 / (kernel * kernel)) ** 2 * floor_proximity
                    density += weight
                    weighted_y += particle.position.y * weight
                if density <= 0.18:
                    points[(column, row)] = (Vec3(x, floor_y + 0.004, z), 0.0)
                    continue
                mean_y = weighted_y / density
                height = floor_y + min(0.07, density * 0.010) + max(0.0, min(0.055, (mean_y - floor_y) * 0.12))
                points[(column, row)] = (Vec3(x, height, z), density)

        triangles: list[Triangle] = []
        for row in range(rows):
            for column in range(columns):
                a, da = points[(column, row)]
                b, db = points[(column + 1, row)]
                c, dc = points[(column + 1, row + 1)]
                d, dd = points[(column, row + 1)]
                if max(da, db, dc, dd) <= 0.18:
                    continue
                u0 = column / columns
                u1 = (column + 1) / columns
                v0 = row / rows
                v1 = (row + 1) / rows
                triangles.append(Triangle(a, b, c, material, (u0, v0), (u1, v0), (u1, v1)))
                triangles.append(Triangle(a, c, d, material, (u0, v0), (u1, v1), (u0, v1)))
        return (Mesh(triangles),) if triangles else ()

    def _constrain_water(self) -> None:
        for particle in self.fluid.particles:
            local = particle.position - self.water_center
            horizontal = Vec3(local.x, 0.0, local.z)
            distance = horizontal.length()
            if distance > self.water_radius:
                normal = horizontal / distance
                particle.position = self.water_center + normal * self.water_radius + Vec3(0.0, local.y, 0.0)
                outward = particle.velocity.dot(normal)
                if outward > 0.0:
                    particle.velocity = particle.velocity - normal * (outward * 1.2)
            if particle.position.y < self.water_center.y - 0.18 and particle.velocity.y < 0.0:
                particle.velocity = Vec3(particle.velocity.x * 0.98, particle.velocity.y * 0.25, particle.velocity.z * 0.98)

    def _bowl_center_y_for_table_contact(self) -> float:
        end_phi = pi / 2.0 + self.bowl_depth * pi / 2.0
        bottom_offset = cos(end_phi) * (self.bowl_radius + self.bowl_thickness)
        return self.table_surface_y + self.bowl_table_clearance - bottom_offset

    def scene(self) -> Scene:
        scene = Scene()
        scene.add(Box((0.0, -0.04, 0.0), (5.2, 0.08, 3.0), Material(color=(48, 58, 60), roughness=0.62, fuzziness=0.08)))
        scene.add(Box((-0.62, 1.64, 0.0), (0.08, 0.08, 1.46), Material(color=(92, 76, 52), roughness=0.5)))
        scene.add(self.cloth.mesh(), self.cloth.wire())
        scene.add(*fan_primitives(Vec3(-2.05, 0.98, 0.0), self.time, facing="x"))
        if self.vessel_intact:
            scene.add(Bowl((self.water_center.x, self.bowl_center_y, self.water_center.z), self.bowl_radius, Material(color=(122, 82, 48), roughness=0.62, fuzziness=0.14, specular=0.05), depth=self.bowl_depth, thickness=self.bowl_thickness))
        if getattr(self.fluid, "gpu_enabled", False) and self.vessel_intact:
            scene.add(
                ParticleWaterSurface(
                    self.fluid,
                    self.water_center,
                    self.water_radius,
                    quality=self.quality,
                    time=self.time,
                    particle_base_y=self.water_center.y - 0.055,
                    vortex_center=self.fan_center,
                    vortex_strength=(0.85 if self.vessel_intact else 0.34) * self.blade_strength,
                    bubbles=self.bubble_surface_events(),
                )
            )
        elif getattr(self.fluid, "gpu_enabled", False):
            scene.add(
                ParticleFluidPuddle(
                    self.fluid,
                    self.floor_bounds_min,
                    self.floor_bounds_max,
                    floor_y=self.table_surface_y,
                    quality=self.quality,
                    time=self.time,
                    kernel_radius=max(self.fluid.rest_distance * 3.1, 0.12),
                    density_threshold=0.18,
                )
            )
            scene.add(ParticleFluidVolume(self.fluid, particle_radius=self.fluid.rest_distance * 0.42, min_y=self.table_surface_y + 0.18))
        elif self.vessel_intact:
            scene.add(water_surface_mesh(self.fluid, self.water_center, self.water_radius, self.time, quality=self.quality))
        else:
            scene.add(*self.spilled_water_primitives())
        scene.add(*self.bubble_primitives())
        scene.add(*self.splash_primitives())
        scene.add(*water_fan_primitives(self.fan_center, self.time))
        scene.add_light(Sun(direction=(-0.35, -0.8, -0.55), color=(255, 245, 228), intensity=0.78))
        scene.add_light(Lamp(position=(-1.6, 1.9, -1.2), color=(135, 180, 255), intensity=3.8))
        scene.add_light(Lamp(position=(1.8, 1.55, -0.8), color=(255, 210, 150), intensity=2.1))
        scene.add_bulletin(TextBulletin("FAN CLOTH WATER\nVECTOR CLOUD FLUID", position=(10, 10), color=(245, 248, 255), background=(4, 7, 10), padding=5))
        return scene


def cloth_texture(width: int = 256, height: int = 256) -> PixelBuffer:
    global _CLOTH_TEXTURE
    if _CLOTH_TEXTURE is not None:
        return _CLOTH_TEXTURE
    buffer = PixelBuffer.new(width, height, (168, 184, 172))
    for y in range(height):
        for x in range(width):
            weave = 0.5 + 0.5 * sin(x * 0.42) * sin(y * 0.38)
            stripe = 1.0 if x % 38 < 3 or y % 42 < 3 else 0.0
            buffer.pixels[y * width + x] = Color(126 + weave * 42 + stripe * 28, 146 + weave * 34 + stripe * 24, 134 + weave * 28 + stripe * 18)
    _CLOTH_TEXTURE = buffer
    return buffer


def water_texture(width: int = 256, height: int = 256) -> PixelBuffer:
    global _WATER_TEXTURE
    if _WATER_TEXTURE is not None:
        return _WATER_TEXTURE
    buffer = PixelBuffer.new(width, height, (64, 128, 164))
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            wave = 0.5 + 0.5 * sin((u * 8.0 + 0.18 * sin(v * tau * 3.0)) * tau)
            ripple = 0.5 + 0.5 * sin((u * 22.0 + v * 17.0) * tau)
            caustic = max(0.0, wave * ripple - 0.55) * 1.6
            buffer.pixels[y * width + x] = Color(34 + caustic * 80, 108 + wave * 32 + caustic * 82, 146 + ripple * 34 + caustic * 70)
    _WATER_TEXTURE = buffer
    return buffer


_WATER_SURFACE_LAYOUT_CACHE: dict[tuple[str, float], tuple[int, int, tuple[tuple[float, float, float], ...], tuple[tuple[int, int], ...]]] = {}
_WATER_SURFACE_ARRAY_CACHE: dict[int, tuple[object, object, object]] = {}


def water_surface_mesh(world: VectorFluidWorld, center: Vec3, radius: float, time: float, *, quality: str) -> Mesh:
    rings, segments, samples, sample_keys = _water_surface_layout(quality, radius)
    material = Material(color=(126, 216, 240), roughness=0.006, specular=1.0, shininess=192.0, reflectivity=0.72, light_transmission=0.72)
    sampler = _ParticleSurfaceSampler(world, center, radius, time)
    heights = sampler.heights_for(samples)
    center_point = Vec3(center.x, heights[0], center.z)
    points: dict[tuple[int, int], Vec3] = {}
    for (ring, segment), (x, z, _edge), height in zip(sample_keys[1:], samples[1:], heights[1:]):
        points[(ring, segment)] = Vec3(center.x + x, height, center.z + z)
    normals = _water_surface_normals(points, center_point, rings, segments)
    triangles: list[Triangle] = []
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        u0 = segment / segments
        u1 = (segment + 1) / segments
        triangles.append(
            Triangle(
                center_point,
                points[(1, next_segment)],
                points[(1, segment)],
                material,
                (0.5, 0.5),
                (u1, 0.04),
                (u0, 0.04),
                Vec3(0.0, 1.0, 0.0),
                normals[(1, next_segment)],
                normals[(1, segment)],
            )
        )
    for ring in range(1, rings):
        v0 = ring / rings
        v1 = (ring + 1) / rings
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = points[(ring, segment)]
            b = points[(ring, next_segment)]
            c = points[(ring + 1, next_segment)]
            d = points[(ring + 1, segment)]
            u0 = segment / segments
            u1 = (segment + 1) / segments
            triangles.append(Triangle(a, b, c, material, (u0, v0), (u1, v0), (u1, v1), normals[(ring, segment)], normals[(ring, next_segment)], normals[(ring + 1, next_segment)]))
            triangles.append(Triangle(a, c, d, material, (u0, v0), (u1, v1), (u0, v1), normals[(ring, segment)], normals[(ring + 1, next_segment)], normals[(ring + 1, segment)]))
    return Mesh(triangles)


def _water_surface_layout(quality: str, radius: float) -> tuple[int, int, tuple[tuple[float, float, float], ...], tuple[tuple[int, int], ...]]:
    key = (quality, round(radius, 5))
    cached = _WATER_SURFACE_LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached
    rings, segments = {"fast": (12, 44), "balanced": (17, 60), "high": (24, 84), "ultra": (32, 112)}.get(quality, (17, 60))
    samples: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    sample_keys: list[tuple[int, int]] = [(0, 0)]
    for ring in range(1, rings + 1):
        amount = ring / rings
        radial = radius * (amount ** 0.72)
        edge = max(0.0, (amount - 0.88) / 0.12)
        for segment in range(segments):
            angle = tau * segment / segments
            samples.append((cos(angle) * radial, sin(angle) * radial, edge))
            sample_keys.append((ring, segment))
    cached = (rings, segments, tuple(samples), tuple(sample_keys))
    _WATER_SURFACE_LAYOUT_CACHE[key] = cached
    return cached


def _water_surface_normals(points: dict[tuple[int, int], Vec3], center_point: Vec3, rings: int, segments: int) -> dict[tuple[int, int], Vec3]:
    normals: dict[tuple[int, int], Vec3] = {}
    for ring in range(1, rings + 1):
        for segment in range(segments):
            current = points[(ring, segment)]
            inner = center_point if ring == 1 else points[(ring - 1, segment)]
            if ring == rings:
                radial = current - inner
            else:
                radial = points[(ring + 1, segment)] - inner
            angular = points[(ring, (segment + 1) % segments)] - points[(ring, (segment - 1) % segments)]
            normal = angular.cross(radial).normalized(Vec3(0.0, 1.0, 0.0))
            if normal.y < 0.0:
                normal = -normal
            normals[(ring, segment)] = normal
    return normals


class _ParticleSurfaceSampler:
    def __init__(self, world: VectorFluidWorld, center: Vec3, radius: float, time: float) -> None:
        self.center = center
        self.radius = radius
        self.time = time
        self.kernel_radius = max(world.rest_distance * 3.4, radius * 0.095)
        try:
            import numpy

            self.numpy = numpy
            arrays = None
            array_getter = getattr(world, "particle_position_velocity_arrays", None)
            if callable(array_getter):
                arrays = array_getter()
            if arrays is not None:
                self.positions, self.velocities = arrays
            else:
                self.positions = numpy.array([(p.position.x, p.position.y, p.position.z) for p in world.particles], dtype=numpy.float32)
                self.velocities = numpy.array([(p.velocity.x, p.velocity.y, p.velocity.z) for p in world.particles], dtype=numpy.float32)
            self.mean_y = float(self.positions[:, 1].mean()) if len(self.positions) else self.center.y
        except Exception:
            self.numpy = None
            self.positions = tuple(p.position for p in world.particles)
            self.velocities = tuple(p.velocity for p in world.particles)
            self.mean_y = sum((position.y for position in self.positions), self.center.y * 0.0) / len(self.positions) if self.positions else self.center.y

    def heights_for(self, samples: tuple[tuple[float, float, float], ...]) -> list[float]:
        if self.numpy is None or len(self.positions) == 0:
            return [self._finish_height(self.height_at(x, z), x, z, edge) for x, z, edge in samples]
        numpy = self.numpy
        coords, edges = _water_sample_arrays(numpy, samples)
        heights = numpy.full((len(samples),), self.center.y, dtype=numpy.float32)
        displacement = (self.positions[:, 1] - self.mean_y) * 0.36 + self.velocities[:, 1] * 0.026
        if len(self.positions) >= 768:
            heights += self._grid_surface_offsets(coords, displacement)
        else:
            heights += self._kernel_surface_offsets(coords, displacement)
        radial = numpy.sqrt(coords[:, 0] * coords[:, 0] + coords[:, 1] * coords[:, 1]) / max(0.001, self.radius)
        wind_ripple = numpy.sin(self.time * tau * 1.45 + coords[:, 0] * 6.4 + coords[:, 1] * 2.7) * 0.012
        cross_ripple = numpy.sin(self.time * tau * 2.0 + coords[:, 0] * 9.0) * numpy.maximum(0.0, 1.0 - numpy.abs(coords[:, 1]) / max(0.001, self.radius)) * 0.007
        heights += (wind_ripple + cross_ripple) * numpy.maximum(0.0, 1.0 - radial * 0.32)
        edge_flatten = edges * edges
        heights = self.center.y + (heights - self.center.y) * (1.0 - edge_flatten * 0.86) + edge_flatten * 0.003
        return [float(value) for value in heights]

    def _kernel_surface_offsets(self, coords, displacement):
        numpy = self.numpy
        offsets = numpy.zeros((len(coords),), dtype=numpy.float32)
        particle_x = self.positions[:, 0]
        particle_z = self.positions[:, 2]
        kernel2 = self.kernel_radius * self.kernel_radius
        chunk_size = 224
        for start in range(0, len(coords), chunk_size):
            stop = min(len(coords), start + chunk_size)
            sample_x = self.center.x + coords[start:stop, 0]
            sample_z = self.center.z + coords[start:stop, 1]
            dx = particle_x[None, :] - sample_x[:, None]
            dz = particle_z[None, :] - sample_z[:, None]
            weight = numpy.maximum(0.0, 1.0 - (dx * dx + dz * dz) / kernel2)
            weight = weight * weight
            total = weight.sum(axis=1)
            valid = total > 1e-6
            if bool(valid.any()):
                chunk_offsets = offsets[start:stop]
                chunk_offsets[valid] = ((weight[valid] * displacement[None, :]).sum(axis=1) / total[valid]) * 0.34
                offsets[start:stop] = chunk_offsets
        return offsets

    def _grid_surface_offsets(self, coords, displacement):
        numpy = self.numpy
        rel_x = self.positions[:, 0] - self.center.x
        rel_z = self.positions[:, 2] - self.center.z
        radius = max(0.001, self.radius)
        mask = (numpy.abs(rel_x) <= radius) & (numpy.abs(rel_z) <= radius)
        if not bool(mask.any()):
            return numpy.zeros((len(coords),), dtype=numpy.float32)
        grid_size = max(40, min(96, int((len(coords) ** 0.5) * 2.35)))
        grid_range = ((-radius, radius), (-radius, radius))
        weighted, _x_edges, _z_edges = numpy.histogram2d(rel_x[mask], rel_z[mask], bins=grid_size, range=grid_range, weights=displacement[mask])
        counts, _x_edges, _z_edges = numpy.histogram2d(rel_x[mask], rel_z[mask], bins=grid_size, range=grid_range)
        for _ in range(3):
            weighted = _box_blur_grid(numpy, weighted)
            counts = _box_blur_grid(numpy, counts)
        field = numpy.divide(weighted, counts, out=numpy.zeros_like(weighted), where=counts > 1e-6)
        u = numpy.clip((coords[:, 0] + radius) / (radius * 2.0) * (grid_size - 1), 0.0, grid_size - 1.001)
        v = numpy.clip((coords[:, 1] + radius) / (radius * 2.0) * (grid_size - 1), 0.0, grid_size - 1.001)
        x0 = numpy.floor(u).astype(numpy.int32)
        z0 = numpy.floor(v).astype(numpy.int32)
        x1 = numpy.minimum(x0 + 1, grid_size - 1)
        z1 = numpy.minimum(z0 + 1, grid_size - 1)
        tx = u - x0
        tz = v - z0
        a = field[x0, z0] * (1.0 - tx) + field[x1, z0] * tx
        b = field[x0, z1] * (1.0 - tx) + field[x1, z1] * tx
        return (a * (1.0 - tz) + b * tz).astype(numpy.float32) * 0.42

    def height_at(self, x: float, z: float) -> float:
        base = self.center.y
        if self.numpy is not None and len(self.positions) > 0:
            dx = self.positions[:, 0] - (self.center.x + x)
            dz = self.positions[:, 2] - (self.center.z + z)
            distance2 = dx * dx + dz * dz
            weight = self.numpy.maximum(0.0, 1.0 - distance2 / (self.kernel_radius * self.kernel_radius))
            weight = weight * weight
            total = float(weight.sum())
            if total > 1e-6:
                displacement = (self.positions[:, 1] - self.mean_y) * 0.36 + self.velocities[:, 1] * 0.026
                base += float((weight * displacement).sum() / total) * 0.34
        else:
            total = 0.0
            displacement = 0.0
            for index, position in enumerate(self.positions):
                dx = self.center.x + x - position.x
                dz = self.center.z + z - position.z
                weight = max(0.0, 1.0 - (dx * dx + dz * dz) / (self.kernel_radius * self.kernel_radius)) ** 2
                if weight <= 0.0:
                    continue
                total += weight
                displacement += weight * ((position.y - self.mean_y) * 0.36 + self.velocities[index].y * 0.026)
            if total > 1e-6:
                base += displacement / total * 0.34
        radial = (x * x + z * z) ** 0.5 / max(0.001, self.radius)
        wind_ripple = sin(self.time * tau * 1.45 + x * 6.4 + z * 2.7) * 0.012
        cross_ripple = sin(self.time * tau * 2.0 + x * 9.0) * max(0.0, 1.0 - abs(z) / max(0.001, self.radius)) * 0.007
        return base + (wind_ripple + cross_ripple) * max(0.0, 1.0 - radial * 0.32)

    def _finish_height(self, height: float, x: float, z: float, edge: float) -> float:
        del x, z
        edge_flatten = edge * edge
        return self.center.y + (height - self.center.y) * (1.0 - edge_flatten * 0.86) + edge_flatten * 0.003


def _water_sample_arrays(numpy, samples: tuple[tuple[float, float, float], ...]):
    key = id(samples)
    cached = _WATER_SURFACE_ARRAY_CACHE.get(key)
    if cached is not None and cached[0] is numpy:
        return cached[1], cached[2]
    coords = numpy.array([(x, z) for x, z, _edge in samples], dtype=numpy.float32)
    edges = numpy.array([edge for _x, _z, edge in samples], dtype=numpy.float32)
    _WATER_SURFACE_ARRAY_CACHE[key] = (numpy, coords, edges)
    return coords, edges


def _box_blur_grid(numpy, grid):
    padded = numpy.pad(grid, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[1:-1, :-2]
        + padded[2:, :-2]
        + padded[:-2, 1:-1]
        + padded[1:-1, 1:-1]
        + padded[2:, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, 2:]
        + padded[2:, 2:]
    ) / 9.0


def fan_primitives(center: Vec3, time: float, *, facing: str) -> tuple[Sphere | Line3 | Mesh, ...]:
    metal = Material(color=(92, 96, 102), roughness=0.24, specular=0.42, shininess=42.0)
    glow = Material(color=(170, 205, 255), emission=(28, 42, 58), roughness=0.18, specular=0.3, shininess=36.0)
    stand = (
        Line3(center + Vec3(0.0, -0.52, 0.0), center, metal),
        Sphere(center + Vec3(0.0, -0.56, 0.0), 0.14, metal),
        Sphere(center, 0.1, glow),
    )
    return (*stand, blade_mesh(center, time * tau * 2.8, 0.42, facing=facing, material=metal))


def water_fan_primitives(center: Vec3, time: float) -> tuple[Sphere | Mesh, ...]:
    metal = Material(color=(112, 118, 122), roughness=0.18, specular=0.52, shininess=56.0)
    return (Sphere(center, 0.055, metal), blade_mesh(center, time * tau * 2.4, 0.28, facing="y", material=metal))


def blade_mesh(center: Vec3, angle: float, radius: float, *, facing: str, material: Material) -> Mesh:
    triangles: list[Triangle] = []
    for index in range(4):
        theta = angle + index * tau / 4.0
        if facing == "x":
            radial = Vec3(0.0, cos(theta), sin(theta))
            tangent = Vec3(0.0, -sin(theta), cos(theta))
        else:
            radial = Vec3(cos(theta), 0.0, sin(theta))
            tangent = Vec3(-sin(theta), 0.0, cos(theta))
        root = center + radial * (radius * 0.22)
        tip = center + radial * radius
        width = radius * 0.08
        a = root - tangent * width
        b = root + tangent * width
        c = tip + tangent * (width * 1.8)
        d = tip - tangent * (width * 1.8)
        triangles.append(Triangle(a, b, c, material))
        triangles.append(Triangle(a, c, d, material))
    return Mesh(triangles)


def make_camera() -> Camera:
    return Camera(position=(0.05, 1.65, -4.2), target=(0.08, 0.82, 0.0), fov_degrees=52)


def make_settings(args: argparse.Namespace) -> RenderSettings:
    return RenderSettings(
        width=args.width,
        height=args.height,
        background=Color(7, 10, 13),
        ambient=args.ambient,
        gamma=args.gamma,
        light_wrap=getattr(args, "light_wrap", 0.08),
        bounce_light=getattr(args, "bounce_light", 0.16),
        smooth_shading=bool(getattr(args, "smooth_shading", True)),
        tone_mapping=True,
        ray_traced_shadows=bool(getattr(args, "ray_traced_shadows", True)),
        reflection_bounces=args.reflection_bounces,
        shadow_samples=getattr(args, "shadow_samples", 3),
        shadow_softness=getattr(args, "shadow_softness", 0.22),
        sphere_segments=args.sphere_segments,
        sphere_rings=args.sphere_rings,
        texture_size=args.texture_size,
    )


class GLFanClothWaterViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        from py_3d.live import LiveFlyCamera, LiveMenu, LiveMenuOption, ModernGLLiveRenderer

        self.args = args
        self.settings = make_settings(args)
        self.sky = SkyPrefab(time_of_day=14.0, cycle_enabled=False, stars_enabled=True, clouds_enabled=True)
        base_camera = make_camera()
        self.camera_controller = LiveFlyCamera.looking_at(
            base_camera.position,
            base_camera.target,
            fov_degrees=base_camera.fov_degrees,
            speed=2.4,
        )
        self.keys: set[str] = set()
        self.paused = False
        self.quality_order = ("fast", "balanced", "high", "ultra")
        self.renderer = ModernGLLiveRenderer(
            args.window_width,
            args.window_height,
            title="py_3d fan cloth water - live",
            vsync=getattr(args, "vsync", True),
        )
        render_context = getattr(self.renderer, "ctx", None)
        fluid_context = render_context if getattr(render_context, "version_code", 0) >= 430 else None
        self.simulation = FanWaterSimulation(quality=args.quality, gpu_context=fluid_context)
        self.renderer.menu = LiveMenu(
            "py_3d fan cloth water",
            (
                LiveMenuOption("done", "Done"),
                LiveMenuOption("quality_next", "Quality preset"),
                LiveMenuOption("smooth_shading", "Surface smoothing"),
                LiveMenuOption("wind_up", "More cloth wind"),
                LiveMenuOption("wind_down", "Less cloth wind"),
                LiveMenuOption("blade_up", "More water swirl"),
                LiveMenuOption("blade_down", "Less water swirl"),
                LiveMenuOption("reflections_up", "More reflections"),
                LiveMenuOption("reflections_down", "Fewer reflections"),
                LiveMenuOption("sky_cycle", "Day/night cycle"),
                LiveMenuOption("sky_time_up", "Time later"),
                LiveMenuOption("sky_time_down", "Time earlier"),
                LiveMenuOption("sky_clouds", "Clouds"),
                LiveMenuOption("sky_stars", "Stars"),
                LiveMenuOption("break_vessel", "Break/repair bowl"),
                LiveMenuOption("pause", "Pause/run physics"),
                LiveMenuOption("reset", "Reset simulation"),
                LiveMenuOption("quit", "Quit demo"),
            ),
            background_blur=getattr(args, "menu_blur", False),
        )
        self._refresh_menu_options()
        self.renderer.set_mouse_captured(True)
        self._last_title_update = 0
        self._cached_scene: Scene | None = None
        self._scene_cache_dirty = True

    def run(self) -> None:
        clock = self.renderer.frame_clock()
        dt = 1.0 / self.args.fps
        running = True
        try:
            while running:
                for event in self.renderer.events():
                    if self.renderer.is_quit_event(event):
                        running = False
                    elif self.renderer.handle_resize_event(event):
                        continue
                    elif self.renderer.menu.visible and self.renderer.is_menu_pointer_event(event):
                        menu_action = self.renderer.handle_menu_mouse_event(event)
                        if menu_action is not None:
                            running = self._handle_menu_action(menu_action)
                    elif self.renderer.is_mouse_button_down_event(event) and not self.renderer.menu.visible:
                        self.renderer.set_mouse_captured(True)
                    elif self.renderer.is_mouse_motion_event(event) and self.renderer.mouse_captured:
                        rel = self.renderer.event_mouse_rel(event)
                        self.camera_controller.look(rel[0], rel[1])
                    elif self.renderer.is_key_down_event(event):
                        running = self.on_key_down(self.renderer.event_key(event))
                    elif self.renderer.is_key_up_event(event):
                        self.on_key_up(self.renderer.event_key(event))

                menu_visible = self.renderer.menu.visible
                if not menu_visible:
                    self.camera_controller.move(self.keys, dt)
                    self.sky.step(dt)
                    if not self.paused:
                        self.simulation.step(dt)
                self._update_hud()
                if menu_visible and self._cached_scene is not None and not self._scene_cache_dirty:
                    scene = self._cached_scene
                else:
                    scene = self.simulation.scene()
                    self._apply_sky(scene)
                    self._cached_scene = scene
                    self._scene_cache_dirty = False
                stats = self.renderer.render(scene, self.camera_controller.smoothed_camera(dt), self.sky.settings_for(self.settings))
                self._update_title(stats)
                dt = max(1.0 / 240.0, min(0.05, clock.tick(self.args.fps) / 1000.0))
        finally:
            self.renderer.close()

    def on_key_down(self, key: int) -> bool:
        menu_action = self.renderer.handle_menu_key(key)
        if menu_action is not None:
            return self._handle_menu_action(menu_action)
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.add(movement_key)
        elif self.renderer.key_matches(key, "p"):
            self.paused = not self.paused
            self._refresh_menu_options()
        elif self.renderer.key_matches(key, "r"):
            self._reset_simulation()
        elif self.renderer.key_matches(key, "leftbracket"):
            self._adjust_wind(-0.15)
        elif self.renderer.key_matches(key, "rightbracket"):
            self._adjust_wind(0.15)
        return True

    def on_key_up(self, key: int) -> None:
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.discard(movement_key)

    def _movement_key(self, key: int) -> str | None:
        if self.renderer.key_matches(key, "w"):
            return "w"
        if self.renderer.key_matches(key, "a"):
            return "a"
        if self.renderer.key_matches(key, "s"):
            return "s"
        if self.renderer.key_matches(key, "d"):
            return "d"
        if self.renderer.key_matches(key, "space"):
            return "space"
        if self.renderer.key_matches(key, "lshift", "rshift"):
            return "shift"
        if self.renderer.key_matches(key, "lctrl", "rctrl"):
            return "ctrl"
        return None

    def _handle_menu_action(self, action: str) -> bool:
        if action in {"handled", "navigate"}:
            return True
        if action == "opened":
            self.keys.clear()
            self.renderer.set_mouse_captured(False)
            self._refresh_menu_options()
            return True
        if action == "quit":
            return False
        if action in {"done", "resume", "cancel"}:
            self.renderer.menu.close()
            self.renderer.set_mouse_captured(True)
            return True
        if action == "quality_next":
            self._cycle_quality()
        elif action == "wind_up":
            self._adjust_wind(0.15)
        elif action == "wind_down":
            self._adjust_wind(-0.15)
        elif action == "blade_up":
            self._adjust_blade(0.15)
        elif action == "blade_down":
            self._adjust_blade(-0.15)
        elif action == "reflections_up":
            self._adjust_reflections(1)
        elif action == "reflections_down":
            self._adjust_reflections(-1)
        elif action == "smooth_shading":
            self._toggle_smooth_shading()
        elif action == "sky_cycle":
            self.sky.toggle_cycle()
        elif action == "sky_time_up":
            self.sky.adjust_time(1.0)
        elif action == "sky_time_down":
            self.sky.adjust_time(-1.0)
        elif action == "sky_clouds":
            self.sky.toggle_clouds()
        elif action == "sky_stars":
            self.sky.toggle_stars()
        elif action == "break_vessel":
            self.simulation.vessel_intact = not self.simulation.vessel_intact
        elif action == "pause":
            self.paused = not self.paused
        elif action == "reset":
            self._reset_simulation()
        self._scene_cache_dirty = True
        self._refresh_menu_options()
        return True

    def _cycle_quality(self) -> None:
        index = self.quality_order.index(self.args.quality) if self.args.quality in self.quality_order else 0
        self.args.quality = self.quality_order[(index + 1) % len(self.quality_order)]
        old_wind = self.simulation.wind_scale
        old_blade = self.simulation.blade_strength
        old_vessel = self.simulation.vessel_intact
        render_context = getattr(self.renderer, "ctx", None)
        fluid_context = render_context if getattr(render_context, "version_code", 0) >= 430 else None
        self.simulation = FanWaterSimulation(quality=self.args.quality, gpu_context=fluid_context)
        self.simulation.wind_scale = old_wind
        self.simulation.blade_strength = old_blade
        self.simulation.vessel_intact = old_vessel

    def _adjust_wind(self, amount: float) -> None:
        self.simulation.wind_scale = max(0.0, min(4.0, self.simulation.wind_scale + amount))

    def _adjust_blade(self, amount: float) -> None:
        self.simulation.blade_strength = max(0.0, min(2.5, self.simulation.blade_strength + amount))

    def _adjust_reflections(self, amount: int) -> None:
        self.settings = replace(self.settings, reflection_bounces=max(0, min(5, self.settings.reflection_bounces + amount)))

    def _toggle_smooth_shading(self) -> None:
        self.settings = replace(self.settings, smooth_shading=not self.settings.smooth_shading)

    def _reset_simulation(self) -> None:
        old_wind = self.simulation.wind_scale
        old_blade = self.simulation.blade_strength
        old_vessel = self.simulation.vessel_intact
        render_context = getattr(self.renderer, "ctx", None)
        fluid_context = render_context if getattr(render_context, "version_code", 0) >= 430 else None
        self.simulation = FanWaterSimulation(quality=self.args.quality, gpu_context=fluid_context)
        self.simulation.wind_scale = old_wind
        self.simulation.blade_strength = old_blade
        self.simulation.vessel_intact = old_vessel

    def _apply_sky(self, scene: Scene) -> Scene:
        scene.lights = [light for light in scene.lights if not isinstance(light, Sun)]
        self.sky.apply(scene)
        return scene

    def _refresh_menu_options(self) -> None:
        from py_3d.live import LiveMenuOption

        menu = self.renderer.menu
        previous_action = menu.selected_action() if menu.options else "done"
        options = (
            LiveMenuOption("done", "Done"),
            LiveMenuOption("quality_next", "Quality", self.args.quality, "Graphics"),
            LiveMenuOption("wind_up", "Wind +", f"{self.simulation.wind_scale:0.2f}x", "Physics"),
            LiveMenuOption("wind_down", "Wind -", f"{self.simulation.wind_scale:0.2f}x", "Physics"),
            LiveMenuOption("blade_up", "Swirl +", f"{self.simulation.blade_strength:0.2f}x", "Physics"),
            LiveMenuOption("blade_down", "Swirl -", f"{self.simulation.blade_strength:0.2f}x", "Physics"),
            LiveMenuOption("reflections_up", "Reflections +", str(self.settings.reflection_bounces), "Graphics"),
            LiveMenuOption("reflections_down", "Reflections -", str(self.settings.reflection_bounces), "Graphics"),
            LiveMenuOption("smooth_shading", "Smoothing", "on" if self.settings.smooth_shading else "off", "Graphics"),
            LiveMenuOption("sky_cycle", "Cycle", "on" if self.sky.cycle_enabled else "off", "Sky"),
            LiveMenuOption("sky_time_up", "Later", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_time_down", "Earlier", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_clouds", "Clouds", "on" if self.sky.clouds_enabled else "off", "Sky"),
            LiveMenuOption("sky_stars", "Stars", "on" if self.sky.stars_enabled else "off", "Sky"),
            LiveMenuOption("break_vessel", "Bowl", "intact" if self.simulation.vessel_intact else "broken", "Physics"),
            LiveMenuOption("pause", "Pause", "paused" if self.paused else "running", "Physics"),
            LiveMenuOption("reset", "Reset", "simulation", "Physics"),
            LiveMenuOption("quit", "Quit demo"),
        )
        menu.options = options
        actions = [option.action for option in options]
        menu.selected_index = actions.index(previous_action) if previous_action in actions else min(menu.selected_index, len(options) - 1)

    def _update_hud(self) -> None:
        self.renderer.hud.set(
            HUDRect((12, 12), (232, 74), (3, 7, 10), alpha=0.55),
            HUDText(
                f"FAN CLOTH WATER\nWIND {self.simulation.wind_scale:0.2f}  SWIRL {self.simulation.blade_strength:0.2f}\nSKY {self.sky.time_of_day:04.1f}H",
                (20, 20),
                color=(238, 245, 255),
                alpha=0.94,
                scale=1,
            ),
        )

    def _update_title(self, stats) -> None:
        ticks = self.renderer.ticks()
        if ticks - self._last_title_update < 400:
            return
        self._last_title_update = ticks
        self.renderer.set_title(
            f"py_3d fan cloth water - live - {self.args.quality} - {self.settings.reflection_bounces} refl "
            f"- {stats.approx_fps:0.1f} fps ({stats.build_seconds * 1000:0.1f} ms build, {stats.draw_seconds * 1000:0.1f} ms draw)"
        )


def render_still(args: argparse.Namespace) -> Path:
    simulation = FanWaterSimulation(quality=args.quality)
    for _ in range(int(args.warmup * args.fps)):
        simulation.step(1.0 / args.fps)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    RenderEngine().render(simulation.scene(), make_camera(), make_settings(args)).to_png(output)
    print(f"Wrote {output}")
    return output


def render_video(args: argparse.Namespace) -> Path:
    output = Path(args.video)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    if ffmpeg is None:
        message = ffmpeg_missing_message()
        if args.require_ffmpeg:
            raise RuntimeError(message)
        frames_dir = output.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = None
    simulation = FanWaterSimulation(quality=args.quality)
    engine = RenderEngine()
    settings = make_settings(args)
    if frames_dir is not None:
        for frame in range(args.frames):
            simulation.step(1.0 / args.fps)
            engine.render(simulation.scene(), make_camera(), settings).to_png(frames_dir / f"frame_{frame:04d}.png")
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
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("could not open ffmpeg stdin")
    try:
        for _frame in range(args.frames):
            simulation.step(1.0 / args.fps)
            process.stdin.write(engine.render(simulation.scene(), make_camera(), settings).to_ppm_bytes())
    finally:
        process.stdin.close()
    result = process.wait()
    if result != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result}")
    print(f"Wrote {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fan-driven cloth and vector-cloud water.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "fan_cloth_water.png")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--require-ffmpeg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--warmup", type=float, default=1.2)
    parser.add_argument("--quality", choices=("fast", "balanced", "high", "ultra"), default="ultra")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--menu-blur", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smooth-shading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ambient", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=1.12)
    parser.add_argument("--light-wrap", type=float, default=0.08)
    parser.add_argument("--bounce-light", type=float, default=0.16)
    parser.add_argument("--ray-traced-shadows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shadow-samples", type=int, default=3)
    parser.add_argument("--shadow-softness", type=float, default=0.22)
    parser.add_argument("--reflection-bounces", type=int, default=3)
    parser.add_argument("--texture-size", type=int, default=256)
    parser.add_argument("--sphere-segments", type=int, default=18)
    parser.add_argument("--sphere-rings", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.live:
        GLFanClothWaterViewer(args).run()
        return
    if args.video is not None:
        render_video(args)
    else:
        render_still(args)


if __name__ == "__main__":
    main()
