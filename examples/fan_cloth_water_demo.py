"""Fan, cloth, and vector-fluid water demonstration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from math import cos, pi, sin, tau
from pathlib import Path
import subprocess

from fruit_bowl_demo import find_ffmpeg, ffmpeg_missing_message
from py_3d import (
    Bowl,
    Box,
    Camera,
    Color,
    HUDRect,
    HUDText,
    Line3,
    Lamp,
    Material,
    Mesh,
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

    def step(self, dt: float, time: float, substeps: int = 3, *, wind_scale: float = 1.0) -> None:
        step_dt = dt / substeps
        for _ in range(substeps):
            forces = [Vec3(0.0, -1.15, 0.0) for _node in self.nodes]
            self._spring_forces(forces, stiffness=48.0)
            for index, node in enumerate(self.nodes):
                if node.pinned:
                    node.velocity = Vec3(0.0, 0.0, 0.0)
                    continue
                wind = self._wind_force(node.position, time) * wind_scale
                node.velocity = (node.velocity + (forces[index] + wind) * step_dt) * 0.985
                node.position = node.position + node.velocity * step_dt
                if node.position.y < 0.14:
                    node.position = Vec3(node.position.x, 0.14, node.position.z)
                    node.velocity = Vec3(node.velocity.x, abs(node.velocity.y) * 0.08, node.velocity.z)

    def _spring_forces(self, forces: list[Vec3], *, stiffness: float) -> None:
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
        for first_index, second_index, rest_length in links:
            first = self.nodes[first_index]
            second = self.nodes[second_index]
            delta = second.position - first.position
            distance = delta.length()
            if distance <= 1e-6:
                continue
            amount = (distance - rest_length) * stiffness
            force = delta / distance * amount
            forces[first_index] = forces[first_index] + force
            forces[second_index] = forces[second_index] - force

    @staticmethod
    def _wind_force(position: Vec3, time: float) -> Vec3:
        source = Vec3(-2.0, 0.98, 0.0)
        offset = position - source
        distance = max(0.12, offset.length())
        alignment = max(0.0, offset.normalized(Vec3(1.0, 0.0, 0.0)).dot(Vec3(1.0, 0.0, 0.0)))
        pulse = 0.82 + 0.18 * sin(time * tau * 1.7 + position.z * 4.2)
        turbulence = Vec3(0.0, sin(time * 3.4 + position.z * 5.0) * 0.32, cos(time * 2.7 + position.y * 4.0) * 0.28)
        return (Vec3(1.0, 0.0, 0.0) + turbulence) * (5.2 * alignment * pulse / (1.0 + distance * distance * 0.9))

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

    def wire(self) -> tuple[Line3, ...]:
        material = Material(color=(218, 228, 220), emission=(18, 20, 18))
        lines: list[Line3] = []
        for row in range(self.rows):
            for column in range(self.columns):
                if column + 1 < self.columns:
                    lines.append(Line3(self.nodes[self._index(column, row)].position, self.nodes[self._index(column + 1, row)].position, material))
                if row + 1 < self.rows:
                    lines.append(Line3(self.nodes[self._index(column, row)].position, self.nodes[self._index(column, row + 1)].position, material))
        return tuple(lines)

    def _index(self, column: int, row: int) -> int:
        return row * self.columns + column


class FanWaterSimulation:
    def __init__(self, *, quality: str = "balanced") -> None:
        self.time = 0.0
        self.quality = quality
        self.wind_scale = 1.0
        self.blade_strength = 1.0
        self.cloth = ClothSheet(columns=14 if quality != "fast" else 10, rows=10 if quality != "fast" else 7)
        self.water_center = Vec3(1.25, 0.64, 0.0)
        self.water_radius = 0.58
        self.fluid = VectorFluidWorld(
            bounds_min=(self.water_center.x - self.water_radius, 0.42, self.water_center.z - self.water_radius),
            bounds_max=(self.water_center.x + self.water_radius, 0.78, self.water_center.z + self.water_radius),
            gravity=(0.0, -1.6, 0.0),
            rest_distance=0.14,
            repel_strength=5.6,
            self_attraction=1.9,
            viscosity=0.22,
            boundary_bounce=0.08,
        )
        rings = (0.0, 0.2, 0.36, 0.5)
        for ring_index, radius in enumerate(rings):
            count = max(1, 6 * ring_index)
            for index in range(count):
                angle = tau * index / count + ring_index * 0.24
                position = self.water_center + Vec3(cos(angle) * radius, 0.0, sin(angle) * radius)
                self.fluid.add_particle(VectorFluidParticle(position, velocity=(-sin(angle) * 0.05, 0.0, cos(angle) * 0.05)))

    def step(self, dt: float) -> None:
        self.time += dt
        self.cloth.step(dt, self.time, wind_scale=self.wind_scale)
        self.fluid.step(dt, substeps=3, external_force=self._fan_blade_force)
        self._constrain_water()

    def _fan_blade_force(self, particle: VectorFluidParticle, dt: float) -> Vec3:
        del dt
        local = particle.position - self.water_center
        horizontal = Vec3(local.x, 0.0, local.z)
        distance = max(0.04, horizontal.length())
        tangent = Vec3(-horizontal.z, 0.0, horizontal.x).normalized(Vec3(0.0, 0.0, 1.0))
        blade_phase = sin(self.time * tau * 2.4 + distance * 8.0)
        swirl = tangent * (1.25 / (1.0 + distance * 3.0))
        lift = Vec3(0.0, 0.2 * blade_phase, 0.0)
        return (swirl + lift) * self.blade_strength

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

    def scene(self) -> Scene:
        scene = Scene()
        scene.add(Box((0.0, -0.04, 0.0), (5.2, 0.08, 3.0), Material(color=(48, 58, 60), roughness=0.62, fuzziness=0.08)))
        scene.add(Box((-0.62, 1.64, 0.0), (0.08, 0.08, 1.46), Material(color=(92, 76, 52), roughness=0.5)))
        scene.add(self.cloth.mesh(), *self.cloth.wire())
        scene.add(*fan_primitives(Vec3(-2.05, 0.98, 0.0), self.time, facing="x"))
        scene.add(Bowl((self.water_center.x, 0.78, self.water_center.z), 0.76, Material(color=(122, 82, 48), roughness=0.62, fuzziness=0.14, specular=0.05), depth=0.82, thickness=0.05))
        scene.add(water_surface_mesh(self.fluid, self.water_center, self.water_radius, self.time, quality=self.quality))
        scene.add(*water_fan_primitives(self.water_center + Vec3(0.0, 0.03, 0.0), self.time))
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


def water_surface_mesh(world: VectorFluidWorld, center: Vec3, radius: float, time: float, *, quality: str) -> Mesh:
    steps = {"fast": 9, "balanced": 13, "high": 17, "ultra": 21}.get(quality, 13)
    material = Material(color=(210, 235, 255), texture=water_texture(), roughness=0.06, specular=0.72, shininess=88.0, reflectivity=0.32, light_transmission=0.35)
    points: dict[tuple[int, int], Vec3] = {}
    for row in range(steps + 1):
        z = -radius + 2.0 * radius * row / steps
        for column in range(steps + 1):
            x = -radius + 2.0 * radius * column / steps
            if x * x + z * z > radius * radius:
                continue
            height = center.y
            for particle in world.particles:
                dx = center.x + x - particle.position.x
                dz = center.z + z - particle.position.z
                influence = max(0.0, 1.0 - (dx * dx + dz * dz) / 0.08)
                height += influence * (particle.position.y - center.y) * 0.28
            height += sin(time * tau * 1.25 + x * 5.2 + z * 3.7) * 0.012
            points[(column, row)] = Vec3(center.x + x, height, center.z + z)
    triangles: list[Triangle] = []
    for row in range(steps):
        for column in range(steps):
            keys = ((column, row), (column + 1, row), (column + 1, row + 1), (column, row + 1))
            if not all(key in points for key in keys):
                continue
            a, b, c, d = (points[key] for key in keys)
            u0 = column / steps
            u1 = (column + 1) / steps
            v0 = row / steps
            v1 = (row + 1) / steps
            triangles.append(Triangle(a, b, c, material, (u0, v0), (u1, v0), (u1, v1)))
            triangles.append(Triangle(a, c, d, material, (u0, v0), (u1, v1), (u0, v1)))
    return Mesh(triangles)


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
        smooth_shading=True,
        tone_mapping=True,
        reflection_bounces=args.reflection_bounces,
        sphere_segments=args.sphere_segments,
        sphere_rings=args.sphere_rings,
        texture_size=args.texture_size,
    )


class GLFanClothWaterViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        import pygame
        from py_3d.live import LiveFlyCamera, LiveMenu, LiveMenuOption, ModernGLLiveRenderer

        self.pygame = pygame
        self.args = args
        self.simulation = FanWaterSimulation(quality=args.quality)
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
            title="py_3d fan cloth water - OpenGL live",
            vsync=getattr(args, "vsync", True),
        )
        self.renderer.menu = LiveMenu(
            "py_3d fan cloth water",
            (
                LiveMenuOption("done", "Done"),
                LiveMenuOption("quality_next", "Quality preset"),
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
                LiveMenuOption("pause", "Pause/run physics"),
                LiveMenuOption("reset", "Reset simulation"),
                LiveMenuOption("quit", "Quit demo"),
            ),
        )
        self._refresh_menu_options()
        self.renderer.set_mouse_captured(True)
        self._last_title_update = 0

    def run(self) -> None:
        clock = self.pygame.time.Clock()
        dt = 1.0 / self.args.fps
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
                    elif event.type == self.pygame.MOUSEMOTION and self.renderer.mouse_captured:
                        self.camera_controller.look(event.rel[0], event.rel[1])
                    elif event.type == self.pygame.KEYDOWN:
                        running = self.on_key_down(event.key)
                    elif event.type == self.pygame.KEYUP:
                        self.on_key_up(event.key)

                self.camera_controller.move(self.keys, dt)
                self.sky.step(dt)
                if not self.paused:
                    self.simulation.step(dt)
                self._update_hud()
                scene = self.simulation.scene()
                self._apply_sky(scene)
                stats = self.renderer.render(scene, self.camera_controller.smoothed_camera(dt), self.sky.settings_for(self.settings))
                self._update_title(stats)
                dt = max(1.0 / 240.0, min(0.05, clock.tick(self.args.fps) / 1000.0))
        finally:
            self.renderer.close()

    def on_key_down(self, key: int) -> bool:
        pygame = self.pygame
        menu_action = self.renderer.menu.handle_key(key, pygame)
        if menu_action is not None:
            return self._handle_menu_action(menu_action)
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.add(movement_key)
        elif key == pygame.K_p:
            self.paused = not self.paused
            self._refresh_menu_options()
        elif key == pygame.K_r:
            self._reset_simulation()
        elif key == pygame.K_LEFTBRACKET:
            self._adjust_wind(-0.15)
        elif key == pygame.K_RIGHTBRACKET:
            self._adjust_wind(0.15)
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
        if key == pygame.K_SPACE:
            return "space"
        if key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
            return "shift"
        if key in (pygame.K_LCTRL, pygame.K_RCTRL):
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
        elif action == "pause":
            self.paused = not self.paused
        elif action == "reset":
            self._reset_simulation()
        self._refresh_menu_options()
        return True

    def _cycle_quality(self) -> None:
        index = self.quality_order.index(self.args.quality) if self.args.quality in self.quality_order else 0
        self.args.quality = self.quality_order[(index + 1) % len(self.quality_order)]
        old_wind = self.simulation.wind_scale
        old_blade = self.simulation.blade_strength
        self.simulation = FanWaterSimulation(quality=self.args.quality)
        self.simulation.wind_scale = old_wind
        self.simulation.blade_strength = old_blade

    def _adjust_wind(self, amount: float) -> None:
        self.simulation.wind_scale = max(0.0, min(2.5, self.simulation.wind_scale + amount))

    def _adjust_blade(self, amount: float) -> None:
        self.simulation.blade_strength = max(0.0, min(2.5, self.simulation.blade_strength + amount))

    def _adjust_reflections(self, amount: int) -> None:
        self.settings = replace(self.settings, reflection_bounces=max(0, min(5, self.settings.reflection_bounces + amount)))

    def _reset_simulation(self) -> None:
        old_wind = self.simulation.wind_scale
        old_blade = self.simulation.blade_strength
        self.simulation = FanWaterSimulation(quality=self.args.quality)
        self.simulation.wind_scale = old_wind
        self.simulation.blade_strength = old_blade

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
            LiveMenuOption("sky_cycle", "Cycle", "on" if self.sky.cycle_enabled else "off", "Sky"),
            LiveMenuOption("sky_time_up", "Later", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_time_down", "Earlier", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_clouds", "Clouds", "on" if self.sky.clouds_enabled else "off", "Sky"),
            LiveMenuOption("sky_stars", "Stars", "on" if self.sky.stars_enabled else "off", "Sky"),
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
        ticks = self.pygame.time.get_ticks()
        if ticks - self._last_title_update < 400:
            return
        self._last_title_update = ticks
        self.renderer.set_title(
            f"py_3d fan cloth water - OpenGL live - {self.args.quality} - {self.settings.reflection_bounces} refl "
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
    parser.add_argument("--quality", choices=("fast", "balanced", "high", "ultra"), default="balanced")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ambient", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=1.12)
    parser.add_argument("--reflection-bounces", type=int, default=2)
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
