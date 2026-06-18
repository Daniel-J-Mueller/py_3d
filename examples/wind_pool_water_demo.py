"""Wind rushing over a particle-fluid water pool."""

from __future__ import annotations

import argparse
from dataclasses import replace
from math import cos, sin, tau
from pathlib import Path

from fan_cloth_water_demo import fan_primitives, water_surface_mesh
from py_3d import (
    Bowl,
    Box,
    Camera,
    Color,
    GPUVectorFluidWorld,
    HUDRect,
    HUDText,
    Lamp,
    Line3,
    Material,
    ParticleWaterSurface,
    RenderEngine,
    RenderSettings,
    Scene,
    SkyPrefab,
    Sphere,
    Sun,
    TextBulletin,
    Vec3,
    VectorFluidParticle,
    VectorFluidWorld,
)


OUTPUT_DIR = Path("USER") / "environments" / "wind_pool_water" / "renderings"


class WindPoolWaterSimulation:
    def __init__(self, *, quality: str = "balanced", gpu_context=None) -> None:
        self.time = 0.0
        self.quality = quality
        self.wind_strength = 1.0
        self.center = Vec3(0.25, 0.48, 0.0)
        self.radius = 1.29
        self.spill_radius = 2.45
        self.vessel_intact = True
        fluid_cls = GPUVectorFluidWorld if gpu_context is not None else VectorFluidWorld
        fluid_args = (gpu_context,) if gpu_context is not None else ()
        self.fluid = fluid_cls.liquid(
            *fluid_args,
            bounds_min=(self.center.x - self.radius, 0.22, self.center.z - self.radius),
            bounds_max=(self.center.x + self.radius, 0.72, self.center.z + self.radius),
            gravity=(0.0, -2.7, 0.0),
            rest_distance=0.065 if gpu_context is not None else 0.148,
            repel_strength=8.6 if gpu_context is not None else 9.8,
            self_attraction=0.58 if gpu_context is not None else 0.9,
            viscosity=0.5 if gpu_context is not None else 0.4,
            boundary_bounce=0.04,
            close_damping=11.0,
            attract_range_factor=1.65 if gpu_context is not None else 2.0,
        )
        if getattr(self.fluid, "gpu_enabled", False):
            self.fluid.sync_python_particles = False
            self.fluid.readback_particles = False
        self._seed_water_particles(gpu_context is not None)

    def _configure_fluid_bounds(self) -> None:
        radius = self.radius if self.vessel_intact else self.spill_radius
        floor_y = 0.22 if self.vessel_intact else 0.08
        ceiling_y = 0.74 if self.vessel_intact else 1.25
        self.fluid.bounds_min = Vec3(self.center.x - radius, floor_y, self.center.z - radius)
        self.fluid.bounds_max = Vec3(self.center.x + radius, ceiling_y, self.center.z + radius)

    def _seed_water_particles(self, gpu_enabled: bool) -> None:
        count = (
            {"fast": 1536, "balanced": 3072, "high": 5120, "ultra": 8192}.get(self.quality, 3072)
            if gpu_enabled
            else {"fast": 128, "balanced": 220, "high": 340, "ultra": 480}.get(self.quality, 220)
        )
        golden_angle = 3.141592653589793 * (3.0 - 5.0 ** 0.5)
        for index in range(count):
            amount = (index + 0.5) / count
            radius = self.radius * 0.96 * (amount ** 0.5)
            angle = index * golden_angle
            wave = 0.012 * sin(angle * 1.3 + radius * 4.0)
            self.fluid.add_particle(VectorFluidParticle(self.center + Vec3(cos(angle) * radius, -0.035 + wave, sin(angle) * radius)))

    def step(self, dt: float) -> None:
        self.time += dt
        self._configure_fluid_bounds()
        if getattr(self.fluid, "gpu_enabled", False):
            self.fluid.force_mode = 2
            self.fluid.force_center = self.center
            self.fluid.cylinder_radius = self.radius if self.vessel_intact else 0.0
            self.fluid.time = self.time
            self.fluid.wind_strength = self.wind_strength
            self.fluid.blade_strength = 0.0
            self.fluid.step(dt, substeps=1)
        else:
            self.fluid.step(dt, substeps=2, external_force=self._wind_force)
            if self.vessel_intact:
                self._constrain_pool()

    def _wind_force(self, particle: VectorFluidParticle, dt: float) -> tuple[float, float, float]:
        del dt
        local_x = particle.position.x - self.center.x
        local_z = particle.position.z - self.center.z
        surface = max(0.0, min(1.0, (particle.position.y - (self.center.y - 0.12)) / 0.22))
        lane = max(0.0, 1.0 - (local_z / max(0.001, self.radius)) ** 2)
        gust = 0.76 + 0.24 * sin(self.time * tau * 1.35 + local_z * 4.4)
        cross = 0.22 * sin(self.time * 3.6 + local_x * 4.8)
        lift = 0.08 * sin(self.time * tau * 2.2 + local_x * 7.0 + local_z * 2.0)
        return (1.55 * self.wind_strength * lane * gust * surface, lift * surface, cross * surface)

    def _constrain_pool(self) -> None:
        for particle in self.fluid.particles:
            local = particle.position - self.center
            horizontal = Vec3(local.x, 0.0, local.z)
            distance = horizontal.length()
            if distance > self.radius:
                normal = horizontal / distance
                particle.position = self.center + normal * self.radius + Vec3(0.0, local.y, 0.0)
                outward = particle.velocity.dot(normal)
                if outward > 0.0:
                    particle.velocity = particle.velocity - normal * (outward * 1.35)
            if particle.position.y < self.center.y - 0.2 and particle.velocity.y < 0.0:
                particle.velocity = Vec3(particle.velocity.x * 0.98, particle.velocity.y * 0.2, particle.velocity.z * 0.98)

    def scene(self) -> Scene:
        scene = Scene()
        stone = Material(color=(72, 78, 76), roughness=0.68, fuzziness=0.08, specular=0.04)
        scene.add(Box((0.0, -0.04, 0.0), (5.6, 0.08, 3.5), stone))
        if self.vessel_intact:
            scene.add(Bowl((self.center.x, 0.72, self.center.z), 1.35, Material(color=(84, 90, 88), roughness=0.6, specular=0.08), depth=0.56, thickness=0.07))
        surface_radius = self.radius if self.vessel_intact else self.spill_radius
        if getattr(self.fluid, "gpu_enabled", False):
            scene.add(
                ParticleWaterSurface(
                    self.fluid,
                    self.center,
                    surface_radius,
                    quality=self.quality,
                    time=self.time,
                    particle_base_y=self.center.y - 0.035,
                )
            )
        else:
            scene.add(water_surface_mesh(self.fluid, self.center, surface_radius, self.time, quality=self.quality))
        scene.add(*fan_primitives(Vec3(-2.15, 0.72, 0.0), self.time, facing="x"))
        scene.add(*wind_stream_lines(self.center, self.time, self.wind_strength))
        scene.add_light(Sun(direction=(-0.3, -0.84, -0.46), color=(255, 244, 224), intensity=0.82))
        scene.add_light(Lamp(position=(-1.8, 1.8, -1.1), color=(120, 170, 255), intensity=3.2))
        scene.add_light(Lamp(position=(1.8, 1.35, 1.0), color=(255, 218, 170), intensity=2.0))
        scene.add_bulletin(TextBulletin("WIND POOL WATER\nPARTICLE LIQUID SURFACE", position=(10, 10), color=(245, 248, 255), background=(5, 8, 10), padding=5))
        return scene


def wind_stream_lines(center: Vec3, time: float, strength: float) -> tuple[Line3, ...]:
    material = Material(color=(164, 210, 255), emission=(22, 34, 48))
    lines: list[Line3] = []
    for index in range(7):
        z = -0.9 + index * 0.3
        y = center.y + 0.28 + 0.035 * sin(time * 2.8 + index)
        start_x = center.x - 1.55
        end_x = center.x + 1.25
        offset = 0.08 * strength * sin(time * tau * 1.2 + index * 0.7)
        lines.append(Line3((start_x, y, z), (end_x, y + offset * 0.25, z + offset), material))
    return tuple(lines)


def make_camera() -> Camera:
    return Camera(position=(0.25, 1.65, -4.15), target=(0.2, 0.54, 0.0), fov_degrees=52)


def make_settings(args: argparse.Namespace) -> RenderSettings:
    return RenderSettings(
        width=args.width,
        height=args.height,
        background=Color(7, 10, 13),
        ambient=args.ambient,
        gamma=args.gamma,
        light_wrap=getattr(args, "light_wrap", 0.08),
        bounce_light=getattr(args, "bounce_light", 0.16),
        smooth_shading=True,
        tone_mapping=True,
        ray_traced_shadows=bool(getattr(args, "ray_traced_shadows", True)),
        reflection_bounces=args.reflection_bounces,
        shadow_samples=getattr(args, "shadow_samples", 3),
        shadow_softness=getattr(args, "shadow_softness", 0.22),
        sphere_segments=args.sphere_segments,
        sphere_rings=args.sphere_rings,
        texture_size=args.texture_size,
    )


class GLWindPoolWaterViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        from py_3d.live import LiveFlyCamera, LiveMenu, LiveMenuOption, ModernGLLiveRenderer

        self.args = args
        self.settings = make_settings(args)
        self.sky = SkyPrefab(time_of_day=14.5, cycle_enabled=False, stars_enabled=True, clouds_enabled=True)
        base_camera = make_camera()
        self.camera_controller = LiveFlyCamera.looking_at(base_camera.position, base_camera.target, fov_degrees=base_camera.fov_degrees, speed=2.3)
        self.keys: set[str] = set()
        self.paused = False
        self.quality_order = ("fast", "balanced", "high", "ultra")
        self.renderer = ModernGLLiveRenderer(args.window_width, args.window_height, title="py_3d wind pool water - live", vsync=getattr(args, "vsync", True))
        render_context = getattr(self.renderer, "ctx", None)
        fluid_context = render_context if getattr(render_context, "version_code", 0) >= 430 else None
        self.simulation = WindPoolWaterSimulation(quality=args.quality, gpu_context=fluid_context)
        self.renderer.menu = LiveMenu(
            "py_3d wind pool water",
            (
                LiveMenuOption("done", "Done"),
                LiveMenuOption("quality_next", "Quality preset"),
                LiveMenuOption("wind_up", "More wind"),
                LiveMenuOption("wind_down", "Less wind"),
                LiveMenuOption("reflections_up", "More reflections"),
                LiveMenuOption("reflections_down", "Fewer reflections"),
                LiveMenuOption("break_vessel", "Break/repair bowl"),
                LiveMenuOption("pause", "Pause/run water"),
                LiveMenuOption("reset", "Reset pool"),
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
                        action = self.renderer.handle_menu_mouse_event(event)
                        if action is not None:
                            running = self._handle_menu_action(action)
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
        action = self.renderer.handle_menu_key(key)
        if action is not None:
            return self._handle_menu_action(action)
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.add(movement_key)
        elif self.renderer.key_matches(key, "p"):
            self.paused = not self.paused
            self._refresh_menu_options()
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
        elif action == "reflections_up":
            self.settings = replace(self.settings, reflection_bounces=max(0, min(5, self.settings.reflection_bounces + 1)))
        elif action == "reflections_down":
            self.settings = replace(self.settings, reflection_bounces=max(0, min(5, self.settings.reflection_bounces - 1)))
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
        self._reset_simulation()

    def _adjust_wind(self, amount: float) -> None:
        self.simulation.wind_strength = max(0.0, min(4.0, self.simulation.wind_strength + amount))

    def _reset_simulation(self) -> None:
        old_wind = self.simulation.wind_strength
        old_vessel = self.simulation.vessel_intact
        render_context = getattr(self.renderer, "ctx", None)
        fluid_context = render_context if getattr(render_context, "version_code", 0) >= 430 else None
        self.simulation = WindPoolWaterSimulation(quality=self.args.quality, gpu_context=fluid_context)
        self.simulation.wind_strength = old_wind
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
            LiveMenuOption("wind_up", "Wind +", f"{self.simulation.wind_strength:0.2f}x", "Physics"),
            LiveMenuOption("wind_down", "Wind -", f"{self.simulation.wind_strength:0.2f}x", "Physics"),
            LiveMenuOption("reflections_up", "Reflections +", str(self.settings.reflection_bounces), "Graphics"),
            LiveMenuOption("reflections_down", "Reflections -", str(self.settings.reflection_bounces), "Graphics"),
            LiveMenuOption("break_vessel", "Bowl", "intact" if self.simulation.vessel_intact else "broken", "Physics"),
            LiveMenuOption("pause", "Pause", "paused" if self.paused else "running", "Physics"),
            LiveMenuOption("reset", "Reset", "pool", "Physics"),
            LiveMenuOption("quit", "Quit demo"),
        )
        menu.options = options
        actions = [option.action for option in options]
        menu.selected_index = actions.index(previous_action) if previous_action in actions else min(menu.selected_index, len(options) - 1)

    def _update_hud(self) -> None:
        self.renderer.hud.set(
            HUDRect((12, 12), (220, 62), (3, 7, 10), alpha=0.55),
            HUDText(f"WIND POOL WATER\nWIND {self.simulation.wind_strength:0.2f}  {self.args.quality.upper()}", (20, 20), color=(238, 245, 255), alpha=0.94, scale=1),
        )

    def _update_title(self, stats) -> None:
        ticks = self.renderer.ticks()
        if ticks - self._last_title_update < 400:
            return
        self._last_title_update = ticks
        self.renderer.set_title(
            f"py_3d wind pool water - live - {self.args.quality} - {stats.approx_fps:0.1f} fps "
            f"({stats.build_seconds * 1000:0.1f} ms build, {stats.draw_seconds * 1000:0.1f} ms draw)"
        )


def render_still(args: argparse.Namespace) -> Path:
    simulation = WindPoolWaterSimulation(quality=args.quality)
    simulation.wind_strength = args.wind_strength
    for _ in range(int(args.warmup * args.fps)):
        simulation.step(1.0 / args.fps)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    RenderEngine().render(simulation.scene(), make_camera(), make_settings(args)).to_png(output)
    print(f"Wrote {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render or run wind rushing over a particle-fluid water pool.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "wind_pool_water.png")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--warmup", type=float, default=1.2)
    parser.add_argument("--quality", choices=("fast", "balanced", "high", "ultra"), default="balanced")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--menu-blur", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wind-strength", type=float, default=1.0)
    parser.add_argument("--ambient", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=1.12)
    parser.add_argument("--light-wrap", type=float, default=0.08)
    parser.add_argument("--bounce-light", type=float, default=0.16)
    parser.add_argument("--ray-traced-shadows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shadow-samples", type=int, default=3)
    parser.add_argument("--shadow-softness", type=float, default=0.22)
    parser.add_argument("--reflection-bounces", type=int, default=3)
    parser.add_argument("--texture-size", type=int, default=256)
    parser.add_argument("--sphere-segments", type=int, default=16)
    parser.add_argument("--sphere-rings", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.live:
        GLWindPoolWaterViewer(args).run()
        return
    render_still(args)


if __name__ == "__main__":
    main()
