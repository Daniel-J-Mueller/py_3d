"""Indefinite deterministic procedural hill biome demo."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from math import cos, radians, sin
from pathlib import Path
from time import perf_counter

from sim_settings import live_settings_for, update_live_settings

from py_3d import (
    Camera,
    HUDRect,
    HUDText,
    GPURenderer,
    Lamp,
    ProceduralEnvironmentConfig,
    ProceduralEnvironmentGenerator,
    RenderEngine,
    RenderSettings,
    Scene,
    SkyPrefab,
    Vec3,
    build_environment_chunk,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "renderings-tests" / "live-renders"
DEMO_SETTINGS = "procedural_hills"


QUALITY_PROFILES = {
    "fast": dict(chunk_resolution=12, shoreline_detail=1, active_radius=1, tree_slots_per_axis=3, tree_density=0.16, bush_slots_per_axis=4, bush_density=0.18, rock_slots_per_axis=3, rock_density=0.08, grass_blades_per_chunk=42, sphere_segments=8, sphere_rings=4),
    "balanced": dict(chunk_resolution=16, shoreline_detail=1, active_radius=1, tree_slots_per_axis=4, tree_density=0.22, bush_slots_per_axis=5, bush_density=0.25, rock_slots_per_axis=4, rock_density=0.1, grass_blades_per_chunk=84, sphere_segments=10, sphere_rings=5),
    "high": dict(chunk_resolution=20, shoreline_detail=2, active_radius=2, tree_slots_per_axis=5, tree_density=0.26, bush_slots_per_axis=6, bush_density=0.3, rock_slots_per_axis=5, rock_density=0.12, grass_blades_per_chunk=112, sphere_segments=12, sphere_rings=6),
    "ultra": dict(chunk_resolution=24, shoreline_detail=2, active_radius=3, tree_slots_per_axis=5, tree_density=0.3, bush_slots_per_axis=7, bush_density=0.34, rock_slots_per_axis=5, rock_density=0.14, grass_blades_per_chunk=150, sphere_segments=14, sphere_rings=7),
}


@dataclass
class TerrainWalker:
    position: Vec3
    yaw_degrees: float = 38.0
    pitch_degrees: float = -7.0
    eye_height: float = 1.75
    standing_eye_height: float = 1.75
    crouch_eye_height: float = 1.12
    speed: float = 4.8
    sprint_multiplier: float = 1.45
    crouch_speed_multiplier: float = 0.62

    @property
    def flat_forward(self) -> Vec3:
        yaw = radians(self.yaw_degrees)
        return Vec3(sin(yaw), 0.0, cos(yaw)).normalized(Vec3(0.0, 0.0, 1.0))

    @property
    def forward(self) -> Vec3:
        yaw = radians(self.yaw_degrees)
        pitch = radians(self.pitch_degrees)
        return Vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)).normalized(Vec3(0.0, 0.0, 1.0))

    @property
    def right(self) -> Vec3:
        forward = self.flat_forward
        return Vec3(forward.z, 0.0, -forward.x)

    def look(self, dx: float, dy: float) -> None:
        self.yaw_degrees += dx * 0.12
        self.pitch_degrees = max(-82.0, min(82.0, self.pitch_degrees - dy * 0.12))

    def step(self, keys: set[str], dt: float, environment: ProceduralEnvironmentGenerator) -> None:
        target_eye_height = self.crouch_eye_height if "crouch" in keys else self.standing_eye_height
        self.eye_height += (target_eye_height - self.eye_height) * min(1.0, dt * 10.0)
        move = Vec3(0.0, 0.0, 0.0)
        if "w" in keys:
            move = move + self.flat_forward
        if "s" in keys:
            move = move - self.flat_forward
        if "d" in keys:
            move = move + self.right
        if "a" in keys:
            move = move - self.right
        if move.length_squared() > 0.0:
            move = move.normalized()
        speed = self.speed
        if "crouch" in keys:
            speed *= self.crouch_speed_multiplier
        elif "sprint" in keys:
            speed *= self.sprint_multiplier
        if environment.is_water_at(self.position.x, self.position.z):
            speed *= 0.55
        next_position = self.position + move * (speed * max(0.0, dt))
        ground = environment.height_at(next_position.x, next_position.z)
        feet_y = max(ground, environment.config.water_level - 0.1)
        self.position = Vec3(next_position.x, feet_y, next_position.z)

    def camera(self) -> Camera:
        eye = self.position + Vec3(0.0, self.eye_height, 0.0)
        return Camera(position=eye, target=eye + self.forward, fov_degrees=68.0, near=0.04, far=700.0)


class ChunkBuildWorker:
    def __init__(self, config: ProceduralEnvironmentConfig, *, enabled: bool = True, mode: str = "process") -> None:
        self.config = config
        self.enabled = enabled
        self.mode = mode
        self.executor: ProcessPoolExecutor | ThreadPoolExecutor | None = self._make_executor()
        self.futures: dict[tuple[int, int], Future] = {}
        self.failed = False

    @property
    def pending_count(self) -> int:
        return len(self.futures)

    def request(self, coord: tuple[int, int]) -> bool:
        if self.executor is None or self.failed or coord in self.futures:
            return False
        try:
            self.futures[coord] = self.executor.submit(build_environment_chunk, self.config, coord)
            return True
        except Exception:
            self.failed = True
            return False

    def collect_ready(self) -> tuple[object, ...]:
        ready: list[object] = []
        for coord, future in tuple(self.futures.items()):
            if not future.done():
                continue
            del self.futures[coord]
            try:
                ready.append(future.result())
            except Exception:
                self.failed = True
        return tuple(ready)

    def reset(self, config: ProceduralEnvironmentConfig) -> None:
        self.close()
        self.config = config
        self.failed = False
        self.futures = {}
        self.executor = self._make_executor()

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = None
        self.futures = {}

    def _make_executor(self) -> ProcessPoolExecutor | ThreadPoolExecutor | None:
        if not self.enabled:
            return None
        if self.mode == "process":
            return ProcessPoolExecutor(max_workers=1)
        return ThreadPoolExecutor(max_workers=1)


class ProceduralEnvironmentViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        from py_3d.live import LiveMenu, LiveMenuOption, ModernGLLiveRenderer

        self.args = args
        self.config = make_config(args)
        self.environment = ProceduralEnvironmentGenerator(self.config)
        self.spawn = find_demo_spawn(self.environment)
        self.environment.chunk(self.environment.chunk_coordinate(self.spawn))
        self.walker = TerrainWalker(self.spawn)
        self.render_camera: Camera | None = None
        self.keys: set[str] = set()
        self.active_radius = args.active_radius if args.active_radius is not None else self.config.active_radius
        self.preload_margin = max(0, int(args.preload_margin))
        self.max_pending_chunks = max(1, int(args.max_pending_chunks))
        self.chunk_activation_rate = max(1, int(args.chunk_activation_rate))
        self.look_smoothing = args.look_smoothing
        self.sky = SkyPrefab(
            time_of_day=args.sky_time,
            cycle_enabled=args.sky_cycle,
            stars_enabled=args.sky_stars,
            clouds_enabled=args.sky_clouds,
            radius=360.0,
            cloud_count=8,
            cloud_seed=args.seed,
        )
        self.settings = make_settings(args)
        self._static_scene_cache_key: tuple | None = None
        self._static_scene: Scene | None = None
        self._frame_scene = Scene()
        self._stream_version = 0
        self._stream_target: tuple[tuple[int, int], ...] = ()
        self._stream_missing_count = 0
        self._activated_chunk_objects: dict[tuple[int, int], int] = {}
        spawn_chunk = self.environment.cached_chunk(self.environment.chunk_coordinate(self.spawn))
        if spawn_chunk is not None:
            self._activated_chunk_objects[spawn_chunk.coord] = len(spawn_chunk.objects)
        self.chunk_worker = ChunkBuildWorker(self.config, enabled=args.chunk_worker, mode=args.chunk_worker_mode)
        self.renderer = ModernGLLiveRenderer(args.window_width, args.window_height, title="py_3d procedural hills", vsync=args.vsync, backend=args.live_backend)
        self.renderer.menu = LiveMenu(
            "py_3d procedural hills",
            (
                LiveMenuOption("done", "Done"),
                LiveMenuOption("radius_up", "More view distance"),
                LiveMenuOption("radius_down", "Less view distance"),
                LiveMenuOption("look_smoothing_up", "More look smoothing"),
                LiveMenuOption("look_smoothing_down", "Less look smoothing"),
                LiveMenuOption("rebuild", "Rebuild visible chunks"),
                LiveMenuOption("seed_next", "Next seed"),
                LiveMenuOption("reset", "Reset position"),
                LiveMenuOption("sky_cycle", "Day/night cycle"),
                LiveMenuOption("sky_time_up", "Time later"),
                LiveMenuOption("sky_time_down", "Time earlier"),
                LiveMenuOption("sky_clouds", "Clouds"),
                LiveMenuOption("snapshot", "Save snapshot"),
                LiveMenuOption("quit", "Quit demo"),
            ),
            background_blur=args.menu_blur,
        )
        self._refresh_menu_options()
        self.renderer.set_mouse_captured(True)
        self._last_title_update = 0

    def run(self, *, frame_limit: int | None = None, auto_walk: bool = False) -> None:
        clock = self.renderer.frame_clock()
        dt = 1.0 / self.args.fps
        running = True
        frame_index = 0
        frame_times: list[float] = []
        received_chunks_per_frame: list[int] = []
        collect_times: list[float] = []
        scene_times: list[float] = []
        render_build_times: list[float] = []
        render_draw_times: list[float] = []
        try:
            while running:
                frame_started = perf_counter()
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
                        dx, dy = self.renderer.event_mouse_rel(event)
                        self.walker.look(dx, dy)
                    elif self.renderer.is_key_down_event(event):
                        running = self.on_key_down(self.renderer.event_key(event))
                    elif self.renderer.is_key_up_event(event):
                        self.on_key_up(self.renderer.event_key(event))

                if not self.renderer.menu.visible:
                    if auto_walk:
                        self.keys.add("w")
                        if frame_index % 160 > 108:
                            self.keys.add("d")
                        else:
                            self.keys.discard("d")
                        self.walker.look(0.35, 0.0)
                    self.walker.step(self.keys, dt, self.environment)
                    self.sky.step(dt)
                collect_started = perf_counter()
                received_chunks = self._collect_streamed_chunks()
                collect_elapsed = perf_counter() - collect_started
                self._advance_chunk_activation()
                scene_started = perf_counter()
                scene = self.scene()
                scene_elapsed = perf_counter() - scene_started
                self._update_hud()
                stats = self.renderer.render(scene, self._smoothed_camera(self.walker.camera(), dt), self.sky.settings_for(self.settings))
                self._update_title(stats)
                self._schedule_chunk_stream()
                dt = max(1.0 / 240.0, min(0.05, clock.tick(self.args.fps) / 1000.0))
                if frame_limit is not None:
                    frame_times.append(perf_counter() - frame_started)
                    received_chunks_per_frame.append(received_chunks)
                    collect_times.append(collect_elapsed)
                    scene_times.append(scene_elapsed)
                    render_build_times.append(stats.build_seconds)
                    render_draw_times.append(stats.draw_seconds)
                frame_index += 1
                if frame_limit is not None and frame_index >= frame_limit:
                    running = False
        finally:
            pending_chunks = self.chunk_worker.pending_count
            self.chunk_worker.close()
            self.renderer.close()
        if frame_limit is not None and frame_times:
            average_ms = sum(frame_times) * 1000.0 / len(frame_times)
            max_ms = max(frame_times) * 1000.0
            max_index = max(range(len(frame_times)), key=lambda index: frame_times[index])
            steady = frame_times[min(20, len(frame_times) - 1) :]
            steady_max_ms = max(steady) * 1000.0 if steady else max_ms
            slow_50 = sum(1 for value in frame_times if value > 0.05)
            slow_100 = sum(1 for value in frame_times if value > 0.1)
            chunk_frames = sum(1 for value in received_chunks_per_frame if value)
            max_collect_ms = max(collect_times) * 1000.0 if collect_times else 0.0
            max_scene_ms = max(scene_times) * 1000.0 if scene_times else 0.0
            max_build_ms = max(render_build_times) * 1000.0 if render_build_times else 0.0
            max_draw_ms = max(render_draw_times) * 1000.0 if render_draw_times else 0.0
            print(
                f"Smoke frames={len(frame_times)} avg_frame_ms={average_ms:0.2f} max_frame_ms={max_ms:0.2f} max_frame_index={max_index} steady_max_frame_ms={steady_max_ms:0.2f} "
                f"slow50={slow_50} slow100={slow_100} chunk_frames={chunk_frames} "
                f"max_collect_ms={max_collect_ms:0.2f} max_scene_ms={max_scene_ms:0.2f} max_build_ms={max_build_ms:0.2f} max_draw_ms={max_draw_ms:0.2f} "
                f"cached_chunks={self.environment.cached_chunk_count} pending_chunks={pending_chunks}"
            )

    def scene(self) -> Scene:
        static_scene = self._static_scene_for_position()
        self._frame_scene.objects = static_scene.objects
        self._frame_scene.bulletins = static_scene.bulletins
        self._frame_scene.portals = static_scene.portals
        self._frame_scene.background = self.sky.background_color()
        self._frame_scene.lights = [
            self.sky.sun_light(),
            Lamp(self.walker.position + Vec3(-3.2, 4.2, -2.0), color=(120, 170, 255), intensity=1.1),
        ]
        return self._frame_scene

    def _static_scene_for_position(self) -> Scene:
        key = self._static_scene_key()
        if self._static_scene is not None and self._static_scene_cache_key == key:
            return self._static_scene
        scene = Scene()
        for chunk in self.environment.cached_chunks_around(self.walker.position, self.active_radius):
            visible_count = self._activated_chunk_objects.get(chunk.coord, self._initial_chunk_object_count(chunk))
            scene.add(*chunk.objects[:visible_count])
        if self.sky.stars_enabled and self.sky.night_amount > 0.18:
            scene.add(*self.sky.star_primitives())
        if self.sky.clouds_enabled and self.sky.daylight_amount() > 0.08:
            scene.add(*self.sky.cloud_primitives())
        self.environment.prune_cache_around(self.walker.position, self.active_radius, margin=self.preload_margin + 1)
        self._static_scene = scene
        self._static_scene_cache_key = key
        return scene

    def _static_scene_key(self) -> tuple:
        chunk = self.environment.chunk_coordinate(self.walker.position)
        clouds_visible = self.sky.clouds_enabled and self.sky.daylight_amount() > 0.08
        stars_visible = self.sky.stars_enabled and self.sky.night_amount > 0.18
        return (
            self.config,
            chunk,
            self.active_radius,
            self._stream_version,
            clouds_visible,
            stars_visible,
            self.sky.cloud_seed,
            self.sky.cloud_count,
            self.sky.star_count,
            round(self.sky.radius, 3),
        )

    def _invalidate_static_scene(self) -> None:
        self._static_scene_cache_key = None
        self._static_scene = None

    def _collect_streamed_chunks(self) -> int:
        active_coords = set(self.environment.chunk_coords_around(self.walker.position, self.active_radius))
        received_visible = False
        count = 0
        for chunk in self.chunk_worker.collect_ready():
            self.environment.store_chunk(chunk)
            self._activated_chunk_objects[chunk.coord] = self._initial_chunk_object_count(chunk) if chunk.coord in active_coords else 0
            if chunk.coord in active_coords:
                received_visible = True
            count += 1
        if received_visible:
            self._stream_version += 1
            self._invalidate_static_scene()
        return count

    def _advance_chunk_activation(self) -> bool:
        changed = False
        for chunk in self.environment.cached_chunks_around(self.walker.position, self.active_radius):
            current = self._activated_chunk_objects.get(chunk.coord, self._initial_chunk_object_count(chunk))
            target = len(chunk.objects)
            if current < target:
                self._activated_chunk_objects[chunk.coord] = min(target, current + self.chunk_activation_rate)
                changed = True
        if changed:
            self._stream_version += 1
            self._invalidate_static_scene()
        return changed

    def _initial_chunk_object_count(self, chunk) -> int:
        count = 1
        if chunk.water is not None:
            count += 1
        if chunk.shorelines is not None:
            count += 1
        if chunk.ripples is not None:
            count += 1
        if chunk.grass is not None:
            count += 1
        for obj in chunk.objects[count:]:
            if type(obj).__name__ == "Mesh":
                break
            count += 1
        return min(len(chunk.objects), max(1, count))

    def _schedule_chunk_stream(self) -> None:
        center = self.environment.chunk_coordinate(self.walker.position)
        radius = self.active_radius + self.preload_margin
        target = self.environment.chunk_coords_around(self.walker.position, radius)
        self._stream_target = target
        self._stream_missing_count = sum(1 for coord in target if self.environment.cached_chunk(coord) is None and coord not in self.chunk_worker.futures)
        current = self.environment.cached_chunk(center)
        if current is None:
            self.environment.chunk(center)
            self._stream_version += 1
            self._invalidate_static_scene()
            return
        request_room = max(0, self.max_pending_chunks - self.chunk_worker.pending_count)
        if request_room <= 0:
            return
        requested = 0
        sync_fallback = self.chunk_worker.failed or not self.chunk_worker.enabled
        for coord in target:
            if self.environment.cached_chunk(coord) is not None or coord in self.chunk_worker.futures:
                continue
            if not sync_fallback and self.chunk_worker.request(coord):
                requested += 1
            else:
                self.environment.chunk(coord)
                self._stream_version += 1
                self._invalidate_static_scene()
                requested += 1
            if requested >= request_room:
                break
        self.environment.prune_cache_around(self.walker.position, self.active_radius, margin=self.preload_margin + 1)
        self._activated_chunk_objects = {
            coord: count
            for coord, count in self._activated_chunk_objects.items()
            if self.environment.cached_chunk(coord) is not None or coord in self.chunk_worker.futures
        }

    def on_key_down(self, key: int) -> bool:
        action = self.renderer.handle_menu_key(key)
        if action is not None:
            return self._handle_menu_action(action)
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.add(movement_key)
        elif self.renderer.key_matches(key, "p"):
            self.save_snapshot()
        elif self.renderer.key_matches(key, "r"):
            self._reset_position()
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
        if self.renderer.key_matches(key, "lshift", "rshift"):
            return "sprint"
        if self.renderer.key_matches(key, "lctrl", "rctrl", "c"):
            return "crouch"
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
        if action == "radius_up":
            self.active_radius = min(4, self.active_radius + 1)
            self._invalidate_static_scene()
        elif action == "radius_down":
            self.active_radius = max(1, self.active_radius - 1)
            self._invalidate_static_scene()
        elif action == "look_smoothing_up":
            self.look_smoothing = max(2.0, min(40.0, self.look_smoothing + 2.0))
        elif action == "look_smoothing_down":
            self.look_smoothing = max(2.0, min(40.0, self.look_smoothing - 2.0))
        elif action == "rebuild":
            self.environment.clear_cache()
            self.environment.chunk(self.environment.chunk_coordinate(self.walker.position))
            self.chunk_worker.reset(self.config)
            self._stream_version += 1
            self._invalidate_static_scene()
        elif action == "seed_next":
            self._next_seed()
        elif action == "reset":
            self._reset_position()
            self._invalidate_static_scene()
        elif action == "sky_cycle":
            self.sky.toggle_cycle()
            self._invalidate_static_scene()
        elif action == "sky_time_up":
            self.sky.adjust_time(1.0)
            self._invalidate_static_scene()
        elif action == "sky_time_down":
            self.sky.adjust_time(-1.0)
            self._invalidate_static_scene()
        elif action == "sky_clouds":
            self.sky.toggle_clouds()
            self._invalidate_static_scene()
        elif action == "snapshot":
            self.save_snapshot()
        self._persist_live_settings()
        self._refresh_menu_options()
        return True

    def _next_seed(self) -> None:
        self.config = replace(self.config, seed=self.config.seed + 1)
        self.environment = ProceduralEnvironmentGenerator(self.config)
        self.spawn = find_demo_spawn(self.environment)
        self.environment.chunk(self.environment.chunk_coordinate(self.spawn))
        self.sky.cloud_seed = self.config.seed
        self.chunk_worker.reset(self.config)
        self._stream_version += 1
        self._invalidate_static_scene()
        self._reset_position()

    def _reset_position(self) -> None:
        self.keys.clear()
        self.walker = TerrainWalker(self.spawn)
        self.render_camera = None

    def _persist_live_settings(self) -> None:
        update_live_settings(
            DEMO_SETTINGS,
            {
                "quality": self.args.quality,
                "seed": self.config.seed,
                "active_radius": self.active_radius,
                "look_smoothing": self.look_smoothing,
                "sky_time": round(self.sky.time_of_day, 3),
                "sky_cycle": self.sky.cycle_enabled,
                "sky_clouds": self.sky.clouds_enabled,
                "sky_stars": self.sky.stars_enabled,
                "reflection_bounces": self.args.reflection_bounces,
                "live_backend": self.args.live_backend,
                "still_renderer": self.args.still_renderer,
                "chunk_worker": self.args.chunk_worker,
                "chunk_worker_mode": self.args.chunk_worker_mode,
                "preload_margin": self.preload_margin,
                "max_pending_chunks": self.max_pending_chunks,
                "chunk_activation_rate": self.chunk_activation_rate,
            },
        )

    def save_snapshot(self) -> None:
        output = Path(self.args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        make_still_render_engine(self.args).render(self.scene(), self.walker.camera(), self.sky.settings_for(self.settings)).to_png(output)
        print(f"Wrote {output}")

    def _refresh_menu_options(self) -> None:
        from py_3d.live import LiveMenuOption

        menu = self.renderer.menu
        previous_action = menu.selected_action() if menu.options else "done"
        options = (
            LiveMenuOption("done", "Done"),
            LiveMenuOption("radius_up", "View Distance +", str(self.active_radius), "World"),
            LiveMenuOption("radius_down", "View Distance -", str(self.active_radius), "World"),
            LiveMenuOption("look_smoothing_up", "Look Smoothing +", f"{self.look_smoothing:0.1f}", "Camera"),
            LiveMenuOption("look_smoothing_down", "Look Smoothing -", f"{self.look_smoothing:0.1f}", "Camera"),
            LiveMenuOption("rebuild", "Rebuild", f"{self.environment.cached_chunk_count} cached", "World"),
            LiveMenuOption("seed_next", "Seed", str(self.config.seed), "World"),
            LiveMenuOption("reset", "Reset", "spawn", "World"),
            LiveMenuOption("sky_cycle", "Cycle", "on" if self.sky.cycle_enabled else "off", "Sky"),
            LiveMenuOption("sky_time_up", "Later", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_time_down", "Earlier", f"{self.sky.time_of_day:04.1f}h", "Sky"),
            LiveMenuOption("sky_clouds", "Clouds", "on" if self.sky.clouds_enabled else "off", "Sky"),
            LiveMenuOption("snapshot", "Snapshot", "PNG", "Demo"),
            LiveMenuOption("quit", "Quit demo"),
        )
        menu.options = options
        actions = [option.action for option in options]
        menu.selected_index = actions.index(previous_action) if previous_action in actions else min(menu.selected_index, len(options) - 1)

    def _update_hud(self) -> None:
        chunk = self.environment.chunk_coordinate(self.walker.position)
        water = "WATER" if self.environment.is_water_at(self.walker.position.x, self.walker.position.z) else "HILL"
        self.renderer.hud.set(
            HUDRect((12, 12), (240, 74), (3, 7, 10), alpha=0.55),
            HUDText(
                f"PROCEDURAL HILLS  {water}\nX {self.walker.position.x:0.1f}  Z {self.walker.position.z:0.1f}\nCHUNK {chunk[0]},{chunk[1]}  CACHE {self.environment.cached_chunk_count}  STREAM {self.chunk_worker.pending_count}/{self._stream_missing_count}",
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
            f"py_3d procedural hills - {getattr(self.renderer, 'backend', type(self.renderer).__name__)} - seed {self.config.seed} - {stats.approx_fps:0.1f} fps "
            f"({stats.build_seconds * 1000:0.1f} ms build, {stats.draw_seconds * 1000:0.1f} ms draw)"
        )

    def _smoothed_camera(self, camera: Camera, dt: float) -> Camera:
        if self.render_camera is None:
            self.render_camera = camera
            return camera
        alpha = min(1.0, dt * self.look_smoothing)
        position = self.render_camera.position + (camera.position - self.render_camera.position) * alpha
        target = self.render_camera.target + (camera.target - self.render_camera.target) * alpha
        self.render_camera = Camera(position=position, target=target, fov_degrees=camera.fov_degrees, near=camera.near, far=camera.far)
        return self.render_camera


def make_config(args: argparse.Namespace) -> ProceduralEnvironmentConfig:
    profile = QUALITY_PROFILES[args.quality]
    return ProceduralEnvironmentConfig(
        seed=args.seed,
        chunk_resolution=profile["chunk_resolution"],
        shoreline_detail=profile["shoreline_detail"],
        active_radius=args.active_radius if args.active_radius is not None else profile["active_radius"],
        tree_slots_per_axis=profile["tree_slots_per_axis"],
        tree_density=profile["tree_density"],
        bush_slots_per_axis=profile["bush_slots_per_axis"],
        bush_density=profile["bush_density"],
        rock_slots_per_axis=profile["rock_slots_per_axis"],
        rock_density=profile["rock_density"],
        grass_blades_per_chunk=profile["grass_blades_per_chunk"],
    )


def make_settings(args: argparse.Namespace) -> RenderSettings:
    profile = QUALITY_PROFILES[args.quality]
    return RenderSettings(
        width=args.width,
        height=args.height,
        background=(8, 12, 18),
        ambient=args.ambient,
        gamma=args.gamma,
        light_wrap=args.light_wrap,
        bounce_light=args.bounce_light,
        tone_mapping=True,
        smooth_shading=True,
        reflection_bounces=args.reflection_bounces,
        max_render_distance=args.max_render_distance,
        sphere_segments=profile["sphere_segments"],
        sphere_rings=profile["sphere_rings"],
    )


def find_demo_spawn(environment: ProceduralEnvironmentGenerator) -> Vec3:
    best: tuple[float, Vec3] | None = None
    step = 3.0
    search_indices = range(-16, 17)
    water_offsets = (-6, -4, -2, 0, 2, 4, 6)
    height_cache: dict[tuple[int, int], float] = {}

    def height_at_index(x_index: int, z_index: int) -> float:
        key = (x_index, z_index)
        cached = height_cache.get(key)
        if cached is None:
            cached = environment.height_at(x_index * step, z_index * step)
            height_cache[key] = cached
        return cached

    def water_score_at_index(x_index: int, z_index: int) -> float:
        score = 0.0
        for dz_index in water_offsets:
            for dx_index in water_offsets:
                if dx_index == 0 and dz_index == 0:
                    continue
                if height_at_index(x_index + dx_index, z_index + dz_index) <= environment.config.water_level:
                    distance = ((dx_index * step) ** 2 + (dz_index * step) ** 2) ** 0.5
                    score += 1.0 / max(1.0, distance)
        return score

    def normal_at_index(x_index: int, z_index: int) -> Vec3:
        return Vec3(
            height_at_index(x_index - 1, z_index) - height_at_index(x_index + 1, z_index),
            step * 2.0,
            height_at_index(x_index, z_index - 1) - height_at_index(x_index, z_index + 1),
        ).normalized(Vec3(0.0, 1.0, 0.0))

    for z_index in search_indices:
        for x_index in search_indices:
            x = x_index * step
            z = z_index * step
            y = height_at_index(x_index, z_index)
            if y <= environment.config.water_level + 0.32:
                continue
            normal = normal_at_index(x_index, z_index)
            if normal.y < 0.78:
                continue
            water_score = water_score_at_index(x_index, z_index)
            hill_score = min(2.0, max(0.0, y - environment.config.water_level) * 0.18)
            flat_score = normal.y
            score = water_score * 1.6 + hill_score + flat_score
            if best is None or score > best[0]:
                best = (score, Vec3(x, y, z))
    if best is not None:
        return best[1]
    return Vec3(0.0, max(environment.height_at(0.0, 0.0), environment.config.water_level), 0.0)


def make_preview_camera(environment: ProceduralEnvironmentGenerator, spawn: Vec3) -> Camera:
    camera_x = spawn.x - 14.0
    camera_z = spawn.z - 18.0
    camera_y = max(spawn.y + 7.0, environment.height_at(camera_x, camera_z) + 3.0)
    position = Vec3(camera_x, camera_y, camera_z)
    target = spawn + Vec3(4.0, 1.1, 5.0)
    return Camera(position=position, target=target, fov_degrees=54.0, near=0.08, far=700.0)


def render_still(args: argparse.Namespace) -> Path:
    environment = ProceduralEnvironmentGenerator(make_config(args))
    spawn = find_demo_spawn(environment)
    scene = environment.scene_around(spawn, args.active_radius)
    scene.add_light(Lamp(spawn + Vec3(-4.0, 4.5, -2.5), color=(120, 170, 255), intensity=1.2))
    sky = SkyPrefab(
        time_of_day=args.sky_time,
        cycle_enabled=args.sky_cycle,
        stars_enabled=args.sky_stars,
        clouds_enabled=args.sky_clouds,
        radius=360.0,
        cloud_count=8,
        cloud_seed=args.seed,
    )
    sky.apply(scene)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_still_render_engine(args).render(scene, make_preview_camera(environment, spawn), sky.settings_for(make_settings(args))).to_png(output)
    print(f"Wrote {output}")
    return output


def make_still_render_engine(args: argparse.Namespace) -> RenderEngine:
    renderer = getattr(args, "still_renderer", "auto")
    if renderer == "cpu":
        return RenderEngine()
    return RenderEngine(GPURenderer(allow_cpu_fallback=renderer == "auto"))


def parse_args() -> argparse.Namespace:
    defaults = live_settings_for(DEMO_SETTINGS)
    parser = argparse.ArgumentParser(description="Render or run an indefinitely streamed procedural hill biome.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", dest="mode", action="store_const", const="live", help="Run the live streamed demo (default).")
    mode.add_argument("--still", dest="mode", action="store_const", const="still", help="Render one PNG preview and exit.")
    parser.set_defaults(mode="live")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "procedural_hills.png")
    parser.add_argument("--quality", choices=tuple(QUALITY_PROFILES), default=defaults.get("quality", "balanced"))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 2701)))
    parser.add_argument("--active-radius", type=int, default=defaults.get("active_radius"))
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--menu-blur", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--live-backend", choices=("auto", "gl", "opengl", "gpu", "pixel", "cpu", "native"), default=str(defaults.get("live_backend", "auto")))
    parser.add_argument("--still-renderer", choices=("auto", "cpu", "gpu"), default=str(defaults.get("still_renderer", "auto")))
    parser.add_argument("--chunk-worker", action=argparse.BooleanOptionalAction, default=bool(defaults.get("chunk_worker", True)))
    parser.add_argument("--chunk-worker-mode", choices=("process", "thread"), default=str(defaults.get("chunk_worker_mode", "process")))
    parser.add_argument("--preload-margin", type=int, default=int(defaults.get("preload_margin", 1)))
    parser.add_argument("--max-pending-chunks", type=int, default=int(defaults.get("max_pending_chunks", 1)))
    parser.add_argument("--chunk-activation-rate", type=int, default=int(defaults.get("chunk_activation_rate", 100)))
    parser.add_argument("--smoke-frames", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--look-smoothing", type=float, default=float(defaults.get("look_smoothing", 18.0)))
    parser.add_argument("--sky-time", type=float, default=float(defaults.get("sky_time", 15.2)))
    parser.add_argument("--sky-cycle", action=argparse.BooleanOptionalAction, default=bool(defaults.get("sky_cycle", True)))
    parser.add_argument("--sky-clouds", action=argparse.BooleanOptionalAction, default=bool(defaults.get("sky_clouds", True)))
    parser.add_argument("--sky-stars", action=argparse.BooleanOptionalAction, default=bool(defaults.get("sky_stars", True)))
    parser.add_argument("--ambient", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=1.12)
    parser.add_argument("--light-wrap", type=float, default=0.14)
    parser.add_argument("--bounce-light", type=float, default=0.18)
    parser.add_argument("--reflection-bounces", type=int, default=int(defaults.get("reflection_bounces", 2)))
    parser.add_argument("--max-render-distance", type=float, default=92.0)
    return parser.parse_args()


def should_run_live(args: argparse.Namespace) -> bool:
    return getattr(args, "mode", "live") == "live"


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if should_run_live(args):
        frame_limit = args.smoke_frames if args.smoke_frames > 0 else None
        ProceduralEnvironmentViewer(args).run(frame_limit=frame_limit, auto_walk=frame_limit is not None)
        return
    render_still(args)


if __name__ == "__main__":
    main()
