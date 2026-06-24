"""Indefinite deterministic procedural hill biome demo."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from math import ceil, cos, radians, sin
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
    canonical_player_movement_key,
    ensure_procedural_world_assets,
    next_camera_mode,
    swayed_tree_primitives,
    update_canonical_live_menu,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "renderings-tests" / "live-renders"
WORLD_ASSET_DIR = ROOT / "USER" / "environments" / "procedural_hills" / "world-assets"
DEMO_SETTINGS = "procedural_hills"


QUALITY_PROFILES = {
    "fast": dict(chunk_resolution=12, shoreline_detail=1, active_radius=1, tree_lod_distance_chunks=2, tree_slots_per_axis=3, tree_density=0.16, bush_slots_per_axis=4, bush_density=0.18, rock_slots_per_axis=3, rock_density=0.08, grass_blades_per_chunk=96, leaf_tufts_per_branch=1, sphere_segments=8, sphere_rings=4, cloud_count=3, star_count=18),
    "balanced": dict(chunk_resolution=16, shoreline_detail=1, active_radius=1, tree_lod_distance_chunks=1, tree_slots_per_axis=4, tree_density=0.22, bush_slots_per_axis=5, bush_density=0.25, rock_slots_per_axis=4, rock_density=0.1, grass_blades_per_chunk=170, leaf_tufts_per_branch=2, sphere_segments=10, sphere_rings=5, cloud_count=5, star_count=36),
    "high": dict(chunk_resolution=20, shoreline_detail=2, active_radius=2, tree_lod_distance_chunks=2, tree_slots_per_axis=5, tree_density=0.26, bush_slots_per_axis=6, bush_density=0.3, rock_slots_per_axis=5, rock_density=0.12, grass_blades_per_chunk=260, leaf_tufts_per_branch=3, sphere_segments=12, sphere_rings=6, cloud_count=8, star_count=54),
    "ultra": dict(chunk_resolution=24, shoreline_detail=2, active_radius=3, tree_lod_distance_chunks=3, tree_slots_per_axis=5, tree_density=0.3, bush_slots_per_axis=7, bush_density=0.34, rock_slots_per_axis=5, rock_density=0.14, grass_blades_per_chunk=380, leaf_tufts_per_branch=4, sphere_segments=14, sphere_rings=7, cloud_count=11, star_count=72),
}

PROCEDURAL_LIVE_ACTIONS = {
    "done",
    "next_camera",
    "look_smoothing_down",
    "look_smoothing_up",
    "sky_cycle",
    "sky_time_down",
    "sky_time_up",
    "sky_clouds",
    "sky_stars",
    "radius_down",
    "radius_up",
    "tree_lod_down",
    "tree_lod_up",
    "wind_down",
    "wind_up",
    "rebuild",
    "seed_next",
    "reset",
    "snapshot",
    "quit",
}


@dataclass
class TerrainWalker:
    position: Vec3
    velocity: Vec3 = Vec3(0.0, 0.0, 0.0)
    yaw_degrees: float = 38.0
    pitch_degrees: float = -7.0
    eye_height: float = 1.75
    standing_eye_height: float = 1.75
    crouch_eye_height: float = 1.12
    speed: float = 4.8
    sprint_multiplier: float = 1.45
    crouch_speed_multiplier: float = 0.62
    jump_speed: float = 4.2
    on_ground: bool = False

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
        dt = max(0.0, dt)
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
        horizontal = move * speed
        vertical_velocity = self.velocity.y - 9.81 * dt
        if "space" in keys and self.on_ground:
            vertical_velocity = self.jump_speed
            self.on_ground = False
        next_position = self.position + Vec3(horizontal.x, vertical_velocity, horizontal.z) * dt
        ground = environment.height_at(next_position.x, next_position.z)
        feet_y = max(ground, environment.config.water_level - 0.1)
        if next_position.y <= feet_y:
            self.position = Vec3(next_position.x, feet_y, next_position.z)
            self.velocity = Vec3(horizontal.x, 0.0, horizontal.z)
            self.on_ground = True
        else:
            self.position = Vec3(next_position.x, next_position.y, next_position.z)
            self.velocity = Vec3(horizontal.x, vertical_velocity, horizontal.z)
            self.on_ground = False

    def camera(self, mode: str = "first") -> Camera:
        if mode == "third":
            yaw = radians(self.yaw_degrees)
            aim = self.forward
            right = Vec3(cos(yaw), 0.0, -sin(yaw)).normalized(Vec3(1.0, 0.0, 0.0))
            shoulder = self.position + Vec3(0.0, self.eye_height * 0.82, 0.0)
            camera_position = shoulder - aim * 3.25 + right * 0.54 + Vec3(0.0, 0.16, 0.0)
            return Camera(position=camera_position, target=shoulder + aim * 18.0, fov_degrees=66.0, near=0.04, far=700.0)
        eye = self.position + Vec3(0.0, self.eye_height, 0.0)
        return Camera(position=eye, target=eye + self.forward, fov_degrees=72.0, near=0.04, far=700.0)


class ChunkBuildWorker:
    def __init__(self, config: ProceduralEnvironmentConfig, *, enabled: bool = True, mode: str = "process", max_workers: int = 1) -> None:
        self.config = config
        self.enabled = enabled
        self.mode = mode
        self.max_workers = max(1, int(max_workers))
        self.executor: ProcessPoolExecutor | ThreadPoolExecutor | None = self._make_executor()
        self.futures: dict[tuple[int, int], Future] = {}
        self.failed = False

    @property
    def pending_count(self) -> int:
        return len(self.futures)

    def request(self, coord: tuple[int, int], *, tree_detail: bool = True) -> bool:
        if self.executor is None or self.failed or coord in self.futures:
            return False
        try:
            self.futures[coord] = self.executor.submit(build_environment_chunk, self.config, coord, tree_detail=tree_detail)
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
            return ProcessPoolExecutor(max_workers=self.max_workers)
        return ThreadPoolExecutor(max_workers=self.max_workers)


@dataclass(frozen=True)
class StreamLoadStats:
    active_chunks_loaded: int
    active_chunks_total: int
    stream_chunks_loaded: int
    stream_chunks_total: int
    active_objects_loaded: int
    active_objects_total: int

    @property
    def active_chunk_percent(self) -> float:
        return _percent(self.active_chunks_loaded, self.active_chunks_total)

    @property
    def stream_chunk_percent(self) -> float:
        return _percent(self.stream_chunks_loaded, self.stream_chunks_total)

    @property
    def active_object_percent(self) -> float:
        return _percent(self.active_objects_loaded, self.active_objects_total)


def _percent(value: int, total: int) -> float:
    return 100.0 if total <= 0 else max(0.0, min(100.0, value * 100.0 / total))


class ProceduralEnvironmentViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        from py_3d.live import LiveFlyCamera, LiveMenu, ModernGLLiveRenderer

        self.args = args
        self.config = make_config(args)
        self.world_asset_manifest = ensure_procedural_world_assets(WORLD_ASSET_DIR, self.config)
        self.environment = ProceduralEnvironmentGenerator(self.config)
        self.spawn = find_demo_spawn(self.environment)
        self.environment.chunk(self.environment.chunk_coordinate(self.spawn))
        self.walker = TerrainWalker(self.spawn)
        preview_camera = make_preview_camera(self.environment, self.spawn)
        self.free_camera = LiveFlyCamera.looking_at(
            preview_camera.position,
            preview_camera.target,
            fov_degrees=preview_camera.fov_degrees,
            speed=self.walker.speed,
        )
        self.camera_mode = args.camera_mode
        self.render_camera: Camera | None = None
        self.keys: set[str] = set()
        self.active_radius = args.active_radius if args.active_radius is not None else self.config.active_radius
        self.tree_lod_distance_chunks = max(1, int(args.tree_lod_distance_chunks))
        self.preload_margin = max(0, int(args.preload_margin))
        self.max_pending_chunks = max(1, int(args.max_pending_chunks))
        self.chunk_activation_rate = max(1, int(args.chunk_activation_rate))
        self.look_smoothing = args.look_smoothing
        self.wind_time = 0.0
        self.wind_strength = max(0.0, min(1.35, float(args.wind_strength)))
        self.wind_direction = Vec3(0.72, 0.0, 0.42).normalized(Vec3(1.0, 0.0, 0.0))
        self._swayed_chunk_tree_cache: dict[tuple[tuple[int, int], int, int, int, int, float], tuple[object, ...]] = {}
        self.sky = SkyPrefab(
            time_of_day=args.sky_time,
            cycle_enabled=args.sky_cycle,
            stars_enabled=args.sky_stars,
            clouds_enabled=args.sky_clouds,
            radius=360.0,
            cloud_count=QUALITY_PROFILES[args.quality]["cloud_count"],
            star_count=QUALITY_PROFILES[args.quality]["star_count"],
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
        self._prime_chunk_activation(self.spawn)
        self.chunk_worker = ChunkBuildWorker(self.config, enabled=args.chunk_worker, mode=args.chunk_worker_mode, max_workers=self.max_pending_chunks)
        self.renderer = ModernGLLiveRenderer(args.window_width, args.window_height, title="py_3d procedural hills", vsync=args.vsync, backend=args.live_backend)
        if getattr(self.renderer, "backend", "").lower() == "pixel":
            self.settings = replace(self.settings, smooth_shading=False)
            self.sky.cloud_count = min(self.sky.cloud_count, 3)
            self.sky.star_count = min(self.sky.star_count, 18)
        self.tree_sway_distance_chunks = self._tree_sway_distance_default(args, getattr(self.renderer, "backend", args.live_backend))
        self.renderer.menu = LiveMenu(
            "py_3d procedural hills",
            background_blur=args.menu_blur,
        )
        self._refresh_menu_options()
        self.renderer.set_mouse_captured(True)
        self._last_title_update = 0
        self._schedule_chunk_stream()

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
                        self.on_mouse_motion(self.renderer.event_mouse_rel(event))
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
                        if self.camera_mode != "global":
                            self.walker.look(0.35, 0.0)
                    if self.camera_mode == "global":
                        self.free_camera.move(self.keys, dt)
                    else:
                        self.walker.step(self.keys, dt, self.environment)
                    self.wind_time += dt
                    self.sky.step(dt)
                collect_started = perf_counter()
                received_chunks = self._collect_streamed_chunks()
                collect_elapsed = perf_counter() - collect_started
                self._advance_chunk_activation()
                scene_started = perf_counter()
                scene = self.scene()
                scene_elapsed = perf_counter() - scene_started
                self._update_hud()
                stats = self.renderer.render(scene, self._smoothed_camera(self.camera(), dt), self.sky.settings_for(self.settings))
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
            load_stats = self._stream_load_stats()
            print(
                f"Smoke frames={len(frame_times)} avg_frame_ms={average_ms:0.2f} max_frame_ms={max_ms:0.2f} max_frame_index={max_index} steady_max_frame_ms={steady_max_ms:0.2f} "
                f"slow50={slow_50} slow100={slow_100} chunk_frames={chunk_frames} "
                f"max_collect_ms={max_collect_ms:0.2f} max_scene_ms={max_scene_ms:0.2f} max_build_ms={max_build_ms:0.2f} max_draw_ms={max_draw_ms:0.2f} "
                f"active_chunks={load_stats.active_chunks_loaded}/{load_stats.active_chunks_total} active_chunks_pct={load_stats.active_chunk_percent:0.1f} "
                f"stream_chunks={load_stats.stream_chunks_loaded}/{load_stats.stream_chunks_total} stream_chunks_pct={load_stats.stream_chunk_percent:0.1f} "
                f"active_objects_pct={load_stats.active_object_percent:0.1f} cached_chunks={self.environment.cached_chunk_count} pending_chunks={pending_chunks}"
            )

    def on_mouse_motion(self, relative: tuple[int, int]) -> None:
        dx, dy = relative
        if self.camera_mode == "global":
            self.free_camera.look(dx, dy)
            return
        self.walker.look(dx, dy)

    def camera(self) -> Camera:
        if self.camera_mode == "global":
            return self.free_camera.camera()
        return self.walker.camera(self.camera_mode)

    def _stream_position(self) -> Vec3:
        return self.free_camera.position if self.camera_mode == "global" else self.walker.position

    def scene(self) -> Scene:
        static_scene = self._static_scene_for_position()
        self._frame_scene.objects = list(static_scene.objects)
        self._frame_scene.add(*self._swayed_tree_objects())
        self._frame_scene.bulletins = static_scene.bulletins
        self._frame_scene.portals = static_scene.portals
        self._frame_scene.background = self.sky.background_color()
        light_position = self._stream_position()
        self._frame_scene.lights = [
            self.sky.sun_light(),
            Lamp(light_position + Vec3(-3.2, 4.2, -2.0), color=(134, 178, 255), intensity=3.2),
        ]
        return self._frame_scene

    def _static_scene_for_position(self) -> Scene:
        key = self._static_scene_key()
        if self._static_scene is not None and self._static_scene_cache_key == key:
            return self._static_scene
        scene = Scene()
        stream_position = self._stream_position()
        center_coord = self.environment.chunk_coordinate(stream_position)
        for chunk in self.environment.cached_chunks_around(stream_position, self.active_radius):
            if (self._chunk_uses_distant_tree_lod(chunk.coord, center_coord) or not chunk.tree_detail_loaded) and chunk.distant_objects:
                scene.add(*chunk.distant_objects)
            else:
                base_objects = chunk.base_objects or chunk.objects
                visible_count = min(self._activated_chunk_objects.get(chunk.coord, self._initial_chunk_object_count(chunk)), len(chunk.objects))
                scene.add(*base_objects[: min(visible_count, len(base_objects))])
                if not self._chunk_uses_tree_sway(chunk.coord, center_coord):
                    tree_visible = self._visible_tree_object_count(chunk, visible_count)
                    if tree_visible > 0:
                        scene.add(*(chunk.tree_objects or chunk.objects[len(base_objects) :])[:tree_visible])
        if self.sky.stars_enabled and self.sky.night_amount > 0.55:
            scene.add(*self.sky.star_primitives())
        if self.sky.clouds_enabled and self.sky.daylight_amount() > 0.08:
            scene.add(*self.sky.cloud_primitives())
        self.environment.prune_cache_around(stream_position, self.active_radius, margin=self.preload_margin + 1)
        self._static_scene = scene
        self._static_scene_cache_key = key
        return scene

    def _static_scene_key(self) -> tuple:
        chunk = self.environment.chunk_coordinate(self._stream_position())
        clouds_visible = self.sky.clouds_enabled and self.sky.daylight_amount() > 0.08
        stars_visible = self.sky.stars_enabled and self.sky.night_amount > 0.55
        return (
            self.config,
            chunk,
            self.active_radius,
            self.tree_lod_distance_chunks,
            self.tree_sway_distance_chunks,
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
        stream_position = self._stream_position()
        active_coords = set(self.environment.chunk_coords_around(stream_position, self.active_radius))
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
        for chunk in self.environment.cached_chunks_around(self._stream_position(), self.active_radius):
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
        if chunk.base_objects:
            return min(len(chunk.base_objects), max(1, 5))
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

    def _chunk_uses_distant_tree_lod(self, coord: tuple[int, int], center_coord: tuple[int, int]) -> bool:
        return self._chunk_distance(coord, center_coord) >= self.tree_lod_distance_chunks

    def _chunk_uses_tree_sway(self, coord: tuple[int, int], center_coord: tuple[int, int]) -> bool:
        return self._chunk_distance(coord, center_coord) <= self.tree_sway_distance_chunks

    def _chunk_needs_tree_detail(self, coord: tuple[int, int], center_coord: tuple[int, int]) -> bool:
        return not self._chunk_uses_distant_tree_lod(coord, center_coord)

    def _chunk_ready_for_distance(self, coord: tuple[int, int], center_coord: tuple[int, int]) -> bool:
        chunk = self.environment.cached_chunk(coord)
        if chunk is None:
            return False
        return not self._chunk_needs_tree_detail(coord, center_coord) or chunk.tree_detail_loaded

    def _chunk_has_base(self, coord: tuple[int, int]) -> bool:
        return self.environment.cached_chunk(coord) is not None

    def _chunk_distance(self, coord: tuple[int, int], center_coord: tuple[int, int]) -> int:
        return max(abs(coord[0] - center_coord[0]), abs(coord[1] - center_coord[1]))

    def _tree_sway_distance_default(self, args: argparse.Namespace, backend: str) -> int:
        del backend
        requested = getattr(args, "tree_sway_distance_chunks", None)
        if requested is not None:
            return max(0, min(self.active_radius, int(requested)))
        return 0

    def _visible_tree_object_count(self, chunk, visible_count: int | None = None) -> int:
        base_count = len(chunk.base_objects or ())
        total_visible = self._activated_chunk_objects.get(chunk.coord, self._initial_chunk_object_count(chunk)) if visible_count is None else visible_count
        return max(0, min(len(chunk.tree_objects or chunk.objects[base_count:]), total_visible - base_count))

    def _visible_tree_count(self, chunk, visible_count: int | None = None) -> int:
        if not chunk.trees:
            return 0
        tree_object_count = self._visible_tree_object_count(chunk, visible_count)
        if tree_object_count <= 0:
            return 0
        objects_per_tree = max(1, ceil(len(chunk.tree_objects or ()) / len(chunk.trees)))
        return max(1, min(len(chunk.trees), ceil(tree_object_count / objects_per_tree)))

    def _swayed_tree_objects(self) -> tuple[object, ...]:
        if self.wind_strength <= 0.001:
            return ()
        stream_position = self._stream_position()
        center_coord = self.environment.chunk_coordinate(stream_position)
        time_bucket = int(self.wind_time * 3.0)
        objects: list[object] = []
        for chunk in self.environment.cached_chunks_around(stream_position, self.active_radius):
            if self._chunk_uses_distant_tree_lod(chunk.coord, center_coord) or not self._chunk_uses_tree_sway(chunk.coord, center_coord):
                continue
            visible_tree_count = self._visible_tree_count(chunk)
            if visible_tree_count <= 0:
                continue
            cache_key = (
                chunk.coord,
                visible_tree_count,
                time_bucket,
                self.config.seed,
                self.config.leaf_tufts_per_branch,
                round(self.wind_strength, 3),
            )
            cached = self._swayed_chunk_tree_cache.get(cache_key)
            if cached is None:
                chunk_objects: list[object] = []
                for tree in chunk.trees[:visible_tree_count]:
                    chunk_objects.extend(
                        swayed_tree_primitives(
                            tree,
                            self.config.seed,
                            leaf_tufts_per_branch=self.config.leaf_tufts_per_branch,
                            wind_time=time_bucket / 3.0,
                            wind_strength=self.wind_strength,
                            wind_direction=self.wind_direction,
                        )
                    )
                cached = tuple(chunk_objects)
                if len(self._swayed_chunk_tree_cache) > 96:
                    self._swayed_chunk_tree_cache.clear()
                self._swayed_chunk_tree_cache[cache_key] = cached
            objects.extend(cached)
        return tuple(objects)

    def _schedule_chunk_stream(self) -> None:
        stream_position = self._stream_position()
        center = self.environment.chunk_coordinate(stream_position)
        radius = self.active_radius + self.preload_margin
        target = self.environment.chunk_coords_around(stream_position, radius)
        active_target = set(self.environment.chunk_coords_around(stream_position, self.active_radius))
        self._stream_target = target
        self._stream_missing_count = sum(1 for coord in target if not self._chunk_has_base(coord) and coord not in self.chunk_worker.futures)
        current = self.environment.cached_chunk(center)
        if current is None or not current.tree_detail_loaded:
            if center not in self.chunk_worker.futures:
                sync_fallback = self.chunk_worker.failed or not self.chunk_worker.enabled
                if not sync_fallback and self.chunk_worker.pending_count < self.max_pending_chunks:
                    self.chunk_worker.request(center, tree_detail=True)
                elif sync_fallback:
                    self.environment.store_chunk(build_environment_chunk(self.config, center, tree_detail=True))
                    self._stream_version += 1
                    self._invalidate_static_scene()
            return
        request_room = max(0, self.max_pending_chunks - self.chunk_worker.pending_count)
        if request_room <= 0:
            return
        sync_fallback = self.chunk_worker.failed or not self.chunk_worker.enabled

        active_base_missing = [
            coord
            for coord in target
            if coord in active_target and not self._chunk_has_base(coord) and coord not in self.chunk_worker.futures
        ]
        active_detail_missing = [
            coord
            for coord in target
            if (
                coord in active_target
                and self._chunk_has_base(coord)
                and self._chunk_needs_tree_detail(coord, center)
                and not self.environment.cached_chunk(coord).tree_detail_loaded
                and coord not in self.chunk_worker.futures
            )
        ]
        stream_base_missing = [
            coord
            for coord in target
            if coord not in active_target and not self._chunk_has_base(coord) and coord not in self.chunk_worker.futures
        ]
        stream_detail_missing = [
            coord
            for coord in target
            if (
                coord not in active_target
                and self._chunk_has_base(coord)
                and self._chunk_needs_tree_detail(coord, center)
                and not self.environment.cached_chunk(coord).tree_detail_loaded
                and coord not in self.chunk_worker.futures
            )
        ]
        if active_base_missing:
            candidates = tuple((coord, False) for coord in active_base_missing)
        elif active_detail_missing:
            candidates = tuple((coord, True) for coord in active_detail_missing)
        elif stream_base_missing:
            request_room = min(request_room, 1)
            candidates = tuple((coord, self._chunk_needs_tree_detail(coord, center)) for coord in stream_base_missing)
        else:
            request_room = min(request_room, 1)
            candidates = tuple((coord, True) for coord in stream_detail_missing)

        requested = 0
        for coord, tree_detail in candidates:
            if not sync_fallback and self.chunk_worker.request(coord, tree_detail=tree_detail):
                requested += 1
            else:
                self.environment.store_chunk(build_environment_chunk(self.config, coord, tree_detail=tree_detail))
                self._stream_version += 1
                self._invalidate_static_scene()
                requested += 1
            if requested >= request_room:
                break
        self.environment.prune_cache_around(stream_position, self.active_radius, margin=self.preload_margin + 1)
        self._activated_chunk_objects = {
            coord: count
            for coord, count in self._activated_chunk_objects.items()
            if self.environment.cached_chunk(coord) is not None or coord in self.chunk_worker.futures
        }

    def _stream_load_stats(self) -> StreamLoadStats:
        stream_position = self._stream_position()
        center_coord = self.environment.chunk_coordinate(stream_position)
        active_target = self.environment.chunk_coords_around(stream_position, self.active_radius)
        stream_target = self.environment.chunk_coords_around(stream_position, self.active_radius + self.preload_margin)
        active_loaded = sum(1 for coord in active_target if self._chunk_has_base(coord))
        stream_loaded = sum(1 for coord in stream_target if self._chunk_has_base(coord))
        active_objects_loaded = 0
        active_objects_total = 0
        for coord in active_target:
            chunk = self.environment.cached_chunk(coord)
            if chunk is None:
                active_objects_total += 1
                continue
            if self._chunk_uses_distant_tree_lod(coord, center_coord):
                total = len(chunk.distant_objects or chunk.base_objects or chunk.objects)
                active_objects_total += total
                active_objects_loaded += total
            elif not chunk.tree_detail_loaded:
                loaded = len(chunk.distant_objects or chunk.base_objects or chunk.objects)
                detail_estimate = max(1, len(chunk.trees) * 3)
                active_objects_total += loaded + detail_estimate
                active_objects_loaded += loaded
            else:
                total = len(chunk.objects)
                active_objects_total += total
                active_objects_loaded += min(total, self._activated_chunk_objects.get(coord, self._initial_chunk_object_count(chunk)))
        return StreamLoadStats(
            active_loaded,
            len(active_target),
            stream_loaded,
            len(stream_target),
            active_objects_loaded,
            active_objects_total,
        )

    def on_key_down(self, key: int) -> bool:
        action = self.renderer.handle_menu_key(key)
        if action is not None:
            return self._handle_menu_action(action)
        if self.renderer.key_matches(key, "v"):
            self._cycle_camera_mode()
        elif self.renderer.key_matches(key, "left"):
            self.walker.yaw_degrees -= 5.0
        elif self.renderer.key_matches(key, "right"):
            self.walker.yaw_degrees += 5.0
        elif self.renderer.key_matches(key, "p"):
            self.save_snapshot()
        elif self.renderer.key_matches(key, "r"):
            self._reset_position()
        else:
            movement_key = self._movement_key(key)
            if movement_key is not None:
                self.keys.add(movement_key)
        return True

    def on_key_up(self, key: int) -> None:
        movement_key = self._movement_key(key)
        if movement_key is not None:
            self.keys.discard(movement_key)

    def _movement_key(self, key: int) -> str | None:
        return canonical_player_movement_key(self.renderer, key, camera_mode=self.camera_mode)

    def _cycle_camera_mode(self) -> None:
        self.camera_mode = next_camera_mode(self.camera_mode)
        self.keys.clear()
        self.render_camera = None
        self._invalidate_static_scene()

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
            self.environment.chunk(self.environment.chunk_coordinate(self._stream_position()))
            self._prime_chunk_activation(self._stream_position())
            self.chunk_worker.reset(self.config)
            self._stream_version += 1
            self._invalidate_static_scene()
        elif action == "next_camera":
            self._cycle_camera_mode()
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
        elif action == "sky_stars":
            self.sky.toggle_stars()
            self._invalidate_static_scene()
        elif action == "tree_lod_up":
            self.tree_lod_distance_chunks = min(6, self.tree_lod_distance_chunks + 1)
            self._invalidate_static_scene()
        elif action == "tree_lod_down":
            self.tree_lod_distance_chunks = max(1, self.tree_lod_distance_chunks - 1)
            self._invalidate_static_scene()
        elif action == "wind_up":
            self.wind_strength = min(1.35, self.wind_strength + 0.08)
            self._swayed_chunk_tree_cache.clear()
        elif action == "wind_down":
            self.wind_strength = max(0.0, self.wind_strength - 0.08)
            self._swayed_chunk_tree_cache.clear()
        elif action == "snapshot":
            self.save_snapshot()
        self._persist_live_settings()
        self._refresh_menu_options()
        return True

    def _next_seed(self) -> None:
        self.config = replace(self.config, seed=self.config.seed + 1)
        self.world_asset_manifest = ensure_procedural_world_assets(WORLD_ASSET_DIR, self.config)
        self.environment = ProceduralEnvironmentGenerator(self.config)
        self.spawn = find_demo_spawn(self.environment)
        self.environment.chunk(self.environment.chunk_coordinate(self.spawn))
        self._prime_chunk_activation(self.spawn)
        self.sky.cloud_seed = self.config.seed
        self.chunk_worker.reset(self.config)
        self._stream_version += 1
        self._invalidate_static_scene()
        self._reset_position()

    def _reset_position(self) -> None:
        self.keys.clear()
        self.walker = TerrainWalker(self.spawn)
        preview_camera = make_preview_camera(self.environment, self.spawn)
        self.free_camera = type(self.free_camera).looking_at(
            preview_camera.position,
            preview_camera.target,
            fov_degrees=preview_camera.fov_degrees,
            speed=self.walker.speed,
        )
        self.render_camera = None

    def _prime_chunk_activation(self, position: Vec3) -> None:
        self._activated_chunk_objects.clear()
        chunk = self.environment.cached_chunk(self.environment.chunk_coordinate(position))
        if chunk is not None:
            self._activated_chunk_objects[chunk.coord] = self._initial_chunk_object_count(chunk)

    def _persist_live_settings(self) -> None:
        update_live_settings(
            DEMO_SETTINGS,
            {
                "quality": self.args.quality,
                "seed": self.config.seed,
                "camera_mode": self.camera_mode,
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
                "tree_lod_distance_chunks": self.tree_lod_distance_chunks,
                "preload_margin": self.preload_margin,
                "max_pending_chunks": self.max_pending_chunks,
                "chunk_activation_rate": self.chunk_activation_rate,
                "wind_strength": round(self.wind_strength, 3),
                "tree_sway_distance_chunks": self.tree_sway_distance_chunks,
            },
        )

    def save_snapshot(self) -> None:
        output = Path(self.args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        make_still_render_engine(self.args).render(self.scene(), self.camera(), self.sky.settings_for(self.settings)).to_png(output)
        print(f"Wrote {output}")

    def _refresh_menu_options(self) -> None:
        update_canonical_live_menu(
            self.renderer.menu,
            details={
                "quality_next": self.args.quality,
                "reflections_down": self.args.reflection_bounces,
                "reflections_up": self.args.reflection_bounces,
                "radius_down": self.active_radius,
                "radius_up": self.active_radius,
                "tree_lod_down": self.tree_lod_distance_chunks,
                "tree_lod_up": self.tree_lod_distance_chunks,
                "wind_down": f"{self.wind_strength:0.2f}x",
                "wind_up": f"{self.wind_strength:0.2f}x",
                "rebuild": f"{self.environment.cached_chunk_count} cached",
                "seed_next": self.config.seed,
                "next_camera": self.camera_mode,
                "look_smoothing_down": f"{self.look_smoothing:0.1f}",
                "look_smoothing_up": f"{self.look_smoothing:0.1f}",
                "sky_cycle": "on" if self.sky.cycle_enabled else "off",
                "sky_time_down": f"{self.sky.time_of_day:04.1f}h",
                "sky_time_up": f"{self.sky.time_of_day:04.1f}h",
                "sky_clouds": "on" if self.sky.clouds_enabled else "off",
                "sky_stars": "on" if self.sky.stars_enabled else "off",
                "reset": "spawn",
                "snapshot": "PNG",
            },
            enabled_actions=PROCEDURAL_LIVE_ACTIONS,
        )

    def _update_hud(self) -> None:
        stream_position = self._stream_position()
        chunk = self.environment.chunk_coordinate(stream_position)
        water = "WATER" if self.environment.is_water_at(self.walker.position.x, self.walker.position.z) else "HILL"
        move_state = "FREE" if self.camera_mode == "global" else ("CROUCH" if "crouch" in self.keys else ("SPRINT" if "sprint" in self.keys else "WALK"))
        load_stats = self._stream_load_stats()
        self.renderer.hud.set(
            HUDRect((12, 12), (334, 98), (3, 7, 10), alpha=0.55),
            HUDText(
                f"PROCEDURAL HILLS  {self.camera_mode.upper()}  {move_state}\n"
                f"X {stream_position.x:0.1f}  Z {stream_position.z:0.1f}  {water}\n"
                f"LOAD {load_stats.active_chunk_percent:0.0f}% CHUNKS  {load_stats.active_object_percent:0.0f}% DETAIL  {load_stats.stream_chunk_percent:0.0f}% PRELOAD\n"
                f"CHUNK {chunk[0]},{chunk[1]}  CACHE {self.environment.cached_chunk_count}  STREAM {self.chunk_worker.pending_count}/{self._stream_missing_count}",
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
        if self.render_camera is None or self.camera_mode == "global":
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
        leaf_tufts_per_branch=profile["leaf_tufts_per_branch"],
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
    camera_x = spawn.x + 14.0
    camera_z = spawn.z - 2.0
    camera_y = max(spawn.y + 2.8, environment.height_at(camera_x, camera_z) + 1.6)
    position = Vec3(camera_x, camera_y, camera_z)
    target = spawn + Vec3(-9.0, 1.2, 1.0)
    return Camera(position=position, target=target, fov_degrees=70.0, near=0.04, far=700.0)


def render_still(args: argparse.Namespace) -> Path:
    config = make_config(args)
    ensure_procedural_world_assets(WORLD_ASSET_DIR, config)
    environment = ProceduralEnvironmentGenerator(config)
    spawn = find_demo_spawn(environment)
    scene = scene_around_with_lod(environment, spawn, args.active_radius, args.tree_lod_distance_chunks)
    scene.add_light(Lamp(spawn + Vec3(-4.0, 4.5, -2.5), color=(134, 178, 255), intensity=3.6))
    sky = SkyPrefab(
        time_of_day=args.sky_time,
        cycle_enabled=args.sky_cycle,
        stars_enabled=args.sky_stars,
        clouds_enabled=args.sky_clouds,
        radius=360.0,
        cloud_count=QUALITY_PROFILES[args.quality]["cloud_count"],
        star_count=QUALITY_PROFILES[args.quality]["star_count"],
        cloud_seed=args.seed,
    )
    sky.apply(scene)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_still_render_engine(args).render(scene, make_preview_camera(environment, spawn), sky.settings_for(make_settings(args))).to_png(output)
    print(f"Wrote {output}")
    return output


def scene_around_with_lod(environment: ProceduralEnvironmentGenerator, position: Vec3, radius: int | None, tree_lod_distance_chunks: int) -> Scene:
    scene = Scene()
    center_coord = environment.chunk_coordinate(position)
    lod_distance = max(1, int(tree_lod_distance_chunks))
    for coord in environment.chunk_coords_around(position, radius):
        distance = max(abs(coord[0] - center_coord[0]), abs(coord[1] - center_coord[1]))
        if distance >= lod_distance:
            chunk = environment.cached_chunk(coord)
            if chunk is None or chunk.tree_detail_loaded:
                chunk = build_environment_chunk(environment.config, coord, tree_detail=False)
                environment.store_chunk(chunk)
        else:
            chunk = environment.chunk(coord)
        distance = max(abs(chunk.coord[0] - center_coord[0]), abs(chunk.coord[1] - center_coord[1]))
        if distance >= lod_distance and chunk.distant_objects:
            scene.add(*chunk.distant_objects)
        else:
            scene.add(*chunk.objects)
    return scene


def make_still_render_engine(args: argparse.Namespace) -> RenderEngine:
    renderer = getattr(args, "still_renderer", "auto")
    if renderer == "cpu":
        return RenderEngine()
    return RenderEngine(GPURenderer(allow_cpu_fallback=renderer == "auto"))


def parse_args() -> argparse.Namespace:
    defaults = live_settings_for(DEMO_SETTINGS)
    quality_default = str(defaults.get("quality", "balanced"))
    if quality_default not in QUALITY_PROFILES:
        quality_default = "balanced"
    preview_parser = argparse.ArgumentParser(add_help=False)
    preview_parser.add_argument("--quality", choices=tuple(QUALITY_PROFILES), default=quality_default)
    preview_args, _ = preview_parser.parse_known_args()
    selected_quality = preview_args.quality if preview_args.quality in QUALITY_PROFILES else quality_default
    active_radius_default = defaults.get("active_radius")
    if active_radius_default is not None:
        active_radius_default = min(int(active_radius_default), QUALITY_PROFILES[selected_quality]["active_radius"])
    tree_lod_default = defaults.get("tree_lod_distance_chunks", QUALITY_PROFILES[selected_quality]["tree_lod_distance_chunks"])
    tree_lod_default = min(max(1, int(tree_lod_default)), QUALITY_PROFILES[selected_quality]["tree_lod_distance_chunks"])
    chunk_activation_default = min(max(1, int(defaults.get("chunk_activation_rate", 8))), 12)
    max_pending_default = min(max(1, int(defaults.get("max_pending_chunks", 1))), 4)
    sky_time_default = float(defaults.get("sky_time", 14.2))
    if sky_time_default < 10.5 or sky_time_default > 15.8:
        sky_time_default = 14.2
    chunk_worker_mode_default = str(defaults.get("chunk_worker_mode", "thread"))
    if chunk_worker_mode_default == "process":
        chunk_worker_mode_default = "thread"
    parser = argparse.ArgumentParser(description="Render or run an indefinitely streamed procedural hill biome.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", dest="mode", action="store_const", const="live", help="Run the live streamed demo (default).")
    mode.add_argument("--still", dest="mode", action="store_const", const="still", help="Render one PNG preview and exit.")
    parser.set_defaults(mode="live")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "procedural_hills.png")
    parser.add_argument("--quality", choices=tuple(QUALITY_PROFILES), default=quality_default)
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 2701)))
    camera_mode_default = str(defaults.get("camera_mode", "first"))
    if camera_mode_default not in {"global", "third", "first"}:
        camera_mode_default = "first"
    parser.add_argument("--camera-mode", choices=("global", "third", "first"), default=camera_mode_default)
    parser.add_argument("--active-radius", type=int, default=active_radius_default)
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
    parser.add_argument("--chunk-worker-mode", choices=("process", "thread"), default=chunk_worker_mode_default)
    parser.add_argument("--tree-lod-distance-chunks", type=int, default=tree_lod_default)
    parser.add_argument("--tree-sway-distance-chunks", type=int, default=defaults.get("tree_sway_distance_chunks"), help=argparse.SUPPRESS)
    parser.add_argument("--preload-margin", type=int, default=int(defaults.get("preload_margin", 1)))
    parser.add_argument("--max-pending-chunks", type=int, default=max_pending_default)
    parser.add_argument("--chunk-activation-rate", type=int, default=chunk_activation_default)
    parser.add_argument("--smoke-frames", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--look-smoothing", type=float, default=float(defaults.get("look_smoothing", 18.0)))
    wind_strength_default = float(defaults.get("wind_strength", 0.0))
    parser.add_argument("--wind-strength", type=float, default=wind_strength_default, help=argparse.SUPPRESS)
    parser.add_argument("--sky-time", type=float, default=sky_time_default)
    parser.add_argument("--sky-cycle", action=argparse.BooleanOptionalAction, default=bool(defaults.get("sky_cycle", True)))
    parser.add_argument("--sky-clouds", action=argparse.BooleanOptionalAction, default=bool(defaults.get("sky_clouds", True)))
    parser.add_argument("--sky-stars", action=argparse.BooleanOptionalAction, default=bool(defaults.get("sky_stars", True)))
    parser.add_argument("--ambient", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=1.12)
    parser.add_argument("--light-wrap", type=float, default=0.66)
    parser.add_argument("--bounce-light", type=float, default=0.68)
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
