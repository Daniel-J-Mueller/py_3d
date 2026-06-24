"""Deterministic infinite procedural environment generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from math import cos, floor, sin, sqrt, tau
from pathlib import Path
from typing import Any, Callable, Iterable

from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .noise import FractalNoise3D
from .primitives import LineSet3, Mesh, Triangle
from .scene import Scene


PROCEDURAL_WORLD_ASSET_FORMAT = "py_3d.procedural_world_assets.v1"
GRASS_PALETTE = (
    Material(color=(76, 126, 66), roughness=0.76, fuzziness=0.1, specular=0.02),
    Material(color=(84, 138, 70), roughness=0.74, fuzziness=0.1, specular=0.02),
    Material(color=(92, 148, 76), roughness=0.74, fuzziness=0.1, specular=0.02),
)
GRASS_TUFT_PALETTE = (
    Material(color=(83, 130, 55), emission=(2, 3, 1), roughness=0.84, fuzziness=0.26, specular=0.012, light_transmission=0.16),
    Material(color=(104, 151, 62), emission=(2, 4, 1), roughness=0.84, fuzziness=0.27, specular=0.012, light_transmission=0.18),
    Material(color=(132, 160, 66), emission=(3, 4, 1), roughness=0.86, fuzziness=0.28, specular=0.012, light_transmission=0.18),
    Material(color=(66, 112, 58), emission=(1, 3, 1), roughness=0.86, fuzziness=0.24, specular=0.01, light_transmission=0.14),
)
MEADOW_PALETTE = (
    Material(color=(94, 143, 77), roughness=0.76, fuzziness=0.11, specular=0.02),
    Material(color=(103, 153, 82), roughness=0.76, fuzziness=0.11, specular=0.02),
    Material(color=(112, 161, 88), roughness=0.78, fuzziness=0.12, specular=0.02),
)
ROCK_PALETTE = (
    Material(color=(91, 97, 90), roughness=0.84, fuzziness=0.06, specular=0.03),
    Material(color=(101, 106, 98), roughness=0.84, fuzziness=0.06, specular=0.03),
    Material(color=(111, 112, 103), roughness=0.84, fuzziness=0.06, specular=0.03),
)
SHORE_PALETTE = (
    Material(color=(101, 105, 76), roughness=0.84, fuzziness=0.13, specular=0.02),
    Material(color=(110, 112, 80), roughness=0.84, fuzziness=0.13, specular=0.02),
)
WATER_SHALLOW = Material(
    color=(78, 157, 179),
    roughness=0.035,
    specular=0.78,
    shininess=155.0,
    reflectivity=0.38,
    light_transmission=0.58,
)
WATER = Material(
    color=(49, 124, 175),
    roughness=0.025,
    specular=0.88,
    shininess=210.0,
    reflectivity=0.5,
    light_transmission=0.5,
)
WATER_DEEP = Material(
    color=(34, 91, 146),
    roughness=0.02,
    specular=0.92,
    shininess=240.0,
    reflectivity=0.56,
    light_transmission=0.42,
)
FOAM = Material(color=(192, 214, 198), emission=(14, 18, 18), roughness=0.18, specular=0.24)
RIPPLE = Material(color=(138, 190, 210), emission=(8, 12, 14), roughness=0.18, specular=0.32)
TRUNK = Material(color=(84, 61, 41), roughness=0.72, fuzziness=0.07, specular=0.03)
BRANCH = Material(color=(74, 54, 37), roughness=0.76, fuzziness=0.07, specular=0.03)
TWIG = Material(color=(60, 45, 33), roughness=0.82, fuzziness=0.08, specular=0.02)
MAPLE_LEAF_PALETTE = (
    Material(color=(58, 135, 63), emission=(2, 4, 2), diffuse=0.94, roughness=0.78, fuzziness=0.16, specular=0.025, light_transmission=0.4),
    Material(color=(72, 154, 72), emission=(2, 5, 2), diffuse=0.94, roughness=0.78, fuzziness=0.17, specular=0.025, light_transmission=0.44),
    Material(color=(92, 171, 78), emission=(3, 6, 2), diffuse=0.92, roughness=0.8, fuzziness=0.18, specular=0.025, light_transmission=0.46),
    Material(color=(107, 151, 61), emission=(3, 5, 2), diffuse=0.92, roughness=0.8, fuzziness=0.18, specular=0.025, light_transmission=0.4),
)
CANOPY_HAZE_PALETTE = (
    Material(color=(56, 118, 62), emission=(3, 6, 3), diffuse=0.86, roughness=0.92, fuzziness=0.38, specular=0.004, light_transmission=0.32),
    Material(color=(70, 137, 68), emission=(3, 7, 3), diffuse=0.86, roughness=0.92, fuzziness=0.38, specular=0.004, light_transmission=0.34),
    Material(color=(86, 152, 72), emission=(3, 7, 3), diffuse=0.84, roughness=0.94, fuzziness=0.4, specular=0.004, light_transmission=0.34),
)
BUSH_PALETTE = (
    Material(color=(46, 103, 58), emission=(3, 7, 3), diffuse=0.86, roughness=0.84, fuzziness=0.18, specular=0.015, light_transmission=0.18),
    Material(color=(70, 126, 66), emission=(4, 8, 3), diffuse=0.84, roughness=0.84, fuzziness=0.18, specular=0.015, light_transmission=0.2),
)


@dataclass(frozen=True)
class ProceduralEnvironmentConfig:
    """Settings for deterministic world generation.

    Every visible feature is derived from the seed and world coordinates. A
    chunk can be evicted from memory and later rebuilt without changing its
    terrain, trees, or water.
    """

    seed: int = 2701
    chunk_size: float = 16.0
    chunk_resolution: int = 12
    active_radius: int = 2
    water_level: float = 0.35
    hill_scale: float = 4.2
    detail_scale: float = 0.62
    tree_slots_per_axis: int = 4
    tree_density: float = 0.28
    bush_slots_per_axis: int = 5
    bush_density: float = 0.24
    rock_slots_per_axis: int = 4
    rock_density: float = 0.12
    grass_blades_per_chunk: int = 180
    shoreline_detail: int = 2
    leaf_tufts_per_branch: int = 3

    def __post_init__(self) -> None:
        if self.chunk_size <= 0.0:
            raise ValueError("chunk size must be positive")
        if self.chunk_resolution <= 0:
            raise ValueError("chunk resolution must be positive")
        if self.active_radius < 0:
            raise ValueError("active radius must be non-negative")
        if self.tree_slots_per_axis <= 0:
            raise ValueError("tree slots per axis must be positive")
        if self.bush_slots_per_axis <= 0:
            raise ValueError("bush slots per axis must be positive")
        if self.rock_slots_per_axis <= 0:
            raise ValueError("rock slots per axis must be positive")
        object.__setattr__(self, "tree_density", clamp(float(self.tree_density), 0.0, 1.0))
        object.__setattr__(self, "bush_density", clamp(float(self.bush_density), 0.0, 1.0))
        object.__setattr__(self, "rock_density", clamp(float(self.rock_density), 0.0, 1.0))
        object.__setattr__(self, "grass_blades_per_chunk", max(0, int(self.grass_blades_per_chunk)))
        object.__setattr__(self, "shoreline_detail", max(1, int(self.shoreline_detail)))
        object.__setattr__(self, "leaf_tufts_per_branch", max(1, int(self.leaf_tufts_per_branch)))


@dataclass(frozen=True)
class WaterSource:
    center: Vec3
    radius: float
    level: float


@dataclass(frozen=True)
class TreeInstance:
    position: Vec3
    trunk_height: float
    trunk_radius: float
    crown_radius: float
    yaw: float = 0.0
    lean: Vec3 = Vec3(0.0, 0.0, 0.0)
    branch_count: int = 5
    species: int = 0
    material_variant: int = 0


@dataclass(frozen=True)
class BushInstance:
    position: Vec3
    radius: float
    height: float
    variant: int = 0


@dataclass(frozen=True)
class RockInstance:
    position: Vec3
    radius: float
    height: float
    yaw: float = 0.0
    variant: int = 0


@dataclass(frozen=True)
class EnvironmentChunk:
    coord: tuple[int, int]
    terrain: Mesh
    water: Mesh | None
    trees: tuple[TreeInstance, ...]
    bushes: tuple[BushInstance, ...]
    rocks: tuple[RockInstance, ...]
    grass: Mesh | None
    shorelines: LineSet3 | None
    ripples: LineSet3 | None
    water_sources: tuple[WaterSource, ...]
    objects: tuple[object, ...]
    distant_objects: tuple[object, ...] = ()
    base_objects: tuple[object, ...] = ()
    tree_objects: tuple[object, ...] = ()
    tree_detail_loaded: bool = True


class ProceduralEnvironmentGenerator:
    """Generate and stream stable chunks for an unbounded hill biome."""

    def __init__(self, config: ProceduralEnvironmentConfig | None = None) -> None:
        self.config = config or ProceduralEnvironmentConfig()
        seed = self.config.seed
        self._hill_noise = FractalNoise3D(seed + 11, octaves=4, lacunarity=2.0, gain=0.54)
        self._ridge_noise = FractalNoise3D(seed + 29, octaves=3, lacunarity=2.1, gain=0.48)
        self._detail_noise = FractalNoise3D(seed + 47, octaves=3, lacunarity=2.35, gain=0.45)
        self._basin_noise = FractalNoise3D(seed + 71, octaves=3, lacunarity=1.9, gain=0.52)
        self._foliage_noise = FractalNoise3D(seed + 89, octaves=3, lacunarity=2.2, gain=0.5)
        self._water_noise = FractalNoise3D(seed + 109, octaves=2, lacunarity=2.0, gain=0.45)
        self._chunk_cache: dict[tuple[int, int], EnvironmentChunk] = {}

    @property
    def cached_chunk_count(self) -> int:
        return len(self._chunk_cache)

    def clear_cache(self) -> None:
        self._chunk_cache.clear()

    def prune_cache(self, keep: Iterable[tuple[int, int]]) -> None:
        keep_set = set(keep)
        for coord in tuple(self._chunk_cache):
            if coord not in keep_set:
                del self._chunk_cache[coord]

    def prune_cache_around(self, position: Vec3 | tuple[float, float, float], radius: int | None = None, *, margin: int = 1) -> None:
        center_x, center_z = self.chunk_coordinate(position)
        active_radius = self.config.active_radius if radius is None else max(0, int(radius))
        keep_radius = active_radius + max(0, int(margin))
        keep = (
            (center_x + dx, center_z + dz)
            for dz in range(-keep_radius, keep_radius + 1)
            for dx in range(-keep_radius, keep_radius + 1)
        )
        self.prune_cache(keep)

    def chunk_coordinate(self, position: Vec3 | tuple[float, float, float] | tuple[float, float]) -> tuple[int, int]:
        if isinstance(position, Vec3):
            x = position.x
            z = position.z
        elif len(position) == 2:
            x = float(position[0])
            z = float(position[1])
        else:
            value = as_vec3(position)
            x = value.x
            z = value.z
        size = self.config.chunk_size
        return floor(x / size), floor(z / size)

    def height_at(self, x: float, z: float) -> float:
        """Sample deterministic terrain height at a world-space x/z point."""

        broad = self._hill_noise.sample_xyz(x * 0.034, 0.0, z * 0.034)
        ridge = abs(self._ridge_noise.sample_xyz(x * 0.053, 2.0, z * 0.053) * 2.0 - 1.0)
        detail = self._detail_noise.sample_xyz(x * 0.15, 8.0, z * 0.15) - 0.5
        basin = self._basin_noise.sample_xyz(x * 0.045, 16.0, z * 0.045)
        basin_drop = max(0.0, 0.42 - basin) * 5.2
        rolling_hills = (broad - 0.32) * self.config.hill_scale
        ridge_lift = (1.0 - ridge) * 1.1
        return rolling_hills + ridge_lift + detail * self.config.detail_scale - basin_drop

    def normal_at(self, x: float, z: float, *, step: float = 0.35) -> Vec3:
        left = self.height_at(x - step, z)
        right = self.height_at(x + step, z)
        back = self.height_at(x, z - step)
        front = self.height_at(x, z + step)
        return Vec3(left - right, step * 2.0, back - front).normalized(Vec3(0.0, 1.0, 0.0))

    def is_water_at(self, x: float, z: float) -> bool:
        return self.height_at(x, z) <= self.config.water_level

    def chunk(self, coord: tuple[int, int]) -> EnvironmentChunk:
        cached = self._chunk_cache.get(coord)
        if cached is not None:
            return cached
        chunk = self._build_chunk(coord)
        self._chunk_cache[coord] = chunk
        return chunk

    def store_chunk(self, chunk: EnvironmentChunk) -> None:
        self._chunk_cache[chunk.coord] = chunk

    def cached_chunk(self, coord: tuple[int, int]) -> EnvironmentChunk | None:
        return self._chunk_cache.get(coord)

    def chunk_coords_around(self, position: Vec3 | tuple[float, float, float], radius: int | None = None) -> tuple[tuple[int, int], ...]:
        center_x, center_z = self.chunk_coordinate(position)
        active_radius = self.config.active_radius if radius is None else max(0, int(radius))
        coords = [
            (center_x + dx, center_z + dz)
            for dz in range(-active_radius, active_radius + 1)
            for dx in range(-active_radius, active_radius + 1)
        ]
        coords.sort(key=lambda item: ((item[0] - center_x) ** 2 + (item[1] - center_z) ** 2, item[1], item[0]))
        return tuple(coords)

    def cached_chunks_around(self, position: Vec3 | tuple[float, float, float], radius: int | None = None) -> tuple[EnvironmentChunk, ...]:
        chunks: list[EnvironmentChunk] = []
        for coord in self.chunk_coords_around(position, radius):
            chunk = self.cached_chunk(coord)
            if chunk is not None:
                chunks.append(chunk)
        return tuple(chunks)

    def chunks_around(self, position: Vec3 | tuple[float, float, float], radius: int | None = None) -> tuple[EnvironmentChunk, ...]:
        return tuple(self.chunk(coord) for coord in self.chunk_coords_around(position, radius))

    def scene_around(self, position: Vec3 | tuple[float, float, float], radius: int | None = None) -> Scene:
        scene = Scene()
        for chunk in self.chunks_around(position, radius):
            scene.add(*chunk.objects)
        return scene

    def _build_chunk(self, coord: tuple[int, int], *, tree_detail: bool = True) -> EnvironmentChunk:
        height_at = self._chunk_height_sampler()
        normal_at = self._chunk_normal_sampler(height_at)
        terrain = self._terrain_mesh(coord, height_at=height_at)
        water, sources, shorelines, ripples = self._water_mesh(coord, height_at=height_at)
        trees = self._trees(coord, height_at=height_at, normal_at=normal_at)
        bushes = self._bushes(coord, height_at=height_at, normal_at=normal_at)
        rocks = self._rocks(coord, height_at=height_at, normal_at=normal_at)
        grass = self._grass(
            coord,
            height_at=height_at,
            normal_at=normal_at,
            density_scale=1.0 if tree_detail else 0.34,
            distant=not tree_detail,
        )
        base_objects: list[object] = [terrain]
        if water is not None:
            base_objects.append(water)
        if shorelines is not None:
            base_objects.append(shorelines)
        if ripples is not None:
            base_objects.append(ripples)
        if grass is not None:
            base_objects.append(grass)
        for rock in rocks:
            base_objects.extend(_rock_primitives(rock))
        for bush in bushes:
            base_objects.extend(_bush_primitives(bush, self.config.seed))
        tree_objects: list[object] = []
        if tree_detail:
            for tree in trees:
                tree_objects.extend(_tree_primitives(tree, self.config.seed, leaf_tufts_per_branch=self.config.leaf_tufts_per_branch))
        objects = [*base_objects, *tree_objects]

        distant_objects: list[object] = [terrain]
        if water is not None:
            distant_objects.append(water)
        if shorelines is not None:
            distant_objects.append(shorelines)
        if ripples is not None:
            distant_objects.append(ripples)
        if grass is not None and not tree_detail:
            distant_objects.append(grass)
        for rock in rocks:
            distant_objects.extend(_rock_primitives(rock))
        for bush in bushes:
            distant_objects.extend(_bush_primitives(bush, self.config.seed))
        for tree in trees:
            distant_objects.extend(_tree_distant_primitives(tree, self.config.seed))
        return EnvironmentChunk(
            coord,
            terrain,
            water,
            trees,
            bushes,
            rocks,
            grass,
            shorelines,
            ripples,
            sources,
            tuple(objects),
            tuple(distant_objects),
            tuple(base_objects),
            tuple(tree_objects),
            bool(tree_detail),
        )

    def _chunk_height_sampler(self) -> Callable[[float, float], float]:
        cache: dict[tuple[float, float], float] = {}

        def sample(x: float, z: float) -> float:
            key = (round(x, 6), round(z, 6))
            cached = cache.get(key)
            if cached is None:
                cached = self.height_at(x, z)
                cache[key] = cached
            return cached

        return sample

    def _chunk_normal_sampler(self, height_at: Callable[[float, float], float]) -> Callable[[float, float], Vec3]:
        cache: dict[tuple[float, float], Vec3] = {}

        def sample(x: float, z: float) -> Vec3:
            key = (round(x, 6), round(z, 6))
            cached = cache.get(key)
            if cached is not None:
                return cached
            step = 0.35
            cached = Vec3(
                height_at(x - step, z) - height_at(x + step, z),
                step * 2.0,
                height_at(x, z - step) - height_at(x, z + step),
            ).normalized(Vec3(0.0, 1.0, 0.0))
            cache[key] = cached
            return cached

        return sample

    def _terrain_mesh(self, coord: tuple[int, int], *, height_at: Callable[[float, float], float] | None = None) -> Mesh:
        height = height_at or self.height_at
        cx, cz = coord
        size = self.config.chunk_size
        resolution = self.config.chunk_resolution
        step = size / resolution
        x0 = cx * size
        z0 = cz * size
        vertices: list[list[Vec3]] = []
        normals: list[list[Vec3]] = []
        for z_index in range(resolution + 1):
            row: list[Vec3] = []
            normal_row: list[Vec3] = []
            z = z0 + z_index * step
            for x_index in range(resolution + 1):
                x = x0 + x_index * step
                sample_x, sample_z = self._terrain_vertex_xz(coord, x_index, z_index, x, z, step)
                row.append(Vec3(sample_x, height(sample_x, sample_z), sample_z))
            vertices.append(row)
        for z_index in range(resolution + 1):
            normal_row: list[Vec3] = []
            for x_index in range(resolution + 1):
                left = vertices[z_index][max(0, x_index - 1)]
                right = vertices[z_index][min(resolution, x_index + 1)]
                back = vertices[max(0, z_index - 1)][x_index]
                front = vertices[min(resolution, z_index + 1)][x_index]
                span_x = max(step, right.x - left.x)
                span_z = max(step, front.z - back.z)
                normal_row.append(Vec3(left.y - right.y, (span_x + span_z) * 0.5, back.y - front.y).normalized(Vec3(0.0, 1.0, 0.0)))
            normals.append(normal_row)

        triangles: list[Triangle] = []
        for z_index in range(resolution):
            for x_index in range(resolution):
                top_left = vertices[z_index][x_index]
                top_right = vertices[z_index][x_index + 1]
                bottom_left = vertices[z_index + 1][x_index]
                bottom_right = vertices[z_index + 1][x_index + 1]
                center = (top_left + top_right + bottom_left + bottom_right) / 4.0
                normal = (normals[z_index][x_index] + normals[z_index][x_index + 1] + normals[z_index + 1][x_index] + normals[z_index + 1][x_index + 1]).normalized(Vec3(0.0, 1.0, 0.0))
                material = self._terrain_material(center, normal)
                if _stable_uint(self.config.seed, cx, cz, x_index, z_index, 401) & 1:
                    triangles.append(
                        Triangle(
                            top_left,
                            bottom_left,
                            top_right,
                            material,
                            normal_a=normals[z_index][x_index],
                            normal_b=normals[z_index + 1][x_index],
                            normal_c=normals[z_index][x_index + 1],
                        )
                    )
                    triangles.append(
                        Triangle(
                            top_right,
                            bottom_left,
                            bottom_right,
                            material,
                            normal_a=normals[z_index][x_index + 1],
                            normal_b=normals[z_index + 1][x_index],
                            normal_c=normals[z_index + 1][x_index + 1],
                        )
                    )
                else:
                    triangles.append(
                        Triangle(
                            top_left,
                            bottom_left,
                            bottom_right,
                            material,
                            normal_a=normals[z_index][x_index],
                            normal_b=normals[z_index + 1][x_index],
                            normal_c=normals[z_index + 1][x_index + 1],
                        )
                    )
                    triangles.append(
                        Triangle(
                            top_left,
                            bottom_right,
                            top_right,
                            material,
                            normal_a=normals[z_index][x_index],
                            normal_b=normals[z_index + 1][x_index + 1],
                            normal_c=normals[z_index][x_index + 1],
                        )
                    )
        return Mesh(triangles)

    def _terrain_vertex_xz(self, coord: tuple[int, int], x_index: int, z_index: int, x: float, z: float, step: float) -> tuple[float, float]:
        if x_index in {0, self.config.chunk_resolution} or z_index in {0, self.config.chunk_resolution}:
            return x, z
        cx, cz = coord
        amount = step * 0.24
        jitter_x = (_stable_unit(self.config.seed, cx, cz, x_index, z_index, 503) - 0.5) * amount
        jitter_z = (_stable_unit(self.config.seed, cx, cz, x_index, z_index, 509) - 0.5) * amount
        return x + jitter_x, z + jitter_z

    def _terrain_material(self, point: Vec3, normal: Vec3 | None = None) -> Material:
        surface_normal = normal or self.normal_at(point.x, point.z)
        variation = self._detail_noise.sample_xyz(point.x * 0.045, 27.0, point.z * 0.045)
        if point.y <= self.config.water_level + 0.18:
            return SHORE_PALETTE[0 if variation < 0.55 else 1]
        if surface_normal.y < 0.78 or point.y > self.config.water_level + 2.75:
            return ROCK_PALETTE[min(len(ROCK_PALETTE) - 1, int(variation * len(ROCK_PALETTE)))]
        meadow = self._detail_noise.sample_xyz(point.x * 0.07, 21.0, point.z * 0.07)
        palette = MEADOW_PALETTE if meadow > 0.54 else GRASS_PALETTE
        return palette[min(len(palette) - 1, int(variation * len(palette)))]

    def _water_mesh(
        self,
        coord: tuple[int, int],
        *,
        height_at: Callable[[float, float], float] | None = None,
    ) -> tuple[Mesh | None, tuple[WaterSource, ...], LineSet3 | None, LineSet3 | None]:
        height = height_at or self.height_at
        cx, cz = coord
        size = self.config.chunk_size
        resolution = self.config.chunk_resolution * self.config.shoreline_detail
        step = size / resolution
        x0 = cx * size
        z0 = cz * size
        heights = [
            [height(x0 + x_index * step, z0 + z_index * step) for x_index in range(resolution + 1)]
            for z_index in range(resolution + 1)
        ]
        water_y_cache: dict[tuple[float, float], float] = {}

        def water_y(x: float, z: float) -> float:
            key = (round(x, 6), round(z, 6))
            cached = water_y_cache.get(key)
            if cached is None:
                cached = self._water_y_at(x, z)
                water_y_cache[key] = cached
            return cached

        triangles: list[Triangle] = []
        wet_centers: list[Vec3] = []
        shoreline_segments: list[tuple[Vec3, Vec3]] = []
        ripple_segments: list[tuple[Vec3, Vec3]] = []
        normal = Vec3(0.0, 1.0, 0.0)
        for z_index in range(resolution):
            z = z0 + z_index * step
            for x_index in range(resolution):
                x = x0 + x_index * step
                corners = (
                    (x, z, self.config.water_level - heights[z_index][x_index]),
                    (x + step, z, self.config.water_level - heights[z_index][x_index + 1]),
                    (x + step, z + step, self.config.water_level - heights[z_index + 1][x_index + 1]),
                    (x, z + step, self.config.water_level - heights[z_index + 1][x_index]),
                )
                polygon = _clip_water_polygon(corners)
                if len(polygon) < 3:
                    continue
                center_x = sum(point[0] for point in polygon) / len(polygon)
                center_z = sum(point[1] for point in polygon) / len(polygon)
                wet_centers.append(Vec3(center_x, self.config.water_level, center_z))
                vertices = [Vec3(px, water_y(px, pz), pz) for px, pz in polygon]
                center_depth = sum(value for _px, _pz, value in corners if value > 0.0) / max(1, sum(1 for _px, _pz, value in corners if value > 0.0))
                material = WATER_SHALLOW if center_depth < 0.34 else WATER_DEEP if center_depth > 1.15 else WATER
                for index in range(1, len(vertices) - 1):
                    triangles.append(
                        Triangle(
                            vertices[0],
                            vertices[index],
                            vertices[index + 1],
                            material,
                            (0.0, 0.0),
                            (0.5, 1.0),
                            (1.0, 0.0),
                            normal,
                            normal,
                            normal,
                        )
                )
                if any(value <= 0.0 for _px, _pz, value in corners) and any(value > 0.0 for _px, _pz, value in corners):
                    shore_points = [vertex for vertex in vertices if abs(height(vertex.x, vertex.z) - self.config.water_level) < 0.035]
                    if len(shore_points) >= 2:
                        shoreline_segments.append((shore_points[0] + Vec3(0.0, 0.018, 0.0), shore_points[1] + Vec3(0.0, 0.018, 0.0)))
                elif _stable_unit(self.config.seed, cx, cz, x_index, z_index, 761) < 0.18:
                    angle = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 769) * tau
                    length = step * (0.18 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 773) * 0.28)
                    center = Vec3(center_x, water_y(center_x, center_z) + 0.024, center_z)
                    offset = Vec3(cos(angle) * length, 0.0, sin(angle) * length)
                    ripple_segments.append((center - offset, center + offset))
        if not triangles:
            return None, (), None, None
        source = _water_source_for(wet_centers, step, self.config.water_level)
        ripple_segments.extend(_source_ripple_segments(source, self.config.seed, coord))
        shorelines = LineSet3(shoreline_segments, FOAM) if shoreline_segments else None
        ripples = LineSet3(ripple_segments, RIPPLE) if ripple_segments else None
        return Mesh(triangles), (source,), shorelines, ripples

    def _water_y_at(self, x: float, z: float) -> float:
        ripple = self._water_noise.sample_xyz(x * 0.17, 5.0, z * 0.17) - 0.5
        long_wave = self._water_noise.sample_xyz(x * 0.045, 11.0, z * 0.045) - 0.5
        return self.config.water_level + 0.018 + ripple * 0.018 + long_wave * 0.026

    def _trees(
        self,
        coord: tuple[int, int],
        *,
        height_at: Callable[[float, float], float] | None = None,
        normal_at: Callable[[float, float], Vec3] | None = None,
    ) -> tuple[TreeInstance, ...]:
        sample_height = height_at or self.height_at
        normal_sample = normal_at or (lambda x, z: self.normal_at(x, z))
        cx, cz = coord
        size = self.config.chunk_size
        slots = self.config.tree_slots_per_axis
        slot_size = size / slots
        x0 = cx * size
        z0 = cz * size
        trees: list[TreeInstance] = []
        for z_index in range(slots):
            for x_index in range(slots):
                if _stable_unit(self.config.seed, cx, cz, x_index, z_index, 101) > self.config.tree_density:
                    continue
                jitter_x = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 113)
                jitter_z = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 127)
                x = x0 + (x_index + 0.18 + jitter_x * 0.64) * slot_size
                z = z0 + (z_index + 0.18 + jitter_z * 0.64) * slot_size
                y = sample_height(x, z)
                normal = normal_sample(x, z)
                if y <= self.config.water_level + 0.22 or normal.y < 0.74:
                    continue
                trunk_height = 4.6 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 139) * 4.4
                crown = 2.05 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 149) * 1.75
                radius = 0.2 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 157) * 0.17
                yaw = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 167) * tau
                lean = Vec3(
                    (_stable_unit(self.config.seed, cx, cz, x_index, z_index, 169) - 0.5) * 0.42,
                    0.0,
                    (_stable_unit(self.config.seed, cx, cz, x_index, z_index, 173) - 0.5) * 0.42,
                )
                branch_count = 5 + int(_stable_unit(self.config.seed, cx, cz, x_index, z_index, 179) * 4.0)
                species = 1 if _stable_unit(self.config.seed, cx, cz, x_index, z_index, 181) > 0.7 else 0
                variant = 1 if _stable_unit(self.config.seed, cx, cz, x_index, z_index, 163) > 0.58 else 0
                trees.append(TreeInstance(Vec3(x, y, z), trunk_height, radius, crown, yaw, lean, branch_count, species, variant))
        return tuple(trees)

    def _bushes(
        self,
        coord: tuple[int, int],
        *,
        height_at: Callable[[float, float], float] | None = None,
        normal_at: Callable[[float, float], Vec3] | None = None,
    ) -> tuple[BushInstance, ...]:
        sample_height = height_at or self.height_at
        normal_sample = normal_at or (lambda x, z: self.normal_at(x, z))
        cx, cz = coord
        size = self.config.chunk_size
        slots = self.config.bush_slots_per_axis
        slot_size = size / slots
        x0 = cx * size
        z0 = cz * size
        bushes: list[BushInstance] = []
        for z_index in range(slots):
            for x_index in range(slots):
                cover = self._foliage_noise.sample_xyz((x0 + x_index * slot_size) * 0.055, 33.0, (z0 + z_index * slot_size) * 0.055)
                probability = self.config.bush_density * (0.55 + cover * 0.75)
                if _stable_unit(self.config.seed, cx, cz, x_index, z_index, 601) > probability:
                    continue
                jitter_x = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 607)
                jitter_z = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 613)
                x = x0 + (x_index + 0.18 + jitter_x * 0.64) * slot_size
                z = z0 + (z_index + 0.18 + jitter_z * 0.64) * slot_size
                y = sample_height(x, z)
                normal = normal_sample(x, z)
                if y <= self.config.water_level + 0.12 or normal.y < 0.68:
                    continue
                radius = 0.42 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 617) * 0.62
                bush_height = 0.34 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 619) * 0.55
                variant = 1 if _stable_unit(self.config.seed, cx, cz, x_index, z_index, 631) > 0.5 else 0
                bushes.append(BushInstance(Vec3(x, y, z), radius, bush_height, variant))
        return tuple(bushes)

    def _rocks(
        self,
        coord: tuple[int, int],
        *,
        height_at: Callable[[float, float], float] | None = None,
        normal_at: Callable[[float, float], Vec3] | None = None,
    ) -> tuple[RockInstance, ...]:
        sample_height = height_at or self.height_at
        normal_sample = normal_at or (lambda x, z: self.normal_at(x, z))
        cx, cz = coord
        size = self.config.chunk_size
        slots = self.config.rock_slots_per_axis
        slot_size = size / slots
        x0 = cx * size
        z0 = cz * size
        rocks: list[RockInstance] = []
        for z_index in range(slots):
            for x_index in range(slots):
                ridge = self._ridge_noise.sample_xyz((x0 + x_index * slot_size) * 0.075, 45.0, (z0 + z_index * slot_size) * 0.075)
                probability = self.config.rock_density * (0.45 + ridge * 0.9)
                if _stable_unit(self.config.seed, cx, cz, x_index, z_index, 1301) > probability:
                    continue
                jitter_x = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 1303)
                jitter_z = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 1307)
                x = x0 + (x_index + 0.18 + jitter_x * 0.64) * slot_size
                z = z0 + (z_index + 0.18 + jitter_z * 0.64) * slot_size
                y = sample_height(x, z)
                normal = normal_sample(x, z)
                if y <= self.config.water_level + 0.05 or normal.y < 0.55:
                    continue
                radius = 0.28 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 1319) * 0.9
                height = 0.16 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 1321) * 0.52
                yaw = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 1327) * tau
                variant = min(len(ROCK_PALETTE) - 1, int(_stable_unit(self.config.seed, cx, cz, x_index, z_index, 1329) * len(ROCK_PALETTE)))
                rocks.append(RockInstance(Vec3(x, y + 0.02, z), radius, height, yaw, variant))
        return tuple(rocks)

    def _grass(
        self,
        coord: tuple[int, int],
        *,
        height_at: Callable[[float, float], float] | None = None,
        normal_at: Callable[[float, float], Vec3] | None = None,
        density_scale: float = 1.0,
        distant: bool = False,
    ) -> Mesh | None:
        sample_height = height_at or self.height_at
        normal_sample = normal_at or (lambda x, z: self.normal_at(x, z))
        cx, cz = coord
        size = self.config.chunk_size
        x0 = cx * size
        z0 = cz * size
        triangles: list[Triangle] = []
        blade_candidates = max(0, int(self.config.grass_blades_per_chunk * max(0.0, density_scale)))
        for index in range(blade_candidates):
            x = x0 + _stable_unit(self.config.seed, cx, cz, index, 701) * size
            z = z0 + _stable_unit(self.config.seed, cx, cz, index, 709) * size
            y = sample_height(x, z)
            normal = normal_sample(x, z)
            if y <= self.config.water_level + 0.08 or normal.y < 0.72:
                continue
            cover = self._foliage_noise.sample_xyz(x * 0.11, 41.0, z * 0.11)
            threshold = 0.66 + cover * 0.28 if distant else 0.70 + cover * 0.24
            if _stable_unit(self.config.seed, cx, cz, index, 719) > threshold:
                continue
            blade_count = (6 + int(_stable_unit(self.config.seed, cx, cz, index, 721) * 5.0)) if distant else (9 + int(_stable_unit(self.config.seed, cx, cz, index, 721) * 9.0))
            variant = min(3, int(_stable_unit(self.config.seed, cx, cz, index, 727) * 4.0))
            yaw = _stable_unit(self.config.seed, cx, cz, index, 733) * tau
            height_scale = 1.12 + cover * 0.52 + _stable_unit(self.config.seed, cx, cz, index, 739) * 0.55 if distant else 1.04 + cover * 0.45 + _stable_unit(self.config.seed, cx, cz, index, 739) * 0.5
            spread_scale = 1.34 + _stable_unit(self.config.seed, cx, cz, index, 743) * 0.76 if distant else 1.12 + _stable_unit(self.config.seed, cx, cz, index, 743) * 0.62
            origin = Vec3(x, y + 0.018, z)
            triangles.extend(
                _transform_triangles(
                    _grass_tuft_template(variant, blade_count),
                    origin,
                    normal,
                    yaw,
                    Vec3(spread_scale, height_scale, spread_scale),
                )
            )
        return Mesh(triangles) if triangles else None


def _water_source_for(centers: Iterable[Vec3], step: float, level: float) -> WaterSource:
    values = tuple(centers)
    average = Vec3(
        sum(point.x for point in values) / len(values),
        level,
        sum(point.z for point in values) / len(values),
    )
    radius = max(step * 0.75, sqrt(len(values)) * step * 0.55)
    return WaterSource(average, radius, level)


def _source_ripple_segments(source: WaterSource, seed: int, coord: tuple[int, int]) -> tuple[tuple[Vec3, Vec3], ...]:
    cx, cz = coord
    segments: list[tuple[Vec3, Vec3]] = []
    ring_count = 2 + int(_stable_unit(seed, cx, cz, 1201) * 2.0)
    for ring_index in range(ring_count):
        radius = source.radius * (0.24 + ring_index * 0.18 + _stable_unit(seed, cx, cz, ring_index, 1207) * 0.035)
        parts = 9 + ring_index * 2
        phase = _stable_unit(seed, cx, cz, ring_index, 1213) * tau
        for index in range(parts):
            if _stable_unit(seed, cx, cz, ring_index, index, 1217) < 0.32:
                continue
            a0 = phase + index * tau / parts
            a1 = phase + (index + 0.58) * tau / parts
            start = source.center + Vec3(cos(a0) * radius, 0.038, sin(a0) * radius)
            end = source.center + Vec3(cos(a1) * radius, 0.038, sin(a1) * radius)
            segments.append((start, end))
    return tuple(segments)


@lru_cache(maxsize=64)
def _grass_tuft_template(variant: int, blade_count: int) -> tuple[Triangle, ...]:
    triangles: list[Triangle] = []
    count = max(3, int(blade_count))
    base_seed = 3701 + variant * 97
    for blade in range(count):
        angle = blade * tau / count + (_stable_unit(base_seed, blade, 3) - 0.5) * 0.34
        height = 0.46 + _stable_unit(base_seed, blade, 5) * 0.58
        width = 0.028 + _stable_unit(base_seed, blade, 7) * 0.028
        bend = 0.08 + _stable_unit(base_seed, blade, 11) * 0.28
        root_radius = _stable_unit(base_seed, blade, 13) * 0.15
        root_angle = angle + (_stable_unit(base_seed, blade, 17) - 0.5) * 0.7
        material = GRASS_TUFT_PALETTE[(variant + blade) % len(GRASS_TUFT_PALETTE)]
        triangles.extend(_grass_blade_triangles(root_angle, root_radius, angle, height, width, bend, material))
    return tuple(triangles)


def _flat_triangle(a: Vec3, b: Vec3, c: Vec3, material: Material, preferred_normal: Vec3 | None = None) -> Triangle:
    fallback = preferred_normal or Vec3(0.0, 1.0, 0.0)
    normal = (b - a).cross(c - a).normalized(fallback)
    if preferred_normal is not None and normal.dot(preferred_normal) < 0.0:
        normal = normal * -1.0
    return Triangle(a, b, c, material, normal_a=normal, normal_b=normal, normal_c=normal)


def _grass_blade_triangles(root_angle: float, root_radius: float, angle: float, height: float, width: float, bend: float, material: Material) -> tuple[Triangle, ...]:
    forward = Vec3(cos(angle), 0.0, sin(angle))
    side = Vec3(-sin(angle), 0.0, cos(angle))
    root = Vec3(cos(root_angle) * root_radius, 0.0, sin(root_angle) * root_radius)
    mid = root + forward * (bend * 0.48) + Vec3(0.0, height * 0.56, 0.0)
    tip = root + forward * bend + Vec3(0.0, height, 0.0)
    base_left = root - side * (width * 0.5)
    base_right = root + side * (width * 0.5)
    mid_left = mid - side * (width * 0.28)
    normal_hint = (side * 0.42 + Vec3(0.0, 0.28, 0.0) - forward * 0.12).normalized(Vec3(0.0, 1.0, 0.0))
    return (
        _flat_triangle(base_left, mid_left, base_right, material, normal_hint),
        _flat_triangle(base_right, mid_left, tip, material, normal_hint),
    )


@lru_cache(maxsize=64)
def _maple_leaf_tuft_template(variant: int, leaf_count: int) -> tuple[Triangle, ...]:
    triangles: list[Triangle] = []
    count = max(3, min(4, int(leaf_count)))
    base_seed = 4903 + variant * 131
    for leaf in range(count):
        angle = leaf * tau / count + (_stable_unit(base_seed, leaf, 19) - 0.5) * 0.9
        radius = 0.18 + _stable_unit(base_seed, leaf, 23) * 0.26
        lift = (_stable_unit(base_seed, leaf, 29) - 0.45) * 0.18
        center = Vec3(cos(angle) * radius, lift, sin(angle) * radius)
        leaf_yaw = angle + (_stable_unit(base_seed, leaf, 31) - 0.5) * 1.25
        leaf_pitch = -0.24 + _stable_unit(base_seed, leaf, 37) * 0.48
        leaf_scale = 0.5 + _stable_unit(base_seed, leaf, 41) * 0.22
        material = MAPLE_LEAF_PALETTE[(variant + leaf) % len(MAPLE_LEAF_PALETTE)]
        triangles.extend(_maple_leaf_triangles(center, leaf_yaw, leaf_pitch, leaf_scale, material))
    return tuple(triangles)


def _maple_leaf_triangles(center: Vec3, yaw: float, pitch: float, scale: float, material: Material) -> tuple[Triangle, ...]:
    outline = (
        (0.0, -0.46),
        (-0.10, -0.20),
        (-0.38, -0.28),
        (-0.24, -0.02),
        (-0.52, 0.08),
        (-0.20, 0.18),
        (-0.34, 0.44),
        (-0.06, 0.30),
        (0.0, 0.58),
        (0.06, 0.30),
        (0.34, 0.44),
        (0.20, 0.18),
        (0.52, 0.08),
        (0.24, -0.02),
        (0.38, -0.28),
        (0.10, -0.20),
    )
    cy = cos(yaw)
    sy = sin(yaw)
    cp = cos(pitch)
    sp = sin(pitch)

    def local(point: tuple[float, float]) -> Vec3:
        x = point[0] * scale
        y = point[1] * scale
        z = y * sp
        y = y * cp
        return center + Vec3(x * cy - z * sy, y, x * sy + z * cy)

    fan_center = local((0.0, 0.04))
    points = tuple(local(point) for point in outline)
    normal_hint = (points[8] - fan_center).cross(points[12] - fan_center).normalized(Vec3(0.0, 1.0, 0.0))
    if normal_hint.y < 0.0:
        normal_hint = normal_hint * -1.0
    triangles: list[Triangle] = []
    for index, point in enumerate(points):
        triangles.append(_flat_triangle(fan_center, point, points[(index + 1) % len(points)], material, normal_hint))
    stem_end = local((0.0, -0.72))
    stem_width = 0.018 * scale
    stem_side = Vec3(cy, 0.0, sy) * stem_width
    stem_top = local((0.0, -0.37))
    triangles.append(_flat_triangle(stem_top - stem_side, stem_end, stem_top + stem_side, material, normal_hint))
    return tuple(triangles)


def _transform_triangles(template: Iterable[Triangle], origin: Vec3, up: Vec3, yaw: float, scale: Vec3) -> tuple[Triangle, ...]:
    tangent, bitangent, normal = _surface_basis(up, yaw)

    def transform(point: Vec3) -> Vec3:
        local = Vec3(point.x * scale.x, point.y * scale.y, point.z * scale.z)
        return origin + tangent * local.x + normal * local.y + bitangent * local.z

    transformed: list[Triangle] = []
    for triangle in template:
        a = transform(triangle.a)
        b = transform(triangle.b)
        c = transform(triangle.c)
        normal_hint = (b - a).cross(c - a).normalized(normal)
        if normal_hint.dot(normal) < -0.35:
            normal_hint = normal_hint * -1.0
        transformed.append(
            Triangle(
                a,
                b,
                c,
                triangle.material,
                triangle.uv_a,
                triangle.uv_b,
                triangle.uv_c,
                normal_hint,
                normal_hint,
                normal_hint,
            )
        )
    return tuple(transformed)


def _surface_basis(up: Vec3, yaw: float) -> tuple[Vec3, Vec3, Vec3]:
    normal = up.normalized(Vec3(0.0, 1.0, 0.0))
    seed_tangent = Vec3(cos(yaw), 0.0, sin(yaw))
    tangent = (seed_tangent - normal * seed_tangent.dot(normal)).normalized()
    if tangent.length_squared() <= 1e-12:
        tangent = normal.cross(Vec3(1.0, 0.0, 0.0)).normalized(Vec3(1.0, 0.0, 0.0))
    bitangent = normal.cross(tangent).normalized(Vec3(0.0, 0.0, 1.0))
    return tangent, bitangent, normal


def _tree_primitives(tree: TreeInstance, seed: int, *, leaf_tufts_per_branch: int = 2, wind_offset: Vec3 | None = None) -> tuple[object, ...]:
    key = _position_key(tree.position)
    static_trunk_top = tree.position + Vec3(tree.lean.x, tree.trunk_height, tree.lean.z)
    wind = wind_offset or Vec3(0.0, 0.0, 0.0)
    sway_height = max(0.1, tree.trunk_height + tree.crown_radius * 1.8)

    def sway_point(point: Vec3, strength: float = 1.0) -> Vec3:
        ratio = clamp((point.y - tree.position.y) / sway_height, 0.0, 1.0)
        return point + wind * (ratio * ratio * strength)

    trunk_top = sway_point(static_trunk_top)
    static_branch_specs = _tree_branch_specs(tree, seed, key, static_trunk_top)
    branch_specs = tuple((sway_point(start, 0.42), sway_point(end, 1.0), radius) for start, end, radius in static_branch_specs)
    wood_triangles = list(_tapered_cylinder_triangles(tree.position, trunk_top, tree.trunk_radius * 1.25, tree.trunk_radius * 0.62, 7, TRUNK))
    for start, end, radius in branch_specs:
        wood_triangles.extend(_tapered_cylinder_triangles(start, end, radius, max(radius * 0.42, tree.trunk_radius * 0.14), 5, BRANCH))

    root_segments: list[tuple[Vec3, Vec3]] = []
    for index in range(4):
        angle = tree.yaw + index * tau / 4.0 + (_stable_unit(seed, *key, index, 801) - 0.5) * 0.55
        length = tree.trunk_radius * (3.2 + _stable_unit(seed, *key, index, 803) * 2.4)
        root_segments.append((tree.position + Vec3(0.0, 0.025, 0.0), tree.position + Vec3(cos(angle) * length, 0.018, sin(angle) * length)))

    objects: list[object] = [Mesh(wood_triangles)]
    if root_segments:
        objects.append(LineSet3(root_segments, TWIG))
    canopy_anchor = trunk_top + Vec3(0.0, tree.crown_radius * (0.55 if tree.species == 0 else 0.78), 0.0)
    fill_triangles: list[Triangle] = []
    leaf_triangles: list[Triangle] = []
    twig_segments: list[tuple[Vec3, Vec3]] = []
    tuft_index = 0
    for branch_index, (start, end, radius) in enumerate(branch_specs):
        branch_axis = (end - start).normalized(Vec3(0.0, 1.0, 0.0))
        for detail in range(max(1, leaf_tufts_per_branch)):
            if detail > 0 and _stable_unit(seed, *key, branch_index, detail, 835) < 0.18:
                continue
            amount = 0.98 if detail == 0 else 0.56 + _stable_unit(seed, *key, branch_index, detail, 837) * 0.3
            base = start + (end - start) * amount
            angle = tree.yaw + branch_index * tau / max(1, tree.branch_count) + detail * 1.73 + (_stable_unit(seed, *key, branch_index, detail, 839) - 0.5) * 0.82
            lift = tree.crown_radius * (0.04 + _stable_unit(seed, *key, branch_index, detail, 841) * 0.18)
            side = Vec3(cos(angle), 0.0, sin(angle))
            center = base + side * (tree.crown_radius * (0.08 + _stable_unit(seed, *key, branch_index, detail, 843) * 0.18)) + Vec3(0.0, lift, 0.0)
            tuft_normal = (branch_axis * 0.44 + side * 0.28 + Vec3(0.0, 0.72, 0.0)).normalized(Vec3(0.0, 1.0, 0.0))
            fill_triangles.extend(
                _leaf_cluster_triangles(
                    center + side * (tree.crown_radius * 0.08),
                    tree.crown_radius * (0.29 + _stable_unit(seed, *key, branch_index, detail, 844) * 0.18),
                    angle,
                    CANOPY_HAZE_PALETTE[(tree.material_variant + branch_index + detail) % len(CANOPY_HAZE_PALETTE)],
                    seed,
                    key,
                    1200 + branch_index * 9 + detail,
                    leaf_count=5,
                )
            )
            leaf_count = 3 + int(_stable_unit(seed, *key, branch_index, detail, 845) * 2.0)
            variant = (tree.material_variant + tree.species + branch_index + detail) % 4
            scale = tree.crown_radius * (0.072 + _stable_unit(seed, *key, branch_index, detail, 847) * 0.052)
            leaf_triangles.extend(
                _transform_triangles(
                    _maple_leaf_tuft_template(variant, leaf_count),
                    center,
                    tuft_normal,
                    angle + _stable_unit(seed, *key, branch_index, detail, 849) * 0.65,
                    Vec3(scale, scale, scale),
                )
            )
            twig_segments.append((base, center))
            tuft_index += 1

    crown_tufts = 15 + tree.species * 5
    for index in range(crown_tufts):
        angle = tree.yaw + index * tau / crown_tufts + (_stable_unit(seed, *key, index, 861) - 0.5) * 0.7
        radius = tree.crown_radius * (0.08 + _stable_unit(seed, *key, index, 863) * 0.48)
        lift = tree.crown_radius * (-0.08 + _stable_unit(seed, *key, index, 865) * 0.62)
        center = canopy_anchor + Vec3(cos(angle) * radius, lift, sin(angle) * radius)
        tuft_normal = Vec3(cos(angle) * 0.25, 0.92, sin(angle) * 0.25).normalized(Vec3(0.0, 1.0, 0.0))
        fill_triangles.extend(
            _leaf_cluster_triangles(
                center,
                tree.crown_radius * (0.32 + _stable_unit(seed, *key, index, 866) * 0.2),
                angle + 0.35,
                CANOPY_HAZE_PALETTE[(tree.material_variant + tree.species + index) % len(CANOPY_HAZE_PALETTE)],
                seed,
                key,
                1400 + index,
                leaf_count=6,
            )
        )
        if index % 2 == 0:
            scale = tree.crown_radius * (0.074 + _stable_unit(seed, *key, index, 867) * 0.052)
            variant = (tree.material_variant + tree.species + index + tuft_index) % 4
            leaf_triangles.extend(
                _transform_triangles(
                    _maple_leaf_tuft_template(variant, 3 + int(_stable_unit(seed, *key, index, 869) * 2.0)),
                    center,
                    tuft_normal,
                    angle,
                    Vec3(scale, scale, scale),
                )
            )

    if fill_triangles or leaf_triangles:
        objects.append(Mesh([*fill_triangles, *leaf_triangles]))
    if twig_segments:
        objects.append(LineSet3(twig_segments, TWIG))

    return tuple(objects)


def swayed_tree_primitives(
    tree: TreeInstance,
    seed: int,
    *,
    leaf_tufts_per_branch: int = 2,
    wind_time: float = 0.0,
    wind_strength: float = 1.0,
    wind_direction: Vec3 = Vec3(0.72, 0.0, 0.42),
) -> tuple[object, ...]:
    """Return a gently wind-swayed tree mesh using coherent deterministic phase."""

    key = _position_key(tree.position)
    direction = wind_direction.normalized(Vec3(1.0, 0.0, 0.0))
    phase = (
        wind_time * 0.72
        + tree.position.x * 0.035
        + tree.position.z * 0.026
        + _stable_unit(seed, *key, 883) * tau
    )
    secondary = wind_time * 1.18 + (tree.position.x - tree.position.z) * 0.012 + _stable_unit(seed, *key, 887) * tau
    sway = sin(phase) * 0.68 + sin(secondary) * 0.22
    amplitude = tree.crown_radius * 0.075 * clamp(wind_strength, 0.0, 2.5)
    wind_offset = direction * (sway * amplitude)
    return _tree_primitives(tree, seed, leaf_tufts_per_branch=leaf_tufts_per_branch, wind_offset=wind_offset)


def _tree_distant_primitives(tree: TreeInstance, seed: int) -> tuple[object, ...]:
    key = _position_key(tree.position)
    trunk_top = tree.position + Vec3(tree.lean.x, tree.trunk_height, tree.lean.z)
    branch_specs = _tree_branch_specs(tree, seed, key, trunk_top)
    wood_triangles = list(_tapered_cylinder_triangles(tree.position, trunk_top, tree.trunk_radius * 1.18, tree.trunk_radius * 0.56, 5, TRUNK))
    for index, (start, end, radius) in enumerate(branch_specs):
        if index % 2:
            continue
        wood_triangles.extend(_tapered_cylinder_triangles(start, end, radius * 0.82, max(radius * 0.28, tree.trunk_radius * 0.1), 4, BRANCH))

    canopy_anchor = trunk_top + Vec3(0.0, tree.crown_radius * (0.5 if tree.species == 0 else 0.7), 0.0)
    haze_triangles: list[Triangle] = []
    card_count = 12 + tree.species * 4
    for index in range(card_count):
        branch_start, branch_end, _branch_radius = branch_specs[index % len(branch_specs)] if branch_specs else (tree.position, trunk_top, tree.trunk_radius)
        angle = tree.yaw + index * tau / card_count + (_stable_unit(seed, *key, index, 891) - 0.5) * 0.78
        amount = 0.58 + _stable_unit(seed, *key, index, 892) * 0.36
        branch_center = branch_start + (branch_end - branch_start) * amount
        radial = Vec3(cos(angle), 0.0, sin(angle))
        crown_center = canopy_anchor + radial * (tree.crown_radius * (_stable_unit(seed, *key, index, 893) * 0.38))
        blend = 0.36 + _stable_unit(seed, *key, index, 894) * 0.42
        center = branch_center * blend + crown_center * (1.0 - blend)
        center = center + Vec3(0.0, (_stable_unit(seed, *key, index, 895) - 0.46) * tree.crown_radius * 0.5, 0.0)
        cluster_radius = tree.crown_radius * (0.42 + _stable_unit(seed, *key, index, 897) * 0.22)
        material = CANOPY_HAZE_PALETTE[(tree.material_variant + tree.species + index) % len(CANOPY_HAZE_PALETTE)]
        haze_triangles.extend(
            _canopy_haze_card_triangles(
                center,
                cluster_radius * 1.18,
                cluster_radius * 0.82,
                angle + (_stable_unit(seed, *key, index, 899) - 0.5) * 0.5,
                material,
                seed,
                key,
                index,
            )
        )

    objects: list[object] = [Mesh(wood_triangles)]
    if haze_triangles:
        objects.append(Mesh(haze_triangles))
    return tuple(objects)


def _canopy_haze_card_triangles(
    center: Vec3,
    width: float,
    height: float,
    yaw: float,
    material: Material,
    seed: int,
    key: tuple[int, int, int],
    card_index: int,
) -> tuple[Triangle, ...]:
    right = Vec3(cos(yaw), 0.0, sin(yaw)).normalized(Vec3(1.0, 0.0, 0.0))
    normal = Vec3(-sin(yaw), 0.12, cos(yaw)).normalized(Vec3(0.0, 0.0, 1.0))
    up = Vec3(0.0, 1.0, 0.0)
    outline = (
        (-0.54, -0.18),
        (-0.48, 0.18),
        (-0.30, 0.44),
        (-0.06, 0.56),
        (0.22, 0.48),
        (0.50, 0.22),
        (0.58, -0.08),
        (0.34, -0.34),
        (0.02, -0.44),
        (-0.32, -0.36),
    )

    def point(local: tuple[float, float], index: int) -> Vec3:
        jitter = (_stable_unit(seed, *key, card_index, index, 905) - 0.5) * 0.12
        depth = (_stable_unit(seed, *key, card_index, index, 907) - 0.5) * width * 0.08
        return center + right * (local[0] * width * (1.0 + jitter)) + up * (local[1] * height) + normal * depth

    fan_center = center + normal * ((_stable_unit(seed, *key, card_index, 911) - 0.5) * width * 0.045)
    points = tuple(point(local, index) for index, local in enumerate(outline))
    triangles: list[Triangle] = []
    for index, current in enumerate(points):
        triangles.append(_flat_triangle(fan_center, current, points[(index + 1) % len(points)], material, normal))
    return tuple(triangles)


def _leaf_cluster_triangles(
    center: Vec3,
    radius: float,
    yaw: float,
    material: Material,
    seed: int,
    key: tuple[int, int, int],
    cluster_index: int,
    *,
    leaf_count: int = 5,
) -> tuple[Triangle, ...]:
    triangles: list[Triangle] = []
    count = max(3, min(4, int((leaf_count + 1) // 2)))
    for index in range(count):
        angle = yaw + index * tau / count + (_stable_unit(seed, *key, cluster_index, index, 1301) - 0.5) * 1.1
        outward = Vec3(cos(angle), 0.0, sin(angle))
        side = Vec3(-sin(angle), 0.0, cos(angle))
        offset = radius * (0.05 + _stable_unit(seed, *key, cluster_index, index, 1303) * 0.42)
        lift = radius * ((_stable_unit(seed, *key, cluster_index, index, 1307) - 0.42) * 0.5)
        depth = radius * ((_stable_unit(seed, *key, cluster_index, index, 1311) - 0.5) * 0.26)
        cluster_center = center + outward * offset + side * depth + Vec3(0.0, lift, 0.0)
        blob_radius = radius * (0.2 + _stable_unit(seed, *key, cluster_index, index, 1313) * 0.1)
        stretch = outward * (blob_radius * (0.18 + _stable_unit(seed, *key, cluster_index, index, 1317) * 0.18))
        stretch = stretch + Vec3(0.0, blob_radius * 0.1, 0.0) + side * (blob_radius * 0.08)
        triangles.extend(
            _ellipsoid_blob_triangles(
                cluster_center,
                blob_radius,
                material,
                stretch=stretch,
                segments=5,
                rings=3,
                scale_y=0.68,
            )
        )
    return tuple(triangles)


def _tree_branch_specs(tree: TreeInstance, seed: int, key: tuple[int, int, int], trunk_top: Vec3) -> tuple[tuple[Vec3, Vec3, float], ...]:
    specs: list[tuple[Vec3, Vec3, float]] = []
    for index in range(tree.branch_count):
        amount = 0.36 + index / max(1, tree.branch_count - 1) * 0.48
        start = tree.position + (trunk_top - tree.position) * amount
        angle = tree.yaw + index * tau / max(1, tree.branch_count) + (_stable_unit(seed, *key, index, 907) - 0.5) * 0.9
        length = tree.crown_radius * (0.38 + _stable_unit(seed, *key, index, 911) * 0.48)
        lift = tree.crown_radius * (0.14 + _stable_unit(seed, *key, index, 919) * 0.34)
        end = start + Vec3(cos(angle) * length, lift, sin(angle) * length)
        radius = tree.trunk_radius * (0.38 + (1.0 - amount) * 0.28)
        specs.append((start, end, radius))
    return tuple(specs)


def _bush_primitives(bush: BushInstance, seed: int) -> tuple[object, ...]:
    key = _position_key(bush.position)
    objects: list[object] = []
    twig_segments: list[tuple[Vec3, Vec3]] = []
    leaf_triangles: list[Triangle] = []
    puff_count = 2 + int(_stable_unit(seed, *key, 941) * 3.0)
    for index in range(puff_count):
        angle = _stable_unit(seed, *key, index, 947) * tau
        offset = bush.radius * (_stable_unit(seed, *key, index, 953) * 0.42)
        center = bush.position + Vec3(cos(angle) * offset, bush.height * (0.42 + index * 0.08), sin(angle) * offset)
        radius = bush.radius * (0.58 + _stable_unit(seed, *key, index, 967) * 0.28)
        stretch = Vec3(cos(angle) * radius * 0.28, bush.height * 0.22, sin(angle) * radius * 0.28)
        leaf_triangles.extend(_ellipsoid_blob_triangles(center, radius, BUSH_PALETTE[(bush.variant + index) % len(BUSH_PALETTE)], stretch=stretch, segments=5, rings=3))
        twig_start = bush.position + Vec3(0.0, 0.035, 0.0)
        twig_end = center + Vec3(cos(angle) * radius * 0.22, -bush.height * 0.12, sin(angle) * radius * 0.22)
        twig_segments.append((twig_start, twig_end))
    if leaf_triangles:
        objects.append(Mesh(leaf_triangles))
    if twig_segments:
        objects.append(LineSet3(twig_segments, TWIG))
    return tuple(objects)


def _rock_primitives(rock: RockInstance) -> tuple[object, ...]:
    material = ROCK_PALETTE[rock.variant % len(ROCK_PALETTE)]
    stretch = Vec3(cos(rock.yaw) * rock.radius * 0.35, -rock.height * 0.12, sin(rock.yaw) * rock.radius * 0.22)
    triangles = _ellipsoid_blob_triangles(
        rock.position + Vec3(0.0, rock.height * 0.35, 0.0),
        rock.radius,
        material,
        stretch=stretch,
        segments=6,
        rings=3,
        scale_y=max(0.18, rock.height / max(0.001, rock.radius)),
    )
    return (Mesh(triangles),)


def _ellipsoid_blob_triangles(
    center: Vec3,
    radius: float,
    material: Material,
    *,
    stretch: Vec3 = Vec3(0.0, 0.0, 0.0),
    segments: int = 7,
    rings: int = 4,
    scale_y: float = 0.72,
) -> tuple[Triangle, ...]:
    rx = max(0.05, radius + abs(stretch.x))
    ry = max(0.05, radius * scale_y + abs(stretch.y))
    rz = max(0.05, radius + abs(stretch.z))
    rings = max(3, int(rings))
    segments = max(5, int(segments))
    rows: list[list[tuple[Vec3, Vec3]]] = []
    for ring in range(rings + 1):
        phi = ring * tau / (rings * 2.0)
        y = cos(phi) * ry
        radial = sin(phi)
        row: list[tuple[Vec3, Vec3]] = []
        for segment in range(segments):
            theta = segment * tau / segments
            local = Vec3(cos(theta) * rx * radial, y, sin(theta) * rz * radial)
            normal = Vec3(local.x / max(0.001, rx), local.y / max(0.001, ry), local.z / max(0.001, rz)).normalized(Vec3(0.0, 1.0, 0.0))
            row.append((center + local, normal))
        rows.append(row)

    triangles: list[Triangle] = []
    for ring in range(rings):
        current = rows[ring]
        following = rows[ring + 1]
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a, na = current[segment]
            b, nb = following[segment]
            c, nc = current[next_segment]
            d, nd = following[next_segment]
            if ring > 0:
                triangles.append(Triangle(a, b, c, material, normal_a=na, normal_b=nb, normal_c=nc))
            if ring < rings - 1:
                triangles.append(Triangle(c, b, d, material, normal_a=nc, normal_b=nb, normal_c=nd))
    return tuple(triangles)


def _tapered_cylinder_triangles(
    start: Vec3,
    end: Vec3,
    start_radius: float,
    end_radius: float,
    segments: int,
    material: Material,
) -> tuple[Triangle, ...]:
    axis = (end - start).normalized(Vec3(0.0, 1.0, 0.0))
    tangent = axis.cross(Vec3(0.0, 1.0, 0.0)).normalized()
    if tangent.length_squared() <= 1e-12:
        tangent = axis.cross(Vec3(1.0, 0.0, 0.0)).normalized(Vec3(1.0, 0.0, 0.0))
    bitangent = axis.cross(tangent).normalized(Vec3(0.0, 0.0, 1.0))
    bottom = _oriented_ring(start, tangent, bitangent, start_radius, segments)
    top = _oriented_ring(end, tangent, bitangent, end_radius, segments)
    triangles: list[Triangle] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        normal_a = (bottom[index] - start).normalized()
        normal_b = (top[index] - end).normalized()
        normal_c = (bottom[next_index] - start).normalized()
        normal_d = (top[next_index] - end).normalized()
        triangles.append(Triangle(bottom[index], top[index], bottom[next_index], material, normal_a=normal_a, normal_b=normal_b, normal_c=normal_c))
        triangles.append(Triangle(bottom[next_index], top[index], top[next_index], material, normal_a=normal_c, normal_b=normal_b, normal_c=normal_d))
        triangles.append(Triangle(bottom[next_index], start, bottom[index], material, normal_a=-axis, normal_b=-axis, normal_c=-axis))
        triangles.append(Triangle(top[index], end, top[next_index], material, normal_a=axis, normal_b=axis, normal_c=axis))
    return tuple(triangles)


def _oriented_ring(center: Vec3, tangent: Vec3, bitangent: Vec3, radius: float, segments: int) -> tuple[Vec3, ...]:
    return tuple(center + tangent * (cos(index * tau / segments) * radius) + bitangent * (sin(index * tau / segments) * radius) for index in range(segments))


def _clip_water_polygon(corners: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float], ...]:
    polygon = list(corners)
    clipped: list[tuple[float, float, float]] = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        previous_inside = previous[2] >= 0.0
        current_inside = current[2] >= 0.0
        if current_inside != previous_inside:
            clipped.append(_waterline_intersection(previous, current))
        if current_inside:
            clipped.append(current)
    return tuple((point[0], point[1]) for point in clipped)


def _waterline_intersection(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    denominator = a[2] - b[2]
    amount = 0.0 if abs(denominator) <= 1e-9 else a[2] / denominator
    amount = clamp(amount, 0.0, 1.0)
    return (
        a[0] + (b[0] - a[0]) * amount,
        a[1] + (b[1] - a[1]) * amount,
        0.0,
    )


def _position_key(position: Vec3) -> tuple[int, int, int]:
    return (round(position.x * 1000.0), round(position.y * 1000.0), round(position.z * 1000.0))


def _stable_unit(seed: int, *values: int) -> float:
    return _stable_uint(seed, *values) / 0xFFFFFFFF


def _stable_uint(seed: int, *values: int) -> int:
    state = (seed ^ 0x9E3779B9) & 0xFFFFFFFF
    for value in values:
        item = int(value) & 0xFFFFFFFF
        state ^= item + 0x9E3779B9 + ((state << 6) & 0xFFFFFFFF) + (state >> 2)
        state = (state ^ (state >> 16)) & 0xFFFFFFFF
        state = (state * 0x85EBCA6B) & 0xFFFFFFFF
        state = (state ^ (state >> 13)) & 0xFFFFFFFF
        state = (state * 0xC2B2AE35) & 0xFFFFFFFF
        state = (state ^ (state >> 16)) & 0xFFFFFFFF
    return state


def ensure_procedural_world_assets(asset_dir: str | Path, config: ProceduralEnvironmentConfig | None = None) -> Path:
    """Write deterministic procedural prefab metadata beside a generated world."""

    active_config = config or ProceduralEnvironmentConfig()
    target = Path(asset_dir)
    target.mkdir(parents=True, exist_ok=True)
    grass_path = target / "grass-tufts.json"
    leaf_path = target / "maple-leaf-tufts.json"
    manifest_path = target / "manifest.json"

    _write_json_if_changed(grass_path, _prefab_asset_payload("grass_tuft_prefabs", (_grass_tuft_template(index, 5 + index) for index in range(4))))
    _write_json_if_changed(leaf_path, _prefab_asset_payload("maple_leaf_tuft_prefabs", (_maple_leaf_tuft_template(index, 3 + (index % 2)) for index in range(4))))
    manifest = {
        "format": PROCEDURAL_WORLD_ASSET_FORMAT,
        "world": "procedural_hills",
        "description": "Deterministic grass and maple leaf tuft prefabs used by procedural hill chunks.",
        "seed": active_config.seed,
        "chunk_size": active_config.chunk_size,
        "placement": {
            "grass_clump_candidates_per_chunk": active_config.grass_blades_per_chunk,
            "leaf_tufts_per_branch": active_config.leaf_tufts_per_branch,
            "tree_slots_per_axis": active_config.tree_slots_per_axis,
            "tree_density": active_config.tree_density,
        },
        "assets": [
            {
                "name": "grass_tuft_prefabs",
                "path": grass_path.name,
                "variants": 4,
                "kind": "low-poly crossed grass blade clumps",
            },
            {
                "name": "maple_leaf_tuft_prefabs",
                "path": leaf_path.name,
                "variants": 4,
                "kind": "3-4 leaf five-lobed maple tufts",
            },
            {
                "name": "distant_canopy_haze",
                "variants": len(CANOPY_HAZE_PALETTE),
                "kind": "layered low-poly canopy cards for blurred outer chunk rings",
            },
        ],
    }
    _write_json_if_changed(manifest_path, manifest)
    return manifest_path


def _prefab_asset_payload(name: str, templates: Iterable[tuple[Triangle, ...]]) -> dict[str, Any]:
    variants = []
    for index, triangles in enumerate(templates):
        variants.append(
            {
                "variant": index,
                "triangle_count": len(triangles),
                "triangles": [_triangle_asset_payload(triangle) for triangle in triangles],
            }
        )
    return {
        "format": PROCEDURAL_WORLD_ASSET_FORMAT,
        "name": name,
        "coordinate_space": "local_y_up",
        "variants": variants,
    }


def _triangle_asset_payload(triangle: Triangle) -> dict[str, Any]:
    payload = {
        "vertices": [
            [round(triangle.a.x, 6), round(triangle.a.y, 6), round(triangle.a.z, 6)],
            [round(triangle.b.x, 6), round(triangle.b.y, 6), round(triangle.b.z, 6)],
            [round(triangle.c.x, 6), round(triangle.c.y, 6), round(triangle.c.z, 6)],
        ],
        "material": _material_asset_payload(triangle.material),
    }
    if triangle.normal_a is not None and triangle.normal_b is not None and triangle.normal_c is not None:
        payload["normals"] = [
            [round(triangle.normal_a.x, 6), round(triangle.normal_a.y, 6), round(triangle.normal_a.z, 6)],
            [round(triangle.normal_b.x, 6), round(triangle.normal_b.y, 6), round(triangle.normal_b.z, 6)],
            [round(triangle.normal_c.x, 6), round(triangle.normal_c.y, 6), round(triangle.normal_c.z, 6)],
        ]
    return payload


def _material_asset_payload(material: Material) -> dict[str, Any]:
    return {
        "color": [material.color.r, material.color.g, material.color.b],
        "emission": [material.emission.r, material.emission.g, material.emission.b],
        "roughness": material.roughness,
        "fuzziness": material.fuzziness,
        "specular": material.specular,
        "light_transmission": material.light_transmission,
    }


def _write_json_if_changed(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def build_environment_chunk(config: ProceduralEnvironmentConfig, coord: tuple[int, int], *, tree_detail: bool = True) -> EnvironmentChunk:
    """Build one deterministic chunk for worker processes."""

    return ProceduralEnvironmentGenerator(config)._build_chunk(coord, tree_detail=tree_detail)
