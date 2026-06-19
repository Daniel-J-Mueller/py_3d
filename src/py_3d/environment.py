"""Deterministic infinite procedural environment generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, floor, sin, sqrt, tau
from typing import Callable, Iterable

from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .noise import FractalNoise3D
from .primitives import LineSet3, Mesh, Triangle
from .scene import Scene


GRASS_PALETTE = (
    Material(color=(64, 108, 59), roughness=0.76, fuzziness=0.1, specular=0.02),
    Material(color=(69, 116, 62), roughness=0.74, fuzziness=0.1, specular=0.02),
    Material(color=(73, 121, 65), roughness=0.74, fuzziness=0.1, specular=0.02),
)
MEADOW_PALETTE = (
    Material(color=(82, 126, 70), roughness=0.76, fuzziness=0.11, specular=0.02),
    Material(color=(88, 132, 73), roughness=0.76, fuzziness=0.11, specular=0.02),
    Material(color=(94, 138, 77), roughness=0.78, fuzziness=0.12, specular=0.02),
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
CANOPY_PALETTE = (
    Material(color=(43, 99, 55), emission=(3, 8, 3), diffuse=0.86, roughness=0.82, fuzziness=0.18, specular=0.015, light_transmission=0.18),
    Material(color=(53, 117, 62), emission=(4, 9, 4), diffuse=0.86, roughness=0.82, fuzziness=0.18, specular=0.015, light_transmission=0.2),
    Material(color=(67, 132, 69), emission=(5, 10, 4), diffuse=0.84, roughness=0.82, fuzziness=0.18, specular=0.015, light_transmission=0.22),
)
BUSH_PALETTE = (
    Material(color=(46, 103, 58), emission=(3, 7, 3), diffuse=0.86, roughness=0.84, fuzziness=0.18, specular=0.015, light_transmission=0.18),
    Material(color=(70, 126, 66), emission=(4, 8, 3), diffuse=0.84, roughness=0.84, fuzziness=0.18, specular=0.015, light_transmission=0.2),
)
LEAF_SPRIG = Material(color=(82, 145, 72), emission=(5, 10, 5), diffuse=0.82, roughness=0.86, fuzziness=0.2, specular=0.01, light_transmission=0.28)
GRASS_BLADE = Material(color=(92, 151, 74), roughness=0.88, fuzziness=0.2, specular=0.01)


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
    grass_blades_per_chunk: int = 80
    shoreline_detail: int = 2

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
    grass: LineSet3 | None
    shorelines: LineSet3 | None
    ripples: LineSet3 | None
    water_sources: tuple[WaterSource, ...]
    objects: tuple[object, ...]


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

    def _build_chunk(self, coord: tuple[int, int]) -> EnvironmentChunk:
        height_at = self._chunk_height_sampler()
        normal_at = self._chunk_normal_sampler(height_at)
        terrain = self._terrain_mesh(coord, height_at=height_at)
        water, sources, shorelines, ripples = self._water_mesh(coord, height_at=height_at)
        trees = self._trees(coord, height_at=height_at, normal_at=normal_at)
        bushes = self._bushes(coord, height_at=height_at, normal_at=normal_at)
        rocks = self._rocks(coord, height_at=height_at, normal_at=normal_at)
        grass = self._grass(coord, height_at=height_at, normal_at=normal_at)
        objects: list[object] = [terrain]
        if water is not None:
            objects.append(water)
        if shorelines is not None:
            objects.append(shorelines)
        if ripples is not None:
            objects.append(ripples)
        if grass is not None:
            objects.append(grass)
        for rock in rocks:
            objects.extend(_rock_primitives(rock))
        for bush in bushes:
            objects.extend(_bush_primitives(bush, self.config.seed))
        for tree in trees:
            objects.extend(_tree_primitives(tree, self.config.seed))
        return EnvironmentChunk(coord, terrain, water, trees, bushes, rocks, grass, shorelines, ripples, sources, tuple(objects))

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
                trunk_height = 5.2 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 139) * 5.6
                crown = 1.55 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 149) * 1.55
                radius = 0.18 + _stable_unit(self.config.seed, cx, cz, x_index, z_index, 157) * 0.16
                yaw = _stable_unit(self.config.seed, cx, cz, x_index, z_index, 167) * tau
                lean = Vec3(
                    (_stable_unit(self.config.seed, cx, cz, x_index, z_index, 169) - 0.5) * 0.42,
                    0.0,
                    (_stable_unit(self.config.seed, cx, cz, x_index, z_index, 173) - 0.5) * 0.42,
                )
                branch_count = 6 + int(_stable_unit(self.config.seed, cx, cz, x_index, z_index, 179) * 5.0)
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
    ) -> LineSet3 | None:
        sample_height = height_at or self.height_at
        normal_sample = normal_at or (lambda x, z: self.normal_at(x, z))
        cx, cz = coord
        size = self.config.chunk_size
        x0 = cx * size
        z0 = cz * size
        segments: list[tuple[Vec3, Vec3]] = []
        for index in range(self.config.grass_blades_per_chunk):
            x = x0 + _stable_unit(self.config.seed, cx, cz, index, 701) * size
            z = z0 + _stable_unit(self.config.seed, cx, cz, index, 709) * size
            y = sample_height(x, z)
            normal = normal_sample(x, z)
            if y <= self.config.water_level + 0.08 or normal.y < 0.72:
                continue
            cover = self._foliage_noise.sample_xyz(x * 0.11, 41.0, z * 0.11)
            if _stable_unit(self.config.seed, cx, cz, index, 719) > 0.35 + cover * 0.48:
                continue
            tuft_count = 2 + int(_stable_unit(self.config.seed, cx, cz, index, 721) * 3.0)
            base_angle = _stable_unit(self.config.seed, cx, cz, index, 733) * tau
            for blade in range(tuft_count):
                spread = (_stable_unit(self.config.seed, cx, cz, index, blade, 735) - 0.5) * 0.22
                angle = base_angle + blade * tau / max(1, tuft_count) + spread
                height_value = 0.16 + _stable_unit(self.config.seed, cx, cz, index, blade, 727) * 0.36
                bend = 0.022 + _stable_unit(self.config.seed, cx, cz, index, blade, 739) * 0.09
                root_offset = (_stable_unit(self.config.seed, cx, cz, index, blade, 743) - 0.5) * 0.08
                start = Vec3(x + cos(angle + 1.7) * root_offset, y + 0.018, z + sin(angle + 1.7) * root_offset)
                end = Vec3(start.x + cos(angle) * bend, y + height_value, start.z + sin(angle) * bend)
                segments.append((start, end))
        return LineSet3(segments, GRASS_BLADE) if segments else None


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


def _tree_primitives(tree: TreeInstance, seed: int) -> tuple[object, ...]:
    key = _position_key(tree.position)
    trunk_top = tree.position + Vec3(tree.lean.x, tree.trunk_height, tree.lean.z)
    branch_specs = _tree_branch_specs(tree, seed, key, trunk_top)
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
    puff_count = 4 + tree.species + int(_stable_unit(seed, *key, 811) * 3.0)
    leaf_segments: list[tuple[Vec3, Vec3]] = []
    leaf_triangles: list[Triangle] = []
    for index in range(puff_count):
        amount = index / max(1, puff_count)
        angle = tree.yaw + amount * tau * 1.618 + (_stable_unit(seed, *key, index, 821) - 0.5) * 0.9
        ring = tree.crown_radius * (0.12 + _stable_unit(seed, *key, index, 823) * 0.62)
        lift = (_stable_unit(seed, *key, index, 827) - 0.38) * tree.crown_radius * (0.78 if tree.species == 0 else 1.08)
        center = canopy_anchor + Vec3(cos(angle) * ring, lift, sin(angle) * ring)
        radius = tree.crown_radius * (0.58 + _stable_unit(seed, *key, index, 829) * 0.38)
        stretch = Vec3(
            cos(angle) * radius * (0.24 + _stable_unit(seed, *key, index, 831) * 0.22),
            radius * (0.08 + _stable_unit(seed, *key, index, 833) * 0.12),
            sin(angle) * radius * (0.24 + _stable_unit(seed, *key, index, 837) * 0.22),
        )
        material = CANOPY_PALETTE[(tree.material_variant + tree.species + index) % len(CANOPY_PALETTE)]
        leaf_triangles.extend(_ellipsoid_blob_triangles(center, radius, material, stretch=stretch, segments=5, rings=3))
        for sprig in range(2):
            sprig_angle = angle + (_stable_unit(seed, *key, index, sprig, 839) - 0.5) * 1.5
            sprig_lift = (_stable_unit(seed, *key, index, sprig, 841) - 0.45) * radius * 0.45
            start = center + Vec3(cos(sprig_angle) * radius * 0.18, sprig_lift, sin(sprig_angle) * radius * 0.18)
            end = start + Vec3(cos(sprig_angle) * radius * 0.46, radius * 0.12, sin(sprig_angle) * radius * 0.46)
            leaf_segments.append((start, end))

    if leaf_triangles:
        objects.append(Mesh(leaf_triangles))
    if leaf_segments:
        objects.append(LineSet3(leaf_segments, LEAF_SPRIG))

    return tuple(objects)


def _tree_branch_specs(tree: TreeInstance, seed: int, key: tuple[int, int, int], trunk_top: Vec3) -> tuple[tuple[Vec3, Vec3, float], ...]:
    specs: list[tuple[Vec3, Vec3, float]] = []
    for index in range(tree.branch_count):
        amount = 0.42 + index / max(1, tree.branch_count - 1) * 0.44
        start = tree.position + (trunk_top - tree.position) * amount
        angle = tree.yaw + index * tau / max(1, tree.branch_count) + (_stable_unit(seed, *key, index, 907) - 0.5) * 0.9
        length = tree.crown_radius * (0.58 + _stable_unit(seed, *key, index, 911) * 0.7)
        lift = tree.crown_radius * (0.12 + _stable_unit(seed, *key, index, 919) * 0.42)
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


def build_environment_chunk(config: ProceduralEnvironmentConfig, coord: tuple[int, int]) -> EnvironmentChunk:
    """Build one deterministic chunk for worker processes."""

    return ProceduralEnvironmentGenerator(config)._build_chunk(coord)
