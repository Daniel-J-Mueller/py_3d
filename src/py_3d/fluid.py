"""Small bounded blob fluid primitives."""

from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, floor, pi, sqrt

from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .primitives import BlobSurface


GPU_FLUID_COMPUTE_SHADER = """
#version 430

struct Particle {
    vec4 position_mass;
    vec4 velocity_pad;
};

layout(local_size_x = 128) in;
layout(std430, binding = 0) readonly buffer InputParticles {
    Particle input_particles[];
};
layout(std430, binding = 1) writeonly buffer OutputParticles {
    Particle output_particles[];
};

const int TILE_SIZE = 128;
shared Particle tile_particles[TILE_SIZE];

uniform int u_count;
uniform float u_dt;
uniform vec3 u_bounds_min;
uniform vec3 u_bounds_max;
uniform vec3 u_gravity;
uniform float u_rest_distance;
uniform float u_repel_strength;
uniform float u_self_attraction;
uniform float u_viscosity;
uniform float u_boundary_bounce;
uniform float u_close_damping;
uniform float u_attract_range_factor;
uniform int u_force_mode;
uniform vec3 u_force_center;
uniform float u_cylinder_radius;
uniform float u_time;
uniform float u_wind_strength;
uniform float u_blade_strength;

vec3 external_force(vec3 position, vec3 velocity) {
    vec3 local = position - u_force_center;
    if (u_force_mode == 1) {
        vec3 horizontal = vec3(local.x, 0.0, local.z);
        float distance = max(0.04, length(horizontal));
        vec3 tangent = normalize(vec3(-horizontal.z, 0.0, horizontal.x) + vec3(0.0, 0.0, 0.0001));
        float blade_phase = sin(u_time * 6.28318530718 * 3.1 + distance * 10.0);
        float fan_plane = exp(-(local.y * local.y) / 0.030);
        float surface = clamp((position.y - (u_force_center.y + 0.02)) / 0.30, 0.0, 1.0);
        float core = exp(-(distance * distance) / 0.055);
        float outer = smoothstep(0.14, 0.46, distance) * (1.0 - smoothstep(0.58, 0.95, distance));
        float wind_wave = sin(u_time * 6.28318530718 * 1.7 + position.z * 8.0);
        vec3 wind_shear = vec3(1.0, 0.035 * wind_wave, 0.14 * sin(u_time * 3.1 + position.x * 4.5)) * (0.38 * u_wind_strength * surface);
        vec3 swirl = tangent * ((1.52 * fan_plane + 0.42 * surface) / (0.48 + distance * 2.35));
        vec3 radial_pull = normalize(horizontal + vec3(0.0001, 0.0, 0.0001)) * (-0.42 * core * fan_plane + 0.10 * outer * surface);
        vec3 vertical = vec3(0.0, (-0.76 * core * fan_plane + 0.24 * outer * surface + 0.10 * blade_phase * fan_plane), 0.0);
        vec3 entrained_air = vec3(0.0, 0.16 * core * surface, 0.0);
        return wind_shear + (swirl + radial_pull + vertical + entrained_air) * u_blade_strength;
    }
    if (u_force_mode == 2) {
        float surface = clamp((position.y - (u_force_center.y - 0.12)) / 0.22, 0.0, 1.0);
        float lane = max(0.0, 1.0 - pow(local.z / max(0.001, u_cylinder_radius), 2.0));
        float gust = 0.76 + 0.24 * sin(u_time * 6.28318530718 * 1.35 + local.z * 4.4);
        float side = 0.22 * sin(u_time * 3.6 + local.x * 4.8);
        float lift = 0.08 * sin(u_time * 6.28318530718 * 2.2 + local.x * 7.0 + local.z * 2.0);
        return vec3(1.55 * u_wind_strength * lane * gust * surface, lift * surface, side * surface);
    }
    return vec3(0.0);
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    bool is_active = index < uint(u_count);

    Particle particle;
    vec3 position = vec3(0.0);
    float mass = 1.0;
    vec3 velocity = vec3(0.0);
    if (is_active) {
        particle = input_particles[index];
        position = particle.position_mass.xyz;
        mass = max(0.000001, particle.position_mass.w);
        velocity = particle.velocity_pad.xyz;
    }
    vec3 force = u_gravity * mass + external_force(position, velocity);
    float attract_range = u_rest_distance * u_attract_range_factor;
    float attract_range2 = attract_range * attract_range;

    int local_index = int(gl_LocalInvocationID.x);
    for (int tile_start = 0; tile_start < u_count; tile_start += TILE_SIZE) {
        int tile_particle_index = tile_start + local_index;
        if (tile_particle_index < u_count) {
            tile_particles[local_index] = input_particles[tile_particle_index];
        }
        barrier();

        int tile_count = min(TILE_SIZE, u_count - tile_start);
        for (int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
            int other_index = tile_start + tile_offset;
            if (other_index == int(index)) {
                continue;
            }
            Particle other = tile_particles[tile_offset];
            vec3 delta = other.position_mass.xyz - position;
            vec3 component_distance = abs(delta);
            if (component_distance.x > attract_range || component_distance.y > attract_range || component_distance.z > attract_range) {
                continue;
            }
            float distance2 = dot(delta, delta);
            if (distance2 <= 0.000000000001 || distance2 > attract_range2) {
                continue;
            }
            float distance = sqrt(distance2);
            vec3 direction = delta / distance;
            if (distance < u_rest_distance) {
                float amount = (u_rest_distance - distance) / u_rest_distance;
                force += direction * (-u_repel_strength * amount);
                force += (other.velocity_pad.xyz - velocity) * (u_close_damping * amount);
            } else {
                float amount = 1.0 - (distance - u_rest_distance) / max(0.000001, attract_range - u_rest_distance);
                force += direction * (u_self_attraction * amount);
            }
        }
        barrier();
    }

    if (!is_active) {
        return;
    }

    velocity = (velocity + force * (u_dt / mass)) * max(0.0, 1.0 - u_viscosity * u_dt);
    position += velocity * u_dt;

    if (position.x < u_bounds_min.x) {
        position.x = u_bounds_min.x;
        velocity.x = abs(velocity.x) * u_boundary_bounce;
    }
    if (position.x > u_bounds_max.x) {
        position.x = u_bounds_max.x;
        velocity.x = -abs(velocity.x) * u_boundary_bounce;
    }
    if (position.y < u_bounds_min.y) {
        position.y = u_bounds_min.y;
        velocity.y = abs(velocity.y) * u_boundary_bounce;
    }
    if (position.y > u_bounds_max.y) {
        position.y = u_bounds_max.y;
        velocity.y = -abs(velocity.y) * u_boundary_bounce;
    }
    if (position.z < u_bounds_min.z) {
        position.z = u_bounds_min.z;
        velocity.z = abs(velocity.z) * u_boundary_bounce;
    }
    if (position.z > u_bounds_max.z) {
        position.z = u_bounds_max.z;
        velocity.z = -abs(velocity.z) * u_boundary_bounce;
    }

    if (u_cylinder_radius > 0.0) {
        vec3 local = position - u_force_center;
        vec2 horizontal = local.xz;
        float distance = length(horizontal);
        if (distance > u_cylinder_radius) {
            vec2 normal = horizontal / max(0.000001, distance);
            position.xz = u_force_center.xz + normal * u_cylinder_radius;
            float outward = dot(velocity.xz, normal);
            if (outward > 0.0) {
                velocity.xz -= normal * outward * 1.25;
            }
        }
    }

    output_particles[index].position_mass = vec4(position, mass);
    output_particles[index].velocity_pad = vec4(velocity, 0.0);
}
"""


GPU_FLUID_BUCKET_SHADER = """
#version 430

struct Particle {
    vec4 position_mass;
    vec4 velocity_pad;
};

layout(local_size_x = 128) in;
layout(std430, binding = 0) readonly buffer InputParticles {
    Particle input_particles[];
};
layout(std430, binding = 2) buffer CellCounts {
    uint cell_counts[];
};
layout(std430, binding = 3) buffer CellIndices {
    int cell_indices[];
};

uniform int u_count;
uniform vec3 u_bounds_min;
uniform ivec3 u_grid_dims;
uniform float u_cell_size;
uniform int u_max_cell_particles;

int cell_index_for(vec3 position) {
    ivec3 cell = ivec3(floor((position - u_bounds_min) / max(0.0001, u_cell_size)));
    cell = clamp(cell, ivec3(0), u_grid_dims - ivec3(1));
    return (cell.z * u_grid_dims.y + cell.y) * u_grid_dims.x + cell.x;
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= uint(u_count)) {
        return;
    }
    int cell_index = cell_index_for(input_particles[index].position_mass.xyz);
    uint slot = atomicAdd(cell_counts[cell_index], 1u);
    if (slot < uint(u_max_cell_particles)) {
        cell_indices[cell_index * u_max_cell_particles + int(slot)] = int(index);
    }
}
"""


GPU_FLUID_GRID_COMPUTE_SHADER = """
#version 430

struct Particle {
    vec4 position_mass;
    vec4 velocity_pad;
};

layout(local_size_x = 128) in;
layout(std430, binding = 0) readonly buffer InputParticles {
    Particle input_particles[];
};
layout(std430, binding = 1) writeonly buffer OutputParticles {
    Particle output_particles[];
};
layout(std430, binding = 2) readonly buffer CellCounts {
    uint cell_counts[];
};
layout(std430, binding = 3) readonly buffer CellIndices {
    int cell_indices[];
};

uniform int u_count;
uniform float u_dt;
uniform vec3 u_bounds_min;
uniform vec3 u_bounds_max;
uniform vec3 u_gravity;
uniform float u_rest_distance;
uniform float u_repel_strength;
uniform float u_self_attraction;
uniform float u_viscosity;
uniform float u_boundary_bounce;
uniform float u_close_damping;
uniform float u_attract_range_factor;
uniform int u_force_mode;
uniform vec3 u_force_center;
uniform float u_cylinder_radius;
uniform float u_time;
uniform float u_wind_strength;
uniform float u_blade_strength;
uniform ivec3 u_grid_dims;
uniform float u_cell_size;
uniform int u_max_cell_particles;
uniform int u_neighbor_limit;

vec3 external_force(vec3 position, vec3 velocity) {
    vec3 local = position - u_force_center;
    if (u_force_mode == 1) {
        vec3 horizontal = vec3(local.x, 0.0, local.z);
        float distance = max(0.04, length(horizontal));
        vec3 tangent = normalize(vec3(-horizontal.z, 0.0, horizontal.x) + vec3(0.0, 0.0, 0.0001));
        float blade_phase = sin(u_time * 6.28318530718 * 3.1 + distance * 10.0);
        float fan_plane = exp(-(local.y * local.y) / 0.030);
        float surface = clamp((position.y - (u_force_center.y + 0.02)) / 0.30, 0.0, 1.0);
        float core = exp(-(distance * distance) / 0.055);
        float outer = smoothstep(0.14, 0.46, distance) * (1.0 - smoothstep(0.58, 0.95, distance));
        float wind_wave = sin(u_time * 6.28318530718 * 1.7 + position.z * 8.0);
        vec3 wind_shear = vec3(1.0, 0.035 * wind_wave, 0.14 * sin(u_time * 3.1 + position.x * 4.5)) * (0.38 * u_wind_strength * surface);
        vec3 swirl = tangent * ((1.52 * fan_plane + 0.42 * surface) / (0.48 + distance * 2.35));
        vec3 radial_pull = normalize(horizontal + vec3(0.0001, 0.0, 0.0001)) * (-0.42 * core * fan_plane + 0.10 * outer * surface);
        vec3 vertical = vec3(0.0, (-0.76 * core * fan_plane + 0.24 * outer * surface + 0.10 * blade_phase * fan_plane), 0.0);
        vec3 entrained_air = vec3(0.0, 0.16 * core * surface, 0.0);
        return wind_shear + (swirl + radial_pull + vertical + entrained_air) * u_blade_strength;
    }
    if (u_force_mode == 2) {
        float surface = clamp((position.y - (u_force_center.y - 0.12)) / 0.22, 0.0, 1.0);
        float lane = max(0.0, 1.0 - pow(local.z / max(0.001, u_cylinder_radius), 2.0));
        float gust = 0.76 + 0.24 * sin(u_time * 6.28318530718 * 1.35 + local.z * 4.4);
        float side = 0.22 * sin(u_time * 3.6 + local.x * 4.8);
        float lift = 0.08 * sin(u_time * 6.28318530718 * 2.2 + local.x * 7.0 + local.z * 2.0);
        return vec3(1.55 * u_wind_strength * lane * gust * surface, lift * surface, side * surface);
    }
    return vec3(0.0);
}

ivec3 cell_coord_for(vec3 position) {
    ivec3 cell = ivec3(floor((position - u_bounds_min) / max(0.0001, u_cell_size)));
    return clamp(cell, ivec3(0), u_grid_dims - ivec3(1));
}

int cell_index_for(ivec3 cell) {
    return (cell.z * u_grid_dims.y + cell.y) * u_grid_dims.x + cell.x;
}

bool accumulate_particle_force(
    inout vec3 force,
    vec3 position,
    vec3 velocity,
    int self_index,
    int other_index,
    float attract_range,
    float attract_range2,
    inout int interactions
) {
    if (other_index == self_index || other_index < 0 || interactions >= u_neighbor_limit) {
        return false;
    }
    Particle other = input_particles[other_index];
    vec3 delta = other.position_mass.xyz - position;
    vec3 component_distance = abs(delta);
    if (component_distance.x > attract_range || component_distance.y > attract_range || component_distance.z > attract_range) {
        return false;
    }
    float distance2 = dot(delta, delta);
    if (distance2 <= 0.000000000001 || distance2 > attract_range2) {
        return false;
    }
    float distance = sqrt(distance2);
    vec3 direction = delta / distance;
    if (distance < u_rest_distance) {
        float amount = (u_rest_distance - distance) / u_rest_distance;
        force += direction * (-u_repel_strength * amount);
        force += (other.velocity_pad.xyz - velocity) * (u_close_damping * amount);
    } else {
        float amount = 1.0 - (distance - u_rest_distance) / max(0.000001, attract_range - u_rest_distance);
        force += direction * (u_self_attraction * amount);
    }
    interactions += 1;
    return true;
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= uint(u_count)) {
        return;
    }

    Particle particle = input_particles[index];
    vec3 position = particle.position_mass.xyz;
    float mass = max(0.000001, particle.position_mass.w);
    vec3 velocity = particle.velocity_pad.xyz;
    vec3 force = u_gravity * mass + external_force(position, velocity);
    float attract_range = max(0.0001, u_rest_distance * u_attract_range_factor);
    float attract_range2 = attract_range * attract_range;
    ivec3 center_cell = cell_coord_for(position);
    int interactions = 0;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (interactions >= u_neighbor_limit) {
                    continue;
                }
                ivec3 cell = center_cell + ivec3(dx, dy, dz);
                if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, u_grid_dims))) {
                    continue;
                }
                int cell_index = cell_index_for(cell);
                int slots = int(min(cell_counts[cell_index], uint(u_max_cell_particles)));
                for (int slot = 0; slot < slots; ++slot) {
                    int other_index = cell_indices[cell_index * u_max_cell_particles + slot];
                    accumulate_particle_force(force, position, velocity, int(index), other_index, attract_range, attract_range2, interactions);
                    if (interactions >= u_neighbor_limit) {
                        break;
                    }
                }
            }
        }
    }

    velocity = (velocity + force * (u_dt / mass)) * max(0.0, 1.0 - u_viscosity * u_dt);
    position += velocity * u_dt;

    if (position.x < u_bounds_min.x) {
        position.x = u_bounds_min.x;
        velocity.x = abs(velocity.x) * u_boundary_bounce;
    }
    if (position.x > u_bounds_max.x) {
        position.x = u_bounds_max.x;
        velocity.x = -abs(velocity.x) * u_boundary_bounce;
    }
    if (position.y < u_bounds_min.y) {
        position.y = u_bounds_min.y;
        velocity.y = abs(velocity.y) * u_boundary_bounce;
    }
    if (position.y > u_bounds_max.y) {
        position.y = u_bounds_max.y;
        velocity.y = -abs(velocity.y) * u_boundary_bounce;
    }
    if (position.z < u_bounds_min.z) {
        position.z = u_bounds_min.z;
        velocity.z = abs(velocity.z) * u_boundary_bounce;
    }
    if (position.z > u_bounds_max.z) {
        position.z = u_bounds_max.z;
        velocity.z = -abs(velocity.z) * u_boundary_bounce;
    }

    if (u_cylinder_radius > 0.0) {
        vec3 local = position - u_force_center;
        vec2 horizontal = local.xz;
        float distance = length(horizontal);
        if (distance > u_cylinder_radius) {
            vec2 normal = horizontal / max(0.000001, distance);
            position.xz = u_force_center.xz + normal * u_cylinder_radius;
            float outward = dot(velocity.xz, normal);
            if (outward > 0.0) {
                velocity.xz -= normal * outward * 1.25;
            }
        }
    }

    output_particles[index].position_mass = vec4(position, mass);
    output_particles[index].velocity_pad = vec4(velocity, 0.0);
}
"""


def _radius_from_volume(volume: float) -> float:
    return ((3.0 * volume) / (4.0 * pi)) ** (1.0 / 3.0)


def _volume_from_radius(radius: float) -> float:
    return 4.0 * pi * radius * radius * radius / 3.0


def _force_components(value: Vec3 | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, Vec3):
        return value.x, value.y, value.z
    return float(value[0]), float(value[1]), float(value[2])


@dataclass
class VectorFluidParticle:
    """A single particle in a vector-cloud fluid."""

    position: Vec3 | tuple[float, float, float]
    velocity: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    mass: float = 1.0

    def __post_init__(self) -> None:
        self.position = as_vec3(self.position)
        self.velocity = as_vec3(self.velocity)
        self.mass = max(1e-6, float(self.mass))


@dataclass
class VectorFluidWorld:
    """A light vector-cloud fluid model with liquid/gas behavior.

    Liquids use short-range repulsion plus medium-range attraction. Gases can
    share the same particles while setting ``self_attraction`` to zero.
    """

    bounds_min: Vec3 | tuple[float, float, float] = Vec3(-2.0, 0.0, -2.0)
    bounds_max: Vec3 | tuple[float, float, float] = Vec3(2.0, 2.0, 2.0)
    gravity: Vec3 | tuple[float, float, float] = Vec3(0.0, -9.81, 0.0)
    rest_distance: float = 0.18
    repel_strength: float = 10.0
    self_attraction: float = 2.6
    viscosity: float = 0.18
    boundary_bounce: float = 0.18
    close_damping: float = 5.0
    attract_range_factor: float = 3.1

    def __post_init__(self) -> None:
        self.bounds_min = as_vec3(self.bounds_min)
        self.bounds_max = as_vec3(self.bounds_max)
        self.gravity = as_vec3(self.gravity)
        self.rest_distance = max(1e-4, float(self.rest_distance))
        self.repel_strength = max(0.0, float(self.repel_strength))
        self.self_attraction = max(0.0, float(self.self_attraction))
        self.viscosity = clamp(float(self.viscosity), 0.0, 1.0)
        self.boundary_bounce = clamp(float(self.boundary_bounce), 0.0, 1.0)
        self.close_damping = max(0.0, float(self.close_damping))
        self.attract_range_factor = max(1.01, float(self.attract_range_factor))
        self.particles: list[VectorFluidParticle] = []

    @classmethod
    def liquid(cls, **kwargs) -> "VectorFluidWorld":
        kwargs.setdefault("gravity", Vec3(0.0, -9.81, 0.0))
        kwargs.setdefault("self_attraction", 2.6)
        kwargs.setdefault("close_damping", 5.0)
        return cls(**kwargs)

    @classmethod
    def gas(cls, **kwargs) -> "VectorFluidWorld":
        kwargs.setdefault("gravity", Vec3(0.0, 0.0, 0.0))
        kwargs.setdefault("self_attraction", 0.0)
        kwargs.setdefault("viscosity", 0.08)
        kwargs.setdefault("close_damping", 2.0)
        return cls(**kwargs)

    def add_particle(self, particle: VectorFluidParticle) -> VectorFluidParticle:
        self.particles.append(particle)
        return particle

    def step(
        self,
        dt: float,
        substeps: int = 1,
        external_force: Callable[[VectorFluidParticle, float], Vec3 | tuple[float, float, float]] | None = None,
    ) -> None:
        if dt < 0.0:
            raise ValueError("fluid dt must be non-negative")
        if substeps <= 0:
            raise ValueError("fluid substeps must be positive")
        step_dt = dt / substeps
        for _ in range(substeps):
            self._step_once(step_dt, external_force)

    def _step_once(self, dt: float, external_force: Callable[[VectorFluidParticle, float], Vec3 | tuple[float, float, float]] | None) -> None:
        particles = self.particles
        count = len(particles)
        if count == 0:
            return
        gravity = self.gravity
        force_x = [gravity.x * particle.mass for particle in particles]
        force_y = [gravity.y * particle.mass for particle in particles]
        force_z = [gravity.z * particle.mass for particle in particles]
        velocity_x = [particle.velocity.x for particle in particles]
        velocity_y = [particle.velocity.y for particle in particles]
        velocity_z = [particle.velocity.z for particle in particles]

        attract_range = self.rest_distance * self.attract_range_factor
        self._accumulate_neighbor_forces(dt, attract_range, force_x, force_y, force_z, velocity_x, velocity_y, velocity_z)

        if external_force is not None:
            for index, particle in enumerate(particles):
                particle.velocity = Vec3(velocity_x[index], velocity_y[index], velocity_z[index])
                external = external_force(particle, dt)
                external_x, external_y, external_z = _force_components(external)
                force_x[index] += external_x
                force_y[index] += external_y
                force_z[index] += external_z

        damping = max(0.0, 1.0 - self.viscosity * dt)
        min_x, min_y, min_z = self.bounds_min.x, self.bounds_min.y, self.bounds_min.z
        max_x, max_y, max_z = self.bounds_max.x, self.bounds_max.y, self.bounds_max.z
        bounce = self.boundary_bounce
        for index, particle in enumerate(particles):
            mass_dt = dt / particle.mass
            vx = (velocity_x[index] + force_x[index] * mass_dt) * damping
            vy = (velocity_y[index] + force_y[index] * mass_dt) * damping
            vz = (velocity_z[index] + force_z[index] * mass_dt) * damping
            px = particle.position.x + vx * dt
            py = particle.position.y + vy * dt
            pz = particle.position.z + vz * dt
            if px < min_x:
                px = min_x
                vx = abs(vx) * bounce
            elif px > max_x:
                px = max_x
                vx = -abs(vx) * bounce
            if py < min_y:
                py = min_y
                vy = abs(vy) * bounce
            elif py > max_y:
                py = max_y
                vy = -abs(vy) * bounce
            if pz < min_z:
                pz = min_z
                vz = abs(vz) * bounce
            elif pz > max_z:
                pz = max_z
                vz = -abs(vz) * bounce
            particle.velocity = Vec3(vx, vy, vz)
            particle.position = Vec3(px, py, pz)

    def _accumulate_neighbor_forces(
        self,
        dt: float,
        interaction_range: float,
        force_x: list[float],
        force_y: list[float],
        force_z: list[float],
        velocity_x: list[float],
        velocity_y: list[float],
        velocity_z: list[float],
    ) -> None:
        particles = self.particles
        if len(particles) < 2:
            return
        cell_size = max(interaction_range, self.rest_distance)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, particle in enumerate(particles):
            position = particle.position
            key = (
                floor(position.x / cell_size),
                floor(position.y / cell_size),
                floor(position.z / cell_size),
            )
            buckets.setdefault(key, []).append(index)

        interaction_range2 = interaction_range * interaction_range
        rest_distance = self.rest_distance
        rest_inv = 1.0 / max(1e-6, rest_distance)
        attract_span_inv = 1.0 / max(1e-6, interaction_range - rest_distance)
        repel_strength = self.repel_strength
        self_attraction = self.self_attraction
        close_damping = self.close_damping
        close_dt = max(0.0, dt)

        for key, indices in buckets.items():
            key_x, key_y, key_z = key
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    for dz_cell in (-1, 0, 1):
                        neighbor_key = (key_x + dx_cell, key_y + dy_cell, key_z + dz_cell)
                        if neighbor_key not in buckets or neighbor_key < key:
                            continue
                        other_indices = buckets[neighbor_key]
                        if neighbor_key == key:
                            for offset, first_index in enumerate(indices):
                                for second_index in indices[offset + 1 :]:
                                    self._accumulate_pair_force(
                                        first_index,
                                        second_index,
                                        interaction_range,
                                        interaction_range2,
                                        rest_distance,
                                        rest_inv,
                                        attract_span_inv,
                                        repel_strength,
                                        self_attraction,
                                        close_damping,
                                        close_dt,
                                        force_x,
                                        force_y,
                                        force_z,
                                        velocity_x,
                                        velocity_y,
                                        velocity_z,
                                    )
                        else:
                            for first_index in indices:
                                for second_index in other_indices:
                                    self._accumulate_pair_force(
                                        first_index,
                                        second_index,
                                        interaction_range,
                                        interaction_range2,
                                        rest_distance,
                                        rest_inv,
                                        attract_span_inv,
                                        repel_strength,
                                        self_attraction,
                                        close_damping,
                                        close_dt,
                                        force_x,
                                        force_y,
                                        force_z,
                                        velocity_x,
                                        velocity_y,
                                        velocity_z,
                                    )

    def _accumulate_pair_force(
        self,
        first_index: int,
        second_index: int,
        interaction_range: float,
        interaction_range2: float,
        rest_distance: float,
        rest_inv: float,
        attract_span_inv: float,
        repel_strength: float,
        self_attraction: float,
        close_damping: float,
        close_dt: float,
        force_x: list[float],
        force_y: list[float],
        force_z: list[float],
        velocity_x: list[float],
        velocity_y: list[float],
        velocity_z: list[float],
    ) -> None:
        first_position = self.particles[first_index].position
        second_position = self.particles[second_index].position
        dx = second_position.x - first_position.x
        if abs(dx) > interaction_range:
            return
        dy = second_position.y - first_position.y
        if abs(dy) > interaction_range:
            return
        dz = second_position.z - first_position.z
        if abs(dz) > interaction_range:
            return
        distance2 = dx * dx + dy * dy + dz * dz
        if distance2 <= 1e-12 or distance2 > interaction_range2:
            return
        distance = sqrt(distance2)
        inv_distance = 1.0 / distance
        direction_x = dx * inv_distance
        direction_y = dy * inv_distance
        direction_z = dz * inv_distance
        if distance < rest_distance:
            amount = (rest_distance - distance) * rest_inv
            scalar = -repel_strength * amount
            damp_amount = min(0.45, close_damping * amount * close_dt)
            if damp_amount > 0.0:
                keep = 1.0 - damp_amount
                average_x = (velocity_x[first_index] + velocity_x[second_index]) * 0.5
                average_y = (velocity_y[first_index] + velocity_y[second_index]) * 0.5
                average_z = (velocity_z[first_index] + velocity_z[second_index]) * 0.5
                velocity_x[first_index] = velocity_x[first_index] * keep + average_x * damp_amount
                velocity_y[first_index] = velocity_y[first_index] * keep + average_y * damp_amount
                velocity_z[first_index] = velocity_z[first_index] * keep + average_z * damp_amount
                velocity_x[second_index] = velocity_x[second_index] * keep + average_x * damp_amount
                velocity_y[second_index] = velocity_y[second_index] * keep + average_y * damp_amount
                velocity_z[second_index] = velocity_z[second_index] * keep + average_z * damp_amount
        else:
            amount = 1.0 - (distance - rest_distance) * attract_span_inv
            scalar = self_attraction * amount
        fx = direction_x * scalar
        fy = direction_y * scalar
        fz = direction_z * scalar
        force_x[first_index] += fx
        force_y[first_index] += fy
        force_z[first_index] += fz
        force_x[second_index] -= fx
        force_y[second_index] -= fy
        force_z[second_index] -= fz

    def _resolve_bounds(self, particle: VectorFluidParticle) -> None:
        x, vx = _bounded_axis(particle.position.x, particle.velocity.x, self.bounds_min.x, self.bounds_max.x, self.boundary_bounce)
        y, vy = _bounded_axis(particle.position.y, particle.velocity.y, self.bounds_min.y, self.bounds_max.y, self.boundary_bounce)
        z, vz = _bounded_axis(particle.position.z, particle.velocity.z, self.bounds_min.z, self.bounds_max.z, self.boundary_bounce)
        particle.position = Vec3(x, y, z)
        particle.velocity = Vec3(vx, vy, vz)

    def _neighbor_pairs(self, interaction_range: float) -> tuple[tuple[int, int], ...]:
        if len(self.particles) < 2:
            return ()
        cell_size = max(interaction_range, self.rest_distance)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, particle in enumerate(self.particles):
            key = (
                floor(particle.position.x / cell_size),
                floor(particle.position.y / cell_size),
                floor(particle.position.z / cell_size),
            )
            buckets.setdefault(key, []).append(index)

        pairs: list[tuple[int, int]] = []
        for key, indices in buckets.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                        if neighbor_key not in buckets or neighbor_key < key:
                            continue
                        other_indices = buckets[neighbor_key]
                        if neighbor_key == key:
                            for offset, first_index in enumerate(indices):
                                for second_index in indices[offset + 1 :]:
                                    pairs.append((first_index, second_index))
                        else:
                            for first_index in indices:
                                for second_index in other_indices:
                                    pairs.append((first_index, second_index))
        return tuple(pairs)

    def _dampen_close_pair(self, first: VectorFluidParticle, second: VectorFluidParticle, closeness: float, dt: float) -> None:
        amount = min(0.45, self.close_damping * closeness * max(0.0, dt))
        if amount <= 0.0:
            return
        average = (first.velocity + second.velocity) * 0.5
        first.velocity = first.velocity * (1.0 - amount) + average * amount
        second.velocity = second.velocity * (1.0 - amount) + average * amount


class GPUVectorFluidWorld(VectorFluidWorld):
    """OpenGL compute-shader particle fluid with a VectorFluidWorld-compatible API."""

    backend = "gpu"

    def __init__(self, ctx, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ctx = ctx
        self._compute_program = None
        self._bucket_program = None
        self._grid_compute_program = None
        self._buffers = []
        self._grid_counts_buffer = None
        self._grid_indices_buffer = None
        self._grid_signature: tuple[tuple[int, int, int], float, int] | None = None
        self._grid_zero_payload = b""
        self._read_index = 0
        self._initialized = False
        self.force_mode = 0
        self.force_center = Vec3(0.0, 0.0, 0.0)
        self.cylinder_radius = 0.0
        self.time = 0.0
        self.wind_strength = 1.0
        self.blade_strength = 1.0
        self.sync_python_particles = True
        self.readback_particles = True
        self.neighbor_search_enabled = True
        self.spatial_grid_min_particles = 3072
        self.neighbor_limit = 72
        self.cell_particle_capacity = 96
        self.spatial_grid_active = False
        self._particle_array = None

    @classmethod
    def liquid(cls, ctx, **kwargs) -> "GPUVectorFluidWorld":
        kwargs.setdefault("gravity", Vec3(0.0, -9.81, 0.0))
        kwargs.setdefault("self_attraction", 2.6)
        kwargs.setdefault("close_damping", 5.0)
        return cls(ctx, **kwargs)

    @classmethod
    def gas(cls, ctx, **kwargs) -> "GPUVectorFluidWorld":
        kwargs.setdefault("gravity", Vec3(0.0, 0.0, 0.0))
        kwargs.setdefault("self_attraction", 0.0)
        kwargs.setdefault("viscosity", 0.08)
        kwargs.setdefault("close_damping", 2.0)
        return cls(ctx, **kwargs)

    @property
    def gpu_enabled(self) -> bool:
        return True

    def step(
        self,
        dt: float,
        substeps: int = 1,
        external_force: Callable[[VectorFluidParticle, float], Vec3 | tuple[float, float, float]] | None = None,
    ) -> None:
        if external_force is not None:
            super().step(dt, substeps=substeps, external_force=external_force)
            return
        if dt < 0.0:
            raise ValueError("fluid dt must be non-negative")
        if substeps <= 0:
            raise ValueError("fluid substeps must be positive")
        if not self.particles:
            return
        self._ensure_gpu_state()
        step_dt = dt / substeps
        for _ in range(substeps):
            self._run_compute_step(step_dt)
        if self.sync_python_particles or self.readback_particles:
            self._sync_particles_from_gpu()
        else:
            self._particle_array = None

    def close(self) -> None:
        for buffer in (*self._buffers, self._grid_counts_buffer, self._grid_indices_buffer):
            if buffer is None:
                continue
            try:
                buffer.release()
            except Exception:
                pass
        self._buffers = []
        self._grid_counts_buffer = None
        self._grid_indices_buffer = None
        self._grid_signature = None
        self._grid_zero_payload = b""
        self._initialized = False

    def _ensure_gpu_state(self) -> None:
        if self._initialized:
            return
        if not hasattr(self._ctx, "compute_shader") or getattr(self._ctx, "version_code", 0) < 430:
            raise RuntimeError("GPU fluid requires an OpenGL 4.3 compute-capable ModernGL context")
        self._compute_program = self._ctx.compute_shader(GPU_FLUID_COMPUTE_SHADER)
        try:
            self._bucket_program = self._ctx.compute_shader(GPU_FLUID_BUCKET_SHADER)
            self._grid_compute_program = self._ctx.compute_shader(GPU_FLUID_GRID_COMPUTE_SHADER)
        except Exception:
            self._bucket_program = None
            self._grid_compute_program = None
            self.neighbor_search_enabled = False
        payload = self._particle_payload()
        self._buffers = [self._ctx.buffer(payload), self._ctx.buffer(reserve=len(payload))]
        self._read_index = 0
        self._initialized = True

    def _particle_payload(self) -> bytes:
        payload = array("f")
        for particle in self.particles:
            payload.extend(
                (
                    particle.position.x,
                    particle.position.y,
                    particle.position.z,
                    particle.mass,
                    particle.velocity.x,
                    particle.velocity.y,
                    particle.velocity.z,
                    0.0,
                )
            )
        return payload.tobytes()

    def _run_compute_step(self, dt: float) -> None:
        use_spatial_grid = self.neighbor_search_enabled and len(self.particles) >= int(self.spatial_grid_min_particles)
        if use_spatial_grid and self._bucket_program is not None and self._grid_compute_program is not None:
            try:
                self._run_grid_compute_step(dt)
                self.spatial_grid_active = True
                return
            except Exception:
                self.neighbor_search_enabled = False
                self.spatial_grid_active = False
        self._run_tiled_compute_step(dt)
        self.spatial_grid_active = False

    def _run_tiled_compute_step(self, dt: float) -> None:
        assert self._compute_program is not None
        read_buffer = self._buffers[self._read_index]
        write_index = 1 - self._read_index
        write_buffer = self._buffers[write_index]
        read_buffer.bind_to_storage_buffer(0)
        write_buffer.bind_to_storage_buffer(1)
        self._apply_compute_uniforms(self._compute_program, dt)
        groups = (len(self.particles) + 127) // 128
        self._compute_program.run(group_x=groups)
        self._read_index = write_index
        self._memory_barrier()

    def _run_grid_compute_step(self, dt: float) -> None:
        assert self._bucket_program is not None
        assert self._grid_compute_program is not None
        if not self._ensure_grid_buffers():
            self._run_tiled_compute_step(dt)
            return
        assert self._grid_counts_buffer is not None
        assert self._grid_indices_buffer is not None
        read_buffer = self._buffers[self._read_index]
        write_index = 1 - self._read_index
        write_buffer = self._buffers[write_index]
        try:
            self._grid_counts_buffer.clear()
        except Exception:
            self._grid_counts_buffer.write(self._grid_zero_payload)
        read_buffer.bind_to_storage_buffer(0)
        self._grid_counts_buffer.bind_to_storage_buffer(2)
        self._grid_indices_buffer.bind_to_storage_buffer(3)
        self._apply_bucket_uniforms(self._bucket_program)
        groups = (len(self.particles) + 127) // 128
        self._bucket_program.run(group_x=groups)
        self._memory_barrier()

        read_buffer.bind_to_storage_buffer(0)
        write_buffer.bind_to_storage_buffer(1)
        self._grid_counts_buffer.bind_to_storage_buffer(2)
        self._grid_indices_buffer.bind_to_storage_buffer(3)
        self._apply_compute_uniforms(self._grid_compute_program, dt)
        self._apply_grid_uniforms(self._grid_compute_program)
        self._set_uniform("u_neighbor_limit", max(1, int(self.neighbor_limit)), self._grid_compute_program)
        self._grid_compute_program.run(group_x=groups)
        self._read_index = write_index
        self._memory_barrier()

    def _ensure_grid_buffers(self) -> bool:
        signature = self._neighbor_grid_signature()
        if signature is None:
            return False
        if self._grid_signature == signature and self._grid_counts_buffer is not None and self._grid_indices_buffer is not None:
            return True
        for buffer in (self._grid_counts_buffer, self._grid_indices_buffer):
            if buffer is None:
                continue
            try:
                buffer.release()
            except Exception:
                pass
        dims, _cell_size, capacity = signature
        cell_count = dims[0] * dims[1] * dims[2]
        if cell_count <= 0:
            self._grid_counts_buffer = None
            self._grid_indices_buffer = None
            self._grid_signature = None
            return False
        self._grid_counts_buffer = self._ctx.buffer(reserve=cell_count * 4)
        self._grid_indices_buffer = self._ctx.buffer(reserve=cell_count * capacity * 4)
        self._grid_zero_payload = b"\x00" * (cell_count * 4)
        self._grid_signature = signature
        return True

    def _neighbor_grid_signature(self) -> tuple[tuple[int, int, int], float, int] | None:
        attract_range = max(0.0001, float(self.rest_distance) * float(self.attract_range_factor))
        cell_size = max(attract_range, float(self.rest_distance), 0.0001)
        extent = self.bounds_max - self.bounds_min
        dims = (
            max(1, int(ceil(max(0.0001, extent.x) / cell_size))),
            max(1, int(ceil(max(0.0001, extent.y) / cell_size))),
            max(1, int(ceil(max(0.0001, extent.z) / cell_size))),
        )
        capacity = max(4, int(self.cell_particle_capacity))
        cell_count = dims[0] * dims[1] * dims[2]
        if cell_count <= 0 or cell_count > 250000:
            return None
        return dims, cell_size, capacity

    def _apply_compute_uniforms(self, program, dt: float) -> None:
        self._set_uniform("u_count", len(self.particles), program)
        self._set_uniform("u_dt", float(dt), program)
        self._set_uniform("u_bounds_min", self.bounds_min.as_tuple(), program)
        self._set_uniform("u_bounds_max", self.bounds_max.as_tuple(), program)
        self._set_uniform("u_gravity", self.gravity.as_tuple(), program)
        self._set_uniform("u_rest_distance", self.rest_distance, program)
        self._set_uniform("u_repel_strength", self.repel_strength, program)
        self._set_uniform("u_self_attraction", self.self_attraction, program)
        self._set_uniform("u_viscosity", self.viscosity, program)
        self._set_uniform("u_boundary_bounce", self.boundary_bounce, program)
        self._set_uniform("u_close_damping", self.close_damping, program)
        self._set_uniform("u_attract_range_factor", self.attract_range_factor, program)
        self._set_uniform("u_force_mode", int(self.force_mode), program)
        self._set_uniform("u_force_center", self.force_center.as_tuple(), program)
        self._set_uniform("u_cylinder_radius", float(self.cylinder_radius), program)
        self._set_uniform("u_time", float(self.time), program)
        self._set_uniform("u_wind_strength", float(self.wind_strength), program)
        self._set_uniform("u_blade_strength", float(self.blade_strength), program)

    def _apply_bucket_uniforms(self, program) -> None:
        self._set_uniform("u_count", len(self.particles), program)
        self._set_uniform("u_bounds_min", self.bounds_min.as_tuple(), program)
        self._apply_grid_uniforms(program)

    def _apply_grid_uniforms(self, program) -> None:
        if self._grid_signature is None:
            return
        dims, cell_size, capacity = self._grid_signature
        self._set_uniform("u_grid_dims", dims, program)
        self._set_uniform("u_cell_size", float(cell_size), program)
        self._set_uniform("u_max_cell_particles", int(capacity), program)

    def _memory_barrier(self) -> None:
        barrier = getattr(self._ctx, "memory_barrier", None)
        if callable(barrier):
            try:
                barrier()
            except TypeError:
                try:
                    barrier(0xFFFFFFFF)
                except Exception:
                    pass
            except Exception:
                pass

    def _sync_particles_from_gpu(self) -> None:
        finish = getattr(self._ctx, "finish", None)
        if callable(finish):
            finish()
        raw = self._buffers[self._read_index].read()
        try:
            import numpy

            self._particle_array = numpy.frombuffer(raw, dtype=numpy.float32).reshape((len(self.particles), 8))
            if not self.sync_python_particles:
                return
            values = self._particle_array
            for index, particle in enumerate(self.particles):
                row = values[index]
                particle.position = Vec3(float(row[0]), float(row[1]), float(row[2]))
                particle.mass = max(1e-6, float(row[3]))
                particle.velocity = Vec3(float(row[4]), float(row[5]), float(row[6]))
            return
        except Exception:
            self._particle_array = None
        values = array("f")
        values.frombytes(raw)
        stride = 8
        for index, particle in enumerate(self.particles):
            offset = index * stride
            particle.position = Vec3(values[offset], values[offset + 1], values[offset + 2])
            particle.mass = max(1e-6, values[offset + 3])
            particle.velocity = Vec3(values[offset + 4], values[offset + 5], values[offset + 6])

    def particle_position_velocity_arrays(self):
        if self._particle_array is None:
            return None
        return self._particle_array[:, 0:3], self._particle_array[:, 4:7]

    def bind_particle_buffer(self, binding: int) -> bool:
        if not self._initialized or not self._buffers:
            return False
        self._buffers[self._read_index].bind_to_storage_buffer(int(binding))
        return True

    def _set_uniform(self, name: str, value, program=None) -> None:
        if program is None:
            program = self._compute_program
        if program is None:
            return
        try:
            uniform = program[name]
        except Exception:
            return
        uniform.value = value


@dataclass
class FluidBlob:
    """A fixed-volume, bounded blob used for early slime-like fluid demos."""

    position: Vec3 | tuple[float, float, float]
    volume: float
    velocity: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    stretch: Vec3 | tuple[float, float, float] = Vec3(0.0, 0.0, 0.0)
    stretchiness: float = 0.45
    viscosity: float = 0.22
    surface_tension: float = 0.55
    wetting: float = 0.0
    stickiness: float = 0.0
    bounciness: float = 0.12
    material: Material = Material(color=(70, 190, 150), roughness=0.18, fuzziness=0.08, light_transmission=0.18)

    def __post_init__(self) -> None:
        self.position = as_vec3(self.position)
        self.velocity = as_vec3(self.velocity)
        self.stretch = as_vec3(self.stretch)
        if self.volume <= 0.0:
            raise ValueError("fluid blob volume must be positive")
        self.stretchiness = clamp(float(self.stretchiness), 0.0, 1.0)
        self.viscosity = clamp(float(self.viscosity), 0.0, 1.0)
        self.surface_tension = clamp(float(self.surface_tension), 0.0, 1.0)
        self.wetting = clamp(float(self.wetting), 0.0, 1.0)
        self.stickiness = clamp(float(self.stickiness), 0.0, 1.0)
        self.bounciness = clamp(float(self.bounciness), 0.0, 1.0)

    @classmethod
    def from_radius(cls, position: Vec3 | tuple[float, float, float], radius: float, **kwargs) -> "FluidBlob":
        if radius <= 0.0:
            raise ValueError("fluid blob radius must be positive")
        return cls(position=position, volume=_volume_from_radius(radius), **kwargs)

    @property
    def radius(self) -> float:
        return _radius_from_volume(self.volume)

    def to_primitive(self) -> BlobSurface:
        return BlobSurface(self.position, self.radius, self.material, self.stretch, self.surface_tension, self.wetting, self.stickiness)


@dataclass
class FluidWorld:
    """A bounded blob world with split and heal behavior."""

    bounds_min: Vec3 | tuple[float, float, float] = Vec3(-2.0, 0.0, -2.0)
    bounds_max: Vec3 | tuple[float, float, float] = Vec3(2.0, 2.0, 2.0)
    gravity: Vec3 | tuple[float, float, float] = Vec3(0.0, -9.81, 0.0)
    heal_distance_factor: float = 0.65

    def __post_init__(self) -> None:
        self.bounds_min = as_vec3(self.bounds_min)
        self.bounds_max = as_vec3(self.bounds_max)
        self.gravity = as_vec3(self.gravity)
        self.blobs: list[FluidBlob] = []

    def add_blob(self, blob: FluidBlob) -> FluidBlob:
        self.blobs.append(blob)
        return blob

    def total_volume(self) -> float:
        return sum(blob.volume for blob in self.blobs)

    def step(self, dt: float, substeps: int = 1) -> None:
        if dt < 0.0:
            raise ValueError("fluid dt must be non-negative")
        if substeps <= 0:
            raise ValueError("fluid substeps must be positive")
        step_dt = dt / substeps
        for _ in range(substeps):
            for blob in list(self.blobs):
                self._step_blob(blob, step_dt)
            self._split_overstretched()
            self._heal_close_blobs()

    def _step_blob(self, blob: FluidBlob, dt: float) -> None:
        sticky_drag = blob.stickiness * (1.0 if blob.position.y <= self.bounds_min.y + blob.radius * 1.08 else 0.0)
        blob.velocity = (blob.velocity + self.gravity * dt) * max(0.0, 1.0 - (blob.viscosity + sticky_drag) * dt)
        blob.position = blob.position + blob.velocity * dt
        blob.stretch = (blob.stretch + blob.velocity * dt) * max(0.0, 1.0 - blob.surface_tension * dt)
        radius = blob.radius
        x, vx = _bounded_axis(blob.position.x, blob.velocity.x, self.bounds_min.x + radius, self.bounds_max.x - radius, blob.bounciness)
        y, vy = _bounded_axis(blob.position.y, blob.velocity.y, self.bounds_min.y + radius, self.bounds_max.y - radius, blob.bounciness)
        z, vz = _bounded_axis(blob.position.z, blob.velocity.z, self.bounds_min.z + radius, self.bounds_max.z - radius, blob.bounciness)
        blob.position = Vec3(x, y, z)
        blob.velocity = Vec3(vx, vy, vz)

    def _split_overstretched(self) -> None:
        additions: list[FluidBlob] = []
        for blob in list(self.blobs):
            limit = blob.radius * (0.45 + blob.stretchiness * 2.4)
            stretch_length = blob.stretch.length()
            if stretch_length <= limit or blob.volume <= 0.01:
                continue
            direction = blob.stretch.normalized(Vec3(1.0, 0.0, 0.0))
            split_volume = blob.volume * 0.42
            blob.volume -= split_volume
            blob.stretch = blob.stretch * 0.2
            additions.append(
                FluidBlob(
                    position=blob.position + direction * (blob.radius * 0.75),
                    volume=split_volume,
                    velocity=blob.velocity + direction * 0.45,
                    stretch=-blob.stretch,
                    stretchiness=blob.stretchiness,
                    viscosity=blob.viscosity,
                    surface_tension=blob.surface_tension,
                    wetting=blob.wetting,
                    stickiness=blob.stickiness,
                    bounciness=blob.bounciness,
                    material=blob.material,
                )
            )
        self.blobs.extend(additions)

    def _heal_close_blobs(self) -> None:
        index = 0
        while index < len(self.blobs):
            first = self.blobs[index]
            merged = False
            second_index = index + 1
            while second_index < len(self.blobs):
                second = self.blobs[second_index]
                heal_distance = (first.radius + second.radius) * self.heal_distance_factor
                if first.position.distance_to(second.position) <= heal_distance:
                    total_volume = first.volume + second.volume
                    first.position = (first.position * first.volume + second.position * second.volume) / total_volume
                    first.velocity = (first.velocity * first.volume + second.velocity * second.volume) / total_volume
                    first.stretch = (first.stretch + second.stretch) * (0.2 + (first.surface_tension + second.surface_tension) * 0.15)
                    first.volume = total_volume
                    del self.blobs[second_index]
                    merged = True
                    continue
                second_index += 1
            if not merged:
                index += 1


def _bounded_axis(position: float, velocity: float, low: float, high: float, bounciness: float) -> tuple[float, float]:
    if position < low:
        return low, abs(velocity) * bounciness
    if position > high:
        return high, -abs(velocity) * bounciness
    return position, velocity
