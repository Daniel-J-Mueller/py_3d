"""Small bounded blob fluid primitives."""

from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import dataclass
from math import floor, pi

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
        float blade_phase = sin(u_time * 6.28318530718 * 2.4 + distance * 8.0);
        float surface = clamp((position.y - (u_force_center.y - 0.1)) / 0.22, 0.0, 1.0);
        float wind_wave = sin(u_time * 6.28318530718 * 1.7 + position.z * 8.0);
        vec3 wind_shear = vec3(1.0, 0.04 * wind_wave, 0.16 * sin(u_time * 3.1 + position.x * 4.5)) * (0.72 * u_wind_strength * surface);
        vec3 swirl = tangent * (1.05 / (1.0 + distance * 3.0));
        vec3 lift = vec3(0.0, 0.16 * blade_phase * surface, 0.0);
        return wind_shear + (swirl + lift) * u_blade_strength;
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
    if (index >= uint(u_count)) {
        return;
    }

    Particle particle = input_particles[index];
    vec3 position = particle.position_mass.xyz;
    float mass = max(0.000001, particle.position_mass.w);
    vec3 velocity = particle.velocity_pad.xyz;
    vec3 force = u_gravity * mass + external_force(position, velocity);
    float attract_range = u_rest_distance * u_attract_range_factor;

    for (int other_index = 0; other_index < u_count; ++other_index) {
        if (other_index == int(index)) {
            continue;
        }
        Particle other = input_particles[other_index];
        vec3 delta = other.position_mass.xyz - position;
        float distance = length(delta);
        if (distance <= 0.000001 || distance > attract_range) {
            continue;
        }
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
        external_force: Callable[[VectorFluidParticle, float], Vec3] | None = None,
    ) -> None:
        if dt < 0.0:
            raise ValueError("fluid dt must be non-negative")
        if substeps <= 0:
            raise ValueError("fluid substeps must be positive")
        step_dt = dt / substeps
        for _ in range(substeps):
            self._step_once(step_dt, external_force)

    def _step_once(self, dt: float, external_force: Callable[[VectorFluidParticle, float], Vec3] | None) -> None:
        forces = [self.gravity * particle.mass for particle in self.particles]
        attract_range = self.rest_distance * self.attract_range_factor
        for first_index, second_index in self._neighbor_pairs(attract_range):
            first = self.particles[first_index]
            second = self.particles[second_index]
            delta = second.position - first.position
            distance = delta.length()
            if distance <= 1e-6 or distance > attract_range:
                continue
            direction = delta / distance
            if distance < self.rest_distance:
                amount = (self.rest_distance - distance) / self.rest_distance
                force = direction * (-self.repel_strength * amount)
                self._dampen_close_pair(first, second, amount, dt)
            else:
                amount = 1.0 - (distance - self.rest_distance) / max(1e-6, attract_range - self.rest_distance)
                force = direction * (self.self_attraction * amount)
            forces[first_index] = forces[first_index] + force
            forces[second_index] = forces[second_index] - force

        for index, particle in enumerate(self.particles):
            if external_force is not None:
                forces[index] = forces[index] + external_force(particle, dt)
            particle.velocity = (particle.velocity + forces[index] * (dt / particle.mass)) * max(0.0, 1.0 - self.viscosity * dt)
            particle.position = particle.position + particle.velocity * dt
            self._resolve_bounds(particle)

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
        self._buffers = []
        self._read_index = 0
        self._initialized = False
        self.force_mode = 0
        self.force_center = Vec3(0.0, 0.0, 0.0)
        self.cylinder_radius = 0.0
        self.time = 0.0
        self.wind_strength = 1.0
        self.blade_strength = 1.0

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
        external_force: Callable[[VectorFluidParticle, float], Vec3] | None = None,
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
        self._sync_particles_from_gpu()

    def close(self) -> None:
        for buffer in self._buffers:
            try:
                buffer.release()
            except Exception:
                pass
        self._buffers = []
        self._initialized = False

    def _ensure_gpu_state(self) -> None:
        if self._initialized:
            return
        if not hasattr(self._ctx, "compute_shader") or getattr(self._ctx, "version_code", 0) < 430:
            raise RuntimeError("GPU fluid requires an OpenGL 4.3 compute-capable ModernGL context")
        self._compute_program = self._ctx.compute_shader(GPU_FLUID_COMPUTE_SHADER)
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
        assert self._compute_program is not None
        read_buffer = self._buffers[self._read_index]
        write_index = 1 - self._read_index
        write_buffer = self._buffers[write_index]
        read_buffer.bind_to_storage_buffer(0)
        write_buffer.bind_to_storage_buffer(1)
        self._set_uniform("u_count", len(self.particles))
        self._set_uniform("u_dt", float(dt))
        self._set_uniform("u_bounds_min", self.bounds_min.as_tuple())
        self._set_uniform("u_bounds_max", self.bounds_max.as_tuple())
        self._set_uniform("u_gravity", self.gravity.as_tuple())
        self._set_uniform("u_rest_distance", self.rest_distance)
        self._set_uniform("u_repel_strength", self.repel_strength)
        self._set_uniform("u_self_attraction", self.self_attraction)
        self._set_uniform("u_viscosity", self.viscosity)
        self._set_uniform("u_boundary_bounce", self.boundary_bounce)
        self._set_uniform("u_close_damping", self.close_damping)
        self._set_uniform("u_attract_range_factor", self.attract_range_factor)
        self._set_uniform("u_force_mode", int(self.force_mode))
        self._set_uniform("u_force_center", self.force_center.as_tuple())
        self._set_uniform("u_cylinder_radius", float(self.cylinder_radius))
        self._set_uniform("u_time", float(self.time))
        self._set_uniform("u_wind_strength", float(self.wind_strength))
        self._set_uniform("u_blade_strength", float(self.blade_strength))
        groups = (len(self.particles) + 127) // 128
        self._compute_program.run(group_x=groups)
        self._read_index = write_index

    def _sync_particles_from_gpu(self) -> None:
        finish = getattr(self._ctx, "finish", None)
        if callable(finish):
            finish()
        raw = self._buffers[self._read_index].read()
        values = array("f")
        values.frombytes(raw)
        stride = 8
        for index, particle in enumerate(self.particles):
            offset = index * stride
            particle.position = Vec3(values[offset], values[offset + 1], values[offset + 2])
            particle.mass = max(1e-6, values[offset + 3])
            particle.velocity = Vec3(values[offset + 4], values[offset + 5], values[offset + 6])

    def _set_uniform(self, name: str, value) -> None:
        try:
            uniform = self._compute_program[name]
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
