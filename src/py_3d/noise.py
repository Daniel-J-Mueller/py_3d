"""Deterministic procedural noise for surface variation."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from .math3d import Vec3, as_vec3, clamp


@dataclass(frozen=True)
class ValueNoise3D:
    """Small deterministic 3D value noise sampler."""

    seed: int = 0

    def sample(self, point: Vec3 | tuple[float, float, float]) -> float:
        value = as_vec3(point)
        x0 = floor(value.x)
        y0 = floor(value.y)
        z0 = floor(value.z)
        xf = value.x - x0
        yf = value.y - y0
        zf = value.z - z0
        u = _smoothstep(xf)
        v = _smoothstep(yf)
        w = _smoothstep(zf)

        c000 = self._corner(x0, y0, z0)
        c100 = self._corner(x0 + 1, y0, z0)
        c010 = self._corner(x0, y0 + 1, z0)
        c110 = self._corner(x0 + 1, y0 + 1, z0)
        c001 = self._corner(x0, y0, z0 + 1)
        c101 = self._corner(x0 + 1, y0, z0 + 1)
        c011 = self._corner(x0, y0 + 1, z0 + 1)
        c111 = self._corner(x0 + 1, y0 + 1, z0 + 1)

        x00 = _lerp(c000, c100, u)
        x10 = _lerp(c010, c110, u)
        x01 = _lerp(c001, c101, u)
        x11 = _lerp(c011, c111, u)
        y0_mix = _lerp(x00, x10, v)
        y1_mix = _lerp(x01, x11, v)
        return _lerp(y0_mix, y1_mix, w)

    def _corner(self, x: int, y: int, z: int) -> float:
        value = x * 374761393 + y * 668265263 + z * 2147483647 + self.seed * 1274126177
        value = (value ^ (value >> 13)) * 1274126177
        value = value ^ (value >> 16)
        return (value & 0xFFFFFFFF) / 0xFFFFFFFF


@dataclass(frozen=True)
class FractalNoise3D:
    """Layered value noise with octave, lacunarity, and gain controls."""

    seed: int = 0
    octaves: int = 3
    lacunarity: float = 2.0
    gain: float = 0.5

    def __post_init__(self) -> None:
        if self.octaves <= 0:
            raise ValueError("noise octaves must be positive")
        if self.lacunarity <= 0.0:
            raise ValueError("noise lacunarity must be positive")
        if self.gain <= 0.0:
            raise ValueError("noise gain must be positive")

    def sample(self, point: Vec3 | tuple[float, float, float]) -> float:
        value = as_vec3(point)
        amplitude = 1.0
        frequency = 1.0
        total = 0.0
        weight = 0.0
        for octave in range(self.octaves):
            noise = ValueNoise3D(self.seed + octave * 1013)
            total += noise.sample(value * frequency) * amplitude
            weight += amplitude
            frequency *= self.lacunarity
            amplitude *= self.gain
        return total / weight


@dataclass(frozen=True)
class SurfacePerturbation:
    """Displace generated vertices along their local surface normal."""

    magnitude: float
    scale: float = 1.0
    seed: int = 0
    octaves: int = 3
    lacunarity: float = 2.0
    gain: float = 0.5
    bias: float = 0.0

    def __post_init__(self) -> None:
        if self.magnitude < 0.0:
            raise ValueError("surface perturbation magnitude must be non-negative")
        if self.scale <= 0.0:
            raise ValueError("surface perturbation scale must be positive")
        object.__setattr__(self, "bias", clamp(float(self.bias), -1.0, 1.0))

    def displacement(self, point: Vec3 | tuple[float, float, float]) -> float:
        sample = FractalNoise3D(self.seed, self.octaves, self.lacunarity, self.gain).sample(as_vec3(point) * self.scale)
        centered = sample * 2.0 - 1.0 + self.bias
        return self.magnitude * clamp(centered, -1.0, 1.0)


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _lerp(a: float, b: float, amount: float) -> float:
    return a + (b - a) * amount
