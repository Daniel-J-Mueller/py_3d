"""Optional GPU renderer scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec

from .buffer import PixelBuffer
from .camera import Camera
from .color import Color
from .render import CPURenderer, RenderSettings, Renderer, _triangles_for
from .scene import Scene


def detect_gpu_backends() -> tuple[str, ...]:
    """Return optional Python GPU packages visible in this environment."""

    candidates = (
        ("moderngl", "OpenGL"),
        ("wgpu", "WebGPU"),
        ("OpenGL", "PyOpenGL"),
    )
    return tuple(label for module_name, label in candidates if find_spec(module_name) is not None)


@dataclass(frozen=True)
class GPUSceneBatch:
    """Flattened triangle data ready for a future GPU upload path."""

    positions: tuple[tuple[float, float, float], ...]
    colors: tuple[Color, ...]
    indices: tuple[tuple[int, int, int], ...]


def build_gpu_scene_batch(scene: Scene, settings: RenderSettings | None = None) -> GPUSceneBatch:
    """Flatten renderable scene geometry into triangle buffers."""

    active_settings = settings or RenderSettings()
    positions: list[tuple[float, float, float]] = []
    colors: list[Color] = []
    indices: list[tuple[int, int, int]] = []
    for obj in scene.objects:
        for triangle in _triangles_for(obj, active_settings):
            index = len(positions)
            positions.extend((triangle.a.as_tuple(), triangle.b.as_tuple(), triangle.c.as_tuple()))
            colors.extend((triangle.material.color, triangle.material.color, triangle.material.color))
            indices.append((index, index + 1, index + 2))
    return GPUSceneBatch(tuple(positions), tuple(colors), tuple(indices))


@dataclass
class GPURenderer:
    """Renderer-compatible GPU entry point.

    This is deliberately a scaffold. It keeps the GPU path behind the same
    ``Renderer`` protocol as the CPU implementation while backend-specific
    rasterization is still being built.
    """

    allow_cpu_fallback: bool = True
    fallback_renderer: Renderer | None = None
    name: str = "GPU Renderer Scaffold"
    backend: str = field(init=False, default="gpu")

    def __post_init__(self) -> None:
        if self.fallback_renderer is None:
            self.fallback_renderer = CPURenderer()
        self.detected_backends = detect_gpu_backends()

    def prepare(self, scene: Scene, settings: RenderSettings | None = None) -> GPUSceneBatch:
        return build_gpu_scene_batch(scene, settings)

    @property
    def is_accelerated(self) -> bool:
        return False

    def render(
        self,
        scene: Scene,
        camera: Camera,
        settings: RenderSettings,
        target: PixelBuffer | None = None,
    ) -> PixelBuffer:
        if self.allow_cpu_fallback:
            return self.fallback_renderer.render(scene, camera, settings, target)
        backends = ", ".join(self.detected_backends) if self.detected_backends else "none"
        raise RuntimeError(
            "GPU renderer scaffold is installed, but GPU rasterization is not implemented yet. "
            f"Detected optional GPU packages: {backends}."
        )
