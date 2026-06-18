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
        ("py_gpu", "py_gpu bridge"),
        ("moderngl", "OpenGL"),
        ("wgpu", "WebGPU"),
        ("OpenGL", "PyOpenGL"),
        ("cupy", "CuPy"),
        ("numba", "Numba"),
        ("torch", "PyTorch"),
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

    This keeps the GPU path behind the same ``Renderer`` protocol as the CPU
    implementation. When the optional ``py_gpu`` package is importable it uses
    that accelerated batch renderer; otherwise it falls back explicitly.
    """

    allow_cpu_fallback: bool = True
    fallback_renderer: Renderer | None = None
    reference_compatible: bool = False
    prefer_backend: str = "auto"
    name: str = "GPU Renderer Scaffold"
    backend: str = field(init=False, default="gpu")

    def __post_init__(self) -> None:
        if self.fallback_renderer is None:
            self.fallback_renderer = CPURenderer()
        self.detected_backends = detect_gpu_backends()
        self.backend_renderer = self._make_backend_renderer()
        if self.backend_renderer is not None:
            self.name = "py_gpu accelerated renderer"

    def prepare(self, scene: Scene, settings: RenderSettings | None = None) -> GPUSceneBatch:
        return build_gpu_scene_batch(scene, settings)

    @property
    def is_accelerated(self) -> bool:
        return self.backend_renderer is not None

    def render(
        self,
        scene: Scene,
        camera: Camera,
        settings: RenderSettings,
        target: PixelBuffer | None = None,
    ) -> PixelBuffer:
        if self.backend_renderer is not None:
            return self.backend_renderer.render(scene, camera, settings, target)
        if self.allow_cpu_fallback:
            return self.fallback_renderer.render(scene, camera, settings, target)
        backends = ", ".join(self.detected_backends) if self.detected_backends else "none"
        raise RuntimeError(
            "GPU renderer is installed, but no accelerated rasterizer is available. "
            f"Detected optional GPU packages: {backends}."
        )

    def _make_backend_renderer(self):
        try:
            from py_gpu.adapters.py3d import Py3DRasterRenderer

            try:
                return Py3DRasterRenderer(
                    reference_compatible=self.reference_compatible,
                    prefer_backend=self.prefer_backend,
                    fast_materials=not self.reference_compatible,
                )
            except TypeError:
                return Py3DRasterRenderer(reference_compatible=self.reference_compatible, prefer_backend=self.prefer_backend)
        except Exception:
            return None
