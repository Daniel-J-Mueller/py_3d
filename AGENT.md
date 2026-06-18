# Agent Guide

This repository is building a primitive 3D pixel drawing and simulation package
for Python. Future agents should protect clarity, small APIs, and deterministic
behavior while the engine grows.

## Product Direction

- Treat `py_3d` as a basic package, not a full game engine.
- Favor Pygame-like ergonomics for simple drawing, but keep this project
  3D-first with explicit depth, camera, lighting, and simulation concepts.
- Support Windows and Linux.
- Keep headless rendering as a first-class use case. A window should not be
  required to test drawing or simulation.
- Build toward fluid dynamics eventually, but do not distort the early engine
  around fluids before core rendering, materials, and collisions are stable.

## Engineering Rules

- Keep modules focused. Split files when rendering, collision, physics, window
  management, and math responsibilities start blending together.
- Prefer small data classes and explicit functions over global mutable engine
  state.
- Public APIs should be easy to inspect in a REPL.
- Preserve deterministic behavior for math, rendering tests, and fixed-step
  physics.
- Keep reference implementations understandable before adding acceleration.
- If adding an optimized path, keep it behind the same public interface as the
  clear path.
- Keep CPU and GPU compatibility in mind early. The pure-Python CPU renderer is
  the behavioral reference; GPU renderers should match its public interface and
  be tested against equivalent scenes.
- Avoid mandatory heavy dependencies unless the project has a clear need and
  tests that justify them.
- Do not add a framework-scale abstraction just because one might be useful
  later.

## Rendering Guidance

- Render into explicit buffers. Window display should consume a buffer rather
  than own all drawing behavior.
- Preserve offline rendering. A render pass should be usable for tests, image
  export, batch rendering, and simulation snapshots without a live event loop.
- Keep depth handling basic and predictable, such as a z-buffer.
- Make camera projection, clipping, and rasterization testable without a GUI.
- Prefer simple Lambert-style lighting before advanced shading.
- Lights should have color channels and clear semantics:
  - `Lamp`: positional emission with optional falloff.
  - `Sun`: directional emission.
- Materials should expose absorption and base color directly.
- Future GPU backends should be optional and should not force a different scene,
  material, camera, or light model.
- Keep acceleration behind the `Renderer` protocol. The pure-Python CPU renderer
  is the correctness reference; optional NumPy, Numba, Cython, OpenGL, Vulkan,
  WebGPU, or platform compute backends should match its behavior.
- Profile before optimizing. Prefer improvements that keep the reference path
  understandable, such as cached projection constants, cached static geometry,
  prepared triangle data, and tight buffer writes.
- Keep text bulletins on the roadmap for 3D views. They should work as overlays
  for labels, debug status, and scene callouts without requiring a GUI window.
- For real-time and wiremesh/wireframe views, plan clicked-in mouse and keyboard
  navigation. Camera controls should be explicit and should not leak into
  headless rendering.
- Keep live viewing dimensions separate from output render dimensions. A window
  can be 960x540 while the render target is 320x180, 1280x720, or any other
  explicit size.

## Collision and Physics Guidance

- Start with simple, robust primitives: sphere, plane, axis-aligned box, oriented
  box, triangle, and ray.
- Keep collision detection separate from time integration.
- Use fixed-step world updates for deterministic examples.
- Keep units explicit in names and docs when possible.
- Add examples only when the underlying behavior is test-covered.

## API Cleanliness

- Keep object construction boring and explicit.
- Avoid hidden registration, singleton engines, or import-time side effects.
- Do not require users to subclass core engine objects for ordinary use.
- Keep error messages specific and actionable.
- Prefer tuples or small vector types consistently; do not mix coordinate
  conventions casually.
- Document coordinate handedness before implementing camera and projection APIs.

## Testing Guidance

- Add tests with new behavior, especially for math, depth, collision, and
  lighting.
- Prefer numeric assertions over screenshot-only tests.
- Use off-screen buffers for render tests.
- Keep random examples seeded.
- Add benchmarks separately from tests when performance work begins.

## Documentation Guidance

- Update `README.md` when public direction or API shape changes.
- Keep examples runnable once the package exists.
- Mark aspirational APIs clearly until implemented.
- Avoid claiming support for features that do not exist yet.

## Change Discipline

- Keep changes narrow and explainable.
- Respect existing user changes in the worktree.
- Avoid broad rewrites during feature work unless the current structure blocks
  the requested feature.
- Prefer incremental foundations: math, buffers, renderers, lights, materials,
  primitives, collisions, then physics examples.
