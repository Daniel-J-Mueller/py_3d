# py_3d

`py_3d` is an early-stage Python package for fast, basic 3D pixel drawing,
rendering, and simulation primitives.

The goal is to become a small, reliable alternative to the simplest parts of
Pygame, with a 3D-first model: draw pixels and primitives, keep depth ordering
simple and predictable, light objects with basic material behavior, and provide
enough collision support to build toy physics, demos, and simulations without
pulling in a full game engine.

This repository is currently at the project-definition stage. The API examples
below describe the intended direction and should guide implementation.

## Project Goals

- Provide a clean Python API for 3D pixel drawing on Windows and Linux.
- Keep the primitive layer simple: points, lines, triangles, boxes, spheres,
  meshes, voxel-like blocks, and blitted pixel buffers.
- Include a basic depth-aware renderer that can draw into an image buffer or
  window surface.
- Support simple light sources:
  - `Lamp`: positional light with distance falloff.
  - `Sun`: directional light with effectively parallel rays.
- Let lights carry RGB color channels and intensity.
- Let objects and materials expose absorption, reflection, and diffuse response
  in a basic, inspectable way.
- Provide a small collision system suitable for examples such as a ball rolling
  down a hill, hitting a wall, or stacking simple objects.
- Stay extensible enough for future simulation work, including fluids.
- Prefer deterministic, explicit behavior over clever hidden state.

## Non-Goals

`py_3d` should not try to become Blender, Unity, Unreal, Panda3D, or a complete
physics engine. It should be a primitive, teachable, hackable foundation.

The package should also avoid copying Pygame's design wholesale. Pygame is a
useful reference point for simple drawing ergonomics, but this project should
favor a smaller, clearer API that treats depth, light, and simulation as
first-class concepts.

## Intended API Shape

The exact module names may change as the package is built, but the public API
should stay close to this level of simplicity:

```python
import py_3d as p3d

screen = p3d.Window(width=960, height=540, title="py_3d demo")
scene = p3d.Scene()

red_ball = p3d.Sphere(
    center=(0, 2, 0),
    radius=0.5,
    material=p3d.Material(color=(220, 40, 35), absorption=(0.2, 0.1, 0.1)),
)

hill = p3d.Box(
    center=(0, 0, 0),
    size=(6, 0.25, 3),
    rotation=(0, 0, -18),
    material=p3d.Material(color=(80, 150, 90), absorption=(0.25, 0.35, 0.25)),
)

wall = p3d.Box(
    center=(2.5, 0.7, 0),
    size=(0.25, 1.4, 3),
    material=p3d.Material(color=(180, 180, 190)),
)

scene.add(red_ball, hill, wall)
scene.add_light(p3d.Sun(direction=(-1, -2, -1), color=(255, 245, 230), intensity=0.9))
scene.add_light(p3d.Lamp(position=(1, 3, 2), color=(120, 170, 255), intensity=0.4))

physics = p3d.World(gravity=(0, -9.81, 0))
physics.add_dynamic(red_ball, mass=1.0)
physics.add_static(hill)
physics.add_static(wall)

while screen.open:
    dt = screen.tick(60)
    physics.step(dt)
    screen.draw(scene)
```

## Core Concepts

### Surfaces and Buffers

The lowest layer should be a pixel buffer with predictable memory layout. A
window is only one possible output target. Rendering to an off-screen buffer
should be supported from the beginning so tests, image export, and headless
simulation are easy.

### Primitives

Primitives are the basic things the package can draw or simulate. They should be
small data objects where possible. Drawing behavior belongs in renderers, and
physics behavior belongs in the collision or world modules.

Useful initial primitives:

- `Point3`
- `Line3`
- `Triangle`
- `Box`
- `Sphere`
- `Plane`
- `Mesh`
- `VoxelGrid`

### Materials

Materials define how objects respond to light. Keep this deliberately simple at
first:

- `color`: base RGB color.
- `absorption`: per-channel light absorption.
- `diffuse`: matte response strength.
- `emission`: optional RGB emission for self-lit objects.

### Lights

Lights should be data-first and explicit:

- `Lamp(position, color, intensity, radius=None)` for local light emission.
- `Sun(direction, color, intensity)` for directional light.

Lighting should be basic but composable. It is better to have an understandable
Lambert-style model than a large physically based system that is hard to extend.

### Rendering

The first renderer should prioritize correctness and clarity:

- Camera projection from 3D world coordinates to 2D pixels.
- Z-buffer or equivalent depth handling.
- Back-face culling where useful.
- Basic shaded triangles and primitive rasterization.
- Optional wireframe and debug-depth modes.

Performance matters, but early optimization should not make the architecture
opaque. When speed work is needed, prefer isolated accelerated paths behind a
stable Python API.

### Collision and Motion

The collision system should start with simple shapes and simple guarantees:

- Broad phase: cheap bounding checks.
- Narrow phase: sphere, plane, box, and triangle interactions.
- Body types: static, dynamic, and kinematic.
- Forces: gravity, impulses, friction, and restitution.
- Deterministic fixed-step simulation support.

The system should be good enough for educational demos and basic simulation
prototypes before it tries to handle advanced rigid-body behavior.

## Proposed Package Layout

```text
py_3d/
  __init__.py
  buffer.py        # Pixel buffers, color packing, image export helpers
  camera.py        # Camera and projection math
  collision.py     # Shape intersection and contact generation
  draw.py          # Immediate-mode primitive drawing helpers
  lights.py        # Lamp, Sun, and lighting utilities
  materials.py     # Material definitions
  math3d.py        # Vectors, matrices, transforms, numeric helpers
  physics.py       # Bodies, world stepping, constraints over time
  primitives.py    # Drawable and collidable primitive data types
  render.py        # Renderers, depth buffers, rasterization
  window.py        # Optional interactive window backend
tests/
examples/
```

This layout is a starting point, not a requirement. Keep modules small and
split them when a file starts mixing unrelated responsibilities.

## Performance Direction

The package should be fast enough for real-time demos while staying readable.
Recommended path:

1. Build a clear pure-Python reference implementation.
2. Add focused benchmarks for rasterization, depth buffering, collision checks,
   and world stepping.
3. Use standard-library tools first.
4. Add optional acceleration only behind stable interfaces.
5. Keep slow-but-clear reference paths available for tests and debugging.

Potential acceleration options can include NumPy, Numba, Cython, Rust, or C
extensions later, but no acceleration dependency should become mandatory without
a strong reason.

## Development

This project targets modern Python on Windows and Linux.

Suggested setup once packaging files exist:

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest
```

Until the package structure is created, use this README and `AGENT.md` as the
source of truth for project direction.

## Testing Expectations

Tests should cover behavior that is easy to accidentally break:

- Projection math and coordinate transforms.
- Depth comparisons and clipping.
- Primitive rasterization edge cases.
- Light color and absorption calculations.
- Collision contact generation.
- Fixed-step simulation determinism.
- Headless rendering to a pixel buffer.

Visual examples are valuable, but they should not replace numeric tests.

## Roadmap

- Define package metadata and initial module layout.
- Implement vector, color, transform, and pixel-buffer primitives.
- Implement immediate-mode 2D and 3D drawing into an off-screen buffer.
- Add camera projection and a basic depth buffer.
- Add `Lamp`, `Sun`, and material absorption.
- Implement basic primitives: line, triangle, box, sphere, plane.
- Add a minimal window backend for Windows and Linux.
- Add collision detection for spheres, planes, and boxes.
- Add a fixed-step physics world with gravity and impulses.
- Build examples:
  - Rotating lit cube.
  - Ball rolling down a hill into a wall.
  - Multiple colored lights.
  - Headless render-to-image test.
- Explore voxel and fluid-friendly data structures.

## Design Principle

Keep the engine primitive, explicit, and composable. A user should be able to
understand how a pixel got its color, why an object collided, and where to
extend the system without reading thousands of lines of framework code.
