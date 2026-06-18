# Rendering Guide

This guide covers the main rendering concepts in `py3dengine`: scenes, cameras,
materials, lights, render settings, textures, shadows, and generated geometry.

## Render Pipeline Overview

A render combines:

1. A `Scene` containing objects, lights, background color, and bulletins.
2. A `Camera` containing position, target, and field of view.
3. A `RenderSettings` object containing size, quality, shading, and budget
   controls.
4. A `Renderer`, usually the CPU reference renderer through `RenderEngine`.

```python
from py_3d import Camera, RenderEngine, RenderSettings, Scene

scene = Scene()
camera = Camera(position=(0, 0, -4), target=(0, 0, 0))
settings = RenderSettings(width=640, height=360)

buffer = RenderEngine().render(scene, camera, settings)
```

The output is a `PixelBuffer`. It can be saved:

```python
buffer.to_png("render.png")
buffer.to_ppm("render.ppm")
```

or sampled in tests:

```python
center = buffer.get_pixel(buffer.width // 2, buffer.height // 2)
```

## Scene Objects

Objects are added to a scene with `scene.add(...)`:

```python
from py_3d import Box, Material, Plane, Scene, Sphere

scene = Scene()
scene.add(
    Sphere((0, 0.6, 0), 0.6, Material(color=(90, 150, 235))),
    Box((1.0, 0.25, 0), (0.5, 0.5, 0.5), Material(color=(230, 160, 70))),
    Plane((0, -0.05, 0), (0, 1, 0), Material(color=(50, 80, 70)), size=5.0),
)
```

Common primitives:

| Primitive | Use |
| --- | --- |
| `Point3` | A point in 3D space. |
| `Line3` | A 3D line segment. |
| `Triangle` | The lowest-level filled 3D surface. |
| `Box` | Generated cube/rectangular prism geometry. |
| `Sphere` | Generated UV sphere geometry. |
| `Bowl` | Concave bowl mesh with optional thickness. |
| `Plane` | Finite plane or slab. |
| `Capsule` | Player/body-style capsule mesh. |
| `Mesh` | A collection of triangles. |
| `BlobSurface` | Deformable blob/fluid-style surface. |

## Cameras

The camera looks from `position` toward `target`:

```python
from py_3d import Camera

camera = Camera(
    position=(2.0, 1.2, -4.0),
    target=(0.0, 0.4, 0.0),
    fov_degrees=48,
)
```

Field of view is tied to projection. A higher FOV sees more of the world and
creates stronger perspective. A lower FOV feels more zoomed in.

Live demos use smoothed render cameras where useful so movement does not flash
or jitter while physics and input continue to update normally.

## RenderSettings

`RenderSettings` controls image dimensions, background, shading, generated
geometry density, and optional effects:

```python
from py_3d import RenderSettings

settings = RenderSettings(
    width=1280,
    height=720,
    background=(8, 10, 14),
    ambient=0.04,
    gamma=1.15,
    smooth_shading=True,
    sphere_segments=24,
    sphere_rings=12,
    texture_size=384,
    tone_mapping=True,
    max_render_distance=10.0,
)
```

Important settings:

| Setting | Meaning |
| --- | --- |
| `width`, `height` | Output render resolution. |
| `background` | Clear color. |
| `ambient` | Small fill light added before direct lights. |
| `gamma` | Display correction. |
| `smooth_shading` | Interpolate vertex normals for generated primitives. |
| `wireframe` | Draw primitive edges instead of filled triangles. |
| `sphere_segments`, `sphere_rings` | Generated sphere/bowl mesh density. |
| `ray_traced_shadows` | Enable direct-light shadow ray checks. |
| `shadow_samples` | Area-light style samples for softer shadows. |
| `shadow_softness` | Spread for soft shadow samples. |
| `reflection_bounces` | Reflection budget used by showcase paths. |
| `edge_highlight` | Overlay sharp/boundary edges. |
| `edge_highlight_threshold_degrees` | Normal angle threshold for edge outlines. |
| `max_render_distance` | Cull geometry beyond a camera distance budget. |

## Lights

Two light types are exposed publicly:

```python
from py_3d import Lamp, Sun

scene.add_light(Sun(direction=(-0.4, -0.8, -1.0), color=(255, 245, 230), intensity=0.9))
scene.add_light(Lamp(position=(-1.2, 1.5, -2.0), color=(120, 170, 255), intensity=2.4))
```

`Sun` is directional. Its rays are effectively parallel.

`Lamp` is positional. It is useful for local highlights, colored lights, and
small presentation lamps.

`SkyPrefab` can add a sun automatically from the current time of day or manual
sun angle.

## Materials

Materials describe visual response:

```python
from py_3d import Material

material = Material(
    color=(230, 140, 70),
    diffuse=0.9,
    absorption=(0.05, 0.1, 0.16),
    emission=(0, 0, 0),
    roughness=0.42,
    fuzziness=0.04,
    specular=0.22,
    shininess=36.0,
    reflectivity=0.12,
)
```

Material fields:

| Field | Meaning |
| --- | --- |
| `color` | Base RGB color. |
| `texture` | Optional `PixelBuffer` sampled by UV coordinates. |
| `diffuse` | Matte light response strength. |
| `absorption` | Per-channel light absorption. |
| `emission` | Self-lit RGB contribution. |
| `roughness` | Dulls sharp visual response. |
| `fuzziness` | Adds deterministic surface variation. |
| `specular` | Highlight strength. |
| `shininess` | Highlight tightness. |
| `reflectivity` | Reflection-like boost used by showcase materials. |
| `light_transmission` | Direct-light transmission during ray shadow checks. |

## Textures

Load a PNG texture:

```python
from py_3d import Material, PixelBuffer

texture = PixelBuffer.from_png("assets/tv-test.png")
material = Material(color=(255, 255, 255), texture=texture)
```

Generated spheres and bowls already create UVs. For custom triangles, pass UVs
when constructing `Triangle`:

```python
from py_3d import Material, Triangle

material = Material(texture=texture, color=(255, 255, 255))
triangle = Triangle(
    (-1, -1, 0),
    (1, -1, 0),
    (0, 1, 0),
    material,
    (0.0, 1.0),
    (1.0, 1.0),
    (0.5, 0.0),
)
```

Planar projection can assign UVs to a triangle set:

```python
from py_3d import planar_project_triangles

triangles = planar_project_triangles(
    triangles,
    center=(0, 0, 0),
    u_axis=(1, 0, 0),
    v_axis=(0, 1, 0),
    scale=1.0,
)
```

## Shadows

The default lighting path is fast and direct. For still images, enable
ray-traced direct shadows:

```python
settings = RenderSettings(
    width=640,
    height=360,
    ray_traced_shadows=True,
    shadow_samples=4,
    shadow_softness=0.12,
)
```

This checks whether direct light from a lamp or sun is blocked by scene
geometry. It is intentionally experimental and slower than the normal path.

Transparent or transmissive objects can let some light through:

```python
glass = Material(color=(180, 220, 255), light_transmission=0.55)
```

## Edge Highlighting

Edge highlighting is useful for technical renders and low-poly showcase images:

```python
settings = RenderSettings(
    edge_highlight=True,
    edge_highlight_threshold_degrees=35,
)
```

It draws boundary edges and adjacent faces whose normals differ by at least the
threshold.

## Generated Geometry Density

Generated meshes use settings to decide how much detail to create:

```python
settings = RenderSettings(sphere_segments=32, sphere_rings=16)
```

Lower values give a low-poly style and cheaper frames. Higher values give
smoother silhouettes and better texture motion.

The user settings file `USER/settings.json` defines named quality presets:

| Preset | Typical use |
| --- | --- |
| `fast` | Lower render size, lower mesh density, safe live preview. |
| `balanced` | Middle ground for demos. |
| `high` | Current showcase target. |
| `ultra` | High render size and higher shadow/reflection budget. |
| `poly` | Low-poly visual style with modern lighting. |

## Text Bulletins

Scenes can include screen-space or world-projected text:

```python
from py_3d import FloatingTextBulletin, TextBulletin

scene.add_bulletin(
    TextBulletin("SCREEN LABEL", position=(12, 12), background=(4, 8, 13), padding=6),
    FloatingTextBulletin("WORLD LABEL", position=(0, 1.2, 0), background=(8, 5, 3), padding=5),
)
```

This is how the examples show labels without requiring a UI framework.

## Asset Imports

Load OBJ or STL directly:

```python
from py_3d import Material, load_obj, load_stl

mesh = load_obj("model.obj", material=Material(color=(200, 200, 220)))
mesh = load_stl("model.stl", material=Material(color=(200, 200, 220)))
```

Prepared mesh assets use the compact `*.py3dmesh.json` format:

```python
from py_3d import load_mesh_asset

mesh = load_mesh_asset("USER/assets/sea_lion/sea_lion.py3dmesh.json")
```

Create a prepared asset from an OBJ:

```bash
python examples/ingest_asset.py assets/sea-lion-import-test/10041_sealion_v1_L3.obj --name sea_lion --output-dir USER/assets --target-triangles 12000 --source-up z --scale-to-height 1.15 --yaw 90
```

## Reproducing The Showcase Renders

Run:

```bash
python examples/rendering_gallery.py
python examples/texture_demo.py
python examples/textured_sphere_polygons.py
python examples/physics_gallery.py
python examples/render_menu_showcase.py
```

For environment stills, see the command list in the repository README.
