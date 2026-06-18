# Live Demos And Menus

The `USER/demos/` folder contains the user-facing demo launcher and direct demo
entry points. These scripts present the renderer as an interactive feature
showcase rather than a library test suite.

## Main Launcher

Open the launcher:

```bash
python USER/demos/00_list_experiences.py
```

List experiences:

```bash
python USER/demos/00_list_experiences.py --list
```

Run an experience by number:

```bash
python USER/demos/00_list_experiences.py --run 1 --quality high
```

Preview a launch command without running it:

```bash
python USER/demos/00_list_experiences.py --run 1 --dry-run --quality safe
```

Render a still preview:

```bash
python USER/demos/00_list_experiences.py --render-preview 3
```

The launcher keeps settings staged until Apply or Done is used. If a launched
process exits with an error, the launcher reverts to the last safe settings.

## Demo Entries

| Script | Experience |
| --- | --- |
| `10_live_fruit_bowl_gpu.py` | Live fruit bowl rendering and physics showcase. |
| `11_live_capsule_walk.py` | FPS, over-shoulder, and free-camera capsule controller. |
| `12_live_fruit_bowl_mirror_prelight.py` | Higher-spec mirror/prelight fruit bowl. |
| `13_live_fruit_bowl_poly_lamp.py` | Low-poly wood bowl, hanging lamp, and baked sign. |
| `14_render_sea_lion_asset.py` | Prepared mesh asset render. |
| `15_render_fan_cloth_water.py` | Fan, cloth, and vector-fluid water scene. |
| `20_render_feature_previews.py` | Batch still previews. |
| `30_render_environment_videos.py` | Batch environment videos. |
| `40_run_feature_tests.py` | User-facing feature test runner. |

## Common Live Controls

Controls vary by demo, but the common live controls are:

| Control | Action |
| --- | --- |
| `Esc` | Open or close the menu. |
| Mouse | Look around while the pointer is captured. |
| `W/A/S/D` | Move. |
| `E` | Grab or drop supported physics objects. |
| Mouse wheel | Adjust held-object distance. |
| `Space` | Jump in player modes or pause in some demo modes. |
| `Shift` | Sprint in FPS/player modes; elevate in free-camera mode. |
| `Ctrl` | Descend in free-camera mode; crouch where supported. |
| `V` | Cycle camera modes where supported. |
| `P` | Save a snapshot where supported. |

## Live Menu Structure

Live menus are built from `LiveMenu` and `LiveMenuOption`:

```python
from py_3d.live import LiveMenu, LiveMenuOption

menu = LiveMenu(
    "Demo Settings",
    (
        LiveMenuOption("quality", "Quality: High", "Render size and mesh density", "Graphics"),
        LiveMenuOption("sky_cycle", "Day/Night Cycle", "Animate sky time", "Sky"),
        LiveMenuOption("pause", "Pause Physics", "Freeze simulation", "Physics"),
        LiveMenuOption("done", "Done"),
        LiveMenuOption("apply", "Apply"),
        LiveMenuOption("cancel", "Exit Menu"),
    ),
)
```

Options with a `group` value are shown under category tabs. Footer actions such
as Done, Apply, Exit Menu, and Quit Demo are drawn in a button bay.

The menu supports:

- Mouse hover and click.
- Mouse wheel scrolling for long option groups.
- Left/right tab navigation.
- Up/down option navigation.
- Enter/Space selection.
- Apply/Done/Exit patterns for settings that should not apply immediately.

## Graphics Settings

Demos that expose quality settings use presets from `USER/settings.json`.

Presets include:

| Preset | Behavior |
| --- | --- |
| `fast` | Safe live preview, lower render size, lower mesh density. |
| `balanced` | Moderate render size and quality. |
| `high` | Main showcase setting. |
| `ultra` | Larger render target and higher shadow/reflection budget. |
| `poly` | Lower mesh density for a deliberate low-poly style. |

Quality changes can affect:

- Render width and height.
- Window width and height.
- Generated sphere and bowl polygons.
- Smooth shading.
- Texture resolution.
- Gamma and tone mapping.
- Light wrap and bounce fill.
- Reflection bounce budget.
- Shadow samples and softness.
- Maximum render distance.

## Fruit Bowl Demo

Run:

```bash
python examples/fruit_bowl_live.py --renderer py_gpu --quality high --light-mode hanging-lamp
```

The fruit bowl demo shows:

- Textured fruit materials.
- Wood and mirror bowl materials.
- Kinematic bowl motion with dynamic fruit collision.
- Grab/drop interaction with `E`.
- Mouse wheel held-object distance.
- Baked in-world sign text.
- Sky controls.
- Quality, polygon, reflection, texture, and tone-map controls.

Still renders:

```bash
python examples/fruit_bowl_demo.py --smooth-shading --output USER/environments/fruit_bowl/renderings/fruit_bowl_smooth.png
python examples/fruit_bowl_demo.py --ray-traced-shadows --reflection-bounces 2 --output USER/environments/fruit_bowl/renderings/fruit_bowl_ray_traced.png
```

## Capsule Walk Demo

Run:

```bash
python examples/capsule_walk_demo.py --renderer py_gpu --camera-mode third
```

Camera modes:

| Mode | Behavior |
| --- | --- |
| `first` | First-person view with player body hidden from the camera. |
| `third` | Over-shoulder camera aimed through the crosshair. |
| `global` | Free/noclip camera for inspection. |

Movement notes:

- Shift sprints in first- and third-person modes.
- Shift elevates the camera in global/free mode.
- Crouch lowers the player camera in FPS/player modes.

## Fan Cloth Water Demo

Run:

```bash
python examples/fan_cloth_water_demo.py --renderer py_gpu
```

The scene demonstrates:

- A spring-mesh cloth driven by fan wind.
- Pinned cloth nodes and gravity.
- A vector-particle water surface in a bowl.
- A spinning blade that stirs water gently.
- Menu controls for wind, swirl, physics pause, reset, sky, and reflections.

Render a still:

```bash
python examples/fan_cloth_water_demo.py --quality fast --width 640 --height 360 --output USER/environments/fan_cloth_water/renderings/fan_cloth_water.png
```

## Rendering Menu Screenshots

The docs menu images are generated with:

```bash
python examples/render_menu_showcase.py
```

Outputs:

- `renderings-tests/demo_launcher_menu.png`
- `renderings-tests/demo_launcher_settings.png`
- `renderings-tests/live_settings_menu.png`

