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
| `16_live_fruit_bowl_rgb_bulbs.py` | Fruit bowl live variant with blinking RGB bulbs. |
| `17_render_hud_demo.py` | FPS and third-person HUD still previews. |
| `18_live_wind_pool_water.py` | Wind-driven vector-fluid pool scene. |
| `19_live_procedural_environment.py` | Streamed procedural hills with HLOD scenery rings, virtualized tree clusters, grass, and water. |
| `20_render_feature_previews.py` | Batch still previews. |
| `30_render_environment_videos.py` | Batch environment videos. |
| `40_run_feature_tests.py` | User-facing feature test runner. |

## Common Live Controls

The live capsule walk demo is the control reference. Any demo with player/FPV
movement or free-camera navigation must use `canonical_player_movement_key`
from `py_3d.live_defaults` instead of copying bindings locally.

| Control | Action |
| --- | --- |
| `Esc` | Open or close the menu. |
| Mouse | Look around while the pointer is captured. |
| `W/A/S/D` | Move. |
| `Space` | Jump in first/third-person modes; elevate in global/free-camera mode. |
| `Shift` | Sprint in first/third-person modes; elevate in global/free-camera mode. |
| `Ctrl` | Crouch in first/third-person modes; descend in global/free-camera mode. |
| `C` | Crouch in first/third-person modes. |
| `V` | Cycle `global -> third -> first -> global` when a demo exposes camera modes. |
| `E` | Grab or drop supported physics objects. |
| Mouse wheel | Adjust held-object distance where grabbing is supported. |
| `P` | Save a snapshot where supported. |

## Live Menu Structure

There is one live settings menu contract. Use `LiveMenu` for the menu shell and
`update_canonical_live_menu` for the option inventory. Demos enable only the
actions they support; all other settings remain visible but disabled/grayed out.

```python
from py_3d import update_canonical_live_menu
from py_3d.live import LiveMenu

menu = LiveMenu("Demo Settings")
update_canonical_live_menu(
    menu,
    details={
        "quality_next": "balanced",
        "sky_cycle": "on",
        "pause": "running",
    },
    enabled_actions={"quality_next", "sky_cycle", "pause", "done", "quit"},
)
```

The canonical inventory covers every setting used by current live demos:
Graphics quality, polygon density, reflections, smooth shading, texture size,
gamma, tone mapping, and wire/fill; sky cycle/time/sun/clouds/stars; world view
distance, tree LOD, rebuild, and seed; physics wind, swirl, vessel, pause, and
reset; camera mode and look smoothing; snapshot; and Done/Apply/Exit/Quit
footer actions.

Options with a `group` value are shown under category tabs. Disabled options are
drawn muted and cannot be activated by mouse click or Enter/Space. Footer
actions such as Done, Apply, Exit Menu, and Quit Demo are drawn in a button bay.

The menu supports:

- Mouse hover and click.
- Mouse wheel scrolling for long option groups.
- Left/right tab navigation.
- Up/down option navigation.
- Enter/Space selection.
- Apply/Done/Exit patterns for settings that should not apply immediately.
- Disabled options for settings not relevant to the current demo.

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
