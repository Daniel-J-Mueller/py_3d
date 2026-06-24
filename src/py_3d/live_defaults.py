"""Shared live-demo controls and settings menu defaults.

The capsule walk demo is the reference for player controls. Live demos that
offer player or FPV movement should call these helpers instead of restating
bindings locally.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .live import LiveMenu, LiveMenuOption


CANONICAL_CAMERA_MODES = ("first", "third", "global")


def next_camera_mode(camera_mode: str) -> str:
    """Cycle camera modes in the same order as the FPV capsule world."""

    return {"global": "third", "third": "first", "first": "global"}.get(camera_mode, "first")


def canonical_player_movement_key(renderer, key: int, *, camera_mode: str = "first") -> str | None:
    """Map renderer key events to the canonical FPV movement vocabulary."""

    if renderer.key_matches(key, "w"):
        return "w"
    if renderer.key_matches(key, "a"):
        return "a"
    if renderer.key_matches(key, "s"):
        return "s"
    if renderer.key_matches(key, "d"):
        return "d"
    if renderer.key_matches(key, "space"):
        return "space"
    if renderer.key_matches(key, "lshift", "rshift"):
        return "shift" if camera_mode == "global" else "sprint"
    if renderer.key_matches(key, "lctrl", "rctrl"):
        return "ctrl" if camera_mode == "global" else "crouch"
    if renderer.key_matches(key, "c"):
        return "crouch"
    return None


@dataclass(frozen=True)
class _LiveSettingSpec:
    action: str
    label: str
    group: str = ""


_CANONICAL_LIVE_SETTINGS: tuple[_LiveSettingSpec, ...] = (
    _LiveSettingSpec("quality_next", "Quality", "Graphics"),
    _LiveSettingSpec("poly_down", "Polygons -", "Graphics"),
    _LiveSettingSpec("poly_up", "Polygons +", "Graphics"),
    _LiveSettingSpec("reflections_down", "Reflections -", "Graphics"),
    _LiveSettingSpec("reflections_up", "Reflections +", "Graphics"),
    _LiveSettingSpec("smooth", "Smooth", "Graphics"),
    _LiveSettingSpec("texture_down", "Texture -", "Graphics"),
    _LiveSettingSpec("texture_up", "Texture +", "Graphics"),
    _LiveSettingSpec("gamma_down", "Gamma -", "Graphics"),
    _LiveSettingSpec("gamma_up", "Gamma +", "Graphics"),
    _LiveSettingSpec("tone_mapping", "Tone Map", "Graphics"),
    _LiveSettingSpec("toggle_render", "Wire/Fill", "Graphics"),
    _LiveSettingSpec("sky_cycle", "Cycle", "Sky"),
    _LiveSettingSpec("sky_time_down", "Earlier", "Sky"),
    _LiveSettingSpec("sky_time_up", "Later", "Sky"),
    _LiveSettingSpec("sky_sun_down", "Sun -", "Sky"),
    _LiveSettingSpec("sky_sun_up", "Sun +", "Sky"),
    _LiveSettingSpec("sky_clouds", "Clouds", "Sky"),
    _LiveSettingSpec("sky_stars", "Stars", "Sky"),
    _LiveSettingSpec("radius_down", "View Distance -", "World"),
    _LiveSettingSpec("radius_up", "View Distance +", "World"),
    _LiveSettingSpec("hlod_down", "HLOD Range -", "World"),
    _LiveSettingSpec("hlod_up", "HLOD Range +", "World"),
    _LiveSettingSpec("tree_lod_down", "Tree LOD -", "World"),
    _LiveSettingSpec("tree_lod_up", "Tree LOD +", "World"),
    _LiveSettingSpec("detail_budget_down", "Detail Budget -", "World"),
    _LiveSettingSpec("detail_budget_up", "Detail Budget +", "World"),
    _LiveSettingSpec("rebuild", "Rebuild", "World"),
    _LiveSettingSpec("seed_next", "Seed", "World"),
    _LiveSettingSpec("wind_down", "Wind -", "Physics"),
    _LiveSettingSpec("wind_up", "Wind +", "Physics"),
    _LiveSettingSpec("blade_down", "Swirl -", "Physics"),
    _LiveSettingSpec("blade_up", "Swirl +", "Physics"),
    _LiveSettingSpec("break_vessel", "Bowl", "Physics"),
    _LiveSettingSpec("pause", "Pause", "Physics"),
    _LiveSettingSpec("reset", "Reset", "Physics"),
    _LiveSettingSpec("next_camera", "Camera", "Camera"),
    _LiveSettingSpec("look_smoothing_down", "Look Smoothing -", "Camera"),
    _LiveSettingSpec("look_smoothing_up", "Look Smoothing +", "Camera"),
    _LiveSettingSpec("snapshot", "Snapshot", "Demo"),
    _LiveSettingSpec("done", "Done"),
    _LiveSettingSpec("apply", "Apply"),
    _LiveSettingSpec("cancel", "Exit Menu"),
    _LiveSettingSpec("quit", "Quit demo"),
)

CANONICAL_LIVE_MENU_ACTIONS = tuple(spec.action for spec in _CANONICAL_LIVE_SETTINGS)
DEFAULT_LIVE_FOOTER_ACTIONS = {"done", "quit"}


def canonical_live_menu_options(
    *,
    details: Mapping[str, object] | None = None,
    enabled_actions: Iterable[str] | None = None,
) -> tuple[LiveMenuOption, ...]:
    """Return the one shared settings inventory for live demos.

    ``enabled_actions`` names actions the current demo can apply. Unsupported
    actions remain visible and grayed out so settings do not disappear between
    demos.
    """

    detail_map = details or {}
    enabled = set(CANONICAL_LIVE_MENU_ACTIONS) if enabled_actions is None else set(enabled_actions) | DEFAULT_LIVE_FOOTER_ACTIONS
    return tuple(
        LiveMenuOption(
            spec.action,
            spec.label,
            str(detail_map.get(spec.action, "" if spec.action in enabled else "n/a")),
            spec.group,
            enabled=spec.action in enabled,
        )
        for spec in _CANONICAL_LIVE_SETTINGS
    )


def update_canonical_live_menu(
    menu: LiveMenu,
    *,
    details: Mapping[str, object] | None = None,
    enabled_actions: Iterable[str] | None = None,
) -> None:
    """Refresh a menu with the canonical inventory while preserving selection."""

    previous_action = menu.selected_action() if menu.options else "done"
    options = canonical_live_menu_options(details=details, enabled_actions=enabled_actions)
    menu.options = options
    actions = [option.action for option in options]
    if previous_action in actions:
        menu.selected_index = actions.index(previous_action)
    else:
        menu.selected_index = actions.index("done")
