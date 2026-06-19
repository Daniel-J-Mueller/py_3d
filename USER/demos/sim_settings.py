"""Shared USER demo settings persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "USER" / "settings.json"
LIVE_SETTINGS_KEY = "live_menu_settings"


def load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def live_settings_for(name: str) -> dict[str, Any]:
    settings = load_settings()
    live_settings = settings.get(LIVE_SETTINGS_KEY, {})
    values = {}
    if isinstance(live_settings.get("global"), dict):
        values.update(live_settings["global"])
    if isinstance(live_settings.get(name), dict):
        values.update(live_settings[name])
    return values


def update_live_settings(name: str, values: dict[str, Any]) -> None:
    settings = load_settings()
    live_settings = settings.setdefault(LIVE_SETTINGS_KEY, {})
    section = live_settings.setdefault(name, {})
    section.update(values)
    save_settings(settings)
