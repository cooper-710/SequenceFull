"""
Utility helpers for loading and persisting application settings.

Settings are stored as JSON and merged with DEFAULT_SETTINGS so new keys
are available automatically without wiping a user's existing preferences.
"""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parent
SETTINGS_DIR = ROOT_DIR / "config"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "general": {
        "theme": "dark",
        "auto_collapse_sidebar": False,
        "show_beta_features": True,
        "auto_check_updates": True,
    },
    "reports": {
        "default_team": "AUTO",
        "default_pitcher_team": "AUTO",
        "default_opponent": "",
        "default_season_start": "2025-03-20",
        "use_next_series": False,
        "auto_open_downloads": True,
        "default_export_format": "pdf",
    },
    "data_sources": {
        "data_directory": str(ROOT_DIR / "data"),
        "use_local_cache": True,
        "auto_refresh_on_startup": False,
        "allow_unverified_sources": False,
    },
    "integrations": {
        "sportradar_api_key": "",
        "statcast_cookie": "",
        "fangraphs_username": "",
        "fangraphs_password": "",
        "trumedia_token": "",
    },
    "notifications": {
        "enable_email": False,
        "email_address": "",
        "notify_on_completion": True,
        "notify_on_failure": True,
    },
}

_lock = threading.RLock()


def _ensure_settings_file() -> None:
    """Create the settings directory and file with defaults if needed."""
    if not SETTINGS_DIR.exists():
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

    if not SETTINGS_PATH.exists():
        save_settings(DEFAULT_SETTINGS)


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_settings() -> Dict[str, Any]:
    """Load settings from disk, falling back to defaults as needed."""
    with _lock:
        _ensure_settings_file()
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as fh:
                stored = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # If file is corrupt, back it up and start fresh
            backup_path = SETTINGS_PATH.with_suffix(".backup.json")
            try:
                SETTINGS_PATH.replace(backup_path)
            except OSError:
                pass
            stored = {}

        merged = deepcopy(DEFAULT_SETTINGS)
        if isinstance(stored, dict):
            merged = _deep_merge(merged, stored)
        return merged


def save_settings(settings: Dict[str, Any]) -> None:
    """Write the provided settings dictionary to disk."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2, sort_keys=True)


def update_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update settings with the provided payload and persist the result."""
    if not isinstance(payload, dict):
        raise ValueError("Settings payload must be an object")

    with _lock:
        current = load_settings()
        updated = _deep_merge(current, payload)
        save_settings(updated)
        return updated


def reset_settings() -> Dict[str, Any]:
    """Reset settings back to defaults."""
    with _lock:
        save_settings(DEFAULT_SETTINGS)
        return deepcopy(DEFAULT_SETTINGS)

