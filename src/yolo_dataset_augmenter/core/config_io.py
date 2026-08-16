from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AugmentationConfig


CURRENT_CONFIG_VERSION = 1


def _migrate_payload(payload: dict[str, Any], version: int) -> dict[str, Any]:
    if version < 0:
        raise ValueError("Config version cannot be negative.")
    if version > CURRENT_CONFIG_VERSION:
        raise ValueError(
            f"Config version {version} is newer than the supported version "
            f"{CURRENT_CONFIG_VERSION}."
        )

    migrated = dict(payload)
    if version == 0:
        # Migrate the legacy flat format.
        return migrated
    return migrated


def load_config(path: Path) -> AugmentationConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")

    if "config_version" in data:
        version = data.get("config_version")
        payload = data.get("config")
        if not isinstance(version, int):
            raise ValueError("config_version must be an integer.")
        if not isinstance(payload, dict):
            raise ValueError("Versioned config must contain a config object.")
    else:
        version = 0
        payload = data

    config = AugmentationConfig.from_mapping(_migrate_payload(payload, version))
    errors = config.validate()
    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))
    return config


def save_config(path: Path, config: AugmentationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = config.validate()
    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))
    payload: dict[str, Any] = {
        "config_version": CURRENT_CONFIG_VERSION,
        "config": config.to_legacy_dict(),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
