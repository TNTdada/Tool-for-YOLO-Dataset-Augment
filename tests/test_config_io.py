from __future__ import annotations

import json

import pytest

from yolo_dataset_augmenter.core.config_io import CURRENT_CONFIG_VERSION, load_config, save_config
from yolo_dataset_augmenter.core.models import AugmentationConfig


def test_config_round_trip(tmp_path):
    config = AugmentationConfig(class_names={0: "person", 1: "ball"})
    path = tmp_path / "settings.json"

    save_config(path, config)
    loaded = load_config(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["config_version"] == CURRENT_CONFIG_VERSION
    assert loaded.class_names == config.class_names
    assert loaded.target_size == config.target_size
    assert loaded.brightness_factor == config.brightness_factor
    assert loaded.random_seed == config.random_seed


def test_default_config_is_valid():
    assert AugmentationConfig().validate() == []


def test_load_config_migrates_original_flat_format(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"target_size": 320, "class_names": {"5": "part"}}),
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded.target_size == 320
    assert loaded.class_names == {5: "part"}
    assert loaded.random_seed == 42


def test_load_config_rejects_future_version(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({"config_version": CURRENT_CONFIG_VERSION + 1, "config": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="newer than"):
        load_config(path)
