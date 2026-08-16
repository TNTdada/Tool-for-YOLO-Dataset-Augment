from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from yolo_dataset_augmenter.app.services import AugmenterService
from yolo_dataset_augmenter.core.engine import (
    CoreEngine,
    CoreTaskError,
    adjust_bboxes,
    augment_image,
    read_image,
    resize_and_pad,
    rotate_bboxes,
    source_group_key,
    split_grouped_pairs,
)
from yolo_dataset_augmenter.core.labels import parse_yolo_line
from yolo_dataset_augmenter.core.models import (
    AugmentationConfig,
    CancellationToken,
    DatasetIssueKind,
    ProcessingMode,
    TaskCancelledError,
    TaskProgress,
)


def _write_image(path: Path, color: tuple[int, int, int] = (30, 60, 90)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), color).save(path)


def _write_sample(dataset: Path, stem: str, label_text: str, suffix: str = ".jpg") -> None:
    _write_image(dataset / "images" / f"{stem}{suffix}")
    label_path = dataset / "labels" / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(label_text, encoding="utf-8")


def _deterministic_config(**overrides: object) -> AugmentationConfig:
    values: dict[str, object] = {
        "target_size": 32,
        "augment_per_image": 1,
        "max_augs_per_image": 1,
        "aug_brightness_enabled": False,
        "aug_noise_enabled": False,
        "aug_occlusion_enabled": False,
        "aug_hflip_enabled": True,
        "aug_vflip_enabled": False,
        "aug_rotate_enabled": False,
        "aug_blur_enabled": False,
        "random_seed": 17,
    }
    values.update(overrides)
    return AugmentationConfig.from_mapping(values)


def _split_groups(yolo_dir: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for image_path in (yolo_dir / split / "images").iterdir():
            key = source_group_key(image_path.stem)
            assert key not in assignments or assignments[key] == split
            assignments[key] = split
    return assignments


def test_scan_dataset_records_explicit_input_policies(tmp_path):
    dataset = tmp_path / "dataset"
    _write_sample(dataset, "valid", "2 0.5 0.5 0.4 0.2\ninvalid line")
    _write_sample(dataset, "empty", "")
    _write_image(dataset / "images" / "missing.jpg")
    _write_sample(dataset, "duplicate", "2 0.5 0.5 0.2 0.2", ".jpg")
    _write_image(dataset / "images" / "duplicate.png")
    _write_sample(dataset, "damaged", "2 0.5 0.5 0.2 0.2")
    (dataset / "images" / "damaged.jpg").write_bytes(b"not an image")
    config = AugmentationConfig(class_names={2: "widget", 99: "unused"})

    result = CoreEngine().scan_dataset(dataset, config)

    assert {pair[0] for pair in result.valid_pairs} == {"valid.jpg", "empty.jpg"}
    assert set(result.invalid_images) == {
        "missing.jpg",
        "duplicate.jpg",
        "duplicate.png",
        "damaged.jpg",
    }
    assert result.classes == [2]
    assert config.class_names == {2: "widget"}
    issue_kinds = {issue.kind for issue in result.issues}
    assert issue_kinds == {
        DatasetIssueKind.MISSING_LABEL,
        DatasetIssueKind.EMPTY_LABEL,
        DatasetIssueKind.INVALID_LABEL_LINE,
        DatasetIssueKind.DUPLICATE_STEM,
        DatasetIssueKind.DAMAGED_IMAGE,
    }
    assert all(not Path(issue.path).is_absolute() for issue in result.issues)
    assert all(issue.path.startswith(("images/", "labels/")) for issue in result.issues)

    packed = CoreEngine().run_task(
        dataset,
        tmp_path / "outputs",
        _deterministic_config(augment_per_image=0),
        ProcessingMode.PACK_ONLY,
    )
    packaged_names = {
        path.name
        for path in (packed.output_dir / "yolo_dataset").glob("*/images/*")
    }
    assert packaged_names == {"valid.jpg", "empty.jpg"}


def test_resize_and_bbox_adjustment_preserve_letterbox_geometry():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    resized, params = resize_and_pad(image, 200)
    adjusted = adjust_bboxes(["0 0.5 0.5 0.5 0.5"], (200, 100), params, 200)
    box = parse_yolo_line(adjusted[0])

    assert resized.shape == (200, 200, 3)
    assert params == (1.0, 0, 50)
    assert box.x_center == pytest.approx(0.5)
    assert box.y_center == pytest.approx(0.5)
    assert box.width == pytest.approx(0.5)
    assert box.height == pytest.approx(0.25)


def test_flip_and_noise_are_reproducible_with_local_random_generators():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))
    flip_config = _deterministic_config(augment_per_image=0)
    _, flipped_boxes = augment_image(
        image.copy(),
        ["3 0.2 0.4 0.1 0.2"],
        flip_config,
        random.Random(5),
        np.random.default_rng(5),
    )
    flipped = parse_yolo_line(flipped_boxes[0])
    assert flipped.x_center == pytest.approx(0.8)
    assert flipped.y_center == pytest.approx(0.4)

    noise_config = _deterministic_config(
        augment_per_image=0,
        aug_hflip_enabled=False,
        aug_noise_enabled=True,
        noise_std=5.0,
    )
    first, _ = augment_image(
        image.copy(), [], noise_config, random.Random(7), np.random.default_rng(7)
    )
    second, _ = augment_image(
        image.copy(), [], noise_config, random.Random(7), np.random.default_rng(7)
    )
    assert np.array_equal(first, second)


def test_rotated_boxes_use_documented_axis_aligned_envelope():
    rotated = rotate_bboxes(["0 0.5 0.5 0.2 0.4"], 45)
    box = parse_yolo_line(rotated[0])

    assert box.x_center == pytest.approx(0.5)
    assert box.y_center == pytest.approx(0.5)
    assert box.width == pytest.approx(0.424264, abs=1e-6)
    assert box.height == pytest.approx(0.424264, abs=1e-6)


def test_grouped_split_never_separates_augmented_siblings():
    pairs = [
        (name, f"{Path(name).stem}.txt")
        for name in (
            "a.jpg",
            "a_aug0.jpg",
            "b.jpg",
            "b_aug0.jpg",
            "c.jpg",
            "c_aug0.jpg",
            "d.jpg",
            "d_aug0.jpg",
        )
    ]
    config = AugmentationConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)

    splits = split_grouped_pairs(pairs, config, random.Random(3))

    assignments: dict[str, str] = {}
    for split, split_pairs in splits.items():
        for image_file, _ in split_pairs:
            group = source_group_key(Path(image_file).stem)
            assert group not in assignments or assignments[group] == split
            assignments[group] = split
    assert {name: len(items) for name, items in splits.items()} == {
        "train": 4,
        "val": 2,
        "test": 2,
    }


def test_all_processing_modes_seeded_split_and_non_contiguous_class_mapping(tmp_path):
    dataset = tmp_path / "dataset"
    _write_sample(dataset, "a", "2 0.5 0.5 0.4 0.4\nbad label")
    _write_sample(dataset, "b", "5 0.4 0.4 0.2 0.2")
    _write_sample(dataset, "c", "")
    output_root = tmp_path / "outputs"
    config = _deterministic_config(
        train_ratio=0.34,
        val_ratio=0.33,
        test_ratio=0.33,
        class_names={2: "bolt", 5: "nut"},
    )
    engine = CoreEngine()

    full_events: list[TaskProgress] = []
    full_first = engine.run_task(
        dataset, output_root, config, ProcessingMode.FULL, full_events.append
    )
    full_second = engine.run_task(dataset, output_root, config, ProcessingMode.FULL)
    augment_only = engine.run_task(dataset, output_root, config, ProcessingMode.AUGMENT_ONLY)
    pack_only = engine.run_task(dataset, output_root, config, ProcessingMode.PACK_ONLY)

    assert full_first.output_dir.name == "run_1"
    assert full_second.output_dir.name == "run_2"
    assert augment_only.output_dir.name == "run_3"
    assert pack_only.output_dir.name == "run_4"
    assert full_first.split_counts == {"train": 2, "val": 2, "test": 2}
    assert pack_only.split_counts == {"train": 1, "val": 1, "test": 1}
    assert not (full_first.output_dir / "images").exists()
    assert not (full_first.output_dir / "labels").exists()
    assert (full_first.output_dir / "yolo_dataset").is_dir()
    assert not list((pack_only.output_dir / "yolo_dataset").glob("*/images/*_aug*.jpg"))
    stage_order = [event.stage for event in full_events]
    assert stage_order.index("pack") < stage_order.index("augment")
    assert _split_groups(full_first.output_dir / "yolo_dataset") == _split_groups(
        full_second.output_dir / "yolo_dataset"
    )
    assert not (augment_only.output_dir / "yolo_dataset").exists()
    assert len(list((augment_only.output_dir / "images").glob("*.jpg"))) == 6

    for split in ("train", "val", "test"):
        split_images = full_first.output_dir / "yolo_dataset" / split / "images"
        originals = [path for path in split_images.glob("*.jpg") if "_aug" not in path.stem]
        augmented = list(split_images.glob("*_aug0.jpg"))
        assert len(originals) == 1
        assert [path.stem.removesuffix("_aug0") for path in augmented] == [
            originals[0].stem
        ]

    yaml_data = yaml.safe_load(
        (full_first.output_dir / "yolo_dataset" / "data.yaml").read_text(encoding="utf-8")
    )
    assert yaml_data["nc"] == 2
    assert yaml_data["names"] == ["bolt", "nut"]
    packaged_labels = list((full_first.output_dir / "yolo_dataset").glob("*/labels/*.txt"))
    class_ids = {
        int(line.split()[0])
        for path in packaged_labels
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert class_ids == {0, 1}
    assert all("bad label" not in path.read_text(encoding="utf-8") for path in packaged_labels)


def test_progress_callback_can_cancel_a_running_task(tmp_path):
    dataset = tmp_path / "dataset"
    _write_sample(dataset, "a", "0 0.5 0.5 0.4 0.4")
    _write_sample(dataset, "b", "0 0.5 0.5 0.4 0.4")
    token = CancellationToken()
    events: list[TaskProgress] = []

    def on_progress(event: TaskProgress) -> None:
        events.append(event)
        if event.stage == "augment" and event.current == 1:
            token.cancel()

    with pytest.raises(TaskCancelledError, match="augment"):
        CoreEngine().run_task(
            dataset,
            tmp_path / "outputs",
            _deterministic_config(augment_per_image=2),
            ProcessingMode.AUGMENT_ONLY,
            on_progress,
            token,
        )

    assert {event.stage for event in events} == {"scan", "augment"}


def test_image_decode_error_contains_stage_and_path(tmp_path):
    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"broken")

    with pytest.raises(CoreTaskError) as error:
        read_image(image_path)

    assert error.value.stage == "read"
    assert error.value.path == image_path
    assert str(image_path) in str(error.value)


def test_preview_can_select_an_image_and_randomize_each_click(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(dataset / "images" / "red.jpg")
    Image.new("RGB", (16, 16), (0, 255, 0)).save(dataset / "images" / "green.jpg")
    (dataset / "labels" / "red.txt").write_text("", encoding="utf-8")
    (dataset / "labels" / "green.txt").write_text("", encoding="utf-8")
    service = AugmenterService()
    service.config = _deterministic_config(
        target_size=16,
        augment_per_image=0,
        aug_hflip_enabled=False,
        aug_noise_enabled=True,
        noise_std=20.0,
    )
    service.scan(dataset)

    original, no_augmentation = service.preview(
        image_file="green.jpg",
        run_augment=False,
    )
    assert no_augmentation is None
    assert tuple(original[8, 8]) == pytest.approx((0, 255, 1), abs=3)

    randomized = []
    for _ in range(3):
        _, augmented = service.preview(
            image_file="green.jpg",
            run_augment=True,
            randomize=True,
        )
        assert augmented is not None
        randomized.append(augmented.tobytes())
    assert len(set(randomized)) > 1
