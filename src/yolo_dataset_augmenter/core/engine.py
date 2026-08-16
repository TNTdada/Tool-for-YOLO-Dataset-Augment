from __future__ import annotations

import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

from .labels import YoloBox, normalized_label_text, parse_yolo_line, parse_yolo_lines
from .models import (
    AugmentationConfig,
    CancellationToken,
    DatasetIssue,
    DatasetIssueKind,
    DatasetScanResult,
    ProcessingMode,
    TaskCancelledError,
    TaskProgress,
    TaskResult,
)


ProgressCallback = Callable[[TaskProgress], None]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
AUGMENTED_SUFFIX = re.compile(r"(?:_aug\d+)+$", re.IGNORECASE)


class CoreTaskError(RuntimeError):
    def __init__(self, stage: str, message: str, path: Path | None = None) -> None:
        self.stage = stage
        self.path = path
        context = f" [{path}]" if path is not None else ""
        super().__init__(f"{stage}: {message}{context}")


def _emit(
    progress: ProgressCallback | None,
    stage: str,
    message: str,
    current: int = 0,
    total: int = 0,
) -> None:
    if progress is not None:
        progress(TaskProgress(stage, message, current, total))


def _check_cancel(cancel_token: CancellationToken | None, stage: str) -> None:
    if cancel_token is not None:
        cancel_token.raise_if_cancelled(stage)


class CoreEngine:
    def scan_dataset(
        self,
        dataset_dir: Path,
        config: AugmentationConfig,
        progress: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> DatasetScanResult:
        image_dir = dataset_dir / "images"
        label_dir = dataset_dir / "labels"
        if not dataset_dir.is_dir() or not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError("Dataset must contain images/ and labels/ folders.")

        image_paths = sorted(
            (
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ),
            key=lambda path: path.name.casefold(),
        )
        stem_groups: dict[str, list[Path]] = defaultdict(list)
        for image_path in image_paths:
            stem_groups[image_path.stem.casefold()].append(image_path)
        duplicate_names = {
            path.name
            for paths in stem_groups.values()
            if len(paths) > 1
            for path in paths
        }

        valid_pairs: list[tuple[str, str]] = []
        invalid_images: list[str] = []
        issues: list[DatasetIssue] = []
        classes: set[int] = set()
        total = len(image_paths)
        _emit(progress, "scan", "Scanning dataset.", 0, total)

        for current, image_path in enumerate(image_paths, start=1):
            _check_cancel(cancel_token, "scan")
            label_path = label_dir / f"{image_path.stem}.txt"
            image_ref = f"images/{image_path.name}"
            label_ref = f"labels/{label_path.name}"

            if image_path.name in duplicate_names:
                invalid_images.append(image_path.name)
                issues.append(
                    DatasetIssue(
                        DatasetIssueKind.DUPLICATE_STEM,
                        image_ref,
                        "Multiple images share the same stem and would overwrite each other.",
                    )
                )
                _emit(progress, "scan", f"Rejected duplicate stem: {image_path.name}", current, total)
                continue

            if not label_path.is_file():
                invalid_images.append(image_path.name)
                issues.append(
                    DatasetIssue(
                        DatasetIssueKind.MISSING_LABEL,
                        image_ref,
                        "The matching label file is missing.",
                    )
                )
                _emit(progress, "scan", f"Missing label: {image_path.name}", current, total)
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                invalid_images.append(image_path.name)
                issues.append(
                    DatasetIssue(
                        DatasetIssueKind.DAMAGED_IMAGE,
                        image_ref,
                        f"The image cannot be decoded ({type(exc).__name__}).",
                    )
                )
                _emit(progress, "scan", f"Damaged image: {image_path.name}", current, total)
                continue

            try:
                label_lines = label_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as exc:
                invalid_images.append(image_path.name)
                issues.append(
                    DatasetIssue(
                        DatasetIssueKind.INVALID_LABEL_LINE,
                        label_ref,
                        f"The label file cannot be read ({type(exc).__name__}).",
                    )
                )
                _emit(progress, "scan", f"Unreadable label: {label_path.name}", current, total)
                continue

            boxes, label_errors = parse_yolo_lines(label_lines)
            if not any(line.strip() for line in label_lines):
                issues.append(
                    DatasetIssue(
                        DatasetIssueKind.EMPTY_LABEL,
                        label_ref,
                        "Empty labels are accepted as negative samples.",
                    )
                )
            for line_number, message in label_errors:
                issues.append(
                    DatasetIssue(
                        DatasetIssueKind.INVALID_LABEL_LINE,
                        label_ref,
                        message,
                        line_number,
                    )
                )
            classes.update(box.class_id for box in boxes)
            valid_pairs.append((image_path.name, label_path.name))
            _emit(progress, "scan", f"Scanned {image_path.name}", current, total)

        sorted_classes = sorted(classes)
        config.class_names = {
            class_id: config.class_names.get(class_id, str(class_id))
            for class_id in sorted_classes
        }
        return DatasetScanResult(
            dataset_dir=dataset_dir,
            valid_pairs=valid_pairs,
            invalid_images=invalid_images,
            classes=sorted_classes,
            issues=issues,
        )

    def random_pair(
        self,
        scan_result: DatasetScanResult,
        seed: int | None = None,
    ) -> tuple[str, str]:
        if not scan_result.valid_pairs:
            raise ValueError("No valid image-label pairs are available.")
        return random.Random(seed).choice(scan_result.valid_pairs)

    def preview_pair(
        self,
        dataset_dir: Path,
        pair: tuple[str, str],
        config: AugmentationConfig,
        run_augment: bool,
        randomize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        image_path = dataset_dir / "images" / pair[0]
        label_path = dataset_dir / "labels" / pair[1]
        image = read_image(image_path)
        bboxes = label_path.read_text(encoding="utf-8").splitlines()
        resized, params = resize_and_pad(image, config.target_size)
        adjusted = adjust_bboxes(bboxes, (image.shape[1], image.shape[0]), params, config.target_size)
        original = draw_bboxes_on_image(resized, adjusted, config)
        if not run_augment:
            return original, None
        seed = None if randomize else config.random_seed
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        augmented_image, augmented_bboxes = augment_image(
            resized.copy(), adjusted.copy(), config, rng, np_rng
        )
        return original, draw_bboxes_on_image(augmented_image, augmented_bboxes, config)

    def run_task(
        self,
        dataset_dir: Path,
        output_root: Path,
        config: AugmentationConfig,
        mode: ProcessingMode,
        progress: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> TaskResult:
        errors = config.validate()
        if errors:
            raise ValueError("Invalid configuration: " + "; ".join(errors))

        scan_result = self.scan_dataset(dataset_dir, config, progress, cancel_token)
        if not scan_result.valid_pairs:
            raise ValueError("No valid image-label pairs are available.")
        _check_cancel(cancel_token, "prepare")
        run_dir = get_next_run_dir(output_root)
        augmentation_rng = random.Random(config.random_seed)
        split_seed = None if config.random_seed is None else config.random_seed + 1
        split_rng = random.Random(split_seed)
        np_rng = np.random.default_rng(config.random_seed)
        split_counts: dict[str, int] = {}

        if mode is ProcessingMode.AUGMENT_ONLY:
            process_dataset(
                scan_result,
                run_dir,
                config,
                progress,
                cancel_token,
                augmentation_rng,
                np_rng,
            )
        if mode in (ProcessingMode.FULL, ProcessingMode.PACK_ONLY):
            split_counts = prepare_yolo_dataset(
                dataset_dir,
                run_dir / "yolo_dataset",
                scan_result.classes,
                config,
                progress,
                cancel_token,
                split_rng,
                scan_result.valid_pairs,
            )
        if mode is ProcessingMode.FULL:
            split_counts = augment_packaged_dataset(
                run_dir / "yolo_dataset",
                config,
                progress,
                cancel_token,
                augmentation_rng,
                np_rng,
            )
        _emit(progress, "complete", "Task finished.", 1, 1)
        return TaskResult(
            output_dir=run_dir,
            mode=mode,
            message="Task finished.",
            split_counts=split_counts,
            issues=list(scan_result.issues),
        )


def read_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise CoreTaskError("read", "Cannot decode image", path) from exc


def safe_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise CoreTaskError("write", "Cannot encode image", path)
    encoded.tofile(str(path))


def get_next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for child in base_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("run_"):
            continue
        try:
            max_idx = max(max_idx, int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return base_dir / f"run_{max_idx + 1}"


def resize_and_pad(image: np.ndarray, target_size: int) -> tuple[np.ndarray, tuple[float, int, int]]:
    height, width = image.shape[:2]
    if width == 0 or height == 0:
        raise ValueError("Image dimensions must be positive.")
    scale = min(target_size / width, target_size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale <= 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    padded[top : top + new_height, left : left + new_width] = resized
    return padded, (scale, left, top)


def adjust_bboxes(
    bboxes: list[str],
    image_size: tuple[int, int],
    params: tuple[float, int, int],
    target_size: int,
) -> list[str]:
    scale, pad_x, pad_y = params
    original_width, original_height = image_size
    adjusted: list[str] = []
    for bbox in bboxes:
        if not bbox.strip():
            continue
        try:
            box = parse_yolo_line(bbox)
        except ValueError:
            continue
        xmin = ((box.x_center - box.width / 2) * original_width * scale + pad_x) / target_size
        xmax = ((box.x_center + box.width / 2) * original_width * scale + pad_x) / target_size
        ymin = ((box.y_center - box.height / 2) * original_height * scale + pad_y) / target_size
        ymax = ((box.y_center + box.height / 2) * original_height * scale + pad_y) / target_size
        xmin, xmax = float(np.clip(xmin, 0.0, 1.0)), float(np.clip(xmax, 0.0, 1.0))
        ymin, ymax = float(np.clip(ymin, 0.0, 1.0)), float(np.clip(ymax, 0.0, 1.0))
        width, height = xmax - xmin, ymax - ymin
        if width <= 0.0 or height <= 0.0:
            continue
        adjusted.append(
            YoloBox(
                box.class_id,
                xmin + width / 2,
                ymin + height / 2,
                width,
                height,
            ).to_line()
        )
    return adjusted


def rotate_bboxes(
    bboxes: list[str], angle: int, cx: float = 0.5, cy: float = 0.5
) -> list[str]:
    """Rotate and clip bounding boxes."""
    angle_rad = -math.radians(angle)
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    rotated: list[str] = []
    for bbox in bboxes:
        try:
            box = parse_yolo_line(bbox)
        except ValueError:
            continue
        corners = [
            (box.x_center - box.width / 2 - cx, box.y_center - box.height / 2 - cy),
            (box.x_center + box.width / 2 - cx, box.y_center - box.height / 2 - cy),
            (box.x_center - box.width / 2 - cx, box.y_center + box.height / 2 - cy),
            (box.x_center + box.width / 2 - cx, box.y_center + box.height / 2 - cy),
        ]
        rotated_corners = [
            (x * cosine - y * sine + cx, x * sine + y * cosine + cy)
            for x, y in corners
        ]
        xs = [corner[0] for corner in rotated_corners]
        ys = [corner[1] for corner in rotated_corners]
        xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
        ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))
        width, height = xmax - xmin, ymax - ymin
        if width > 0.01 and height > 0.01:
            rotated.append(
                YoloBox(
                    box.class_id,
                    xmin + width / 2,
                    ymin + height / 2,
                    width,
                    height,
                ).to_line()
            )
    return rotated


def augment_image(
    image: np.ndarray,
    bboxes: list[str],
    config: AugmentationConfig,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[str]]:
    rng = rng or random.Random()
    np_rng = np_rng or np.random.default_rng()
    enabled = [
        name
        for name, is_enabled in (
            ("brightness", config.aug_brightness_enabled),
            ("noise", config.aug_noise_enabled),
            ("occlusion", config.aug_occlusion_enabled),
            ("hflip", config.aug_hflip_enabled),
            ("vflip", config.aug_vflip_enabled),
            ("rotate", config.aug_rotate_enabled),
            ("blur", config.aug_blur_enabled),
        )
        if is_enabled
    ]
    if not enabled:
        return image, bboxes

    count = rng.randint(1, min(config.max_augs_per_image, len(enabled)))
    augmentations = rng.sample(enabled, k=count)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for augmentation in augmentations:
        if augmentation == "brightness":
            image_pil = ImageEnhance.Brightness(image_pil).enhance(
                rng.uniform(*config.brightness_factor)
            )
        elif augmentation == "noise":
            image_np = np.array(image_pil).astype(np.float32)
            noise = np_rng.normal(0, config.noise_std, image_np.shape)
            image_pil = Image.fromarray(np.clip(image_np + noise, 0, 255).astype(np.uint8))
        elif augmentation == "occlusion":
            image_np = np.array(image_pil)
            height, width, _ = image_np.shape
            for _ in range(config.occlusion_count):
                occ_width = min(width, rng.randint(*config.occlusion_size))
                occ_height = min(height, rng.randint(*config.occlusion_size))
                occ_x = rng.randint(0, width - occ_width)
                occ_y = rng.randint(0, height - occ_height)
                image_np[occ_y : occ_y + occ_height, occ_x : occ_x + occ_width] = np_rng.integers(
                    0, 256, (occ_height, occ_width, 3), dtype=np.uint8
                )
            image_pil = Image.fromarray(image_np)
        elif augmentation == "hflip":
            image_pil = image_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes, _ = parse_yolo_lines(bboxes)
            bboxes = [
                YoloBox(box.class_id, 1.0 - box.x_center, box.y_center, box.width, box.height).to_line()
                for box in boxes
            ]
        elif augmentation == "vflip":
            image_pil = image_pil.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            boxes, _ = parse_yolo_lines(bboxes)
            bboxes = [
                YoloBox(box.class_id, box.x_center, 1.0 - box.y_center, box.width, box.height).to_line()
                for box in boxes
            ]
        elif augmentation == "rotate":
            angle = rng.randint(*config.rotation_range)
            image_pil = image_pil.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=(114, 114, 114),
            )
            bboxes = rotate_bboxes(bboxes, angle)
        elif augmentation == "blur":
            image_pil = image_pil.filter(
                ImageFilter.GaussianBlur(radius=rng.uniform(*config.blur_radius))
            )

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR), bboxes


def draw_bboxes_on_image(
    image: np.ndarray, bboxes: list[str], config: AugmentationConfig
) -> np.ndarray:
    drawn = image.copy()
    height, width = drawn.shape[:2]
    boxes, _ = parse_yolo_lines(bboxes)
    for box in boxes:
        xmin = max(0, round((box.x_center - box.width / 2) * width))
        ymin = max(0, round((box.y_center - box.height / 2) * height))
        xmax = min(width, round((box.x_center + box.width / 2) * width))
        ymax = min(height, round((box.y_center + box.height / 2) * height))
        class_name = config.class_names.get(box.class_id, str(box.class_id))
        cv2.rectangle(drawn, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            drawn,
            class_name,
            (xmin, max(15, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)


def process_dataset(
    scan_result: DatasetScanResult,
    output_dir: Path,
    config: AugmentationConfig,
    progress: ProgressCallback | None = None,
    cancel_token: CancellationToken | None = None,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> None:
    rng = rng or random.Random(config.random_seed)
    np_rng = np_rng or np.random.default_rng(config.random_seed)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    total = len(scan_result.valid_pairs) * (config.augment_per_image + 1)
    current = 0
    _emit(progress, "augment", "Processing images.", current, total)

    for image_file, label_file in scan_result.valid_pairs:
        image_path = scan_result.dataset_dir / "images" / image_file
        label_path = scan_result.dataset_dir / "labels" / label_file
        try:
            _check_cancel(cancel_token, "augment")
            image = read_image(image_path)
            bboxes = label_path.read_text(encoding="utf-8").splitlines()
            resized, params = resize_and_pad(image, config.target_size)
            adjusted = adjust_bboxes(
                bboxes, (image.shape[1], image.shape[0]), params, config.target_size
            )
            base_name = image_path.stem
            safe_imwrite(images_out / f"{base_name}.jpg", resized)
            (labels_out / f"{base_name}.txt").write_text(
                "\n".join(adjusted), encoding="utf-8"
            )
            current += 1
            _emit(progress, "augment", f"Processed {image_file}", current, total)

            for index in range(config.augment_per_image):
                _check_cancel(cancel_token, "augment")
                augmented_image, augmented_bboxes = augment_image(
                    resized.copy(), adjusted.copy(), config, rng, np_rng
                )
                safe_imwrite(images_out / f"{base_name}_aug{index}.jpg", augmented_image)
                (labels_out / f"{base_name}_aug{index}.txt").write_text(
                    "\n".join(augmented_bboxes), encoding="utf-8"
                )
                current += 1
                _emit(progress, "augment", f"Augmented {image_file}", current, total)
        except TaskCancelledError:
            raise
        except CoreTaskError:
            raise
        except Exception as exc:
            raise CoreTaskError("augment", "Failed to process sample", image_path) from exc


def augment_packaged_dataset(
    dataset_dir: Path,
    config: AugmentationConfig,
    progress: ProgressCallback | None = None,
    cancel_token: CancellationToken | None = None,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> dict[str, int]:
    """Augment samples within assigned splits."""
    rng = rng or random.Random(config.random_seed)
    np_rng = np_rng or np.random.default_rng(config.random_seed)
    split_pairs = {
        split: _collect_pairs(dataset_dir / split)
        for split in ("train", "val", "test")
    }
    total = sum(len(pairs) for pairs in split_pairs.values()) * (
        config.augment_per_image + 1
    )
    current = 0
    _emit(progress, "augment", "Augmenting packaged YOLO dataset.", current, total)

    for split, pairs in split_pairs.items():
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        for image_file, label_file in pairs:
            image_path = images_dir / image_file
            label_path = labels_dir / label_file
            try:
                _check_cancel(cancel_token, "augment")
                image = read_image(image_path)
                bboxes = label_path.read_text(encoding="utf-8").splitlines()
                resized, params = resize_and_pad(image, config.target_size)
                adjusted = adjust_bboxes(
                    bboxes,
                    (image.shape[1], image.shape[0]),
                    params,
                    config.target_size,
                )
                safe_imwrite(image_path, resized)
                label_path.write_text("\n".join(adjusted), encoding="utf-8")
                current += 1
                _emit(
                    progress,
                    "augment",
                    f"Processed {split}/{image_file}",
                    current,
                    total,
                )

                for index in range(config.augment_per_image):
                    _check_cancel(cancel_token, "augment")
                    augmented_image, augmented_bboxes = augment_image(
                        resized.copy(), adjusted.copy(), config, rng, np_rng
                    )
                    base_name = image_path.stem
                    safe_imwrite(
                        images_dir / f"{base_name}_aug{index}.jpg",
                        augmented_image,
                    )
                    (labels_dir / f"{base_name}_aug{index}.txt").write_text(
                        "\n".join(augmented_bboxes), encoding="utf-8"
                    )
                    current += 1
                    _emit(
                        progress,
                        "augment",
                        f"Augmented {split}/{image_file}",
                        current,
                        total,
                    )
            except TaskCancelledError:
                raise
            except CoreTaskError:
                raise
            except Exception as exc:
                raise CoreTaskError(
                    "augment", "Failed to augment packaged sample", image_path
                ) from exc

    return {
        split: len(pairs) * (config.augment_per_image + 1)
        for split, pairs in split_pairs.items()
    }


def source_group_key(stem: str) -> str:
    return AUGMENTED_SUFFIX.sub("", stem).casefold()


def _collect_pairs(input_dir: Path) -> list[tuple[str, str]]:
    image_dir = input_dir / "images"
    label_dir = input_dir / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError("Input directory must contain images/ and labels/ folders.")
    pairs: list[tuple[str, str]] = []
    for image_path in sorted(image_dir.iterdir(), key=lambda path: path.name.casefold()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_file = f"{image_path.stem}.txt"
        if (label_dir / label_file).is_file():
            pairs.append((image_path.name, label_file))
    return pairs


def _allocate_split_counts(total: int, config: AugmentationConfig) -> dict[str, int]:
    ratios = {
        "train": config.train_ratio,
        "val": config.val_ratio,
        "test": config.test_ratio,
    }
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    priority = {"train": 0, "val": 1, "test": 2}
    order = sorted(raw, key=lambda name: (-(raw[name] - counts[name]), priority[name]))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def split_grouped_pairs(
    pairs: list[tuple[str, str]],
    config: AugmentationConfig,
    rng: random.Random,
) -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pair in pairs:
        groups[source_group_key(Path(pair[0]).stem)].append(pair)
    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    counts = _allocate_split_counts(len(group_keys), config)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    split_keys = {
        "train": group_keys[:train_end],
        "val": group_keys[train_end:val_end],
        "test": group_keys[val_end:],
    }
    return {
        split: [pair for key in keys for pair in groups[key]]
        for split, keys in split_keys.items()
    }


def prepare_yolo_dataset(
    input_dir: Path,
    output_dir: Path,
    classes: list[int],
    config: AugmentationConfig,
    progress: ProgressCallback | None = None,
    cancel_token: CancellationToken | None = None,
    rng: random.Random | None = None,
    source_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, int]:
    rng = rng or random.Random(config.random_seed)
    pairs = list(source_pairs) if source_pairs is not None else _collect_pairs(input_dir)
    splits = split_grouped_pairs(pairs, config, rng)
    class_mapping = {class_id: index for index, class_id in enumerate(sorted(classes))}
    total = len(pairs)
    current = 0
    _emit(progress, "pack", "Packaging YOLO dataset.", current, total)

    for split, split_pairs in splits.items():
        images_out = output_dir / split / "images"
        labels_out = output_dir / split / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image_file, label_file in split_pairs:
            _check_cancel(cancel_token, "pack")
            image_source = input_dir / "images" / image_file
            label_source = input_dir / "labels" / label_file
            try:
                shutil.copy2(image_source, images_out / image_file)
                label_text = normalized_label_text(
                    label_source.read_text(encoding="utf-8").splitlines(), class_mapping
                )
                (labels_out / label_file).write_text(label_text, encoding="utf-8")
            except Exception as exc:
                raise CoreTaskError("pack", "Failed to package sample", image_source) from exc
            current += 1
            _emit(progress, "pack", f"Packaged {image_file}", current, total)

    sorted_classes = sorted(classes)
    data = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(sorted_classes),
        "names": [config.class_names.get(class_id, str(class_id)) for class_id in sorted_classes],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return {split: len(split_pairs) for split, split_pairs in splits.items()}
