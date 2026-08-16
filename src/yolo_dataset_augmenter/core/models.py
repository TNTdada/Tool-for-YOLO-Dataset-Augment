from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any


class ProcessingMode(str, Enum):
    FULL = "full"
    AUGMENT_ONLY = "augment_only"
    PACK_ONLY = "pack_only"


class DatasetIssueKind(str, Enum):
    MISSING_LABEL = "missing_label"
    EMPTY_LABEL = "empty_label"
    INVALID_LABEL_LINE = "invalid_label_line"
    DUPLICATE_STEM = "duplicate_stem"
    DAMAGED_IMAGE = "damaged_image"


class TaskCancelledError(RuntimeError):
    """Task cancelled."""


@dataclass(slots=True)
class AugmentationConfig:
    target_size: int = 640
    augment_per_image: int = 5
    max_augs_per_image: int = 3
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    class_names: dict[int, str] = field(default_factory=dict)
    aug_brightness_enabled: bool = True
    brightness_factor: tuple[float, float] = (0.6, 1.2)
    aug_noise_enabled: bool = True
    noise_std: float = 10.0
    aug_occlusion_enabled: bool = True
    occlusion_size: tuple[int, int] = (40, 100)
    occlusion_count: int = 2
    aug_hflip_enabled: bool = True
    aug_vflip_enabled: bool = True
    aug_rotate_enabled: bool = True
    rotation_range: tuple[int, int] = (-15, 15)
    aug_blur_enabled: bool = True
    blur_radius: tuple[float, float] = (0.1, 0.5)
    random_seed: int | None = 42

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.target_size <= 0:
            errors.append("target_size must be positive.")
        if self.augment_per_image < 0:
            errors.append("augment_per_image cannot be negative.")
        if self.max_augs_per_image <= 0:
            errors.append("max_augs_per_image must be positive.")
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 0.001:
            errors.append("train/val/test ratios must sum to 1.0.")
        if any(r < 0 for r in (self.train_ratio, self.val_ratio, self.test_ratio)):
            errors.append("split ratios cannot be negative.")
        if self.random_seed is not None and self.random_seed < 0:
            errors.append("random_seed cannot be negative.")
        if self.brightness_factor[0] > self.brightness_factor[1] or self.brightness_factor[0] < 0:
            errors.append("brightness_factor must be an ordered non-negative range.")
        if self.noise_std < 0:
            errors.append("noise_std cannot be negative.")
        if (
            self.occlusion_size[0] <= 0
            or self.occlusion_size[0] > self.occlusion_size[1]
            or self.occlusion_count < 0
        ):
            errors.append("occlusion settings must use positive ordered sizes and a non-negative count.")
        if self.rotation_range[0] > self.rotation_range[1]:
            errors.append("rotation_range must be ordered.")
        if self.blur_radius[0] < 0 or self.blur_radius[0] > self.blur_radius[1]:
            errors.append("blur_radius must be an ordered non-negative range.")
        if any(class_id < 0 for class_id in self.class_names):
            errors.append("class IDs cannot be negative.")
        return errors

    def to_legacy_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AugmentationConfig":
        normalized = dict(data)
        if "class_names" in normalized:
            normalized["class_names"] = {
                int(key): str(value) for key, value in normalized["class_names"].items()
            }
        for key in ("brightness_factor", "occlusion_size", "rotation_range", "blur_radius"):
            if key in normalized and isinstance(normalized[key], list):
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)


@dataclass(slots=True)
class DatasetScanResult:
    dataset_dir: Path
    valid_pairs: list[tuple[str, str]]
    invalid_images: list[str]
    classes: list[int]
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.valid_pairs) + len(self.invalid_images)

    @property
    def valid_count(self) -> int:
        return len(self.valid_pairs)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_images)

    @property
    def warning_count(self) -> int:
        return len(self.issues)


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    kind: DatasetIssueKind
    path: str
    message: str
    line_number: int | None = None


@dataclass(slots=True)
class TaskProgress:
    stage: str
    message: str
    current: int = 0
    total: int = 0

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return min(100, max(0, int(self.current / self.total * 100)))


@dataclass(slots=True)
class TaskResult:
    output_dir: Path
    mode: ProcessingMode
    message: str
    split_counts: dict[str, int] = field(default_factory=dict)
    issues: list[DatasetIssue] = field(default_factory=list)


@dataclass(slots=True)
class CancellationToken:
    _event: Event = field(default_factory=Event, init=False, repr=False)

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self, stage: str = "task") -> None:
        if self.is_cancelled:
            raise TaskCancelledError(f"Task cancelled during {stage}.")
