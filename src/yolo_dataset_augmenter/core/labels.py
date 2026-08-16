from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True, slots=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )

    def with_class_id(self, class_id: int) -> YoloBox:
        return replace(self, class_id=class_id)


def parse_yolo_line(line: str) -> YoloBox:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError("Detection labels must contain exactly five fields.")
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
    except ValueError as exc:
        raise ValueError("Label fields must contain an integer class ID and numeric box values.") from exc

    values = (x_center, y_center, width, height)
    if class_id < 0:
        raise ValueError("Class ID cannot be negative.")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Bounding-box values must be finite.")
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        raise ValueError("Bounding-box centers must be normalized to [0, 1].")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError("Bounding-box dimensions must be normalized to (0, 1].")
    return YoloBox(class_id, x_center, y_center, width, height)


def parse_yolo_lines(lines: Iterable[str]) -> tuple[list[YoloBox], list[tuple[int, str]]]:
    boxes: list[YoloBox] = []
    errors: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            boxes.append(parse_yolo_line(line))
        except ValueError as exc:
            errors.append((line_number, str(exc)))
    return boxes, errors


def normalized_label_text(lines: Iterable[str], class_mapping: dict[int, int] | None = None) -> str:
    boxes, _ = parse_yolo_lines(lines)
    if class_mapping is not None:
        boxes = [
            box.with_class_id(class_mapping[box.class_id])
            for box in boxes
            if box.class_id in class_mapping
        ]
    return "\n".join(box.to_line() for box in boxes)
