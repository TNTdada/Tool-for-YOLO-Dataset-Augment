from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from yolo_dataset_augmenter import __version__
from yolo_dataset_augmenter.core.config_io import save_config
from yolo_dataset_augmenter.core.engine import CoreEngine, ProgressCallback
from yolo_dataset_augmenter.core.models import (
    AugmentationConfig,
    CancellationToken,
    DatasetScanResult,
    ProcessingMode,
    TaskResult,
)


class AugmenterService:
    def __init__(self, engine: CoreEngine | None = None) -> None:
        self.engine = engine or CoreEngine()
        self.config = AugmentationConfig()
        self.scan_result: DatasetScanResult | None = None

    def scan(self, dataset_dir: Path) -> DatasetScanResult:
        self.scan_result = self.engine.scan_dataset(dataset_dir, self.config)
        if not self.config.class_names:
            self.config.class_names = {class_id: str(class_id) for class_id in self.scan_result.classes}
        return self.scan_result

    def preview(
        self,
        image_file: str | None = None,
        run_augment: bool = True,
        randomize: bool = False,
    ):
        if self.scan_result is None:
            raise ValueError("Scan a dataset before previewing samples.")
        if image_file is None:
            seed = None if randomize else self.config.random_seed
            pair = self.engine.random_pair(self.scan_result, seed)
        else:
            try:
                pair = next(
                    pair for pair in self.scan_result.valid_pairs if pair[0] == image_file
                )
            except StopIteration as exc:
                raise ValueError(f"Preview image is not in the current scan: {image_file}") from exc
        return self.engine.preview_pair(
            self.scan_result.dataset_dir,
            pair,
            self.config,
            run_augment=run_augment,
            randomize=randomize,
        )

    def run(
        self,
        output_root: Path,
        mode: ProcessingMode,
        progress: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> TaskResult:
        if self.scan_result is None:
            raise ValueError("Scan a dataset before running a task.")
        errors = self.config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        result = self.engine.run_task(
            self.scan_result.dataset_dir,
            output_root,
            self.config,
            mode,
            progress,
            cancel_token,
        )
        self._write_run_artifacts(result)
        return result

    def _write_run_artifacts(self, result: TaskResult) -> None:
        if self.scan_result is None:
            raise ValueError("Scan a dataset before writing run artifacts.")
        save_config(result.output_dir / "run_config.json", self.config)
        summary = {
            "summary_version": 2,
            "application_version": __version__,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "mode": result.mode.value,
            "dataset_name": self.scan_result.dataset_dir.name,
            "output_name": result.output_dir.name,
            "split_counts": result.split_counts,
            "issues": [
                {
                    "kind": issue.kind.value,
                    "path": issue.path,
                    "line_number": issue.line_number,
                    "message": issue.message,
                }
                for issue in result.issues
            ],
        }
        (result.output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
