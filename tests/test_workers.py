from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6.QtWidgets import QApplication

from yolo_dataset_augmenter.core.engine import CoreEngine
from yolo_dataset_augmenter.core.models import AugmentationConfig, ProcessingMode
from yolo_dataset_augmenter.ui.workers import ScanWorker, TaskWorker


def _app() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


def _wait_until(predicate, timeout: float = 5.0) -> None:
    app = _app()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out while waiting for a Qt worker.")


def _dataset(root: Path, count: int = 2) -> Path:
    dataset = root / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    for index in range(count):
        Image.new("RGB", (16, 12), (index * 20, 40, 80)).save(
            dataset / "images" / f"sample_{index}.jpg"
        )
        (dataset / "labels" / f"sample_{index}.txt").write_text(
            "3 0.5 0.5 0.4 0.4",
            encoding="utf-8",
        )
    return dataset


def test_scan_worker_reports_success_and_progress(tmp_path):
    dataset = _dataset(tmp_path)
    results = []
    events = []
    failures = []
    worker = ScanWorker(dataset, AugmentationConfig())
    worker.succeeded.connect(results.append)
    worker.progress.connect(events.append)
    worker.failed.connect(lambda message, details: failures.append((message, details)))

    worker.start()
    _wait_until(lambda: worker.isFinished())

    assert not failures
    assert results[0].valid_count == 2
    assert events[0].stage == "scan"
    assert events[-1].current == 2


def test_task_worker_writes_config_and_summary_artifacts(tmp_path):
    dataset = _dataset(tmp_path)
    config = AugmentationConfig(
        target_size=32,
        augment_per_image=0,
        class_names={3: "part"},
        random_seed=8,
    )
    scan_result = CoreEngine().scan_dataset(dataset, config)
    results = []
    failures = []
    worker = TaskWorker(
        scan_result,
        tmp_path / "outputs",
        config,
        ProcessingMode.PACK_ONLY,
    )
    worker.succeeded.connect(results.append)
    worker.failed.connect(lambda message, details: failures.append((message, details)))

    worker.start()
    _wait_until(lambda: worker.isFinished())

    assert not failures
    result = results[0]
    assert (result.output_dir / "run_config.json").is_file()
    summary_path = result.output_dir / "run_summary.json"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["summary_version"] == 2
    assert summary["mode"] == "pack_only"
    assert summary["application_version"] == "3.0.0"
    assert summary["dataset_name"] == dataset.name
    assert summary["output_name"] == result.output_dir.name
    assert summary["split_counts"] == result.split_counts
    assert "dataset_dir" not in summary
    assert "output_dir" not in summary
    assert str(tmp_path) not in summary_text


def test_workers_forward_failure_and_cancellation(tmp_path):
    failures = []
    failed_worker = ScanWorker(tmp_path / "missing", AugmentationConfig())
    failed_worker.failed.connect(lambda message, details: failures.append((message, details)))
    failed_worker.start()
    _wait_until(lambda: failed_worker.isFinished())
    assert "images/ and labels/" in failures[0][0]
    assert "Traceback" in failures[0][1]

    dataset = _dataset(tmp_path / "cancel")
    cancelled = []
    cancelled_worker = ScanWorker(dataset, AugmentationConfig())
    cancelled_worker.cancelled.connect(cancelled.append)
    cancelled_worker.cancel()
    cancelled_worker.start()
    _wait_until(lambda: cancelled_worker.isFinished())
    assert "cancelled during scan" in cancelled[0]
