from __future__ import annotations

import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from yolo_dataset_augmenter.app.services import AugmenterService
from yolo_dataset_augmenter.core.engine import CoreEngine
from yolo_dataset_augmenter.core.models import (
    AugmentationConfig,
    CancellationToken,
    DatasetScanResult,
    ProcessingMode,
    TaskCancelledError,
)


class ScanWorker(QThread):
    progress = pyqtSignal(object)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)
    cancelled = pyqtSignal(str)

    def __init__(
        self,
        dataset_dir: Path,
        config: AugmentationConfig,
        engine: CoreEngine | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.config = config
        self.engine = engine or CoreEngine()
        self.cancel_token = CancellationToken()

    def cancel(self) -> None:
        self.cancel_token.cancel()

    def run(self) -> None:
        try:
            result = self.engine.scan_dataset(
                self.dataset_dir,
                self.config,
                self.progress.emit,
                self.cancel_token,
            )
            self.succeeded.emit(result)
        except TaskCancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())


class TaskWorker(QThread):
    progress = pyqtSignal(object)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)
    cancelled = pyqtSignal(str)

    def __init__(
        self,
        scan_result: DatasetScanResult,
        output_root: Path,
        config: AugmentationConfig,
        mode: ProcessingMode,
        engine: CoreEngine | None = None,
    ) -> None:
        super().__init__()
        self.scan_result = scan_result
        self.output_root = output_root
        self.config = config
        self.mode = mode
        self.engine = engine or CoreEngine()
        self.cancel_token = CancellationToken()

    def cancel(self) -> None:
        self.cancel_token.cancel()

    def run(self) -> None:
        try:
            service = AugmenterService(self.engine)
            service.config = self.config
            service.scan_result = self.scan_result
            result = service.run(
                self.output_root,
                self.mode,
                self.progress.emit,
                self.cancel_token,
            )
            self.succeeded.emit(result)
        except TaskCancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
