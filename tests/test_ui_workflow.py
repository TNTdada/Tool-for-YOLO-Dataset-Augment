from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from yolo_dataset_augmenter.core.models import ProcessingMode
from yolo_dataset_augmenter.ui.main_window import MainWindow


_QAPP: QApplication | None = None


def _app() -> QApplication:
    global _QAPP
    if _QAPP is None:
        instance = QApplication.instance()
        _QAPP = instance if isinstance(instance, QApplication) else QApplication([])
    return _QAPP


def _wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _app().processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out while waiting for the UI workflow.")


def _dataset(root: Path, count: int = 3) -> Path:
    dataset = root / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    for index in range(count):
        Image.new("RGB", (24, 16), (20, index * 20, 100)).save(
            dataset / "images" / f"image_{index}.jpg"
        )
        (dataset / "labels" / f"image_{index}.txt").write_text(
            "4 0.5 0.5 0.4 0.4",
            encoding="utf-8",
        )
    return dataset


class QuietMainWindow(MainWindow):
    def __init__(self, settings_dir: Path) -> None:
        self.errors: list[tuple[str, str, str]] = []
        super().__init__(settings_dir)

    def show_error(self, title: str, message: str, details: str = "") -> None:
        self.errors.append((title, message, details))


def test_complete_ui_workflow_and_local_config_restore(tmp_path):
    _app()
    dataset = _dataset(tmp_path)
    settings_dir = tmp_path / "settings"
    output_dir = tmp_path / "outputs"
    window = QuietMainWindow(settings_dir)
    window.show()
    _app().processEvents()
    assert window.sidebar_scroll.width() == 318
    assert not window.class_table.isVisible()
    assert window.log_box.height() == 86
    assert abs(
        window.sidebar_toggle_button.geometry().center().x()
        - window.sidebar_scroll.geometry().right()
    ) <= 2

    window.sidebar_toggle_button.click()
    _app().processEvents()
    assert window.sidebar_scroll.width() == 880
    assert window.class_table.isVisible()

    window.log_toggle_button.click()
    _app().processEvents()
    assert window._log_expanded
    assert not window._right_sections[0].isVisible()
    assert window.log_box.height() > 300
    window.log_toggle_button.click()
    _app().processEvents()
    assert not window._log_expanded
    window.dataset_path.setText(str(dataset))
    window.output_path.setText(str(output_dir))

    window.scan_dataset()
    assert window.cancel_button.isEnabled()
    _wait_until(lambda: not window._busy)

    assert not window.errors
    assert window.service.scan_result is not None
    assert window.service.scan_result.valid_count == 3
    assert window.class_table.rowCount() == 1
    assert window.preview_image_combo.count() == 4
    assert window.preview_button.isEnabled()
    class_name_item = window.class_table.item(0, 1)
    assert class_name_item is not None
    class_name_item.setText("component")
    preview_index = window.preview_image_combo.findData("image_1.jpg")
    window.preview_image_combo.setCurrentIndex(preview_index)
    window.preview_sample()
    assert window.original_preview.pixmap() is not None
    assert window.augmented_preview.pixmap() is not None

    window.en_button.click()
    assert window.language == "en"
    assert window.windowTitle() == "YOLO Dataset Augmenter"
    assert window.scan_button.text() == "Scan Dataset"
    assert window.preview_image_combo.itemText(0) == "Random Image"

    mode_index = window.mode_combo.findData(ProcessingMode.PACK_ONLY)
    window.mode_combo.setCurrentIndex(mode_index)
    window.start_task()
    assert window.cancel_button.isEnabled()
    _wait_until(lambda: not window._busy)

    assert not window.errors
    assert window.last_result is not None
    assert window.progress_bar.value() == 100
    assert "Status: Success" in window.result_summary.text()
    assert window.open_output_button.isEnabled()
    assert (window.last_result.output_dir / "run_config.json").is_file()
    assert (window.last_result.output_dir / "run_summary.json").is_file()

    exported = tmp_path / "exported.json"
    window.target_size.setValue(320)
    window.export_config_to_path(exported)
    window.target_size.setValue(640)
    window.load_config_from_path(exported)
    assert window.target_size.value() == 320

    window.train_ratio.setValue(0.5)
    window.val_ratio.setValue(0.5)
    window.test_ratio.setValue(0.5)
    with pytest.raises(ValueError, match="sum to 1.0"):
        window.config_from_widgets()
    window.train_ratio.setValue(0.7)
    window.val_ratio.setValue(0.2)
    window.test_ratio.setValue(0.1)
    window.close()
    _app().processEvents()

    restored = QuietMainWindow(settings_dir)
    assert restored.dataset_path.text() == str(dataset)
    assert restored.output_path.text() == str(output_dir)
    assert restored.target_size.value() == 320
    assert restored.language == "en"
    assert restored.scan_button.text() == "Scan Dataset"
    restored_class_name = restored.class_table.item(0, 1)
    assert restored_class_name is not None
    assert restored_class_name.text() == "component"
    restored.close()


def test_sidebar_handle_english_layout_and_mapping_scroll_are_independent(tmp_path):
    _app()
    window = QuietMainWindow(tmp_path / "settings")
    window.show()
    assert window.mode_combo.itemText(0) == "全套流程"
    assert "建议" in window.target_size.toolTip()
    assert "640" in window.target_size.toolTip()
    assert (
        window._parameter_labels["target_size"].toolTip()
        == window.target_size.toolTip()
    )
    assert "1.0 表示不变" in window.brightness_min.toolTip()
    full_tooltip = window.mode_combo.itemData(
        window.mode_combo.findData(ProcessingMode.FULL),
        Qt.ItemDataRole.ToolTipRole,
    )
    assert isinstance(full_tooltip, str)
    assert "先把原始样本划分" in full_tooltip
    window.en_button.click()
    _app().processEvents()

    assert window.mode_combo.itemText(0) == "Full Pipeline"
    assert window.mode_combo.itemText(1) == "Augment Only"
    assert window.mode_combo.itemText(2) == "Package Only"
    assert "Recommended" in window.target_size.toolTip()
    assert "consistent or compatible" in window.target_size.toolTip()
    assert (
        window._parameter_labels["target_size"].toolTip()
        == window.target_size.toolTip()
    )
    full_tooltip = window.mode_combo.itemData(
        window.mode_combo.findData(ProcessingMode.FULL),
        Qt.ItemDataRole.ToolTipRole,
    )
    assert isinstance(full_tooltip, str)
    assert "Splits source samples" in full_tooltip
    assert "currently shown in the UI" in window.rerun_button.toolTip()
    compact_buttons = (
        window.import_button,
        window.export_button,
        window.reset_button,
        window.scan_button,
        window.preview_button,
        window.start_button,
        window.cancel_button,
        window.rerun_button,
        window.open_output_button,
    )
    assert all(button.toolTip() for button in compact_buttons)
    assert all(
        button.geometry().right() <= window.task_group.contentsRect().right()
        for button in compact_buttons
    )

    sidebar_bar = window.sidebar_scroll.verticalScrollBar()
    assert sidebar_bar is not None
    sidebar_bar.setValue(sidebar_bar.maximum())
    handle_position = window.sidebar_toggle_button.pos()
    window.sidebar_toggle_button.click()
    _app().processEvents()
    assert window.sidebar_scroll.width() == 880
    assert window.sidebar_toggle_button.pos() != handle_position
    assert abs(
        window.sidebar_toggle_button.geometry().center().x()
        - window.sidebar_scroll.geometry().right()
    ) <= 2
    expanded_sidebar_bar = window.sidebar_scroll.verticalScrollBar()
    assert expanded_sidebar_bar is not None
    assert expanded_sidebar_bar.maximum() == 0
    assert all(
        button.geometry().right() <= window.task_group.contentsRect().right()
        for button in compact_buttons
    )

    window._populate_class_table({index: f"class_{index}" for index in range(100)})
    _app().processEvents()
    table_bar = window.class_table.verticalScrollBar()
    assert table_bar is not None
    assert table_bar.maximum() > 0
    sidebar_bar.setValue(0)
    table_bar.setValue(table_bar.maximum())
    assert sidebar_bar.value() == 0
    window.close()


def test_ui_cancel_restores_controls_and_reports_cancelled_state(tmp_path):
    _app()
    dataset = _dataset(tmp_path, count=20)
    window = QuietMainWindow(tmp_path / "settings")
    window.dataset_path.setText(str(dataset))
    window.output_path.setText(str(tmp_path / "outputs"))
    window.scan_dataset()
    _wait_until(lambda: not window._busy)
    assert window.service.scan_result is not None

    window.augment_per_image.setValue(100)
    window.target_size.setValue(128)
    mode_index = window.mode_combo.findData(ProcessingMode.AUGMENT_ONLY)
    window.mode_combo.setCurrentIndex(mode_index)
    window.start_task()
    window.cancel_operation()
    _wait_until(lambda: not window._busy)

    assert "已取消" in window.status_label.text()
    assert "已取消" in window.result_summary.text()
    assert window.start_button.isEnabled()
    assert window.rerun_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    assert not window.errors
    window.close()
