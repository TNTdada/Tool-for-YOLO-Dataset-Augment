from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtCore import QStandardPaths, Qt, QTimer, QUrl
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_dataset_augmenter.app.services import AugmenterService
from yolo_dataset_augmenter.core.config_io import load_config, save_config
from yolo_dataset_augmenter.core.models import (
    AugmentationConfig,
    DatasetScanResult,
    ProcessingMode,
    TaskProgress,
    TaskResult,
)
from yolo_dataset_augmenter.ui.i18n import TEXTS, ui_text, ui_tooltip
from yolo_dataset_augmenter.ui.workers import ScanWorker, TaskWorker


def ndarray_to_pixmap(image) -> QPixmap:
    height, width, channels = image.shape
    bytes_per_line = channels * width
    qimage = QImage(
        image.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qimage)


def _int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def _float_spin(
    minimum: float,
    maximum: float,
    value: float,
    step: float = 0.1,
    decimals: int = 2,
) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    widget.setValue(value)
    return widget


class MainWindow(QMainWindow):
    def __init__(self, settings_dir: Path | None = None) -> None:
        super().__init__()
        self.service = AugmenterService()
        self._settings_dir_override = settings_dir
        self._scan_worker: ScanWorker | None = None
        self._task_worker: TaskWorker | None = None
        self._busy = False
        self.language = "zh"
        self._sidebar_expanded = False
        self._log_expanded = False
        self._last_progress: TaskProgress | None = None
        self._active_mode: ProcessingMode | None = None
        self._has_run_request = False
        self._result_state: tuple[str, str] | None = None
        self.last_result: TaskResult | None = None

        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("选择包含 images 和 labels 的目录")
        self.output_path = QLineEdit(str(Path.cwd() / "outputs"))

        config = self.service.config
        self.target_size = _int_spin(1, 4096, config.target_size)
        self.augment_per_image = _int_spin(0, 100, config.augment_per_image)
        self.max_augs_per_image = _int_spin(1, 7, config.max_augs_per_image)
        self.random_seed = _int_spin(
            -1,
            2_147_483_647,
            -1 if config.random_seed is None else config.random_seed,
        )
        self.random_seed.setSpecialValueText("随机")

        self.brightness_enabled = QCheckBox("启用")
        self.brightness_min = _float_spin(0.0, 5.0, config.brightness_factor[0])
        self.brightness_max = _float_spin(0.0, 5.0, config.brightness_factor[1])
        self.noise_enabled = QCheckBox("启用")
        self.noise_std = _float_spin(0.0, 255.0, config.noise_std, 1.0)
        self.occlusion_enabled = QCheckBox("启用")
        self.occlusion_min = _int_spin(1, 4096, config.occlusion_size[0])
        self.occlusion_max = _int_spin(1, 4096, config.occlusion_size[1])
        self.occlusion_count = _int_spin(0, 100, config.occlusion_count)
        self.hflip_enabled = QCheckBox("水平翻转")
        self.vflip_enabled = QCheckBox("垂直翻转")
        self.rotate_enabled = QCheckBox("启用")
        self.rotation_min = _int_spin(-180, 180, config.rotation_range[0])
        self.rotation_max = _int_spin(-180, 180, config.rotation_range[1])
        self.blur_enabled = QCheckBox("启用")
        self.blur_min = _float_spin(0.0, 20.0, config.blur_radius[0], 0.1)
        self.blur_max = _float_spin(0.0, 20.0, config.blur_radius[1], 0.1)

        self.train_ratio = _float_spin(0.0, 1.0, config.train_ratio, 0.05, 3)
        self.val_ratio = _float_spin(0.0, 1.0, config.val_ratio, 0.05, 3)
        self.test_ratio = _float_spin(0.0, 1.0, config.test_ratio, 0.05, 3)
        self.ratio_total = QLabel()

        self.class_table = QTableWidget(0, 2)
        self.class_table.setHorizontalHeaderLabels(["类别 ID", "类别名称"])
        header = self.class_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.class_table.setMinimumHeight(360)
        self.class_table.setMaximumHeight(520)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("全套流程", ProcessingMode.FULL)
        self.mode_combo.addItem("只增强", ProcessingMode.AUGMENT_ONLY)
        self.mode_combo.addItem("只分类包装", ProcessingMode.PACK_ONLY)

        self.preview_image_combo = QComboBox()
        self.preview_image_combo.addItem("随机选择图片", None)

        self.scan_button = QPushButton("扫描数据集")
        self.preview_button = QPushButton("预览样本")
        self.start_button = QPushButton("开始任务")
        self.cancel_button = QPushButton("取消任务")
        self.rerun_button = QPushButton("重新运行")
        self.open_output_button = QPushButton("打开输出目录")
        self.import_button = QPushButton("导入配置")
        self.export_button = QPushButton("导出配置")
        self.reset_button = QPushButton("恢复默认")
        self.sidebar_toggle_button = QPushButton("›")
        self.sidebar_toggle_button.setFixedSize(22, 72)
        self.sidebar_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sidebar_toggle_button.setStyleSheet(
            "QPushButton { color: white; background: #1677d2; border: 1px solid #0f5ea8; "
            "border-radius: 5px; font-size: 20px; padding: 0; }"
            "QPushButton:hover { background: #2b8be0; }"
        )
        self.log_toggle_button = QPushButton("▲ 展开日志")
        self.zh_button = QPushButton("中文")
        self.en_button = QPushButton("English")
        for button in (self.zh_button, self.en_button):
            button.setFlat(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(
                "QPushButton { color: #1677d2; border: none; padding: 0 2px; font-size: 11px; }"
                "QPushButton:hover { text-decoration: underline; }"
            )

        self.scan_summary = QLabel("尚未扫描数据集。")
        self.scan_summary.setWordWrap(True)
        self.status_label = QLabel("空闲：请选择数据集并开始扫描。")
        self.stage_label = QLabel("阶段：空闲")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.original_preview = QLabel("原图预览")
        self.augmented_preview = QLabel("增强预览")
        self.result_summary = QLabel("尚无运行结果。")
        self.result_summary.setWordWrap(True)
        self.result_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(86)

        self._locked_groups: list[QWidget] = []
        self._detail_widgets: list[QWidget] = []
        self._right_sections: list[QWidget] = []
        self._tooltip_targets: list[tuple[QWidget, str]] = []
        self._parameter_labels: dict[str, QWidget] = {}
        self._build()
        self._connect_signals()
        self.apply_config_to_widgets(config)
        self.load_local_state()
        self.set_language(self.language)
        self._apply_sidebar_mode()
        self._apply_log_mode()
        self._update_ratio_state()
        self._refresh_actions()

    @property
    def settings_dir(self) -> Path:
        if self._settings_dir_override is not None:
            return self._settings_dir_override
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
        root = Path(base) if base else Path.home() / ".config"
        if root.name.casefold() == "yolodatasetaugmenter":
            return root
        return root / "YOLODatasetAugmenter"

    def _build(self) -> None:
        self.setWindowTitle("YOLO 数据集增强工具")
        self.resize(1380, 860)
        root = QWidget()
        root_layout = QHBoxLayout(root)

        controls = QWidget()
        controls_layout = QGridLayout(controls)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setRowStretch(4, 1)
        self.controls_layout = controls_layout

        path_group = QGroupBox("路径")
        path_form = QFormLayout(path_group)
        path_form.addRow("数据集", self._path_row(self.dataset_path, self.choose_dataset))
        path_form.addRow("输出根目录", self._path_row(self.output_path, self.choose_output))
        controls_layout.addWidget(path_group, 0, 0)

        basic_group = QGroupBox("基础参数")
        basic_form = QFormLayout(basic_group)
        basic_form.addRow("目标尺寸", self.target_size)
        basic_form.addRow("每图增强数量", self.augment_per_image)
        basic_form.addRow("单次最多增强方式", self.max_augs_per_image)
        basic_form.addRow("随机种子", self.random_seed)
        self._register_form_tooltip(basic_form, self.target_size, "target_size")
        self._register_form_tooltip(
            basic_form, self.augment_per_image, "augment_per_image"
        )
        self._register_form_tooltip(
            basic_form, self.max_augs_per_image, "max_operations"
        )
        self._register_form_tooltip(basic_form, self.random_seed, "random_seed")
        controls_layout.addWidget(basic_group, 1, 0)

        augmentation_group = QGroupBox("增强参数")
        augmentation_form = QFormLayout(augmentation_group)
        brightness_row = self._range_row(
            self.brightness_enabled, self.brightness_min, self.brightness_max
        )
        augmentation_form.addRow("亮度", brightness_row)
        noise_row = self._range_row(self.noise_enabled, self.noise_std)
        augmentation_form.addRow("噪声标准差", noise_row)
        occlusion_size_row = self._range_row(
            self.occlusion_enabled, self.occlusion_min, self.occlusion_max
        )
        augmentation_form.addRow("遮挡尺寸", occlusion_size_row)
        augmentation_form.addRow("遮挡次数", self.occlusion_count)
        flip_row = QWidget()
        flip_layout = QHBoxLayout(flip_row)
        flip_layout.setContentsMargins(0, 0, 0, 0)
        flip_layout.addWidget(self.hflip_enabled)
        flip_layout.addWidget(self.vflip_enabled)
        augmentation_form.addRow("翻转", flip_row)
        rotation_row = self._range_row(
            self.rotate_enabled, self.rotation_min, self.rotation_max
        )
        augmentation_form.addRow("旋转角度", rotation_row)
        blur_row = self._range_row(
            self.blur_enabled, self.blur_min, self.blur_max
        )
        augmentation_form.addRow("模糊半径", blur_row)
        self._register_form_tooltip(
            augmentation_form,
            brightness_row,
            "brightness",
            self.brightness_enabled,
            self.brightness_min,
            self.brightness_max,
        )
        self._register_form_tooltip(
            augmentation_form,
            noise_row,
            "noise_std",
            self.noise_enabled,
            self.noise_std,
        )
        self._register_form_tooltip(
            augmentation_form,
            occlusion_size_row,
            "occlusion_size",
            self.occlusion_enabled,
            self.occlusion_min,
            self.occlusion_max,
        )
        self._register_form_tooltip(
            augmentation_form,
            self.occlusion_count,
            "occlusion_count",
        )
        self._register_form_tooltip(
            augmentation_form,
            flip_row,
            "flip",
            self.hflip_enabled,
            self.vflip_enabled,
        )
        self._register_form_tooltip(
            augmentation_form,
            rotation_row,
            "rotation",
            self.rotate_enabled,
            self.rotation_min,
            self.rotation_max,
        )
        self._register_form_tooltip(
            augmentation_form,
            blur_row,
            "blur_radius",
            self.blur_enabled,
            self.blur_min,
            self.blur_max,
        )
        controls_layout.addWidget(augmentation_group, 2, 0)

        split_group = QGroupBox("数据集划分")
        split_form = QFormLayout(split_group)
        split_form.addRow("Train", self.train_ratio)
        split_form.addRow("Val", self.val_ratio)
        split_form.addRow("Test", self.test_ratio)
        split_form.addRow("比例总和", self.ratio_total)
        self._register_form_tooltip(split_form, self.train_ratio, "train_ratio")
        self._register_form_tooltip(split_form, self.val_ratio, "val_ratio")
        self._register_form_tooltip(split_form, self.test_ratio, "test_ratio")
        self._register_form_tooltip(split_form, self.ratio_total, "ratio_total")
        controls_layout.addWidget(split_group, 3, 0)

        class_group = QGroupBox("类别名称映射")
        class_layout = QVBoxLayout(class_group)
        class_layout.addWidget(self.class_table)
        self._register_tooltip("class_mapping", class_group, self.class_table)
        controls_layout.addWidget(class_group, 0, 1, 3, 1)

        task_group = QGroupBox("任务与配置")
        task_layout = QGridLayout(task_group)
        mode_label = QLabel("执行模式")
        task_layout.addWidget(mode_label, 0, 0)
        task_layout.addWidget(self.mode_combo, 0, 1)
        task_layout.addWidget(self.import_button, 1, 0)
        task_layout.addWidget(self.export_button, 1, 1)
        task_layout.addWidget(self.reset_button, 2, 0, 1, 2)
        task_layout.addWidget(self.scan_button, 3, 0)
        task_layout.addWidget(self.preview_button, 3, 1)
        task_layout.addWidget(self.start_button, 4, 0)
        task_layout.addWidget(self.cancel_button, 4, 1)
        task_layout.addWidget(self.rerun_button, 5, 0)
        task_layout.addWidget(self.open_output_button, 5, 1)
        task_layout.setColumnStretch(0, 1)
        task_layout.setColumnStretch(1, 1)
        self._register_tooltip("processing_mode", mode_label, self.mode_combo)
        for button, tooltip_key in (
            (self.import_button, "import_config"),
            (self.export_button, "export_config"),
            (self.reset_button, "reset_config"),
            (self.scan_button, "scan_dataset"),
            (self.preview_button, "preview_sample"),
            (self.start_button, "start_task"),
            (self.cancel_button, "cancel_task"),
            (self.rerun_button, "rerun_task"),
            (self.open_output_button, "open_output"),
        ):
            self._register_tooltip(tooltip_key, button)
        controls_layout.addWidget(task_group, 3, 0)
        self.task_group = task_group
        self.class_group = class_group
        self.split_group = split_group

        for widget in (
            self.target_size,
            self.augment_per_image,
            self.max_augs_per_image,
            self.random_seed,
            self.brightness_min,
            self.brightness_max,
            self.noise_std,
            self.occlusion_min,
            self.occlusion_max,
            self.occlusion_count,
            self.rotation_min,
            self.rotation_max,
            self.blur_min,
            self.blur_max,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
        ):
            widget.setMaximumWidth(105)

        self._detail_widgets = [
            self.max_augs_per_image,
            self.random_seed,
            split_group,
            class_group,
            self.brightness_min,
            self.brightness_max,
            self.noise_std,
            self.occlusion_min,
            self.occlusion_max,
            self.occlusion_count,
            self.rotation_min,
            self.rotation_max,
            self.blur_min,
            self.blur_max,
        ]
        for field in (self.max_augs_per_image, self.random_seed):
            label = basic_form.labelForField(field)
            if label is not None:
                self._detail_widgets.append(label)
        occlusion_count_label = augmentation_form.labelForField(self.occlusion_count)
        if occlusion_count_label is not None:
            self._detail_widgets.append(occlusion_count_label)

        self._locked_groups = [path_group, basic_group, augmentation_group, split_group, class_group]
        self._locked_groups.extend([self.mode_combo, self.import_button, self.export_button, self.reset_button])

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll = scroll

        sidebar_placeholder = QWidget()
        sidebar_placeholder.setFixedWidth(318)
        self.sidebar_placeholder = sidebar_placeholder

        content = QWidget()
        content_layout = QVBoxLayout(content)

        language_row = QHBoxLayout()
        language_row.addStretch(1)
        language_row.addWidget(self.zh_button)
        language_row.addWidget(QLabel("|"))
        language_row.addWidget(self.en_button)
        content_layout.addLayout(language_row)

        status_group = QGroupBox("任务状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.stage_label)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.scan_summary)
        content_layout.addWidget(status_group)

        preview_selector = QWidget()
        preview_selector_layout = QHBoxLayout(preview_selector)
        preview_selector_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_image_label = QLabel("预览图片")
        preview_selector_layout.addWidget(self.preview_image_label)
        preview_selector_layout.addWidget(self.preview_image_combo, 1)
        content_layout.addWidget(preview_selector)

        preview_container = QWidget()
        preview_row = QHBoxLayout(preview_container)
        preview_row.setContentsMargins(0, 0, 0, 0)
        for label in (self.original_preview, self.augmented_preview):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(360, 300)
            label.setStyleSheet("border: 1px solid #777; background: #202124; color: #ddd;")
            preview_row.addWidget(label)
        content_layout.addWidget(preview_container, 1)

        result_group = QGroupBox("运行结果")
        result_layout = QVBoxLayout(result_group)
        result_layout.addWidget(self.result_summary)
        content_layout.addWidget(result_group)

        log_group = QGroupBox()
        log_layout = QVBoxLayout(log_group)
        log_header = QHBoxLayout()
        self.log_title_label = QLabel("运行日志（可选择并复制）")
        log_header.addWidget(self.log_title_label)
        log_header.addStretch(1)
        log_header.addWidget(self.log_toggle_button)
        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log_box)
        content_layout.addWidget(log_group)

        self._right_sections = [status_group, preview_selector, preview_container, result_group]
        self.log_group = log_group

        root_layout.addWidget(sidebar_placeholder)
        root_layout.addWidget(content, 1)
        self.setCentralWidget(root)
        scroll.setParent(root)
        self.sidebar_toggle_button.setParent(root)
        QTimer.singleShot(0, self._position_sidebar)

    def _path_row(self, line_edit: QLineEdit, callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("浏览")
        button.clicked.connect(callback)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return row

    def _range_row(self, *widgets: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return row

    def _register_tooltip(self, key: str, *widgets: QWidget) -> None:
        self._tooltip_targets.extend((widget, key) for widget in widgets)

    def _register_form_tooltip(
        self,
        form: QFormLayout,
        field: QWidget,
        key: str,
        *controls: QWidget,
    ) -> None:
        label = form.labelForField(field)
        if label is not None:
            self._parameter_labels[key] = label
            self._register_tooltip(key, label)
        self._register_tooltip(key, field, *controls)

    def _apply_tooltips(self) -> None:
        for widget, key in self._tooltip_targets:
            widget.setToolTip(ui_tooltip(self.language, key))

        mode_tooltips = {
            ProcessingMode.FULL: "mode_full",
            ProcessingMode.AUGMENT_ONLY: "mode_augment",
            ProcessingMode.PACK_ONLY: "mode_package",
        }
        for index in range(self.mode_combo.count()):
            mode = self.mode_combo.itemData(index)
            if isinstance(mode, ProcessingMode):
                self.mode_combo.setItemData(
                    index,
                    ui_tooltip(self.language, mode_tooltips[mode]),
                    Qt.ItemDataRole.ToolTipRole,
                )

    def set_language(self, language: str) -> None:
        self.language = "en" if language == "en" else "zh"
        for widget in self.findChildren(QPushButton):
            widget.setText(self._translated_existing(widget.text()))
        for widget in self.findChildren(QCheckBox):
            widget.setText(self._translated_existing(widget.text()))
        for widget in self.findChildren(QLabel):
            widget.setText(self._translated_existing(widget.text()))
        for group in self.findChildren(QGroupBox):
            group.setTitle(self._translated_existing(group.title()))

        current_mode = self._selected_mode()
        mode_labels = (
            ("全套流程", ProcessingMode.FULL),
            ("只增强", ProcessingMode.AUGMENT_ONLY),
            ("只分类包装", ProcessingMode.PACK_ONLY),
        )
        self.mode_combo.clear()
        for text, mode in mode_labels:
            self.mode_combo.addItem(ui_text(self.language, text), mode)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(current_mode))
        self._apply_tooltips()
        self.class_table.setHorizontalHeaderLabels(
            ["类别 ID", "类别名称"]
            if self.language == "zh"
            else ["Class ID", "Class Name"]
        )
        self.dataset_path.setPlaceholderText(
            ui_text(self.language, "选择包含 images 和 labels 的目录")
        )
        self.random_seed.setSpecialValueText(ui_text(self.language, "随机"))
        if self.preview_image_combo.count() > 0:
            self.preview_image_combo.setItemText(
                0, ui_text(self.language, "随机选择图片")
            )
        self.setWindowTitle(ui_text(self.language, "YOLO 数据集增强工具"))
        self.zh_button.setStyleSheet(self._language_button_style(self.language == "zh"))
        self.en_button.setStyleSheet(self._language_button_style(self.language == "en"))
        self._apply_sidebar_mode()
        self._apply_log_mode()
        self._update_ratio_state()
        if self.service.scan_result is not None:
            self._render_scan_summary(self.service.scan_result)
        if self.last_result is not None:
            self._render_task_result(self.last_result)
        elif self._result_state is not None:
            self._render_result_state()
        if self._last_progress is not None:
            self._render_progress(self._last_progress)

    def _translated_existing(self, text: str) -> str:
        for chinese, english in TEXTS.items():
            if text in {chinese, english}:
                return ui_text(self.language, chinese)
        return text

    def _language_button_style(self, active: bool) -> str:
        weight = "font-weight: 600; text-decoration: underline;" if active else ""
        return (
            "QPushButton { color: #1677d2; border: none; padding: 0 2px; "
            f"font-size: 11px; {weight} }}"
            "QPushButton:hover { text-decoration: underline; }"
        )

    def _toggle_sidebar(self) -> None:
        self._sidebar_expanded = not self._sidebar_expanded
        self._apply_sidebar_mode()

    def _apply_sidebar_mode(self) -> None:
        for widget in self._detail_widgets:
            widget.setVisible(self._sidebar_expanded)
        self.controls_layout.addWidget(
            self.task_group,
            3,
            1 if self._sidebar_expanded else 0,
        )
        self.sidebar_toggle_button.setText("‹" if self._sidebar_expanded else "›")
        self.sidebar_toggle_button.setToolTip(
            ui_text(
                self.language,
                "收起详细参数" if self._sidebar_expanded else "展开详细参数",
            )
        )
        self._position_sidebar()

    def _position_sidebar(self) -> None:
        if not hasattr(self, "sidebar_scroll"):
            return
        central = self.centralWidget()
        if central is None:
            return
        rect = self.sidebar_placeholder.geometry()
        width = 880 if self._sidebar_expanded else rect.width()
        available = max(rect.width(), central.width() - rect.x() - 12)
        sidebar_width = min(width, available)
        self.sidebar_scroll.setGeometry(
            rect.x(),
            rect.y(),
            sidebar_width,
            rect.height(),
        )
        handle_x = rect.x() + sidebar_width - self.sidebar_toggle_button.width() // 2
        handle_y = rect.y() + max(
            0,
            (rect.height() - self.sidebar_toggle_button.height()) // 2,
        )
        self.sidebar_toggle_button.move(handle_x, handle_y)
        self.sidebar_scroll.raise_()
        self.sidebar_toggle_button.raise_()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._position_sidebar()

    def _toggle_log_panel(self) -> None:
        self._log_expanded = not self._log_expanded
        self._apply_log_mode()

    def _apply_log_mode(self) -> None:
        for section in self._right_sections:
            section.setVisible(not self._log_expanded)
        self.log_toggle_button.setText(
            ui_text(
                self.language,
                "▼ 收起日志" if self._log_expanded else "▲ 展开日志",
            )
        )
        if self._log_expanded:
            self.log_box.setMinimumHeight(0)
            self.log_box.setMaximumHeight(16_777_215)
            self.log_box.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.log_group.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
        else:
            self.log_box.setFixedHeight(86)
            self.log_group.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Maximum,
            )

    def _connect_signals(self) -> None:
        self.scan_button.clicked.connect(self.scan_dataset)
        self.preview_button.clicked.connect(self.preview_sample)
        self.start_button.clicked.connect(self.start_task)
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.rerun_button.clicked.connect(self.rerun_task)
        self.open_output_button.clicked.connect(self.open_output_dir)
        self.import_button.clicked.connect(self.import_config)
        self.export_button.clicked.connect(self.export_config)
        self.reset_button.clicked.connect(self.reset_config)
        self.sidebar_toggle_button.clicked.connect(self._toggle_sidebar)
        self.log_toggle_button.clicked.connect(self._toggle_log_panel)
        self.zh_button.clicked.connect(lambda: self.set_language("zh"))
        self.en_button.clicked.connect(lambda: self.set_language("en"))
        for ratio in (self.train_ratio, self.val_ratio, self.test_ratio):
            ratio.valueChanged.connect(self._update_ratio_state)
        for checkbox, widgets in (
            (self.brightness_enabled, (self.brightness_min, self.brightness_max)),
            (self.noise_enabled, (self.noise_std,)),
            (self.occlusion_enabled, (self.occlusion_min, self.occlusion_max, self.occlusion_count)),
            (self.rotate_enabled, (self.rotation_min, self.rotation_max)),
            (self.blur_enabled, (self.blur_min, self.blur_max)),
        ):
            checkbox.toggled.connect(
                lambda checked, controlled=widgets: [
                    widget.setEnabled(checked) for widget in controlled
                ]
            )

    def log(self, message: str) -> None:
        self.log_box.append(message)

    def _update_ratio_state(self) -> None:
        total = self.train_ratio.value() + self.val_ratio.value() + self.test_ratio.value()
        valid = abs(total - 1.0) <= 0.001
        invalid_hint = "（必须为 1.000）" if self.language == "zh" else " (must equal 1.000)"
        self.ratio_total.setText(f"{total:.3f}" + (" ✓" if valid else invalid_hint))
        self.ratio_total.setStyleSheet("color: #188038;" if valid else "color: #d93025;")

    def apply_config_to_widgets(self, config: AugmentationConfig) -> None:
        self.target_size.setValue(config.target_size)
        self.augment_per_image.setValue(config.augment_per_image)
        self.max_augs_per_image.setValue(config.max_augs_per_image)
        self.random_seed.setValue(-1 if config.random_seed is None else config.random_seed)
        self.brightness_enabled.setChecked(config.aug_brightness_enabled)
        self.brightness_min.setValue(config.brightness_factor[0])
        self.brightness_max.setValue(config.brightness_factor[1])
        self.noise_enabled.setChecked(config.aug_noise_enabled)
        self.noise_std.setValue(config.noise_std)
        self.occlusion_enabled.setChecked(config.aug_occlusion_enabled)
        self.occlusion_min.setValue(config.occlusion_size[0])
        self.occlusion_max.setValue(config.occlusion_size[1])
        self.occlusion_count.setValue(config.occlusion_count)
        self.hflip_enabled.setChecked(config.aug_hflip_enabled)
        self.vflip_enabled.setChecked(config.aug_vflip_enabled)
        self.rotate_enabled.setChecked(config.aug_rotate_enabled)
        self.rotation_min.setValue(config.rotation_range[0])
        self.rotation_max.setValue(config.rotation_range[1])
        self.blur_enabled.setChecked(config.aug_blur_enabled)
        self.blur_min.setValue(config.blur_radius[0])
        self.blur_max.setValue(config.blur_radius[1])
        self.train_ratio.setValue(config.train_ratio)
        self.val_ratio.setValue(config.val_ratio)
        self.test_ratio.setValue(config.test_ratio)
        self._populate_class_table(config.class_names)
        self._update_ratio_state()

    def _populate_class_table(self, class_names: dict[int, str]) -> None:
        self.class_table.setRowCount(0)
        for row, class_id in enumerate(sorted(class_names)):
            self.class_table.insertRow(row)
            id_item = QTableWidgetItem(str(class_id))
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.class_table.setItem(row, 0, id_item)
            self.class_table.setItem(row, 1, QTableWidgetItem(class_names[class_id]))

    def _populate_preview_images(self, result: DatasetScanResult) -> None:
        current = self.preview_image_combo.currentData()
        self.preview_image_combo.clear()
        self.preview_image_combo.addItem(
            ui_text(self.language, "随机选择图片"),
            None,
        )
        for image_file, _ in result.valid_pairs:
            self.preview_image_combo.addItem(image_file, image_file)
        if isinstance(current, str):
            index = self.preview_image_combo.findData(current)
            if index >= 0:
                self.preview_image_combo.setCurrentIndex(index)

    def _render_scan_summary(self, result: DatasetScanResult) -> None:
        if self.language == "en":
            text = (
                f"Valid samples: {result.valid_count} | Rejected images: {result.invalid_count} | "
                f"Classes: {result.classes} | Issues: {result.warning_count}"
            )
        else:
            text = (
                f"有效样本：{result.valid_count} ｜ 拒绝图片：{result.invalid_count} ｜ "
                f"类别：{result.classes} ｜ 问题记录：{result.warning_count}"
            )
        self.scan_summary.setText(text)

    def config_from_widgets(self) -> AugmentationConfig:
        class_names: dict[int, str] = {}
        for row in range(self.class_table.rowCount()):
            id_item = self.class_table.item(row, 0)
            name_item = self.class_table.item(row, 1)
            if id_item is None:
                continue
            class_id = int(id_item.text())
            class_name = name_item.text().strip() if name_item is not None else ""
            if not class_name:
                if self.language == "en":
                    raise ValueError(f"The name for class {class_id} cannot be empty.")
                raise ValueError(f"类别 {class_id} 的名称不能为空。")
            class_names[class_id] = class_name

        seed_value = self.random_seed.value()
        config = AugmentationConfig(
            target_size=self.target_size.value(),
            augment_per_image=self.augment_per_image.value(),
            max_augs_per_image=self.max_augs_per_image.value(),
            train_ratio=self.train_ratio.value(),
            val_ratio=self.val_ratio.value(),
            test_ratio=self.test_ratio.value(),
            class_names=class_names,
            aug_brightness_enabled=self.brightness_enabled.isChecked(),
            brightness_factor=(self.brightness_min.value(), self.brightness_max.value()),
            aug_noise_enabled=self.noise_enabled.isChecked(),
            noise_std=self.noise_std.value(),
            aug_occlusion_enabled=self.occlusion_enabled.isChecked(),
            occlusion_size=(self.occlusion_min.value(), self.occlusion_max.value()),
            occlusion_count=self.occlusion_count.value(),
            aug_hflip_enabled=self.hflip_enabled.isChecked(),
            aug_vflip_enabled=self.vflip_enabled.isChecked(),
            aug_rotate_enabled=self.rotate_enabled.isChecked(),
            rotation_range=(self.rotation_min.value(), self.rotation_max.value()),
            aug_blur_enabled=self.blur_enabled.isChecked(),
            blur_radius=(self.blur_min.value(), self.blur_max.value()),
            random_seed=None if seed_value < 0 else seed_value,
        )
        errors = config.validate()
        if errors:
            raise ValueError("；".join(errors))
        return config

    def sync_config(self) -> AugmentationConfig:
        config = self.config_from_widgets()
        self.service.config = config
        return config

    def _config_snapshot(self) -> AugmentationConfig:
        return AugmentationConfig.from_mapping(self.sync_config().to_legacy_dict())

    def choose_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, ui_text(self.language, "选择 YOLO 数据集目录")
        )
        if path:
            self.dataset_path.setText(path)

    def choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, ui_text(self.language, "选择输出根目录")
        )
        if path:
            self.output_path.setText(path)

    def load_config_from_path(self, path: Path) -> None:
        config = load_config(path)
        self.service.config = config
        self.apply_config_to_widgets(config)
        self.log(f"已载入配置：{path}")

    def export_config_to_path(self, path: Path) -> None:
        save_config(path, self.sync_config())
        self.log(f"已导出配置：{path}")

    def import_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入配置",
            str(self.settings_dir),
            "JSON 配置 (*.json);;所有文件 (*)",
        )
        if not path:
            return
        try:
            self.load_config_from_path(Path(path))
            self.save_local_state()
            self.status_label.setText(ui_text(self.language, "配置导入成功。"))
        except Exception as exc:
            self.show_error(ui_text(self.language, "配置导入失败"), str(exc))

    def export_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出配置",
            str(self.settings_dir / "augment_config.json"),
            "JSON 配置 (*.json)",
        )
        if not path:
            return
        try:
            target = Path(path)
            if target.suffix.lower() != ".json":
                target = target.with_suffix(".json")
            self.export_config_to_path(target)
            self.status_label.setText(ui_text(self.language, "配置导出成功。"))
        except Exception as exc:
            self.show_error(ui_text(self.language, "配置导出失败"), str(exc))

    def reset_config(self) -> None:
        self.service.config = AugmentationConfig()
        self.apply_config_to_widgets(self.service.config)
        self.status_label.setText(ui_text(self.language, "已恢复默认配置。"))
        self.log("配置已恢复默认值。")

    def load_local_state(self) -> None:
        config_path = self.settings_dir / "last_config.json"
        state_path = self.settings_dir / "ui_state.json"
        try:
            if config_path.is_file():
                config = load_config(config_path)
                self.service.config = config
                self.apply_config_to_widgets(config)
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(state, dict):
                    self.dataset_path.setText(str(state.get("dataset_path", "")))
                    self.output_path.setText(
                        str(state.get("output_path", self.output_path.text()))
                    )
                    self.language = "en" if state.get("language") == "en" else "zh"
            if config_path.is_file() or state_path.is_file():
                self.log(f"已恢复用户配置目录：{self.settings_dir}")
        except Exception as exc:
            self.log(f"恢复最近配置失败，将使用默认值：{exc}")

    def save_local_state(self) -> None:
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        config = self.config_from_widgets()
        save_config(self.settings_dir / "last_config.json", config)
        state = {
            "dataset_path": self.dataset_path.text().strip(),
            "output_path": self.output_path.text().strip(),
            "language": self.language,
        }
        (self.settings_dir / "ui_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def show_error(self, title: str, message: str, details: str = "") -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(message)
        if details:
            box.setDetailedText(details)
        box.exec()

    def _dataset_dir(self) -> Path:
        text = self.dataset_path.text().strip()
        if not text:
            raise ValueError(
                "Please choose a dataset folder first."
                if self.language == "en"
                else "请先选择数据集目录。"
            )
        path = Path(text)
        if not path.is_dir():
            raise ValueError(
                f"Dataset folder does not exist: {path}"
                if self.language == "en"
                else f"数据集目录不存在：{path}"
            )
        return path

    def _current_scan(self) -> DatasetScanResult:
        result = self.service.scan_result
        if result is None:
            raise ValueError(
                "Scan the dataset first." if self.language == "en" else "请先扫描数据集。"
            )
        if result.dataset_dir.resolve() != self._dataset_dir().resolve():
            raise ValueError(
                "The dataset path changed. Scan it again."
                if self.language == "en"
                else "数据集路径已变化，请重新扫描。"
            )
        return result

    def scan_dataset(self) -> None:
        if self._busy:
            return
        try:
            dataset_dir = self._dataset_dir()
            config = self._config_snapshot()
        except Exception as exc:
            self.status_label.setText(ui_text(self.language, "路径或配置无效。"))
            self.show_error(ui_text(self.language, "无法开始扫描"), str(exc))
            return

        self.service.scan_result = None
        self._has_run_request = False
        self.last_result = None
        self._result_state = None
        self.result_summary.setText(ui_text(self.language, "尚无运行结果。"))
        self.preview_image_combo.clear()
        self.preview_image_combo.addItem(ui_text(self.language, "随机选择图片"), None)
        worker = ScanWorker(dataset_dir, config)
        self._scan_worker = worker
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(self._scan_succeeded)
        worker.failed.connect(self._scan_failed)
        worker.cancelled.connect(self._scan_cancelled)
        worker.finished.connect(lambda: self._scan_finished(worker))
        self._set_busy(True)
        self._active_mode = None
        self.status_label.setText(ui_text(self.language, "正在后台扫描数据集……"))
        self.log(f"开始扫描：{dataset_dir}")
        worker.start()

    def _scan_succeeded(self, result: DatasetScanResult) -> None:
        if self._scan_worker is not None:
            self.service.config = self._scan_worker.config
        self.service.scan_result = result
        self._populate_class_table(self.service.config.class_names)
        self._populate_preview_images(result)
        self._render_scan_summary(result)
        self.status_label.setText(
            ui_text(self.language, "扫描完成，可以预览或执行任务。")
        )
        self._log_issues(result)
        try:
            self.save_local_state()
        except Exception as exc:
            self.log(f"保存最近配置失败：{exc}")

    def _scan_failed(self, message: str, details: str) -> None:
        self.service.scan_result = None
        self.status_label.setText(
            ui_text(self.language, "扫描失败。请检查路径和数据集结构。")
        )
        self.log(f"扫描失败：{message}")
        self.show_error(ui_text(self.language, "扫描失败"), message, details)

    def _scan_cancelled(self, message: str) -> None:
        self.service.scan_result = None
        self.status_label.setText(ui_text(self.language, "扫描已取消。"))
        self.log(message)

    def _scan_finished(self, worker: ScanWorker) -> None:
        if self._scan_worker is worker:
            self._scan_worker = None
        worker.deleteLater()
        self._set_busy(False)

    def preview_sample(self) -> None:
        try:
            self._current_scan()
            self.sync_config()
            selected = self.preview_image_combo.currentData()
            image_file = selected if isinstance(selected, str) else None
            original, augmented = self.service.preview(
                image_file=image_file,
                run_augment=True,
                randomize=True,
            )
            self.original_preview.setPixmap(
                ndarray_to_pixmap(original).scaled(
                    self.original_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            if augmented is not None:
                self.augmented_preview.setPixmap(
                    ndarray_to_pixmap(augmented).scaled(
                        self.augmented_preview.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            self.status_label.setText(ui_text(self.language, "预览已刷新。"))
            self.log("已生成原图与增强预览。")
        except Exception as exc:
            self.status_label.setText(ui_text(self.language, "预览失败。"))
            self.show_error(ui_text(self.language, "预览失败"), str(exc))

    def _selected_mode(self) -> ProcessingMode:
        mode = self.mode_combo.currentData()
        if isinstance(mode, ProcessingMode):
            return mode
        return ProcessingMode(str(mode))

    def start_task(self) -> None:
        if self._busy:
            return
        try:
            scan_result = self._current_scan()
            config = self._config_snapshot()
            output_text = self.output_path.text().strip()
            if not output_text:
                raise ValueError(
                    "Choose an output root folder first."
                    if self.language == "en"
                    else "请先选择输出根目录。"
                )
            output_root = Path(output_text)
            if output_root.exists() and not output_root.is_dir():
                raise ValueError(
                    f"The output path is not a folder: {output_root}"
                    if self.language == "en"
                    else f"输出路径不是目录：{output_root}"
                )
            mode = self._selected_mode()
        except Exception as exc:
            self.status_label.setText(ui_text(self.language, "任务参数无效。"))
            self.show_error(ui_text(self.language, "无法开始任务"), str(exc))
            return

        worker = TaskWorker(scan_result, output_root, config, mode)
        self._has_run_request = True
        self._task_worker = worker
        self._active_mode = mode
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(self._task_succeeded)
        worker.failed.connect(self._task_failed)
        worker.cancelled.connect(self._task_cancelled)
        worker.finished.connect(lambda: self._task_finished(worker))
        self._set_busy(True)
        self.progress_bar.setValue(0)
        self._result_state = ("running", "")
        self.status_label.setText(ui_text(self.language, "任务正在后台执行……"))
        self.result_summary.setText(
            ui_text(self.language, "任务执行中，完成后将在此显示结果。")
        )
        self.log(f"开始任务：模式={mode.value}，输出根目录={output_root}")
        worker.start()

    def rerun_task(self) -> None:
        self.start_task()

    def cancel_operation(self) -> None:
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.cancel()
            self.status_label.setText(ui_text(self.language, "正在取消扫描……"))
            self.log("已请求取消扫描。")
        if self._task_worker is not None and self._task_worker.isRunning():
            self._task_worker.cancel()
            self.status_label.setText(ui_text(self.language, "正在取消任务……"))
            self.log("已请求取消任务。")

    def _task_succeeded(self, result: TaskResult) -> None:
        self.last_result = result
        self._result_state = ("success", "")
        self.service.config = (
            self._task_worker.config if self._task_worker is not None else self.service.config
        )
        self._render_task_result(result)
        self.status_label.setText(
            ui_text(self.language, "任务完成。可以打开输出目录或重新运行。")
        )
        self.log(f"任务完成：{result.output_dir}")
        self._log_issues(result)
        try:
            self.save_local_state()
        except Exception as exc:
            self.log(f"保存最近配置失败：{exc}")

    def _task_failed(self, message: str, details: str) -> None:
        self._result_state = ("failed", message)
        self.status_label.setText(ui_text(self.language, "任务执行失败。"))
        self._render_result_state()
        self.log(f"任务失败：{message}")
        self.show_error(ui_text(self.language, "任务失败"), message, details)

    def _task_cancelled(self, message: str) -> None:
        self._result_state = ("cancelled", message)
        self.status_label.setText(
            ui_text(self.language, "任务已取消；输出目录可能包含未完成文件。")
        )
        self._render_result_state()
        self.log(message)

    def _task_finished(self, worker: TaskWorker) -> None:
        if self._task_worker is worker:
            self._task_worker = None
        worker.deleteLater()
        self._active_mode = None
        self._set_busy(False)

    def _on_progress(self, event: TaskProgress) -> None:
        self._last_progress = event
        self._render_progress(event)
        self.progress_bar.setValue(self._overall_percent(event))
        if event.message:
            self.log(f"[{event.stage}] {event.message}")

    def _render_progress(self, event: TaskProgress) -> None:
        if self.language == "en":
            stage_names = {
                "scan": "Scan",
                "augment": "Augment",
                "pack": "Package",
                "complete": "Complete",
            }
            prefix = "Stage"
        else:
            stage_names = {
                "scan": "扫描",
                "augment": "增强",
                "pack": "打包",
                "complete": "完成",
            }
            prefix = "阶段"
        self.stage_label.setText(
            f"{prefix}：{stage_names.get(event.stage, event.stage)} ｜ {event.message}"
        )

    def _render_task_result(self, result: TaskResult) -> None:
        separator = ", " if self.language == "en" else "，"
        split_text = separator.join(
            f"{name}={count}" for name, count in result.split_counts.items()
        )
        if self.language == "en":
            split_text = split_text or "This mode does not create dataset splits"
            text = (
                f"Status: Success\nOutput: {result.output_dir}\n"
                f"Split counts: {split_text}\nScan issues: {len(result.issues)}\n"
                f"Config snapshot: {result.output_dir / 'run_config.json'}\n"
                f"Run summary: {result.output_dir / 'run_summary.json'}"
            )
        else:
            split_text = split_text or "当前模式不生成划分目录"
            text = (
                f"状态：成功\n输出目录：{result.output_dir}\n"
                f"划分数量：{split_text}\n扫描问题：{len(result.issues)}\n"
                f"配置快照：{result.output_dir / 'run_config.json'}\n"
                f"运行摘要：{result.output_dir / 'run_summary.json'}"
            )
        self.result_summary.setText(text)

    def _render_result_state(self) -> None:
        if self._result_state is None:
            return
        state, message = self._result_state
        if state == "running":
            self.result_summary.setText(
                ui_text(self.language, "任务执行中，完成后将在此显示结果。")
            )
        elif state == "failed":
            prefix = "Status: Failed\nReason: " if self.language == "en" else "状态：失败\n原因："
            self.result_summary.setText(prefix + message)
        elif state == "cancelled":
            prefix = "Status: Cancelled\n" if self.language == "en" else "状态：已取消\n"
            self.result_summary.setText(prefix + message)

    def _overall_percent(self, event: TaskProgress) -> int:
        if self._active_mode is None:
            return event.percent
        if event.stage == "complete":
            return 100
        if self._active_mode is ProcessingMode.FULL:
            starts = {"scan": 0, "pack": 10, "augment": 40}
            weights = {"scan": 10, "pack": 30, "augment": 60}
        elif self._active_mode is ProcessingMode.AUGMENT_ONLY:
            starts = {"scan": 0, "augment": 10}
            weights = {"scan": 10, "augment": 90}
        else:
            starts = {"scan": 0, "pack": 20}
            weights = {"scan": 20, "pack": 80}
        start = starts.get(event.stage, 0)
        weight = weights.get(event.stage, 0)
        return min(100, start + round(event.percent * weight / 100))

    def _log_issues(self, result: DatasetScanResult | TaskResult) -> None:
        if not result.issues:
            return
        for issue in result.issues[:50]:
            line = f"第 {issue.line_number} 行" if issue.line_number is not None else ""
            self.log(f"[数据问题:{issue.kind.value}] {issue.path} {line} {issue.message}")
        if len(result.issues) > 50:
            self.log(f"另有 {len(result.issues) - 50} 条问题未在日志区域展开。")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in self._locked_groups:
            widget.setEnabled(not busy)
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        has_scan = self.service.scan_result is not None
        self.scan_button.setEnabled(not self._busy)
        self.preview_button.setEnabled(not self._busy and has_scan)
        self.start_button.setEnabled(not self._busy and has_scan)
        self.cancel_button.setEnabled(self._busy)
        self.rerun_button.setEnabled(not self._busy and has_scan and self._has_run_request)
        self.open_output_button.setEnabled(not self._busy and self.last_result is not None)

    def open_output_dir(self) -> None:
        if self.last_result is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_result.output_dir)))

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if a0 is None:
            return
        workers = [worker for worker in (self._scan_worker, self._task_worker) if worker is not None]
        for worker in workers:
            worker.cancel()
        if any(not worker.wait(3000) for worker in workers):
            self.show_error(
                ui_text(self.language, "暂时无法退出"),
                "后台任务尚未安全结束，请稍后重试。"
                if self.language == "zh"
                else "The background task has not stopped safely. Try again shortly.",
            )
            a0.ignore()
            return
        try:
            self.save_local_state()
        except Exception as exc:
            self.log(f"退出时保存配置失败：{exc}")
        a0.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("TNTdada")
    app.setApplicationName("YOLODatasetAugmenter")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
