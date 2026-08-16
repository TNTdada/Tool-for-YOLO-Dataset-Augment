from __future__ import annotations


TEXTS: dict[str, str] = {
    "YOLO 数据集增强工具": "YOLO Dataset Augmenter",
    "路径": "Paths",
    "基础参数": "Basic Settings",
    "增强参数": "Augmentation",
    "数据集划分": "Dataset Split",
    "类别名称映射": "Class Name Mapping",
    "任务与配置": "Task and Configuration",
    "任务状态": "Task Status",
    "运行结果": "Run Result",
    "数据集": "Dataset",
    "输出根目录": "Output Root",
    "目标尺寸": "Target Size",
    "每图增强数量": "Augments / Image",
    "单次最多增强方式": "Max Operations",
    "随机种子": "Random Seed",
    "亮度": "Brightness",
    "噪声标准差": "Noise Std. Dev.",
    "遮挡尺寸": "Occlusion Size",
    "遮挡次数": "Occlusion Count",
    "翻转": "Flip",
    "旋转角度": "Rotation Angle",
    "模糊半径": "Blur Radius",
    "比例总和": "Ratio Total",
    "执行模式": "Processing Mode",
    "启用": "Enabled",
    "水平翻转": "Horizontal Flip",
    "垂直翻转": "Vertical Flip",
    "浏览": "Browse",
    "导入配置": "Import Config",
    "导出配置": "Export Config",
    "恢复默认": "Reset Defaults",
    "扫描数据集": "Scan Dataset",
    "预览样本": "Preview Sample",
    "开始任务": "Start Task",
    "取消任务": "Cancel Task",
    "重新运行": "Run Again",
    "打开输出目录": "Open Output",
    "全套流程": "Full Pipeline",
    "只增强": "Augment Only",
    "只分类包装": "Package Only",
    "尚未扫描数据集。": "No dataset has been scanned.",
    "空闲：请选择数据集并开始扫描。": "Idle: choose a dataset and start scanning.",
    "阶段：空闲": "Stage: Idle",
    "原图预览": "Original Preview",
    "增强预览": "Augmented Preview",
    "尚无运行结果。": "No run result yet.",
    "运行日志（可选择并复制）": "Run Log (selectable and copyable)",
    "预览图片": "Preview Image",
    "随机选择图片": "Random Image",
    "▲ 展开日志": "▲ Expand Log",
    "▼ 收起日志": "▼ Collapse Log",
    "展开详细参数": "Expand detailed settings",
    "收起详细参数": "Collapse detailed settings",
    "选择包含 images 和 labels 的目录": "Choose a folder containing images and labels",
    "随机": "Random",
    "扫描完成，可以预览或执行任务。": "Scan complete. You can preview or run a task.",
    "正在后台扫描数据集……": "Scanning the dataset in the background...",
    "扫描失败。请检查路径和数据集结构。": "Scan failed. Check the path and dataset structure.",
    "扫描已取消。": "Scan cancelled.",
    "预览已刷新。": "Preview refreshed.",
    "预览失败。": "Preview failed.",
    "任务正在后台执行……": "The task is running in the background...",
    "任务完成。可以打开输出目录或重新运行。": "Task complete. Open the output folder or run again.",
    "任务执行失败。": "Task failed.",
    "任务已取消；输出目录可能包含未完成文件。": "Task cancelled; the output folder may contain partial files.",
    "正在取消扫描……": "Cancelling scan...",
    "正在取消任务……": "Cancelling task...",
    "路径或配置无效。": "The path or configuration is invalid.",
    "任务参数无效。": "Task parameters are invalid.",
    "配置导入成功。": "Configuration imported.",
    "配置导出成功。": "Configuration exported.",
    "已恢复默认配置。": "Default configuration restored.",
    "任务执行中，完成后将在此显示结果。": "Task running. Results will appear here when complete.",
    "选择 YOLO 数据集目录": "Choose YOLO Dataset Folder",
    "选择输出根目录": "Choose Output Root Folder",
    "无法开始扫描": "Cannot Start Scan",
    "扫描失败": "Scan Failed",
    "预览失败": "Preview Failed",
    "无法开始任务": "Cannot Start Task",
    "任务失败": "Task Failed",
    "配置导入失败": "Configuration Import Failed",
    "配置导出失败": "Configuration Export Failed",
    "暂时无法退出": "Cannot Exit Yet",
}


TOOLTIPS: dict[str, tuple[str, str]] = {
    "target_size": (
        "<b>作用：</b>图像会按比例缩放并补边为该正方形边长，标注框同步换算。<br>"
        "<b>建议：</b>常用 640；速度或显存优先可用 320–512，小目标可尝试 960–1280。应与训练输入尺寸一致或兼容。",
        "<b>Purpose:</b> Images are resized proportionally and padded to this square size; bounding boxes are adjusted with them.<br>"
        "<b>Recommended:</b> 640 is common. Use 320–512 for speed or lower memory use, and try 960–1280 for small objects. Keep it consistent or compatible with the training input size.",
    ),
    "augment_per_image": (
        "<b>作用：</b>每个有效原样本额外生成的增强副本数，仅影响“只增强”和“全套流程”。设为 0 时不生成随机增强，只保留缩放后的原样本。<br>"
        "<b>建议：</b>先用 1–5；小数据集可尝试 5–10。数值越大，运行时间和磁盘占用越高。",
        "<b>Purpose:</b> Number of additional augmented copies generated for each valid source sample. It affects Augment Only and Full Pipeline. At 0, no random copies are created and only the resized source sample is kept.<br>"
        "<b>Recommended:</b> Start with 1–5; try 5–10 for a small dataset. Larger values increase runtime and disk usage.",
    ),
    "max_operations": (
        "<b>作用：</b>每个增强副本最多随机组合多少种已启用的增强操作；它不是生成图片的数量，实际组合数不会超过已启用操作数。<br>"
        "<b>建议：</b>1–3。组合过多可能产生不符合真实场景的样本。",
        "<b>Purpose:</b> Maximum number of enabled augmentation operations randomly combined in each augmented copy. This is not the number of output images, and it never exceeds the number of enabled operations.<br>"
        "<b>Recommended:</b> 1–3. Combining too many operations can produce unrealistic samples.",
    ),
    "random_seed": (
        "<b>作用：</b>固定整数可复现批处理任务的增强结果和数据集划分；选择“随机”则每次运行不同。预览按钮仍会在每次点击时重新随机。<br>"
        "<b>建议：</b>正式对比实验使用固定值（如 42）；仅在需要生成新变体时使用随机。",
        "<b>Purpose:</b> A fixed integer makes batch augmentation and dataset splitting reproducible. Random produces a different run each time. The Preview button still randomizes on every click.<br>"
        "<b>Recommended:</b> Use a fixed value such as 42 for controlled experiments; use Random only when new variants are desired.",
    ),
    "brightness": (
        "<b>作用：</b>从最小值到最大值随机选择亮度系数；1.0 表示不变，小于 1.0 变暗，大于 1.0 变亮。<br>"
        "<b>建议：</b>常用 0.7–1.3；光照差异较大时可尝试 0.5–1.5，并先检查预览。",
        "<b>Purpose:</b> Randomly selects a brightness factor between the minimum and maximum. 1.0 is unchanged, values below 1.0 darken, and values above 1.0 brighten.<br>"
        "<b>Recommended:</b> 0.7–1.3 is common. Try 0.5–1.5 for wider lighting variation and inspect the preview first.",
    ),
    "noise_std": (
        "<b>作用：</b>高斯噪声的标准差，采用 0–255 像素强度尺度；0 表示不添加噪声。<br>"
        "<b>建议：</b>5–20；超过 30 通常已是较强噪声，应结合真实成像条件判断。",
        "<b>Purpose:</b> Standard deviation of Gaussian noise on the 0–255 pixel-intensity scale. 0 adds no noise.<br>"
        "<b>Recommended:</b> 5–20. Values above 30 are usually strong and should reflect realistic imaging conditions.",
    ),
    "occlusion_size": (
        "<b>作用：</b>随机遮挡矩形宽和高的像素范围，按缩放补边后的目标图像计算。<br>"
        "<b>建议：</b>约为目标边长的 3%–15%；目标尺寸 640 时可从 20–100 开始。",
        "<b>Purpose:</b> Pixel range used for both width and height of random occlusion rectangles, measured after resizing and padding.<br>"
        "<b>Recommended:</b> About 3%–15% of the target side length; for a target size of 640, start around 20–100.",
    ),
    "occlusion_count": (
        "<b>作用：</b>每次执行遮挡增强时放置的随机遮挡块数量；0 表示不放置遮挡块。<br>"
        "<b>建议：</b>1–3，目标密集或目标很小时应适当减少。",
        "<b>Purpose:</b> Number of random blocks placed whenever the occlusion operation is selected. 0 places no blocks.<br>"
        "<b>Recommended:</b> 1–3; use fewer for dense scenes or very small objects.",
    ),
    "flip": (
        "<b>作用：</b>水平翻转和垂直翻转可独立启用，标注框会同步变换。<br>"
        "<b>建议：</b>仅在翻转后类别语义仍成立时启用。水平翻转较常用；垂直翻转通常只适合航拍、显微等方向不敏感场景。",
        "<b>Purpose:</b> Horizontal and vertical flips can be enabled independently; bounding boxes are transformed with the image.<br>"
        "<b>Recommended:</b> Enable a flip only when class semantics remain valid. Horizontal flip is common; vertical flip is usually suitable only for orientation-insensitive data such as aerial or microscopy images.",
    ),
    "rotation": (
        "<b>作用：</b>在最小角度与最大角度之间随机旋转，输出仍使用轴对齐 YOLO 检测框。<br>"
        "<b>建议：</b>普通检测任务使用 ±5°–15°；除非真实场景存在大角度变化，否则避免过大范围。",
        "<b>Purpose:</b> Randomly rotates between the minimum and maximum angles. Output remains in axis-aligned YOLO detection boxes.<br>"
        "<b>Recommended:</b> Use about ±5°–15° for typical detection tasks. Avoid wider ranges unless large rotations occur in real data.",
    ),
    "blur_radius": (
        "<b>作用：</b>高斯模糊半径的随机范围；0 表示不模糊，数值越大图像越模糊。<br>"
        "<b>建议：</b>0.1–1.0；超过 2.0 通常较强，应先通过预览确认目标仍可辨认。",
        "<b>Purpose:</b> Random range for the Gaussian blur radius. 0 means no blur; larger values produce stronger blur.<br>"
        "<b>Recommended:</b> 0.1–1.0. Values above 2.0 are usually strong; verify that objects remain recognizable in Preview.",
    ),
    "train_ratio": (
        "<b>作用：</b>分配给训练集的原始样本比例；增强在划分后进行。三个比例之和必须为 1.000。<br>"
        "<b>建议：</b>0.7–0.8，并确保验证集和测试集仍有足够原始样本。",
        "<b>Purpose:</b> Fraction of source samples assigned to training. Augmentation happens after splitting. All three ratios must total 1.000.<br>"
        "<b>Recommended:</b> 0.7–0.8, while keeping enough source samples for validation and testing.",
    ),
    "val_ratio": (
        "<b>作用：</b>分配给验证集的原始样本比例，用于训练期间调参与评估。三个比例之和必须为 1.000。<br>"
        "<b>建议：</b>0.1–0.2；小数据集应优先保证验证样本数量足够。",
        "<b>Purpose:</b> Fraction of source samples assigned to validation for tuning and evaluation during training. All three ratios must total 1.000.<br>"
        "<b>Recommended:</b> 0.1–0.2; for small datasets, prioritize having enough validation samples.",
    ),
    "test_ratio": (
        "<b>作用：</b>分配给独立测试集的原始样本比例。三个比例之和必须为 1.000。<br>"
        "<b>建议：</b>0.1–0.2；测试集应保留到最终评估，避免参与调参。",
        "<b>Purpose:</b> Fraction of source samples assigned to the independent test set. All three ratios must total 1.000.<br>"
        "<b>Recommended:</b> 0.1–0.2. Reserve the test set for final evaluation and do not use it for tuning.",
    ),
    "ratio_total": (
        "<b>作用：</b>显示 Train、Val、Test 比例之和；只有接近 1.000 时配置才有效。",
        "<b>Purpose:</b> Shows the sum of Train, Val, and Test ratios. The configuration is valid only when the total is approximately 1.000.",
    ),
    "class_mapping": (
        "<b>作用：</b>为扫描到的源类别 ID 指定名称。打包时源 ID 会按升序映射为连续 ID，名称按同一顺序写入 data.yaml。<br>"
        "<b>建议：</b>使用简短、明确且不重复的类别名称。",
        "<b>Purpose:</b> Assigns names to scanned source class IDs. During packaging, source IDs are remapped in ascending order to contiguous IDs, and names are written to data.yaml in the same order.<br>"
        "<b>Recommended:</b> Use short, clear, and unique class names.",
    ),
    "processing_mode": (
        "<b>作用：</b>决定任务是否增强、是否划分数据集以及最终输出目录结构。将鼠标停留在各个模式选项上可查看具体区别。",
        "<b>Purpose:</b> Controls whether the task augments, splits, and packages the dataset, and determines the final output structure. Hover over each mode option for its exact behavior.",
    ),
    "mode_full": (
        "<b>全套流程：</b>先把原始样本划分到 train、val、test，再在各自集合内增强。输出增强后的标准 YOLO 数据集和 data.yaml。<br>"
        "<b>适用：</b>希望直接得到可训练数据集，并从流程上避免同源样本跨集合。",
        "<b>Full Pipeline:</b> Splits source samples into train, val, and test first, then augments inside each split. Outputs an augmented standard YOLO dataset with data.yaml.<br>"
        "<b>Use when:</b> You want a training-ready dataset while structurally preventing related samples from crossing splits.",
    ),
    "mode_augment": (
        "<b>只增强：</b>输出平铺的 images 和 labels，包含缩放后的原样本及增强副本；不划分 train/val/test，也不生成 data.yaml。<br>"
        "<b>适用：</b>只需生成增强素材，或准备自行管理数据集划分。",
        "<b>Augment Only:</b> Outputs flat images and labels folders containing resized source samples and augmented copies. It does not create train/val/test splits or data.yaml.<br>"
        "<b>Use when:</b> You only need augmented samples or will manage dataset splitting separately.",
    ),
    "mode_package": (
        "<b>只分类包装：</b>不执行增强；将原始有效样本划分为标准 YOLO 数据集，并生成 data.yaml 和连续类别 ID。<br>"
        "<b>适用：</b>原始数据已足够，只需要规范划分和包装。",
        "<b>Package Only:</b> Performs no augmentation. It splits valid source samples into a standard YOLO dataset, writes data.yaml, and remaps classes to contiguous IDs.<br>"
        "<b>Use when:</b> The source data is sufficient and only standardized splitting and packaging are needed.",
    ),
    "import_config": (
        "从 JSON 配置文件载入并替换当前参数；不会自动扫描或启动任务。",
        "Load a JSON configuration and replace the current parameters. This does not scan the dataset or start a task.",
    ),
    "export_config": (
        "将当前参数校验后保存为带版本号的 JSON 配置文件；不会执行数据处理。",
        "Validate and save the current parameters as a versioned JSON configuration. No dataset processing is performed.",
    ),
    "reset_config": (
        "恢复程序默认参数；不会删除数据、输出结果或已选择的路径。",
        "Restore the default parameters. This does not delete data, output results, or selected paths.",
    ),
    "scan_dataset": (
        "后台检查 images/labels 配对、图片可读性、标签格式和类别 ID。更换数据集路径后应重新扫描。",
        "Check images/labels pairing, image readability, label format, and class IDs in the background. Scan again after changing the dataset path.",
    ),
    "preview_sample": (
        "使用当前参数预览所选图片；每次点击都会重新随机增强。预览不会写入输出文件。",
        "Preview the selected image with the current parameters. Every click generates a new random augmentation. Preview does not write output files.",
    ),
    "start_task": (
        "使用当前路径、参数和执行模式启动后台任务；开始前必须完成有效扫描。",
        "Start a background task using the current paths, parameters, and processing mode. A valid scan is required first.",
    ),
    "cancel_task": (
        "请求安全取消当前扫描或处理任务。取消在当前最小处理单元结束后生效，未完成输出可能保留。",
        "Request safe cancellation of the current scan or processing task. Cancellation takes effect after the current atomic step, and partial output may remain.",
    ),
    "rerun_task": (
        "使用界面中当前的路径、参数和模式再次启动任务；不保证与上一次配置完全相同。",
        "Start another task using the paths, parameters, and mode currently shown in the UI. These may differ from the previous run.",
    ),
    "open_output": (
        "在文件管理器中打开最近一次成功任务的 run_N 输出目录。",
        "Open the run_N folder from the most recent successful task in the file manager.",
    ),
}


def ui_text(language: str, text: str) -> str:
    if language == "en":
        return TEXTS.get(text, text)
    return text


def ui_tooltip(language: str, key: str) -> str:
    chinese, english = TOOLTIPS[key]
    return english if language == "en" else chinese
