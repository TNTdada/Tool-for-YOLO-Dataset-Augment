import os
import random
import shutil
import time
import math
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端解决 pyinstaller 打包后非交互渲染问题
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


config = {
    'target_size': 640,
    'augment_per_image': 5,
    'max_augs_per_image': 3,
    'train_ratio': 0.7,
    'val_ratio': 0.2,
    'test_ratio': 0.1,
    'class_names': {},
    
    'aug_brightness_enabled': True,
    'brightness_factor': (0.6, 1.2),
    
    'aug_noise_enabled': True,
    'noise_std': 10.0,
    
    'aug_occlusion_enabled': True,
    'occlusion_size': (40, 100),
    'occlusion_count': 2,
    
    'aug_hflip_enabled': True,
    'aug_vflip_enabled': True,
    
    'aug_rotate_enabled': True,
    'rotation_range': (-15, 15),
    
    'aug_blur_enabled': True,
    'blur_radius': (0.1, 0.5),
}

dataset_info = {
    'valid_pairs': [],
    'invalid_images': [],
    'classes': set()
}

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_next_run_dir(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            try:
                idx = int(d.name.split("_")[1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return base_dir / f"run_{max_idx + 1}"

def safe_imwrite(path, img):
    ext = os.path.splitext(path)[1].lower() or '.jpg'
    cv2.imencode(ext, img)[1].tofile(str(path))


def scan_dataset_thoroughly(src: Path):
    if not (src.is_dir() and (src / "images").exists() and (src / "labels").exists()):
        print("[!] 错误：请拖入有效的数据集文件夹（必需包含 images 和 labels 子文件夹）")
        return False
    
    img_dir = src / "images"
    lbl_dir = src / "labels"
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    if not img_dir.exists():
        return False
        
    base_images = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in image_exts]
    
    valid_pairs = []
    invalid_images = []
    classes = set()
    
    print("\n>>> 正在扫描数据集，请稍候...")
    for img_file in base_images:
        base_name = os.path.splitext(img_file)[0]
        lbl_file = f"{base_name}.txt"
        lbl_path = lbl_dir / lbl_file
        
        if lbl_path.exists():
            valid_pairs.append((img_file, lbl_file))
            try:
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    for line in f.read().splitlines():
                        if not line.strip(): continue
                        c_id = int(line.split()[0])
                        classes.add(c_id)
            except: pass
        else:
            invalid_images.append(img_file)
            
    dataset_info['valid_pairs'] = valid_pairs
    dataset_info['invalid_images'] = invalid_images
    dataset_info['classes'] = sorted(list(classes))
    
    if not config['class_names'] or set(config['class_names'].keys()) != set(dataset_info['classes']):
        config['class_names'] = {c: str(c) for c in dataset_info['classes']}
        
    print("================ 扫描报告 ================")
    print(f"- 总图片数: {len(base_images)}")
    print(f"- 有效图像-标签对: {len(valid_pairs)}")
    print(f"- 缺失标签的数据: {len(invalid_images)} 张")
    print(f"- 探测到目标类别 ID: {dataset_info['classes']}")
    print("========================================\n")
    return True


def resize_and_pad(img, target_size=640):
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img, (1, 0, 0)

    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    return padded, (scale, left, top)


def adjust_bboxes(bboxes, img_size, params):
    scale, pad_x, pad_y = params
    target_size = config['target_size']

    adjusted = []
    for bbox in bboxes:
        if not bbox.strip(): continue
        try:
            parts = bbox.split()
            class_id = int(parts[0])
            x_center, y_center = float(parts[1]), float(parts[2])
            width, height = float(parts[3]), float(parts[4])
        except: continue

        orig_w, orig_h = img_size
        x_center = (x_center * orig_w * scale + pad_x) / target_size
        y_center = (y_center * orig_h * scale + pad_y) / target_size
        width = (width * orig_w * scale) / target_size
        height = (height * orig_h * scale) / target_size

        x_center = np.clip(x_center, 0.0, 1.0)
        y_center = np.clip(y_center, 0.0, 1.0)
        width = np.clip(width, 0.0, 1.0)
        height = np.clip(height, 0.0, 1.0)
        adjusted.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return adjusted

def rotate_bboxes(bboxes, angle, cx=0.5, cy=0.5):
    angle_rad = -math.radians(angle)
    new_bboxes = []
    for bbox in bboxes:
        try:
            parts = bbox.split()
            c_id, x, y, bw, bh = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            corners = [
                (x - bw / 2 - cx, y - bh / 2 - cy),
                (x + bw / 2 - cx, y - bh / 2 - cy),
                (x - bw / 2 - cx, y + bh / 2 - cy),
                (x + bw / 2 - cx, y + bh / 2 - cy)
            ]
            rot_corners = []
            for corner_x, corner_y in corners:
                rx = corner_x * math.cos(angle_rad) - corner_y * math.sin(angle_rad)
                ry = corner_x * math.sin(angle_rad) + corner_y * math.cos(angle_rad)
                rot_corners.append((rx + cx, ry + cy))
            
            xs, ys = [c[0] for c in rot_corners], [c[1] for c in rot_corners]
            new_xmin, new_xmax = max(0.0, min(1.0, min(xs))), max(0.0, min(1.0, max(xs)))
            new_ymin, new_ymax = max(0.0, min(1.0, min(ys))), max(0.0, min(1.0, max(ys)))
            
            new_w, new_h = new_xmax - new_xmin, new_ymax - new_ymin
            new_x, new_y = new_xmin + new_w / 2, new_ymin + new_h / 2
            
            if new_w > 0.01 and new_h > 0.01:
                new_bboxes.append(f"{c_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")
        except: continue
    return new_bboxes


def augment_image(img, bboxes):
    enabled_augs = []
    if config['aug_brightness_enabled']: enabled_augs.append('brightness')
    if config['aug_noise_enabled']: enabled_augs.append('noise')
    if config['aug_occlusion_enabled']: enabled_augs.append('occlusion')
    if config['aug_hflip_enabled']: enabled_augs.append('hflip')
    if config['aug_vflip_enabled']: enabled_augs.append('vflip')
    if config['aug_rotate_enabled']: enabled_augs.append('rotate')
    if config['aug_blur_enabled']: enabled_augs.append('blur')

    if not enabled_augs:
        return img, bboxes

    max_possible_augs = min(config['max_augs_per_image'], len(enabled_augs))
    num_augs = random.randint(1, max_possible_augs)
    augmentations = random.sample(enabled_augs, k=num_augs)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for aug in augmentations:
        if aug == "brightness":
            factor = random.uniform(*config['brightness_factor'])
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(factor)
        elif aug == "noise":
            img_np = np.array(img_pil).astype(np.float32)
            noise = np.random.normal(0, config['noise_std'], img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
        elif aug == "occlusion":
            img_np = np.array(img_pil)
            h, w, _ = img_np.shape
            for _ in range(config['occlusion_count']):
                occ_w = random.randint(*config['occlusion_size'])
                occ_h = random.randint(*config['occlusion_size'])
                occ_x = random.randint(0, max(1, w - occ_w))
                occ_y = random.randint(0, max(1, h - occ_h))
                img_np[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w] = (
                    np.random.randint(0, 255, (occ_h, occ_w, 3)))
            img_pil = Image.fromarray(img_np)
        elif aug == "hflip":
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            new_bboxes = []
            for bbox in bboxes:
                try:
                    parts = bbox.split()
                    x_center = 1.0 - float(parts[1])
                    new_bboxes.append(f"{parts[0]} {x_center:.6f} {parts[2]} {parts[3]} {parts[4]}")
                except: continue
            bboxes = new_bboxes
        elif aug == "vflip":
            img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
            new_bboxes = []
            for bbox in bboxes:
                try:
                    parts = bbox.split()
                    y_center = 1.0 - float(parts[2])
                    new_bboxes.append(f"{parts[0]} {parts[1]} {y_center:.6f} {parts[3]} {parts[4]}")
                except: continue
            bboxes = new_bboxes
        elif aug == "rotate":
            angle = random.randint(*config['rotation_range'])
            bg = Image.new('RGB', img_pil.size, (114, 114, 114))
            rotated = img_pil.rotate(angle, resample=Image.BILINEAR, expand=False)
            bg.paste(rotated, (0, 0))
            img_pil = bg
            bboxes = rotate_bboxes(bboxes, angle)
        elif aug == "blur":
            blur_radius = random.uniform(*config['blur_radius'])
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), bboxes


def draw_bboxes_on_image(img, bboxes_str_list, global_config):
    img_draw = img.copy()
    h, w = img_draw.shape[:2]
    for bbox in bboxes_str_list:
        if not bbox.strip(): continue
        parts = bbox.split()
        c_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        
        abs_cx, abs_cy = int(cx * w), int(cy * h)
        abs_bw, abs_bh = int(bw * w), int(bh * h)
        
        xmin = max(0, int(abs_cx - abs_bw / 2))
        ymin = max(0, int(abs_cy - abs_bh / 2))
        xmax = min(w, int(abs_cx + abs_bw / 2))
        ymax = min(h, int(abs_cy + abs_bh / 2))
        
        c_name = global_config['class_names'].get(c_id, str(c_id))
        
        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        (text_w, text_h), _ = cv2.getTextSize(c_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_draw, (xmin, ymin - text_h - 5), (xmin + text_w, ymin), (0, 255, 0), -1)
        cv2.putText(img_draw, c_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

def run_preview_process(dataset_dir_str, num_to_show, run_augment, sample_pairs, current_config):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    global config
    config.update(current_config) 
    
    dataset_dir = Path(dataset_dir_str)
    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"
    
    if not run_augment:
        fig, axes = plt.subplots(1, num_to_show, figsize=(5 * num_to_show, 5), squeeze=False)
        fig.canvas.manager.set_window_title("数据集预览")
        for i, (img_file, lbl_file) in enumerate(sample_pairs):
            ax = axes[0][i]
            img_path = img_dir / img_file
            lbl_path = lbl_dir / lbl_file
            try:
                pil_img = Image.open(img_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    bboxes = f.read().splitlines()
                resized_img, params = resize_and_pad(img, config['target_size'])
                orig_size = (img.shape[1], img.shape[0])
                bboxes = adjust_bboxes(bboxes, orig_size, params)
                draw_img = draw_bboxes_on_image(resized_img, bboxes, config)
                ax.imshow(draw_img)
                ax.set_title(f"原始图像 {i+1}")
                ax.axis('off')
            except Exception as e:
                ax.set_title(f"读取异常")
                ax.axis('off')
    else:
        fig, axes = plt.subplots(2, num_to_show, figsize=(5 * num_to_show, 9), squeeze=False)
        fig.canvas.manager.set_window_title("数据集增强对照预览")
        for i, (img_file, lbl_file) in enumerate(sample_pairs):
            ax_orig = axes[0][i]
            ax_aug = axes[1][i]
            img_path = img_dir / img_file
            lbl_path = lbl_dir / lbl_file
            try:
                pil_img = Image.open(img_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    bboxes = f.read().splitlines()
                resized_img, params = resize_and_pad(img, config['target_size'])
                orig_size = (img.shape[1], img.shape[0])
                bboxes = adjust_bboxes(bboxes, orig_size, params)
                draw_orig = draw_bboxes_on_image(resized_img, bboxes, config)
                ax_orig.imshow(draw_orig)
                ax_orig.set_title(f"原图 {i+1}")
                ax_orig.axis('off')

                augmented_img, augmented_bboxes = augment_image(resized_img.copy(), bboxes.copy())
                draw_aug = draw_bboxes_on_image(augmented_img, augmented_bboxes, config)
                ax_aug.imshow(draw_aug)
                ax_aug.set_title(f"增强效果 {i+1}")
                ax_aug.axis('off')
            except Exception as e:
                ax_orig.set_title("加载失败")
                ax_orig.axis('off')
                ax_aug.axis('off')
                
    plt.tight_layout()
    plt.show(block=True) 

# -----------------------------

def process_dataset(dataset_dir: Path, output_dir: Path):
    img_base_dir = dataset_dir / "images"
    label_base_dir = dataset_dir / "labels"

    img_out_dir = output_dir / "images"
    label_out_dir = output_dir / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    base_pairs = dataset_info['valid_pairs']
    if not base_pairs:
        print("[!] 未找到有效图片数据。")
        return

    print(f"[*] 找到 {len(base_pairs)} 张有效图片，开始增强流程...")
    total_images = len(base_pairs) * (config['augment_per_image'] + 1)

    with tqdm(total=total_images, desc="处理进度", unit='img', ncols=80) as pbar:
        for img_file, lbl_file in base_pairs:
            img_path = img_base_dir / img_file
            label_path = label_base_dir / lbl_file
            base_name = os.path.splitext(img_file)[0]

            try:
                pil_img = Image.open(img_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                continue

            resized_img, params = resize_and_pad(img, config['target_size'])
            orig_size = (img.shape[1], img.shape[0])

            bboxes = []
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    bboxes = f.read().splitlines()
                bboxes = adjust_bboxes(bboxes, orig_size, params)
            except: pass

            save_path = img_out_dir / f"{base_name}.jpg"
            safe_imwrite(save_path, resized_img)

            if bboxes:
                with open(label_out_dir / f"{base_name}.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(bboxes))
            pbar.update(1)

            for i in range(config['augment_per_image']):
                try:
                    augmented_img, augmented_bboxes = augment_image(resized_img.copy(), bboxes.copy())
                    new_img_name = f"{base_name}_aug{i}.jpg"
                    new_label_name = f"{base_name}_aug{i}.txt"

                    safe_imwrite(img_out_dir / new_img_name, augmented_img)

                    if augmented_bboxes:
                        with open(label_out_dir / new_label_name, 'w', encoding='utf-8') as f:
                            f.write("\n".join(augmented_bboxes))
                    pbar.update(1)
                except Exception as e:
                    print(f"\n[!] augmenting image error {img_file} (aug{i}): {e}")

def create_dir_structure(base_dir):
    dirs = {
        'train': ['images', 'labels'],
        'val': ['images', 'labels'],
        'test': ['images', 'labels']
    }
    for split in dirs:
        for subdir in dirs[split]:
            path = os.path.join(base_dir, split, subdir)
            os.makedirs(path, exist_ok=True)
    return os.path.join(base_dir, 'data.yaml')

def split_dataset(pairs, train_ratio, val_ratio, test_ratio):
    random.shuffle(pairs)
    total = len(pairs)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_set = pairs[:train_count]
    val_set = pairs[train_count:train_count + val_count]
    test_set = pairs[train_count + val_count:]

    print(f"[*] 分割占比 - 训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")
    return train_set, val_set, test_set

def copy_files(dataset_dir, split, pairs, images_source, labels_source):
    images_target = os.path.join(dataset_dir, split, 'images')
    labels_target = os.path.join(dataset_dir, split, 'labels')

    for img_file, label_file in pairs:
        shutil.copy2(os.path.join(images_source, img_file), os.path.join(images_target, img_file))
        shutil.copy2(os.path.join(labels_source, label_file), os.path.join(labels_target, label_file))

def generate_yaml(yaml_path):
    c_names = []
    for c in dataset_info['classes']:
        c_names.append(config['class_names'][c])
        
    data = {
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(c_names),
        'names': c_names
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"[OK] 已创建配置文件: {yaml_path}")


def prepare_yolo_dataset(input_dir, output_dir):
    yaml_path = create_dir_structure(output_dir)
    
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    pairs = []
    if (input_dir / "images").exists():
        for img_file in os.listdir(input_dir / "images"):
            name, ext = os.path.splitext(img_file)
            if ext.lower() not in valid_exts: continue
            label_file = f"{name}.txt"
            if os.path.exists(os.path.join(input_dir / "labels", label_file)):
                pairs.append((img_file, label_file))

    if not pairs:
        print("[!] 未找到对应的图像标签文件。")
        return

    train_set, val_set, test_set = split_dataset(
        pairs, config['train_ratio'], config['val_ratio'], config['test_ratio']
    )

    print("[*] 正在分发拷贝至 Train/Val/Test 集合...")
    copy_files(output_dir, 'train', train_set, input_dir / "images", input_dir / "labels")
    copy_files(output_dir, 'val', val_set, input_dir / "images", input_dir / "labels")
    copy_files(output_dir, 'test', test_set, input_dir / "images", input_dir / "labels")

    generate_yaml(yaml_path)
    print(f"\n[OK] YOLO 格式数据集整理完毕: {os.path.abspath(output_dir)}")


def status(b): 
    return "ON " if b else "OFF"

def show_menu():
    print("\n" + "="*55)
    print("   Tool-for-YOLO-Dataset-Augment")
    print("   YOLO 目标检测数据集增强与划分工具 v2.0")
    print("="*55)

    print("\n┌── [数据集概览] ──────────────────────────────────────────────")
    print(f"│ [+] 标签完备图片: {len(dataset_info['valid_pairs'])} 张 │  [-] 异常/缺失图像: {len(dataset_info['invalid_images'])} 张")
    print(f"│ [>] 分类 IDs: {dataset_info['classes']}")
    print(f"│ [>] 类别映射: {config['class_names']}")
    print("└──────────────────────────────────────────────────────────────")
    
    print("\n┌── [通用策略与划分原则] ──────────────────────────────────────")
    print(f"│ 取样尺寸: {config['target_size']}x{config['target_size']} (Letterbox)")
    print(f"│ 生成倍数: 每行原图增广 {config['augment_per_image']} 张  |  随机重叠上限: {config['max_augs_per_image']} 种")
    print(f"│ 数据配比: 训练集 {config['train_ratio']} | 验证集 {config['val_ratio']} | 测试集 {config['test_ratio']}")
    print("└──────────────────────────────────────────────────────────────")

    print("\n┌── [增强手段与属性配置] ──────────────────────────────────────")
    print(f"│ [亮度变换]  {status(config['aug_brightness_enabled']):<4} │ 范围: {config['brightness_factor']}")
    print(f"│ [高斯噪声]  {status(config['aug_noise_enabled']):<4} │ 噪声标准差: {config['noise_std']}")
    print(f"│ [随机遮挡]  {status(config['aug_occlusion_enabled']):<4} │ 遮挡数量: {config['occlusion_count']}, 尺寸区间: {config['occlusion_size']}")
    print(f"│ [水平镜像]  {status(config['aug_hflip_enabled']):<4} │ -")
    print(f"│ [垂直镜像]  {status(config['aug_vflip_enabled']):<4} │ -")
    print(f"│ [随机旋转]  {status(config['aug_rotate_enabled']):<4} │ 角度区间: {config['rotation_range']}")
    print(f"│ [高斯模糊]  {status(config['aug_blur_enabled']):<4} │ 模糊半径: {config['blur_radius']}")
    print("└──────────────────────────────────────────────────────────────")
    
    print("\n【任务执行表】")
    print("  1 = 完整执行: 数据增强 + 坐标变换 + YOLO集合打包")
    print("  2 = 仅生成数据增强及包围框换算")
    print("  3 = 仅执行切割装配打包 (不进行图像内容增强)")
    
    print("\n【功能预览 (在后台处理，不影响终端命令)】")
    print("  p1, p3, p5 = 预览读取原图的自适应边框效果(抽取其中数字对应的展示数量)")
    print("  a1, a3, a5 = 开启对照: 预览原图在当前激活参数下的模拟增强结果")
    
    print("\n【系统命令】")
    print(" 'set' = 进入修改环境参数设置")
    print(" 'q'   = 安全退出")
    
    return input("\n>> 请指派命令：").strip().lower()

def print_set_menu():
    print("\n================= 参数设置与修改 =================\n")
    print("┌── [基础配置] ────────────────────────────────────────────────")
    print(f"│ 1. 目标图像大小约束               (当前: {config['target_size']})")
    print(f"│ 2. 每张图片生成倍数及组合数目     (当前倍数: {config['augment_per_image']}, 组合上限: {config['max_augs_per_image']})")
    print(f"│ 3. 比例分割设定                   (当前: Train:{config['train_ratio']}, Val:{config['val_ratio']}, Test:{config['test_ratio']})")
    print(f"│ 4. 修改标签名映射 (ID -> Name)    (当前: {config['class_names']})")
    print("└──────────────────────────────────────────────────────────────")
    
    print("\n┌── [增强手段开关与参数设置] ──────────────────────────────────")
    print(f"│ 11. 亮度变换   (状态: {status(config['aug_brightness_enabled']):<4}) | 参数: 范围 {config['brightness_factor']}")
    print(f"│ 12. 高斯噪声   (状态: {status(config['aug_noise_enabled']):<4}) | 参数: 强度 {config['noise_std']}")
    print(f"│ 13. 随机遮挡   (状态: {status(config['aug_occlusion_enabled']):<4}) | 参数: 数量 {config['occlusion_count']}, 尺寸 {config['occlusion_size']}")
    print(f"│ 14. 水平镜像   (状态: {status(config['aug_hflip_enabled']):<4}) | -")
    print(f"│ 15. 垂直镜像   (状态: {status(config['aug_vflip_enabled']):<4}) | -")
    print(f"│ 16. 几何旋转   (状态: {status(config['aug_rotate_enabled']):<4}) | 参数: 角度区间 {config['rotation_range']}")
    print(f"│ 17. 高斯模糊   (状态: {status(config['aug_blur_enabled']):<4}) | 参数: 半径区间 {config['blur_radius']}")
    print("└──────────────────────────────────────────────────────────────")
    print("\n  'b' - 返回主菜单\n")

def toggle_choice(name, var1, current_val):
    ans = input(f"是否开启 [{name}] 功能？(y 开启 / n 关闭) (当前: {status(current_val)}): ").strip().lower()
    if ans == 'y': return True
    elif ans == 'n': return False
    return current_val

def adjust_config():
    while True:
        clear_console()
        print_set_menu()
        choice = input("请选择您要进入配置的轨道代号：").strip().lower()

        if choice == '1':
            try: 
                config['target_size'] = max(1, int(input(f"指定目标图像宽/高 (当前={config['target_size']}): ")))
                print("[OK] 设置成功！")
            except: print("[!] 输入无效！")
            
        elif choice == '2':
            try: 
                v1 = input(f"每张原图生成变异图片数量 (当前={config['augment_per_image']}): ")
                if v1.strip(): config['augment_per_image'] = max(1, int(v1))
                
                v2 = input(f"单张图片最多同时应用几种增强效果 (当前={config['max_augs_per_image']}): ")
                if v2.strip(): config['max_augs_per_image'] = max(1, int(v2))
                print("[OK] 设置成功！")
            except: print("[!] 输入无效！")
            
        elif choice == '3':
            try:
                ratios = input(f"训练层,验证层,评测层 分配值 (当前={config['train_ratio']},{config['val_ratio']},{config['test_ratio']}): ")
                if ratios.strip():
                    tr, vr, te = map(float, ratios.split(','))
                    if abs(tr + vr + te - 1.0) < 0.001 and all(r >= 0 for r in [tr, vr, te]):
                        config['train_ratio'], config['val_ratio'], config['test_ratio'] = tr, vr, te
                        print("[OK] 设置成功！")
                    else: print("[!] 比例总和必须等于 1.0！")
            except: print("[!] 输入格式有误。")
            
        elif choice == '4':
            print(f"\n当前图像的目标类别 ID: {dataset_info['classes']}")
            names = input(f"请按顺序输入实际类别名称(逗号分隔,需填写{len(dataset_info['classes'])}个): ").strip()
            if names:
                name_list = [n.strip() for n in names.split(',')]
                if len(name_list) == len(dataset_info['classes']):
                    for i, cid in enumerate(dataset_info['classes']):
                        config['class_names'][cid] = name_list[i]
                    print("[OK] 类别名称映射更新成功！")
                else: print(f"[!] 错误：应当提供 {len(dataset_info['classes'])} 个实体名称。")
                
        elif choice == '11':
            config['aug_brightness_enabled'] = toggle_choice("亮度变换", "aug_brightness_enabled", config['aug_brightness_enabled'])
            if config['aug_brightness_enabled']:
                try:
                    val = input(f"波动范围限制 (如 0.6,1.2) [直接回车保留原有设定]: ")
                    if val.strip(): config['brightness_factor'] = tuple(map(float, val.split(',')))
                except: pass
                
        elif choice == '12':
            config['aug_noise_enabled'] = toggle_choice("高斯噪声", "aug_noise_enabled", config['aug_noise_enabled'])
            if config['aug_noise_enabled']:
                try:
                    val = input(f"噪声标准差 (如 10.0) [直接回车保留原有设定]: ")
                    if val.strip(): config['noise_std'] = float(val)
                except: pass
                
        elif choice == '13':
            config['aug_occlusion_enabled'] = toggle_choice("随机遮挡", "aug_occlusion_enabled", config['aug_occlusion_enabled'])
            if config['aug_occlusion_enabled']:
                try:
                    v1 = input("遮挡块数量 (如 2) [直接回车保留原有设定]: ")
                    if v1.strip(): config['occlusion_count'] = int(v1)
                    v2 = input("遮挡块尺寸区间 (如 40,100) [直接回车保留原有设定]: ")
                    if v2.strip(): config['occlusion_size'] = tuple(map(int, v2.split(',')))
                except: pass
                
        elif choice == '14':
            config['aug_hflip_enabled'] = toggle_choice("水平镜像", "aug_hflip_enabled", config['aug_hflip_enabled'])
        elif choice == '15':
            config['aug_vflip_enabled'] = toggle_choice("垂直镜像", "aug_vflip_enabled", config['aug_vflip_enabled'])
            
        elif choice == '16':
            config['aug_rotate_enabled'] = toggle_choice("几何旋转", "aug_rotate_enabled", config['aug_rotate_enabled'])
            if config['aug_rotate_enabled']:
                try:
                    val = input("旋转角度区间限制 (如 -15,15) [直接回车保留原有设定]: ")
                    if val.strip(): config['rotation_range'] = tuple(map(int, val.split(',')))
                except: pass
                
        elif choice == '17':
            config['aug_blur_enabled'] = toggle_choice("高斯模糊", "aug_blur_enabled", config['aug_blur_enabled'])
            if config['aug_blur_enabled']:
                try:
                    val = input("模糊滤波半径区间 (如 0.1,0.5) [直接回车保留原有设定]: ")
                    if val.strip(): config['blur_radius'] = tuple(map(float, val.split(',')))
                except: pass
                
        elif choice == 'b':
            break
        else:
            print("[!] 未知选项")
            
        if choice != 'b':
            input("\n操作已记录，按回车键回到配置列表...")


def main_loop():
    global_out_dir = Path('augmented_dataset')

    while True:
        clear_console()
        print("============================================")
        print("   Tool-for-YOLO-Dataset-Augment")
        print("   YOLO 目标检测数据集增强与划分工具 v2.0")
        print("============================================")
        user_path = input("\n>>> [1] 请输入或直接拖入用于处理的图像母目录路径然后回车（输入 q 退出）：\n").strip('"').strip("'")
        if user_path.lower() == 'q': break

        dataset_path = Path(user_path)
        if not scan_dataset_thoroughly(dataset_path):
            input("\n按回车键重新选择路径...")
            continue
            
        input("\n数据集挂载成功！请按回车键进入控制台核心...")

        while True:
            clear_console()
            cmd = show_menu()
            
            if cmd == 'q': break
            elif cmd == 'set': adjust_config()
                
            elif cmd in ['p1', 'p3', 'p5']:
                cnt = int(cmd[-1])
                num_to_show = min(cnt, len(dataset_info['valid_pairs']))
                if num_to_show == 0: continue
                sample_pairs = random.sample(dataset_info['valid_pairs'], num_to_show)
                p = multiprocessing.Process(target=run_preview_process, 
                    args=(str(dataset_path), num_to_show, False, sample_pairs, config.copy()))
                p.start()
                time.sleep(0.5)
                print("\n[OK] 预览窗口已成功下发！")
                input("按回车键继续指令操作...")
                
            elif cmd in ['a1', 'a3', 'a5']:
                cnt = int(cmd[-1])
                num_to_show = min(cnt, len(dataset_info['valid_pairs']))
                if num_to_show == 0: continue
                sample_pairs = random.sample(dataset_info['valid_pairs'], num_to_show)
                p = multiprocessing.Process(target=run_preview_process, 
                    args=(str(dataset_path), num_to_show, True, sample_pairs, config.copy()))
                p.start()
                time.sleep(0.5)
                print("\n[OK] 增强效果对照窗口已启动！")
                input("按回车键继续指令操作...")
                
            elif cmd in ['1', '2', '3']:
                # 二次计算与验证
                total_original = len(dataset_info['valid_pairs'])
                # 如果执行1或2，将会生产增强图；如果是3，只做划分。
                est_img_cnt = total_original * (config['augment_per_image'] + 1) if cmd in ['1', '2'] else total_original
                
                # 检查默认字典是否发生更改
                default_names = {c: str(c) for c in dataset_info['classes']}
                label_warning = "标签名似乎使用了默认数字ID，如有需要请前往 'set' 命令中修改" if default_names == config['class_names'] else "类名重映射设置正常"
                
                print(f"\n--- [确认执行方案 {cmd}] ---")
                print(f"预计生成图片总数: ~{est_img_cnt} 张")
                print(f"类名标签检查: {label_warning}")
                ans = input("\n请确认是否开始执行？ (输入 y 继续)：").strip().lower()
                
                if ans != 'y': 
                    continue

                run_dir = get_next_run_dir(global_out_dir)
                if cmd in ['1', '2']: process_dataset(dataset_path, run_dir)
                if cmd in ['1', '3']:
                    src_dir = run_dir if cmd == '1' else dataset_path
                    prepare_yolo_dataset(src_dir, run_dir / "yolo_dataset")

                print(f"\n[SUCCESS] 运算完毕！程序处理完成的数据已存放在目录：{run_dir}")
                input("按回车键返回主菜单...")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\n[!] 执行中断...")
