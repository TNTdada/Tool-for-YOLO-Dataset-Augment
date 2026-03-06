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
import matplotlib.pyplot as plt

# 终端的 matplotlib 图表在生成时偶尔会加载中文字体失败，因此配置全局兜底字库
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


config = {
    # --- 全局设计与打包输出 ---
    'target_size': 640,
    'augment_per_image': 5,
    'max_augs_per_image': 3,
    'train_ratio': 0.7,
    'val_ratio': 0.2,
    'test_ratio': 0.1,
    'class_names': {},

    # --- 独立增强模块的开关与精细参数 --- 
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
    """扫描现有文件夹，生成顺序递增的输出目录 run_1, run_2..."""
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
    """完美支持中文路径的安全保存"""
    ext = os.path.splitext(path)[1].lower() or '.jpg'
    cv2.imencode(ext, img)[1].tofile(str(path))


def scan_dataset_thoroughly(src: Path):
    if not (src.is_dir() and (src / "images").exists() and (src / "labels").exists()):
        print("⚠ 错误：请拖入有效的数据集文件夹（必须包含 images 和 labels 子文件夹）")
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
    
    print("\n>>> 正在快速扫描数据集，请稍候...")
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
    print(f"- 有效标注对: {len(valid_pairs)}")
    print(f"- 无标注/失效数据: {len(invalid_images)} 张")
    print(f"- 提取到目标类别 ID 列表: {dataset_info['classes']}")
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
        if not bbox.strip():
            continue
        try:
            parts = bbox.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except:
            continue

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
            
            xs = [c[0] for c in rot_corners]
            ys = [c[1] for c in rot_corners]
            new_xmin, new_xmax = max(0.0, min(1.0, min(xs))), max(0.0, min(1.0, max(xs)))
            new_ymin, new_ymax = max(0.0, min(1.0, min(ys))), max(0.0, min(1.0, max(ys)))
            
            new_w = new_xmax - new_xmin
            new_h = new_ymax - new_ymin
            new_x = new_xmin + new_w / 2
            new_y = new_ymin + new_h / 2
            
            if new_w > 0.01 and new_h > 0.01:
                new_bboxes.append(f"{c_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")
        except:
            continue
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
        return img, bboxes  # 如果全部关闭增强效果，直接返回

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


# --- 可视化预览系统 ---
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
        fig.canvas.manager.set_window_title("数据集预览 - 独立窗口")
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
                ax.set_title(f"未增强图像 {i+1}")
                ax.axis('off')
            except Exception as e:
                ax.set_title(f"读取异常")
                ax.axis('off')
    else:
        fig, axes = plt.subplots(2, num_to_show, figsize=(5 * num_to_show, 9), squeeze=False)
        fig.canvas.manager.set_window_title("数据集增强对比预览 - 独立窗口")
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
                ax_aug.set_title(f"施加混合增强后 {i+1}")
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
        print("未找到有效带标签图片数据。")
        return

    print(f"找到 {len(base_pairs)} 张有效图片启动增广...")
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
                    print(f"\nError augmenting image {img_file} (aug{i}): {e}")

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

    print(f"数据分割就绪: 训练集={len(train_set)}, 验证集={len(val_set)}, 测试集={len(test_set)}")
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
    print(f"✓ 已创建自动配置 YAML 文件: {yaml_path}")


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
        print("错误: 未找到对应的图像标签组合，无法进行分割拼装！")
        return

    train_set, val_set, test_set = split_dataset(
        pairs, config['train_ratio'], config['val_ratio'], config['test_ratio']
    )

    print("正将文件分发拷贝至 Train/Val/Test 标准集合中...")
    copy_files(output_dir, 'train', train_set, input_dir / "images", input_dir / "labels")
    copy_files(output_dir, 'val', val_set, input_dir / "images", input_dir / "labels")
    copy_files(output_dir, 'test', test_set, input_dir / "images", input_dir / "labels")

    generate_yaml(yaml_path)
    print(f"\n★ 最终 YOLO 格式数据集整理完毕在: {os.path.abspath(output_dir)}")


def status(b): 
    return "ON" if b else "OFF"

def show_menu():
    print("\n" + "="*50)
    print("      极速 Yolo 数据集扫描与强扩频工具 v2.3.2")
    print("="*50)

    print("\n┌── [数据集特征与概览] ──────────────────────────────────────────")
    print(f"│ ✔ 有效图片(带标签): {len(dataset_info['valid_pairs'])} 张 │  ✖ 无效/残缺图片: {len(dataset_info['invalid_images'])} 张")
    print(f"│ ▶ 分类 IDs: {dataset_info['classes']}")
    print(f"│ ▶ 类别映射: {config['class_names']}")
    print("└─────────────────────────────────────────────────────────────")
    
    print("\n┌── [全局与打包配置] ──────────────────────────────────────────")
    print(f"│ > 目标尺寸  : {config['target_size']}x{config['target_size']} (Letterbox自适应)")
    print(f"│ > 增广倍数  : 每张图扩充 {config['augment_per_image']} 张  |  随机叠放极值: {config['max_augs_per_image']} 种")
    print(f"│ > 数据分配  : 训练 {config['train_ratio']} | 验证 {config['val_ratio']} | 测试 {config['test_ratio']}")
    print("└─────────────────────────────────────────────────────────────")

    print("\n┌── [增强手段独立开关与参数] ──────────────────────────────────")
    print(f"│ > [亮度变换]  {status(config['aug_brightness_enabled']):<4} │ 范围: {config['brightness_factor']}")
    print(f"│ > [高斯噪声]  {status(config['aug_noise_enabled']):<4} │ 强度(标准差): {config['noise_std']}")
    print(f"│ > [随机遮挡]  {status(config['aug_occlusion_enabled']):<4} │ 遮挡块数: {config['occlusion_count']}, 尺寸区间: {config['occlusion_size']}")
    print(f"│ > [水平镜像]  {status(config['aug_hflip_enabled']):<4} │ ")
    print(f"│ > [垂直镜像]  {status(config['aug_vflip_enabled']):<4} │ ")
    print(f"│ > [随机旋转]  {status(config['aug_rotate_enabled']):<4} │ 角度变动区间: {config['rotation_range']}")
    print(f"│ > [高斯模糊]  {status(config['aug_blur_enabled']):<4} │ 模糊半径区间: {config['blur_radius']}")
    print("└─────────────────────────────────────────────────────────────")
    
    print("\n【任务执行表】")
    print("  1 = 一键执行全套(数据增强 + YOLO分类打包)")
    print("  2 = 仅生成数据增强图片(并绑定改好坐标的txt)")
    print("  3 = 仅执行切割装配(不进行图片内容增强变换)")
    
    print("\n【图形化预览 (窗口挂起在后台不阻塞界面)】")
    print("  p1, p3, p5 = 验证原图与原始边框(数量选择)")
    print("  a1, a3, a5 = 开启上下对照：观测原图被随机抽卡施加的混合增益效果")
    
    print("\n【系统命令】")
    print(" 'set' = 进入修改全量参数调整层级")
    print(" 'q'   = 安全退出进程")
    
    return input("\n>> 请指派接下来干什么：").strip().lower()


def toggle_choice(name, var1, current_val):
    ans = input(f"是否开启 [{name}] 功能？(y 开启 / n 关闭) (当前: {status(current_val)}): ").strip().lower()
    if ans == 'y': return True
    elif ans == 'n': return False
    return current_val

def adjust_config():
    while True:
        clear_console()
        print("\n================= 参数全能修改空间 =================\n")
        print(" [通用构建配置] ")
        print("   1. 目标图像大小约束")
        print("   2. 图片生成扩张倍数 / 最大同时施布变异种数")
        print("   3. Yolo 数据集训练层分类百分比权限")
        print("   4. 重排目标分类词典映射 (ID -> Label Name)")
        print("\n [核心增压手段修改] ")
        print("  11. 亮度变换 (启用/停用/配置范围)")
        print("  12. 高斯噪声 (启用/停用/配置强度)")
        print("  13. 随机遮挡 (启用/停用/配置遮挡数和块大小)")
        print("  14. 水平镜像翻转 (启用/停用)")
        print("  15. 垂直镜像翻转 (启用/停用)")
        print("  16. 几何旋转 (启用/停用/配置角度限值)")
        print("  17. 模糊泛化 (启用/停用/配置发糊半径)")
        print("\n  'b' - 返回至主界面监控中心")
        
        choice = input("\n请选择您要进入配置的轨道代号：").strip().lower()

        if choice == '1':
            try: 
                config['target_size'] = max(1, int(input(f"指定等比压缩宽/高 (当前={config['target_size']}): ")))
                print("设置成功！")
            except: print("输入无效！")
            
        elif choice == '2':
            try: 
                v1 = input(f"每张原图随机繁衍出多少张变异图 (当前={config['augment_per_image']}): ")
                if v1.strip(): config['augment_per_image'] = max(1, int(v1))
                
                v2 = input(f"单张画最大可同屏跌落多少种技能 (当前={config['max_augs_per_image']}): ")
                if v2.strip(): config['max_augs_per_image'] = max(1, int(v2))
                print("设置成功！")
            except: print("输入无效！")
            
        elif choice == '3':
            try:
                ratios = input(f"训练层,检验层,评测层 分配值 (当前={config['train_ratio']},{config['val_ratio']},{config['test_ratio']}): ")
                if ratios.strip():
                    tr, vr, te = map(float, ratios.split(','))
                    if abs(tr + vr + te - 1.0) < 0.001 and all(r >= 0 for r in [tr, vr, te]):
                        config['train_ratio'], config['val_ratio'], config['test_ratio'] = tr, vr, te
                        print("矩阵分配成功！")
                    else: print("加权数不合要求(请加起来等于1.0)！")
            except: print("格式验证错误。")
            
        elif choice == '4':
            print(f"\n目前解析到的全量标签 ID 为: {dataset_info['classes']}")
            names = input(f"请输入从低到高所对应的真实类型名(逗号分隔,需填写{len(dataset_info['classes'])}个): ").strip()
            if names:
                name_list = [n.strip() for n in names.split(',')]
                if len(name_list) == len(dataset_info['classes']):
                    for i, cid in enumerate(dataset_info['classes']):
                        config['class_names'][cid] = name_list[i]
                    print("✅ 名称词典更新成功！")
                else: print(f"❌ 错误：需提供 {len(dataset_info['classes'])} 个实体。")
                
        elif choice == '11':
            config['aug_brightness_enabled'] = toggle_choice("亮度变换", "aug_brightness_enabled", config['aug_brightness_enabled'])
            if config['aug_brightness_enabled']:
                try:
                    val = input(f"双边波幅极限 (如 0.6,1.2) [回车不改]: ")
                    if val.strip(): 
                        config['brightness_factor'] = tuple(map(float, val.split(',')))
                        print("修改成功")
                except: print("有误")
                
        elif choice == '12':
            config['aug_noise_enabled'] = toggle_choice("高斯噪声", "aug_noise_enabled", config['aug_noise_enabled'])
            if config['aug_noise_enabled']:
                try:
                    val = input(f"噪声绝对强度标准差 (如 10.0) [回车不改]: ")
                    if val.strip(): config['noise_std'] = float(val)
                except: pass
                
        elif choice == '13':
            config['aug_occlusion_enabled'] = toggle_choice("随机遮挡", "aug_occlusion_enabled", config['aug_occlusion_enabled'])
            if config['aug_occlusion_enabled']:
                try:
                    v1 = input("一图贴几块黑斑 (输入正整数) [回车跳]: ")
                    if v1.strip(): config['occlusion_count'] = int(v1)
                    v2 = input("一块黑斑动态长宽在什么范围区间 (例 40,100) [回车跳]: ")
                    if v2.strip(): config['occlusion_size'] = tuple(map(int, v2.split(',')))
                except: pass
                
        elif choice == '14':
            config['aug_hflip_enabled'] = toggle_choice("水平镜像翻转", "aug_hflip_enabled", config['aug_hflip_enabled'])
        elif choice == '15':
            config['aug_vflip_enabled'] = toggle_choice("垂直镜像翻转", "aug_vflip_enabled", config['aug_vflip_enabled'])
            
        elif choice == '16':
            config['aug_rotate_enabled'] = toggle_choice("几何不对称扭转", "aug_rotate_enabled", config['aug_rotate_enabled'])
            if config['aug_rotate_enabled']:
                try:
                    val = input("请输入偏移左倾与右摆极大极小极限角度 (例 -15,15) [回车不改]: ")
                    if val.strip(): config['rotation_range'] = tuple(map(int, val.split(',')))
                except: pass
                
        elif choice == '17':
            config['aug_blur_enabled'] = toggle_choice("发烧级近视高斯模糊", "aug_blur_enabled", config['aug_blur_enabled'])
            if config['aug_blur_enabled']:
                try:
                    val = input("模糊滤波光照半径区间 (例 0.1,0.5) [回车不改]: ")
                    if val.strip(): config['blur_radius'] = tuple(map(float, val.split(',')))
                except: pass
                
        elif choice == 'b':
            break
        else:
            print("未知选项")
            
        if choice != 'b':
            input("\n操作结束，按回车键回到配置列表...")


def main_loop():
    global_out_dir = Path('augmented_dataset')

    while True:
        clear_console()
        print("============================================")
        print("   专业 Yolo 数据集扫描与全维度强扩频工具 v2.3.2")
        print("============================================")
        user_path = input("\n>>> [第一步] 请拖入准备就绪的数据集母目录路径然后回车（输入 q 退出）：\n").strip('"').strip("'")
        if user_path.lower() == 'q': break

        dataset_path = Path(user_path)
        if not scan_dataset_thoroughly(dataset_path):
            input("\n按回车键重新选择路径...")
            continue
            
        input("\n数据集挂载成功！请按回车键进入中控菜单核心...")

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
                print("\n✅ 预览独立窗口已下发")
                input("按回车键刷新菜单...")
                
            elif cmd in ['a1', 'a3', 'a5']:
                cnt = int(cmd[-1])
                num_to_show = min(cnt, len(dataset_info['valid_pairs']))
                if num_to_show == 0: continue
                sample_pairs = random.sample(dataset_info['valid_pairs'], num_to_show)
                p = multiprocessing.Process(target=run_preview_process, 
                    args=(str(dataset_path), num_to_show, True, sample_pairs, config.copy()))
                p.start()
                time.sleep(0.5)
                print("\n✅ 倍镜独立窗口发车完毕")
                input("按回车键刷新菜单...")
                
            elif cmd in ['1', '2', '3']:
                if input(f"\n即将启动方案 [{cmd}]。如果资源过多将占用高额硬盘且无可回退。\n确认执行？ (输入 y 启动)：").strip().lower() != 'y': 
                    continue

                run_dir = get_next_run_dir(global_out_dir)
                if cmd in ['1', '2']: process_dataset(dataset_path, run_dir)
                if cmd in ['1', '3']:
                    src_dir = run_dir if cmd == '1' else dataset_path
                    prepare_yolo_dataset(src_dir, run_dir / "yolo_dataset")

                print(f"\n🎉 惊动天地的运算完毕！物理引擎完整切片存放进：{run_dir}")
                input("请按回车键凯旋总部控制台...")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\n⚠ 安全切断...")
