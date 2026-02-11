import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

AUGMENTATION_TYPES = ['brightness', 'noise', 'occlusion', 'hflip', 'vflip', 'rotate', 'blur']
config = {
    'target_size': 640,             # 统一图像大小
    'augment_per_image': 5,         # 每张图片生成的增强图片数量
    'max_augs_per_image': 3,        # 每张增强图片中最多使用的处理方式数量
    'brightness_factor': (0.6, 1.2),
    'noise_std': 10.0,
    'occlusion_size': (40, 100),
    'occlusion_count': 2,
    'blur_radius': (0.1, 0.5),
    'rotation_range': (-15, 15),
    'train_ratio': 0.6,             # 训练集比例
    'val_ratio': 0.3,              # 验证集比例
    'test_ratio': 0.1              # 测试集比例
}

def scan_dataset(src: Path):
    if src.is_dir() and (src / "images").exists() and (src / "labels").exists():
        return src
    else:
        print("⚠ 请拖入有效的数据集文件夹（必须包含 images 和 labels 子文件夹）")
        return None

def resize_and_pad(img, target_size=640):
    """将图像等比例缩放并填充到目标尺寸"""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 等比例缩放
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建目标尺寸画布并填充114(灰色)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    # 返回处理后的图像和缩放填充参数
    return padded, (scale, left, top)

def adjust_bboxes(bboxes, img_size, params):
    """调整边界框坐标"""
    scale, pad_x, pad_y = params
    target_size = config['target_size']  # 目标尺寸

    adjusted = []
    for bbox in bboxes:
        if not bbox.strip():  # 跳过空行
            continue

        try:
            parts = bbox.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except:
            print(f"Invalid bbox format: {bbox}")
            continue

        # 调整坐标
        orig_w, orig_h = img_size
        x_center = (x_center * orig_w * scale + pad_x) / target_size
        y_center = (y_center * orig_h * scale + pad_y) / target_size
        width = (width * orig_w * scale) / target_size
        height = (height * orig_h * scale) / target_size

        # 确保坐标在[0,1]范围内
        x_center = np.clip(x_center, 0.0, 1.0)
        y_center = np.clip(y_center, 0.0, 1.0)
        width = np.clip(width, 0.0, 1.0)
        height = np.clip(height, 0.0, 1.0)

        adjusted.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return adjusted

def augment_image(img, bboxes):
    """应用多种数据增强技术"""
    # 随机选择1到max_augs_per_image种增强方式
    num_augs = random.randint(1, config['max_augs_per_image'])
    augmentations = random.sample(AUGMENTATION_TYPES, k=num_augs)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for aug in augmentations:
        if aug == "brightness":  # 亮度调整
            factor = random.uniform(*config['brightness_factor'])
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(factor)

        elif aug == "noise":  # 高斯噪声
            img_np = np.array(img_pil)
            noise = np.random.normal(0, config['noise_std'], img_np.shape).astype(np.uint8)
            img_np = np.clip(img_np + noise, 0, 255)
            img_pil = Image.fromarray(img_np)

        elif aug == "occlusion":  # 随机遮挡
            img_np = np.array(img_pil)
            h, w, _ = img_np.shape
            for _ in range(config['occlusion_count']):
                occ_w = random.randint(*config['occlusion_size'])
                occ_h = random.randint(*config['occlusion_size'])
                occ_x = random.randint(0, w - occ_w)
                occ_y = random.randint(0, h - occ_h)
                img_np[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w] = (
                    np.random.randint(0, 255, (occ_h, occ_w, 3)))
            img_pil = Image.fromarray(img_np)

        elif aug == "hflip":  # 水平翻转
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            new_bboxes = []
            for bbox in bboxes:
                if not bbox.strip():
                    continue
                try:
                    parts = bbox.split()
                    class_id = parts[0]
                    x_center = 1.0 - float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    new_bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                except:
                    continue
            bboxes = new_bboxes

        elif aug == "vflip":  # 垂直翻转
            img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
            new_bboxes = []
            for bbox in bboxes:
                if not bbox.strip():
                    continue
                try:
                    parts = bbox.split()
                    class_id = parts[0]
                    x_center = float(parts[1])
                    y_center = 1.0 - float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    new_bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                except:
                    continue
            bboxes = new_bboxes

        elif aug == "rotate":  # 随机旋转
            angle = random.randint(*config['rotation_range'])
            # 创建白色背景旋转
            bg = Image.new('RGB', img_pil.size, (114, 114, 114))
            rotated = img_pil.rotate(angle, resample=Image.BILINEAR, expand=False)
            bg.paste(rotated, (0, 0))
            img_pil = bg

        elif aug == "blur":  # 高斯模糊
            blur_radius = random.uniform(*config['blur_radius'])
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), bboxes

def process_dataset(dataset_dir: Path, output_dir: Path):
    """处理整个数据集进行增强"""
    img_base_dir = dataset_dir / "images"
    label_base_dir = dataset_dir / "labels"

    img_out_dir = output_dir / "images"
    label_out_dir = output_dir / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    # 获取基础图像列表
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    base_images = [f for f in os.listdir(img_base_dir)
                   if os.path.splitext(f)[1].lower() in image_exts]

    print(f"Found {len(base_images)} base images for augmentation")

    # 计算预计处理的图像总数
    total_images = len(base_images) * (config['augment_per_image'] + 1)  # +1 表示原始图像

    # 为每张图像创建多个增强版本
    with tqdm(total=total_images, desc="Processing images", unit='img', ncols=80) as pbar:
        for img_file in base_images:
            img_path = img_base_dir / img_file
            base_name = os.path.splitext(img_file)[0]
            label_path = label_base_dir / f"{base_name}.txt"

            # 读取基础图像（使用PIL避免警告）
            try:
                pil_img = Image.open(img_path)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"\nError reading image {img_path}: {e}")
                continue

            # 统一图像大小
            resized_img, params = resize_and_pad(img, config['target_size'])
            orig_size = (img.shape[1], img.shape[0])  # 原始图像大小

            # 读取标注
            bboxes = []
            if label_path.exists():
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        bboxes = f.read().splitlines()
                    # 调整标注框
                    bboxes = adjust_bboxes(bboxes, orig_size, params)
                except Exception as e:
                    print(f"\nError reading label {label_path}: {e}")

            # 保存原始图像和标注
            save_path = img_out_dir / img_file
            cv2.imwrite(str(save_path), resized_img)
            if bboxes:
                with open(label_out_dir / f"{base_name}.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(bboxes))
            pbar.update(1)

            # 创建多个增强版本
            for i in range(config['augment_per_image']):
                try:
                    # 应用增强
                    augmented_img, augmented_bboxes = augment_image(resized_img.copy(), bboxes.copy())

                    # 确保增强后的图像保持统一尺寸
                    augmented_img, _ = resize_and_pad(augmented_img, config['target_size'])

                    # 生成新文件名
                    new_img_name = f"{base_name}_aug{i}.jpg"
                    new_label_name = f"{base_name}_aug{i}.txt"

                    # 保存增强后的图像
                    cv2.imwrite(str(img_out_dir / new_img_name), augmented_img)

                    # 保存增强后的标注
                    if augmented_bboxes:
                        with open(label_out_dir / new_label_name, 'w', encoding='utf-8') as f:
                            f.write("\n".join(augmented_bboxes))
                    pbar.update(1)
                except Exception as e:
                    print(f"\nError augmenting image {img_file} (aug{i}): {e}")

def create_dir_structure(base_dir):
    """创建YOLO数据集目录结构"""
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

def get_image_label_pairs(images_dir, labels_dir):
    """获取图像和标签文件对"""
    pairs = []
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for img_file in os.listdir(images_dir):
        name, ext = os.path.splitext(img_file)
        if ext.lower() not in valid_exts:
            continue

        label_file = f"{name}.txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {img_file}")
            continue

        pairs.append((img_file, label_file))

    return pairs

def split_dataset(pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """划分数据集"""
    # 验证比例总和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0 (current sum: {total_ratio})")

    # 随机打乱数据集
    random.shuffle(pairs)

    # 计算各分集数量
    total = len(pairs)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    # 划分数据集
    train_set = pairs[:train_count]
    val_set = pairs[train_count:train_count + val_count]
    test_set = pairs[train_count + val_count:]

    print(f"Dataset split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_set, val_set, test_set

def copy_files(dataset_dir, split, pairs, images_source, labels_source):
    """复制文件到指定目录"""
    images_target = os.path.join(dataset_dir, split, 'images')
    labels_target = os.path.join(dataset_dir, split, 'labels')

    for img_file, label_file in pairs:
        # 复制图像
        img_src = os.path.join(images_source, img_file)
        img_dst = os.path.join(images_target, img_file)
        shutil.copy2(img_src, img_dst)

        # 复制标签
        label_src = os.path.join(labels_source, label_file)
        label_dst = os.path.join(labels_target, label_file)
        shutil.copy2(label_src, label_dst)

def generate_yaml(yaml_path, dataset_dir, class_names):
    """生成YOLO格式的data.yaml文件"""
    data = {
        # 'path': os.path.abspath(dataset_dir),
        # 'train': os.path.join('train', 'images'),
        # 'val': os.path.join('val', 'images'),
        # 'test': os.path.join('test', 'images'),
        'train': '../train/images',
        'val': '../val/images',
        'test': '../test/images',
        'nc': len(class_names),
        'names': class_names
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Generated YAML config at: {yaml_path}")

def prepare_yolo_dataset(input_dir, output_dir, class_names):
    # 创建目录结构
    dataset_dir = output_dir
    yaml_path = create_dir_structure(dataset_dir)

    # 获取图像和标签文件对
    pairs = get_image_label_pairs(input_dir / "images", input_dir / "labels")

    if not pairs:
        print("Error: No valid image-label pairs found!")
        return

    print(f"Found {len(pairs)} image-label pairs")

    # 划分数据集
    train_set, val_set, test_set = split_dataset(
        pairs,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio']
    )

    # 复制文件到对应目录
    print("Copying files to train set...")
    copy_files(dataset_dir, 'train', train_set, input_dir / "images", input_dir / "labels")

    print("Copying files to validation set...")
    copy_files(dataset_dir, 'val', val_set, input_dir / "images", input_dir / "labels")

    print("Copying files to test set...")
    copy_files(dataset_dir, 'test', test_set, input_dir / "images", input_dir / "labels")

    # 生成YAML配置文件
    generate_yaml(yaml_path, dataset_dir, class_names)

    print("\nDataset preparation complete!")
    print(f"YOLO dataset created at: {os.path.abspath(dataset_dir)}")

def show_menu():
    print("\n--------- 数据集增强和整理菜单 ---------")
    print(f"目标图像大小：{config['target_size']}x{config['target_size']}")
    print(f"增强图片数量：{config['augment_per_image']} 张/原始图片")
    print(f"每张增强图片最多使用 {config['max_augs_per_image']} 种处理方式")
    print(f"亮度调整范围：{config['brightness_factor'][0]}-{config['brightness_factor'][1]}")
    print(f"噪声强度：{config['noise_std']}")
    print(f"遮挡块大小范围：{config['occlusion_size'][0]}-{config['occlusion_size'][1]}")
    print(f"遮挡块数量：{config['occlusion_count']}")
    print(f"模糊强度范围：{config['blur_radius'][0]}-{config['blur_radius'][1]}")
    print(f"旋转角度范围：{config['rotation_range'][0]}-{config['rotation_range'][1]}")
    print(f"训练集比例：{config['train_ratio']}")
    print(f"验证集比例：{config['val_ratio']}")
    print(f"测试集比例：{config['test_ratio']}")
    print("----------------------------")
    print("1      执行增强和整理")
    print("2      仅执行数据增强")
    print("3      仅执行数据整理")
    print("set    修改增强和整理参数")
    print("q      退出程序")
    return input("请输入指令：").strip().lower()

def adjust_config():
    while True:
        print("\n可调整的参数：")
        print("1. 目标图像大小")
        print(f"   当前值：{config['target_size']}")
        print("2. 增强图片数量（每张原始图片生成多少张增强图片）")
        print(f"   当前值：{config['augment_per_image']}")
        print("3. 每张增强图片中最多使用的处理方式数量")
        print(f"   当前值：{config['max_augs_per_image']}")
        print("4. 亮度调整范围")
        print(f"   当前值：{config['brightness_factor'][0]}-{config['brightness_factor'][1]}")
        print("5. 噪声强度")
        print(f"   当前值：{config['noise_std']}")
        print("6. 遮挡块大小范围")
        print(f"   当前值：{config['occlusion_size'][0]}-{config['occlusion_size'][1]}")
        print("7. 遮挡块数量")
        print(f"   当前值：{config['occlusion_count']}")
        print("8. 模糊强度范围")
        print(f"   当前值：{config['blur_radius'][0]}-{config['blur_radius'][1]}")
        print("9. 旋转角度范围")
        print(f"   当前值：{config['rotation_range'][0]}-{config['rotation_range'][1]}")
        print("10. 训练集比例")
        print(f"   当前值：{config['train_ratio']}")
        print("11. 验证集比例")
        print(f"   当前值：{config['val_ratio']}")
        print("12. 测试集比例")
        print(f"   当前值：{config['test_ratio']}")
        print("b. 返回主菜单")
        choice = input("请选择要调整的参数：").strip().lower()

        if choice == '1':
            try:
                new = int(input("新的目标图像大小："))
                if new < 1:
                    print("数值必须大于 0")
                    continue
                config['target_size'] = new
                print(f"目标图像大小已设置为：{new}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '2':
            try:
                new = int(input("新的增强图片数量（建议 1-10）："))
                if new < 1:
                    print("数值必须大于 0")
                    continue
                config['augment_per_image'] = new
                print(f"增强图片数量已设置为：{new}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '3':
            try:
                new = int(input("每张增强图片中最多使用的处理方式数量（建议 1-5）："))
                if new < 1:
                    print("数值必须大于 0")
                    continue
                config['max_augs_per_image'] = new
                print(f"每张增强图片中最多使用的处理方式数量已设置为：{new}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '4':
            try:
                min_val = float(input("新亮度下限（建议 0.5-1.0）："))
                max_val = float(input("新亮水上限（建议 1.0-1.5）："))
                if min_val >= max_val:
                    print("下限必须小于上限")
                    continue
                config['brightness_factor'] = (min_val, max_val)
                print(f"亮度调整范围已设置为：{min_val}-{max_val}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '5':
            try:
                new = float(input("新噪声强度（建议 5.0-20.0）："))
                if new < 0:
                    print("数值不能为负")
                    continue
                config['noise_std'] = new
                print(f"噪声强度已设置为：{new}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '6':
            try:
                min_val = int(input("遮挡块大小下限（建议 10-50）："))
                max_val = int(input("遮挡块大小上限（建议 50-100）："))
                if min_val >= max_val:
                    print("下限必须小于上限")
                    continue
                config['occlusion_size'] = (min_val, max_val)
                print(f"遮挡块大小范围已设置为：{min_val}-{max_val}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '7':
            try:
                new = int(input("新遮挡块数量（建议 1-3）："))
                if new < 1:
                    print("数值必须大于 0")
                    continue
                config['occlusion_count'] = new
                print(f"遮挡块数量已设置为：{new}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '8':
            try:
                min_val = float(input("模糊强度下限（建议 0.1-0.3）："))
                max_val = float(input("模糊强度上限（建议 0.3-0.8）："))
                if min_val >= max_val:
                    print("下限必须小于上限")
                    continue
                config['blur_radius'] = (min_val, max_val)
                print(f"模糊强度范围已设置为：{min_val}-{max_val}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '9':
            try:
                min_val = int(input("旋转角度下限（建议 -15 到 15 之间）："))
                max_val = int(input("旋转角度上限（建议 -15 到 15 之间）："))
                if min_val >= max_val:
                    print("下限必须小于上限")
                    continue
                config['rotation_range'] = (min_val, max_val)
                print(f"旋转角度范围已设置为：{min_val}-{max_val}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '10':
            try:
                new = float(input("新的训练集比例（建议 0.6-0.8）："))
                if new < 0 or new > 1:
                    print("数值必须在 0 到 1 之间")
                    continue
                config['train_ratio'] = new
                print(f"训练集比例已设置为：{new}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '11':
            try:
                new = float(input("新的验证集比例（建议 0.1-0.2）："))
                if new < 0 or new > 1:
                    print("数值必须在 0 到 1 之间")
                    continue
                config['val_ratio'] = new
                print(f"验证集比例已设置为：{new}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == '12':
            try:
                new = float(input("新的测试集比例（建议 0.1-0.2）："))
                if new < 0 or new > 1:
                    print("数值必须在 0 到 1 之间")
                    continue
                config['test_ratio'] = new
                print(f"测试集比例已设置为：{new}")
            except ValueError:
                print("格式错误，未修改")

        elif choice == 'b':
            break

        else:
            print("未知选项")

def main_loop():
    global_out_dir = Path('augmented_dataset')
    global_out_dir.mkdir(exist_ok=True)

    print("=== 数据集增强和整理工具（循环模式）===")
    print("随时拖入数据集文件夹（包含 images 和 labels 子文件夹），或输入 q 退出程序\n")

    while True:
        user_path = input(">>> 拖入数据集文件夹后回车（输入 q 退出）：").strip('"')
        if user_path.lower() == 'q':
            print("拜拜~")
            break

        dataset_dir = scan_dataset(Path(user_path))
        if not dataset_dir:
            print("⚠ 未找到有效的数据集文件夹，重来\n")
            continue

        while True:
            cmd = show_menu()
            if cmd == 'q':
                print("放弃本次任务，回到顶层等待新文件\n")
                break
            elif cmd == 'set':
                adjust_config()
            elif cmd == '1':
                confirm = input("确定开始？输入 y 继续：").strip().lower()
                if confirm != 'y':
                    continue

                # 执行数据增强
                augmented_dir = global_out_dir / f"run_{int(time.time())}"
                process_dataset(dataset_dir, augmented_dir)

                # 准备YOLO数据集
                class_names = input("请输入类别名称，用逗号分隔（例如：cat,dog,person）：").strip()
                if not class_names:
                    print("类别名称不能为空")
                    continue

                yolo_dataset_dir = augmented_dir / "yolo_dataset"
                prepare_yolo_dataset(augmented_dir, yolo_dataset_dir, class_names.split(','))

                print(f"✓ 本次任务完成，YOLO数据集已保存到：{yolo_dataset_dir}\n")
            elif cmd == '2':
                confirm = input("确定开始仅执行数据增强？输入 y 继续：").strip().lower()
                if confirm != 'y':
                    continue

                # 执行数据增强
                augmented_dir = global_out_dir / f"run_{int(time.time())}"
                process_dataset(dataset_dir, augmented_dir)

                print(f"✓ 数据增强完成，增强数据已保存到：{augmented_dir}\n")
            elif cmd == '3':
                confirm = input("确定开始仅执行数据整理？输入 y 继续：").strip().lower()
                if confirm != 'y':
                    continue

                # 准备YOLO数据集
                class_names = input("请输入类别名称，用逗号分隔（例如：cat,dog,person）：").strip()
                if not class_names:
                    print("类别名称不能为空")
                    continue

                yolo_dataset_dir = global_out_dir / f"run_{int(time.time())}"
                prepare_yolo_dataset(dataset_dir, yolo_dataset_dir, class_names.split(','))

                print(f"✓ 数据整理完成，YOLO数据集已保存到：{yolo_dataset_dir}\n")
            else:
                print("未知指令")

if __name__ == '__main__':
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n强制退出")