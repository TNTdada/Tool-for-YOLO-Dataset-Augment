# Tool-for-YOLO-Dataset-Augment
**A Simple and Efficient Tool for YOLO Dataset Augmentation and Splitting**

[English](README_en.md) | [简体中文](README_zh.md)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-2.0-brightgreen.svg)

## 📖 Introduction
This project is an efficient dataset preprocessing tool specifically designed for YOLO object detection models. It provides 7 physical transformation augmentation schemes for existing labeled data (images and corresponding annotation files) through a Terminal User Interface (TUI). It supports automatic dataset splitting into **Train / Val / Test** structures and generates the `data.yaml` file, enabling the rapid augmentation and packaging of raw datasets for training.

---

## ✨ Features
The script supports fully automated batch processing of raw datasets with customizable parameters, including image resizing, augmentation quantity/methods, and dataset split ratios. Key augmentation algorithms include:

1. **Brightness Transformation**
   Randomly adjusts image brightness to simulate strong exposure under direct sunlight or low-light conditions (overcast/dusk), enhancing the model's adaptability to varying lighting.

2. **Gaussian Noise**
   Injects "salt-and-pepper" style noise into images to simulate artifacts from low-quality cameras, night vision equipment, or high ISO settings, improving model robustness against low-quality inputs.

3. **Random Occlusion**
   Randomly places opaque color blocks on the image to simulate partial obstructions. This forces the model to learn local features of the target. Users can limit the size of occlusion blocks to prevent complete target coverage.

4. **& 5. Horizontal/Vertical Flip**
   Flips images horizontally or vertically to increase the model's ability to recognize targets in various orientations.

6. **Geometric Rotation**
   Rotates the image around its center axis, with remaining areas filled in black. 
   > ⚠️ **Note**: Augmented bounding boxes are created by calculating the "Axis-Aligned Bounding Box (AABB)" of the original box after rotation. This may lead to a slight decrease in annotation precision; therefore, large-angle rotations are generally not recommended.

7. **Gaussian Blur**
   Blurs image details and edges to simulate focus failure or motion blur (motion trailing) caused by fast-moving targets.

---

## 🚀 Usage Guide

### 1. Data Preparation
Ensure the raw dataset is labeled, with images and label files placed in `images` and `labels` folders respectively.

![Directory Structure 1](assets/Snipaste_2026-03-07_12-12-28.jpeg)
![Directory Structure 2](assets/Snipaste_2026-03-07_12-13-01.jpeg)
![Directory Structure 3](assets/Snipaste_2026-03-07_12-13-29.jpeg)

### 2. Mounting the Dataset
Run the script and drag the raw dataset folder into the window (or enter the absolute path) and press Enter to enter the main interface, where scanning results will be displayed.

![Scanning Results](assets/Snipaste_2026-03-07_12-14-46.jpeg)

### 3. Parameter Adjustment
The main menu displays various parameter information. Input `set` to adjust each parameter.

![Main Menu Parameters](assets/Snipaste_2026-03-07_12-16-24.jpeg)
![Parameter Settings Interface](assets/Snipaste_2026-03-07_12-19-23.jpeg)

### 4. Label Mapping and Original Preview
Since a `data.yaml` file is generated during dataset splitting, and default label information only contains numeric indices, label mapping is required. If unsure of the mapping, use the **Original Preview (p-series commands)** to randomly view annotations and ensure the correct correspondence between label names and indices.

![Original Preview 1](assets/Snipaste_2026-03-07_12-27-28.jpeg)
![Original Preview 2](assets/Snipaste_2026-03-07_12-28-52.jpeg)
![Original Preview 3](assets/Snipaste_2026-03-07_12-29-37.jpeg)

### 5. Augmentation Preview
During parameter setup, use the **Augmentation Preview (a-series commands)** to randomly select images and apply current settings to compare "Before vs. After" effects. Adjust parameters further if the results do not meet expectations.

![Augmentation Comparison Preview](assets/Snipaste_2026-03-07_12-38-00.jpeg)

### 6. Execution and Results
Once configured, proceed to augment and split the entire dataset. Results will be saved in the `augmented_dataset` directory:
- `images` and `labels`: The full set of augmented data.
- `yolo_dataset`: The packaged dataset, ready for YOLO training.

![Output Results 1](assets/Snipaste_2026-03-07_13-01-09.jpeg)
![Output Results 2](assets/Snipaste_2026-03-07_13-01-52.jpeg)
![Output Results 3](assets/Snipaste_2026-03-07_13-02-18.jpeg)

---

## 📅 Upcoming Updates
- [ ] Persistent configuration memory to avoid re-setting parameters on every launch.
- [ ] Graphical User Interface (GUI) version based on PyQt6.

## 🙏 Acknowledgements / Credits
The demonstration images and test datasets used in this README are provided by **Roboflow Universe**:

- **Dataset Name**: Football Players Detection
- **Source Link**: [Roboflow Universe - Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- **Usage**: Solely for functional demonstration and effect preview.
