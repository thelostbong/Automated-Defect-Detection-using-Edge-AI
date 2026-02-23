# ESPDL Evaluation and Quantization Project

This project provides tools and scripts to quantize and evaluate YOLO11 models specifically for the ESP32-P4 platform. It utilizes `esp-ppq` for quantization and `ultralytics` for model evaluation.

## Prerequisites

Before getting started, ensure you have the following installed on your system:
- **Python 3.10** (Strict requirement)
- **Anaconda** or **Miniconda** (Recommended for environment management)

## Installation Guide

Follow these steps to set up the project from scratch.

### 1. Clone the Repository
Download the project files to your local machine.

### 2. Create a Python Environment
Create a dedicated Python environment with Python 3.10 to ensure compatibility.

#### Using Conda (Recommended):
```bash
conda create -n espdl_env python=3.10 -y
conda activate espdl_env
```

#### Using venv (Standard Python):
```bash
python -m venv espdl_env
# Windows
.\espdl_env\Scripts\activate
# Linux/Mac
source espdl_env/bin/activate
```

### 3. Install Dependencies
Install the required libraries and packages.

**Core Dependencies:**
```bash
# Install Ultralytics 8.3 as required
pip install ultralytics==8.3

# Install PyTorch and Torchvision (Ensure these match your CUDA version if you plan to use GPU)
pip install torch torchvision

# Install ONNX tools
pip install onnx onnx-simplifier

# Install Image Processing tools
pip install Pillow
```

**ESP-PPQ Installation:**
This project relies on `esp-ppq` for quantization.
```bash
pip install esp-ppq
```
*Note: If `esp-ppq` is not available on PyPI, please install it from the official Espressif source or wheel file provided with your ESP-DL SDK.*

## Configuration

### Dataset Path
Before running the evaluation, you must update the dataset path in the configuration file.
1. Open `merged_val/data.yaml`.
2. Update the `train` and `val` paths to point to the absolute path of your `merged_val/images` directory on your local system.

**Example `merged_val/data.yaml`:**
```yaml
train: C:\path\to\your\project\merged_val\images
val: C:\path\to\your\project\merged_val\images
nc: 2
names: ['defect_colour', 'defect_missingpiece']
```

## Usage

### 1. Quantize the Model
Run the quantization script to convert and optimize your ONNX model for the ESP32-P4 target. This script uses calibration images from the `calib_images` directory.

```bash
python quantize_yolo11n.py
```
**Output:** This will generate `best.espdl` and updated ONNX files.

### 2. Evaluate the Model
Run the evaluation script to validate the quantized model's performance using the dataset defined in `merged_val/data.yaml`.

```bash
python yolo11n_eval.py
```
**Output:** Evaluation metrics effectively demonstrating the model's accuracy on the validation set.

## Project Structure

- **`quantize_yolo11n.py`**: Main script for performing model quantization using `esp-ppq`.
- **`yolo11n_eval.py`**: Script for evaluating the model using `ultralytics` validators.
- **`merge_datasets.py`**: Utility script for dataset management.
- **`best.pt`**: Original PyTorch model checkpoint.
- **`best.onnx`**: Exported ONNX model.
- **`best.espdl`**: Quantized model format for ESP32.
- **`merged_val/`**: Contains the validation dataset and `data.yaml` configuration.
- **`calib_images/`**: Directory containing images used for quantization calibration.
