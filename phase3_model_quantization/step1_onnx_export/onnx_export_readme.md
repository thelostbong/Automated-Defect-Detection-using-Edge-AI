# YOLO ONNX Export for ESP32 - Setup and Implementation Guide

This guide describes how to set up the environment and run the project from scratch. This project exports a YOLO model (specifically `best.pt`) to an ONNX format optimized for ESP32P4 device (removing the default detection head and modifying attention mechanisms), and verifies the exported model.

## 1. Prerequisites (Python Environment)

To ensure compatibility, please use **Python 3.10**.

### Step 1.1: Install Python 3.10
If you do not have Python 3.10 installed, download and install it from [python.org](https://www.python.org/downloads/).

### Step 1.2: Create a Virtual Environment
Open your terminal (Command Prompt or PowerShell) and navigate to the project directory.

Run the following command to create a virtual environment named `venv`:

```bash
python -m venv venv
```

### Step 1.3: Activate the Virtual Environment
Activate the environment to isolate your dependencies.

*   **Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\Activate
    ```
*   **Windows (Command Prompt):**
    ```cmd
    venv\Scripts\activate.bat
    ```
*   **Linux/macOS:**
    ```bash
    source venv/bin/activate
    ```

---

## 2. Install Project Dependencies

Once the environment is active, install the required libraries. This project requires `ultralytics` version 8.3 and several ONNX-related tools.

Run the following command:

```bash
pip install ultralytics==8.3.0 torch onnx>=1.14.0 onnxsim onnxruntime numpy
```

### Dependency List
*   **Python**: 3.10
*   **ultralytics**: 8.3.0 (Required for YOLO model handling)
*   **torch**: (Required for PyTorch model operations)
*   **onnx**: >= 1.14.0 (Standard Open Neural Network Exchange)
*   **onnxsim**: (Used to simplify the exported ONNX graph)
*   **onnxruntime**: (Used to valid and infer the model)
*   **numpy**: (Used for data manipulation in verification)

---

## 3. Project Files

Ensure your project directory contains the following files:

*   `best.pt`: The source PyTorch model weight file you wish to export.
*   `export_onnx.py`: The script that handles the conversion from `.pt` to `.onnx` with ESP32-specific modifications.
*   `analyse_onnx.py`: The script used to verify the structure and functionality of the exported `.onnx` file.

---

## 4. Usage Instructions

### Step 4.1: Export the Model
To convert the `best.pt` model to ONNX format, run the export script:

```bash
python export_onnx.py
```

**What this does:**
*   Loads `best.pt` using `ultralytics` 8.3.0.
*   Applies patches to `Detect` and `Attention` modules for ESP32 compatibility.
*   Exports the model to `best.onnx`.
*   Simplifies the ONNX graph using `onnxsim`.

### Step 4.2: Verify the Exported Model
After the export is complete, verify the generated `best.onnx` file to ensure it is valid and functional:

```bash
python analyse_onnx.py
```

**What this does:**
*   Loads `best.onnx`.
*   Checks the model schema and structure.
*   Prints the Input and Output tensor shapes.
*   Runs a dummy inference with random data to confirm the model executes without errors.

---
*   **Missing `best.pt`**: Ensure your model file is named `best.pt` and is located in the root of the directory, or update the filename in `export_onnx.py` (line 141).

The generated output is best.onnx
