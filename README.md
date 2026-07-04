<h1 align="center">LEGO Space Shuttle Defect Detection on Edge AI</h1>

<p align="center">
  A two-stage YOLOv11n pipeline that spots colour and missing-piece defects in LEGO space shuttles, quantized to INT8 and running on-device on an ESP32-P4 — no cloud, no GPU at inference.
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/thelostbong/Automated-Defect-Detection-using-Edge-AI" alt="License">
  <img src="https://img.shields.io/github/last-commit/thelostbong/Automated-Defect-Detection-using-Edge-AI" alt="Last commit">
  <img src="https://img.shields.io/badge/model-YOLOv11n-informational" alt="Model">
  <img src="https://img.shields.io/badge/target-ESP32--P4-blue" alt="Target">
  <img src="https://img.shields.io/badge/ESP--IDF-v5.5.1-red" alt="ESP-IDF">
</p>

<p align="center">
  <a href="main_report.pdf">Full report (PDF)</a> ·
  <a href="#results">Results</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#citation">Cite</a>
</p>

<!-- HERO: replace with a GIF of the ESP32-P4 running the live UVC stream with red overlay boxes (see punch-list). Until then, INT8 predictions on the held-out set stand in. -->
<p align="center">
  <img src="phase2_model_development/defect_model_and_results/val_batch0_pred.jpg" alt="Defect predictions on held-out validation images" width="70%">
</p>
<p align="center"><em>Model 2 predictions on held-out validation images: colour and missing-piece defects boxed with confidence.</em></p>

## Overview

Manufacturing QA needs a defect check that runs at the line, not in a datacentre. The catch on a microcontroller is a threefold squeeze: a skewed dataset (defects are rarer than good parts), a tiny compute and memory budget, and the need to keep up with a moving product.

The approach here splits the job across two small YOLOv11n models instead of forcing one model to do everything. Stage 1 finds the shuttle (a single-class detector, tuned for recall so nothing slips past). Stage 2 classifies the defect on what stage 1 found (`defect_colour` vs `defect_missingpiece`). Splitting the task keeps each model's class balance manageable and each head small enough to quantize to INT8 and run on an ESP32-P4's NPU. The device presents itself to a host over USB as a standard UVC webcam and draws detection boxes straight onto the video, so there are no custom drivers to install to see it work.

The design decisions, the failure modes (including a preprocessing mismatch that flattened confidence to near zero until training and deployment pipelines were aligned), and the full inference breakdown live in the [report](main_report.pdf). This README is the trailer.

## Results

All figures below are read straight from the training logs (`results.csv`) and the INT8 evaluation run. Both models are YOLOv11n (2.6M params), trained from the `yolo11n.pt` checkpoint at 640px, then exported and quantized to 512px for the target.

**Stage 1 — shuttle detection (1 class, 15 epochs)**

| Metric | Value | Measured on |
|---|---|---|
| mAP@0.5 | 0.995 | 178-image held-out split |
| mAP@0.5:0.95 | 0.963 | 178-image held-out split |
| Precision | 0.991 | 178-image held-out split |
| Recall | 1.000 | 178-image held-out split |

Recall of 1.0 on the validation split is the point of stage 1: a missed shuttle here can never be classified downstream, so this stage is tuned to let nothing through.

**Stage 2 — defect classification (2 classes, 20 epochs)**

| Metric | Value | Measured on |
|---|---|---|
| mAP@0.5 | 0.984 | 178-image held-out split |
| mAP@0.5:0.95 | 0.682 | 178-image held-out split |
| Precision | 0.987 | 178-image held-out split |
| Recall | 0.917 | 178-image held-out split |

The gap between mAP@0.5 (0.984) and mAP@0.5:0.95 (0.682) is honest to report: the model finds defects reliably but its boxes are looser at strict IoU. Tightening localization is the main open item (see [Roadmap](#roadmap)).

**INT8 quantization (FP32 → INT8, ESP-PPQ, target ESP32-P4)**

| | FP32 | INT8 |
|---|---|---|
| mAP@0.5 | 0.984 | 0.984 |
| Model size (per model) | ~6 MB | ~1.5 MB |

The INT8 mAP was measured on the 11-image `merged_val` set committed under `phase3_model_quantization/step2_espdl_eval_final/merged_val/` — a small set, so read the "no measurable drop" result as "no drop on this set" rather than a guarantee across all inputs. Size drops ~75% per model.

<p align="center">
  <img src="phase3_model_quantization/step2_espdl_eval_final/runs/val_batch0_pred.jpg" alt="INT8 model predictions after quantization" width="70%">
</p>
<p align="center"><em>INT8 model predictions after quantization, running through the ESP-DL evaluation path.</em></p>

## How it works

```
Camera (OV5647, 1920x1080 RGB565)
        │
        ▼
Preprocess → 512x512 RGB, INT8
        │
        ▼
Stage 1: YOLOv11n INT8 — locate shuttle
        │
        ▼
Stage 2: YOLOv11n INT8 — classify defect
        │
        ▼
DFL decode + NMS  →  boxes mapped back to 1920x1080
        │
        ▼
Overlay drawn on the UVC video stream
```

On the ESP32-P4, video streams continuously at 30 FPS while inference runs on roughly every 60th frame (about one AI pass every two seconds). A full inference pass takes on the order of 4.7 s; running it on every frame is neither possible nor needed, since a defect on a static part does not change between frames. The frame-timing breakdown and memory/power figures are in the [report](main_report.pdf).

## Dataset

1,773 images collected with the ESP32-P4 + OV5647 rig in Phase 1, annotated in Roboflow with manual verification, in YOLOv11 format:

| Class | Images |
|---|---|
| good shuttle | 650 |
| colour defect | 320 |
| missing piece | 803 |

Split 90/10 into 1,595 training and 178 validation images. The class skew is exactly why the task is split across two models rather than trained as one three-class detector.

## Hardware & stack

- **Board:** ESP32-P4-Function-EV-Board (dual-core 400 MHz, NPU)
- **Camera:** OV5647 5MP MIPI-CSI, 1920×1080 @ 30fps
- **Host link:** USB UVC (appears as a standard webcam, MJPEG)
- **Firmware:** ESP-IDF v5.5.1, ESP-DL
- **Training / export:** Python 3.10, Ultralytics 8.3.0, PyTorch, ONNX (opset 13)
- **Quantization:** ESP-PPQ (INT8, calibration-based PTQ)

## Quickstart

The repo is organized as four phases. Each has its own README with the full setup; the short version:

```bash
git clone https://github.com/thelostbong/Automated-Defect-Detection-using-Edge-AI.git
cd Automated-Defect-Detection-using-Edge-AI
```

> [!NOTE]
> Weights, ONNX/ESPDL models, and the report are tracked with Git LFS. Run `git lfs install` before cloning, or the `.pt` / `.onnx` / `.espdl` / `.pdf` files come down as small pointer stubs.

**Train (Phase 2)** — Google Colab or a local GPU:

```bash
pip install ultralytics==8.3.0 torch torchvision
# open phase2_model_development/object_detection.ipynb
# Stage 1: YOLO("yolo11n.pt").train(data="good/data.yaml",  epochs=15, imgsz=640)
# Stage 2: YOLO("yolo11n.pt").train(data="data/train.yaml", epochs=20, imgsz=640)
```

**Export + quantize (Phase 3)** — Python 3.10:

```bash
cd phase3_model_quantization/step1_onnx_export
python -m venv venv && source venv/bin/activate      # Windows: .\venv\Scripts\activate
pip install ultralytics==8.3.0 torch onnx onnxsim onnxruntime
python export_onnx.py        # -> best.onnx (headless, 6 output tensors)

cd ../step2_espdl_eval_final
pip install esp-ppq
# point merged_val/data.yaml at your local merged_val/images path
python quantize_yolo11n.py    # -> best.espdl (INT8)
python yolo11n_eval.py        # evaluate INT8 vs FP32
```

**Deploy (Phase 4)** — ESP-IDF v5.5.1:

```bash
cd phase4_model_deployment
# unzip uvc_ai_final.zip, then:
idf.py set-target esp32p4
idf.py build
idf.py -p <PORT> flash monitor
```

Then open any UVC-capable viewer (e.g. PotPlayer), select the ESP32-P4 camera, and detections render on the stream as boxed labels like `defect_missingpiece: 85%`.

Key inference knobs live in `uvc_example.cpp`: `AI_PROCESS_EVERY_N_FRAMES` (60), `AI_SCORE_THRESHOLD` (0.10), `AI_NMS_THRESHOLD` (0.45).

## Repository structure

```
.
├── phase1_data_collection/        # ESP32-P4 web server for dataset capture (zipped)
├── phase2_model_development/
│   ├── object_detection.ipynb     # training notebook
│   ├── good_model_and_results/    # Stage 1 weights, curves, results.csv
│   └── defect_model_and_results/  # Stage 2 weights, curves, results.csv
├── phase3_model_quantization/
│   ├── step1_onnx_export/          # headless ONNX export (export_onnx.py)
│   └── step2_espdl_eval_final/     # ESP-PPQ INT8 quantization + eval, merged_val set
├── phase4_model_deployment/        # ESP-IDF UVC + inference firmware (zipped)
├── labelled_dataset/               # Roboflow YOLOv11 exports (zipped)
├── main_report.pdf                 # full write-up: design, failures, timing, references
└── LICENSE
```

## Roadmap

- Tighten stage-2 localization — the mAP@0.5:0.95 of 0.682 is the weakest headline number and the clearest thing to improve (attention blocks, harder augmentation).
- Re-evaluate INT8 on a proper held-out set, not the 11-image `merged_val`, to put a real number on quantization loss.
- Measure a true end-to-end accuracy: run both stages on a labeled end-to-end test set rather than reporting the two stage mAPs separately.
- Cut the ~4.7 s inference latency by profiling preprocessing and the NPU path.
- Balance the dataset with synthetic defect samples to reduce the colour/missing-piece skew.

## Citation

```bibtex
@techreport{mohammed2025legoedgeai,
  title  = {Automated LEGO Space Shuttle Defect Detection on Edge AI},
  author = {Mohammed, Nayeemuddin},
  year   = {2025},
  institution = {Deggendorf Institute of Technology},
  note   = {Quality Data Acquisition project. https://github.com/thelostbong/Automated-Defect-Detection-using-Edge-AI}
}
```

## License · Acknowledgements · Contact

Released under the MIT License — see [LICENSE](LICENSE).

Built at the Deggendorf Institute of Technology under Prof. Dr. Tim Weber. Thanks to Espressif (ESP32-P4, ESP-DL, ESP-PPQ), Ultralytics (YOLOv11), and Roboflow (annotation).

**Nayeemuddin Mohammed** — M.Sc. Applied AI for Digital Production Management, THD
[GitHub](https://github.com/thelostbong) · [LinkedIn](https://linkedin.com/in/nayeemuddin-mohammed-03/) · nayeemuddin.mohammed@th-deg.de
