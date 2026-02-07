# SiliconSight
**Autonomous Semiconductor Wafer Defect Detection using Vision Transformers**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

**SiliconSight** is an autonomous computer vision system for **binary defect classification** in semiconductor wafer manufacturing  .

The system performs wafer-level **PASS / REJECT** decisions by analyzing **wafer map representations**, a critical quality-control step in modern fabrication pipelines  . The primary objective is **zero missed defects (false negatives)** under strict industrial constraints  .

Instead of traditional CNN-based approaches, SiliconSight leverages **Self-Supervised Vision Transformers (ViT)**—specifically **Meta's DINOv2**—to capture **global, non-linear spatial defect patterns** (rings, clusters, arcs, edge defects) that CNNs often struggle to model  .

This repository demonstrates a **production-oriented, end-to-end ML lifecycle**, including  :

- Domain-aware preprocessing  
- Extreme class-imbalance handling  
- Transformer fine-tuning  
- Manufacturing-focused evaluation  
- Robustness stress testing  
- Near real-time inference simulation  

All experiments are optimized for **Apple Silicon (MPS)** and **NVIDIA CUDA** hardware  .

---

## Table of Contents

- [Objective](#objective)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Virtual Environment Setup](#-virtual-environment-setup-recommended)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Performance Benchmark](#performance-benchmark)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [License](#license)

---

## Objective

Autonomous **binary classification** of semiconductor wafers  :

- **Clean (PASS)**
- **Defected (REJECT)**

with **zero tolerance for false negatives**  .

---

## Features

- **Vision Transformer backbone** (DINOv2) for global pattern recognition
- **100% accuracy** on balanced test set with perfect recall
- **~29ms inference latency** (~34 FPS) on Apple M4 MacBook Air
- **Robustness to Gaussian noise** (>99% accuracy up to σ ≈ 0.1–0.2)
- **Production-ready demo** with live wafer streaming simulation
- **Hardware acceleration** via MPS (Apple Silicon) and CUDA (NVIDIA)

---

## Prerequisites

- **Python**: 3.9 or higher  
- **Hardware**: 
  - Apple Silicon Mac (automatically uses MPS acceleration)  
  - NVIDIA GPU with CUDA support (optional)  
  - CPU fallback available
- **Operating System**: macOS, Linux, or Windows

---

## Installation

### Virtual Environment Setup (Recommended)

To ensure dependency isolation and reproducibility, this project should be run inside a Python virtual environment   . Creating a virtual environment prevents conflicts with system-wide packages and guarantees consistent behavior across machines  .

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/siliconsight.git
cd siliconsight
```

#### Step 2: Create a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
```

**Windows:**
```bash
python -m venv venv
```

#### Step 3: Activate the Virtual Environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

After activation, your terminal prompt should display `(venv)`  .

#### Step 4: Upgrade pip and Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

If this runs without errors and displays hardware acceleration status, the environment is set up correctly  .

#### Step 6: Deactivate the Environment (When Done)

```bash
deactivate
```

> ** Important Notes:**
> - Always activate your environment before running any scripts  
> - Add `venv/` to your `.gitignore` file to avoid committing environment files  
> - Apple Silicon users automatically use MPS acceleration if available  
> - NVIDIA users will use CUDA if properly installed  

---

## Dataset

### Source
- **File**: `Wafer_Map_Datasets.npz`
- **Total Samples**: ~38,000 wafers
- **Class Distribution**: ~95% Defected, <5% Clean  

### Structure

| Array  | Description                          |
|-------:|--------------------------------------|
| `arr_0`| Wafer maps (2D integer grids)        |
| `arr_1`| Defect annotations (multi-label)     |

Each wafer map is a sparse categorical matrix  :

| Value | Meaning        |
|------:|----------------|
| 0     | Background     |
| 1     | Normal Die     |
| 2     | Defect         |

---

## Repository Structure

```
DEEPTECH/
├── demo/
│   ├── assets/
│   │   ├── clean/           # Archived clean wafers
│   │   └── defected/        # Archived defective wafers
│   └── demo.py              # Production-line simulation
├── figures/                 # Generated visualizations
├── hackathon_model/         # Trained model checkpoints
├── results/                 # Evaluation outputs
├── src/
│   ├── dataset.py           # Dataset loader
│   ├── model.py             # DINOv2 + classification head
│   ├── preprocess.py        # RGB encoding & preprocessing
│   ├── train.py             # Training pipeline
│   ├── inference.py         # Single wafer inference
│   ├── evaluate.py          # Metrics computation
│   ├── generate_figures.py # Visualization generation
│   └── generate_results.py # Full evaluation report
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

## Usage

### Train the Model

```bash
python -m src.train
```

Trains DINOv2 on balanced wafer dataset  . Model checkpoints are saved to `./hackathon_model/`  .

### Evaluate Performance

```bash
python src/generate_results.py
```

Generates comprehensive evaluation metrics and confusion matrix  . Results are saved to `results/`  .

### Generate Figures

```bash
python src/generate_figures.py
```

Creates ROC curves, confidence histograms, and visualization plots  .

### Run Production Demo

```bash
python demo/demo.py
```

Launches live wafer streaming simulation with dual-view rendering  . Predictions are archived to `demo/assets/`  .

### Single Wafer Inference

```bash
python src/inference.py --random
```

Performs inference on a randomly selected wafer from the test set  .

---

## Model Architecture

### Backbone
- **Model**: `facebook/dinov2-base`  
- **Architecture**: Vision Transformer (ViT)  
- **Training Paradigm**: Self-Supervised Learning (DINO)  

### Why DINOv2?
DINOv2 excels at wafer map defect analysis because it  :
- Learns **global structural features**
- Avoids over-reliance on local texture
- Excels at **non-local pattern detection**

### Classification Head
- Linear layer on the `[CLS]` token  
- Output: 2 logits → Clean / Defected  

### Training Configuration

| Component        | Value          |
|------------------|----------------|
| Loss             | Cross-Entropy  |
| Optimizer        | AdamW          |
| Learning Rate    | 2e-5           |
| Batch Size       | 32             |
| Epochs           | 3              |
| Hardware         | MPS / CUDA     |

*A low learning rate preserves pretrained DINOv2 representations while adapting to domain-specific patterns*  .

---

## Evaluation

### Metrics (Balanced Test Set)

| Metric                                 | Value   |
| -------------------------------------- | ------- |
| Accuracy                               | 100.00% |
| Precision                              | 1.0000  |
| Recall (Sensitivity)                   | 1.0000  |
| F1 Score                               | 1.0000  |
| Matthews Correlation Coefficient (MCC) | 1.0000  |

**Primary metric**: MCC, due to robustness under class imbalance  .

### Diagnostic Outputs
- Confusion Matrix (zero misclassifications)  
- ROC Curve (AUC = 1.00)  
- Prediction confidence histograms  

All artifacts are stored in `results/`  .

---

## Performance Benchmark

**Hardware**: Apple M4 MacBook Air (MPS)  

| Metric            | Value          |
| ----------------- | -------------- |
| Inference Latency | ~29 ms / wafer |
| Throughput        | ~34 FPS        |

Suitable for **near real-time production screening**  .

### Robustness Stress Testing

Gaussian noise injection during inference  :
- Accuracy remained >99% up to σ ≈ 0.1–0.2  
- Confirms reliance on **structural patterns** rather than pixel artifacts  

---

## Limitations

- Operates on wafer map data (not raw microscopy)  
- Binary classification only  
- Dataset-specific preprocessing  
- Generalization may vary across fabs and processes  

---

## Acknowledgements

This project makes use of publicly available wafer map datasets from the following repositories:

- **WaferMap (MixedWM38 Dataset)**  
  Repository: [https://github.com/Junliangwangdhu/WaferMap](https://github.com/Junliangwangdhu/WaferMap)  
  Provides wafer map data with multiple single and mixed defect patterns.

These datasets were used **strictly for research and educational purposes**.  
All rights and credits belong to the original authors and contributors .

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed changes.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---



