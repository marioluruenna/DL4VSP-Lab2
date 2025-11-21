# MEGA for Video Object Detection – DL4VSP Lab 2 (Session 1)

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This repository is a **course repository** based on the original MEGA implementation:

> Original GitHub: https://github.com/Scalsol/mega.pytorch  
> Paper: **Memory Enhanced Global-Local Aggregation for Video Object Detection**  
> Yihong Chen, Yue Cao, Han Hu, Liwei Wang – CVPR 2020

This repo is used in the course **Deep Learning for Video Signal Processing (DL4VSP)** to complete **Lab 2 – Session 1**.

In this lab we focus **only on inference**, not training. The goal is:

- To set up and build `mega.pytorch` in a clean Python environment.
- To run the **demo on an image folder** corresponding to a dog video (provided as frames).
- To compare:
  - **BASE**: single-frame baseline (ResNet-101).
  - **MEGA**: memory enhanced global-local aggregation (ResNet-101).
- To document all requirements and installation steps so that **almost anyone can reproduce the results**.

All necessary fixes (Apex removal, Pillow / OpenCV compatibility, `cv2.putText` bug) are **already applied** in this repository.

---

## 1. Repository structure

After cloning this repository, the relevant structure is:

DL4VSP-Lab2/  
├── README.md  
├── image_folder.zip  
├── configs/  
├── demo/  
├── mega_core/  
├── tools/  
├── tests/ (optional)  
├── requirements.txt  
├── setup.py  
└── ... (other original MEGA files)

Additional directories created at runtime:

- `data/` → created by the user; will contain `data/image_folder/` after unzipping `image_folder.zip`.
- `outputs/` → created by the user when running the demos; stores visualized results (`base_dog`, `mega_dog`).

---

## 2. Requirements (Task 1.2 – environment and setup)

The code has been tested with the following environment:

- Python **3.7**
- PyTorch **1.2.0**
- torchvision **0.4.0**
- CUDA-capable GPU (for the GPU build), but CPU-only is also possible with the CPU build of PyTorch.
- `numpy == 1.21.6`
- `pillow < 7`
- `opencv-python` (4.x)

Important notes:

- You do **NOT** need to install NVIDIA Apex (`apex`).  
  This repository includes a small dummy AMP implementation so that the code runs without Apex.
- All patches required to avoid:
  - Apex import errors,
  - `PILLOW_VERSION` errors with Pillow and torchvision,
  - `cv2.putText` type errors,
  are already included in the code.
- You will need to install the BASE and MEGA .pth files from the original repo.

The instructions below assume you have `conda` available (e.g., Anaconda or Miniconda) on a Linux machine.

---

## 3. Environment setup (conda)

All commands below assume that the root of this repo is:

DL4VSP-Lab2/

(i.e., the directory that contains this `README.md`, `configs/`, `demo/`, etc.)

### 3.1 Create and activate the environment

Create a fresh environment named `mega_lab2` with Python 3.7:

    conda create -n mega_lab2 python=3.7 -y
    conda activate mega_lab2

### 3.2 Install PyTorch 1.2.0 and torchvision 0.4.0

Choose one **depending on your hardware**.

GPU example (with CUDA 10.0):

    conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch -y

CPU-only example:

    conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch -y

Quick sanity check:

    python -c "import torch; print('torch', torch.__version__)"
    # Expected output: torch 1.2.0

If the version is different, adjust your installation to match `1.2.0`.

### 3.3 Install Python dependencies from the project

From the root of the repo:

    cd ~/DL4VSP-Lab2
    pip install -r requirements.txt

This installs the Python dependencies listed by the original project.

### 3.4 Fix numpy / pillow / opencv versions

To avoid common runtime errors (NumPy/OpenCV API mismatch and Pillow/torchvision compatibility issues), explicitly install:

    pip install "numpy==1.21.6" "pillow<7" opencv-python

Then verify that imports work correctly:

    python -c "import numpy as np; print('numpy', np.__version__)"
    python -c "import cv2; print('cv2', cv2.__version__)"

Both commands should run without throwing errors.

### 3.5 Build MEGA C++/CUDA extensions

The MEGA project contains custom C++/CUDA extensions (under `mega_core`) which must be compiled.

From the root of the repo:

    cd ~/DL4VSP-Lab2
    python setup.py build_ext install

This will:

- Compile the extensions.
- Install them into the current `mega_lab2` environment.

If this step fails, check that your compiler and CUDA toolkit (if using GPU) are correctly set up.

---

## 4. Files required for Session 1

For **Lab 2 – Session 1**, you need the following items:

1. BASE checkpoint (ResNet-101)  
   File: `R_101.pth`  
   → Single-frame baseline model.

2. MEGA checkpoint (ResNet-101)  
   File: `MEGA_R_101.pth`  
   → Full MEGA model.

3. Dog image folder (frames)  
   Provided in this repo as a zip: `image_folder.zip`, which contains a folder `image_folder/` with frames:
   - `000000.JPEG`
   - `000001.JPEG`
   - `000002.JPEG`
   - …

In this repository, **the compressed dog image folder is already tracked** and placed in the repo root:

- `DL4VSP-Lab2/image_folder.zip`

If for some reason these files are missing (e.g., due to size limits or a shallow clone), they should be downloaded from the course platform and placed in the same locations.

### 4.1 Check that the checkpoints are present

From the repo root:

    cd ~/DL4VSP-Lab2
    ls *.pth

Expected:

    R_101.pth  MEGA_R_101.pth

If they are missing, copy/download them into the repo root and re-run the command.

### 4.2 Unzip the dog image folder into `data/`

This repository includes `image_folder.zip` already in the root. To create the directory `data/image_folder/` with all frames:

    cd ~/DL4VSP-Lab2
    mkdir -p data
    cd data
    unzip ../image_folder.zip
    ls image_folder | head
    # 000000.JPEG
    # 000001.JPEG
    # 000002.JPEG
    # ...

## 5. Running the demos (BASE and MEGA) on the dog sequence

The repository includes a general demo script:

- `demo/demo.py`

For this lab we only use the **image folder** mode (no video input), applied to the dog frames.

In all commands below, start from the repo root:

    cd ~/DL4VSP-Lab2
    conda activate mega_lab2

### 5.1 BASE – inference on `data/image_folder`

1. Create the output directory for BASE (inside `outputs/`)

```bash
   mkdir -p outputs/base_dog
```
2. Run the demo using the BASE model:
```bash
    python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth \
      --suffix ".JPEG" \
      --visualize-path data/image_folder \
      --output-folder outputs/base_dog
  ```
Where:

- `base` → selects the single-frame baseline method.
- `configs/vid_R_101_C4_1x.yaml` → BASE configuration (ResNet-101 backbone).
- `R_101.pth` → BASE checkpoint file (tracked in the repo root).
- `--suffix ".JPEG"` → extension of the input images (`000000.JPEG`, `000001.JPEG`, …).
- `--visualize-path data/image_folder` → folder with input frames.
- `--output-folder outputs/base_dog` → folder where the visualized outputs will be stored.

If everything is configured correctly, you should see a progress bar such as:

    100%|██████████████████████████████████████| 20/20 [00:02,  9.5it/s]

Then you can inspect the outputs:

    ls outputs/base_dog | head
    # 000000.jpg
    # 000001.jpg
    # 000002.jpg
    # ...

Each of these is a dog frame with **BASE detections** (bounding boxes, class labels) drawn on the image.

### 5.2 MEGA – inference on `data/image_folder`

1. Create the output directory for MEGA (inside `outputs/`):
```bash
    `mkdir -p outputs/mega_dog`
```
2. Run the demo using the MEGA model:
```bash
    `python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth \
        --suffix ".JPEG" \
        --visualize-path data/image_folder \
        --output-folder outputs/mega_dog`
```
Where:

- `mega` → selects the MEGA method.
- `configs/MEGA/vid_R_101_C4_MEGA_1x.yaml` → MEGA configuration (ResNet-101 backbone).
- `MEGA_R_101.pth` → MEGA checkpoint file (tracked in the repo root).
- The other arguments are as in the BASE case.

After it finishes, inspect:

    ls outputs/mega_dog | head
    # 000000.jpg
    # 000001.jpg
    # 000002.jpg
    # ...

Now you have:

- `outputs/base_dog/` → BASE visualizations.
- `outputs/mega_dog/` → MEGA visualizations.

These two sets of outputs are designed to:

- Compare temporal stability of detections between BASE and MEGA.
- Analyze missed vs extra detections.
- Discuss the impact of temporal aggregation (MEGA) vs single-frame processing (BASE).
