# RL-Agent

An automated farming agent using Computer Vision and Reinforcement Learning. This project utilizes object detection models (YOLOv11, RF-DETR) to identify game elements and control inputs.

## Features

*   **Object Detection**: Supports Ultralytics YOLOv11 and Roboflow RF-DETR.
*   **Sequential Training**: Train on multiple datasets in sequence (Dataset 1 -> Dataset 2 -> ...), automatically carrying over the best weights.
*   **Auto-Labeling**: Inference script to generate labels for new frames using a trained model.
*   **Smart Pipeline**: Automatically converts YOLO datasets to COCO format for RF-DETR training.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/splat1745/RL-Agent.git
    cd RL-Agent
    ```

2.  **Install Dependencies**:
    ```bash
    pip install ultralytics opencv-python numpy pyyaml torch
    ```

3.  **Install RF-DETR (Optional, for Transformer models)**:
    ```bash
    pip install rfdetr
    ```

## Usage

The `pipeline.py` script is the main entry point for dataset management and training.

### 1. Sampling Frames
Select random frames from your recordings for manual labeling.
```bash
python pipeline.py sample [count] [subdir]
# Example: Sample 300 frames from session_01
python pipeline.py sample 300 session_01
```

### 2. Training
Train a model. The script supports both YOLO and RF-DETR.

**Arguments:**
*   `[model_type]`: `yolo` or `rfdetr` (or `rtdetr` for Ultralytics version).
*   `[model_size]`: `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (extra large).
*   `--datasets`: List of dataset paths to train on sequentially.
*   `--weights`: Path to initial weights (optional).
*   `--epochs`: Number of epochs per dataset (default: 70).
*   `--batch`: Batch size (default: 4 for RF-DETR/RT-DETR, 32 for YOLO).
*   `--workers`: Number of dataloader workers (default: 4).
*   `--grad-accum`: Gradient accumulation steps (RF-DETR only, default: 4).
*   `--device`: Device to run on (e.g., `0`, `0,1`, `cuda`).

### Performance Tuning
To halve training time, you need to maximize GPU usage.
*   **Increase Batch Size**: Use `--batch 16` or `--batch 32` (depending on VRAM).
*   **Decrease Gradient Accumulation**: If you increase batch size, you can decrease `--grad-accum` to keep the effective batch size (Batch * Accum) constant, or keep it high for better convergence.
*   **Workers**: Default is now 8. If your CPU is strong, this ensures the GPU doesn't wait for data.

**Examples:**

*   **High Performance (RTX 4090/5090):**
    Doubles the workload per step compared to default.
    ```bash
    python pipeline.py train rfdetr s --batch 16 --grad-accum 1
    ```

*   **Max Performance (If VRAM allows):**
    ```bash
    python pipeline.py train rfdetr s --batch 32 --grad-accum 1
    ```

*   **Train RF-DETR Small on specific datasets:**
    ```bash
    python pipeline.py train rfdetr s --datasets "T:\Data\ds1" "T:\Data\ds2"
    ```

*   **Train YOLO Small (Auto-detects datasets in T:\Auto-Farmer-Data):**
    ```bash
    python pipeline.py train yolo s
    ```

*   **Resume Training:**
    ```bash
    python pipeline.py train rfdetr s --weights "runs/last.pth" --datasets "T:\Data\new_ds"
    ```

### 3. Inference & Auto-Labeling
Run inference on a folder of images and generate YOLO-format labels.
```bash
python pipeline.py infer <model_path> <frames_dir>
# Example
python pipeline.py infer runs/train/weights/best.pt data/frames/session_01
```

## Directory Structure

*   `pipeline.py`: Main CLI tool for training and dataset management.
*   `main.py`: Real-time RL agent.
*   `detection/`: Inference logic.
*   `control/`: Keyboard/Mouse control logic.

## Running the RL Agent

The `main.py` script runs the real-time RL agent with object detection.

**Arguments:**
*   `--model`, `-m`: Path to detection model (`.pt` for YOLO, `.pth` for RF-DETR).
*   `--setup`, `-s`: Force run the setup wizard.

**Examples:**

*   **Run with default model:**
    ```bash
    python main.py
    ```

*   **Run with custom YOLO model:**
    ```bash
    python main.py --model "path/to/best.pt"
    ```

*   **Run with RF-DETR model:**
    ```bash
    python main.py --model "D:\Auto-Farmer-Data\runs\seq_train_rfdetr_s\dataset5_run\checkpoint_best_ema.pth"
    ```

*   **Force setup wizard:**
    ```bash
    python main.py --setup
    ```
