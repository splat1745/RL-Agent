# RL-Agent

A real-time **Roblox PvP combat agent** that uses Computer Vision and Reinforcement Learning to play autonomously. It captures the game screen live, detects characters and throwables, and controls the game via DirectInput keyboard/mouse injection.

> **Platform: Windows only** — requires `dxcam`, `win32api`, `win32gui`, and DirectInput via `ctypes`.

---

## Features

- **Live Screen Capture**: DXcam captures the Roblox window at up to 60 FPS with minimal latency
- **Object Detection**: YOLOv11 and RF-DETR detect `character` and `throwable` classes
- **TwoStreamNetwork**: MobileNetV3 (full frame) + custom CNNs (crop + optical flow) + MLP (vector) fused through a stacked LSTM
- **PPO Reinforcement Learning**: Proximal Policy Optimization with truncated BPTT
- **Imitation Learning**: Record human gameplay → train behavior cloning policy
- **Continuous Trainer**: Watches for new trajectory files and trains PPO automatically
- **Config Wizard**: GUI tool for selecting health bar and UI ROIs
- **26-Action Space**: WASD, dashes, skills 1–4, jump, block, M1, and mouse turns

---

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/splat1745/RL-Agent.git
    cd RL-Agent
    ```

2. **Install core dependencies**:
    ```bash
    pip install torch torchvision ultralytics opencv-python numpy pyyaml pywin32 dxcam
    ```

3. **Install RF-DETR** (optional, for transformer-based detection):
    ```bash
    pip install rfdetr
    ```

4. **First-time setup** — run the config wizard to select the game window and health bar ROI:
    ```bash
    python main.py --setup
    ```
    This creates `config.json` which all scripts use.

---

## Full Workflow

### Step 1 — Train a Detection Model

**Capture raw frames for labeling:**
```bash
python collect_data.py
# Press [C] to capture, [A] for auto-capture, [Q] to quit
# Frames saved to datasets/raw_images/
```

**Sample frames to upload to Roboflow for labeling:**
```bash
python pipeline.py sample 300 session_01
```

**Auto-label existing frames with an existing model:**
```bash
python auto_label.py
# Edit VIDEO_DIR and OUTPUT_DIR in auto_label.py first
```

**Train the detection model:**
```bash
# RF-DETR small (recommended)
python pipeline.py train rfdetr s --batch 8 --grad-accum 2

# YOLO small
python pipeline.py train yolo s --batch 32

# Resume from checkpoint
python pipeline.py train rfdetr s --weights runs/last.pth
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_type` | — | `yolo`, `rfdetr`, or `rtdetr` |
| `model_size` | — | `n`, `s`, `m`, `l`, `x` |
| `--datasets` | auto-detect | Dataset paths to train on sequentially |
| `--epochs` | 70 | Epochs per dataset |
| `--batch` | 4 (rfdetr) / 32 (yolo) | Batch size |
| `--grad-accum` | 4 | Gradient accumulation (RF-DETR only) |
| `--device` | `0` | CUDA device index |

---

### Step 2 — Record Imitation Data & Train Policy

**Record human gameplay sessions:**
```bash
python main.py --model runs/best.pt
# At the preview window, press [I] to enter Imitation Mode
# Play normally. Sessions auto-saved as data/trajectories/traj_*.pkl
```

**Train the behavior cloning (imitation) policy:**
```bash
python train_imitation.py --data_dir data/trajectories --epochs 30 --lr 1e-4
```
Outputs: `models/imitation_policy.pth`

---

### Step 3 — PPO Fine-Tuning

**Option A — Continuous trainer** (recommended): runs in the background, picks up new trajectory files automatically:
```bash
python continuous_trainer.py --data_dir data/trajectories --model models/policy.pth
```

**Option B — Offline batch training:**
```bash
python train_offline.py
```

---

### Step 4 — Run the Live Agent

```bash
python main.py --model runs/best.pt
```

At the preview window:

| Key | Mode |
|-----|------|
| `S` | Start PPO agent |
| `T` | Start Temporal agent |
| `D` | Direct Policy (rule-based) |
| `I` | Imitation recording mode |
| `P` | Quit |

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--model`, `-m` | Path to detection weights (`.pt` YOLO or `.pth` RF-DETR) |
| `--setup`, `-s` | Force re-run the config/ROI wizard |

---

## Module Overview

| File | Purpose |
|------|---------|
| `main.py` | Live agent entry point — capture → detect → decide → act |
| `pipeline.py` | Detection model training CLI |
| `collect_data.py` | Frame capture tool for building datasets |
| `auto_label.py` | Auto-label frames with YOLO + optical-flow projectile detection |
| `train_imitation.py` | Behavior cloning from `.pkl` session recordings |
| `train_offline.py` | Offline PPO training on stored trajectories |
| `continuous_trainer.py` | Background PPO trainer watching for new `traj_*.pkl` files |
| `data_pipeline.py` | Video frame extraction + R3D-18 feature extraction |
| `rl/agent.py` | `PPOAgent` — select actions, PPO update with truncated BPTT |
| `rl/network.py` | `TwoStreamNetwork` — MobileNetV3 + crop CNN + flow CNN + LSTM |
| `rl/memory.py` | Experience replay buffer |
| `detection/inference.py` | Full perception stack — YOLO/RF-DETR + LK tracker + health OCR |
| `capture/capture.py` | DXcam threaded screen capture |
| `control/actions.py` | `ACTION_MAP` — 26-action space definition |
| `control/keyboard_mouse.py` | DirectInput `SendInput` keyboard + mouse injection |
| `control/input_listener.py` | Human input polling (for imitation recording) |
| `utils/config_wizard.py` | GUI ROI selector → `config.json` |
| `utils/reward.py` | Reward calculation (health, blocking, spacing, damage reflection) |
| `utils/state.py` | `StateManager` — health tracking, hit history, action history |

---

## Performance Tips

- **Detection training**: Use `--batch 16 --grad-accum 1` on an RTX 4090 to cut training time in half vs. defaults
- **GPU memory**: The dataset loader automatically offloads to CPU RAM when GPU hits 70% utilization during imitation training
- **PYTORCH_ALLOC_CONF**: `continuous_trainer.py` sets `expandable_segments:True` automatically to reduce fragmentation
- **Half precision**: Detection inference uses FP16 automatically when CUDA is available
