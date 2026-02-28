# ROBLOX PvP RL AGENT — PROJECT PLAN

## GOAL

Build a real-time reinforcement learning agent for Roblox PvP combat that:
- Captures the game screen live via `mss` (Linux/Windows)
- Detects characters and throwables using YOLOv11 / RF-DETR
- Learns combat behavior (attack timing, spacing, blocking, combos, projectile dodging) via imitation learning + PPO
- Controls the game via `pynput` (Linux) / DirectInput (Windows) keyboard & mouse injection

> **Platform**: Linux (Ubuntu) and Windows — uses `mss` + `pynput` + `xdotool` on Linux; `dxcam` + `pywin32` + `ctypes` on Windows

---

## ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                       LIVE AGENT  (main.py)                     │
│                                                                 │
│  DXcam Screen Capture  ──►  YOLOv11/RF-DETR Detection          │
│         ▼                          ▼                            │
│  Optical Flow (LK)        Observation Vector Builder            │
│         ▼                          ▼                            │
│       ╔══════════════════════════════════════╗                  │
│       ║        TwoStreamNetwork              ║                  │
│       ║  MobileNetV3 (full 160×160)          ║                  │
│       ║  Custom CNN  (crop  128×128)         ║                  │
│       ║  Flow CNN    (flow   160×160)        ║                  │
│       ║  MLP Vector  (151-dim)               ║                  │
│       ║        ── Fusion MLP ──              ║                  │
│       ║     Stacked LSTM (2L × 768)          ║                  │
│       ║   Multi-Branch Action Head           ║                  │
│       ╚══════════════════════════════════════╝                  │
│                         ▼                                       │
│           26-Action Space → DirectInput SendInput               │
└─────────────────────────────────────────────────────────────────┘
```

### Action Space (26 Actions)

| ID | Action | ID | Action |
|----|--------|----|----|
| 0 | Idle | 13 | Skill 3 |
| 1–4 | WASD | 14 | Skill 4 |
| 5–6 | Turn L/R | 15 | R+2 Combo |
| 7–10 | Directional Dashes | 16 | G |
| 11–12 | Skill 1/2 | 17 | Jump (Space) |
| 18 | Block (F) | 19 | M1 Attack |
| 20–25 | Mouse micro/small/large turns | | |

---

## FOLDER STRUCTURE

```
RL-Agent/
├── main.py                  # Live agent entry point (PPO / Imitation / Direct)
├── pipeline.py              # Detection model training CLI (YOLO, RF-DETR)
├── collect_data.py          # Manual/auto frame capture tool
├── auto_label.py            # Auto-label frames with YOLO + optical flow
├── train_imitation.py       # Behavior cloning from .pkl session recordings
├── train_offline.py         # Offline PPO training on stored trajectories
├── continuous_trainer.py    # Watches for new traj_*.pkl and trains PPO live
├── data_pipeline.py         # Video frame extraction + R3D-18 feature extraction
├── extract_frames.py        # Standalone frame extraction utility
├── capture/
│   └── capture.py           # Cross-platform screen capture (mss/Linux, dxcam/Windows)
├── detection/
│   └── inference.py         # YOLOv11/RF-DETR + LK tracker + health OCR (2400+ lines)
├── control/
│   ├── actions.py           # ACTION_MAP (26 actions)
│   ├── keyboard_mouse.py    # Cross-platform keyboard/mouse (pynput/Linux, DirectInput/Windows)
│   ├── input_listener.py    # Cross-platform human input polling (pynput/Linux, win32api/Windows)
│   └── policy.py            # DirectPolicy (rule-based fallback)
├── rl/
│   ├── agent.py             # PPOAgent with truncated BPTT
│   ├── network.py           # TwoStreamNetwork (MobileNetV3 + crop CNN + LSTM)
│   ├── agent_v2.py          # Experimental agent v2
│   ├── network_v2.py        # Experimental network v2
│   ├── temporal_agent.py    # Temporal-focused agent variant
│   ├── temporal_network.py  # Temporal network variant
│   ├── muzero_agent.py      # MuZero agent variant
│   ├── muzero_network.py    # MuZero network
│   ├── memory.py            # Experience replay buffer
│   ├── policy.py            # RL policy helpers
│   └── pretrain.py          # Pretraining utilities
├── utils/
│   ├── config_wizard.py     # GUI health-bar ROI selection → config.json
│   ├── reward.py            # Reward shaping (health, blocking, spacing)
│   ├── state.py             # StateManager (health delta, hit history)
│   ├── visualization.py     # Detection overlay drawing
│   ├── feature_builder.py   # Feature construction helpers
│   └── video_sampler.py     # Video sampling utility
├── main_agent/
│   ├── run_agent.py         # Advanced multi-sensor agent runner
│   └── sensors/
│       ├── object_detection/detector.py
│       ├── pose_estimation/pose_estimator.py
│       └── combat_state/inference.py
└── data/                    # Auto-created; stores frames, labels, trajectories
    ├── frames/
    ├── labels/
    └── trajectories/        # traj_*.pkl from live sessions
```

---

## TRAINING PIPELINE

### Stage 1 — Detection Model

1. **Capture raw frames**: `python collect_data.py`
2. **Label in Roboflow** (classes: `character`, `throwable`)
3. **Auto-label unlabeled frames**: `python auto_label.py`
4. **Train detection model**: `python pipeline.py train rfdetr s`
5. Best weights auto-saved to `runs/`

### Stage 2 — Imitation Learning (Behavior Cloning)

1. Launch `main.py` and press `[I]` in the preview window to enter Imitation Mode
2. Play normally; sessions auto-saved as `data/trajectories/traj_*.pkl`
3. Train: `python train_imitation.py --data_dir data/trajectories --epochs 30`
4. Outputs: `models/imitation_policy.pth`

### Stage 3 — PPO Fine-Tuning

1. Run the live agent with the imitation policy; new `traj_*.pkl` files are written automatically
2. `continuous_trainer.py` watches the trajectories folder and trains on each new file
3. Or run offline: `python train_offline.py`

---

## NETWORK DETAILS

**TwoStreamNetwork** (`rl/network.py`)

| Stream | Input Shape | Architecture | Output Dim |
|--------|-------------|-------------|------------|
| Full Frame | `[B, 16, 160, 160]` (4-frame RGBD stack) | MobileNetV3-Large + 2-layer MLP | 1024 |
| Fine Crop | `[B, 16, 128, 128]` | 4-layer CNN + AdaptiveAvgPool | 512 |
| Optical Flow | `[B, 2, 160, 160]` | 3-layer CNN + AdaptiveAvgPool | 128 |
| Vector | `[B, 151]` | 2-layer MLP | 128 |
| Fusion | concat → 1792 | 3-layer MLP | 1024 |
| Temporal | sequence | Stacked LSTM (2L × 768) | 768 |
| Head | 768 | Combo-intention MLP → actor + critic | action_dim / 1 |

---

## REWARD DESIGN

- **Chase**: negative Δdist to enemy × 2.0
- **Target lock**: bonus for keeping enemy in screen center
- **Block**: +2.0 for successful block (enemy attacking + no health loss)
- **Action penalties**: penalize retreating or turning away when in range
- **Delayed damage reflection** (1.618 s golden-ratio window): penalize moves followed by taking damage
- **Health tracking** via `StateManager` (`utils/state.py`)

---

## COMPONENT STATUS

| Component | Status |
|-----------|--------|
| Screen capture (mss/dxcam cross-platform) | Complete |
| Detection (YOLOv11 / RF-DETR) | Complete |
| TwoStreamNetwork + PPO | Complete |
| Imitation learning pipeline | Complete |
| Continuous PPO trainer | Complete |
| Config wizard (health ROI) | Complete |
| Auto-labeling (YOLO + optical flow) | Complete |
| Linux / cross-platform input & capture | Complete |
| Advanced sensor suite (main_agent/) | Partial |
| MuZero / temporal variants | Experimental |

---

## FUTURE IMPROVEMENTS

- Curriculum training: static target → moving → bots → PvP
- Domain randomization: lighting, UI skin, projectile speed
- Model quantization (FP8/INT8) for inference speedup
- Ensemble distillation: train multiple policies, distill into one student model
- Integrate pose estimation into the live observation vector
