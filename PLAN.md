# INSTRUCTIONAL PIPELINE: VIDEO-BASED ROBLOX RL AGENT

## GOAL
Build an RL agent that learns directly from recorded gameplay videos and understands:
- Timing-based attacks
- Projectiles
- Movement spacing
- Recovery frames
- Heavy-attack windups
- PvP dynamics

The pipeline requires no live keyboard input. All training comes from video data.

## 1. PREPARATION
**Hardware**
- Windows 10/11
- NVIDIA GPU (e.g., 128-core CUDA support)
- 64GB+ storage recommended
- Python 3.11
- PyTorch 2.x with CUDA support
- OpenCV, TorchVision, PyTorch Lightning

**Folder Structure**
```
/RobloxRL/ (Current Workspace)
    /data/
        /raw_videos/      (On T: Drive)
        /frames/          (On T: Drive)
        /preproc/         (On T: Drive)
        /labels/
        /poses/
        /combat_states/
        /observations/
    /models/
        /base_models/
        /student_model/
    /main_agent/
        /sensors/
        /policy/
        /rl/
        /utils/
```

## 2. VIDEO DATA COLLECTION
Record 3–5 hours of gameplay in 30–60 FPS using OBS.
Include:
- Normal fights
- Blocking & dodging
- Projectile spam
- Different maps, lighting, UI visible

Save videos to: `/data/raw_videos/` (Mapped to T: Drive)

## 3. FRAME EXTRACTION
Use OpenCV to extract frames directly from video.
Extract at 10–20 FPS.
Save as PNGs: `/data/frames/session_1/frame_00001.png`

## 4. PREPROCESSING (VIDEO-BASED)
Instead of frame-by-frame manual preprocessing:
Use 3D convolutional video models (e.g., R3D or TimeSformer) to automatically extract motion, appearance, and temporal features from video segments.

Steps:
1. Input N-frame clips (e.g., 16–32 frames each)
2. Extract feature embeddings capturing:
   - Movement patterns
   - Projectile motion
   - Pose changes
   - Action timing
3. Save embeddings as .npy: `/data/preproc/session_1_clip_0001.npy`

## 5. AUTO-LABELING
Generate pseudo-labels from video:
- **Projectile detection**: Use optical flow or video features to detect fast-moving objects. Cluster by color/motion.
- **Player / Enemy detection**: Use pretrained YOLOv8 or MMPose with humanoid fine-tuning.
- **Pose estimation**: Run video-based pose model on clips.

Save labels to `/data/labels/` and `/data/poses/`.

## 6. DERIVE COMBAT STATES
From pose + motion embeddings, generate combat states automatically:
- Windup
- Attack frame
- Recovery
- Idle / Dash / Projectile incoming

Save as: `/data/combat_states/session_1_clip_0001.json`

## 7. SENSOR FUSION → OBSERVATION VECTOR
Combine features per clip into a compact RL input.
Save vectors: `/data/observations/session_1_clip_0001.pkl`

## 8. BEHAVIOR CLONING (IMITATION LEARNING)
Align action proxies from video (Fast motion = attack, Freeze = idle).
Train supervised policy network.

## 9. MULTI-MODEL ENSEMBLE & DISTILLATION
Train 10–20 separate networks.
Distill into a single student model.

## 10. REINFORCEMENT LEARNING (VIDEO-ONLY)
Environment simulation: Replay video clips as environment.
RL Algorithm: PPO or SAC.
Rewards based on dodges, hits, spacing, etc.

## 11. CURRICULUM TRAINING
Stages: Static target -> Moving target -> Simple bots -> Hard bots -> PvP.

## 12. DOMAIN RANDOMIZATION
Randomize projectile speed, lighting, UI style.

## 13. FINAL AGENT SETUP
Run inference on new videos or integrate with live gameplay screen later.

## 14. OPTIONAL OPTIMIZATIONS
Quantize networks, run policy on CPU, use CUDA graphs.