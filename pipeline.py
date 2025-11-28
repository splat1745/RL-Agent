"""
Auto-Farmer Pipeline Commands:

1. Sample Frames for Labeling:
   python pipeline.py sample [count] [subdir]
   Example: python pipeline.py sample 300 session_01

2. Train Model (Combined Dataset):
   python pipeline.py train "path/to/your/weights.pt"

3. Run Inference & Auto-Label:
   python pipeline.py infer <model_path> <frames_dir>
   Example: python pipeline.py infer runs/train/weights/best.pt data/frames/session_01

4. Main Agent:
   python main.py [model_path] [--setup]
   Example: python main.py runs/train/weights/best.pt --setup
"""
import os
import shutil
import random
import glob
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from collections import defaultdict
import sys

# --- CONFIGURATION ---
DATA_ROOT = r"T:\Auto-Farmer-Data"
FRAMES_DIR = os.path.join(DATA_ROOT, "frames")
UPLOAD_DIR = os.path.join(DATA_ROOT, "roboflow_upload")
DATASET_DIR = os.path.join(DATA_ROOT, "dataset") # Expects 'train/images', 'valid/images' here
AUTO_LABELS_DIR = os.path.join(DATA_ROOT, "auto_labels")

CLASSES = [
    "character", "throwable"
]

def preprocess_dataset(dataset_root, target_classes):
    """
    Normalizes a dataset:
    1. Reads its data.yaml to understand current class mapping.
    2. Converts all labels to the target_classes mapping.
    3. Removes 'obstacle' class (or any class not in target_classes).
    4. Converts polygons to bounding boxes.
    """
    marker_path = os.path.join(dataset_root, "processed_v5.marker")
    if os.path.exists(marker_path):
        print(f"Dataset at {dataset_root} already processed. Skipping.")
        return

    yaml_path = os.path.join(dataset_root, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"Warning: {yaml_path} not found. Cannot map classes correctly.")
        return

    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    source_names = data_cfg.get('names', [])
    if isinstance(source_names, dict):
        source_map = source_names
    else:
        source_map = {i: n for i, n in enumerate(source_names)}

    print(f"Processing {dataset_root}...")
    print(f"Source classes: {source_map}")
    
    # Build mapping from Source Index -> Target Index
    index_map = {}
    for src_idx, src_name in source_map.items():
        name = src_name.lower().strip()
        # Aliases
        if name == "item": name = "throwable"
        if name == "building": name = "obstacle"
        if name == "player": name = "character"
        if name == "enemy": name = "character"
        
        if name in target_classes:
            index_map[src_idx] = target_classes.index(name)
        else:
            print(f"Dropping class '{name}' (Index {src_idx})")
            index_map[src_idx] = None 

    # Process all txt files
    label_files = glob.glob(os.path.join(dataset_root, "**", "*.txt"), recursive=True)
    
    for lf in label_files:
        if "classes.txt" in lf: continue
        if "data.yaml" in lf: continue
        
        with open(lf, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            try:
                src_idx = int(parts[0])
            except ValueError:
                continue
                
            target_idx = index_map.get(src_idx)
            
            if target_idx is None:
                continue # Drop
            
            # Convert Polygon to Box if needed
            coords = [float(x) for x in parts[1:]]
            
            # YOLO Box: x_c, y_c, w, h (4 values)
            # YOLO Polygon: x1, y1, x2, y2 ... (>= 6 values usually)
            
            if len(coords) > 4:
                # Polygon -> Box
                xs = coords[0::2]
                ys = coords[1::2]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                w = max_x - min_x
                h = max_y - min_y
                xc = min_x + w/2
                yc = min_y + h/2
                
                # Clamp to 0-1 just in case
                xc = max(0, min(1, xc))
                yc = max(0, min(1, yc))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                new_line = f"{target_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                new_lines.append(new_line)
            else:
                # Box -> Box (just update index)
                new_line = f"{target_idx} " + " ".join(parts[1:]) + "\n"
                new_lines.append(new_line)
        
        # Always write back to ensure consistency
        with open(lf, 'w') as f:
            f.writelines(new_lines)

    # Update data.yaml to reflect new classes
    data_cfg['names'] = target_classes
    data_cfg['nc'] = len(target_classes)
    with open(yaml_path, 'w') as f:
        yaml.dump(data_cfg, f, sort_keys=False)

    # Create marker
    with open(marker_path, 'w') as f:
        f.write("processed_v5")
    print("Processing complete.")

# --- 1. SAMPLING ---
def sample_frames(n=200, subdir=None):
    """Selects N random frames for manual labeling."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    search_path = FRAMES_DIR
    if subdir:
        search_path = os.path.join(FRAMES_DIR, subdir)
        
    print(f"Searching for frames in: {search_path}")
    
    # Find all pngs recursively
    all_frames = glob.glob(os.path.join(search_path, "**", "*.png"), recursive=True)
    if not all_frames:
        print(f"No frames found in {search_path}")
        return

    # Filter out duplicates (already in upload or dataset)
    existing_files = set()
    if os.path.exists(UPLOAD_DIR):
        existing_files.update(os.listdir(UPLOAD_DIR))
    
    train_imgs = os.path.join(DATASET_DIR, "train", "images")
    if os.path.exists(train_imgs):
        existing_files.update(os.listdir(train_imgs))
        
    valid_imgs = os.path.join(DATASET_DIR, "valid", "images")
    if os.path.exists(valid_imgs):
        existing_files.update(os.listdir(valid_imgs))
        
    available_frames = [f for f in all_frames if os.path.basename(f) not in existing_files]
    print(f"Found {len(all_frames)} total frames. {len(available_frames)} are new (not in dataset/upload).")

    if not available_frames:
        print("No new frames to sample.")
        return

    selected = random.sample(available_frames, min(n, len(available_frames)))
    
    print(f"Copying {len(selected)} frames to {UPLOAD_DIR}...")
    for src in selected:
        dst = os.path.join(UPLOAD_DIR, os.path.basename(src))
        shutil.copy2(src, dst)
    print(f"Done. Upload files in '{UPLOAD_DIR}' to Roboflow.")

# --- 2. TRAINING ---
def run_scheduled_training(model_source, data_yaml, project_dir, base_name):
    """
    Executes the 3-phase scheduled augmentation training strategy.
    Returns the path to the best weights from the final phase.
    """
    # Common settings
    common_args = {
        'project': project_dir,
        'workers': 0,
        'device': 0,
        'amp': True,
        'cos_lr': True,
        'optimizer': 'AdamW',
        'patience': 0, # Disable early stopping
        'imgsz': 640,
        'batch': 32,
    }

    # --- Phase 1: Light Augs (0-70 epochs) ---
    print(f"\n--- Starting Phase 1: Light Augmentations ({base_name}) ---")
    if isinstance(model_source, str):
        model = YOLO(model_source)
    else:
        model = model_source
        
    p1_name = f"{base_name}_p1"
    model.train(
        data=data_yaml,
        epochs=70,
        name=p1_name,
        
        # Light Augs (Dataset is already augmented)
        mosaic=0.2,          # Reduced from 0.3
        mixup=0.1,        
        copy_paste=0.0,      # Disabled
        shear=0.0,           # Disabled
        perspective=0.0,     # Disabled
        
        # Minimal Geom
        translate=0.05,      # Reduced from 0.20
        scale=0.10,          # Reduced from 0.35
        fliplr=0.5,          # Keep horizontal flip
        flipud=0.0,
        degrees=0.2,         
        
        **common_args
    )
    
    # Load weights for Phase 2
    # Dynamically get the save directory from the trainer to handle auto-incremented names
    if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
        p1_weights = os.path.join(str(model.trainer.save_dir), "weights", "last.pt")
    else:
        p1_weights = os.path.join(project_dir, p1_name, "weights", "last.pt")
    
    # --- Phase 2: Fine Tuning (50-170 epochs -> 120 epochs) ---
    print(f"\n--- Starting Phase 2: Fine Tuning ({base_name}) ---")
    # Recreate the same architecture used to start Phase 1 and load phase-1 weights into it.
    # This prevents Ultralytics from inferring a different architecture when given only a weights file.
    arch_src = None
    if isinstance(model_source, str):
        arch_src = model_source
    else:
        # Try to recover the original YAML or source if available on the YOLO object
        try:
            arch_src = getattr(model_source, 'model').yaml
        except Exception:
            arch_src = None

    if arch_src:
        try:
            next_model = YOLO(arch_src)
            # load checkpoint and attempt to apply model weights
            ckpt = torch.load(p1_weights, map_location='cpu')
            state = None
            if isinstance(ckpt, dict):
                state = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt.get('state_dict')
            if state is not None and hasattr(next_model, 'model') and hasattr(next_model.model, 'load_state_dict'):
                # state may be a state_dict or a dict with nested tensors
                next_model.model.load_state_dict(state, strict=False)
                model = next_model
            else:
                # fallback to letting YOLO infer from weights
                model = YOLO(p1_weights)
        except Exception:
            model = YOLO(p1_weights)
    else:
        model = YOLO(p1_weights)
    p2_name = f"{base_name}_p2"
    model.train(
        data=data_yaml,
        epochs=100,
        name=p2_name,
        
        # Very Light / Clean
        mosaic=0.0,
        mixup=0.1,
        copy_paste=0.0,  
    
        # Minimal Geom
        translate=0.05,
        scale=0.05,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        
        **common_args
    )
    
    # Load weights for Phase 3
    if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
        p2_weights = os.path.join(str(model.trainer.save_dir), "weights", "last.pt")
    else:
        p2_weights = os.path.join(project_dir, p2_name, "weights", "last.pt")

    # --- Phase 3: Clean Training (170-320 epochs -> 150 epochs) ---
    print(f"\n--- Starting Phase 3: Clean Training ({base_name}) ---")
    # Same logic as above: try to recreate the original architecture then load p2 weights
    if arch_src:
        try:
            next_model = YOLO(arch_src)
            ckpt = torch.load(p2_weights, map_location='cpu')
            state = None
            if isinstance(ckpt, dict):
                state = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt.get('state_dict')
            if state is not None and hasattr(next_model, 'model') and hasattr(next_model.model, 'load_state_dict'):
                next_model.model.load_state_dict(state, strict=False)
                model = next_model
            else:
                model = YOLO(p2_weights)
        except Exception:
            model = YOLO(p2_weights)
    else:
        model = YOLO(p2_weights)
    p3_name = f"{base_name}_p3"
    results = model.train(
        data=data_yaml,
        epochs=150,
        name=p3_name,
        
        # Clean (No Mosaic/Mixup)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        
        # Reduced Geom
        translate=0.15,
        scale=0.30,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        
        **common_args
    )
    
    # Return best weights from the final clean phase
    final_best = os.path.join(project_dir, p3_name, "weights", "best.pt")
    if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
        final_best = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")
    elif results and getattr(results, 'best', None):
        final_best = str(results.best)
        
    return final_best

def train_model(initial_weights=None):
    r"""
    Combined training on both datasets (Local + T: Drive) with obstacles removed.
    Uses 3-Phase Scheduled Augmentations.
    """
    runs_dir = os.path.join(DATA_ROOT, "runs")

    # --- 1. Filter Datasets (Remove Obstacles) ---
    print("\n=== STEP 1: Preparing Datasets ===")
    local_dataset = os.path.abspath("datasets")
    
    # Find highest numbered dataset folder in T: drive
    t_datasets = glob.glob(os.path.join(DATASET_DIR, "dataset*"))
    best_dataset = None
    max_num = -1
    
    for ds in t_datasets:
        base = os.path.basename(ds)
        # Extract number
        try:
            num_str = base.replace("dataset", "")
            if num_str.isdigit():
                num = int(num_str)
                if num > max_num:
                    max_num = num
                    best_dataset = ds
        except Exception:
            pass
            
    train_paths = [os.path.join(local_dataset, 'train', 'images')]
    val_paths = [os.path.join(local_dataset, 'valid', 'images')]
    
    preprocess_dataset(local_dataset, CLASSES)
    
    if best_dataset:
        print(f"Selected latest dataset: {best_dataset}")
        
        # Verify structure
        t_train = os.path.join(best_dataset, 'train', 'images')
        t_valid = os.path.join(best_dataset, 'valid', 'images')
        
        if os.path.exists(t_train) and os.path.exists(t_valid):
            preprocess_dataset(best_dataset, CLASSES)
            train_paths.append(t_train)
            val_paths.append(t_valid)
        else:
            print(f"Warning: {best_dataset} missing 'train/images' or 'valid/images'. Skipping.")
    else:
        print("No external datasets found in T: drive.")
    
    # --- 2. Create Combined YAML ---
    combined_yaml = os.path.join(DATA_ROOT, "combined_data.yaml")
    print(f"Creating combined data config at {combined_yaml}...")
    
    content = {
        'path': DATA_ROOT, 
        'train': train_paths,
        'val': val_paths,
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    with open(combined_yaml, 'w') as f:
        yaml.dump(content, f, sort_keys=False)

    # --- 3. Train (One Pass) ---
    print("\n=== STEP 2: Starting Combined Training ===")
    
    if initial_weights:
        initial_model = initial_weights
        print(f"Resuming/Retraining from provided weights: {initial_model}")
    else:
        # Start with COCO checkpoint (pass the architecture path string so we can
        # reliably recreate the same architecture across phases)
        initial_model = "yolo11s.pt"

    # Run 3-Phase Training (pass the architecture string)
    best_weights = run_scheduled_training(
        initial_model,
        combined_yaml,
        runs_dir,
        "combined_training"
    )
    
    print(f"Training complete. Final Best Model: {best_weights}")

# --- 3. INFERENCE & AUTO-LABELING ---
def auto_label_inference(model_path, target_dir=None):
    """
    Runs inference with strict filters:
    - Conf >= 0.25
    - IoU >= 0.45
    - Area >= 0.1%
    - Persistence >= 3 frames
    """
    if target_dir is None:
        # Default to processing a session from frames dir if not specified
        # For now, let's just pick the first session folder found
        sessions = glob.glob(os.path.join(FRAMES_DIR, "*"))
        if sessions:
            target_dir = sessions[0]
        else:
            print("No session folders found.")
            return

    print(f"Auto-labeling frames in: {target_dir}")
    model = YOLO(model_path)
    
    frame_files = sorted(glob.glob(os.path.join(target_dir, "*.png")))
    if not frame_files:
        print("No frames found.")
        return

    # Track history: { track_id: [detection_info, ...] }
    track_history = defaultdict(list)
    
    # Run Tracking
    # stream=True returns a generator
    results = model.track(
        source=frame_files,
        conf=0.45,
        iou=0.45,
        persist=True,
        stream=True,
        verbose=False,
        tracker="bytetrack.yaml"
    )
    
    frames_data = [] # Store results to write later after filtering
    
    print("Running inference...")
    for i, result in enumerate(results):
        frame_path = result.path
        img_h, img_w = result.orig_shape
        img_area = img_h * img_w
        min_box_area = img_area * 0.001 # 0.1% threshold
        
        current_frame_dets = []
        
        if result.boxes:
            for box in result.boxes:
                # Check ID
                if box.id is None: continue
                track_id = int(box.id[0])
                
                # Check Area
                xyxy = box.xyxy[0].cpu().numpy()
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                if (w * h) < min_box_area:
                    continue
                
                # Store
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                det = {
                    'track_id': track_id,
                    'cls': cls,
                    'box': xyxy, # pixels
                    'conf': conf
                }
                
                track_history[track_id].append(det)
                current_frame_dets.append(det)
        
        frames_data.append({
            'path': frame_path,
            'shape': (img_h, img_w),
            'dets': current_frame_dets
        })
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(frame_files)} frames...")

    # Filter Tracks (Persistence >= 3)
    valid_ids = {tid for tid, hist in track_history.items() if len(hist) >= 3}
    print(f"Tracks found: {len(track_history)}. Valid (>=3 frames): {len(valid_ids)}")

    # Write Labels
    if not os.path.exists(AUTO_LABELS_DIR):
        os.makedirs(AUTO_LABELS_DIR)
        
    session_name = os.path.basename(target_dir)
    session_label_dir = os.path.join(AUTO_LABELS_DIR, session_name)
    if not os.path.exists(session_label_dir):
        os.makedirs(session_label_dir)

    count = 0
    for frame_data in frames_data:
        img_h, img_w = frame_data['shape']
        txt_name = os.path.splitext(os.path.basename(frame_data['path']))[0] + ".txt"
        txt_path = os.path.join(session_label_dir, txt_name)
        
        valid_dets = [d for d in frame_data['dets'] if d['track_id'] in valid_ids]
        
        if valid_dets:
            with open(txt_path, 'w') as f:
                for det in valid_dets:
                    # YOLO Format: class x_center y_center width height (normalized)
                    b = det['box']
                    w_px = b[2] - b[0]
                    h_px = b[3] - b[1]
                    xc = (b[0] + w_px/2) / img_w
                    yc = (b[1] + h_px/2) / img_h
                    wn = w_px / img_w
                    hn = h_px / img_h
                    
                    f.write(f"{det['cls']} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
            count += 1
            
    print(f"Generated labels for {count} frames in {session_label_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py [sample|train|infer <model> <dir>]")
        sys.exit(1)
        
    cmd = sys.argv[1]
    
    if cmd == "sample":
        count = 200
        subdir = None
        
        # Parse args: sample [count] [subdir]
        if len(sys.argv) >= 3:
            try:
                count = int(sys.argv[2])
            except ValueError:
                print(f"Invalid count '{sys.argv[2]}', defaulting to 200")
        
        if len(sys.argv) >= 4:
            subdir = sys.argv[3]
            
        sample_frames(count, subdir)
    elif cmd == "train":
        weights_path = None
        if len(sys.argv) >= 3:
            weights_path = sys.argv[2]
        train_model(weights_path)
    elif cmd == "infer":
        if len(sys.argv) >= 4:
            auto_label_inference(sys.argv[2], sys.argv[3])
        else:
            print("Usage: python pipeline.py infer <path_to_best.pt> <path_to_frames_folder>")
    else:
        print(f"Unknown command: {cmd}")
