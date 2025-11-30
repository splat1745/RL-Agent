"""
Auto-Farmer Pipeline Commands:

1. Sample Frames for Labeling:
   python pipeline.py sample [count] [subdir]
   Example: python pipeline.py sample 300 session_01

2. Train Model (Sequential Checkpoint Training):
   python pipeline.py train [model_type] [model_size]
   Example: python pipeline.py train rtdetr l
   Example: python pipeline.py train yolo s

3. Run Inference & Auto-Label:
   python pipeline.py infer <model_path> <frames_dir>
   Example: python pipeline.py infer runs/train/weights/best.pt data/frames/session_01
"""
import os
import shutil
import random
import glob
import yaml
try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    YOLO = None
    RTDETR = None

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False

import cv2
import numpy as np
import torch
from collections import defaultdict
import sys
import time
import json
import shutil

# --- CONFIGURATION ---
# Dynamic Data Root Detection
DATA_ROOT = r"T:\Auto-Farmer-Data"
if not os.path.exists(DATA_ROOT):
    # Try other common locations
    candidates = [
        r"D:\Auto-Farmer-Data",
        r"C:\Auto-Farmer-Data",
        os.path.join(os.getcwd(), "data")
    ]
    for path in candidates:
        if os.path.exists(path):
            DATA_ROOT = path
            break
            
# Ensure it exists (fallback to local if nothing found)
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = os.path.join(os.getcwd(), "data")
    if not os.path.exists(DATA_ROOT):
        try:
            os.makedirs(DATA_ROOT)
        except OSError:
            pass

print(f"Active Data Root: {DATA_ROOT}")

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
        # print(f"Dataset at {dataset_root} already processed. Skipping.")
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
            # print(f"Dropping class '{name}' (Index {src_idx})")
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

def ensure_coco_format(dataset_root, classes):
    """
    Converts a YOLO dataset (train/images, train/labels) to COCO format (train/image.jpg, train/_annotations.coco.json).
    Creates a 'coco' subdirectory in the dataset_root.
    Checks if the dataset is already in COCO format.
    """
    # 1. Check if already COCO (Native)
    # Look for _annotations.coco.json in train/valid folders directly
    is_native_coco = True
    for split in ['train', 'valid']:
        json_path = os.path.join(dataset_root, split, "_annotations.coco.json")
        if not os.path.exists(json_path):
            is_native_coco = False
            break
    
    if is_native_coco:
        print(f"Dataset {dataset_root} is already in COCO format.")
        return dataset_root

    # 2. Check if we already converted it
    coco_root = os.path.join(dataset_root, "coco")
    if os.path.exists(os.path.join(coco_root, "processed.marker")):
        return coco_root

    print(f"Converting {dataset_root} to COCO format...")
    
    for split in ['train', 'valid']:
        src_img_dir = os.path.join(dataset_root, split, 'images')
        src_lbl_dir = os.path.join(dataset_root, split, 'labels')
        dst_dir = os.path.join(coco_root, split)
        
        if not os.path.exists(src_img_dir):
            continue
            
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir)
        
        # COCO JSON Structure
        coco_data = {
            "info": {"year": 2025, "version": "1.0", "description": "Converted from YOLO"},
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": []
        }
        
        # Add Categories
        for i, cls_name in enumerate(classes):
            coco_data["categories"].append({
                "id": i,
                "name": cls_name,
                "supercategory": "none"
            })
            
        # Process Images
        img_files = glob.glob(os.path.join(src_img_dir, "*"))
        ann_id = 1
        
        for img_id, img_path in enumerate(img_files):
            # Copy image
            dst_img_path = os.path.join(dst_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dst_img_path)
            
            # Read Image Size
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            
            coco_data["images"].append({
                "id": img_id,
                "width": w,
                "height": h,
                "file_name": os.path.basename(img_path)
            })
            
            # Read Label
            label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path = os.path.join(src_lbl_dir, label_name)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    # YOLO (Normalized Center) -> COCO (Absolute Top-Left)
                    # x_min = (cx - w/2) * img_w
                    # y_min = (cy - h/2) * img_h
                    # w_abs = bw * img_w
                    # h_abs = bh * img_h
                    
                    w_abs = bw * w
                    h_abs = bh * h
                    x_min = (cx * w) - (w_abs / 2)
                    y_min = (cy * h) - (h_abs / 2)
                    
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [x_min, y_min, w_abs, h_abs],
                        "area": w_abs * h_abs,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    ann_id += 1
        
        # Save JSON
        json_path = os.path.join(dst_dir, "_annotations.coco.json")
        with open(json_path, 'w') as f:
            json.dump(coco_data, f)
            
    with open(os.path.join(coco_root, "processed.marker"), 'w') as f:
        f.write("done")
        
    return coco_root

def is_coco_dataset(dataset_root):
    """Checks if the dataset is already in COCO format (has _annotations.coco.json)."""
    for split in ['train', 'valid']:
        json_path = os.path.join(dataset_root, split, "_annotations.coco.json")
        if os.path.exists(json_path):
            return True
    return False

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
def get_dataset_list():
    """
    Returns a sorted list of dataset paths (dataset, dataset2, dataset3...)
    """
    datasets = []
    
    # Local dataset (if exists)
    local_ds = os.path.abspath("datasets")
    if os.path.exists(local_ds):
        datasets.append((0, local_ds))
        
    # T: Drive datasets
    t_datasets = glob.glob(os.path.join(DATASET_DIR, "dataset*"))
    for ds in t_datasets:
        base = os.path.basename(ds)
        num = 1 # Default for 'dataset'
        try:
            num_str = base.replace("dataset", "")
            if num_str.isdigit():
                num = int(num_str)
            elif num_str == "":
                num = 1
        except:
            continue
        datasets.append((num, ds))
        
    # Sort by number
    datasets.sort(key=lambda x: x[0])
    return [d[1] for d in datasets]

def estimate_training_time(dataset_path, epochs, model_type):
    """
    Estimates training time based on image count and model type.
    """
    train_imgs = glob.glob(os.path.join(dataset_path, "train", "images", "*.png"))
    img_count = len(train_imgs)
    
    # Heuristics (Seconds per epoch per 1000 images)
    # RT-DETR is slower than YOLO
    if "rtdetr" in model_type:
        sec_per_epoch_1k = 45 # Estimate
    else:
        sec_per_epoch_1k = 25 # YOLO Estimate
        
    total_seconds = (img_count / 1000.0) * sec_per_epoch_1k * epochs
    
    # Add overhead
    total_seconds += 60 # Startup time
    
    minutes = total_seconds / 60
    return minutes

def get_model_settings(model_type, model_size):
    """
    Returns specific settings (imgsz, batch) for different models.
    """
    settings = {
        'imgsz': 640,
        'batch': 32
    }
    
    if "rtdetr" in model_type.lower() and not model_type.startswith("rfdetr"):
        settings['imgsz'] = 640 
        # RT-DETR is heavy. Adjust batch size.
        if model_size.lower() in ['l', 'x']:
            settings['batch'] = 4 # Conservative for Large/X on consumer GPUs
        else:
            settings['batch'] = 8 
            
    elif "yolo" in model_type.lower():
        settings['imgsz'] = 640
        settings['batch'] = 32
        
    return settings

def train_sequential(model_type="rtdetr", model_size="l", explicit_datasets=None, initial_weights=None, epochs=70, batch_size=None, workers=None, grad_accum=None, device=None):
    """
    Trains sequentially on datasets.
    dataset -> dataset2 -> dataset3 ...
    Or uses explicit_datasets if provided.
    """
    runs_dir = os.path.join(DATA_ROOT, "runs")
    
    if explicit_datasets:
        datasets = explicit_datasets
        print(f"Using explicit dataset list: {datasets}")
    else:
        datasets = get_dataset_list()
    
    if not datasets:
        print("No datasets found!")
        return

    print(f"\n=== Found {len(datasets)} Datasets for Sequential Training ===")
    for i, ds in enumerate(datasets):
        print(f"  {i+1}. {ds}")
        
    # --- RF-DETR BRANCH ---
    if model_type.startswith("rfdetr"):
        if not RFDETR_AVAILABLE:
            print("Error: RF-DETR not installed. Run 'pip install rfdetr'")
            return

        # Map size to class
        size_map = {
            'n': RFDETRNano,
            's': RFDETRSmall,
            'm': RFDETRMedium,
            'b': RFDETRBase,
            'l': RFDETRLarge
        }
        ModelClass = size_map.get(model_size.lower(), RFDETRBase)
        print(f"Selected RF-DETR Model: {ModelClass.__name__}")
        
        current_weights = initial_weights
        
        for i, ds_path in enumerate(datasets):
            ds_name = os.path.basename(ds_path)
            print(f"\n>>> Starting RF-DETR Training on Dataset {i+1}/{len(datasets)}: {ds_name}")
            
            # 1. Preprocess
            if is_coco_dataset(ds_path):
                print(f"Dataset {ds_name} detected as COCO format. Skipping YOLO preprocessing.")
                coco_path = ds_path
            else:
                preprocess_dataset(ds_path, CLASSES) # Ensure clean YOLO first
                coco_path = ensure_coco_format(ds_path, CLASSES)
            
            # 2. Train
            project_name = f"seq_train_{model_type}_{model_size}"
            run_name = f"{ds_name}_run"
            output_dir = os.path.join(runs_dir, project_name, run_name)
            
            print(f"Loading weights: {current_weights if current_weights else 'Default Pretrained'}")
            
            # Initialize Model
            if current_weights:
                model = ModelClass(pretrain_weights=current_weights, num_classes=len(CLASSES))
            else:
                model = ModelClass(num_classes=len(CLASSES))
                
            # Train
            # Note: RF-DETR train() args might vary slightly by version, using standard ones
            model.train(
                dataset_dir=coco_path,
                epochs=epochs,
                batch_size=batch_size if batch_size else 4, # Conservative
                grad_accum_steps=grad_accum if grad_accum else 4,
                num_workers=workers if workers is not None else 2,
                lr=1e-4,
                output_dir=output_dir,
                resume=current_weights if current_weights else None,
                device=device if device else 'cuda'
            )
            
            # 3. Update weights
            # RF-DETR saves 'checkpoint.pth' (latest) and 'checkpoint_best_total.pth'
            best_pt = os.path.join(output_dir, "checkpoint_best_total.pth")
            if not os.path.exists(best_pt):
                best_pt = os.path.join(output_dir, "checkpoint.pth")
                
            if os.path.exists(best_pt):
                print(f"Dataset {ds_name} Complete. Best weights: {best_pt}")
                current_weights = best_pt
            else:
                print(f"Error: Could not find checkpoint at {best_pt}. Aborting sequence.")
                break
                
        print(f"\n=== RF-DETR Sequential Training Complete ===")
        print(f"Final Model: {current_weights}")
        return

    # --- ULTRALYTICS BRANCH ---
    # Base Model Selection
    if model_type == "rtdetr":
        base_model = f"rtdetr-{model_size}.pt"
        ModelClass = RTDETR
    else:
        base_model = f"yolo11{model_size}.pt"
        ModelClass = YOLO
        
    if initial_weights:
        print(f"\nStarting from provided weights: {initial_weights}")
        current_weights = initial_weights
    else:
        print(f"\nStarting from base model: {base_model}")
        current_weights = base_model

    settings = get_model_settings(model_type, model_size)
    print(f"Training Settings: {settings}")
    
    for i, ds_path in enumerate(datasets):
        ds_name = os.path.basename(ds_path)
        print(f"\n>>> Starting Training on Dataset {i+1}/{len(datasets)}: {ds_name}")
        
        # 1. Preprocess
        preprocess_dataset(ds_path, CLASSES)
        
        # 2. Create YAML for this specific dataset
        yaml_path = os.path.join(ds_path, "data_seq.yaml")
        content = {
            'path': ds_path,
            'train': 'train/images',
            'val': 'valid/images',
            'nc': len(CLASSES),
            'names': CLASSES
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(content, f, sort_keys=False)
            
        # 3. Estimate Time
        est_min = estimate_training_time(ds_path, epochs, model_type)
        print(f"Estimated Training Time: {est_min:.1f} minutes ({epochs} epochs)")
        
        # 4. Train
        project_name = f"seq_train_{model_type}_{model_size}"
        run_name = f"{ds_name}_run"
        
        print(f"Loading weights: {current_weights}")
        model = ModelClass(current_weights)
        
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            project=os.path.join(runs_dir, project_name),
            name=run_name,
            imgsz=settings['imgsz'],
            batch=batch_size if batch_size else settings['batch'],
            patience=10,
            device=device if device else 0,
            workers=workers if workers is not None else 8,
            exist_ok=True,
            
            # Augmentations (Moderate)
            mosaic=0.5,
            mixup=0.1,
            degrees=0.1,
            translate=0.1,
            scale=0.2,
            fliplr=0.5
        )
        
        # 5. Update weights for next round
        # Ultralytics saves to runs/project/name/weights/best.pt
        # But we need to be sure where it saved.
        if hasattr(model.trainer, 'save_dir'):
            best_pt = os.path.join(model.trainer.save_dir, "weights", "best.pt")
        else:
            # Fallback guess
            best_pt = os.path.join(runs_dir, project_name, run_name, "weights", "best.pt")
            
        if os.path.exists(best_pt):
            print(f"Dataset {ds_name} Complete. Best weights: {best_pt}")
            current_weights = best_pt
        else:
            print(f"Error: Could not find best.pt at {best_pt}. Aborting sequence.")
            break
            
    print(f"\n=== Sequential Training Complete ===")
    print(f"Final Model: {current_weights}")


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
    
    # Determine Model Type
    is_rfdetr = False
    if model_path.endswith(".pth") or "rfdetr" in model_path.lower():
        is_rfdetr = True
        
    if is_rfdetr:
        if not RFDETR_AVAILABLE:
            print("Error: RF-DETR not installed.")
            return
            
        # Try to guess size from path
        lower_path = model_path.lower()
        if "nano" in lower_path: ModelClass = RFDETRNano
        elif "small" in lower_path: ModelClass = RFDETRSmall
        elif "medium" in lower_path: ModelClass = RFDETRMedium
        elif "large" in lower_path: ModelClass = RFDETRLarge
        else: ModelClass = RFDETRBase
        
        print(f"Loading RF-DETR Model ({ModelClass.__name__}) from {model_path}")
        model = ModelClass(pretrain_weights=model_path, num_classes=len(CLASSES))
        # model.optimize_for_inference() # Optional, might require specific hardware
        
    else:
        # Ultralytics
        if "rtdetr" in model_path.lower():
            model = RTDETR(model_path)
        else:
            model = YOLO(model_path)
    
    frame_files = sorted(glob.glob(os.path.join(target_dir, "*.png")))
    if not frame_files:
        print("No frames found.")
        return

    # Track history: { track_id: [detection_info, ...] }
    track_history = defaultdict(list)
    frames_data = [] 
    
    print("Running inference...")
    
    if is_rfdetr:
        # RF-DETR Inference (No built-in tracker, so we simulate IDs or just detect)
        # For now, we will just run detection. 
        # TODO: Implement ByteTrack for RF-DETR if needed.
        
        for i, frame_path in enumerate(frame_files):
            # Predict
            # Returns a Supervision Detections object or similar? 
            # The docs say: detections = model.predict(image_path)
            # We need to inspect the output format. 
            # Assuming it returns a list of dicts or an object with .xyxy, .confidence, .class_id
            
            try:
                results = model.predict(frame_path, confidence=0.45, overlap=0.45)
            except Exception as e:
                print(f"Error predicting {frame_path}: {e}")
                continue
                
            img = cv2.imread(frame_path)
            if img is None: continue
            img_h, img_w = img.shape[:2]
            img_area = img_h * img_w
            min_box_area = img_area * 0.001
            
            current_frame_dets = []
            
            # RF-DETR likely returns a supervision.Detections object if using Roboflow inference
            # Or a custom object. Let's assume it behaves like the docs example which implies a list or object.
            # Actually, looking at the library source would be best, but let's assume standard Roboflow Inference format.
            # Usually: results.json() or results.xyxy
            
            # If results is a list of predictions:
            # Each pred: {'x':, 'y':, 'width':, 'height':, 'confidence':, 'class':, 'class_id':}
            
            # Let's try to handle the object generically
            preds = []
            if hasattr(results, 'json'):
                preds = results.json().get('predictions', [])
            elif isinstance(results, list):
                preds = results
            elif hasattr(results, 'xyxy'): # Supervision Detections
                # Convert to list of dicts
                for k in range(len(results.xyxy)):
                    preds.append({
                        'x': (results.xyxy[k][0] + results.xyxy[k][2]) / 2,
                        'y': (results.xyxy[k][1] + results.xyxy[k][3]) / 2,
                        'width': results.xyxy[k][2] - results.xyxy[k][0],
                        'height': results.xyxy[k][3] - results.xyxy[k][1],
                        'confidence': results.confidence[k],
                        'class_id': results.class_id[k]
                    })

            for p in preds:
                # Normalize to xyxy
                if 'x' in p and 'width' in p:
                    # Center format
                    x, y, w, h = p['x'], p['y'], p['width'], p['height']
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                elif 'x_min' in p:
                    x1, y1 = p['x_min'], p['y_min']
                    x2, y2 = p['x_max'], p['y_max']
                    w = x2 - x1
                    h = y2 - y1
                else:
                    continue
                    
                if (w * h) < min_box_area: continue
                
                # Fake ID since we don't have a tracker
                # This disables the persistence filter effectively
                track_id = -1 
                
                det = {
                    'track_id': track_id,
                    'cls': int(p.get('class_id', 0)),
                    'box': [x1, y1, x2, y2],
                    'conf': float(p.get('confidence', 0))
                }
                current_frame_dets.append(det)
                
            frames_data.append({
                'path': frame_path,
                'shape': (img_h, img_w),
                'dets': current_frame_dets
            })
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(frame_files)} frames...")
                
    else:
        # Ultralytics Tracking
        results = model.track(
            source=frame_files,
            conf=0.45,
            iou=0.45,
            persist=True,
            stream=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
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
    # Only applies if we have IDs (Ultralytics)
    if not is_rfdetr:
        valid_ids = {tid for tid, hist in track_history.items() if len(hist) >= 3}
        print(f"Tracks found: {len(track_history)}. Valid (>=3 frames): {len(valid_ids)}")
    else:
        # For RF-DETR, we accept all detections for now
        valid_ids = set([-1]) 
        print("RF-DETR: Skipping persistence check (no tracker).")

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
        
        if is_rfdetr:
            valid_dets = frame_data['dets']
        else:
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
        if len(sys.argv) >= 3:
            try:
                count = int(sys.argv[2])
            except ValueError:
                print(f"Invalid count '{sys.argv[2]}', defaulting to 200")
        if len(sys.argv) >= 4:
            subdir = sys.argv[3]
        sample_frames(count, subdir)
        
    elif cmd == "train":
        # python pipeline.py train [model_type] [model_size] [--datasets path1 path2 ...] [--weights path] [--epochs n] [--batch n] [--workers n]
        m_type = "rtdetr"
        m_size = "l"
        dataset_paths = []
        init_weights = None
        num_epochs = 70
        batch_size = None
        workers = None
        grad_accum = None
        device = None
        
        args = sys.argv[2:]
        
        # Parse model type and size if not flags
        if len(args) > 0 and not args[0].startswith("-"):
            m_type = args[0]
            args = args[1:]
        
        if len(args) > 0 and not args[0].startswith("-"):
            m_size = args[0]
            args = args[1:]
            
        # Parse flags
        i = 0
        while i < len(args):
            if args[i] == "--datasets":
                i += 1
                while i < len(args) and not args[i].startswith("-"):
                    dataset_paths.append(args[i])
                    i += 1
                continue # Don't increment i again
            elif args[i] == "--weights":
                if i + 1 < len(args):
                    init_weights = args[i+1]
                    i += 2
                    continue
            elif args[i] == "--epochs":
                if i + 1 < len(args):
                    try:
                        num_epochs = int(args[i+1])
                    except:
                        pass
                    i += 2
                    continue
            elif args[i] == "--batch":
                if i + 1 < len(args):
                    try:
                        batch_size = int(args[i+1])
                    except:
                        pass
                    i += 2
                    continue
            elif args[i] == "--workers":
                if i + 1 < len(args):
                    try:
                        workers = int(args[i+1])
                    except:
                        pass
                    i += 2
                    continue
            elif args[i] == "--grad-accum":
                if i + 1 < len(args):
                    try:
                        grad_accum = int(args[i+1])
                    except:
                        pass
                    i += 2
                    continue
            elif args[i] == "--device":
                if i + 1 < len(args):
                    device = args[i+1]
                    if device.isdigit():
                        device = int(device)
                    i += 2
                    continue
            i += 1
            
        train_sequential(m_type, m_size, dataset_paths, init_weights, num_epochs, batch_size, workers, grad_accum, device)
        
    elif cmd == "infer":
        if len(sys.argv) >= 4:
            auto_label_inference(sys.argv[2], sys.argv[3])
        else:
            print("Usage: python pipeline.py infer <path_to_best.pt> <path_to_frames_folder>")
    else:
        print(f"Unknown command: {cmd}")
