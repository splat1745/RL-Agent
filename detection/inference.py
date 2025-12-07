import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from collections import deque, defaultdict
import json
import os
import traceback
import warnings

# Filter RF-DETR warnings
warnings.filterwarnings("ignore", message=".*Using a different number of positional encodings.*")
warnings.filterwarnings("ignore", message=".*Using patch size.*")

# Try to import RF-DETR
try:
    from rfdetr import RFDETRBase, RFDETRSmall, RFDETRNano, RFDETRMedium, RFDETRLarge
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False

# --- Configuration ---
# Updated: Now supports RF-DETR .pth files
MODEL_PATH = r"D:\Auto-Farmer-Data\runs\seq_train_rfdetr_s\dataset5_run\checkpoint_best_ema.pth"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.20 # Lowered from 0.25
IOU_THRESHOLD = 0.45
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HALF_PRECISION = True if DEVICE == 'cuda' else False

print(f"Perception Config: Device={DEVICE}, FP16={HALF_PRECISION}, Torch={torch.__version__}, CUDA={torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# Class mapping (Adjust based on your model's training)
# Based on pipeline.py CLASSES list (Obstacles removed)
# Swapped 0 and 1 based on observation that Player (center) is detected as Class 1
# Added Class 2 as throwable based on logs
CLASS_MAP = {
    0: "throwable",
    1: "character",
    2: "throwable"
}
# Fallback for COCO if using pretrained yolo11n.pt directly without custom training
# We will assume 'person' (0) is player/enemy for now if not custom.

class SimpleTracker:
    """
    A lightweight tracker using Lucas-Kanade Optical Flow.
    Replaces cv2.TrackerKCF when opencv-contrib is missing.
    """
    def __init__(self):
        self.bbox = None
        self.prev_gray = None
        self.pts = None

    def init(self, frame, bbox):
        self.bbox = bbox # (x, y, w, h)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Create grid of points within bbox
        x, y, w, h = map(int, bbox)
        # Ensure bounds
        h_img, w_img = self.prev_gray.shape
        x = max(0, x); y = max(0, y)
        w = min(w, w_img - x); h = min(h, h_img - y)
        
        if w <= 0 or h <= 0: return
        
        # Grid of points
        step = max(4, min(w, h) // 4)
        grid_y, grid_x = np.mgrid[y+step//2:y+h:step, x+step//2:x+w:step]
        self.pts = np.vstack((grid_x.flatten(), grid_y.flatten())).T.astype(np.float32).reshape(-1, 1, 2)

    def update(self, frame):
        if self.pts is None or len(self.pts) == 0:
            return False, self.bbox
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # LK Flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.pts, None, winSize=(15, 15), maxLevel=2)
        
        # Select good points
        if next_pts is not None:
            good_new = next_pts[status == 1]
            good_old = self.pts[status == 1]
        else:
            return False, self.bbox
        
        if len(good_new) < len(self.pts) * 0.5: # Lost too many points
            return False, self.bbox
            
        # Estimate motion
        motion = good_new - good_old
        dx = np.median(motion[:, 0])
        dy = np.median(motion[:, 1])
        
        # Update bbox
        x, y, w, h = self.bbox
        self.bbox = (x + dx, y + dy, w, h)
        
        # Update state
        self.prev_gray = gray
        self.pts = good_new.reshape(-1, 1, 2)
        
        return True, self.bbox

class Perception:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE, half=HALF_PRECISION, fast_mode=False):
        self.device = device
        self.half = half
        self.fast_mode = fast_mode
        self.model = self._init_model(model_path)
        
        # Optimization
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Tracking / Velocity State
        self.player_history = deque(maxlen=3) # Store (cx, cy, time)
        self.last_obs = None
        self.last_det = {} # Store latest detections for visualization
        
        # --- Kalman Filter for Player Tracking ---
        # State: [x, y, vx, vy]
        # Measurement: [x, y]
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,5,0],[0,0,0,5]], np.float32) * 0.03
        self.last_kalman_pred = None # Store last prediction

        # Track History for Player ID (ID -> deque of (cx, cy))
        self.track_histories = defaultdict(lambda: deque(maxlen=15))
        
        # Enemy Memory
        self.last_enemy_pos = (1.0, 1.0) # (dx, dy)
        self.last_enemy_time = 0
        
        # Enemy Analysis State
        self.enemy_rois = defaultdict(lambda: deque(maxlen=5)) # ID -> deque of ROIs (gray)
        self.enemy_states = {} # ID -> State String
        
        # Global Motion & Smoothing
        self.smoothed_boxes = {} # ID -> xywh (smoothed)
        self.prev_gray = None
        self.prev_gray_full = None # For motion mask
        self.motion_mask = None
        self.prev_pts = None # For KLT tracking
        self.global_motion_history = deque(maxlen=30) # Store (dx, dy) per frame
        
        # Health ROI
        self.health_roi = None
        self.health_template = None
        self.health_bar_box = None # (x, y, w, h) relative to template
        
        # Enemy Health Template
        self.enemy_health_template = None
        if os.path.exists("images/EnemyHealth.png"):
            self.enemy_health_template = cv2.imread("images/EnemyHealth.png")
            if self.enemy_health_template is not None:
                print("Loaded EnemyHealth.png template.")

        # Hit Template
        self.hit_template = None
        if os.path.exists("images/TemplateHit.png"):
            self.hit_template = cv2.imread("images/TemplateHit.png")
            if self.hit_template is not None:
                print("Loaded TemplateHit.png template.")
        
        # UI Templates for Cooldowns and Bars
        self.ui_templates = {}
        self.ui_rois = {} # name -> (x, y, w, h)
        
        ui_files = {
            "move_1": "images/Lapse Blue.png",
            "move_2": "images/Reversal Red.png",
            "move_3": "images/Rapid Punches.png",
            "move_4": "images/Twofold Kick.png",
            "evasive_bar": "images/EvasiveBar.png",
            "special_bar": "images/SpecialBar.png"
        }
        
        for name, filename in ui_files.items():
            if os.path.exists(filename):
                tmpl = cv2.imread(filename)
                if tmpl is not None:
                    self.ui_templates[name] = tmpl
                    print(f"Loaded UI template: {filename}")
        
        if os.path.exists("images/Health.png"):
            self.health_template = cv2.imread("images/Health.png")
            if self.health_template is not None:
                print("Loaded Health.png template.")
                # Analyze template to find the "Max Health" bar area
                hsv_t = cv2.cvtColor(self.health_template, cv2.COLOR_BGR2HSV)
                # Green/Yellow/Red mask
                lower_green = np.array([35, 50, 50])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv_t, lower_green, upper_green)
                
                # Also check Red/Yellow just in case template is not full green
                lower_yellow = np.array([20, 50, 50])
                upper_yellow = np.array([35, 255, 255])
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv_t, lower_yellow, upper_yellow))
                
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 50, 50])
                upper_red2 = np.array([180, 255, 255])
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv_t, lower_red1, upper_red1))
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv_t, lower_red2, upper_red2))
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    self.health_bar_box = cv2.boundingRect(c)
                    print(f"Found Health Bar in Template: {self.health_bar_box}")
        
        self._load_config()
        
        # Debug Visualization
        self.debug_rois = {}
        self.debug_health = None
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.skip_frames = 0 # Dynamic frame skipping
        self.fps_log = []
        
        # --- Depth Estimation ---
        self.depth_model = None
        self.depth_transform = None
        try:
            print("Loading MiDaS Small for Depth Estimation...")
            
            # SSL Certificate Fix for Windows
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Check for timm
            try:
                import timm
            except ImportError:
                print("Warning: 'timm' library not found. Installing it is recommended for MiDaS: pip install timm")

            self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.depth_transform = midas_transforms.small_transform
            print("MiDaS Loaded.")
        except Exception as e:
            print(f"Error loading MiDaS: {e}. Depth estimation disabled.")
            traceback.print_exc()

        # --- Pixel-Based State ---
        self.frame_stack = deque(maxlen=4) # Stores last 4 frames (160x160)
        self.crop_stack = deque(maxlen=4)  # Stores last 4 crops (96x96)
        self.last_gray = None # For optical flow
        self.last_depth_vis = None # For visualization
        self.last_depth_map = None
        self.depth_interval = 5 # Run depth estimation every 5 frames
        
        # --- New Logic State ---
        # --- New Logic State ---
        self.brightness_history = defaultdict(lambda: deque(maxlen=5))
        self.ragdoll_counters = defaultdict(int)
        self.last_player_health = 1.0
        self.filtered_health = 1.0
        self.object_duration = {} # ID -> frames persisted

        # --- Optimization & Advanced Hit Detection ---
        # Check if tracker is available
        dummy_tracker = self._create_tracker()
        if dummy_tracker is None:
            print("Warning: No tracker found. Switching to per-frame detection (Performance may suffer).")
            self.detection_interval = 1
        else:
            # Use SimpleTracker or KCF
            self.detection_interval = 4
            
        self.trackers = {} # ID -> cv2.Tracker
        self.tracker_failures = defaultdict(int)
        
        # Hit Detection State
        self.hit_signals = defaultdict(lambda: deque(maxlen=3)) # ID -> deque of bools
        self.prev_crops = {} # ID -> prev_gray_crop
        self.hit_cooldowns = defaultdict(float) # ID -> timestamp

    def _create_tracker(self):
        """Creates a lightweight tracker (KCF or CSRT or SimpleTracker)."""
        # Try KCF first (Fastest)
        try:
            return cv2.TrackerKCF_create()
        except AttributeError:
            try:
                return cv2.legacy.TrackerKCF_create()
            except AttributeError:
                # Fallback to CSRT if KCF missing
                try:
                    return cv2.TrackerCSRT_create()
                except AttributeError:
                    try:
                        return cv2.legacy.TrackerCSRT_create()
                    except AttributeError:
                        # Fallback to SimpleTracker (Optical Flow)
                        return SimpleTracker()

    def detect_hit_confirm(self, frame, enemy_box, player_box, tid):
        """
        Advanced Hit Detection Pipeline:
        1. Expand Box & Crop
        2. Edge Detection & Masking (Define Character Bounds)
        3. Frame Differencing (Subtract Unchanging Pixels)
        4. Average Brightness Check within Mask
        5. Template Match (Override)
        """
        if frame is None: return 0.0
        
        x, y, w, h = enemy_box
        h_img, w_img = frame.shape[:2]
        
        # 1. Expand Box (~6px)
        pad = 6
        x1 = int(max(0, x - w/2 - pad))
        y1 = int(max(0, y - h/2 - pad))
        x2 = int(min(w_img, x + w/2 + pad))
        y2 = int(min(h_img, y + h/2 + pad))
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return 0.0
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 2. Template Match Check (Keep as strong signal)
        has_template_match = False
        if self.hit_template is not None:
            th, tw = self.hit_template.shape[:2]
            ch, cw = crop.shape[:2]
            if ch >= th and cw >= tw:
                res = cv2.matchTemplate(crop, self.hit_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > 0.6: 
                    has_template_match = True

        # 3. Edge Detection & Masking
        # "ensure that edge detection is used, and anything within the bounds of edge detection is read"
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect edges and form a rough body mask
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours to create a filled mask of the character
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        
        if contours:
            # Assume largest contour is the character
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [c], -1, 255, -1) # Fill contour
        else:
            # Fallback: Use dilated edges if no closed contour
            mask = dilated_edges

        # 4. Brighter Average Color with Background Subtraction
        # "subtract unchanging pixels (character instead of background)"
        brightness_hit = False
        
        if tid in self.prev_crops:
            prev_gray = self.prev_crops[tid]
            
            # Resize prev to match current (handle box size fluctuations)
            if prev_gray.shape != gray.shape:
                prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))
            
            # Subtract: Current - Prev (Detect Brightening only)
            # cv2.subtract clips negatives to 0
            diff = cv2.subtract(gray, prev_gray)
            
            # Filter: Only consider pixels that got SIGNIFICANTLY brighter (> 40)
            # This removes noise from slight movement/texture shifts
            _, diff_thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
            
            # Apply Mask: Only check pixels within the character bounds
            # diff_thresh is 255 where change > 40, 0 otherwise
            masked_diff = cv2.bitwise_and(diff_thresh, diff_thresh, mask=mask)
            
            # Calculate ratio of character that flashed
            mask_pixels = cv2.countNonZero(mask)
            if mask_pixels > 0:
                flash_pixels = cv2.countNonZero(masked_diff)
                flash_ratio = flash_pixels / mask_pixels
                
                # Threshold: If > 15% of the character body flashed bright
                if flash_ratio > 0.15: 
                    brightness_hit = True
        
        # Update History
        self.prev_crops[tid] = gray

        # 5. Combine Signals
        is_hit = brightness_hit or has_template_match
            
        self.hit_signals[tid].append(is_hit)
        
        # Persistence check (2 out of 3 frames)
        if sum(self.hit_signals[tid]) >= 2:
            return 1.0
            
        return 0.0

    def _load_config(self):
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    data = json.load(f)
                    if "health_roi" in data and "bbox" in data["health_roi"]:
                        self.health_roi = data["health_roi"]["bbox"] # [x, y, w, h]
                        print(f"Loaded Health ROI: {self.health_roi}")
            except Exception as e:
                print(f"Error loading config: {e}")

        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    data = json.load(f)
                    if "health_roi" in data and "bbox" in data["health_roi"]:
                        self.health_roi = data["health_roi"]["bbox"] # [x, y, w, h]
                        print(f"Loaded Health ROI: {self.health_roi}")
            except Exception as e:
                print(f"Error loading config: {e}")

    def _init_model(self, path):
        print(f"Loading model {path} to {self.device}...")
        
        # Detect model type from file extension
        self.is_rfdetr = path.endswith('.pth')
        
        try:
            if self.is_rfdetr:
                # RF-DETR Model
                if not RFDETR_AVAILABLE:
                    raise ImportError("RF-DETR not installed. Run: pip install rfdetr")
                
                # Determine model size from path
                path_lower = path.lower()
                if 'nano' in path_lower:
                    ModelClass = RFDETRNano
                elif 'small' in path_lower or 'rfdetr_s' in path_lower or 'rfdetr-s' in path_lower:
                    ModelClass = RFDETRSmall
                elif 'medium' in path_lower:
                    ModelClass = RFDETRMedium
                elif 'large' in path_lower:
                    ModelClass = RFDETRLarge
                else:
                    # Default to Small for seq_train_rfdetr_s paths
                    if 'rfdetr_s' in path_lower:
                        ModelClass = RFDETRSmall
                    else:
                        ModelClass = RFDETRBase
                
                print(f"Detected RF-DETR model: {ModelClass.__name__}")
                
                # Load with pretrained weights
                model = ModelClass(pretrain_weights=path, num_classes=len(CLASS_MAP))
                
                # Move to device and set precision
                # RF-DETR wrapper structure: model -> model.model -> model.model.model (nn.Module)
                try:
                    fp8_success = False
                    if hasattr(model, 'model') and hasattr(model.model, 'model'):
                        internal_model = model.model.model
                        if hasattr(internal_model, 'to'):
                            internal_model.to(self.device)
                            
                            # Fast Mode Optimization
                            if self.fast_mode:
                                print("Fast Mode Enabled: Using FP16 and torch.compile (if available).")
                                # Note: Native FP8 (float8_e4m3fn) is skipped because 'addmm_cuda' is not implemented 
                                # for this dtype on the current setup, causing crashes.
                                
                                # Apply torch.compile for speed
                                    # Compile (Disabled on Windows due to missing Triton)
                                    # if hasattr(torch, 'compile'):
                                    #     print("Enabling torch.compile() for extra speed...")
                                    #     try:
                                    #         model.model.model = torch.compile(internal_model, mode="reduce-overhead")
                                    #     except Exception as e:
                                    #         print(f"torch.compile failed: {e}")

                            if self.half:
                                internal_model.half()
                                print(f"Moved internal RF-DETR model to {self.device} (FP16={self.half})")
                        
                        # Update wrapper device reference if it exists
                        if hasattr(model.model, 'device'):
                            model.model.device = self.device
                            
                    # Optimize for inference
                    if hasattr(model, 'optimize_for_inference'):
                        dtype = torch.float16 if self.half else torch.float32
                        # Note: optimize_for_inference might re-compile, so we do it after moving
                        try:
                            model.optimize_for_inference(dtype=dtype)
                            print("RF-DETR optimized for inference.")
                        except Exception as opt_e:
                            print(f"Warning: Could not optimize RF-DETR: {opt_e}")
                            
                except Exception as e:
                    print(f"Warning: Could not move RF-DETR to device: {e}")
                
                print(f"RF-DETR Model loaded.")
                return model
            else:
                # YOLO Model
                model = YOLO(path)
                
                # Warmup
                print("Warming up YOLO model...")
                dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                _ = model(dummy, device=self.device, half=self.half, verbose=False)
                print("YOLO Model ready.")
                return model
        except Exception as e:
            print(f"Error initializing model: {e}")
            traceback.print_exc()
            raise

    def _run_rfdetr_inference(self, img):
        """
        Runs RF-DETR inference and returns results in a format compatible with the tracking pipeline.
        Returns a list of detection dicts with simulated track IDs.
        """
        try:
            # Convert BGR to RGB for RF-DETR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # RF-DETR predict method
            if self.half and self.device == 'cuda':
                with torch.amp.autocast('cuda'):
                    detections = self.model.predict(img_rgb, threshold=CONF_THRESHOLD)
            else:
                detections = self.model.predict(img_rgb, threshold=CONF_THRESHOLD)
            
            results = []
            
            # RF-DETR returns a supervision.Detections object or similar
            # We need to extract boxes, confidence, and class IDs
            if hasattr(detections, 'xyxy'):
                # Supervision Detections object
                for i in range(len(detections.xyxy)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    conf = detections.confidence[i] if hasattr(detections, 'confidence') else 0.5
                    cls_id = detections.class_id[i] if hasattr(detections, 'class_id') else 0
                    
                    # Check for normalized coordinates (0-1)
                    if x2 <= 1.0 and y2 <= 1.0:
                        h_img, w_img = img.shape[:2]
                        x1 *= w_img
                        x2 *= w_img
                        y1 *= h_img
                        y2 *= h_img
                    
                    # Convert to center format
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w/2
                    cy = y1 + h/2
                    
                    # Generate a pseudo track ID based on position (simple spatial hashing)
                    # This is a workaround since RF-DETR doesn't have built-in tracking
                    tid = int(cx / 50) * 100 + int(cy / 50)
                    
                    results.append({
                        'id': tid,
                        'cls': int(cls_id),
                        'conf': float(conf),
                        'xywh': np.array([cx, cy, w, h])
                    })
            elif isinstance(detections, list):
                # List of prediction dicts
                for i, det in enumerate(detections):
                    if 'x' in det and 'width' in det:
                        # Center format
                        cx, cy = det['x'], det['y']
                        w, h = det['width'], det['height']
                    elif 'x_min' in det:
                        # Corner format
                        x1, y1 = det['x_min'], det['y_min']
                        x2, y2 = det['x_max'], det['y_max']
                        w = x2 - x1
                        h = y2 - y1
                        cx = x1 + w/2
                        cy = y1 + h/2
                    else:
                        continue
                    
                    conf = det.get('confidence', 0.5)
                    cls_id = det.get('class_id', det.get('class', 0))
                    
                    # Check for normalized coordinates
                    if cx <= 1.0 and cy <= 1.0 and w <= 1.0:
                         h_img, w_img = img.shape[:2]
                         cx *= w_img
                         cy *= h_img
                         w *= w_img
                         h *= h_img
                    
                    tid = int(cx / 50) * 100 + int(cy / 50)
                    
                    results.append({
                        'id': tid,
                        'cls': int(cls_id),
                        'conf': float(conf),
                        'xywh': np.array([cx, cy, w, h])
                    })
            
            # Debug: Print first detection if available
            if results and self.frame_count % 60 == 0:
                print(f"DEBUG: RF-DETR Raw Detection 0: {results[0]}")
            
            return results
        except Exception as e:
            print(f"RF-DETR inference error: {e}")
            traceback.print_exc()
            return []

    def preprocess(self, frame):
        """
        Resizes and prepares frame for inference.
        frame: BGR numpy array from capture.py
        """
        if frame is None:
            return None
        
        # Resize
        # Note: Ultralytics handles resizing internally usually, but doing it here gives control
        # and ensures consistent input size.
        # However, passing the raw frame to model() is often faster as they have optimized pipelines.
        # But the prompt asked to "Resize to 640x640 with cv2.INTER_LINEAR".
        try:
            img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            return img
        except Exception:
            return None

    def _estimate_global_motion(self, frame_gray):
        """
        Estimates global camera motion (dx, dy) using sparse optical flow on background features.
        Returns (dx, dy) in pixels.
        """
        # Optimization: Resize to small resolution for flow calculation
        small_h, small_w = 160, 160
        small_gray = cv2.resize(frame_gray, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        scale_x = frame_gray.shape[1] / small_w
        scale_y = frame_gray.shape[0] / small_h
        
        # Initialize points if needed (KLT style)
        if self.prev_pts is None or len(self.prev_pts) < 50:
            # Create mask to ignore known moving objects (from last frame)
            mask = np.ones_like(small_gray, dtype=np.uint8) * 255
            
            if self.last_det:
                def mask_rect(rect):
                    if rect:
                        x, y, w, h = rect[:4]
                        # Scale to small
                        x /= scale_x
                        y /= scale_y
                        w /= scale_x
                        h /= scale_y
                        
                        x1 = int(max(0, x - w/2))
                        y1 = int(max(0, y - h/2))
                        x2 = int(min(small_w, x + w/2))
                        y2 = int(min(small_h, y + h/2))
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

                mask_rect(self.last_det.get("player"))
                for enemy in self.last_det.get("enemies", []):
                    mask_rect(enemy)

            # Use GFTT but only when points are low
            self.prev_pts = cv2.goodFeaturesToTrack(small_gray, mask=mask, maxCorners=200, qualityLevel=0.01, minDistance=5)
        
        if self.prev_gray is None:
            self.prev_gray = small_gray
            return 0.0, 0.0
            
        dx, dy = 0.0, 0.0
        
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            # Track to current frame
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, small_gray, self.prev_pts, None, winSize=(15, 15), maxLevel=2)
            
            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = self.prev_pts[st==1]
                
                if len(good_new) > 0:
                    flows = good_new - good_old
                    
                    # Robust estimation using RANSAC-like approach (Median + MAD)
                    dxs = flows[:, 0]
                    dys = flows[:, 1]
                    
                    med_dx = np.median(dxs)
                    med_dy = np.median(dys)
                    
                    # Filter outliers
                    mad_dx = np.median(np.abs(dxs - med_dx))
                    mad_dy = np.median(np.abs(dys - med_dy))
                    
                    valid_mask = (np.abs(dxs - med_dx) < 3 * mad_dx) & (np.abs(dys - med_dy) < 3 * mad_dy)
                    
                    if np.sum(valid_mask) > 0:
                        dx = np.mean(dxs[valid_mask])
                        dy = np.mean(dys[valid_mask])
                    else:
                        dx, dy = med_dx, med_dy
                    
                    # Scale back to original resolution
                    dx *= scale_x
                    dy *= scale_y
                    
                    # Update points for next frame
                    self.prev_pts = good_new.reshape(-1, 1, 2)
                else:
                    self.prev_pts = None
            else:
                self.prev_pts = None
        
        self.prev_gray = small_gray
        return dx, dy

    def generate_motion_mask(self, current_gray, prev_gray, dx, dy):
        """
        Generates a motion mask by compensating for camera motion.
        dx, dy: Camera motion (pixel displacement).
        """
        rows, cols = current_gray.shape
        
        # Create transformation matrix for translation
        # We shift prev by -dx, -dy to align with current
        M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        
        # Warp previous frame to align with current
        warped_prev = cv2.warpAffine(prev_gray, M, (cols, rows))
        
        # Calculate difference
        diff = cv2.absdiff(current_gray, warped_prev)
        
        # Threshold
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up noise
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask

    def detect(self, frame, mouse_movement=(0,0)):
        """
        Optimized Detection Pipeline:
        1. Global Motion Estimation
        2. YOLO (Every N frames) OR Tracking (Inter-frame)
        3. Hit Confirmation (Every frame)
        4. Logic & State Analysis
        """
        if frame is None: return {}
        
        # Initialize state if missing (Hot-fix for __init__)
        if not hasattr(self, 'object_state'):
            self.object_state = {} # ID -> {'tracker': obj, 'box': xywh, 'cls': int, 'conf': float, 'failures': int}

        # Preprocess (Resize)
        img = self.preprocess(frame)
        if img is None: return {}
        
        # Global Motion Estimation (Visual Odometry)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g_dx, g_dy = self._estimate_global_motion(gray)
        self.global_motion_history.append((g_dx, g_dy))
        
        # Motion Mask Generation
        if self.prev_gray_full is not None:
            self.motion_mask = self.generate_motion_mask(gray, self.prev_gray_full, g_dx, g_dy)
        else:
            self.motion_mask = np.zeros_like(gray)
        self.prev_gray_full = gray.copy()
        
        # Motion Blur Protection: Skip YOLO if camera is spinning too fast
        # This prevents false positives from motion blur
        # Check Mouse Input: If mouse is moving but flow is 0, flow failed -> Unstable
        m_dx, m_dy = mouse_movement
        flow_failed = (abs(m_dx) > 5 or abs(m_dy) > 5) and (abs(g_dx) < 1.0 and abs(g_dy) < 1.0)
        
        # Drastically increased speed limit to allow detection during turns
        is_stable = (abs(g_dx) < 300.0 and abs(g_dy) < 300.0) and not flow_failed

        # --- HIT DETECTION VIA SHAKE (User Request) ---
        # "subtracting mouse movement to compare background movement"
        # We assume Background Flow (g_dx, g_dy) is roughly proportional to Mouse (m_dx, m_dy)
        # Usually Mouse Move Right -> Camera Pans Right -> Background Moves Left (Negative Flow)
        # So Flow ~ -Mouse * Sensitivity. 
        # We use a heuristic sensitivity of 1.5 (pixels flow per pixel mouse) based on typical sensitivity.
        sensitivity = 1.5
        expected_dx = -m_dx * sensitivity
        expected_dy = -m_dy * sensitivity
        
        residual_dx = g_dx - expected_dx
        residual_dy = g_dy - expected_dy
        
        shake_magnitude = np.sqrt(residual_dx**2 + residual_dy**2)
        
        # If shake is high, it might be a hit (Camera Shake effect)
        # We only count it if the residual is significantly larger than the expected motion (Signal to Noise)
        is_shake_hit = 0.0
        if shake_magnitude > 10.0: # Threshold for significant shake
             # Further filter: If mouse is moving fast, residual might just be sensitivity mismatch.
             # So we trust shake more when mouse is slow.
             if abs(m_dx) < 5 and abs(m_dy) < 5:
                 is_shake_hit = 1.0
             elif shake_magnitude > 30.0: # Huge shake even with mouse movement
                 is_shake_hit = 1.0

        # Decide: Run YOLO or Track?
        run_yolo = (self.frame_count % self.detection_interval == 0) and is_stable
        
        # Clear Debug
        self.debug_rois = {}
        self.debug_health = None
        
        current_objects = {} # ID -> Box
        
        if run_yolo and self.model is not None:
            # --- DETECTION STEP ---
            if self.is_rfdetr:
                # RF-DETR Inference
                rfdetr_results = self._run_rfdetr_inference(img)
                
                for det in rfdetr_results:
                    tid = det['id']
                    cls_id = det['cls']
                    conf = det['conf']
                    xywh = det['xywh']
                    
                    x, y, w, h = xywh
                    x1, y1 = int(max(0, x - w/2)), int(max(0, y - h/2))
                    x2, y2 = int(min(INPUT_SIZE, x + w/2)), int(min(INPUT_SIZE, y + h/2))
                    
                    motion_score = 0.0
                    if self.motion_mask is not None:
                        roi_motion = self.motion_mask[y1:y2, x1:x2]
                        if roi_motion.size > 0:
                            motion_score = cv2.countNonZero(roi_motion) / roi_motion.size
                    
                    if conf < 0.5 and motion_score < 0.05:
                        continue
                    
                    tl_x, tl_y = x - w/2, y - h/2
                    bbox = (tl_x, tl_y, w, h)
                    
                    if tid not in self.object_state:
                        tracker = self._create_tracker()
                        if tracker:
                            tracker.init(img, bbox)
                        self.object_state[tid] = {'tracker': tracker, 'box': xywh, 'cls': cls_id, 'conf': conf, 'failures': 0}
                    else:
                        tracker = self.object_state[tid]['tracker']
                        if tracker:
                            tracker = self._create_tracker()
                            if tracker:
                                tracker.init(img, bbox)
                        self.object_state[tid]['tracker'] = tracker
                        self.object_state[tid]['box'] = xywh
                        self.object_state[tid]['conf'] = conf
                        self.object_state[tid]['failures'] = 0
                        
                    current_objects[tid] = xywh
            else:
                # YOLO Tracking
                results = self.model.track(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False, half=self.half, device=self.device, persist=True, tracker="bytetrack.yaml")
            
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        if box.id is None: continue
                        tid = int(box.id[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        xywh = box.xywh[0].cpu().numpy()
                        
                        # Motion Validation
                        # Check if the box contains motion
                        x, y, w, h = xywh
                        x1, y1 = int(max(0, x - w/2)), int(max(0, y - h/2))
                        x2, y2 = int(min(INPUT_SIZE, x + w/2)), int(min(INPUT_SIZE, y + h/2))
                        
                        motion_score = 0.0
                        if self.motion_mask is not None:
                            roi_motion = self.motion_mask[y1:y2, x1:x2]
                            if roi_motion.size > 0:
                                motion_score = cv2.countNonZero(roi_motion) / roi_motion.size
                        
                        # Filter: If confidence is low (< 0.5) AND motion is low (< 0.05), discard
                        # This allows keeping low-conf moving objects, but rejecting low-conf static ones
                        if conf < 0.5 and motion_score < 0.05:
                            continue
                        
                        # Update/Create Tracker
                        tl_x, tl_y = x - w/2, y - h/2
                        bbox = (tl_x, tl_y, w, h)
                        
                        if tid not in self.object_state:
                            # New Object
                            tracker = self._create_tracker()
                            if tracker:
                                tracker.init(img, bbox)
                            self.object_state[tid] = {'tracker': tracker, 'box': xywh, 'cls': cls_id, 'conf': conf, 'failures': 0}
                        else:
                            # Existing Object - Re-initialize to correct drift
                            tracker = self.object_state[tid]['tracker']
                            if tracker:
                                tracker = self._create_tracker()
                                if tracker:
                                    tracker.init(img, bbox)
                            self.object_state[tid]['tracker'] = tracker
                            self.object_state[tid]['box'] = xywh
                            self.object_state[tid]['conf'] = conf
                            self.object_state[tid]['failures'] = 0
                            
                        current_objects[tid] = xywh
            
            # Remove stale objects
            seen_ids = set(current_objects.keys())
            existing_ids = set(self.object_state.keys())
            to_remove = existing_ids - seen_ids
            for tid in to_remove:
                del self.object_state[tid]
                
        else:
            # --- TRACKING STEP ---
            to_remove = []
            for tid, state in self.object_state.items():
                tracker = state['tracker']
                success = False
                bbox = None
                
                if tracker:
                    success, bbox = tracker.update(img)
                
                if success and bbox is not None:
                    tl_x, tl_y, w, h = bbox
                    cx = tl_x + w/2
                    cy = tl_y + h/2
                    xywh = np.array([cx, cy, w, h])
                    state['box'] = xywh
                    state['failures'] = 0
                    current_objects[tid] = xywh
                else:
                    # Tracker Failed - Try Global Motion Compensation
                    prev_xywh = state['box']
                    cx, cy, w, h = prev_xywh
                    cx += g_dx
                    cy += g_dy
                    
                    # Check Motion Mask at predicted location
                    # If there is motion there, maybe we can keep it alive longer?
                    motion_score = 0.0
                    if self.motion_mask is not None:
                        x1, y1 = int(max(0, cx - w/2)), int(max(0, cy - h/2))
                        x2, y2 = int(min(INPUT_SIZE, cx + w/2)), int(min(INPUT_SIZE, cy + h/2))
                        roi_motion = self.motion_mask[y1:y2, x1:x2]
                        if roi_motion.size > 0:
                            motion_score = cv2.countNonZero(roi_motion) / roi_motion.size
                    
                    if 0 <= cx <= INPUT_SIZE and 0 <= cy <= INPUT_SIZE:
                        xywh = np.array([cx, cy, w, h])
                        state['box'] = xywh
                        state['failures'] += 1
                        
                        # If moving, be more lenient with failures
                        if motion_score > 0.1:
                            state['failures'] = max(0, state['failures'] - 0.5) # Reduce failure count
                            
                        current_objects[tid] = xywh
                    else:
                        state['failures'] += 10
                    
                    if state['failures'] > 5:
                        to_remove.append(tid)
            
            for tid in to_remove:
                del self.object_state[tid]
                if tid in current_objects:
                    del current_objects[tid]

        # --- STABILITY FILTER ---
        # Update Durations
        # Only pass objects that have existed for > 1.0 second (approx 30 frames at 30fps)
        valid_objects = {}
        
        for tid, xywh in current_objects.items():
            if tid not in self.object_duration:
                self.object_duration[tid] = 0
            
            self.object_duration[tid] += 1
            
            # Threshold: 30 frames (approx 1s)
            # Exception: Player is always valid if found
            is_player_candidate = (CLASS_MAP.get(self.object_state[tid]['cls']) == "character") # We need to be careful not to filter player if it's new
            # Actually, player selection happens later. 
            # If we filter here, we might lose the player.
            # But we want to filter "labels created".
            
            # Let's just track stable flags but pass everything to the logic that finds player.
            # Then filter the "enemies" list.
            
            is_stable = self.object_duration[tid] > 30
            self.object_state[tid]['is_stable'] = is_stable
            
            if is_stable:
               valid_objects[tid] = xywh

        # --- CONSTRUCT OUTPUT ---
        det_dict = {
            "player": None,
            "enemies": [],
            "items": [],
            "obstacles": [],
            "goal": None,
            "health": 1.0,
            "is_shake_hit": is_shake_hit
        }
        
        character_candidates = []
        
        # Use ALL objects for Player candidate search (player might be new)
        # But use VALID/STABLE objects for enemies/others
        
        for tid, xywh in current_objects.items():
            state = self.object_state[tid]
            cls_id = state['cls']
            conf = state['conf']
            is_stable = state.get('is_stable', False)
            
            # Update History
            self.track_histories[tid].append(xywh[:2])
            
            label = CLASS_MAP.get(cls_id, "unknown")
            
            if label == "character":
                # Add to candidates for player identification
                character_candidates.append((tid, xywh, conf, is_stable))
            elif label == "throwable" and is_stable:
                 # "make item detection confidence 100%" -> Only accept if conf is extremely high (effectively disabling it)
                if conf > 0.99:
                    det_dict["items"].append(xywh)

        # Identify Player
        # Updated: Center (offset left) Downward
        # x=0.45 (Left of center), y=0.6 (Downward)
        target_center = np.array([INPUT_SIZE * 0.45, INPUT_SIZE * 0.6])
        best_score = float('inf')
        player_candidate = None
        
        for tid, xywh, conf, is_stable in character_candidates:
            hist = self.track_histories[tid]
            cx, cy = xywh[:2]
            dist_to_center = np.linalg.norm(np.array([cx, cy]) - target_center)
            
            movement_score = 0.0
            if len(hist) >= 5:
                movement_score = np.linalg.norm(np.array(hist[-1]) - np.array(hist[-5]))
            
            # Increased weight for distance to center to prevent identity swapping
            score = (movement_score * 1.0) + (dist_to_center * 3.0)
            if score < best_score:
                best_score = score
                player_candidate = (tid, xywh, conf)

        
        # Health Logic
        raw_health = self.get_health(frame)
        
        # Hysteresis: If health jumps to 1.0 from [0.1, 0.8], ignore it (assume false positive/UI glitch)
        if raw_health > 0.95 and 0.1 <= self.filtered_health <= 0.8:
             current_health = self.filtered_health
        else:
             current_health = raw_health
             self.filtered_health = current_health

        det_dict["health"] = current_health
        p_damage_visual = 1.0 if (current_health < self.last_player_health - 0.005) else 0.0
        self.last_player_health = current_health
        
        if player_candidate:
            pid, p_xywh, p_conf = player_candidate
            
            # Player Ragdoll
            p_speed = 0.0
            if len(self.player_history) >= 2:
                prev_cx, prev_cy, _ = self.player_history[-1]
                p_speed = np.sqrt((p_xywh[0]-prev_cx)**2 + (p_xywh[1]-prev_cy)**2) / INPUT_SIZE
            
            p_ragdoll = self.detect_ragdoll(p_xywh, p_speed, pid)
            det_dict["player"] = (p_xywh[0], p_xywh[1], p_xywh[2], p_xywh[3], p_conf, p_damage_visual, p_ragdoll)
            
            # Enemies
            for tid, xywh, conf, is_stable in character_candidates:
                if tid != pid:
                    # ONLY Include Stable Enemies
                    if not is_stable:
                        continue
                        
                    # Analyze Enemy
                    state_str = self.analyze_enemy(tid, xywh, p_xywh, img)
                    
                    # HIT CONFIRMATION (The new logic)
                    hit_score = self.detect_hit_confirm(img, xywh, p_xywh, tid)
                    
                    e_ragdoll = self.detect_ragdoll(xywh, 0.0, tid)
                    
                    det_dict["enemies"].append((xywh[0], xywh[1], xywh[2], xywh[3], conf, state_str, hit_score, e_ragdoll))
        else:
            # Fallback if no player found
            if character_candidates:
                # Assume closest to center is player
                pass # Logic already handled by best_score loop above, if player_candidate is None, we have no player.

        return det_dict

    def detect_flash(self, gray_roi, tid):
        """
        Detects sudden brightness increase (White Flash) in ROI.
        Returns intensity of flash (0.0 to 1.0).
        """
        if gray_roi is None or gray_roi.size == 0: return 0.0
        
        # Calculate average brightness directly from Gray ROI
        current_brightness = np.mean(gray_roi)
        
        history = self.brightness_history[tid]
        flash_intensity = 0.0
        
        if len(history) > 0:
            avg_brightness = sum(history) / len(history)
            # Check for spike
            if current_brightness > avg_brightness + 30: # Threshold for flash
                flash_intensity = min(1.0, (current_brightness - avg_brightness) / 100.0)
        
        history.append(current_brightness)
        return flash_intensity

    def detect_ragdoll(self, xywh, velocity, tid):
        """
        Detects if character is ragdolled.
        Criteria: (Width > Height * 1.2) AND (Low Velocity) for > 15 frames.
        """
        w, h = xywh[2], xywh[3]
        
        # Aspect Ratio Check (Prone)
        is_prone = w > (h * 1.2)
        
        # Velocity Check (Still)
        # velocity is normalized, so threshold should be small
        is_still = velocity < 0.02 
        
        if is_prone and is_still:
            self.ragdoll_counters[tid] += 1
        else:
            self.ragdoll_counters[tid] = 0
            
        # Return normalized duration (capped at 1.0 for 60 frames)
        duration = self.ragdoll_counters[tid]
        if duration > 15:
            return min(1.0, (duration - 15) / 45.0)
        return 0.0

    def analyze_enemy(self, tid, xywh, p_xywh, frame):
        """
        Analyzes enemy movement and animation to determine intent.
        Returns a state string: "Idle", "Running", "Dashing", "Retreating"
        """
        cx, cy, w, h = xywh
        px, py, pw, ph = p_xywh
        
        # 1. Calculate Velocity relative to Player
        # We need history for this specific enemy
        hist = self.track_histories[tid]
        if len(hist) < 2:
            return "Idle"
            
        # Use average velocity over last few frames to smooth jitter
        window = 5
        if len(hist) >= window:
            start_cx, start_cy = hist[-window]
            # Raw screen displacement
            screen_dx = cx - start_cx
            screen_dy = cy - start_cy
            
            # Calculate Camera Displacement over the same period
            cam_dx = 0.0
            cam_dy = 0.0
            if len(self.global_motion_history) >= (window - 1):
                recent_motions = list(self.global_motion_history)[-(window-1):]
                for mx, my in recent_motions:
                    cam_dx += mx
                    cam_dy += my
            
            vx = (screen_dx - cam_dx) / (window - 1)
            vy = (screen_dy - cam_dy) / (window - 1)
        else:
            prev_cx, prev_cy = hist[-2]
            screen_dx = cx - prev_cx
            screen_dy = cy - prev_cy
            
            cam_dx, cam_dy = 0.0, 0.0
            if len(self.global_motion_history) > 0:
                cam_dx, cam_dy = self.global_motion_history[-1]
                
            vx = screen_dx - cam_dx
            vy = screen_dy - cam_dy
            
        speed = np.sqrt(vx*vx + vy*vy)
        
        # Direction to Player
        dx = px - cx
        dy = py - cy
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 0:
            dx /= dist
            dy /= dist
            
        # Dot product to check if moving towards player
        if speed > 0:
            norm_vx = vx / speed
            norm_vy = vy / speed
            dot = norm_vx * dx + norm_vy * dy
        else:
            dot = 0
            
        # 2. Determine State (Simplified - No Optical Flow)
        state = "Idle"
        
        # Thresholds
        SPEED_IDLE = 2.5 
        SPEED_FAST = 6.0 
        
        if speed > SPEED_FAST:
            if dot > 0.5: 
                state = "Dashing" 
            elif dot < -0.3: 
                state = "Retreating"
            else:
                state = "Strafing" 
        elif speed > SPEED_IDLE:
            if dot > 0.5:
                state = "Walking"
            elif dot < -0.3:
                state = "Retreating"
            else:
                state = "Moving"
        else:
            state = "Idle"
            
        self.enemy_states[tid] = state
        return state

    def get_health(self, frame):
        """
        Calculates health percentage from the top right health bar.
        Uses template matching with Health.png if available.
        """
        if frame is None: return 1.0
        
        h, w = frame.shape[:2]
        roi = None
        using_preset = False
        
        # 1. Try Template Matching first
        if self.health_template is not None:
            # Search in top right quadrant
            search_y = int(h * 0.25)
            search_x = int(w * 0.6)
            search_roi = frame[0:search_y, search_x:w]
            
            th, tw = self.health_template.shape[:2]
            sh, sw = search_roi.shape[:2]
            
            if th <= sh and tw <= sw:
                # Edge matching
                gray_search = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
                gray_template = cv2.cvtColor(self.health_template, cv2.COLOR_BGR2GRAY)
                
                edges_search = cv2.Canny(gray_search, 50, 150)
                edges_template = cv2.Canny(gray_template, 50, 150)
                
                res = cv2.matchTemplate(edges_search, edges_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                if max_val > 0.4: # Moderate threshold for edges
                    top_left = max_loc
                    # Global coords
                    gx = search_x + top_left[0]
                    gy = top_left[1]
                    
                    # ROI is the matched area
                    roi = frame[gy:gy+th, gx:gx+tw]
                    using_preset = True # Treat as preset/fixed ROI
                    
                    # If we have the bar box from template, crop to it
                    if self.health_bar_box:
                        bx, by, bw, bh = self.health_bar_box
                        # Ensure bounds
                        bx = max(0, min(bx, tw-1))
                        by = max(0, min(by, th-1))
                        bw = max(1, min(bw, tw - bx))
                        bh = max(1, min(bh, th - by))
                        
                        roi = roi[by:by+bh, bx:bx+bw]
                        # Update debug visualization to show this crop
                        # We can't easily update the main frame debug, but roi is used below
        
        if roi is None:
            if self.health_roi:
                rx, ry, rw, rh = self.health_roi
                # Ensure within bounds
                rx = max(0, min(rx, w-1))
                ry = max(0, min(ry, h-1))
                rw = max(1, min(rw, w - rx))
                rh = max(1, min(rh, h - ry))
                
                roi = frame[ry:ry+rh, rx:rx+rw]
                using_preset = True
            else:
                # ROI: Top right corner. Adjust based on game UI.
                # Assuming 1920x1080, top right 300x50 area.
                # Scaled to INPUT_SIZE (640x640)
                # Top right is x > 0.8*W, y < 0.1*H
                roi_x1 = int(w * 0.80) 
                roi_y1 = int(h * 0.0)
                roi_x2 = int(w * 1.0)
                roi_y2 = int(h * 0.15)
                
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
        if roi is None or roi.size == 0: return 1.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define ranges for Green, Yellow, Red
        # Green
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Yellow
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Red
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50]) # Widened range (was 170)
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        
        # Combine masks
        mask_health = cv2.bitwise_or(mask_green, mask_yellow)
        mask_health = cv2.bitwise_or(mask_health, mask_red)
        
        # Debug Visualization: Create BGR version of mask
        vis_health = cv2.cvtColor(mask_health, cv2.COLOR_GRAY2BGR)
        
        if using_preset:
            # Logic: Bar drains Right-to-Left.
            # 1. Check Left side (Anchor). If Left is empty, Bar is likely hidden (Full HP).
            # 2. If Left is present, scan Left-to-Right to find where it ends.
            
            h_roi, w_roi = roi.shape[:2]
            if h_roi == 0 or w_roi == 0: 
                self.debug_health = vis_health
                return 1.0
            
            # Analyze columns
            # Sum of white pixels in each column
            col_sums = np.sum(mask_health, axis=0) / 255 # shape (w_roi,)
            
            # Threshold for a "filled" column
            # We assume the bar fills a significant portion of the vertical height of the ROI
            # Let's say at least 30% of the pixels in the column must be colored
            fill_threshold = h_roi * 0.3
            
            # Check Anchor (First 10% of width)
            anchor_width = max(1, int(w_roi * 0.1))
            anchor_cols = col_sums[:anchor_width]
            filled_anchor_cols = np.sum(anchor_cols > fill_threshold)
            
            # Draw Anchor Box on Debug
            cv2.rectangle(vis_health, (0, 0), (anchor_width, h_roi), (255, 0, 0), 1)
            
            # If less than 20% (relaxed from 50%) of the anchor columns are filled, assume bar is hidden
            # Also check if the *entire* bar is empty (sum of all cols)
            total_filled = np.sum(col_sums > fill_threshold)
            
            if filled_anchor_cols < (anchor_width * 0.2) and total_filled < (w_roi * 0.1):
                cv2.putText(vis_health, "Hidden (Full)", (5, h_roi//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                self.debug_health = vis_health
                return 1.0
            
            # Calculate Fill Percentage
            # Scan from Left to Right to find the "end" of the bar
            # We look for the last column that is "filled" and connected to the start.
            # This handles noise on the right side (empty space) better.
            
            last_filled_idx = -1
            
            # We allow small gaps (e.g. 1-2 pixels) to account for noise/grid lines
            gap_tolerance = 2
            current_gap = 0
            
            for i in range(w_roi):
                if col_sums[i] > fill_threshold:
                    last_filled_idx = i
                    current_gap = 0
                else:
                    current_gap += 1
                    if current_gap > gap_tolerance:
                        break # Stop if gap is too large
            
            # Draw End Line
            if last_filled_idx >= 0:
                cv2.line(vis_health, (last_filled_idx, 0), (last_filled_idx, h_roi), (0, 0, 255), 2)
            
            # Health is ratio of filled width to total width
            health_pct = (last_filled_idx + 1) / w_roi
            
            cv2.putText(vis_health, f"{health_pct*100:.0f}%", (w_roi//2, h_roi//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self.debug_health = vis_health
            
            return min(1.0, max(0.0, health_pct))
        else:
            # Fallback Heuristic: Find largest contour width
            contours, _ = cv2.findContours(mask_health, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Find largest contour (the bar)
                c = max(contours, key=cv2.contourArea)
                x, y, w_bar, h_bar = cv2.boundingRect(c)
                
                # Assuming max width is the width of the ROI (minus some padding)
                # Let's normalize by ROI width for now
                health_pct = w_bar / (roi.shape[1] * 0.9) # 90% of ROI width is max
                
                cv2.rectangle(vis_health, (x, y), (x+w_bar, y+h_bar), (0, 255, 0), 1)
                self.debug_health = vis_health
                return min(1.0, health_pct)
            
            self.debug_health = vis_health
            return 0.0

    def get_enemy_health(self, frame, enemy_box):
        """
        Estimates health of a specific enemy.
        enemy_box: (x, y, w, h)
        """
        if frame is None or enemy_box is None: return 1.0
        
        x, y, w, h = enemy_box
        
        # ROI: Above the head
        # Assume bar is roughly same width as character, and slightly above
        bar_w = int(w * 1.2)
        bar_h = int(h * 0.2) # 20% of height
        
        bx = int(x - (bar_w - w)/2)
        by = int(y - h/2 - bar_h - 5) # 5 pixels gap
        
        # Clamp
        bx = max(0, min(bx, frame.shape[1] - 1))
        by = max(0, min(by, frame.shape[0] - 1))
        bar_w = min(bar_w, frame.shape[1] - bx)
        bar_h = min(bar_h, frame.shape[0] - by)
        
        if bar_w <= 0 or bar_h <= 0: return 1.0
        
        roi = frame[by:by+bar_h, bx:bx+bar_w]
        
        # If template exists, use it to refine ROI
        if self.enemy_health_template is not None:
            th, tw = self.enemy_health_template.shape[:2]
            rh, rw = roi.shape[:2]
            
            # Check if ROI is large enough for template
            if rh >= th and rw >= tw:
                # Match template in this larger ROI
                res = cv2.matchTemplate(roi, self.enemy_health_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                if max_val > 0.5:
                    # Found template
                    tx, ty = max_loc
                    roi = roi[ty:ty+th, tx:tx+tw]
        
        # Calculate Health (Red/Green ratio)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Red Mask (Enemy bars are usually red)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        
        # Calculate fill
        # Assuming bar fills from left? Or is it centered?
        # Usually centered or left.
        # Simple pixel count ratio for now
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0: return 1.0
        
        red_pixels = cv2.countNonZero(mask_red)
        
        # Heuristic: If > 5% red, assume it's the bar
        # But we need percentage.
        # If we found the template, we know the max width.
        # If not, we are guessing.
        
        # Let's assume the ROI width IS the max width.
        health_pct = red_pixels / (total_pixels * 0.8) # Normalize
        
        return min(1.0, health_pct)

    def scan_ui(self, frame):
        """
        Scans the frame for UI elements and updates self.ui_rois.
        """
        if frame is None: return
        h, w = frame.shape[:2]
        
        # Search area: Bottom half for moves/bars
        search_roi = frame[int(h*0.5):h, 0:w]
        
        for name, tmpl in self.ui_templates.items():
            # Use matchTemplate to find the UI element
            res = cv2.matchTemplate(search_roi, tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > 0.8: # High confidence
                th, tw = tmpl.shape[:2]
                tx, ty = max_loc
                # Convert back to global coords
                gx = tx
                gy = ty + int(h*0.5)
                self.ui_rois[name] = (gx, gy, tw, th)
                print(f"Found UI Element: {name} at {gx},{gy}")

    def detect_cooldowns(self, frame):
        """
        Returns list of 4 floats (0.0=Ready, 1.0=Cooldown) for moves 1-4.
        """
        cooldowns = [0.0, 0.0, 0.0, 0.0]
        if frame is None: return cooldowns
        
        for i in range(1, 5):
            name = f"move_{i}"
            if name in self.ui_rois:
                x, y, w, h = self.ui_rois[name]
                # Ensure bounds
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0: continue
                
                # Check for Blue Tint (Cooldown Overlay)
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Define Blue Range
                # OpenCV HSV: H(0-179), S(0-255), V(0-255)
                # Blue is around 120. Range 100-140.
                lower_blue = np.array([100, 50, 50])
                upper_blue = np.array([140, 255, 255])
                
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                blue_ratio = cv2.countNonZero(mask) / (w * h)
                
                # Threshold: If > 20% blue, assume cooldown
                if blue_ratio > 0.2:
                    cooldowns[i-1] = 1.0
                    
        return cooldowns

    def detect_bars(self, frame):
        """
        Detects Mode, Evasive, and Special bars.
        Returns (mode_lvl, evasive_lvl, special_lvl)
        """
        evasive = 0.0
        special = 0.0
        mode = 0.0 # Placeholder
        
        # Check Evasive
        if "evasive_bar" in self.ui_rois:
            x, y, w, h = self.ui_rois["evasive_bar"]
            roi = frame[y:y+h, x:x+w]
            tmpl = self.ui_templates["evasive_bar"]
            
            if roi.shape == tmpl.shape:
                res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > 0.8:
                    evasive = 1.0
        
        # Check Special
        if "special_bar" in self.ui_rois:
            x, y, w, h = self.ui_rois["special_bar"]
            roi = frame[y:y+h, x:x+w]
            tmpl = self.ui_templates["special_bar"]
            
            if roi.shape == tmpl.shape:
                res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > 0.8:
                    special = 1.0
                    
        return mode, evasive, special
        

    def compute_velocity(self, cx, cy):
        """
        Computes velocity based on history buffer.
        Returns (vx, vy) normalized.
        """
        now = time.time()
        self.player_history.append((cx, cy, now))
        
        if len(self.player_history) < 2:
            return 0.0, 0.0
        
        # Simple finite difference between last two points
        # (cx_t - cx_t-1)
        prev_cx, prev_cy, prev_time = self.player_history[-2]
        dt = now - prev_time
        if dt <= 0: return 0.0, 0.0
        
        # Velocity in pixels per second? Or just per frame?
        # Prompt says: "vx = (cx_t - cx_t-1)" -> per frame difference.
        # But normalized.
        
        screen_vx = (cx - prev_cx)
        screen_vy = (cy - prev_cy)
        
        # Compensate for camera motion to get world velocity
        cam_dx, cam_dy = 0.0, 0.0
        if len(self.global_motion_history) > 0:
            cam_dx, cam_dy = self.global_motion_history[-1]
            
        vx = screen_vx - cam_dx
        vy = screen_vy - cam_dy
        
        # Normalize by Input Size to keep it in small range [-1, 1] roughly
        vx /= INPUT_SIZE
        vy /= INPUT_SIZE
        
        # If time difference is very small (e.g. same frame or super fast), 
        # we might want to scale by dt to get per-second velocity if needed.
        # But for RL, per-step velocity is often fine if step rate is constant.
        # However, the test expects > 0.
        
        return vx, vy

    def estimate_depth(self, frame):
        """
        Estimates depth map from frame using MiDaS.
        Returns normalized depth map (0..1) as numpy array [H, W].
        """
        if self.depth_model is None or self.depth_transform is None:
            return np.zeros(frame.shape[:2], dtype=np.float32)
            
        # Optimization: Cache depth map
        if self.frame_count % self.depth_interval != 0 and self.last_depth_map is not None:
            return self.last_depth_map

        try:
            # Optimization: Resize frame to smaller size for faster inference
            # MiDaS Small works well with 256x256 or 384x384
            small_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            input_batch = self.depth_transform(small_frame).to(self.device)
            
            with torch.no_grad():
                prediction = self.depth_model(input_batch)
                
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
            depth_map = prediction.cpu().numpy()
            
            # Normalize to 0..1
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max - depth_min > 1e-6:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_map = np.zeros_like(depth_map)
            
            self.last_depth_map = depth_map
            return depth_map
        except Exception as e:
            # print(f"Depth Error: {e}")
            return np.zeros(frame.shape[:2], dtype=np.float32)

    def compute_raycasts(self, depth_map, grid_size=10, num_radial=16):
        """
        Samples depth map at a grid of points AND radial directions.
        Returns flattened array of depth values.
        """
        h, w = depth_map.shape
        
        # 1. Grid Sampling (10x10)
        step_x = w // (grid_size + 1)
        step_y = h // (grid_size + 1)
        
        grid_rays = []
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                y = i * step_y
                x = j * step_x
                # Sample 3x3 area for robustness
                # Ensure bounds
                y1, y2 = max(0, y-1), min(h, y+2)
                x1, x2 = max(0, x-1), min(w, x+2)
                val = np.mean(depth_map[y1:y2, x1:x2])
                grid_rays.append(val)
        
        # 2. Radial Sampling (16 directions from center)
        radial_rays = []
        cx, cy = w // 2, h // 2
        max_radius = min(cx, cy)
        
        for i in range(num_radial):
            angle = (2 * np.pi * i) / num_radial
            
            # Walk the ray to find obstacle distance
            hit_dist = 1.0 # Default max distance (normalized)
            
            # We walk from center outwards
            # Start a bit away from player center to avoid self-occlusion
            for r in range(20, max_radius, 5): 
                tx = int(cx + r * np.cos(angle))
                ty = int(cy + r * np.sin(angle))
                
                if tx < 0 or tx >= w or ty < 0 or ty >= h:
                    break
                
                d_val = depth_map[ty, tx]
                # Assuming depth_map is 0 (far) to 1 (close)
                # If d_val > 0.5, we hit something close
                if d_val > 0.5: 
                    hit_dist = r / max_radius
                    break
            
            radial_rays.append(hit_dist)
                
        return np.concatenate([np.array(grid_rays, dtype=np.float32), np.array(radial_rays, dtype=np.float32)])

    def get_obs(self, frame, last_action=None, mouse_movement=(0,0)):
        """
        Main function to get observation vector from frame.
        Returns np.array of shape (151,)
        """
        # Performance check
        t0 = time.time()
        
        # Periodic UI Scan (every 60 frames) or if missing
        if self.frame_count % 60 == 0 or not self.ui_rois:
            self.scan_ui(frame)
        
        det_dict = self.detect(frame, mouse_movement)
        self.last_det = det_dict # Save for visualization
        
        # Extract Player
        player = det_dict.get("player")
        
        p_flash = 0.0
        p_ragdoll = 0.0
        
        if player is not None:
            if len(player) >= 7:
                cx, cy, w, h, _, p_flash, p_ragdoll = player[:7]
            elif len(player) == 5:
                cx, cy, w, h, _ = player
            else:
                cx, cy, w, h = player[:4]
                
            vx, vy = self.compute_velocity(cx, cy)
            
            # Normalize Pos (0..1)
            px_norm = cx / INPUT_SIZE
            py_norm = cy / INPUT_SIZE
        else:
            # Fallback
            if len(self.player_history) > 0:
                cx, cy, _ = self.player_history[-1]
                px_norm = cx / INPUT_SIZE
                py_norm = cy / INPUT_SIZE
                vx, vy = 0.0, 0.0
            else:
                px_norm, py_norm = 0.5, 0.5
                vx, vy = 0.0, 0.0
                cx, cy = INPUT_SIZE/2, INPUT_SIZE/2

        # Nearest Enemy
        enemy_dx, enemy_dy = 1.0, 1.0 # Default: Far away
        enemy_flash = 0.0
        found_enemy = False
        best_enemy = None # Initialize best_enemy
        
        # Screen Center (for aiming)
        screen_cx = INPUT_SIZE * 0.5
        screen_cy = INPUT_SIZE * 0.5
        
        if det_dict["enemies"]:
            # Find nearest to player (cx, cy)
            best_dist = float('inf')
            
            for item in det_dict["enemies"]:
                # Handle variable tuple lengths
                if len(item) >= 4:
                    ecx, ecy, ew, eh = item[:4]
                else:
                    continue

                # Calculate distance to player for selection
                p_dx = (ecx - cx)
                p_dy = (ecy - cy)
                dist_p = p_dx*p_dx + p_dy*p_dy
                
                if dist_p < best_dist:
                    best_dist = dist_p
                    # Calculate aiming delta (relative to screen center)
                    enemy_dx = (ecx - screen_cx) / INPUT_SIZE
                    enemy_dy = (ecy - screen_cy) / INPUT_SIZE
                    best_enemy = item
            
            if best_enemy and len(best_enemy) >= 7:
                # item: (x, y, w, h, conf, state, flash, ragdoll)
                enemy_flash = best_enemy[6]
            
            # Update Memory
            self.last_enemy_pos = (enemy_dx, enemy_dy)
            self.last_enemy_time = time.time()
            found_enemy = True
        else:
            # Use Memory if recent (< 1.0s)
            if time.time() - self.last_enemy_time < 1.0:
                enemy_dx, enemy_dy = self.last_enemy_pos
        
        # Time since last seen (Normalized: 0..1 for 0..5s)
        time_since_seen = time.time() - self.last_enemy_time
        time_since_seen_norm = min(time_since_seen / 5.0, 1.0)
        
        # Nearest Obstacle
        obs_dx, obs_dy = 1.0, 1.0 # Default: Far away
        if det_dict["obstacles"]:
            best_dist = float('inf')
            for (ocx, ocy, ow, oh) in det_dict["obstacles"]:
                dx = (ocx - cx) / INPUT_SIZE
                dy = (ocy - cy) / INPUT_SIZE
                dist = dx*dx + dy*dy
                if dist < best_dist:
                    best_dist = dist
                    obs_dx = dx
                    obs_dy = dy

        # Goal
        dist_goal = 1.0 # Default high
        if det_dict["goal"] is not None:
            gcx, gcy, gw, gh = det_dict["goal"]
            dx = (gcx - cx) / INPUT_SIZE
            dy = (gcy - cy) / INPUT_SIZE
            dist_goal = np.sqrt(dx*dx + dy*dy)

        # Player Bars
        mode_lvl, evasive_lvl, special_lvl = self.detect_bars(frame)
        
        # Cooldowns
        cooldowns = self.detect_cooldowns(frame)
        
        # Calculate Relative Size (Enemy Area / Player Area)
        relative_size = 0.0
        is_overlapping = 0.0
        leaks_above = 0.0
        
        # Player Box Area (Zoom Estimation)
        player_area_norm = 0.0
        
        # Depth & Raycasts
        depth_map = self.estimate_depth(frame)
        raycasts = self.compute_raycasts(depth_map, grid_size=10, num_radial=16) # 100 + 16 = 116 values
        
        # Occlusion & Relative Position
        is_occluded = 0.0
        
        if player is not None:
            # player: (cx, cy, w, h, ...)
            p_area = player[2] * player[3]
            player_area_norm = p_area / (INPUT_SIZE * INPUT_SIZE)
            
            if found_enemy and best_enemy is not None:
                # best_enemy: (x, y, w, h, ...)
                e_area = best_enemy[2] * best_enemy[3]
                if p_area > 0:
                    relative_size = e_area / p_area
                
                # Check Overlap
                # Convert center-xywh to top-left-xywh
                px, py, pw, ph = player[:4]
                ex, ey, ew, eh = best_enemy[:4]
                
                px1, py1 = px - pw/2, py - ph/2
                px2, py2 = px + pw/2, py + ph/2
                
                ex1, ey1 = ex - ew/2, ey - eh/2
                ex2, ey2 = ex + ew/2, ey + eh/2
                
                # Overlap condition
                if (px1 < ex2 and px2 > ex1 and py1 < ey2 and py2 > ey1):
                    is_overlapping = 1.0
                    
                # Leaks Above Condition (Enemy top is higher than Player top)
                # In image coords, higher means smaller Y
                if ey1 < py1:
                    leaks_above = 1.0
                    
                # Occlusion Check
                # Sample depth at enemy center
                cx_int, cy_int = int(ex), int(ey)
                if 0 <= cx_int < INPUT_SIZE and 0 <= cy_int < INPUT_SIZE:
                     d_val = depth_map[cy_int, cx_int]
                     # Heuristic: If depth is HIGH (close) but enemy is SMALL (far), it's an occlusion.
                     # Expected depth ~ sqrt(area)
                     expected_d = np.sqrt(e_area) / INPUT_SIZE * 5.0 
                     if d_val > (expected_d + 0.3): 
                         is_occluded = 1.0
        
        # Ego Motion (Global Flow)
        g_dx, g_dy = 0.0, 0.0
        if len(self.global_motion_history) > 0:
            g_dx, g_dy = self.global_motion_history[-1]
            # Normalize roughly (-1..1 for typical fast movement)
            g_dx /= 20.0
            g_dy /= 20.0
            
        # New Features: Heading, Relative Pos, Motion Prediction
        heading_to_target = np.arctan2(enemy_dy, enemy_dx) # Radians
        
        is_left = 1.0 if enemy_dx < -0.1 else 0.0
        is_right = 1.0 if enemy_dx > 0.1 else 0.0
        is_above = 1.0 if enemy_dy < -0.1 else 0.0 # Up on screen
        is_below = 1.0 if enemy_dy > 0.1 else 0.0 # Down on screen
        
        # Motion Prediction (Kalman)
        # self.kalman.statePost -> [x, y, vx, vy]
        # We want predicted displacement
        k_state = self.kalman.statePost
        # These are in pixels/frame (internal state)
        # We want normalized predicted velocity
        pred_vx = float(k_state[2]) / INPUT_SIZE * 10.0 # Scale up
        pred_vy = float(k_state[3]) / INPUT_SIZE * 10.0
        
        # Build Vector
        # Base: 27 + 8 = 35
        # Raycasts: 116
        # Total: 151
        
        health = det_dict.get("health", 1.0)
        is_shake_hit = det_dict.get("is_shake_hit", 0.0)
        
        base_obs = [
            px_norm, py_norm, 
            vx, vy, 
            enemy_dx, enemy_dy, 
            obs_dx, obs_dy, 
            dist_goal,
            time_since_seen_norm,
            health,        
            enemy_flash,   
            mode_lvl,
            evasive_lvl,
            special_lvl,
            p_ragdoll,     
            cooldowns[0], cooldowns[1], cooldowns[2], cooldowns[3],
            relative_size,
            is_overlapping,
            leaks_above,
            is_shake_hit,
            player_area_norm,
            g_dx, g_dy,
            # New Features
            heading_to_target,
            is_left, is_right, is_above, is_below,
            is_occluded,
            pred_vx, pred_vy
        ]
        
        obs = np.concatenate([np.array(base_obs, dtype=np.float32), raycasts])
        
        # Sanity check / Clip
        obs = np.nan_to_num(obs, nan=0.0)
        obs = np.clip(obs, -1.0, 1.0) 
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            fps = 1.0 / dt if dt > 0 else 0
            self.fps_log.append(fps)
            if len(self.fps_log) >= 10:
                avg_fps = sum(self.fps_log) / len(self.fps_log)
                with open("perf_log.txt", "a") as f:
                    f.write(f"Timestamp: {time.time():.2f}, FPS: {avg_fps:.2f}\n")
                self.fps_log = []

        return obs

    def get_pixel_obs(self, frame):
        """
        Generates pixel-based observation dictionary.
        Returns:
            {
                'full': np.array [16, 160, 160], # 4 frames * (RGB + Depth)
                'crop': np.array [16, 128, 128],
                'flow': np.array [2, 160, 160]
            }
        """
        if frame is None: return None
        
        # Depth Estimation
        depth_map = self.estimate_depth(frame) # [H, W] 0..1
        self.last_depth_vis = (depth_map * 255).astype(np.uint8)
        self.last_depth_vis = cv2.applyColorMap(self.last_depth_vis, cv2.COLORMAP_MAGMA)
        
        # 1. Full Frame (160x160)
        full_img = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_LINEAR)
        full_depth = cv2.resize(depth_map, (160, 160), interpolation=cv2.INTER_LINEAR)
        
        # Normalize RGB to 0-1
        full_norm = full_img.astype(np.float32) / 255.0
        
        # Stack RGB + Depth -> [160, 160, 4]
        full_combined = np.dstack([full_norm, full_depth])
        
        # Transpose to [C, H, W] -> [4, 160, 160]
        full_tensor = np.transpose(full_combined, (2, 0, 1))
        
        # 2. Fine Crop (128x128)
        # Use cached detection if available (populated by get_obs)
        if self.last_det:
            det_dict = self.last_det
        else:
            det_dict = self.detect(frame)
            self.last_det = det_dict
        
        player = det_dict.get("player")
        
        # Kalman Filter Prediction & Update
        pred = self.kalman.predict()
        # Extract scalar values
        pred_x = float(pred[0])
        pred_y = float(pred[1])
        
        if player:
            cx, cy = player[:2]
            # Correct with measurement
            meas = np.array([[np.float32(cx)], [np.float32(cy)]])
            self.kalman.correct(meas)
            # Use corrected state
            state = self.kalman.statePost
            cx = float(state[0])
            cy = float(state[1])
        else:
            # Use prediction if no detection
            cx, cy = pred_x, pred_y
            
        self.last_kalman_pred = (cx, cy)
        
        # Map cx, cy (which are in INPUT_SIZE 640 coords) to frame coords
        h, w = frame.shape[:2]
        scale_x = w / INPUT_SIZE
        scale_y = h / INPUT_SIZE
        
        pcx = int(cx * scale_x)
        pcy = int(cy * scale_y)
        
        # Crop Size: 128x128 (Bigger)
        crop_size = 128
        half_crop = crop_size // 2
        
        # Offset: Shift Upwards to see in front
        # Shift by 1/4 of crop size
        offset_y = int(crop_size * 0.25)
        
        center_x = pcx
        center_y = max(half_crop, pcy - offset_y)
        
        x1 = max(0, center_x - half_crop)
        y1 = max(0, center_y - half_crop)
        x2 = min(w, center_x + half_crop)
        y2 = min(h, center_y + half_crop)
        
        # Ensure valid crop
        if x1 >= x2 or y1 >= y2:
            # Fallback to center if invalid
            x1 = max(0, w//2 - half_crop)
            y1 = max(0, h//2 - half_crop)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
        
        crop_img = frame[y1:y2, x1:x2]
        crop_depth = depth_map[y1:y2, x1:x2]
        
        # Pad if out of bounds
        if crop_img.shape[0] != crop_size or crop_img.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            padded_depth = np.zeros((crop_size, crop_size), dtype=np.float32)
            ph, pw = crop_img.shape[:2]
            
            # Safety check for broadcasting error
            if ph > crop_size or pw > crop_size:
                # This should not happen given x1, x2 logic, but if it does, resize
                crop_img = cv2.resize(crop_img, (crop_size, crop_size))
                crop_depth = cv2.resize(crop_depth, (crop_size, crop_size))
                padded = crop_img
                padded_depth = crop_depth
            else:
                padded[:ph, :pw] = crop_img
                padded_depth[:ph, :pw] = crop_depth
            
            crop_img = padded
            crop_depth = padded_depth
            
        crop_norm = crop_img.astype(np.float32) / 255.0
        
        # Stack RGB + Depth -> [128, 128, 4]
        crop_combined = np.dstack([crop_norm, crop_depth])
        
        crop_tensor = np.transpose(crop_combined, (2, 0, 1))
        
        # 3. Optical Flow (Farneback)
        # Compute on 160x160 gray
        gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        
        if self.last_gray is None:
            flow = np.zeros((160, 160, 2), dtype=np.float32)
        else:
            # Initialize flow array to avoid None type error
            flow_init = np.zeros((160, 160, 2), dtype=np.float32)
            flow = cv2.calcOpticalFlowFarneback(self.last_gray, gray, flow_init, 0.5, 3, 15, 3, 5, 1.2, 0)
            
        self.last_gray = gray
        
        # Visualization of flow
        # Map flow to HSV
        hsv = np.zeros((160, 160, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # Manual normalization to avoid cv2.normalize NoneType issues
        if np.max(mag) > 0:
            hsv[..., 2] = (mag / np.max(mag) * 255).astype(np.uint8)
        else:
            hsv[..., 2] = 0
            
        self.last_flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Normalize Flow
        # flow = flow / (np.max(|flow|) + 1e-6)
        max_mag = np.max(mag)
        if max_mag > 1e-6:
            flow = flow / max_mag
        else:
            flow = flow * 0 # Avoid noise
            
        flow_tensor = np.transpose(flow, (2, 0, 1)) # [2, 160, 160]
        
        # 4. Stack Frames
        self.frame_stack.append(full_tensor)
        self.crop_stack.append(crop_tensor)
        
        # Fill stack if empty
        while len(self.frame_stack) < 4:
            self.frame_stack.append(full_tensor)
            self.crop_stack.append(crop_tensor)
            
        # Concatenate Stacks
        # [4, 4, 160, 160] -> [16, 160, 160]
        full_stack = np.concatenate(list(self.frame_stack), axis=0)
        crop_stack = np.concatenate(list(self.crop_stack), axis=0)
        
        return {
            'full': full_stack,
            'crop': crop_stack,
            'flow': flow_tensor
        }

    def visualize_obs(self):
        """
        Returns a visualization image of the current observation state.
        """
        if not self.frame_stack or not self.crop_stack:
            return np.zeros((300, 300, 3), dtype=np.uint8)
            
        # Get latest frames from stack
        # frame_stack: [C, H, W] where C=16 (4 channels * 4 frames)
        # We want the last 4 channels (latest frame: RGBD)
        
        # full_tensor is [4, 160, 160] (RGBD)
        full_tensor = self.frame_stack[-1] 
        crop_tensor = self.crop_stack[-1] 
        
        # Extract RGB [3, H, W]
        full_rgb = full_tensor[:3, :, :]
        crop_rgb = crop_tensor[:3, :, :]

        # Extract Depth [1, H, W] -> [H, W]
        full_depth = full_tensor[3, :, :]
        
        # Convert back to HWC and uint8
        full_img = (np.transpose(full_rgb, (1, 2, 0)) * 255).astype(np.uint8)
        crop_img = (np.transpose(crop_rgb, (1, 2, 0)) * 255).astype(np.uint8)

        # Convert Depth to Color Map for visualization
        depth_vis = (full_depth * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        
        # Create a composite image
        # Canvas: 400x350
        canvas = np.zeros((350, 400, 3), dtype=np.uint8)
        
        # Place Full (160x160) at (0,0)
        h, w = full_img.shape[:2]
        canvas[0:h, 0:w] = full_img
        cv2.putText(canvas, "Full Input", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Place Crop (128x128) at (170, 0)
        ch, cw = crop_img.shape[:2]
        canvas[0:ch, 170:170+cw] = crop_img
        cv2.putText(canvas, "Crop Input", (175, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Place Flow if available
        if hasattr(self, 'last_flow_vis') and self.last_flow_vis is not None:
             fh, fw = self.last_flow_vis.shape[:2]
             # Resize to fit if needed, flow is 160x160
             canvas[170:170+fh, 0:fw] = self.last_flow_vis
             cv2.putText(canvas, "Flow Input", (5, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Place Depth at (170, 170)
        dh, dw = depth_vis.shape[:2]
        canvas[170:170+dh, 170:170+dw] = depth_vis
        
        # Debug Text for Depth
        d_min = full_depth.min()
        d_max = full_depth.max()
        cv2.putText(canvas, f"Depth: {d_min:.2f}-{d_max:.2f}", (175, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Debug Text for UI / Cooldowns
        # We need to access the latest cooldowns. They are not stored in the stack.
        # We can try to read them from the last observation if we had access, but we don't here easily.
        # However, we can check self.ui_rois count.
        ui_count = len(self.ui_rois)
        cv2.putText(canvas, f"UI ROIs: {ui_count}", (5, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Check if MiDaS is loaded
        midas_status = "OK" if self.depth_model is not None else "FAIL"
        cv2.putText(canvas, f"MiDaS: {midas_status}", (175, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
             
        return canvas
        

# Global instance
perception_pipeline = None

def init_perception(model_path=None, fast_mode=False):
    global perception_pipeline
    if perception_pipeline is None:
        if model_path:
            perception_pipeline = Perception(model_path=model_path, fast_mode=fast_mode)
        else:
            perception_pipeline = Perception(fast_mode=fast_mode)
    return perception_pipeline

if __name__ == "__main__":
    # Test run
    print("Initializing Perception...")
    pipeline = init_perception()
    
    # Dummy frame
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    print("Running inference loop...")
    try:
        while True:
            obs = pipeline.get_obs(dummy_frame)
            # print(obs)
            # time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopped.")
