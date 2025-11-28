import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from collections import deque, defaultdict
import json
import os

# --- Configuration ---
MODEL_PATH = r"T:\Auto-Farmer-Data\runs\combined_training_p3\weights\best.pt" # Updated to combined model
INPUT_SIZE = 640
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HALF_PRECISION = True if DEVICE == 'cuda' else False

print(f"Perception Config: Device={DEVICE}, FP16={HALF_PRECISION}, Torch={torch.__version__}, CUDA={torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# Class mapping (Adjust based on your model's training)
# Based on pipeline.py CLASSES list (Obstacles removed)
CLASS_MAP = {
    0: "character",
    1: "throwable"
}
# Fallback for COCO if using pretrained yolo11n.pt directly without custom training
# We will assume 'person' (0) is player/enemy for now if not custom.

class Perception:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE, half=HALF_PRECISION):
        self.device = device
        self.half = half
        self.model = self._init_model(model_path)
        
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
        self.prev_pts = None # For KLT tracking
        self.global_motion_history = deque(maxlen=30) # Store (dx, dy) per frame
        
        # Health ROI
        self.health_roi = None
        self.health_template = None
        self.health_bar_box = None # (x, y, w, h) relative to template
        
        # Enemy Health Template
        self.enemy_health_template = None
        if os.path.exists("EnemyHealth.png"):
            self.enemy_health_template = cv2.imread("EnemyHealth.png")
            if self.enemy_health_template is not None:
                print("Loaded EnemyHealth.png template.")
        
        # UI Templates for Cooldowns and Bars
        self.ui_templates = {}
        self.ui_rois = {} # name -> (x, y, w, h)
        
        ui_files = {
            "move_1": "Lapse Blue.png",
            "move_2": "Reversal Red.png",
            "move_3": "Rapid Punches.png",
            "move_4": "Twofold Kick.png",
            "evasive_bar": "EvasiveBar.png",
            "special_bar": "SpecialBar.png"
        }
        
        for name, filename in ui_files.items():
            if os.path.exists(filename):
                tmpl = cv2.imread(filename)
                if tmpl is not None:
                    self.ui_templates[name] = tmpl
                    print(f"Loaded UI template: {filename}")
        
        if os.path.exists("Health.png"):
            self.health_template = cv2.imread("Health.png")
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
        
        # --- Pixel-Based State ---
        self.frame_stack = deque(maxlen=4) # Stores last 4 frames (160x160)
        self.crop_stack = deque(maxlen=4)  # Stores last 4 crops (96x96)
        self.last_gray = None # For optical flow
        
        # --- New Logic State ---
        self.brightness_history = defaultdict(lambda: deque(maxlen=5))
        self.ragdoll_counters = defaultdict(int)
        self.last_player_health = 1.0

        # --- Optimization & Advanced Hit Detection ---
        self.detection_interval = 4 # Run YOLO every 4 frames
        self.trackers = {} # ID -> cv2.Tracker
        self.tracker_failures = defaultdict(int)
        
        # Hit Detection State
        self.hit_signals = defaultdict(lambda: deque(maxlen=3)) # ID -> deque of bools
        self.prev_edges = {} # ID -> edge_map
        self.hit_cooldowns = defaultdict(float) # ID -> timestamp

    def _create_tracker(self):
        """Creates a lightweight tracker (KCF or CSRT)."""
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
                        print("Warning: No tracker found. Tracking disabled.")
                        return None

    def detect_hit_confirm(self, frame, enemy_box, player_box, tid):
        """
        Advanced Hit Detection Pipeline:
        1. Expand Box & Crop
        2. HSV Flash Detection (V > thresh, S < thresh)
        3. Edge Change Detection (Canny Diff)
        4. Player Suppression
        5. Signal Smoothing
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
        
        # 2. HSV Flash Mask
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # White Flash: High Value (>200), Low Saturation (<50)
        # Adjust thresholds as needed
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        flash_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Cluster Size Check (Remove noise)
        # Morphological Open
        kernel = np.ones((3,3), np.uint8)
        flash_mask = cv2.morphologyEx(flash_mask, cv2.MORPH_OPEN, kernel)
        
        flash_pixels = cv2.countNonZero(flash_mask)
        has_flash = flash_pixels > 20 # Threshold for cluster size
        
        # 3. Edge Change Map (Impact)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        has_edge_change = False
        if tid in self.prev_edges:
            prev = self.prev_edges[tid]
            if prev.shape == edges.shape:
                # Compute difference
                diff = cv2.absdiff(edges, prev)
                diff_pixels = cv2.countNonZero(diff)
                # If significant edge change (impact frame often changes edges drastically)
                if diff_pixels > 50: 
                    has_edge_change = True
        
        self.prev_edges[tid] = edges
        
        # 4. Suppress Player
        # If player box overlaps with this crop, mask it out
        if player_box:
            px, py, pw, ph = player_box
            # Convert player box to crop coordinates
            px1 = int(px - pw/2) - x1
            py1 = int(py - ph/2) - y1
            px2 = int(px + pw/2) - x1
            py2 = int(py + ph/2) - y1
            
            # Clip to crop bounds
            px1 = max(0, px1)
            py1 = max(0, py1)
            px2 = min(crop.shape[1], px2)
            py2 = min(crop.shape[0], py2)
            
            if px1 < px2 and py1 < py2:
                # Zero out player area in flash mask
                flash_mask[py1:py2, px1:px2] = 0
                # Re-check flash
                if cv2.countNonZero(flash_mask) <= 20:
                    has_flash = False

        # 5. Combine & Smooth
        # Hit = Flash AND (Edge Change OR Persist)
        # Actually user said: "Reward hit only if: hit mask persists 2 frames OR hit-stun..."
        
        is_hit = False
        if has_flash:
            is_hit = True
            
        self.hit_signals[tid].append(is_hit)
        
        # Logic: Hit if signal persists for 2 frames in the last 3
        if sum(self.hit_signals[tid]) >= 2:
            return 1.0
            
        # Fallback: Hit-Stun (Size Jitter or Freeze)
        # We can check size jitter from tracking history in detect()
        # For now, return 0.0
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
        try:
            model = YOLO(path)
            # model.to() is not always needed for YOLO object if arguments handle it, 
            # but explicit move is good.
            # However, YOLO object itself isn't a nn.Module, model.model is.
            # Ultralytics handles device automatically during inference usually, 
            # but we can force it.
            
            # Warmup
            print("Warming up model...")
            # We pass device to the call
            dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            _ = model(dummy, device=self.device, half=self.half, verbose=False)
            print("Model ready.")
            return model
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

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
        # Initialize points if needed (KLT style)
        if self.prev_pts is None or len(self.prev_pts) < 50:
            # Create mask to ignore known moving objects (from last frame)
            mask = np.ones_like(frame_gray, dtype=np.uint8) * 255
            
            if self.last_det:
                def mask_rect(rect):
                    if rect:
                        x, y, w, h = rect[:4]
                        x1 = int(max(0, x - w/2))
                        y1 = int(max(0, y - h/2))
                        x2 = int(min(frame_gray.shape[1], x + w/2))
                        y2 = int(min(frame_gray.shape[0], y + h/2))
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

                mask_rect(self.last_det.get("player"))
                for enemy in self.last_det.get("enemies", []):
                    mask_rect(enemy)

            # Use GFTT but only when points are low
            # Increase maxCorners and minDistance for better coverage
            self.prev_pts = cv2.goodFeaturesToTrack(frame_gray, mask=mask, maxCorners=500, qualityLevel=0.01, minDistance=10)
        
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return 0.0, 0.0
            
        dx, dy = 0.0, 0.0
        
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            # Track to current frame
            # Note: nextPts=None is valid in Python OpenCV bindings but type checkers might complain.
            # We pass None as positional argument for nextPts.
            # Explicitly passing None for nextPts causes issues in some versions/type checkers.
            # We can pass an empty array or let it be created.
            # The signature is calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, ...)
            # If we pass None, it should work. If not, we can try passing None as keyword arg?
            # Or just rely on positional.
            # The error "None is not assignable to UMat" suggests strict type checking.
            # Let's try passing None explicitly.
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_pts, None, winSize=(21, 21), maxLevel=2)
            
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
                    
                    # Update points for next frame
                    self.prev_pts = good_new.reshape(-1, 1, 2)
                else:
                    self.prev_pts = None
            else:
                self.prev_pts = None
        
        self.prev_gray = frame_gray
        return dx, dy

    def detect(self, frame):
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
        
        # Decide: Run YOLO or Track?
        run_yolo = (self.frame_count % self.detection_interval == 0)
        
        # Clear Debug
        self.debug_rois = {}
        self.debug_health = None
        
        current_objects = {} # ID -> Box
        
        if run_yolo and self.model is not None:
            # --- YOLO STEP ---
            results = self.model.track(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False, half=self.half, device=self.device, persist=True, tracker="bytetrack.yaml")
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    if box.id is None: continue
                    tid = int(box.id[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xywh = box.xywh[0].cpu().numpy()
                    
                    # Update/Create Tracker
                    x, y, w, h = xywh
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
                
                if success:
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
                    
                    if 0 <= cx <= INPUT_SIZE and 0 <= cy <= INPUT_SIZE:
                        xywh = np.array([cx, cy, w, h])
                        state['box'] = xywh
                        state['failures'] += 1
                        current_objects[tid] = xywh
                    else:
                        state['failures'] += 10
                    
                    if state['failures'] > 5:
                        to_remove.append(tid)
            
            for tid in to_remove:
                del self.object_state[tid]

        # --- CONSTRUCT OUTPUT ---
        det_dict = {
            "player": None,
            "enemies": [],
            "items": [],
            "obstacles": [],
            "goal": None,
            "health": 1.0
        }
        
        character_candidates = []
        
        for tid, xywh in current_objects.items():
            state = self.object_state[tid]
            cls_id = state['cls']
            conf = state['conf']
            
            # Update History
            self.track_histories[tid].append(xywh[:2])
            
            label = CLASS_MAP.get(cls_id, "unknown")
            
            if label == "character":
                character_candidates.append((tid, xywh, conf))
            elif label == "throwable":
                det_dict["items"].append(xywh)

        # Identify Player
        target_center = np.array([INPUT_SIZE * 0.45, INPUT_SIZE * 0.5])
        best_score = float('inf')
        player_candidate = None
        
        for tid, xywh, conf in character_candidates:
            hist = self.track_histories[tid]
            cx, cy = xywh[:2]
            dist_to_center = np.linalg.norm(np.array([cx, cy]) - target_center)
            
            movement_score = 0.0
            if len(hist) >= 5:
                movement_score = np.linalg.norm(np.array(hist[-1]) - np.array(hist[-5]))
            
            score = (movement_score * 2.0) + (dist_to_center * 1.0)
            if score < best_score:
                best_score = score
                player_candidate = (tid, xywh, conf)
        
        # Health Logic
        current_health = self.get_health(frame)
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
            for tid, xywh, conf in character_candidates:
                if tid != pid:
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
        lower_red2 = np.array([170, 50, 50])
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
            
            # If less than 50% of the anchor columns are filled, assume bar is hidden
            if filled_anchor_cols < (anchor_width * 0.5):
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
                # print(f"Found {name} at {gx},{gy}")

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

    def get_obs(self, frame, last_action=None):
        """
        Main function to get observation vector from frame.
        Returns np.array of shape (20,)
        """
        # Performance check
        t0 = time.time()
        
        # Periodic UI Scan (every 60 frames)
        if self.frame_count % 60 == 0:
            self.scan_ui(frame)
        
        det_dict = self.detect(frame)
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
        
        if det_dict["enemies"]:
            # Find nearest to player (cx, cy)
            best_dist = float('inf')
            best_enemy = None
            
            for item in det_dict["enemies"]:
                # Handle variable tuple lengths
                if len(item) >= 4:
                    ecx, ecy, ew, eh = item[:4]
                else:
                    continue

                dx = (ecx - cx) / INPUT_SIZE
                dy = (ecy - cy) / INPUT_SIZE
                dist = dx*dx + dy*dy
                if dist < best_dist:
                    best_dist = dist
                    enemy_dx = dx
                    enemy_dy = dy
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
        
        # Build Vector
        # [px, py, vx, vy, edx, edy, odx, ody, dist_goal, time_since_seen, health, 
        #  enemy_health, mode, evasive, special, is_ragdolled, cd1, cd2, cd3, cd4]
        
        # MAPPING UPDATE:
        # health -> health (Back to healthbar)
        # enemy_health -> enemy_flash
        # is_ragdolled -> p_ragdoll
        
        health = det_dict.get("health", 1.0)
        
        obs = np.array([
            px_norm, py_norm, 
            vx, vy, 
            enemy_dx, enemy_dy, 
            obs_dx, obs_dy, 
            dist_goal,
            time_since_seen_norm,
            health,        # Was p_flash. Now health.
            enemy_flash,   # Was enemy_health
            mode_lvl,
            evasive_lvl,
            special_lvl,
            p_ragdoll,     # Was is_ragdolled (logic updated)
            cooldowns[0], cooldowns[1], cooldowns[2], cooldowns[3]
        ], dtype=np.float32)
        
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
                'full': np.array [12, 160, 160],
                'crop': np.array [12, 96, 96],
                'flow': np.array [2, 160, 160]
            }
        """
        if frame is None: return None
        
        # 1. Full Frame (160x160)
        full_img = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_LINEAR)
        # Normalize to 0-1
        full_norm = full_img.astype(np.float32) / 255.0
        # Transpose to [C, H, W]
        full_tensor = np.transpose(full_norm, (2, 0, 1))
        
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
        
        # Pad if out of bounds
        if crop_img.shape[0] != crop_size or crop_img.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            ph, pw = crop_img.shape[:2]
            
            # Safety check for broadcasting error
            if ph > crop_size or pw > crop_size:
                # This should not happen given x1, x2 logic, but if it does, resize
                crop_img = cv2.resize(crop_img, (crop_size, crop_size))
                padded = crop_img
            else:
                padded[:ph, :pw] = crop_img
            
            crop_img = padded
            
        crop_norm = crop_img.astype(np.float32) / 255.0
        crop_tensor = np.transpose(crop_norm, (2, 0, 1))
        
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
        # [4, 3, 160, 160] -> [12, 160, 160]
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
        # frame_stack: [C, H, W] where C=12 (3 channels * 4 frames)
        # We want the last 3 channels (latest frame)
        full_tensor = self.frame_stack[-1] # [3, 160, 160]
        crop_tensor = self.crop_stack[-1] # [3, 96, 96]
        
        # Convert back to HWC and uint8
        full_img = (np.transpose(full_tensor, (1, 2, 0)) * 255).astype(np.uint8)
        crop_img = (np.transpose(crop_tensor, (1, 2, 0)) * 255).astype(np.uint8)
        
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
             
        return canvas
        

# Global instance
perception_pipeline = None

def init_perception(model_path=None):
    global perception_pipeline
    if perception_pipeline is None:
        if model_path:
            perception_pipeline = Perception(model_path=model_path)
        else:
            perception_pipeline = Perception()
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
