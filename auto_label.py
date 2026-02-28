import cv2
import numpy as np
import os
import glob
import json
import threading
import queue
from ultralytics import YOLO
from tqdm import tqdm

# Configuration (cross-platform paths)
_data_root = os.environ.get("AUTO_FARMER_DATA") or os.path.expanduser("~/Auto-Farmer-Data")
VIDEO_DIR = os.path.join(_data_root, "frames")
OUTPUT_DIR = os.path.join(_data_root, "labels")
CONF_THRESHOLD = 0.4  # High confidence for players to avoid false positives
PROJECTILE_MIN_AREA = 10
PROJECTILE_MAX_AREA = 500
PROJECTILE_MIN_SPEED = 5.0 # Pixels per frame

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ThreadedFrameLoader:
    """
    Reads frames from disk in a separate thread to prevent I/O blocking.
    """
    def __init__(self, file_list, queue_size=256):
        self.file_list = file_list
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        for fpath in self.file_list:
            if self.stop_event.is_set():
                break
            frame = cv2.imread(fpath)
            if frame is not None:
                self.queue.put((fpath, frame))
        self.queue.put(None) # Sentinel

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def stop(self):
        self.stop_event.set()
        # Drain queue to allow worker to exit if blocked on put
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass

class ProjectileDetector:
    def __init__(self):
        self.prev_gray_small = None
        self.scale = 0.5 # Downscale factor for optical flow speedup
        
    def detect(self, frame, player_boxes):
        """
        Detects potential projectiles based on motion and brightness.
        Uses downscaled optical flow for performance.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale for faster optical flow
        small_gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale)
        
        if self.prev_gray_small is None:
            self.prev_gray_small = small_gray
            return []
            
        # 1. Optical Flow (Farneback) on downscaled image
        # Parameters tuned for speed vs accuracy
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray_small, small_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 2. Threshold Motion
        # Adjust speed threshold for scale
        scaled_speed_thresh = PROJECTILE_MIN_SPEED * self.scale
        _, motion_mask_small = cv2.threshold(mag, scaled_speed_thresh, 255, cv2.THRESH_BINARY)
        motion_mask_small = motion_mask_small.astype(np.uint8)
        
        # Upscale mask back to original size
        motion_mask = cv2.resize(motion_mask_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 3. Mask out Players (to avoid detecting moving limbs as projectiles)
        for box in player_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # Dilate player box slightly to be safe
            cv2.rectangle(motion_mask, (max(0, x1-10), max(0, y1-10)), (min(frame.shape[1], x2+10), min(frame.shape[0], y2+10)), 0, -1)
            
        # 4. Brightness Filter (Projectiles are often glowing)
        # This is heuristic-based. Can be tuned.
        # _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Combine: Moving AND Bright (Optional: OR just Moving if projectiles aren't bright)
        # For robustness, let's use Motion Mask primarily, refined by contours
        final_mask = motion_mask
        
        # 5. Find Contours
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if PROJECTILE_MIN_AREA < area < PROJECTILE_MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Aspect ratio check? Projectiles might be long
                aspect_ratio = float(w)/h
                if 0.2 < aspect_ratio < 5.0:
                    detections.append([x, y, x+w, y+h])
                    
        self.prev_gray_small = small_gray
        return detections

class CombatStateDetector:
    def __init__(self):
        self.entity_states = {} # { track_id: { 'hp': 1.0, 'pos': (x,y), 'velocity': 0 } }
        self.ui_state = { 'mode_bar_hist': None, 'main_hp': None }
        
    def get_main_player_health(self, frame):
        """
        Scans the top right corner for the main player's health bar.
        Returns a value representing health (ratio of green pixels).
        """
        h, w = frame.shape[:2]
        # ROI: Top right corner. 
        # Adjust these based on "very top right". 
        # Assuming 1920x1080, top right 300x50 area.
        roi_x1 = int(w * 0.80) 
        roi_y1 = int(h * 0.0)
        roi_x2 = int(w * 1.0)
        roi_y2 = int(h * 0.15)
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0: return 1.0
        
        # Convert to HSV for Green detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Green mask (Health)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        green_pixels = cv2.countNonZero(mask_green)
        
        if total_pixels == 0: return 0.0
        
        # Return ratio of green pixels in the ROI
        return green_pixels / total_pixels

    def get_entity_health_percentage(self, frame, bbox):
        """
        Scans the area immediately above the bounding box for a health bar.
        Returns float 0.0-1.0
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Look slightly above the head
        roi_h = 15
        roi_y = max(0, y1 - roi_h - 5)
        roi_b = max(0, y1 - 5)
        
        if roi_b <= roi_y: return 1.0 # Out of bounds
        
        roi = frame[roi_y:roi_b, x1:x2]
        if roi.size == 0: return 1.0
        
        # Convert to HSV for Green/Red detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Green mask (Health)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        green_pixels = cv2.countNonZero(mask_green)
        
        if total_pixels == 0: return 1.0
        
        ratio = green_pixels / total_pixels
        estimated_hp = min(1.0, ratio * 1.5) 
        return estimated_hp

    def detect_events(self, frame, detections):
        events = []
        
        # --- 1. Main Player Health (UI Top Right) ---
        current_main_hp = self.get_main_player_health(frame)
        
        if self.ui_state['main_hp'] is None:
            self.ui_state['main_hp'] = current_main_hp
        else:
            prev_main_hp = self.ui_state['main_hp']
            hp_diff = prev_main_hp - current_main_hp
            
            # Detect Hit (Health Drop)
            if hp_diff > 0.01: # Small threshold for pixel noise
                events.append({
                    "frame_event": "player_got_hit",
                    "damage": hp_diff
                })
            
            # Detect Death (Low Health)
            # Assuming max health ratio is around 0.1-0.3 depending on ROI size
            # If it drops near zero
            if current_main_hp < 0.005 and prev_main_hp > 0.005:
                 events.append({
                    "frame_event": "player_died"
                })
                
            self.ui_state['main_hp'] = current_main_hp

        # --- 2. Entity Analysis (Enemies/Other Players) ---
        current_ids = set()
        
        for obj in detections:
            tid = obj['track_id']
            if tid == -1: continue
            
            current_ids.add(tid)
            cls_name = obj['class_name']
            bbox = obj['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Only check entity health for enemies or other players (not main player if we could distinguish)
            # For now, check all, but label event as "entity_..."
            current_hp = self.get_entity_health_percentage(frame, bbox)
            
            # Initialize state if new
            if tid not in self.entity_states:
                self.entity_states[tid] = {
                    'hp': current_hp,
                    'pos': (center_x, center_y),
                    'velocity': 0,
                    'class': cls_name
                }
                continue
                
            prev_state = self.entity_states[tid]
            
            # Detect Entity Health Changes (Player landed a hit on enemy)
            hp_diff = prev_state['hp'] - current_hp
            if hp_diff > 0.05: 
                events.append({
                    "frame_event": "entity_damaged", # Likely player landed hit
                    "target_id": tid,
                    "target_class": cls_name,
                    "damage_est": hp_diff
                })
            
            # Detect Entity Death
            if current_hp < 0.05 and prev_state['hp'] > 0.05:
                 events.append({
                    "frame_event": "entity_died",
                    "target_id": tid,
                    "target_class": cls_name
                })

            # Detect Movement Actions (Dodge/Dash)
            dx = center_x - prev_state['pos'][0]
            dy = center_y - prev_state['pos'][1]
            velocity = np.sqrt(dx*dx + dy*dy)
            
            if velocity > 30:
                events.append({
                    "frame_event": "entity_dodge",
                    "actor_id": tid,
                    "actor_class": cls_name,
                    "velocity": velocity
                })

            # Update State
            self.entity_states[tid]['hp'] = current_hp
            self.entity_states[tid]['pos'] = (center_x, center_y)
            self.entity_states[tid]['velocity'] = velocity

        # --- 3. UI Mode Changes (Bottom Center) ---
        h, w = frame.shape[:2]
        # ROI: Bottom center 20% width, bottom 10% height
        ui_roi = frame[int(h*0.9):h, int(w*0.4):int(w*0.6)]
        
        # Calculate color histogram
        hist = cv2.calcHist([ui_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        
        if self.ui_state['mode_bar_hist'] is not None:
            # Compare with previous
            score = cv2.compareHist(self.ui_state['mode_bar_hist'], hist, cv2.HISTCMP_CORREL)
            if score < 0.95: # Significant change
                 events.append({
                    "frame_event": "mode_change",
                    "score": score
                })
        
        self.ui_state['mode_bar_hist'] = hist
        
        return events

def process_session(session_path, output_path, player_model):
    session_name = os.path.basename(session_path)
    print(f"Processing session: {session_name}")
    
    frame_files = sorted(glob.glob(os.path.join(session_path, "*.png")))
    if not frame_files:
        print("No frames found.")
        return

    projectile_detector = ProjectileDetector()
    combat_detector = CombatStateDetector()
    
    # Data structure for labels
    session_labels = {}
    
    # Use Threaded Loader
    loader = ThreadedFrameLoader(frame_files)
    
    try:
        for i, (frame_file, frame) in enumerate(tqdm(loader, total=len(frame_files))):
            frame_id = os.path.basename(frame_file)
            
            # --- 1. Object Detection (Custom YOLO) ---
            # Use 'track' mode for ID persistence. Detect all classes.
            # tracker="bytetrack.yaml" is faster than botsort
            results = player_model.track(frame, persist=True, verbose=False, conf=CONF_THRESHOLD, tracker="bytetrack.yaml")
            
            detected_objects = []
            # Boxes to mask out from projectile detector (players, enemies, etc.)
            mask_boxes = []
            
            if results and results[0].boxes:
                boxes = results[0].boxes
                for box in boxes:
                    # xyxy, conf, id
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = player_model.names[cls_id]
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    detected_objects.append({
                        "bbox": coords,
                        "conf": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "track_id": track_id
                    })
                    
                    # Add to mask for optical flow to ignore known objects
                    mask_boxes.append(coords)
            
            # --- 2. Projectile Detection (Optical Flow for missed fast objects) ---
            # Pass all detected object boxes to mask them out
            motion_projectiles = projectile_detector.detect(frame, mask_boxes)
            
            # --- 3. Combat State & Event Detection ---
            combat_events = combat_detector.detect_events(frame, detected_objects)

            # Store
            session_labels[frame_id] = {
                "detections": detected_objects,
                "motion_projectiles": motion_projectiles,
                "events": combat_events
            }
            
            # Visualization (Optional - save every 200th frame to check)
            if i % 200 == 0:
                vis_frame = frame.copy()
                for obj in detected_objects:
                    p = obj["bbox"]
                    label = f"{obj['class_name']} {obj['track_id']}"
                    cv2.rectangle(vis_frame, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 255, 0), 2)
                    cv2.putText(vis_frame, label, (int(p[0]), int(p[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                for event in combat_events:
                    cv2.putText(vis_frame, f"EVENT: {event['frame_event']}", (50, 50 + (combat_events.index(event)*30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                debug_dir = os.path.join(OUTPUT_DIR, "debug", session_name)
                ensure_dir(debug_dir)
                cv2.imwrite(os.path.join(debug_dir, frame_id), vis_frame)
                
    except KeyboardInterrupt:
        print("Interrupted! Stopping loader...")
        loader.stop()
        raise
    except Exception as e:
        print(f"Error processing session {session_name}: {e}")
        loader.stop()

    # Save JSON
    output_file = os.path.join(output_path, f"{session_name}_labels.json")
    with open(output_file, 'w') as f:
        json.dump(session_labels, f)
    print(f"Saved labels to {output_file}")

def main():
    ensure_dir(OUTPUT_DIR)
    
    # Load Custom YOLO Model
    model_path = r"T:\Auto-Farmer-Data\runs\combined_training_p3\weights\best.pt"
    print(f"Loading custom model from {model_path}...")
    try:
        model = YOLO(model_path)
        print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Error loading custom model: {e}")
        print("Falling back to yolov8n.pt")
        model = YOLO("yolov8n.pt")

    # Find all sessions
    sessions = glob.glob(os.path.join(VIDEO_DIR, "*"))
    sessions = [s for s in sessions if os.path.isdir(s)]
    
    print(f"Found {len(sessions)} sessions.")
    
    for session in sessions:
        process_session(session, OUTPUT_DIR, model)

if __name__ == "__main__":
    main()
