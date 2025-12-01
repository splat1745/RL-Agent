import cv2
import json
import os
import time
import numpy as np

CONFIG_FILE = "config.json"

class ConfigWizard:
    def __init__(self, capture_service):
        self.capture = capture_service
        self.points = []
        self.zoom_level = 2
        self.mouse_pos = (0, 0)

    def select_health_bar(self):
        print("Taking screenshot for setup...")
        # Ensure capture is running
        if not self.capture.running:
            self.capture.start()
            time.sleep(1.0) # Warmup
        
        frame = self.capture.get_latest_frame()
        if frame is None:
            print("Failed to capture frame. Make sure the game is visible.")
            return None

        print("Please select 4 points for the health bar (Top-Left, Top-Right, Bottom-Right, Bottom-Left).")
        print("Controls: Click to add point, 'r' to reset, 'Enter' to confirm, 'q' to quit.")
        
        window_name = "Health Bar Selection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        self.points = []
        
        while True:
            display = frame.copy()
            
            # Draw points
            for i, pt in enumerate(self.points):
                cv2.circle(display, pt, 3, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display, self.points[i-1], pt, (0, 255, 0), 1)
            
            # Close loop if 4 points
            if len(self.points) == 4:
                cv2.line(display, self.points[3], self.points[0], (0, 255, 0), 1)
                
            # Draw magnifier
            self._draw_magnifier(display, frame)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.points = []
            elif key == 13: # Enter
                if len(self.points) == 4:
                    self._save_config()
                    break
                else:
                    print(f"Need 4 points. You have {len(self.points)}.")

        cv2.destroyWindow(window_name)
        return self.points

    def _mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))

    def _draw_magnifier(self, display, original):
        x, y = self.mouse_pos
        h, w = original.shape[:2]
        
        # Magnifier size
        mag_w, mag_h = 200, 200
        zoom = self.zoom_level
        
        # Source region
        src_w = int(mag_w / zoom)
        src_h = int(mag_h / zoom)
        
        x1 = max(0, x - src_w // 2)
        y1 = max(0, y - src_h // 2)
        x2 = min(w, x1 + src_w)
        y2 = min(h, y1 + src_h)
        
        # Adjust if out of bounds
        if x2 - x1 < src_w: x1 = x2 - src_w
        if y2 - y1 < src_h: y1 = y2 - src_h
        
        # Ensure valid slice
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        
        roi = original[y1:y2, x1:x2]
        if roi.size == 0: return
        
        try:
            mag_roi = cv2.resize(roi, (mag_w, mag_h), interpolation=cv2.INTER_NEAREST)
            
            # Draw crosshair on magnifier
            cv2.line(mag_roi, (mag_w//2, 0), (mag_w//2, mag_h), (0, 0, 255), 1)
            cv2.line(mag_roi, (0, mag_h//2), (mag_w, mag_h//2), (0, 0, 255), 1)
            
            # Overlay magnifier
            # Default: Top Left
            disp_x = 10
            disp_y = 10
            
            # If mouse is in Top Left, move to Top Right
            if x < 250 and y < 250:
                disp_x = w - mag_w - 10
            
            display[disp_y:disp_y+mag_h, disp_x:disp_x+mag_w] = mag_roi
            cv2.rectangle(display, (disp_x, disp_y), (disp_x+mag_w, disp_y+mag_h), (255, 255, 255), 2)
            cv2.putText(display, f"Zoom x{zoom}", (disp_x, disp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            pass

    def validate_config(self):
        """Checks if config exists and has a valid health ROI."""
        if not os.path.exists(CONFIG_FILE):
            return False
            
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            
            # Check if health_roi exists and has valid bbox
            if "health_roi" not in data:
                return False
            
            if "bbox" not in data["health_roi"]:
                return False
                
            bbox = data["health_roi"]["bbox"]
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                return False
            
            # Optional: Check resolution but only warn, don't invalidate
            if "resolution" in data:
                saved_w, saved_h = data["resolution"]
                
                # Ensure capture is running to check resolution
                if not self.capture.running:
                    self.capture.start()
                    for _ in range(10):
                        if self.capture.get_latest_frame() is not None:
                            break
                        time.sleep(0.1)
                    
                frame = self.capture.get_latest_frame()
                if frame is not None:
                    curr_h, curr_w = frame.shape[:2]
                    
                    if saved_w != curr_w or saved_h != curr_h:
                        print(f"Warning: Resolution changed (Saved: {saved_w}x{saved_h}, Current: {curr_w}x{curr_h}).")
                        print("Health bar position may need reconfiguration. Use --setup to reconfigure.")
                        # Still return True - let user decide if they need to reconfigure
                
            return True
        except Exception as e:
            print(f"Config validation error: {e}")
            return False

    def _save_config(self):
        # Convert points to list of lists for JSON
        points_list = [list(pt) for pt in self.points]
        
        # Calculate bounding box for convenience
        pts = np.array(self.points)
        bx, by, bw, bh = cv2.boundingRect(pts)
        
        # Get Resolution
        res_w, res_h = 0, 0
        if self.capture.frame is not None:
            res_h, res_w = self.capture.frame.shape[:2]
        
        data = {
            "resolution": [res_w, res_h],
            "health_roi": {
                "points": points_list,
                "bbox": [bx, by, bw, bh]
            }
        }
        
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Config saved to {CONFIG_FILE}")
