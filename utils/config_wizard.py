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

    def _draw_magnifier_with_offset(self, display, original, y_offset):
        """
        Draw magnifier for bottom-half display.
        Samples from full frame using mouse position + y_offset.
        Pads with black when near edges to allow precise edge selection.
        """
        x, y = self.mouse_pos
        # Adjust Y to sample from correct location in full frame
        sample_y = y + y_offset
        
        h, w = original.shape[:2]
        display_h, display_w = display.shape[:2]
        
        # Magnifier size
        mag_w, mag_h = 200, 200
        zoom = self.zoom_level
        
        # Source region size (in pixels from original)
        src_w = int(mag_w / zoom)
        src_h = int(mag_h / zoom)
        
        # Calculate source region CENTERED on mouse position (no clamping)
        x1 = x - src_w // 2
        y1 = sample_y - src_h // 2
        x2 = x1 + src_w
        y2 = y1 + src_h
        
        # Create padded region with black background
        padded = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        
        # Calculate valid source region (what we can actually sample from original)
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)
        
        # Calculate where to place in padded region
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # Copy valid region
        if src_x2 > src_x1 and src_y2 > src_y1:
            padded[dst_y1:dst_y2, dst_x1:dst_x2] = original[src_y1:src_y2, src_x1:src_x2]
        
        try:
            mag_roi = cv2.resize(padded, (mag_w, mag_h), interpolation=cv2.INTER_NEAREST)
            
            # Draw crosshair on magnifier (always centered)
            cv2.line(mag_roi, (mag_w//2, 0), (mag_w//2, mag_h), (0, 0, 255), 1)
            cv2.line(mag_roi, (0, mag_h//2), (mag_w, mag_h//2), (0, 0, 255), 1)
            
            # Overlay magnifier - position in top left or top right
            disp_x = 10
            disp_y = 10
            
            # If mouse is in Top Left, move to Top Right
            if x < 250 and y < 250:
                disp_x = display_w - mag_w - 10
            
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

    def select_cooldown_bars(self):
        """
        Wizard to select 4 cooldown bar ROIs.
        Each ROI is a rectangle around one move's cooldown indicator.
        """
        print("Taking screenshot for cooldown setup...")
        
        if not self.capture.running:
            self.capture.start()
            time.sleep(0.3)
        
        frame = self.capture.get_latest_frame()
        if frame is None:
            print("Failed to capture frame.")
            return None

        print("\n=== COOLDOWN BAR SELECTION ===")
        print("Select 4 cooldown bars by clicking TOP-LEFT then BOTTOM-RIGHT of each.")
        print("Controls:")
        print("  - Click to add point (2 clicks per bar)")
        print("  - 'r' to reset current bar")
        print("  - 'n' to skip to next bar")  
        print("  - 'Enter' to confirm all 4 bars")
        print("  - 'q' to quit\n")
        
        # Show bottom half of screen at full resolution (moves are at bottom)
        h, w = frame.shape[:2]
        self.y_offset = h // 2  # Store offset for coordinate mapping
        bottom_half = frame[self.y_offset:, :]  # Bottom half
        
        window_name = "Cooldown Bar Selection (Bottom Half)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, w, h // 2)  # Full width, half height
        cv2.setMouseCallback(window_name, self._cooldown_mouse_callback)
        
        self.cooldown_rois = []  # List of 4 ROIs: [(x, y, w, h), ...]
        self.current_bar_points = []
        self.current_bar_index = 0
        
        while True:
            display = bottom_half.copy()
            
            # Draw completed ROIs (adjust Y for display since we're showing bottom half)
            colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            labels = ["Move 1", "Move 2", "Move 3", "Move 4"]
            
            for i, roi in enumerate(self.cooldown_rois):
                x, y, w, h = roi
                # Adjust Y coordinate for bottom-half display
                display_y = y - self.y_offset
                if display_y >= 0:  # Only draw if visible in bottom half
                    cv2.rectangle(display, (x, display_y), (x + w, display_y + h), colors[i], 2)
                    cv2.putText(display, labels[i], (x, display_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
            
            # Draw current selection points
            if len(self.current_bar_points) == 1:
                pt = self.current_bar_points[0]
                cv2.circle(display, pt, 5, (0, 0, 255), -1)
                # Draw line to mouse
                cv2.line(display, pt, self.mouse_pos, (0, 0, 255), 1)
                cv2.rectangle(display, pt, self.mouse_pos, (0, 0, 255), 1)
            
            # Status text
            status = f"Selecting Move {self.current_bar_index + 1}/4"
            if len(self.current_bar_points) == 0:
                status += " - Click TOP-LEFT corner"
            else:
                status += " - Click BOTTOM-RIGHT corner"
            
            cv2.putText(display, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw magnifier (use full frame with offset for proper boundary handling)
            self._draw_magnifier_with_offset(display, frame, self.y_offset)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_bar_points = []
            elif key == ord('n'):
                # Skip to next bar
                if self.current_bar_index < 3:
                    self.current_bar_index += 1
                    self.current_bar_points = []
            elif key == 13:  # Enter
                if len(self.cooldown_rois) == 4:
                    self._save_cooldown_config()
                    break
                else:
                    print(f"Need 4 cooldown bars. You have {len(self.cooldown_rois)}.")
        
        cv2.destroyWindow(window_name)
        return self.cooldown_rois
    
    def _cooldown_mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_bar_points) < 2:
                self.current_bar_points.append((x, y))
                
                if len(self.current_bar_points) == 2:
                    # Convert to ROI (x, y, w, h)
                    # Add y_offset to get actual frame coordinates
                    x1, y1 = self.current_bar_points[0]
                    x2, y2 = self.current_bar_points[1]
                    
                    rx = min(x1, x2)
                    ry = min(y1, y2) + self.y_offset  # Add offset to get real Y
                    rw = abs(x2 - x1)
                    rh = abs(y2 - y1)
                    
                    if len(self.cooldown_rois) < 4:
                        self.cooldown_rois.append((rx, ry, rw, rh))
                        print(f"Move {len(self.cooldown_rois)} ROI: x={rx}, y={ry}, w={rw}, h={rh}")
                        self.current_bar_index = len(self.cooldown_rois)
                    
                    self.current_bar_points = []
    
    def _save_cooldown_config(self):
        """Save cooldown ROIs to config file."""
        # Load existing config
        data = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
        
        # Add cooldown ROIs
        data["cooldown_rois"] = self.cooldown_rois
        
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Cooldown config saved to {CONFIG_FILE}")
    
    @staticmethod
    def load_cooldown_rois():
        """Load cooldown ROIs from config file."""
        if not os.path.exists(CONFIG_FILE):
            return None
        
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            return data.get("cooldown_rois", None)
        except:
            return None


class CooldownDetector:
    """
    Detects if moves are on cooldown by checking for blue color in ROIs.
    Blue indicates cooldown in progress.
    """
    
    # HSV range for blue (cooldown indicator)
    BLUE_LOW = np.array([100, 100, 100])   # H: 100-130 for blue
    BLUE_HIGH = np.array([130, 255, 255])
    
    # Minimum percentage of blue pixels to consider "on cooldown"
    COOLDOWN_THRESHOLD = 0.15  # 15% of ROI is blue
    
    def __init__(self, cooldown_rois=None):
        """
        Args:
            cooldown_rois: List of 4 tuples [(x, y, w, h), ...]
        """
        self.rois = cooldown_rois or []
        self.move_on_cooldown = [False, False, False, False]
    
    def update(self, frame):
        """
        Update cooldown status for all 4 moves.
        
        Args:
            frame: BGR frame from capture
            
        Returns:
            List of 4 booleans: [move1_on_cd, move2_on_cd, move3_on_cd, move4_on_cd]
        """
        if frame is None or len(self.rois) != 4:
            return self.move_on_cooldown
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for i, roi in enumerate(self.rois):
            x, y, w, h = roi
            
            # Ensure valid bounds
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            
            if x2 <= x or y2 <= y:
                self.move_on_cooldown[i] = False
                continue
            
            # Extract ROI
            roi_hsv = hsv[y:y2, x:x2]
            
            # Detect blue pixels (horizontal strip detection)
            # We check if there's blue across the horizontal extent
            blue_mask = cv2.inRange(roi_hsv, self.BLUE_LOW, self.BLUE_HIGH)
            
            # For horizontal detection: check each row
            # If any row has significant blue percentage, it's on cooldown
            h_roi, w_roi = blue_mask.shape[:2]
            
            on_cooldown = False
            for row in range(h_roi):
                row_pixels = blue_mask[row, :]
                blue_ratio = np.sum(row_pixels > 0) / w_roi
                
                if blue_ratio > self.COOLDOWN_THRESHOLD:
                    on_cooldown = True
                    break
            
            self.move_on_cooldown[i] = on_cooldown
        
        return self.move_on_cooldown
    
    def get_available_moves(self):
        """
        Get indices of moves that are NOT on cooldown.
        
        Returns:
            List of move indices (0-3) that are available
        """
        return [i for i, on_cd in enumerate(self.move_on_cooldown) if not on_cd]
    
    def is_move_available(self, move_index):
        """Check if a specific move (0-3) is available."""
        if 0 <= move_index < 4:
            return not self.move_on_cooldown[move_index]
        return True  # Default to available for invalid indices

