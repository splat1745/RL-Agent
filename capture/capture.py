# Requirements: pip install dxcam numpy pywin32
import time
import threading
import numpy as np
import win32gui
import win32api
import dxcam

import win32con

def list_window_titles():
    """Lists all visible window titles."""
    titles = []
    def enum_windows_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)
    win32gui.EnumWindows(enum_windows_callback, None)
    return titles

class ScreenCapture:
    def __init__(self, window_title=None, target_fps=60):
        self.window_title = window_title
        self.target_fps = target_fps
        self.running = False
        self.lock = threading.Lock()
        self.frame = None
        self.camera = None # Initialized in start()
        self.region = None
        self.thread = None

    def _get_monitor_for_window(self, hwnd):
        """Finds the monitor index that contains the center of the window."""
        try:
            monitor_handle = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
            # Get all monitors
            monitors = win32api.EnumDisplayMonitors()
            for i, (h_monitor, _, _) in enumerate(monitors):
                if h_monitor == monitor_handle:
                    return i
        except Exception as e:
            print(f"Error finding monitor for window: {e}")
        return 0 # Default to primary

    def find_window_info(self):
        """Returns (region, monitor_index) for the target window."""
        if not self.window_title:
            return None, 0
        
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd:
            self.hwnd = hwnd # Store hwnd
            # GetWindowRect returns (left, top, right, bottom)
            rect = win32gui.GetWindowRect(hwnd)
            monitor_idx = self._get_monitor_for_window(hwnd)
            
            # We need to adjust coordinates relative to the monitor if dxcam requires it.
            # dxcam.create(output_idx=...) creates a camera for that monitor.
            # The region passed to camera.start() should be relative to that monitor's top-left?
            # Actually, dxcam usually expects screen coordinates, but let's verify.
            # If using multiple monitors, dxcam might expect coordinates relative to the specific output.
            # Let's assume for now we pass the rect as is, but we might need to subtract monitor offset.
            
            # Get monitor info to adjust coordinates
            monitors = win32api.EnumDisplayMonitors()
            if monitor_idx < len(monitors):
                monitor_handle = monitors[monitor_idx][0]
                # PyHANDLE usually works with GetMonitorInfo at runtime despite type hints
                monitor_info = win32api.GetMonitorInfo(monitor_handle) # type: ignore
                monitor_rect = monitor_info['Monitor'] # (left, top, right, bottom)
                
                # Adjust rect to be relative to monitor
                # region = (left, top, right, bottom)
                # dxcam expects (left, top, right, bottom) relative to the monitor
                
                rel_left = max(0, rect[0] - monitor_rect[0])
                rel_top = max(0, rect[1] - monitor_rect[1])
                rel_right = min(monitor_rect[2] - monitor_rect[0], rect[2] - monitor_rect[0])
                rel_bottom = min(monitor_rect[3] - monitor_rect[1], rect[3] - monitor_rect[1])
                
                if rel_right - rel_left > 0 and rel_bottom - rel_top > 0:
                    return (rel_left, rel_top, rel_right, rel_bottom), monitor_idx

        return None, 0

    def _init_camera(self, monitor_idx):
        """
        Initializes the dxcam camera.
        Attempts to find the correct device and output index corresponding to the monitor_idx.
        """
        # Try direct mapping first (monitor_idx -> output_idx on default device)
        try:
            return dxcam.create(output_idx=monitor_idx, output_color="BGR")
        except IndexError:
            pass # Failed, try searching
        except Exception as e:
            print(f"dxcam create failed: {e}")

        print(f"Direct mapping for monitor {monitor_idx} failed. Searching for correct dxcam output...")
        
        # We don't have the monitor rect here easily unless we pass it or recalculate.
        # Let's just try to find *any* working camera on other devices if possible,
        # or just iterate and print what we find.
        # A better approach: Iterate all devices and outputs, and if we find one that works, use it?
        # But which one is the *right* one?
        # Without resolution matching, it's a guess.
        # Let's try to match resolution if we can get it.
        
        target_w, target_h = 0, 0
        try:
            monitors = win32api.EnumDisplayMonitors()
            if monitor_idx < len(monitors):
                monitor_handle = monitors[monitor_idx][0]
                monitor_info = win32api.GetMonitorInfo(monitor_handle) # type: ignore
                rect = monitor_info['Monitor']
                target_w = rect[2] - rect[0]
                target_h = rect[3] - rect[1]
        except Exception:
            pass

        for device_idx in range(4): # Check first 4 GPUs
            for output_idx in range(4): # Check first 4 outputs
                try:
                    camera = dxcam.create(device_idx=device_idx, output_idx=output_idx, output_color="BGR")
                    # If we have a target resolution, check it
                    if target_w > 0 and target_h > 0:
                        # dxcam doesn't expose width/height directly on the instance easily?
                        # Actually it does: camera.width, camera.height (after start? or immediately?)
                        # It seems to be available immediately.
                        if camera.width == target_w and camera.height == target_h:
                            print(f"Resolved to Device {device_idx}, Output {output_idx}")
                            return camera
                    else:
                        # If we can't check resolution, just return the first one we find?
                        # That's risky.
                        pass
                    
                    # If not returned, maybe we should keep searching?
                    # But we can't have multiple instances easily without cleaning up?
                    # dxcam is singleton-ish.
                except IndexError:
                    break # No more outputs on this device
                except Exception:
                    continue
        
        print("Warning: Could not find exact match for monitor. Defaulting to primary (Device 0, Output 0).")
        return dxcam.create(output_idx=0, output_color="BGR")

    def start(self):
        if self.running:
            return

        region, monitor_idx = self.find_window_info()
        self.region = region
        
        if self.region:
            print(f"Capturing window '{self.window_title}' on monitor {monitor_idx} at region: {self.region}")
        else:
            print(f"Window '{self.window_title}' not found or invalid. Capturing full screen on monitor {monitor_idx}.")
            self.region = None # Full screen

        # Initialize camera for the specific monitor
        if self.camera is None:
             self.camera = self._init_camera(monitor_idx)
        
        self.running = True
        
        # We use dxcam's built-in threading for maximum performance (producer-consumer)
        # video_mode=True allows using get_latest_frame()
        if self.region:
            self.camera.start(target_fps=self.target_fps, region=self.region, video_mode=True)
        else:
            self.camera.start(target_fps=self.target_fps, video_mode=True)
        
        # We also start a small management thread to update our global buffer variable
        # explicitly if needed, or we can just rely on camera.get_latest_frame().
        # The user asked for a "Global frame buffer". 
        # Let's maintain a copy in self.frame for easy access.
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.camera and self.camera.is_capturing:
            self.camera.stop()
        if self.thread:
            self.thread.join()
        # Release camera resource if needed, though dxcam doesn't have a close/release method explicitly shown in common usage
        # but we can set it to None to force recreation on next start if monitor changes
        self.camera = None

    def _update_loop(self):
        while self.running:
            if self.camera:
                img = self.camera.get_latest_frame()
                if img is not None:
                    with self.lock:
                        self.frame = img
            time.sleep(0.001) # Slight sleep to prevent busy waiting in this management loop

    def get_latest_frame(self):
        """Returns the latest captured frame (thread-safe)."""
        with self.lock:
            return self.frame

    def select_window(self):
        """Interactive window selection."""
        titles = list_window_titles()
        print("\nAvailable Windows:")
        for i, title in enumerate(titles):
            print(f"{i}: {title}")
        
        try:
            choice = int(input("\nSelect window ID (or -1 for full screen): "))
            if 0 <= choice < len(titles):
                self.window_title = titles[choice]
                print(f"Selected: {self.window_title}")
            else:
                self.window_title = None
                print("Selected: Full Screen")
        except ValueError:
            print("Invalid input. Defaulting to Full Screen.")
            self.window_title = None

    def focus(self):
        """Brings the target window to the foreground."""
        if hasattr(self, 'hwnd') and self.hwnd:
            try:
                # Windows sometimes blocks SetForegroundWindow. 
                # We can try to force it by sending an Alt key press or attaching thread input, 
                # but usually a simple call works if we are just starting.
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(self.hwnd)
                print(f"Focused window: {self.window_title}")
            except Exception as e:
                print(f"Failed to focus window: {e}")

# Global instance for easy import
# Usage: from capture import capture_service; capture_service.start(); frame = capture_service.get_latest_frame()
capture_service = ScreenCapture()

if __name__ == "__main__":
    # Example usage
    print("Screen Capture Service")
    capture_service.select_window()
    
    capture_service.start()
    
    try:
        last_time = time.time()
        frames = 0
        while True:
            frame = capture_service.get_latest_frame()
            if frame is not None:
                # Do something with frame
                pass
            
            frames += 1
            if time.time() - last_time >= 1.0:
                print(f"FPS: {frames}")
                frames = 0
                last_time = time.time()
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping...")
        capture_service.stop()
