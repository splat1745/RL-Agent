# Requirements (Linux):  pip install mss numpy
# Requirements (Windows): pip install dxcam numpy pywin32
# Optional (Linux, for window-specific capture): apt install wmctrl  OR  apt install xdotool
import sys
import time
import threading
import subprocess
import numpy as np

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import win32gui
    import win32api
    import win32con
    import dxcam
else:
    import mss as _mss


# ---------------------------------------------------------------------------
# Window enumeration helpers
# ---------------------------------------------------------------------------

def list_window_titles():
    """Returns a list of visible window titles."""
    if IS_WINDOWS:
        titles = []
        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                t = win32gui.GetWindowText(hwnd)
                if t:
                    titles.append(t)
        win32gui.EnumWindows(_cb, None)
        return titles
    else:
        # Try wmctrl first, then xdotool
        for cmd in [["wmctrl", "-l"], ["xdotool", "search", "--onlyvisible", "--name", ""]]:
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=3).decode()
                titles = []
                if cmd[0] == "wmctrl":
                    for line in out.splitlines():
                        parts = line.split(None, 3)
                        if len(parts) >= 4:
                            titles.append(parts[3])
                else:
                    # xdotool returns window IDs; get names
                    for wid in out.splitlines():
                        try:
                            name = subprocess.check_output(
                                ["xdotool", "getwindowname", wid.strip()],
                                stderr=subprocess.DEVNULL, timeout=2
                            ).decode().strip()
                            if name:
                                titles.append(name)
                        except Exception:
                            pass
                return titles
            except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
                continue
        print("Warning: wmctrl/xdotool not found. Window listing unavailable.")
        print("  Install with:  sudo apt install wmctrl   OR   sudo apt install xdotool")
        return []


def _find_window_geometry_linux(title):
    """
    Returns (left, top, width, height) dict for the first visible window whose
    title contains `title` (case-insensitive), or None if not found.
    Tries xdotool then wmctrl.
    """
    # --- xdotool ---
    try:
        ids_raw = subprocess.check_output(
            ["xdotool", "search", "--onlyvisible", "--name", title],
            stderr=subprocess.DEVNULL, timeout=3
        ).decode().strip()
        if ids_raw:
            wid = ids_raw.splitlines()[0].strip()
            geom_raw = subprocess.check_output(
                ["xdotool", "getwindowgeometry", "--shell", wid],
                stderr=subprocess.DEVNULL, timeout=3
            ).decode()
            info = {}
            for line in geom_raw.strip().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
            return {
                "left":   int(info["X"]),
                "top":    int(info["Y"]),
                "width":  int(info["WIDTH"]),
                "height": int(info["HEIGHT"]),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.CalledProcessError, KeyError, ValueError):
        pass

    # --- wmctrl -l -G ---
    try:
        out = subprocess.check_output(
            ["wmctrl", "-l", "-G"], stderr=subprocess.DEVNULL, timeout=3
        ).decode()
        for line in out.splitlines():
            if title.lower() in line.lower():
                parts = line.split()
                if len(parts) >= 7:
                    return {
                        "left":   int(parts[2]),
                        "top":    int(parts[3]),
                        "width":  int(parts[4]),
                        "height": int(parts[5]),
                    }
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.CalledProcessError, ValueError):
        pass

    return None


def _focus_window_linux(title):
    """Raise/focus a window by title on Linux."""
    for cmd in [["wmctrl", "-a", title], ["xdotool", "search", "--name", title, "windowactivate"]]:
        try:
            subprocess.run(cmd, stderr=subprocess.DEVNULL, timeout=3)
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue


# ---------------------------------------------------------------------------
# ScreenCapture class
# ---------------------------------------------------------------------------

class ScreenCapture:
    def __init__(self, window_title=None, target_fps=60):
        self.window_title = window_title
        self.target_fps = target_fps
        self.running = False
        self.lock = threading.Lock()
        self.frame = None
        self.thread = None

        # Windows-only
        self.camera = None   # dxcam instance
        self.region = None   # (left, top, right, bottom) for dxcam

        # Linux-only
        self._mss_region = None  # {"left":…,"top":…,"width":…,"height":…} for mss

    # ------------------------------------------------------------------
    # Windows helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _win_get_monitor_for_window(self, hwnd):
        try:
            mh = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
            for i, (h, _, _) in enumerate(win32api.EnumDisplayMonitors()):
                if h == mh:
                    return i
        except Exception as e:
            print(f"Error finding monitor: {e}")
        return 0

    def _win_find_window_info(self):
        """Returns (dxcam_region, monitor_idx) on Windows."""
        if not self.window_title:
            return None, 0
        hwnd = win32gui.FindWindow(None, self.window_title)
        if not hwnd:
            return None, 0
        self.hwnd = hwnd
        rect = win32gui.GetWindowRect(hwnd)
        monitor_idx = self._win_get_monitor_for_window(hwnd)
        monitors = win32api.EnumDisplayMonitors()
        if monitor_idx < len(monitors):
            minfo = win32api.GetMonitorInfo(monitors[monitor_idx][0])  # type: ignore
            mr = minfo["Monitor"]
            rl = max(0, rect[0] - mr[0])
            rt = max(0, rect[1] - mr[1])
            rr = min(mr[2] - mr[0], rect[2] - mr[0])
            rb = min(mr[3] - mr[1], rect[3] - mr[1])
            if rr - rl > 0 and rb - rt > 0:
                return (rl, rt, rr, rb), monitor_idx
        return None, 0

    def _win_init_camera(self, monitor_idx):
        try:
            return dxcam.create(output_idx=monitor_idx, output_color="BGR")
        except Exception:
            pass
        print("Warning: falling back to primary monitor for DXcam.")
        return dxcam.create(output_idx=0, output_color="BGR")

    # ------------------------------------------------------------------
    # Linux helpers
    # ------------------------------------------------------------------

    def _linux_find_region(self):
        """Returns mss region dict or None (full screen)."""
        if not self.window_title:
            return None
        geo = _find_window_geometry_linux(self.window_title)
        if geo:
            print(f"Found window '{self.window_title}' at {geo}")
            return geo
        print(f"Window '{self.window_title}' not found via wmctrl/xdotool. Capturing full screen.")
        print("  Install wmctrl:  sudo apt install wmctrl")
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self.running:
            return

        if IS_WINDOWS:
            region, monitor_idx = self._win_find_window_info()
            self.region = region
            if self.region:
                print(f"DXcam: capturing region {self.region} on monitor {monitor_idx}")
            else:
                print(f"DXcam: full-screen capture on monitor {monitor_idx}")
            if self.camera is None:
                self.camera = self._win_init_camera(monitor_idx)
            self.running = True
            if self.region:
                self.camera.start(target_fps=self.target_fps, region=self.region, video_mode=True)
            else:
                self.camera.start(target_fps=self.target_fps, video_mode=True)
            self.thread = threading.Thread(target=self._win_update_loop, daemon=True)
        else:
            self._mss_region = self._linux_find_region()
            self.running = True
            self.thread = threading.Thread(target=self._linux_update_loop, daemon=True)

        self.thread.start()

    def stop(self):
        self.running = False
        if IS_WINDOWS and self.camera and self.camera.is_capturing:
            self.camera.stop()
            self.camera = None
        if self.thread:
            self.thread.join(timeout=2)

    def _win_update_loop(self):
        while self.running:
            img = self.camera.get_latest_frame()
            if img is not None:
                with self.lock:
                    self.frame = img
            time.sleep(0.001)

    def _linux_update_loop(self):
        interval = 1.0 / self.target_fps
        with _mss.mss() as sct:
            # Determine capture region
            if self._mss_region:
                region = self._mss_region
            else:
                # Full primary monitor (monitors[1] is the first physical screen)
                region = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

            while self.running:
                t0 = time.perf_counter()
                raw = sct.grab(region)
                # mss returns BGRA; drop alpha for BGR (OpenCV-compatible)
                img = np.array(raw)[:, :, :3]
                with self.lock:
                    self.frame = img
                elapsed = time.perf_counter() - t0
                remaining = interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

    def get_latest_frame(self):
        """Returns the latest captured frame as a BGR numpy array (thread-safe)."""
        with self.lock:
            return self.frame

    def select_window(self):
        """Interactive window selection (prints a numbered list)."""
        titles = list_window_titles()
        print("\nAvailable Windows:")
        for i, title in enumerate(titles):
            print(f"  {i}: {title}")
        try:
            choice = int(input("\nSelect window number (or -1 for full screen): "))
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
        """Bring the target window to the foreground."""
        if IS_WINDOWS:
            if hasattr(self, "hwnd") and self.hwnd:
                try:
                    win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(self.hwnd)
                    print(f"Focused: {self.window_title}")
                except Exception as e:
                    print(f"Failed to focus window: {e}")
        else:
            if self.window_title:
                _focus_window_linux(self.window_title)

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
