import sys
import time
import threading

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import win32api
    import win32con
    import ctypes

    # DirectInput scan codes (Windows only)
    DIK_1 = 0x02
    DIK_2 = 0x03
    DIK_3 = 0x04
    DIK_4 = 0x05
    DIK_Q = 0x10
    DIK_W = 0x11
    DIK_R = 0x13
    DIK_A = 0x1E
    DIK_S = 0x1F
    DIK_D = 0x20
    DIK_F = 0x21
    DIK_G = 0x22
    DIK_SPACE = 0x39

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort),
                    ("wParamH", ctypes.c_ushort)]

    class INPUT_I(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", INPUT_I)]

    def press_key(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = ctypes.pointer(extra)
        inp = INPUT()
        inp.type = 1
        inp.ii.ki = KEYBDINPUT(0, hexKeyCode, 0x0008, 0, ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

    def release_key(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = ctypes.pointer(extra)
        inp = INPUT()
        inp.type = 1
        inp.ii.ki = KEYBDINPUT(0, hexKeyCode, 0x0008 | 0x0002, 0, ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

    def click():
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

    def move_mouse(x, y):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)

    def _get_cursor_pos():
        return win32api.GetCursorPos()

else:
    # Linux: use pynput for keyboard + mouse injection
    from pynput.keyboard import Key as _Key, Controller as _KbCtrl
    from pynput.mouse import Controller as _MsCtrl, Button as _Button

    _kb = _KbCtrl()
    _mouse = _MsCtrl()

    # Map string key names to pynput key objects
    _KEY_MAP = {
        'w': 'w', 'a': 'a', 's': 's', 'd': 'd',
        'q': 'q', 'f': 'f', 'r': 'r', 'g': 'g',
        '1': '1', '2': '2', '3': '3', '4': '4',
        'space': _Key.space,
    }

    def press_key(key_name: str):
        """Press a key by name (e.g. 'w', '1', 'space')."""
        k = _KEY_MAP.get(key_name, key_name)
        _kb.press(k)

    def release_key(key_name: str):
        """Release a key by name."""
        k = _KEY_MAP.get(key_name, key_name)
        _kb.release(k)

    def click():
        _mouse.press(_Button.left)
        time.sleep(0.05)
        _mouse.release(_Button.left)

    def move_mouse(x, y):
        _mouse.move(int(x), int(y))

    def _get_cursor_pos():
        pos = _mouse.position
        return (int(pos[0]), int(pos[1]))


# ---------------------------------------------------------------------------
# InputController — same public API on both platforms
# ---------------------------------------------------------------------------

class InputController:
    def __init__(self):
        self.current_key = None

        # Mouse vertical constraint state
        try:
            self.start_mouse_pos = _get_cursor_pos()
        except Exception:
            self.start_mouse_pos = (0, 0)
        self.accumulated_dy = 0

        # Thread-safe recent mouse movement accumulator
        self.recent_mouse_dx = 0
        self.recent_mouse_dy = 0
        self._lock = threading.Lock()

        # Cooldown management (populated by caller if needed)
        self.last_used = {}
        self.cooldowns = {}

    def move_mouse_constrained(self, dx, dy):
        """Move mouse with vertical pitch clamp (no looking past horizon or floor)."""
        new_accum = self.accumulated_dy + dy
        max_down = self.start_mouse_pos[1]

        if new_accum < 0:
            dy = -self.accumulated_dy
            new_accum = 0
        elif new_accum > max_down:
            dy = max_down - self.accumulated_dy
            new_accum = max_down

        self.accumulated_dy = new_accum

        if dx != 0 or dy != 0:
            move_mouse(int(dx), int(dy))
            with self._lock:
                self.recent_mouse_dx += dx
                self.recent_mouse_dy += dy

    def get_recent_movement(self):
        """Returns and resets accumulated mouse Δx, Δy since last call."""
        with self._lock:
            dx, dy = self.recent_mouse_dx, self.recent_mouse_dy
            self.recent_mouse_dx = 0
            self.recent_mouse_dy = 0
        return dx, dy

    def execute(self, action_name, duration=0.1):
        # Cooldown check
        check_name = "2" if action_name in ("r_2", "2") else action_name
        if check_name in self.cooldowns:
            if time.time() - self.last_used.get(check_name, 0) < self.cooldowns[check_name]:
                return
            self.last_used[check_name] = time.time()

        # Release held key if switching
        if self.current_key and self.current_key != action_name:
            self._release(self.current_key)
            self.current_key = None

        if action_name == "idle":
            time.sleep(duration)
            return

        if action_name in ("click", "m1"):
            click()
            time.sleep(0.1)
            return

        # Evasive dashes: direction + Q simultaneously
        _DASH_MAP = {
            "dash_left":    ("a", "q"),
            "dash_right":   ("d", "q"),
            "dash_back":    ("s", "q"),
            "dash_forward": ("w", "q"),
        }
        if action_name in _DASH_MAP:
            dir_key, mod_key = _DASH_MAP[action_name]
            self._press(dir_key)
            self._press(mod_key)
            time.sleep(0.05)
            self._release(mod_key)
            self._release(dir_key)
            time.sleep(duration)
            return

        # Tap skills / special keys
        if action_name in ("1", "2", "3", "4", "f", "r", "g"):
            self._press(action_name)
            time.sleep(0.09)
            self._release(action_name)
            time.sleep(duration)
            return

        # Held movement keys (WASD, Q)
        if action_name in ("w", "a", "s", "d", "q"):
            self._press(action_name)
            self.current_key = action_name
            time.sleep(duration)
            return

        # Mouse turning actions
        _TURN_ACTIONS = {
            "turn_left":         (-35, 0),
            "turn_right":        (35,  0),
            "turn_left_micro":   (-5,  0),
            "turn_right_micro":  (5,   0),
            "turn_left_small":   (-10, 0),
            "turn_right_small":  (10,  0),
            "turn_left_large":   (-35, 0),
            "turn_right_large":  (35,  0),
        }
        if action_name in _TURN_ACTIONS:
            mx, my = _TURN_ACTIONS[action_name]
            if action_name in ("turn_left_micro", "turn_right_micro"):
                # Single-step micro turns
                self.move_mouse_constrained(mx, my)
                time.sleep(duration)
            else:
                steps = max(1, int(duration / 0.02))
                for _ in range(steps):
                    self.move_mouse_constrained(mx, my)
                    time.sleep(0.01)

    def _press(self, key: str):
        if IS_WINDOWS:
            _WIN_DIK = {
                'w': DIK_W, 'a': DIK_A, 's': DIK_S, 'd': DIK_D,
                'q': DIK_Q, 'f': DIK_F, 'r': DIK_R, 'g': DIK_G,
                '1': DIK_1, '2': DIK_2, '3': DIK_3, '4': DIK_4,
            }
            if key in _WIN_DIK:
                press_key(_WIN_DIK[key])
        else:
            press_key(key)

    def _release(self, key: str):
        if IS_WINDOWS:
            _WIN_DIK = {
                'w': DIK_W, 'a': DIK_A, 's': DIK_S, 'd': DIK_D,
                'q': DIK_Q, 'f': DIK_F, 'r': DIK_R, 'g': DIK_G,
                '1': DIK_1, '2': DIK_2, '3': DIK_3, '4': DIK_4,
            }
            if key in _WIN_DIK:
                release_key(_WIN_DIK[key])
        else:
            release_key(key)
