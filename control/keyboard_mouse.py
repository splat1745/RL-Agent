import time
import win32api
import win32con
import ctypes
import threading

# DirectInput Key Codes
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

# C struct definitions
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_ushort),
                ("wParamH", ctypes.c_ushort)]

class INPUT_I(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT),
                ("mi", MOUSEINPUT),
                ("hi", HARDWAREINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", INPUT_I)]

def press_key(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = ctypes.pointer(extra)
    # type=1 is INPUT_KEYBOARD
    # 0x0008 is KEYEVENTF_SCANCODE
    inp = INPUT()
    inp.type = 1
    inp.ii.ki = KEYBDINPUT(0, hexKeyCode, 0x0008, 0, ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

def release_key(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = ctypes.pointer(extra)
    # 0x0008 | 0x0002 (KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)
    inp = INPUT()
    inp.type = 1
    inp.ii.ki = KEYBDINPUT(0, hexKeyCode, 0x0008 | 0x0002, 0, ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

def move_mouse(x, y):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)

class InputController:
    def __init__(self):
        self.current_key = None
        # Mouse Constraint State
        try:
            self.start_mouse_pos = win32api.GetCursorPos()
        except:
            self.start_mouse_pos = (0, 0)
        self.accumulated_dy = 0 # Tracks vertical camera movement relative to start
        
        # Mouse Movement Tracking (Thread-Safe)
        self.recent_mouse_dx = 0
        self.recent_mouse_dy = 0
        self._lock = threading.Lock()

        # Cooldown Management
        self.last_used = {}
        # Removed hardcoded cooldowns to rely on visual detection
        self.cooldowns = {}

    def move_mouse_constrained(self, dx, dy):
        """
        Moves mouse with vertical constraint.
        Constraint: 
        - Cannot look UP past start (accumulated_dy >= 0)
        - Cannot look DOWN past start_y pixels (accumulated_dy <= start_y)
        Note: Mouse Down -> dy > 0. Mouse Up -> dy < 0.
        """
        # Proposed new accumulated dy
        new_accum = self.accumulated_dy + dy
        
        # Clamp
        # Limit UP: 0 (Start Pitch)
        # Limit DOWN: self.start_mouse_pos[1] (Screen Height roughly)
        max_down = self.start_mouse_pos[1]
        
        if new_accum < 0:
            # Trying to go too high
            dy = -self.accumulated_dy # Move just enough to reach 0
            new_accum = 0
        elif new_accum > max_down:
            # Trying to go too low
            dy = max_down - self.accumulated_dy
            new_accum = max_down
            
        self.accumulated_dy = new_accum
        
        if dx != 0 or dy != 0:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
            with self._lock:
                self.recent_mouse_dx += dx
                self.recent_mouse_dy += dy

    def get_recent_movement(self):
        """Returns accumulated mouse movement since last call."""
        with self._lock:
            dx = self.recent_mouse_dx
            dy = self.recent_mouse_dy
            self.recent_mouse_dx = 0
            self.recent_mouse_dy = 0
        return dx, dy

    def execute(self, action_name, duration=0.1):
        # Cooldown Check
        check_name = action_name
        if action_name == "r_2": check_name = "2" # Share cooldown key
        if action_name == "2": check_name = "2"
        
        if check_name in self.cooldowns:
            last = self.last_used.get(check_name, 0)
            if time.time() - last < self.cooldowns[check_name]:
                # On Cooldown - Ignore
                return
            # Mark as used
            self.last_used[check_name] = time.time()

        # Release previous key if different (and not a combo/instant)
        if self.current_key and self.current_key != action_name:
            self._release(self.current_key)
            self.current_key = None
        
        if action_name == "idle":
            time.sleep(duration)
            return
        
        if action_name == "click":
            click()
            time.sleep(duration)
            return
            
        if action_name == "m1_3": # 3x M1 (Replaces Click)
            click()
            time.sleep(0.1)
            click()
            time.sleep(0.1)
            click()
            time.sleep(duration)
            return

        if action_name == "reg_m4": # 4x M1
            for _ in range(4):
                click()
                time.sleep(0.1)
            time.sleep(duration)
            return

        if action_name == "down_m1": # 3x M1 + Space + M1
            # 3x M1
            for _ in range(3):
                click()
                time.sleep(0.1)
            # Space
            press_key(DIK_SPACE)
            time.sleep(0.05)
            release_key(DIK_SPACE)
            time.sleep(0.1)
            # M1
            click()
            time.sleep(duration)
            return

        if action_name == "up_m1": # 3x M1 + Hold Space + M1
            # 3x M1
            for _ in range(3):
                click()
                time.sleep(0.1)
            # Hold Space
            press_key(DIK_SPACE)
            time.sleep(0.1)
            # M1
            click()
            time.sleep(0.1)
            # Release Space
            release_key(DIK_SPACE)
            time.sleep(duration)
            return

        # Evasive Actions (Q + Direction)
        if action_name == "dash_left":
            press_key(DIK_A)
            press_key(DIK_Q)
            time.sleep(0.05)
            release_key(DIK_Q)
            release_key(DIK_A)
            time.sleep(duration)
            return

        if action_name == "dash_right":
            press_key(DIK_D)
            press_key(DIK_Q)
            time.sleep(0.05)
            release_key(DIK_Q)
            release_key(DIK_D)
            time.sleep(duration)
            return

        if action_name == "dash_back":
            press_key(DIK_S)
            press_key(DIK_Q)
            time.sleep(0.05)
            release_key(DIK_Q)
            release_key(DIK_S)
            time.sleep(duration)
            return

        if action_name == "dash_forward":
            press_key(DIK_W)
            press_key(DIK_Q)
            time.sleep(0.05)
            release_key(DIK_Q)
            release_key(DIK_W)
            time.sleep(duration)
            return

        # Combos
        # r_1 removed

        if action_name == "r_2":
            # R + 2: Press R, then 2
            press_key(DIK_R)
            time.sleep(0.05)
            self._press("2")
            time.sleep(0.05)
            self._release("2")
            time.sleep(0.05)
            release_key(DIK_R)
            time.sleep(duration)
            return

        # Explicit Tap for Skills (1-4, F, R, G)
        if action_name in ["1", "2", "3", "4", "f", "r", "g"]:
            self._press(action_name)
            time.sleep(0.05) # Short press
            self._release(action_name)
            time.sleep(duration)
            return

        # Hold for Movement (WASD)
        if action_name in ["w", "a", "s", "d", "q"]:
            self._press(action_name)
            self.current_key = action_name
            time.sleep(duration) # Hold the key
            
        elif action_name == "turn_left":
            # Move mouse in chunks to simulate smooth turning over duration
            steps = int(duration / 0.02)
            if steps < 1: steps = 1
            for _ in range(steps):
                self.move_mouse_constrained(-35, 0) # Increased speed for responsiveness
                time.sleep(0.01)
                
        elif action_name == "turn_right":
            steps = int(duration / 0.02)
            if steps < 1: steps = 1
            for _ in range(steps):
                self.move_mouse_constrained(35, 0) # Increased speed for responsiveness
                time.sleep(0.01)

        elif action_name == "turn_left_micro":
            self.move_mouse_constrained(-5, 0)
            time.sleep(duration)

        elif action_name == "turn_right_micro":
            self.move_mouse_constrained(5, 0)
            time.sleep(duration)

        elif action_name == "turn_left_small":
            steps = int(duration / 0.02)
            if steps < 1: steps = 1
            for _ in range(steps):
                self.move_mouse_constrained(-10, 0) # Total ~20-30 depending on duration
                time.sleep(0.01)

        elif action_name == "turn_right_small":
            steps = int(duration / 0.02)
            if steps < 1: steps = 1
            for _ in range(steps):
                self.move_mouse_constrained(10, 0)
                time.sleep(0.01)

        elif action_name == "turn_left_large":
            steps = int(duration / 0.02)
            if steps < 1: steps = 1
            for _ in range(steps):
                self.move_mouse_constrained(-35, 0)
                time.sleep(0.01)

        elif action_name == "turn_right_large":
            steps = int(duration / 0.02)
            if steps < 1: steps = 1
            for _ in range(steps):
                self.move_mouse_constrained(35, 0)
                time.sleep(0.01)

        # Removed look_up and look_down from RL control

    def _press(self, key):
        if key == "w": press_key(DIK_W)
        elif key == "a": press_key(DIK_A)
        elif key == "s": press_key(DIK_S)
        elif key == "d": press_key(DIK_D)
        elif key == "q": press_key(DIK_Q)
        elif key == "f": press_key(DIK_F)
        elif key == "r": press_key(DIK_R)
        elif key == "g": press_key(DIK_G)
        elif key == "1": press_key(DIK_1)
        elif key == "2": press_key(DIK_2)
        elif key == "3": press_key(DIK_3)
        elif key == "4": press_key(DIK_4)

    def _release(self, key):
        if key == "w": release_key(DIK_W)
        elif key == "a": release_key(DIK_A)
        elif key == "s": release_key(DIK_S)
        elif key == "d": release_key(DIK_D)
        elif key == "q": release_key(DIK_Q)
        elif key == "f": release_key(DIK_F)
        elif key == "r": release_key(DIK_R)
        elif key == "g": release_key(DIK_G)
        elif key == "1": release_key(DIK_1)
        elif key == "2": release_key(DIK_2)
        elif key == "3": release_key(DIK_3)
        elif key == "4": release_key(DIK_4)
