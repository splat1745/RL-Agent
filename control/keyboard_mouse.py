import time
import win32api
import win32con
import ctypes

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

    def execute(self, action_name, duration=0.1):
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
            
        # Evasive Actions (Q + Direction)
        if action_name == "dash_left":
            # Hold A, Tap Q
            press_key(DIK_A)
            time.sleep(0.05)
            press_key(DIK_Q)
            time.sleep(0.05)
            release_key(DIK_Q)
            time.sleep(duration)
            release_key(DIK_A)
            return

        if action_name == "dash_right":
            # Hold D, Tap Q
            press_key(DIK_D)
            time.sleep(0.05)
            press_key(DIK_Q)
            time.sleep(0.05)
            release_key(DIK_Q)
            time.sleep(duration)
            release_key(DIK_D)
            return

        if action_name == "dash_back":
            # Hold S, Tap Q
            press_key(DIK_S)
            time.sleep(0.05)
            press_key(DIK_Q)
            time.sleep(0.05)
            release_key(DIK_Q)
            time.sleep(duration)
            release_key(DIK_S)
            return

        # Combos
        if action_name == "r_1":
            # R + 1: Press 1, then Spam R
            self._press("1")
            time.sleep(0.05)
            self._release("1")
            time.sleep(0.05)
            # Spam R
            for _ in range(3):
                press_key(DIK_R)
                time.sleep(0.03)
                release_key(DIK_R)
                time.sleep(0.03)
            time.sleep(duration)
            return

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

        if action_name in ["w", "a", "s", "d", "q", "f", "r", "g", "1", "2", "3", "4"]:
            self._press(action_name)
            self.current_key = action_name
            time.sleep(duration) # Hold the key
            
        elif action_name == "turn_left":
            # Move mouse in chunks to simulate smooth turning over duration
            steps = int(duration / 0.02)
            for _ in range(steps):
                self.move_mouse_constrained(-15, 0) # Reduced speed to prevent blur/exploits
                time.sleep(0.01)
                
        elif action_name == "turn_right":
            steps = int(duration / 0.02)
            for _ in range(steps):
                self.move_mouse_constrained(15, 0) # Reduced speed to prevent blur/exploits
                time.sleep(0.01)

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
