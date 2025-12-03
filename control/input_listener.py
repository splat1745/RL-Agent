import win32api
import win32con
import time

class InputListener:
    def __init__(self):
        self.last_mouse_pos = win32api.GetCursorPos()
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        
        # Key Codes
        self.VK_LBUTTON = 0x01
        self.VK_SPACE = 0x20
        self.VK_1 = 0x31
        self.VK_2 = 0x32
        self.VK_3 = 0x33
        self.VK_4 = 0x34
        self.VK_A = 0x41
        self.VK_D = 0x44
        self.VK_F = 0x46
        self.VK_G = 0x47
        self.VK_Q = 0x51
        self.VK_R = 0x52
        self.VK_S = 0x53
        self.VK_W = 0x57

    def is_pressed(self, vk_code):
        # GetAsyncKeyState returns a short. If the high-order bit is 1, the key is down.
        return win32api.GetAsyncKeyState(vk_code) & 0x8000

    def get_user_action(self):
        """
        Polls hardware state and returns the corresponding action ID.
        Prioritizes Combos > Skills > Movement > Mouse > Idle.
        """
        
        # 1. Check Combos / Modifiers
        is_q = self.is_pressed(self.VK_Q)
        is_r = self.is_pressed(self.VK_R)
        
        if is_q:
            if self.is_pressed(self.VK_W): return 10 # Dash Forward
            if self.is_pressed(self.VK_A): return 8  # Dash Left
            if self.is_pressed(self.VK_S): return 7  # Dash Back
            if self.is_pressed(self.VK_D): return 9  # Dash Right
            
        if is_r:
            if self.is_pressed(self.VK_2): return 15 # R + 2
            
        # 2. Check Single Keys
        if self.is_pressed(self.VK_1): return 11
        if self.is_pressed(self.VK_2): return 12
        if self.is_pressed(self.VK_3): return 13
        if self.is_pressed(self.VK_4): return 14
        
        if self.is_pressed(self.VK_F): return 18
        if self.is_pressed(self.VK_G): return 16
        if self.is_pressed(self.VK_SPACE): return 17
        if self.is_pressed(self.VK_LBUTTON): return 19 # M1
        
        # 3. Check Movement (WASD)
        # Note: Agent can only do one thing at a time. 
        # If user presses W and A, we pick one (W).
        if self.is_pressed(self.VK_W): return 1
        if self.is_pressed(self.VK_A): return 2
        if self.is_pressed(self.VK_S): return 3
        if self.is_pressed(self.VK_D): return 4
        
        # 4. Check Mouse Movement
        current_pos = win32api.GetCursorPos()
        dx = current_pos[0] - self.last_mouse_pos[0]
        self.last_mouse_pos = current_pos
        
        # Thresholds for turning
        # These need to be tuned to match the "feel" of the agent's discrete turns
        if abs(dx) > 2:
            if dx < -40: return 24 # Left Large
            if dx < -15: return 22 # Left Small
            if dx < -2:  return 20 # Left Micro
            
            if dx > 40: return 25 # Right Large
            if dx > 15: return 23 # Right Small
            if dx > 2:  return 21 # Right Micro
            
        return 0 # Idle
