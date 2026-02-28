import sys
import time

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import win32api
    import win32con


class InputListener:
    """
    Polls human input state and maps it to the 26-action action space.

    Windows: uses win32api.GetAsyncKeyState for zero-latency polling.
    Linux:   uses a pynput Listener to maintain a set of currently pressed keys.
    """

    # VK / action constants (shared)
    VK_LBUTTON = 0x01
    VK_SPACE    = 0x20
    VK_1 = 0x31; VK_2 = 0x32; VK_3 = 0x33; VK_4 = 0x34
    VK_A = 0x41; VK_D = 0x44; VK_F = 0x46; VK_G = 0x47
    VK_Q = 0x51; VK_R = 0x52; VK_S = 0x53; VK_W = 0x57

    def __init__(self):
        # ---- screen size ----
        self.screen_width, self.screen_height = self._get_screen_size()

        # ---- per-platform setup ----
        if IS_WINDOWS:
            self.last_mouse_pos = win32api.GetCursorPos()
        else:
            from pynput import keyboard as _pykb, mouse as _pymouse
            from pynput.keyboard import Key as _Key

            self._Key = _Key
            self._pressed_chars = set()    # lowercase single chars, e.g. {'w','a'}
            self._pressed_special = set()  # pynput Key objects, e.g. {Key.space}
            self._left_button_down = False

            # Map VK codes → identifiers stored in the pressed sets
            self._VK_TO_ID = {
                self.VK_SPACE:   _Key.space,
                self.VK_1: '1', self.VK_2: '2', self.VK_3: '3', self.VK_4: '4',
                self.VK_A: 'a', self.VK_D: 'd', self.VK_F: 'f', self.VK_G: 'g',
                self.VK_Q: 'q', self.VK_R: 'r', self.VK_S: 's', self.VK_W: 'w',
            }

            def _on_press(key):
                try:
                    c = key.char
                    if c:
                        self._pressed_chars.add(c.lower())
                except AttributeError:
                    self._pressed_special.add(key)

            def _on_release(key):
                try:
                    c = key.char
                    if c:
                        self._pressed_chars.discard(c.lower())
                except AttributeError:
                    self._pressed_special.discard(key)

            def _on_mouse_click(x, y, button, pressed):
                if button == _pymouse.Button.left:
                    self._left_button_down = pressed

            self._kb_listener = _pykb.Listener(
                on_press=_on_press, on_release=_on_release)
            self._kb_listener.daemon = True
            self._kb_listener.start()

            self._ms_listener = _pymouse.Listener(on_click=_on_mouse_click)
            self._ms_listener.daemon = True
            self._ms_listener.start()

            self._mouse_ctrl = _pymouse.Controller()
            self.last_mouse_pos = self._mouse_ctrl.position

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_screen_size():
        if IS_WINDOWS:
            import win32api as _w32
            return _w32.GetSystemMetrics(0), _w32.GetSystemMetrics(1)
        try:
            import subprocess
            out = subprocess.check_output(
                ["xdpyinfo"], stderr=subprocess.DEVNULL, timeout=3
            ).decode()
            for line in out.splitlines():
                if "dimensions:" in line:
                    part = line.split("dimensions:")[1].strip().split()[0]  # e.g. '1920x1080'
                    w, h = part.split("x")
                    return int(w), int(h)
        except Exception:
            pass
        return 1920, 1080  # sensible fallback

    def is_pressed(self, vk_code) -> bool:
        if IS_WINDOWS:
            return bool(win32api.GetAsyncKeyState(vk_code) & 0x8000)
        else:
            if vk_code == self.VK_LBUTTON:
                return self._left_button_down
            ident = self._VK_TO_ID.get(vk_code)
            if ident is None:
                return False
            if isinstance(ident, str):
                return ident in self._pressed_chars
            return ident in self._pressed_special

    def _cursor_pos(self):
        if IS_WINDOWS:
            return win32api.GetCursorPos()
        pos = self._mouse_ctrl.position
        return (int(pos[0]), int(pos[1]))

    # ------------------------------------------------------------------
    # Action mapping (identical logic on both platforms)
    # ------------------------------------------------------------------

    def get_user_action(self) -> int:
        """
        Polls hardware state and returns the corresponding action ID.
        Priority: Combos > Skills > Movement > Mouse > Idle.
        """
        is_q = self.is_pressed(self.VK_Q)
        is_r = self.is_pressed(self.VK_R)

        if is_q:
            if self.is_pressed(self.VK_W): return 10  # Dash Forward
            if self.is_pressed(self.VK_A): return 8   # Dash Left
            if self.is_pressed(self.VK_S): return 7   # Dash Back
            if self.is_pressed(self.VK_D): return 9   # Dash Right

        if is_r:
            if self.is_pressed(self.VK_2): return 15  # R + 2

        if self.is_pressed(self.VK_1): return 11
        if self.is_pressed(self.VK_2): return 12
        if self.is_pressed(self.VK_3): return 13
        if self.is_pressed(self.VK_4): return 14

        if self.is_pressed(self.VK_F):       return 18
        if self.is_pressed(self.VK_G):       return 16
        if self.is_pressed(self.VK_SPACE):   return 17
        if self.is_pressed(self.VK_LBUTTON): return 19  # M1

        if self.is_pressed(self.VK_W): return 1
        if self.is_pressed(self.VK_A): return 2
        if self.is_pressed(self.VK_S): return 3
        if self.is_pressed(self.VK_D): return 4

        # Mouse movement
        current_pos = self._cursor_pos()
        dx = current_pos[0] - self.last_mouse_pos[0]
        self.last_mouse_pos = current_pos

        if abs(dx) > 2:
            if dx < -40: return 24   # Left Large
            if dx < -15: return 22   # Left Small
            if dx <  -2: return 20   # Left Micro
            if dx >  40: return 25   # Right Large
            if dx >  15: return 23   # Right Small
            if dx >   2: return 21   # Right Micro

        return 0  # Idle

    def get_raw_input(self) -> dict:
        """Returns a dict of raw key/button states for imitation recording."""
        return {
            'W':     self.is_pressed(self.VK_W),
            'A':     self.is_pressed(self.VK_A),
            'S':     self.is_pressed(self.VK_S),
            'D':     self.is_pressed(self.VK_D),
            'Space': self.is_pressed(self.VK_SPACE),
            'M1':    self.is_pressed(self.VK_LBUTTON),
            '1':     self.is_pressed(self.VK_1),
            '2':     self.is_pressed(self.VK_2),
            '3':     self.is_pressed(self.VK_3),
            '4':     self.is_pressed(self.VK_4),
            'Q':     self.is_pressed(self.VK_Q),
            'R':     self.is_pressed(self.VK_R),
            'F':     self.is_pressed(self.VK_F),
            'G':     self.is_pressed(self.VK_G),
        }
