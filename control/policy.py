import numpy as np
from control.actions import ACTION_MAP

class DirectPolicy:
    def __init__(self):
        self.scan_timer = 0
        self.last_seen_time = 0
        self.last_edx = 0.0
        self.recovery_step = 0
        self.recovery_timer = 0
        self.vertical_scan_dir = 1 # 1 for up, -1 for down
        
    def select_action(self, vector_obs):
        """
        Determines action based on vector observation.
        vector_obs: [px, py, vx, vy, edx, edy, odx, ody, dist_goal, time_since_seen, ...]
        """
        # Default
        action_idx = 0 # Idle
        
        if len(vector_obs) < 10:
            return action_idx
            
        # Extract relevant features
        # edx, edy are normalized direction to enemy
        edx = vector_obs[4]
        edy = vector_obs[5]
        time_since_seen = vector_obs[9]
        
        # If enemy not seen recently (approx 0.25s), enter recovery/scan mode
        # Threshold 0.05 corresponds to 0.25s (since normalized by 5.0)
        if time_since_seen > 0.05:
            # Recovery Logic: Backtrack if we just lost it
            if self.recovery_step < 3:
                self.recovery_timer += 1
                
                # Cycle: Adjust (5 frames) -> Wait/Vertical (15 frames)
                # Total cycle = 20 frames (~1 second at 20fps)
                cycle_len = 20
                
                if self.recovery_timer > cycle_len:
                    self.recovery_step += 1
                    self.recovery_timer = 0
                    self.vertical_scan_dir *= -1 # Toggle up/down
                    return 0 # Transition
                
                if self.recovery_timer <= 5:
                    # Phase 1: Horizontal Adjustment (Undo overshoot)
                    if self.last_edx > 0: 
                        return 5 # Turn Left
                    else: 
                        return 6 # Turn Right
                else:
                    # Phase 2: Wait & Vertical Scan
                    if self.vertical_scan_dir > 0:
                        return 19 # Look Up
                    else:
                        return 20 # Look Down
            
            # If recovery failed, start Burst Scan
            self.scan_timer += 1
            # Burst turn: Turn for 5 frames, wait for 10 frames
            # Assuming ~20-30 FPS
            cycle = 15
            phase = self.scan_timer % cycle
            
            if phase < 5:
                return 6 # Turn Right
            else:
                return 0 # Idle (Let camera settle for detection)
        else:
            # Enemy Visible
            self.scan_timer = 0
            self.recovery_step = 0
            self.recovery_timer = 0
            self.last_edx = edx
            
        # Calculate distance
        dist = np.sqrt(edx*edx + edy*edy)
        
        # 1. Aiming (Horizontal)
        # Center is 0.0. Range -1 to 1.
        # Threshold for turning
        # Tightened to 0.05 to force "lock on" before moving
        aim_threshold = 0.05
        
        if edx < -aim_threshold:
            return 5 # Turn Left
        elif edx > aim_threshold:
            return 6 # Turn Right
            
        # 2. Movement (Vertical/Distance)
        # If aimed reasonably well, move
        target_dist = 0.2 # Keep some distance
        
        if dist > target_dist:
            return 1 # W (Forward)
        elif dist < 0.1:
            return 3 # S (Back)
            
        return 0 # Idle if in sweet spot

    @staticmethod
    def calculate_aim_assist(vector_obs, gain=300.0):
        """
        Calculates mouse movement (dx, dy) to lock onto target.
        """
        if len(vector_obs) < 10: return 0, 0
        
        edx = vector_obs[4]
        edy = vector_obs[5]
        time_since_seen = vector_obs[9]
        
        if time_since_seen < 0.1:
            move_x = int(edx * gain)
            move_y = int(edy * gain * 0.5)
            return move_x, move_y
        return 0, 0

