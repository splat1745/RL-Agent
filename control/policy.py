import numpy as np
from control.actions import ACTION_MAP

class DirectPolicy:
    def __init__(self):
        pass
        
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
        
        # If enemy not seen recently, maybe turn to find?
        if time_since_seen > 0.1:
            # Spin to find
            return 6 # Turn Right
            
        # Calculate distance
        dist = np.sqrt(edx*edx + edy*edy)
        
        # 1. Aiming (Horizontal)
        # Center is 0.0. Range -1 to 1.
        # Threshold for turning
        aim_threshold = 0.15
        
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
