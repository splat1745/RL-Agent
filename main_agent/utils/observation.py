import numpy as np

class ObservationBuilder:
    def __init__(self):
        pass
        
    def build(self, det_dict, pose_data, combat_state, self_hp=100, cooldowns=None):
        """
        Combines all sensor data into a single observation vector.
        Step 9: Sensor Fusion
        """
        
        # Placeholder for vector construction
        # {
        #   "self_pos": [...],
        #   "self_hp": 83,
        #   "enemy_pos": [...],
        #   "enemy_state": "windup",
        #   ...
        # }
        
        # For RL, we usually flatten this to a numpy array
        
        obs = {
            "enemy_pos": det_dict.get("enemies", []),
            "combat_state": combat_state,
            "hp": self_hp
        }
        
        return obs
