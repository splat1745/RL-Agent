import numpy as np

class RewardCalculator:
    def __init__(self):
        self.last_hp = 1.0
        self.last_enemy_hp = 1.0 # If we can track it
        self.survival_time = 0
        self.last_pos = None # For movement
        
    def calculate(self, perception, action_id):
        """
        Calculates reward based on visual heuristics.
        perception: The Perception object (access to health, tracker, etc.)
        action_id: The action taken
        """
        reward = 0.0
        
        # 1. Health Delta (Self)
        # Assuming perception.filtered_health is 0.0-1.0
        current_hp = perception.filtered_health
        hp_change = current_hp - self.last_hp
        
        if hp_change < 0:
            # Took damage
            reward += hp_change * 5.0 # Penalty (e.g. -0.1 -> -0.5)
            
        self.last_hp = current_hp
        
        # 2. Survival Reward
        # Tiny positive reward for just staying alive per step
        reward += 0.001
        
        # 3. Movement Reward (Optical Flow or Tracker)
        # Encourages not standing still
        # perception.last_flow contains flow magnitude?
        # Or check perception input (which has flow)
        # Let's use flow magnitude if available in perception
        # Assuming perception.last_flow is (2, H, W)
        if hasattr(perception, 'last_flow_mag'):
             # If we track scalar magnitude
             reward += min(perception.last_flow_mag * 0.01, 0.05)
        
        # 4. Hit Registration (Visual Flash or Ragdoll)
        # perception.ragdoll_counters -> dict of detected ragdolls/hits?
        # If we detect a "Hit" UI element or particle
        # This part requires perception to flag "HitDetected"
        if getattr(perception, 'hit_detected', False):
             reward += 0.5
             perception.hit_detected = False # Reset
             
        # 5. Enemy Health (If available)
        # ...
        
        return reward
        
    def reset(self):
        self.last_hp = 1.0
        self.survival_time = 0
