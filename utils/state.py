import time

class StateManager:
    def __init__(self):
        self.prev_dist_enemy = None
        self.prev_dist_goal = None
        self.prev_time_since_seen = None
        self.prev_health = 1.0
        self.prev_enemy_health = 1.0
        
        # New State Tracking
        self.consecutive_m1_count = 0
        self.last_action = None
        self.action_history = [] # List of (action_idx, timestamp, health_at_time)
        
    def update(self, obs, action_idx):
        # obs: [px, py, vx, vy, edx, edy, odx, ody, dist_goal, time_since_seen, health, enemy_health, ...]
        edx, edy = obs[4], obs[5]
        dist_enemy = (edx**2 + edy**2)**0.5
        
        dist_goal = obs[8]
        time_since_seen = obs[9]
        
        current_health = 1.0
        if len(obs) > 10:
            current_health = obs[10]
            
        if len(obs) > 11:
            self.prev_enemy_health = obs[11]
        
        # Update M1 Count
        if action_idx == 18: # Click
            self.consecutive_m1_count += 1
        else:
            self.consecutive_m1_count = 0
            
        # Update History
        now = time.time()
        self.action_history.append((action_idx, now, current_health))
        # Keep history short (last 5 seconds)
        self.action_history = [x for x in self.action_history if now - x[1] < 5.0]
        
        self.prev_health = current_health
        self.prev_dist_enemy = dist_enemy
        self.prev_dist_goal = dist_goal
        self.prev_time_since_seen = time_since_seen
        self.last_action = action_idx
        
    def get_deltas(self, obs):
        edx, edy = obs[4], obs[5]
        dist_enemy = (edx**2 + edy**2)**0.5
        dist_goal = obs[8]
        
        delta_enemy = 0.0
        if self.prev_dist_enemy is not None:
            delta_enemy = dist_enemy - self.prev_dist_enemy
            
        delta_goal = 0.0
        if self.prev_dist_goal is not None:
            delta_goal = self.prev_dist_goal - dist_goal # Positive if got closer
            
        return delta_enemy, delta_goal

