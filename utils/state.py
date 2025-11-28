class StateManager:
    def __init__(self):
        self.prev_dist_enemy = None
        self.prev_dist_goal = None
        self.prev_time_since_seen = None
        self.prev_health = 1.0
        self.prev_enemy_health = 1.0
        
    def update(self, obs):
        # obs: [px, py, vx, vy, edx, edy, odx, ody, dist_goal, time_since_seen, health, enemy_health, ...]
        edx, edy = obs[4], obs[5]
        dist_enemy = (edx**2 + edy**2)**0.5
        
        dist_goal = obs[8]
        time_since_seen = obs[9]
        
        # Handle health if present
        if len(obs) > 10:
            self.prev_health = obs[10]
            
        if len(obs) > 11:
            self.prev_enemy_health = obs[11]
        
        self.prev_dist_enemy = dist_enemy
        self.prev_dist_goal = dist_goal
        self.prev_time_since_seen = time_since_seen
        
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
