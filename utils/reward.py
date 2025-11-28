import numpy as np

def calculate_reward(obs, action_idx, state_manager):
    """
    Calculates reward based on simplified logic:
    1. Get closer to enemy
    2. Hit enemy (Flash)
    3. Avoid damage (Health drop)
    """
    # obs: [px, py, vx, vy, edx, edy, odx, ody, dist_goal, time_since_seen, health, enemy_flash, ...]
    
    current_health = 1.0
    enemy_flash = 0.0
    
    if len(obs) > 11:
        current_health = obs[10]
        enemy_flash = obs[11]
        
    # Get deltas from state manager
    # delta_enemy = dist_current - dist_prev
    delta_enemy, _ = state_manager.get_deltas(obs)
    
    reward = 0.0
    
    # 1. Distance Reward (Chase)
    # Reward getting closer (negative delta)
    # Increased weight to encourage aggressive chasing
    reward += -delta_enemy * 2.0
    
    # Proximity Bonus (Maintain close range)
    # obs[4], obs[5] are edx, edy (normalized)
    if len(obs) > 5:
        edx, edy = obs[4], obs[5]
        dist = np.sqrt(edx*edx + edy*edy)
        if dist < 0.15: # Close range (approx 100 pixels)
            reward += 0.05 # Small bonus per frame for staying close
    
    # 2. Attack Reward (Hit)
    if enemy_flash > 0.1:
        reward += 2.0
        
    # 3. Damage Penalty (Get Hit)
    if state_manager.prev_health is not None:
        # Use a small threshold to avoid float noise, though health is usually discrete-ish
        if current_health < state_manager.prev_health - 0.001:
            reward -= 2.0

     # add constant penalty to always encourage improvement
    reward -= 0.01       
    state_manager.update(obs)
    
    return reward
