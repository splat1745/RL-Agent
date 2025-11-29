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
    
    # Proximity Penalty (Stay within range)
    # obs[4], obs[5] are edx, edy (normalized)
    if len(obs) > 5:
        edx, edy = obs[4], obs[5]
        dist = np.sqrt(edx*edx + edy*edy)
        
        # Penalize if too far away (e.g. > 30% of screen width)
        if dist > 0.3:
            reward -= 0.1
            
        # Target Lock Reward: Encourage keeping enemy in center
        # Boosted to strongly encourage facing the enemy
        # Max +0.5 when centered
        reward += max(0, (0.5 - dist) * 1.0)

        # Action Penalty: Don't look away or retreat if you have the target!
        # If enemy is visible (dist < 1.0 implies we have a valid coordinate, usually)
        # And enemy is reasonably centered (dist < 0.4)
        if dist < 0.4:
            # 5: Turn Left, 6: Turn Right
            if action_idx in [5, 6]:
                reward -= 0.5 # Penalty for turning away when locked on
            
            # 3: S, 7: Dash Back
            if action_idx in [3, 7]:
                reward -= 0.5 # Penalty for retreating
            
            # 7: Dash Back, 8: Dash Left, 9: Dash Right
            if action_idx in [7, 8, 9]:
                reward -= 0.2 # Slight penalty for dashing when locked on (unless dodging, but we don't know that yet)

    # Lost Target / Visibility Penalty
    # obs[9] is time_since_seen_norm (0..1)
    if len(obs) > 9:
        time_since_seen = obs[9]
        
        # Immediate penalty for not seeing enemy this frame
        # time_since_seen is normalized. If > 0 (approx), it means we didn't see it this frame.
        # Using a small epsilon because of float precision/execution time
        if time_since_seen > 0.01:
            reward -= 0.1 # Constant pressure to find enemy
            
        # Heavy penalty if lost for longer (> 0.25s)
        if time_since_seen > 0.05:
            reward -= 0.5
    
    # 2. Attack Reward (Hit) - REMOVED

    # Lost Target / Visibility Penalty
    # obs[9] is time_since_seen_norm (0..1)
    if len(obs) > 9:
        time_since_seen = obs[9]
        
        # Immediate penalty for not seeing enemy this frame
        # time_since_seen is normalized. If > 0 (approx), it means we didn't see it this frame.
        # Using a small epsilon because of float precision/execution time
        if time_since_seen > 0.01:
            reward -= 0.1 # Constant pressure to find enemy
            
        # Heavy penalty if lost for longer (> 0.25s)
        if time_since_seen > 0.05:
            reward -= 0.5
    
    # 2. Attack Reward (Hit) - REMOVED
    # if enemy_flash > 0.1:
    #    reward += 2.0
        
    # 3. Damage Penalty (Get Hit)
    if state_manager.prev_health is not None:
        # Use a small threshold to avoid float noise, though health is usually discrete-ish
        if current_health < state_manager.prev_health - 0.001:
            reward -= 2.0

     # add constant penalty to always encourage improvement
    reward -= 0.01       
    state_manager.update(obs)
    
    return reward
