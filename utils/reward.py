import numpy as np
import time

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
    dist = 1.0
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

        # --- Move Usage Rewards ---
        # Center Constraint: Only reward moves if enemy is within 3% of center (dist < 0.03)
        # 3% of screen width (normalized -1 to 1) is 0.06 range, so dist < 0.03
        is_centered = dist < 0.03
        
        # Move 1 (Tracking) & Move 2 (Projectile): Reward if enemy is visible
        # "Using move 1 and move 2 while enemy bounding box is small = small reward"
        # relative_size is at index 20
        relative_size = 0.0
        if len(obs) > 20:
            relative_size = obs[20]

        if action_idx in [11, 12]: # "1", "2"
            if is_centered and relative_size > 0: # Enemy visible and centered
                if relative_size < 0.3: # Far away / Small
                    reward += 0.2 # Small reward for using ranged moves at range
                else:
                    reward += 0.1 # Smaller reward if close (maybe should be penalty? User didn't specify)
            else:
                reward -= 0.1 # Wasted move (not centered or no enemy)
            
        # Moves 3 & 4 (Close Up): Reward if enemy is close (Relative Size check)
        if action_idx in [13, 14]: # "3", "4"
            # "Using move 3 or 4 while enemy is far = -reward while when enemy close = +reward"
            if is_centered and relative_size >= 0.7: # Close and Centered
                reward += 1.0 # Big reward for correct usage
            elif relative_size > 0: # Visible but far
                reward -= 0.5 # Penalty for using close-up move when far
            else:
                reward -= 0.5 # Wasted move

        # Mode Usage (G): Reward if health is low
        if action_idx == 17: # "g"
            if current_health < 0.4:
                reward += 2.0 # Big reward for using mode when low health
            else:
                reward -= 0.5 # Penalty for wasting mode when healthy
                
        # M1 Usage (Click)
        if action_idx == 18: # "click"
            # "When doing at least 3 concecutive m1s and no health is lost, then assume he hit them"
            # We check if this is the 3rd (or more) consecutive click
            if state_manager.consecutive_m1_count >= 3:
                # Check if health lost in this frame (simple check)
                # Ideally we check over the duration of the clicks, but "no health lost" usually implies "didn't get hit back immediately"
                if current_health >= state_manager.prev_health:
                    reward += 0.5 # Reward for sustained combo without taking damage
            
            # Basic M1 reward if close and centered
            if is_centered and relative_size > 0.5:
                reward += 0.1

    # Delayed Safety Reward (Dash/Moves)
    # "if no health is lost after 1.6 seconds then assume a positive reward"
    now = time.time()
    for act, ts, hp in state_manager.action_history:
        # Check if action was ~1.6s ago (1.5 to 1.7 window)
        if 1.5 < (now - ts) < 1.7:
            # Check if health has dropped since then
            # We compare current_health to hp (health at time of action)
            # Allow small float error
            if current_health >= hp - 0.001:
                # It was a safe move!
                # Give reward based on action type
                if act == 10: # Dash Forward
                    reward += 0.5
                elif act in [7, 8, 9]: # Evasive Dashes
                    reward += 0.5
                elif act in [11, 12, 13, 14, 18]: # Attacks
                    reward += 0.2 # Safe attack

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
    state_manager.update(obs, action_idx)
    
    return reward
