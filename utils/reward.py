import numpy as np
import time

def calculate_reward(obs, action_idx, state_manager):
    """
    Calculates reward based on strict spatial and temporal logic.
    Enforces "No Random Moves" policy and "Damage Reflection".
    """
    # obs: [px, py, vx, vy, edx, edy, odx, ody, dist_goal, time_since_seen, health, enemy_flash, ..., relative_size, is_overlapping, leaks_above]
    
    current_health = 1.0
    enemy_flash = 0.0
    
    if len(obs) > 11:
        current_health = obs[10]
        enemy_flash = obs[11]
        
    # Get deltas from state manager
    delta_enemy, _ = state_manager.get_deltas(obs)
    
    reward = 0.0
    
    # 1. Distance Reward (Chase)
    # Reward getting closer (negative delta)
    reward += -delta_enemy * 2.0
    
    # Proximity Penalty (Stay within range)
    dist = 1.0
    if len(obs) > 5:
        edx, edy = obs[4], obs[5]
        dist = np.sqrt(edx*edx + edy*edy)
        
        # Penalize if too far away (e.g. > 30% of screen width)
        if dist > 0.3:
            reward -= 0.1
            
        # Target Lock Reward: Encourage keeping enemy in center
        # "reward more through aiming moves"
        # Increased weight from 1.0 to 2.0
        reward += max(0, (0.5 - dist) * 2.0)

        # Action Penalty: Don't look away or retreat if you have the target!
        if dist < 0.4:
            # 5: Turn Left, 6: Turn Right, 22: Slow Left, 23: Slow Right
            if action_idx in [5, 6, 22, 23]:
                reward -= 0.5 
            # 3: S, 7: Dash Back
            if action_idx in [3, 7]:
                reward -= 0.5 
            # 7: Dash Back, 8: Dash Left, 9: Dash Right
            if action_idx in [7, 8, 9]:
                reward -= 0.2 
                
    # --- BLOCK SUCCESS LOGIC ---
    # "implement a method of figuring out if the block was successful or not"
    # Logic: Action is Block (18) AND Enemy is Attacking (enemy_flash > 0.1) AND No Health Loss
    if action_idx == 18: # Block
        if enemy_flash > 0.1: # Enemy is attacking
            # Check if health was lost in this step (approx)
            # state_manager.prev_health is from previous step
            damage_taken = 0.0
            if state_manager.prev_health is not None:
                damage_taken = max(0.0, state_manager.prev_health - current_health)
            
            if damage_taken < 0.001:
                # Successful Block!
                reward += 2.0
            else:
                # Failed Block (Chip damage or broken guard)
                reward += 0.1 # Still better than taking full damage?
        else:
            # Blocking for no reason
            reward -= 0.1

    # --- STRICT DELAYED REWARD LOGIC ---
    # "CANNOT do random moves and get rewarded UNLESS..."
    # 1. Enemy bounding box leaks above player (leaks_above > 0.5)
    # 2. Is in front of player (is_overlapping > 0.5 covers touching/in-front)
    # 3. Is in center of screen (dist < 0.1)
    # 4. Doesn't get damaged 1.618 seconds after doing move (Golden Ratio Delay)
    
    now = time.time()
    for act, ts, past_obs in state_manager.action_history:
        # Check if action was ~1.618s ago (1.55 to 1.7 window)
        if 1.55 < (now - ts) < 1.7:
            # Only check for Moves (1-4), M1 (19), and Combos (15)
            if act in [11, 12, 13, 14, 19, 15]:
                
                # Condition 4: No damage taken since then (Improved Hit Detection)
                # Check if any hit occurred in the window [ts, now]
                was_hit = state_manager.was_hit_in_window(ts, now)
                
                if not was_hit:
                    # Check Past Context
                    p_edx, p_edy = past_obs[4], past_obs[5]
                    p_dist = np.sqrt(p_edx*p_edx + p_edy*p_edy)
                    
                    p_is_overlapping = 0.0
                    p_leaks_above = 0.0
                    p_time_since_seen = 1.0 # Default to not seen
                    
                    if len(past_obs) > 9:
                        p_time_since_seen = past_obs[9]
                    if len(past_obs) > 21:
                        p_is_overlapping = past_obs[21]
                    if len(past_obs) > 22:
                        p_leaks_above = past_obs[22]
                    
                    # Conditions 1, 2, 3 + Visibility Check
                    # Center (< 0.1), Overlapping (> 0.5), Leaks Above (> 0.5), Visible (< 0.1)
                    if p_dist < 0.1 and p_is_overlapping > 0.5 and p_leaks_above > 0.5 and p_time_since_seen < 0.1:
                        # SUCCESS!
                        # "reward more for successful attacks"
                        reward += 3.0 # Increased from 2.0
                        
                        # Bonus for specific moves if needed
                        if act in [13, 14, 15, 19]: # Close range moves / Combos
                            reward += 1.5
                    else:
                        # Failed spatial conditions -> Penalty for wasting move
                        reward -= 0.5
                else:
                    # Failed safety condition -> Penalty for unsafe move (Got hit during/after move)
                    reward -= 1.0

    # --- CONSECUTIVE M1 REWARD ---
    # Reward chaining M1s (19)
    if action_idx == 19:
        # Only reward if enemy is visible
        is_visible = False
        if len(obs) > 9 and obs[9] < 0.1:
            is_visible = True
            
        if state_manager.consecutive_m1_count > 0 and is_visible:
            reward += 0.1 * min(5, state_manager.consecutive_m1_count) # Small bonus for chaining
    
    # --- REFLECTIVE LEARNING: DAMAGE ANALYSIS ---
    # "Whenever the agent takes damage, it reflects off of the raw video from 3 seconds ago"
    # UPDATED: "reduce trama memory down to 4 seconds but allow overlapping"
    if state_manager.prev_health is not None:
        if current_health < state_manager.prev_health - 0.001:
            # Damage Taken!
            reward -= 1.0 # Immediate penalty
            
            # Reflect on last 4 seconds
            reflection_window = 4.0
            
            for act, ts, past_obs in state_manager.action_history:
                if now - ts <= reflection_window:
                    # Penalize Passivity
                    if act == 0: # Idle
                        reward -= 0.2 # Don't just stand there!
                    elif act in [1, 2, 3, 4]: # Walking
                        reward -= 0.1 # Walking is better than idle, but maybe dash?
                    
                    # Penalize Unsafe Attacks (Trading)
                    if act in [11, 12, 13, 14, 19, 15]:
                        reward -= 0.1 # You attacked but got hit. Be careful.
                    
                    # Innovation: Penalize "Blindness"
                    # If we got hit and weren't looking at the enemy
                    p_edx, p_edy = past_obs[4], past_obs[5]
                    p_dist = np.sqrt(p_edx*p_edx + p_edy*p_edy)
                    if p_dist > 0.5:
                        reward -= 0.2 # You got hit because you weren't looking!

    # Mode Usage (G): Reward if health is low (Immediate is fine for utility)
    if action_idx == 16: # "g"
        # "Dont reward pressing G unless health was gained... only if health is between 70% and 30%"
        if 0.3 <= current_health <= 0.7:
             # Check if health increased
             if state_manager.prev_health is not None and current_health > state_manager.prev_health + 0.01:
                 reward += 2.0
             else:
                 # Penalize using it if no health gained (cooldown or full hp or interrupted)
                 reward -= 0.1 
        else:
            # Outside range
            reward -= 0.5 

    # Lost Target / Visibility Penalty
    if len(obs) > 9:
        time_since_seen = obs[9]
        if time_since_seen > 0.01:
            reward -= 0.1 
        if time_since_seen > 0.05:
            reward -= 0.5
            
    # Rapid Action Switching Penalty
    if len(state_manager.action_history) > 0:
        last_action_idx = state_manager.action_history[-1][0]
        if action_idx != last_action_idx:
            reward -= 0.05

    # Constant penalty
    reward -= 0.01       
    state_manager.update(obs, action_idx)
    
    return reward
