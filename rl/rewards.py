import numpy as np
import time

class RewardCalculator:
    """
    Simplified reward calculator for vision-only RL agent.
    
    Design Principles:
    - Immediate feedback (no delayed rewards)
    - No rewards during damage states (prevent exploitation)
    - Action cooldowns to prevent move spamming mid-animation
    - Simple, clear signals
    """
    
    # Attack action indices
    ATTACK_ACTIONS = {
        12,  # m1 (left click)
        13,  # m2 (right click)
    }
    
    # Skill actions (longer cooldown)
    SKILL_ACTIONS = {
        14,  # 1
        15,  # 2
        16,  # 3
        18,  # 4
        19,  # r_2
        20,  # g
        21,  # f
    }
    
    # Dash actions
    DASH_ACTIONS = {
        7,   # dash_forward
        8,   # dash_back
        9,   # dash_left
        10,  # dash_right
    }
    
    # Movement actions (no cooldown)
    MOVEMENT_ACTIONS = {
        1, 2, 3, 4,  # w, a, s, d
        17,  # space (jump)
    }
    
    # Action cooldowns in seconds (approximate animation durations)
    ACTION_COOLDOWNS = {
        # Attacks
        12: 0.4,  # m1
        13: 0.5,  # m2
        # Skills (longer animations)
        14: 0.8,  # 1
        15: 0.8,  # 2
        16: 0.8,  # 3
        18: 0.8,  # 4
        19: 1.0,  # r_2 (special)
        20: 0.5,  # g
        21: 0.5,  # f
        # Dashes
        7: 0.3,   # dash_forward
        8: 0.3,   # dash_back
        9: 0.3,   # dash_left
        10: 0.3,  # dash_right
        # Jump
        17: 0.4,  # space
    }
    
    def __init__(self):
        # HP Tracking
        self.last_hp = 1.0
        
        # Damage cooldown - prevent rewarding attacks while being hit
        self.last_damage_time = 0
        self.DAMAGE_COOLDOWN = 0.5  # 500ms after taking damage = "stunned" state
        
        # Track if ANY damage was taken recently
        self.any_recent_damage = False
        self.last_damage_check_time = 0
        self.RECENT_DAMAGE_WINDOW = 1.0  # 1 second window
        
        # Action cooldown tracking
        self.last_action_id = None
        self.last_action_time = 0
        self.current_action_cooldown = 0
        
        # Action spam tracking
        self.action_switch_count = 0
        self.action_switch_window_start = 0
        self.ACTION_SWITCH_WINDOW = 2.0  # 2 second window
        self.MAX_SWITCHES_BEFORE_PENALTY = 10  # Too many switches in window
        
        # Idle tracking
        self.consecutive_idle = 0
        
        # Cooldown detector (set externally if available)
        self.cooldown_detector = None
        
        # Mapping from skill action IDs to cooldown move indices (0-3)
        # Maps game action IDs to cooldown slot indices
        self.SKILL_TO_COOLDOWN_SLOT = {
            14: 0,  # 1 -> Move 1
            15: 1,  # 2 -> Move 2
            16: 2,  # 3 -> Move 3
            18: 3,  # 4 -> Move 4
        }
        
        # Stats
        self.total_damage_taken = 0
        self.total_attacks = 0
        self.total_successful_attacks = 0
        self.action_spam_penalties = 0
        
    def calculate(self, perception, action_id):
        """
        Calculate immediate reward.
        """
        reward = 0.0
        current_time = time.time()
        
        # --- 1. HP Delta (Primary Signal) ---
        current_hp = getattr(perception, 'filtered_health', 1.0)
        hp_delta = current_hp - self.last_hp
        
        if hp_delta < 0:
            # Took ANY damage
            damage_penalty = hp_delta * 5.0  # -0.1 HP -> -0.5 reward
            reward += damage_penalty
            self.total_damage_taken += abs(hp_delta)
            
            # Mark damage event
            self.last_damage_time = current_time
            self.any_recent_damage = True
            self.last_damage_check_time = current_time
        
        # Reset recent damage flag after window expires
        if self.any_recent_damage and (current_time - self.last_damage_check_time > self.RECENT_DAMAGE_WINDOW):
            self.any_recent_damage = False
        
        # --- 2. Check if "stunned/being attacked" state ---
        is_being_attacked = self._is_being_attacked(current_time)
        
        # --- 3. Action Cooldown Check ---
        action_valid, cooldown_penalty = self._check_action_cooldown(action_id, current_time)
        reward += cooldown_penalty
        
        # --- 4. Action Spam Penalty ---
        spam_penalty = self._check_action_spam(action_id, current_time)
        reward += spam_penalty
        
        # --- 5. Attack Actions (only if valid and not being attacked) ---
        if action_id in self.ATTACK_ACTIONS:
            self.total_attacks += 1
            
            if action_valid and not is_being_attacked:
                if self._enemy_visible(perception):
                    reward += 0.05  # Small immediate attack reward
                    self.total_successful_attacks += 1
        
        # --- 6. Skill Actions (only if valid, not being attacked, and not on cooldown) ---
        if action_id in self.SKILL_ACTIONS:
            # Check if skill is on cooldown via cooldown detector
            skill_on_cooldown = False
            if self.cooldown_detector and action_id in self.SKILL_TO_COOLDOWN_SLOT:
                slot = self.SKILL_TO_COOLDOWN_SLOT[action_id]
                skill_on_cooldown = not self.cooldown_detector.is_move_available(slot)
            
            if skill_on_cooldown:
                # Penalty for using skill on cooldown
                reward -= 0.05
            elif action_valid and not is_being_attacked:
                if self._enemy_visible(perception):
                    reward += 0.03  # Reward for using skills at appropriate times
        
        # --- 7. Movement vs Idle ---
        if action_id in self.MOVEMENT_ACTIONS:
            self.consecutive_idle = 0
            if not is_being_attacked:
                reward += 0.001
        elif action_id == 0:  # Idle
            self.consecutive_idle += 1
            if self.consecutive_idle > 20:
                reward -= 0.01
        else:
            self.consecutive_idle = max(0, self.consecutive_idle - 1)
        
        # --- 8. Survival Bonus ---
        if not is_being_attacked:
            reward += 0.001
        
        # --- 9. Engagement Bonus ---
        if self._enemy_visible(perception) and not is_being_attacked:
            reward += 0.002
        
        # Update state
        self.last_hp = current_hp
        
        return reward
    
    def _is_being_attacked(self, current_time):
        """
        Check if agent is currently in a "being attacked" state.
        """
        # Recently took damage
        if current_time - self.last_damage_time < self.DAMAGE_COOLDOWN:
            return True
        
        # ANY damage in recent window
        if self.any_recent_damage:
            return True
        
        return False
    
    def _check_action_cooldown(self, action_id, current_time):
        """
        Check if action is valid given current cooldown.
        Returns (is_valid, penalty).
        """
        # Check if we're still in cooldown from previous action
        if self.last_action_id is not None:
            time_since_last = current_time - self.last_action_time
            
            if time_since_last < self.current_action_cooldown:
                # Still in cooldown - trying to switch mid-animation
                if action_id != self.last_action_id:
                    # Penalize trying to switch action mid-animation
                    return False, -0.02
                else:
                    # Same action held - that's fine
                    return True, 0.0
        
        # Action is valid, update tracking
        if action_id != self.last_action_id:
            self.last_action_id = action_id
            self.last_action_time = current_time
            self.current_action_cooldown = self.ACTION_COOLDOWNS.get(action_id, 0.0)
        
        return True, 0.0
    
    def _check_action_spam(self, action_id, current_time):
        """
        Penalize rapidly switching between different actions.
        """
        penalty = 0.0
        
        # Reset window if expired
        if current_time - self.action_switch_window_start > self.ACTION_SWITCH_WINDOW:
            self.action_switch_count = 0
            self.action_switch_window_start = current_time
        
        # Count action switches
        if action_id != self.last_action_id and action_id != 0:  # Don't count idle
            self.action_switch_count += 1
            
            if self.action_switch_count > self.MAX_SWITCHES_BEFORE_PENALTY:
                # Too many switches - spamming
                penalty = -0.03
                self.action_spam_penalties += 1
        
        return penalty
    
    def _enemy_visible(self, perception):
        """
        Check if any enemy is currently visible/detected.
        """
        detections = getattr(perception, 'last_det', {})
        
        if isinstance(detections, dict):
            enemies = detections.get('enemies', [])
            return len(enemies) > 0
        elif isinstance(detections, list):
            for det in detections:
                if det.get('cls', 0) != 1:
                    return True
        
        return False
    
    def reset(self):
        """Reset state for new episode."""
        self.last_hp = 1.0
        self.last_damage_time = 0
        self.any_recent_damage = False
        self.last_damage_check_time = 0
        self.last_action_id = None
        self.last_action_time = 0
        self.current_action_cooldown = 0
        self.action_switch_count = 0
        self.consecutive_idle = 0
    
    def get_stats(self):
        """Return tracking stats for logging."""
        return {
            'total_damage_taken': self.total_damage_taken,
            'total_attacks': self.total_attacks,
            'total_successful_attacks': self.total_successful_attacks,
            'attack_success_rate': self.total_successful_attacks / max(1, self.total_attacks),
            'spam_penalties': self.action_spam_penalties,
        }
