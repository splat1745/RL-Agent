"""
Feature Builder - Clean interface for building observations.

Handles:
1. RGB frame preprocessing with ImageNet normalization
2. Crop extraction around player (Kalman-smoothed)
3. Optical flow (single-channel magnitude)
4. Vector observation with z-score normalization
5. Kalman estimates fed as inputs (not overrides)

Usage:
    builder = FeatureBuilder()
    pixel_obs, vector_obs = builder.build(frame, det_dict, last_action, mouse_movement)
"""

import cv2
import numpy as np
from collections import deque


class KalmanTracker:
    """
    Kalman filter for smooth position/velocity estimation.
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[2:, 2:] *= 5.0  # Higher noise for velocity
        
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        
        # Initial state
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False
        self.confidence = 0.0
        self.frames_since_update = 0
        
    def predict(self):
        """Predict next state."""
        pred = self.kf.predict()
        self.frames_since_update += 1
        
        # Decay confidence if no measurement
        self.confidence = max(0.0, self.confidence - 0.1)
        
        return {
            'x': float(pred[0]),
            'y': float(pred[1]),
            'vx': float(pred[2]),
            'vy': float(pred[3]),
            'confidence': self.confidence
        }
        
    def update(self, x, y):
        """Update with measurement."""
        if not self.initialized:
            self.kf.statePost[0] = x
            self.kf.statePost[1] = y
            self.initialized = True
            
        meas = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(meas)
        
        self.frames_since_update = 0
        self.confidence = 1.0
        
        state = self.kf.statePost
        return {
            'x': float(state[0]),
            'y': float(state[1]),
            'vx': float(state[2]),
            'vy': float(state[3]),
            'confidence': self.confidence
        }
        
    def get_state(self):
        """Get current state estimate."""
        state = self.kf.statePost
        return {
            'x': float(state[0]),
            'y': float(state[1]),
            'vx': float(state[2]),
            'vy': float(state[3]),
            'confidence': self.confidence
        }


class RunningStats:
    """
    Running mean/std for online normalization using Welford's algorithm.
    """
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        
    def update(self, x):
        """Update with single observation or batch."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    def normalize(self, x):
        """Z-score normalization."""
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)
        
    def save(self):
        return {'mean': self.mean.copy(), 'var': self.var.copy(), 'count': self.count}
        
    def load(self, state):
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


class FeatureBuilder:
    """
    Builds pixel and vector observations from raw frame + detections.
    """
    def __init__(
        self,
        full_size=160,
        crop_size=128,
        input_size=640,
        vector_dim=151
    ):
        self.full_size = full_size
        self.crop_size = crop_size
        self.input_size = input_size
        self.vector_dim = vector_dim
        
        # Kalman trackers
        self.player_tracker = KalmanTracker()
        self.enemy_tracker = KalmanTracker()
        
        # Running stats for vector normalization
        self.vector_stats = RunningStats(shape=(vector_dim,))
        
        # Flow computation state
        self.last_gray = None
        
        # Memory state
        self.last_enemy_pos = (0.5, 0.5)  # Normalized
        self.last_enemy_time = 0.0
        
        # Action history for temporal features
        self.action_history = deque(maxlen=10)
        
    def build(self, frame, det_dict, last_action=None, mouse_movement=(0, 0)):
        """
        Build observation from frame and detections.
        
        Args:
            frame: BGR image [H, W, 3]
            det_dict: Detection results with 'player', 'enemies', etc.
            last_action: Previous action index
            mouse_movement: (dx, dy) mouse delta
            
        Returns:
            pixel_obs: dict with 'full', 'crop', 'flow'
            vector_obs: np.array [vector_dim] (normalized)
        """
        import time
        
        if frame is None:
            return None, None
            
        h, w = frame.shape[:2]
        
        # 1. Process player detection
        player = det_dict.get("player")
        player_state = self._update_player(player)
        
        # 2. Process enemy detection
        enemies = det_dict.get("enemies", [])
        enemy_state = self._update_enemy(enemies, player_state)
        
        # 3. Build pixel observations
        pixel_obs = self._build_pixel_obs(frame, player_state)
        
        # 4. Build vector observations
        vector_obs = self._build_vector_obs(
            player_state, enemy_state, det_dict,
            last_action, mouse_movement
        )
        
        # 5. Update action history
        if last_action is not None:
            self.action_history.append(last_action)
            
        return pixel_obs, vector_obs
        
    def _update_player(self, player):
        """Update player tracker and return state."""
        self.player_tracker.predict()
        
        if player is not None:
            cx, cy = player[:2]
            state = self.player_tracker.update(cx, cy)
        else:
            state = self.player_tracker.get_state()
            
        return state
        
    def _update_enemy(self, enemies, player_state):
        """Update enemy tracker (nearest to player)."""
        import time
        
        self.enemy_tracker.predict()
        
        if enemies:
            # Find nearest enemy to player
            px, py = player_state['x'], player_state['y']
            best_dist = float('inf')
            best_enemy = None
            
            for e in enemies:
                if len(e) >= 2:
                    ex, ey = e[:2]
                    dist = (ex - px)**2 + (ey - py)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_enemy = (ex, ey)
                        
            if best_enemy:
                state = self.enemy_tracker.update(*best_enemy)
                
                # Update memory
                self.last_enemy_pos = (
                    best_enemy[0] / self.input_size,
                    best_enemy[1] / self.input_size
                )
                self.last_enemy_time = time.time()
                
                return state
                
        # Return prediction if no detection
        state = self.enemy_tracker.get_state()
        state['time_since_seen'] = time.time() - self.last_enemy_time
        return state
        
    def _build_pixel_obs(self, frame, player_state):
        """Build pixel observations (RGB, flow)."""
        h, w = frame.shape[:2]
        
        # 1. Full frame RGB [3, 160, 160]
        full_img = cv2.resize(frame, (self.full_size, self.full_size))
        full_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        full_norm = full_rgb.astype(np.float32) / 255.0
        full_tensor = np.transpose(full_norm, (2, 0, 1))
        
        # 2. Crop around player [3, 128, 128]
        crop_tensor = self._extract_crop(frame, player_state)
        
        # 3. Optical flow magnitude [1, 160, 160]
        flow_tensor = self._compute_flow(full_img)
        
        return {
            'full': full_tensor,
            'crop': crop_tensor,
            'flow': flow_tensor
        }
        
    def _extract_crop(self, frame, player_state):
        """Extract crop centered on player."""
        h, w = frame.shape[:2]
        
        # Get player position in frame coords
        scale_x = w / self.input_size
        scale_y = h / self.input_size
        
        pcx = int(player_state['x'] * scale_x)
        pcy = int(player_state['y'] * scale_y)
        
        half = self.crop_size // 2
        
        # Offset upward to see in front
        offset_y = int(self.crop_size * 0.25)
        center_y = max(half, pcy - offset_y)
        
        x1 = max(0, pcx - half)
        y1 = max(0, center_y - half)
        x2 = min(w, pcx + half)
        y2 = min(h, center_y + half)
        
        # Handle edge cases
        if x1 >= x2 or y1 >= y2:
            x1 = max(0, w // 2 - half)
            y1 = max(0, h // 2 - half)
            x2 = min(w, x1 + self.crop_size)
            y2 = min(h, y1 + self.crop_size)
            
        crop = frame[y1:y2, x1:x2]
        
        # Pad if needed
        if crop.shape[0] != self.crop_size or crop.shape[1] != self.crop_size:
            padded = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            ph = min(crop.shape[0], self.crop_size)
            pw = min(crop.shape[1], self.crop_size)
            padded[:ph, :pw] = crop[:ph, :pw]
            crop = padded
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_norm = crop_rgb.astype(np.float32) / 255.0
        return np.transpose(crop_norm, (2, 0, 1))
        
    def _compute_flow(self, full_img):
        """Compute optical flow magnitude."""
        gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        
        if self.last_gray is None:
            flow_mag = np.zeros((self.full_size, self.full_size), dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self.last_gray, gray,
                np.zeros((self.full_size, self.full_size, 2), dtype=np.float32),
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Normalize to [0, 1]
            max_mag = flow_mag.max()
            if max_mag > 1e-6:
                flow_mag = flow_mag / max_mag
            else:
                flow_mag = flow_mag * 0
                
        self.last_gray = gray
        return flow_mag[np.newaxis, ...]
        
    def _build_vector_obs(self, player_state, enemy_state, det_dict, last_action, mouse_movement):
        """
        Build normalized vector observation.
        
        Includes:
        - Player position/velocity (from Kalman)
        - Enemy position/velocity/confidence (from Kalman)
        - Obstacle info
        - Goal info
        - UI state (cooldowns, bars)
        - Kalman confidence values
        - Mouse movement
        - Last action one-hot
        """
        import time
        
        obs = np.zeros(self.vector_dim, dtype=np.float32)
        idx = 0
        
        # Player state (4)
        obs[idx:idx+4] = [
            player_state['x'] / self.input_size,
            player_state['y'] / self.input_size,
            player_state['vx'] / self.input_size * 10,  # Scale velocity
            player_state['vy'] / self.input_size * 10
        ]
        idx += 4
        
        # Enemy state (6) - includes Kalman estimates
        screen_cx = self.input_size / 2
        screen_cy = self.input_size / 2
        
        enemy_dx = (enemy_state['x'] - screen_cx) / self.input_size
        enemy_dy = (enemy_state['y'] - screen_cy) / self.input_size
        
        obs[idx:idx+6] = [
            enemy_dx,
            enemy_dy,
            enemy_state.get('vx', 0) / self.input_size * 10,
            enemy_state.get('vy', 0) / self.input_size * 10,
            enemy_state.get('confidence', 0),
            min(enemy_state.get('time_since_seen', 5.0) / 5.0, 1.0)
        ]
        idx += 6
        
        # Obstacle (2)
        obstacles = det_dict.get("obstacles", [])
        if obstacles:
            # Nearest obstacle
            px, py = player_state['x'], player_state['y']
            best_dist = float('inf')
            best_obs = (1.0, 1.0)
            
            for o in obstacles:
                if len(o) >= 2:
                    ox, oy = o[:2]
                    dx = (ox - px) / self.input_size
                    dy = (oy - py) / self.input_size
                    dist = dx**2 + dy**2
                    if dist < best_dist:
                        best_dist = dist
                        best_obs = (dx, dy)
                        
            obs[idx:idx+2] = best_obs
        else:
            obs[idx:idx+2] = [1.0, 1.0]
        idx += 2
        
        # Goal (1)
        goal = det_dict.get("goal")
        if goal:
            gx, gy = goal[:2]
            px, py = player_state['x'], player_state['y']
            dx = (gx - px) / self.input_size
            dy = (gy - py) / self.input_size
            obs[idx] = np.sqrt(dx**2 + dy**2)
        else:
            obs[idx] = 1.0
        idx += 1
        
        # UI state (health, bars, cooldowns) - up to 10 values
        obs[idx] = det_dict.get("health", 1.0)
        idx += 1
        
        # Cooldowns (4 skills)
        cooldowns = det_dict.get("cooldowns", {})
        for i in range(4):
            obs[idx + i] = cooldowns.get(f"skill_{i}", 0.0)
        idx += 4
        
        # Bars (evasive, special, mode)
        obs[idx] = det_dict.get("evasive_bar", 1.0)
        obs[idx + 1] = det_dict.get("special_bar", 1.0)
        obs[idx + 2] = det_dict.get("mode_bar", 0.0)
        idx += 3
        
        # Mouse movement (2)
        obs[idx:idx+2] = [
            mouse_movement[0] / 100.0,  # Normalize
            mouse_movement[1] / 100.0
        ]
        idx += 2
        
        # Player tracker confidence (1)
        obs[idx] = player_state.get('confidence', 0.0)
        idx += 1
        
        # Last action one-hot (26) - or just the index if space limited
        if last_action is not None and idx + 26 <= self.vector_dim:
            obs[idx + last_action] = 1.0
            idx += 26
            
        # Fill remaining with zeros
        # (Already initialized to zeros)
        
        # Update running stats and normalize
        self.vector_stats.update(obs.reshape(1, -1))
        normalized = self.vector_stats.normalize(obs)
        
        return normalized
        
    def get_kalman_states(self):
        """Get current Kalman filter states for logging."""
        return {
            'player': self.player_tracker.get_state(),
            'enemy': self.enemy_tracker.get_state()
        }
        
    def save_stats(self):
        """Save normalization stats."""
        return self.vector_stats.save()
        
    def load_stats(self, state):
        """Load normalization stats."""
        self.vector_stats.load(state)
