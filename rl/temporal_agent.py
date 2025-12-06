"""
Temporal MuZero Agent with multi-stage training.

Training Stages:
- Stage 1: Pretrain (supervised) - Use existing offline weights
- Stage 2: RL Warmup - Freeze backbone, train policy/value/HP heads
- Stage 3: Hybrid - Full multi-objective training
- Stage 4: Safety Override - danger forecast > 0.4 triggers evasive action

All training runs through main.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import time

from rl.temporal_network import TemporalMuZeroNetwork, DEFAULT_SKILL_TO_ACTION


class TemporalReplayBuffer:
    """
    Replay buffer for 30-frame temporal sequences.
    Stores full sequences for training HP prediction and sequence prediction.
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, frames, actions, cooldowns, rewards, hp_deltas, done):
        """
        Store a transition sequence.
        
        Args:
            frames: [T, 3, H, W] numpy array - 30 frames
            actions: [T] - action indices
            cooldowns: [4] - cooldown state at end
            rewards: [T] - rewards per step
            hp_deltas: [T] - HP changes per step (negative = damage)
            done: bool
        """
        self.buffer.append({
            'frames': frames,
            'actions': actions,
            'cooldowns': cooldowns,
            'rewards': rewards,
            'hp_deltas': hp_deltas,
            'done': done
        })
        
    def sample(self, batch_size):
        """Sample a batch of sequences."""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'frames': torch.FloatTensor(np.stack([b['frames'] for b in batch])),
            'actions': torch.LongTensor(np.stack([b['actions'] for b in batch])),
            'cooldowns': torch.FloatTensor(np.stack([b['cooldowns'] for b in batch])),
            'rewards': torch.FloatTensor(np.stack([b['rewards'] for b in batch])),
            'hp_deltas': torch.FloatTensor(np.stack([b['hp_deltas'] for b in batch])),
            'done': torch.BoolTensor([b['done'] for b in batch])
        }
    
    def __len__(self):
        return len(self.buffer)


class TemporalAgent:
    """
    Temporal MuZero Agent with multi-stage training.
    """
    
    # Loss weight configurations per stage
    STAGE_CONFIGS = {
        2: {  # RL Warmup
            'policy': 0.4,
            'value': 0.4,
            'hp_loss': 0.2,
            'sequence': 0.0,
            'freeze_backbone': True
        },
        3: {  # Hybrid
            'policy': 0.35,
            'value': 0.25,
            'hp_loss': 0.20,
            'sequence': 0.20,
            'freeze_backbone': False
        }
    }
    
    def __init__(self, action_dim=26, hidden_dim=384, seq_len=30, device='cuda', stage=2):
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.device = device
        self.stage = stage
        
        # Network
        freeze = self.STAGE_CONFIGS.get(stage, {}).get('freeze_backbone', True)
        self.network = TemporalMuZeroNetwork(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            freeze_backbone=freeze
        ).to(device)
        
        # Optimizer with warmup + cosine annealing learning rate
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        self.base_lr = 3e-4  # Higher initial LR
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.base_lr, weight_decay=1e-5)
        
        # Learning rate scheduler: warmup 1000 steps, then cosine decay
        self.warmup_steps = 1000
        self.max_steps = 100000
        
        # Gradient accumulation for larger effective batches
        self.gradient_accumulation_steps = 4
        self.accumulated_steps = 0
        
        # Replay buffer
        self.buffer = TemporalReplayBuffer(capacity=10000)
        
        # Rolling buffers for online play
        self.frame_buffer = deque(maxlen=seq_len)
        self.action_buffer = deque(maxlen=seq_len)
        self.reward_buffer = deque(maxlen=seq_len)
        self.hp_delta_buffer = deque(maxlen=seq_len)
        
        # Hidden states for recurrent processing
        self.hidden_states = None
        
        # Skill to action mapping
        self.skill_to_action = DEFAULT_SKILL_TO_ACTION
        
        # Training step counter
        self.train_steps = 0
        
        # Danger threshold for safety override
        self.danger_threshold = 0.4
    def get_lr(self):
        """Calculate learning rate with warmup + cosine annealing."""
        if self.train_steps < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.train_steps / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.train_steps - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def update_lr(self):
        """Update optimizer learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    def preprocess_frame(self, frame):
        """
        Preprocess frame for network.
        frame: [H, W, 3] numpy BGR
        Returns: [3, 128, 128] numpy float32
        """
        import cv2
        resized = cv2.resize(frame, (128, 128))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if resized.shape[2] == 3 else resized
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))  # [3, 128, 128]
        return transposed
    
    def observe(self, frame, reward=0.0, hp_delta=0.0):
        """
        Add observation to buffers.
        Called each step during online play.
        """
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
        self.reward_buffer.append(reward)
        self.hp_delta_buffer.append(hp_delta)
        
    def is_ready(self):
        """Check if we have enough frames for inference."""
        return len(self.frame_buffer) >= self.seq_len
    
    def select_action(self, cooldowns, epsilon=0.1):
        """
        Select action using current state.
        
        Args:
            cooldowns: [4] numpy array - 0=available, 1=on cooldown
            epsilon: exploration rate
            
        Returns:
            action_idx: int
            action_info: dict with policy, value, hp_pred, seq_pred, action_danger
        """
        if not self.is_ready():
            return np.random.randint(self.action_dim), {}
        
        # Prepare tensors
        frames = torch.FloatTensor(np.stack(list(self.frame_buffer))).unsqueeze(0).to(self.device)  # [1, T, 3, H, W]
        actions = torch.LongTensor(list(self.action_buffer)).unsqueeze(0).to(self.device)  # [1, T]
        cooldowns_t = torch.FloatTensor(cooldowns).unsqueeze(0).to(self.device)  # [1, 4]
        
        # Pad actions if needed
        if actions.shape[1] < self.seq_len:
            pad = torch.zeros(1, self.seq_len - actions.shape[1], dtype=torch.long, device=self.device)
            actions = torch.cat([pad, actions], dim=1)
        
        # Get network output (7 outputs now including damage_info)
        action_idx, policy, value, hp_pred, seq_pred, self.hidden_states, damage_info = self.network.get_action(
            frames, actions, cooldowns_t, self.skill_to_action, self.hidden_states
        )
        
        # Safety override: if danger predicted in next 3 frames
        if hp_pred[:3].max() > self.danger_threshold:
            # Use sequence predictor's first action
            action_idx = int(seq_pred[0])
        
        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(self.action_dim)
        
        # Record action
        self.action_buffer.append(action_idx)
        
        return action_idx, {
            'policy': policy,
            'value': value,
            'hp_pred': hp_pred,
            'seq_pred': seq_pred,
            'danger': hp_pred[:3].max(),
            'action_danger': damage_info.get('action_danger', None)
        }
    
    def store_sequence(self, cooldowns, done=False):
        """
        Store current sequence in replay buffer.
        Called periodically during training.
        """
        if len(self.frame_buffer) < self.seq_len:
            return
        
        self.buffer.push(
            frames=np.stack(list(self.frame_buffer)),
            actions=np.array(list(self.action_buffer)),
            cooldowns=cooldowns,
            rewards=np.array(list(self.reward_buffer)),
            hp_deltas=np.array(list(self.hp_delta_buffer)),
            done=done
        )
    
    def train_step(self, batch_size=8):
        """
        Perform one training step.
        
        Returns:
            dict with loss values
        """
        if len(self.buffer) < batch_size:
            return None
        
        batch = self.buffer.sample(batch_size)
        
        # Move to device
        frames = batch['frames'].to(self.device)
        actions = batch['actions'].to(self.device)
        cooldowns = batch['cooldowns'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        hp_deltas = batch['hp_deltas'].to(self.device)
        
        # Update learning rate
        current_lr = self.update_lr()
        
        # Forward pass (6 outputs with damage_info)
        policy_logits, value, hp_pred, seq_logits, _, damage_info = self.network(
            frames, actions, cooldowns
        )
        
        # --- Compute losses ---
        config = self.STAGE_CONFIGS.get(self.stage, self.STAGE_CONFIGS[3])
        losses = {}
        
        # 1. Policy loss (maximize expected reward)
        # Use advantage estimation
        returns = self._compute_returns(rewards, gamma=0.99)
        advantages = returns[:, -1] - value.squeeze(-1)
        
        policy_probs = F.softmax(policy_logits, dim=-1)
        action_probs = policy_probs.gather(1, actions[:, -1:]).squeeze(-1)
        policy_loss = -(torch.log(action_probs + 1e-8) * advantages.detach()).mean()
        losses['policy'] = policy_loss * config['policy']
        
        # 2. Value loss
        value_loss = F.mse_loss(value.squeeze(-1), returns[:, -1])
        losses['value'] = value_loss * config['value']
        
        # 3. HP loss prediction (binary classification)
        # Target: did we take damage in each of the next 30 frames?
        hp_targets = (hp_deltas < 0).float()  # [B, 30]
        hp_loss = F.binary_cross_entropy(hp_pred, hp_targets)
        losses['hp_loss'] = hp_loss * config['hp_loss']
        
        # 4. Sequence prediction loss (predict next 5 actions)
        if config['sequence'] > 0:
            # Target: actual next 5 actions (shift by 1)
            seq_targets = actions[:, -5:]  # Last 5 actions as proxy
            seq_loss = F.cross_entropy(
                seq_logits.reshape(-1, self.action_dim),
                seq_targets.reshape(-1)
            )
            losses['sequence'] = seq_loss * config['sequence']
        else:
            losses['sequence'] = torch.tensor(0.0)
        
        # 5. Damage attribution loss - learn which frames/actions caused damage
        frame_scores = damage_info['frame_scores']  # [B, T]
        action_danger = damage_info['action_danger']  # [B, 26]
        
        # Target: frames with HP loss should have high attribution
        frame_damage_target = (hp_deltas < 0).float()  # [B, T]
        frame_attr_loss = F.binary_cross_entropy(frame_scores, frame_damage_target)
        losses['frame_attr'] = frame_attr_loss * 0.1
        
        # Action danger: actions taken when damage occurred should be penalized
        actions_one_hot = F.one_hot(actions[:, -1], num_classes=26).float()  # [B, 26]
        damage_mask = (hp_deltas[:, -1] < 0).float().unsqueeze(-1)  # [B, 1]
        action_danger_target = actions_one_hot * damage_mask  # [B, 26]
        action_danger_loss = F.binary_cross_entropy(action_danger, action_danger_target)
        losses['action_danger'] = action_danger_loss * 0.1
        
        # Total loss
        total_loss = sum(losses.values())
        
        # Gradient accumulation
        total_loss = total_loss / self.gradient_accumulation_steps
        total_loss.backward()
        self.accumulated_steps += 1
        
        if self.accumulated_steps >= self.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulated_steps = 0
        
        self.train_steps += 1
        
        result = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        result['lr'] = current_lr
        return result
    
    def _compute_returns(self, rewards, gamma=0.99):
        """Compute discounted returns."""
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        returns[:, -1] = rewards[:, -1]
        
        for t in range(T - 2, -1, -1):
            returns[:, t] = rewards[:, t] + gamma * returns[:, t + 1]
        
        return returns
    
    def set_stage(self, stage):
        """Switch training stage."""
        self.stage = stage
        config = self.STAGE_CONFIGS.get(stage, self.STAGE_CONFIGS[3])
        
        # Update backbone freezing
        if config['freeze_backbone']:
            for i, layer in enumerate(self.network.features):
                if i < 6:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            for param in self.network.features.parameters():
                param.requires_grad = True
        
        # Rebuild optimizer with new trainable params
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
        
        print(f"Switched to Stage {stage}")
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stage': self.stage,
            'train_steps': self.train_steps
        }, path)
        
    def load(self, path):
        """Load model checkpoint. Handles architecture changes gracefully."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load network with strict=False to handle architecture changes
        missing, unexpected = self.network.load_state_dict(checkpoint['network_state_dict'], strict=False)
        if missing:
            print(f"Missing keys (will be randomly initialized): {missing}")
        if unexpected:
            print(f"Unexpected keys (ignored): {unexpected}")
        
        # Try to load optimizer, but don't fail if it doesn't match
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (ValueError, KeyError) as e:
            print(f"Optimizer state incompatible, reinitializing: {e}")
            trainable_params = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
        
        self.stage = checkpoint.get('stage', 2)
        self.train_steps = checkpoint.get('train_steps', 0)
        print(f"Loaded checkpoint from {path} (Stage {self.stage}, Step {self.train_steps})")
    
    def reset_episode(self):
        """Reset for new episode."""
        self.frame_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.hp_delta_buffer.clear()
        self.hidden_states = None
        
        # Fill action buffer with zeros
        for _ in range(self.seq_len):
            self.action_buffer.append(0)
