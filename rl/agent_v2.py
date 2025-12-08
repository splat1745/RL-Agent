"""
Agent V2 - Updated PPO Agent for PolicyNetworkV2

Key Changes:
1. Uses RGB frames (3 channels per frame, not RGBD)
2. Single-channel flow (magnitude)
3. Running normalization for vector observations
4. Action masking for cooldowns
5. Proper multi-objective loss computation
"""

import os
import torch
import torch.optim as optim
import numpy as np

from .network_v2 import (
    PolicyNetworkV2, 
    PolicyNetworkV2Loss, 
    RunningMeanStd,
    build_action_mask,
    DEFAULT_SKILL_TO_ACTION
)


class PPOAgentV2:
    """
    PPO Agent using the V2 network architecture.
    """
    def __init__(
        self,
        action_dim=26,
        vector_dim=151,
        lr_policy=3e-4,
        lr_encoder=1e-4,
        gamma=0.995,
        gae_lambda=0.95,
        eps_clip=0.2,
        k_epochs=4,
        weight_decay=1e-5,
        max_grad_norm=0.5,
        freeze_backbone=True
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.max_grad_norm = max_grad_norm
        self.action_dim = action_dim
        self.vector_dim = vector_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Network
        self.policy = PolicyNetworkV2(
            action_dim=action_dim,
            vector_dim=vector_dim,
            lstm_hidden=512,
            freeze_backbone=freeze_backbone
        ).to(self.device)
        
        # Old policy for PPO
        self.policy_old = PolicyNetworkV2(
            action_dim=action_dim,
            vector_dim=vector_dim,
            lstm_hidden=512,
            freeze_backbone=freeze_backbone
        ).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Optimizer with different LRs for different parts
        encoder_params = list(self.policy.full_encoder.parameters())
        other_params = [
            p for n, p in self.policy.named_parameters() 
            if 'full_encoder' not in n and p.requires_grad
        ]
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': lr_encoder},
            {'params': other_params, 'lr': lr_policy}
        ], weight_decay=weight_decay)
        
        # Loss function
        self.loss_fn = PolicyNetworkV2Loss(
            c_value=0.5,
            c_dynamics=0.2,
            c_hp=0.2,
            c_seq=0.15,
            c_entropy=0.01,
            clip_ratio=eps_clip
        )
        
        # Running normalization for vector observations
        self.vec_normalizer = RunningMeanStd(shape=(vector_dim,))
        
        # Hidden state management
        self.hidden_state = None
        self.train_hidden = None
        
        # Skill to action mapping for cooldown masking
        self.skill_to_action = DEFAULT_SKILL_TO_ACTION

    def reset_hidden(self):
        """Reset LSTM hidden state (e.g., at episode start)."""
        self.hidden_state = None
        self.train_hidden = None

    def normalize_vector(self, vector_obs, update_stats=True):
        """
        Normalize vector observation using running stats.
        
        Args:
            vector_obs: numpy array [vector_dim] or [B, vector_dim]
            update_stats: whether to update running mean/std
            
        Returns:
            normalized vector
        """
        if update_stats:
            if vector_obs.ndim == 1:
                self.vec_normalizer.update(vector_obs.reshape(1, -1))
            else:
                self.vec_normalizer.update(vector_obs)
        
        return self.vec_normalizer.normalize(vector_obs).astype(np.float32)

    def select_action(self, pixel_obs, vector_obs, cooldowns=None):
        """
        Select action for a single observation.
        
        Args:
            pixel_obs: dict with:
                'full': [3, 160, 160] RGB in [0, 1]
                'crop': [3, 128, 128] RGB in [0, 1]
                'flow': [1, 160, 160] magnitude in [0, 1]
            vector_obs: [vector_dim] raw features
            cooldowns: [num_skills] cooldown times (optional)
            
        Returns:
            action: int
            log_prob: float
            value: float
        """
        with torch.no_grad():
            # Normalize vector obs
            vec_norm = self.normalize_vector(vector_obs, update_stats=True)
            
            # Convert to tensors
            full = torch.FloatTensor(pixel_obs['full']).unsqueeze(0).to(self.device)
            crop = torch.FloatTensor(pixel_obs['crop']).unsqueeze(0).to(self.device)
            flow = torch.FloatTensor(pixel_obs['flow']).unsqueeze(0).to(self.device)
            vector = torch.FloatTensor(vec_norm).unsqueeze(0).to(self.device)
            
            # Build action mask from cooldowns
            action_mask = None
            if cooldowns is not None:
                action_mask = build_action_mask(
                    torch.FloatTensor(cooldowns).unsqueeze(0),
                    self.skill_to_action,
                    self.action_dim
                ).to(self.device)
            
            # Forward pass
            action_probs, value, self.hidden_state = self.policy_old(
                full, crop, flow, vector,
                hidden_state=self.hidden_state,
                seq_len=1,
                action_mask=action_mask,
                return_aux=False
            )
            
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: list of rewards
            values: list of value estimates
            dones: list of done flags
            next_value: value of next state
            
        Returns:
            advantages: numpy array
            returns: numpy array
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values[:-1], dtype=np.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def update(self, memory, dynamics_targets=None, hp_targets=None, seq_targets=None):
        """
        Update policy using collected experiences.
        
        Args:
            memory: Memory object with states, actions, rewards, etc.
            dynamics_targets: dict with 'next_pos', 'next_vel', 'damage_occurred'
            hp_targets: [N, 3] binary HP loss in next 3 frames
            seq_targets: [N, 5] next 5 actions
        """
        # Prepare batches
        full_batch = torch.stack([
            torch.FloatTensor(s[0]['full']) for s in memory.states
        ]).to(self.device)
        
        crop_batch = torch.stack([
            torch.FloatTensor(s[0]['crop']) for s in memory.states
        ]).to(self.device)
        
        flow_batch = torch.stack([
            torch.FloatTensor(s[0]['flow']) for s in memory.states
        ]).to(self.device)
        
        # Normalize vector observations
        vector_batch = torch.stack([
            torch.FloatTensor(self.normalize_vector(s[1], update_stats=False)) 
            for s in memory.states
        ]).to(self.device)
        
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(self.device)
        
        # Compute GAE
        with torch.no_grad():
            _, values, _ = self.policy_old(
                full_batch, crop_batch, flow_batch, vector_batch,
                seq_len=len(memory.states)
            )
            values = values.squeeze().cpu().numpy().tolist()
            
        advantages, returns = self.compute_gae(
            memory.rewards, values, memory.is_terminals, 0.0
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Prepare auxiliary targets
        if dynamics_targets is None:
            # Default dummy targets
            dynamics_targets = {
                'next_pos': torch.zeros(len(memory.states), 2).to(self.device),
                'next_vel': torch.zeros(len(memory.states), 2).to(self.device),
                'damage_occurred': torch.zeros(len(memory.states), 1).to(self.device)
            }
        else:
            dynamics_targets = {
                k: torch.FloatTensor(v).to(self.device) 
                for k, v in dynamics_targets.items()
            }
            
        if hp_targets is None:
            hp_targets = torch.zeros(len(memory.states), 3).to(self.device)
        else:
            hp_targets = torch.FloatTensor(hp_targets).to(self.device)
            
        if seq_targets is None:
            seq_targets = torch.zeros(len(memory.states), 5, dtype=torch.long).to(self.device)
        else:
            seq_targets = torch.LongTensor(seq_targets).to(self.device)
        
        # Initialize hidden state for training
        if self.train_hidden is None:
            h0 = torch.zeros(2, 1, 512).to(self.device)
            c0 = torch.zeros(2, 1, 512).to(self.device)
            self.train_hidden = (h0, c0)
        
        initial_hidden = (
            self.train_hidden[0].detach(),
            self.train_hidden[1].detach()
        )
        
        # Training epochs
        for epoch in range(self.k_epochs):
            # Forward pass with auxiliary outputs
            action_probs, value, current_hidden, aux_outputs = self.policy(
                full_batch, crop_batch, flow_batch, vector_batch,
                hidden_state=initial_hidden,
                seq_len=len(memory.states),
                return_aux=True
            )
            
            aux_outputs['value'] = value
            
            # Compute losses
            losses = self.loss_fn.compute(
                self.policy,
                old_logprobs,
                old_actions,
                advantages,
                returns,
                aux_outputs,
                dynamics_targets,
                hp_targets,
                seq_targets
            )
            
            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 
                self.max_grad_norm
            )
            
            self.optimizer.step()
        
        # Update hidden state for next batch
        self.train_hidden = (
            current_hidden[0].detach(),
            current_hidden[1].detach()
        )
        
        # Sync old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return {
            'total_loss': losses['total'].item(),
            'policy_loss': losses['policy'].item(),
            'value_loss': losses['value'].item(),
            'dynamics_loss': losses['dynamics'].item(),
            'hp_loss': losses['hp'].item(),
            'seq_loss': losses['seq'].item(),
            'entropy': losses['entropy'].item()
        }

    def save(self, filename):
        """Save model with atomic write."""
        tmp_filename = filename + ".tmp"
        
        state = {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'vec_normalizer': self.vec_normalizer.state_dict()
        }
        
        torch.save(state, tmp_filename)
        
        try:
            os.replace(tmp_filename, filename)
        except OSError:
            if os.path.exists(filename):
                os.remove(filename)
            os.rename(tmp_filename, filename)
    
    def load(self, filename):
        """Load model with compatibility handling."""
        # Fix for PyTorch 2.6+ weights_only=True default
        state = torch.load(filename, map_location=self.device, weights_only=False)
        
        # Load policy
        if 'policy' in state:
            self._load_compatible(self.policy, state['policy'])
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            # Old format: state dict directly
            self._load_compatible(self.policy, state)
            self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Load optimizer if available
        if 'optimizer' in state:
            try:
                self.optimizer.load_state_dict(state['optimizer'])
            except Exception as e:
                print(f"Could not load optimizer state: {e}")
        
        # Load normalizer if available
        if 'vec_normalizer' in state:
            self.vec_normalizer.load_state_dict(state['vec_normalizer'])
    
    def _load_compatible(self, model, state_dict):
        """Load state dict with compatibility for mismatched layers."""
        current_state = model.state_dict()
        
        filtered = {}
        skipped = []
        
        for k, v in state_dict.items():
            if k in current_state:
                if v.shape == current_state[k].shape:
                    filtered[k] = v
                else:
                    skipped.append(f"{k}: {v.shape} vs {current_state[k].shape}")
            else:
                skipped.append(f"{k}: not in model")
        
        if skipped:
            print(f"Skipped {len(skipped)} incompatible layers:")
            for s in skipped[:5]:
                print(f"  {s}")
            if len(skipped) > 5:
                print(f"  ... and {len(skipped) - 5} more")
        
        current_state.update(filtered)
        model.load_state_dict(current_state)
