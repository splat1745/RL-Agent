"""
Network V2 - Redesigned Architecture following best practices:

Key Changes:
1. RGB frames (not grayscale) - Full: 160x160x3, Crop: 128x128x3
2. Single-channel flow magnitude (not 2-channel)
3. Proper normalization: frames + z-score vector obs
4. Semantic dynamics prediction (not raw 151-d)
5. Probabilistic predictions (Gaussian + BCE)
6. Action masking BEFORE softmax
7. MobileNetV3-Small backbone (not Large)
8. LSTM hidden: 512
9. HP-loss prediction head (3 frames)
10. Sequence predictor head (5 actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math


class RunningMeanStd:
    """
    Running mean and standard deviation for online normalization.
    Uses Welford's algorithm for numerical stability.
    """
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        
    def update(self, x):
        """Update running stats with a batch of observations."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
        
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
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
        """Normalize input using running mean/std."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
    
    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}
    
    def load_state_dict(self, state):
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


class ImageNormalizer(nn.Module):
    """
    Channel-wise normalization for RGB frames.
    Uses ImageNet mean/std by default.
    """
    def __init__(self):
        super().__init__()
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x):
        """
        x: [B, 3, H, W] in range [0, 1]
        Returns: normalized tensor
        """
        return (x - self.mean) / self.std


class GaussianHead(nn.Module):
    """
    Probabilistic prediction head that outputs mean and log-variance.
    Used for position/velocity predictions.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mean_net = nn.Linear(in_features, out_features)
        self.logvar_net = nn.Linear(in_features, out_features)
        
        # Initialize logvar to small values for stable training
        nn.init.constant_(self.logvar_net.bias, -2.0)
        nn.init.zeros_(self.logvar_net.weight)
        
    def forward(self, x):
        """
        Returns: (mean, logvar) each [B, out_features]
        """
        mean = self.mean_net(x)
        logvar = self.logvar_net(x)
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mean, logvar
    
    def nll_loss(self, mean, logvar, target):
        """
        Negative log-likelihood loss for Gaussian.
        """
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + (target - mean).pow(2) / var)
        return nll.mean()


class DynamicsHead(nn.Module):
    """
    Semantic dynamics prediction head.
    Predicts meaningful future states, not raw 151-d vector.
    
    Outputs:
        - pos_x, pos_y (Gaussian): enemy position
        - vel_x, vel_y (Gaussian): enemy velocity  
        - damage_prob (BCE): probability of taking damage
        - latent (64-d): compact representation for other predictions
    """
    def __init__(self, hidden_dim=512, latent_dim=64):
        super().__init__()
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Position prediction (Gaussian)
        self.pos_head = GaussianHead(256, 2)  # x, y
        
        # Velocity prediction (Gaussian)
        self.vel_head = GaussianHead(256, 2)  # vx, vy
        
        # Damage probability (BCE)
        self.damage_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Latent prediction (for regularization)
        self.latent_head = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        """
        x: [B, hidden_dim]
        Returns dict with predictions
        """
        shared = self.shared(x)
        
        pos_mean, pos_logvar = self.pos_head(shared)
        vel_mean, vel_logvar = self.vel_head(shared)
        damage_logit = self.damage_head(shared)
        latent = self.latent_head(shared)
        
        return {
            'pos_mean': pos_mean,
            'pos_logvar': pos_logvar,
            'vel_mean': vel_mean,
            'vel_logvar': vel_logvar,
            'damage_logit': damage_logit,
            'latent': latent
        }
    
    def compute_loss(self, preds, targets):
        """
        Compute dynamics loss.
        
        targets: dict with:
            - 'next_pos': [B, 2] actual next enemy position
            - 'next_vel': [B, 2] actual next enemy velocity
            - 'damage_occurred': [B, 1] binary damage flag
            - 'next_latent': [B, 64] (optional, for MSE regularization)
        """
        # Position NLL
        pos_nll = self.pos_head.nll_loss(
            preds['pos_mean'], preds['pos_logvar'], targets['next_pos']
        )
        
        # Velocity NLL
        vel_nll = self.vel_head.nll_loss(
            preds['vel_mean'], preds['vel_logvar'], targets['next_vel']
        )
        
        # Damage BCE
        damage_bce = F.binary_cross_entropy_with_logits(
            preds['damage_logit'], targets['damage_occurred']
        )
        
        # Optional latent MSE
        latent_mse = 0.0
        if 'next_latent' in targets:
            latent_mse = F.mse_loss(preds['latent'], targets['next_latent'])
        
        # Combined loss
        total = pos_nll + vel_nll + damage_bce + 0.5 * latent_mse
        
        return {
            'total': total,
            'pos_nll': pos_nll,
            'vel_nll': vel_nll,
            'damage_bce': damage_bce,
            'latent_mse': latent_mse
        }


class HPPredictionHead(nn.Module):
    """
    Predicts probability of HP loss in next N frames.
    Uses BCE loss per frame.
    """
    def __init__(self, hidden_dim=512, horizon=3):
        super().__init__()
        self.horizon = horizon
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, horizon)
        )
        
    def forward(self, x):
        """Returns [B, horizon] logits for HP loss per frame."""
        return self.net(x)
    
    def compute_loss(self, logits, targets):
        """
        logits: [B, horizon]
        targets: [B, horizon] binary (1 if HP dropped in that frame)
        """
        return F.binary_cross_entropy_with_logits(logits, targets)


class SequencePredictorHead(nn.Module):
    """
    Predicts next N actions (for behavior cloning / sequence learning).
    Uses cross-entropy per step.
    """
    def __init__(self, hidden_dim=512, action_dim=26, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * action_dim)
        )
        
    def forward(self, x):
        """Returns [B, seq_len, action_dim] logits."""
        out = self.net(x)
        return out.view(-1, self.seq_len, self.action_dim)
    
    def compute_loss(self, logits, targets):
        """
        logits: [B, seq_len, action_dim]
        targets: [B, seq_len] action indices
        """
        B, S, A = logits.shape
        logits_flat = logits.view(B * S, A)
        targets_flat = targets.view(B * S)
        return F.cross_entropy(logits_flat, targets_flat)


class PolicyNetworkV2(nn.Module):
    """
    Complete policy network with all improvements.
    
    Architecture:
    - MobileNetV3-Small backbone → 256-d per frame
    - Flow encoder (single-channel magnitude)
    - Vector encoder with normalization
    - Fusion → LSTM (512 hidden)
    - Multi-head outputs with action masking
    """
    def __init__(
        self,
        action_dim=26,
        vector_dim=151,
        lstm_hidden=512,
        freeze_backbone=True
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.vector_dim = vector_dim
        self.lstm_hidden = lstm_hidden
        
        # --- Image Normalization ---
        self.img_norm = ImageNormalizer()
        
        # --- 1. Full Frame Encoder (MobileNetV3-Small) ---
        # Input: [B, 3, 160, 160] RGB
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except AttributeError:
            weights = 'DEFAULT'
            
        backbone = models.mobilenet_v3_small(weights=weights)
        self.full_encoder = backbone.features
        
        # Freeze early layers
        if freeze_backbone:
            for i, layer in enumerate(self.full_encoder):
                if i < 6:  # Freeze first 6 blocks
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.full_pool = nn.AdaptiveAvgPool2d(1)
        # MobileNetV3-Small output: 576 → project to 256
        self.full_proj = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU()
        )
        self.full_feature_dim = 256
        
        # --- 2. Crop Encoder (MobileNetV3-Small, shared weights) ---
        # Input: [B, 3, 128, 128] RGB
        # Reuse backbone features
        self.crop_proj = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU()
        )
        self.crop_feature_dim = 256
        
        # --- 3. Flow Encoder (Single-channel magnitude) ---
        # Input: [B, 1, 160, 160]
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.flow_feature_dim = 64
        
        # --- 4. Vector Encoder ---
        # Expects normalized input (z-score)
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.vector_feature_dim = 128
        
        # --- 5. Fusion ---
        self.fusion_dim = (
            self.full_feature_dim + 
            self.crop_feature_dim + 
            self.flow_feature_dim + 
            self.vector_feature_dim
        )  # 256 + 256 + 64 + 128 = 704
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # --- 6. LSTM (2 layers, 512 hidden) ---
        self.lstm = nn.LSTM(512, lstm_hidden, num_layers=2, batch_first=True)
        
        # --- 7. Policy Head (with action masking) ---
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # --- 8. Value Head ---
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # --- 9. Dynamics Head (Semantic Predictions) ---
        self.dynamics_head = DynamicsHead(lstm_hidden, latent_dim=64)
        
        # --- 10. HP Prediction Head (3 frames) ---
        self.hp_head = HPPredictionHead(lstm_hidden, horizon=3)
        
        # --- 11. Sequence Predictor Head (5 actions) ---
        self.seq_head = SequencePredictorHead(lstm_hidden, action_dim, seq_len=5)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize non-pretrained layers."""
        for module in [
            self.full_proj, self.crop_proj, self.flow_encoder,
            self.vector_encoder, self.fusion_layer, self.policy_head,
            self.value_head
        ]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # LSTM init
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.full_encoder.parameters():
            param.requires_grad = False
        print("Backbone frozen.")
        
    def unfreeze_last_layers(self, num_layers=3):
        """Unfreeze last N layers of backbone for fine-tuning."""
        total = len(list(self.full_encoder))
        for i, layer in enumerate(self.full_encoder):
            if i >= total - num_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"Unfroze last {num_layers} backbone layers.")
    
    def encode_frame(self, frame):
        """
        Encode a single RGB frame.
        frame: [B, 3, H, W] in range [0, 1]
        Returns: [B, 576] features
        """
        x = self.img_norm(frame)
        x = self.full_encoder(x)
        x = self.full_pool(x)
        x = x.flatten(1)
        return x
    
    def forward(
        self,
        full_frames,
        crop_frames,
        flow,
        vector_obs,
        hidden_state=None,
        seq_len=1,
        action_mask=None,
        return_aux=False
    ):
        """
        Forward pass.
        
        Args:
            full_frames: [B*S, 3, 160, 160] RGB in [0, 1]
            crop_frames: [B*S, 3, 128, 128] RGB in [0, 1]
            flow: [B*S, 1, 160, 160] magnitude in [0, 1]
            vector_obs: [B*S, vector_dim] normalized (z-score)
            hidden_state: LSTM hidden state
            seq_len: sequence length (for reshaping)
            action_mask: [B*S, action_dim] bool mask (True = allowed)
            return_aux: whether to return auxiliary predictions
            
        Returns:
            action_probs: [B*S, action_dim]
            value: [B*S, 1]
            hidden_state: new LSTM hidden
            aux_outputs: dict (if return_aux=True)
        """
        # 1. Encode full frames
        x_full = self.encode_frame(full_frames)  # [B*S, 576]
        x_full = self.full_proj(x_full)  # [B*S, 256]
        
        # 2. Encode crop (reuse backbone)
        x_crop = self.encode_frame(crop_frames)  # [B*S, 576]
        x_crop = self.crop_proj(x_crop)  # [B*S, 256]
        
        # 3. Encode flow
        x_flow = self.flow_encoder(flow)  # [B*S, 64]
        
        # 4. Encode vector
        x_vec = self.vector_encoder(vector_obs)  # [B*S, 128]
        
        # 5. Fusion
        fused = torch.cat([x_full, x_crop, x_flow, x_vec], dim=1)  # [B*S, 704]
        fused = self.fusion_layer(fused)  # [B*S, 512]
        
        # 6. LSTM
        batch_size = fused.size(0) // seq_len
        fused_seq = fused.view(batch_size, seq_len, -1)  # [B, S, 512]
        lstm_out, new_hidden = self.lstm(fused_seq, hidden_state)  # [B, S, 512]
        lstm_flat = lstm_out.reshape(-1, self.lstm_hidden)  # [B*S, 512]
        
        # 7. Policy (with masking BEFORE softmax)
        policy_logits = self.policy_head(lstm_flat)  # [B*S, action_dim]
        
        if action_mask is not None:
            # Set masked logits to -inf
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)
        
        action_probs = F.softmax(policy_logits, dim=-1)
        
        # 8. Value
        value = self.value_head(lstm_flat)  # [B*S, 1]
        
        if return_aux:
            # Auxiliary predictions
            dynamics_pred = self.dynamics_head(lstm_flat)
            hp_pred = self.hp_head(lstm_flat)
            seq_pred = self.seq_head(lstm_flat)
            
            aux_outputs = {
                'dynamics': dynamics_pred,
                'hp_logits': hp_pred,
                'seq_logits': seq_pred,
                'policy_logits': policy_logits  # Raw logits for entropy
            }
            return action_probs, value, new_hidden, aux_outputs
        
        return action_probs, value, new_hidden
    
    def get_action(self, full_frame, crop_frame, flow, vector_obs, hidden_state=None, action_mask=None):
        """
        Inference: get action for single observation.
        
        Args:
            full_frame: [3, 160, 160] or [1, 3, 160, 160]
            crop_frame: [3, 128, 128] or [1, 3, 128, 128]
            flow: [1, 160, 160] or [1, 1, 160, 160]
            vector_obs: [vector_dim] or [1, vector_dim]
            hidden_state: LSTM state
            action_mask: [action_dim] or [1, action_dim] bool
            
        Returns:
            action: int
            log_prob: float
            value: float
            hidden_state: new state
        """
        # Add batch dim if needed
        if full_frame.dim() == 3:
            full_frame = full_frame.unsqueeze(0)
        if crop_frame.dim() == 3:
            crop_frame = crop_frame.unsqueeze(0)
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)
        if vector_obs.dim() == 1:
            vector_obs = vector_obs.unsqueeze(0)
        if action_mask is not None and action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
            
        with torch.no_grad():
            action_probs, value, hidden_state = self.forward(
                full_frame, crop_frame, flow, vector_obs,
                hidden_state=hidden_state,
                seq_len=1,
                action_mask=action_mask,
                return_aux=False
            )
            
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item(), hidden_state


class PolicyNetworkV2Loss:
    """
    Combined loss function with proper weighting.
    
    Total = L_policy + c_v*L_value + c_dy*L_dynamics + c_hp*L_hp + c_seq*L_seq - c_ent*Entropy
    """
    def __init__(
        self,
        c_value=0.5,
        c_dynamics=0.2,
        c_hp=0.2,
        c_seq=0.15,
        c_entropy=0.01,
        clip_ratio=0.2
    ):
        self.c_value = c_value
        self.c_dynamics = c_dynamics
        self.c_hp = c_hp
        self.c_seq = c_seq
        self.c_entropy = c_entropy
        self.clip_ratio = clip_ratio
        
    def compute(
        self,
        network,
        old_log_probs,
        actions,
        advantages,
        returns,
        aux_outputs,
        dynamics_targets,
        hp_targets,
        seq_targets
    ):
        """
        Compute total loss.
        
        Args:
            network: PolicyNetworkV2
            old_log_probs: [B] from old policy
            actions: [B] action indices
            advantages: [B] GAE advantages
            returns: [B] discounted returns
            aux_outputs: dict from forward(..., return_aux=True)
            dynamics_targets: dict for dynamics head
            hp_targets: [B, 3] binary HP loss targets
            seq_targets: [B, 5] action sequence targets
        """
        # Policy loss (PPO clipped)
        probs = aux_outputs['policy_logits']  # [B, A]
        dist = torch.distributions.Categorical(logits=probs)
        log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_pred = aux_outputs.get('value', None)
        if value_pred is not None:
            value_loss = F.mse_loss(value_pred.squeeze(), returns)
        else:
            value_loss = torch.tensor(0.0)
        
        # Dynamics loss
        dynamics_loss_dict = network.dynamics_head.compute_loss(
            aux_outputs['dynamics'], dynamics_targets
        )
        dynamics_loss = dynamics_loss_dict['total']
        
        # HP loss
        hp_loss = network.hp_head.compute_loss(aux_outputs['hp_logits'], hp_targets)
        
        # Sequence loss
        seq_loss = network.seq_head.compute_loss(aux_outputs['seq_logits'], seq_targets)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total
        total_loss = (
            policy_loss +
            self.c_value * value_loss +
            self.c_dynamics * dynamics_loss +
            self.c_hp * hp_loss +
            self.c_seq * seq_loss -
            self.c_entropy * entropy
        )
        
        return {
            'total': total_loss,
            'policy': policy_loss,
            'value': value_loss,
            'dynamics': dynamics_loss,
            'hp': hp_loss,
            'seq': seq_loss,
            'entropy': entropy,
            'dynamics_breakdown': dynamics_loss_dict
        }


# --- Utility: Build action mask from cooldowns ---
def build_action_mask(cooldowns, skill_to_action_map, action_dim=26):
    """
    Build action mask from cooldown vector.
    
    Args:
        cooldowns: [B, num_skills] or [num_skills] cooldown times (0 = available)
        skill_to_action_map: dict {skill_idx: action_idx}
        action_dim: total number of actions
        
    Returns:
        mask: [B, action_dim] or [action_dim] bool (True = allowed)
    """
    if isinstance(cooldowns, np.ndarray):
        cooldowns = torch.from_numpy(cooldowns)
        
    if cooldowns.dim() == 1:
        cooldowns = cooldowns.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
        
    B = cooldowns.shape[0]
    mask = torch.ones(B, action_dim, dtype=torch.bool, device=cooldowns.device)
    
    for skill_idx, action_idx in skill_to_action_map.items():
        # If cooldown > 0, disable action
        unavailable = cooldowns[:, skill_idx] > 0.5
        mask[unavailable, action_idx] = False
        
    if squeeze:
        mask = mask.squeeze(0)
        
    return mask


# Default skill to action mapping
DEFAULT_SKILL_TO_ACTION = {
    0: 14,  # Skill 1
    1: 15,  # Skill 2
    2: 16,  # Skill 3
    3: 18,  # Skill 4
}
