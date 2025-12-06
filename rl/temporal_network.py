"""
Temporal MuZero Network with ConvLSTM + Transformer + Multihead Outputs

Architecture:
- Input: 30 RGB frames (90 channels at 128x128) + 30 past actions + 4 cooldowns
- MobileNetV3-Small backbone (early layers frozen)
- ConvLSTM for spatial-temporal features
- Transformer encoder (2L, 4H) for sequence understanding
- Multihead outputs:
  1. Policy distribution (cooldown-masked)
  2. Value estimation
  3. HP-loss prediction (next 30 frames)
  4. Action sequence predictor (next 5 actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import mobilenet_v3_small


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell for spatial-temporal processing.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 
            4 * hidden_dim,  # i, f, g, o gates
            kernel_size=kernel_size,
            padding=padding
        )
        
    def forward(self, x, hidden):
        """
        x: [B, C, H, W]
        hidden: (h, c) each [B, hidden_dim, H, W]
        """
        h, c = hidden
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM for processing frame sequences.
    """
    def __init__(self, input_dim, hidden_dims, kernel_size=3):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        
        self.cells = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            in_dim = input_dim if i == 0 else hidden_dims[i-1]
            self.cells.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size))
    
    def forward(self, x, hidden_states=None):
        """
        x: [B, T, C, H, W] - sequence of frames
        Returns: [B, hidden_dim, H, W] - final hidden state
        """
        B, T, C, H, W = x.shape
        device = x.device
        
        if hidden_states is None:
            hidden_states = [
                cell.init_hidden(B, H, W, device) 
                for cell in self.cells
            ]
        
        # Process sequence
        for t in range(T):
            x_t = x[:, t]  # [B, C, H, W]
            
            for layer_idx, cell in enumerate(self.cells):
                h, c = hidden_states[layer_idx]
                h_next, c_next = cell(x_t, (h, c))
                hidden_states[layer_idx] = (h_next, c_next)
                x_t = h_next  # Input to next layer
        
        # Return final hidden state from last layer
        return hidden_states[-1][0], hidden_states


class TemporalTransformer(nn.Module):
    """
    Small transformer for sequence understanding.
    Encodes temporal features + action embeddings.
    """
    def __init__(self, d_model=384, nhead=4, num_layers=2, action_dim=26):
        super().__init__()
        
        self.d_model = d_model
        self.action_embed = nn.Embedding(action_dim, d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 30, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, frame_features, past_actions):
        """
        frame_features: [B, T, d_model] - from ConvLSTM (pooled)
        past_actions: [B, T] - action indices
        Returns: [B, T, d_model]
        """
        B, T, D = frame_features.shape
        
        # Embed actions and add to features
        action_emb = self.action_embed(past_actions)  # [B, T, d_model]
        x = frame_features + action_emb + self.pos_embed[:, :T]
        
        # Transformer encoding
        x = self.transformer(x)
        
        return x


class MotionEncoder(nn.Module):
    """
    Encodes motion features from frame differences.
    Helps model understand what's moving and why damage occurred.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # Process frame differences
        self.motion_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )
        
    def forward(self, frames):
        """
        frames: [B, T, 3, H, W]
        Returns: [B, T-1, out_channels, 8, 8] motion features
        """
        B, T, C, H, W = frames.shape
        
        # Compute frame differences
        diffs = frames[:, 1:] - frames[:, :-1]  # [B, T-1, 3, H, W]
        
        # Process each diff
        diffs_flat = diffs.reshape(B * (T - 1), C, H, W)
        motion_feats = self.motion_conv(diffs_flat)  # [B*(T-1), 64, 8, 8]
        
        _, C_out, H_out, W_out = motion_feats.shape
        motion_feats = motion_feats.view(B, T - 1, C_out, H_out, W_out)
        
        return motion_feats


class DamageAttributionHead(nn.Module):
    """
    Learns which frames/actions caused damage.
    Uses attention to look backward from damage events.
    """
    def __init__(self, hidden_dim=384, seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        
        # Attention to find causal frames
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Predict which frames contributed to damage
        self.frame_attribution = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Per-frame damage contribution score
        )
        
        # Predict which action types are dangerous
        self.action_danger = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 26)  # Danger score per action type
        )
        
    def forward(self, sequence_features, damage_occurred):
        """
        sequence_features: [B, T, hidden_dim]
        damage_occurred: [B] bool - did damage happen in this sequence?
        
        Returns:
            frame_scores: [B, T] - contribution of each frame to damage
            action_danger: [B, 26] - danger level of each action type
        """
        B, T, D = sequence_features.shape
        
        # Self-attention to find causal relationships
        attended, attn_weights = self.causal_attention(
            sequence_features, sequence_features, sequence_features
        )
        
        # Frame attribution scores
        frame_scores = self.frame_attribution(attended).squeeze(-1)  # [B, T]
        frame_scores = torch.sigmoid(frame_scores)
        
        # Action danger (from final attended features)
        action_danger = self.action_danger(attended[:, -1])  # [B, 26]
        action_danger = torch.sigmoid(action_danger)
        
        return frame_scores, action_danger, attn_weights


class TemporalMuZeroNetwork(nn.Module):
    """
    Full temporal network with multihead outputs.
    """
    def __init__(self, action_dim=26, hidden_dim=384, seq_len=30, freeze_backbone=True):
        super().__init__()
        
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Backbone: MobileNetV3-Small (modified for 3ch per frame)
        backbone = mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.features = backbone.features
        
        # Freeze early layers if specified
        if freeze_backbone:
            for i, layer in enumerate(self.features):
                if i < 6:  # Freeze first 6 blocks
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Get feature dimension
        self.backbone_out_dim = 576  # MobileNetV3-Small output channels
        
        # Spatial pooling to reduce memory before ConvLSTM
        self.spatial_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # ConvLSTM for temporal processing
        self.conv_lstm = ConvLSTM(
            input_dim=self.backbone_out_dim,
            hidden_dims=[256, hidden_dim],
            kernel_size=3
        )
        
        # Global pool for transformer input
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Transformer for sequence understanding
        self.transformer = TemporalTransformer(
            d_model=hidden_dim,
            nhead=4,
            num_layers=2,
            action_dim=action_dim
        )
        
        # Cooldown embedding (4 skills)
        self.cooldown_embed = nn.Linear(4, hidden_dim)
        
        # --- Multihead Outputs ---
        
        # Head 1: Policy Distribution
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Head 2: Value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Head 3: HP Loss Prediction (next 30 frames)
        self.hp_loss_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 30)  # P(damage at t+1...t+30)
        )
        
        # Head 4: Action Sequence Predictor (next 5 actions)
        self.sequence_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 5 * action_dim)  # 5 future actions
        )
        
        # Feature projection for transformer input (backbone_out_dim -> hidden_dim)
        self.feature_proj = nn.Linear(self.backbone_out_dim, hidden_dim)
        
        # --- NEW: Motion and Causal Understanding ---
        
        # Motion encoder for understanding what's moving
        self.motion_encoder = MotionEncoder(in_channels=3, out_channels=64)
        self.motion_proj = nn.Linear(64, hidden_dim)  # Project motion to hidden dim
        
        # Damage attribution head - learns WHY damage occurred
        self.damage_attribution = DamageAttributionHead(hidden_dim=hidden_dim, seq_len=seq_len)
        
    def extract_frame_features(self, frames):
        """
        Extract features from a batch of frames.
        frames: [B, T, 3, H, W]
        Returns: [B, T, C, H', W']
        """
        B, T, C, H, W = frames.shape
        
        # Reshape for batch processing
        x = frames.view(B * T, C, H, W)
        
        # MobileNet features
        x = self.features(x)  # [B*T, 576, H/32, W/32]
        
        # Spatial pooling
        x = self.spatial_pool(x)  # [B*T, 576, 8, 8]
        
        # Reshape back
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, T, C_out, H_out, W_out)
        
        return x
    
    def forward(self, frames, past_actions, cooldowns, hidden_states=None):
        """
        Forward pass.
        
        Args:
            frames: [B, T, 3, H, W] - 30 RGB frames
            past_actions: [B, T] - past action indices
            cooldowns: [B, 4] - current cooldown state (0=available, 1=on cooldown)
            hidden_states: Optional ConvLSTM hidden states
            
        Returns:
            policy: [B, action_dim] - action distribution (pre-softmax logits)
            value: [B, 1] - value estimate
            hp_pred: [B, 30] - HP loss prediction (sigmoid applied)
            seq_pred: [B, 5, action_dim] - sequence prediction (logits)
            hidden_states: Updated ConvLSTM hidden states
            damage_info: dict with frame_scores, action_danger, attn_weights
        """
        B, T = frames.shape[:2]
        
        # 1. Extract frame features
        frame_features = self.extract_frame_features(frames)  # [B, T, C, 8, 8]
        
        # 2. Extract motion features (what's moving between frames)
        motion_features = self.motion_encoder(frames)  # [B, T-1, 64, 8, 8]
        # Pool and project motion
        motion_pooled = motion_features.mean(dim=(-1, -2))  # [B, T-1, 64]
        motion_proj = self.motion_proj(motion_pooled)  # [B, T-1, hidden_dim]
        # Pad to match T length
        motion_padded = F.pad(motion_proj, (0, 0, 1, 0))  # [B, T, hidden_dim]
        
        # 3. ConvLSTM temporal processing
        conv_out, hidden_states = self.conv_lstm(frame_features, hidden_states)  # [B, hidden_dim, 8, 8]
        
        # 4. Prepare transformer input (global pooled features per timestep)
        transformer_input = self._get_per_timestep_features(frame_features)  # [B, T, hidden_dim]
        
        # 5. Add motion context to transformer input
        transformer_input = transformer_input + motion_padded
        
        # 6. Transformer with action context
        transformer_out = self.transformer(transformer_input, past_actions)  # [B, T, hidden_dim]
        
        # 7. Use last timestep output
        final_features = transformer_out[:, -1]  # [B, hidden_dim]
        
        # 8. Add cooldown context
        cooldown_features = self.cooldown_embed(cooldowns.float())  # [B, hidden_dim]
        final_features = final_features + cooldown_features
        
        # 9. Multihead outputs
        policy_logits = self.policy_head(final_features)  # [B, action_dim]
        value = self.value_head(final_features)  # [B, 1]
        hp_pred = torch.sigmoid(self.hp_loss_head(final_features))  # [B, 30]
        seq_logits = self.sequence_head(final_features).reshape(B, 5, self.action_dim)  # [B, 5, action_dim]
        
        # 10. Damage attribution - understand WHY damage happens
        damage_occurred = torch.zeros(B, dtype=torch.bool, device=frames.device)  # Will be set by caller
        frame_scores, action_danger, attn_weights = self.damage_attribution(transformer_out, damage_occurred)
        
        damage_info = {
            'frame_scores': frame_scores,
            'action_danger': action_danger,
            'attn_weights': attn_weights
        }
        
        return policy_logits, value, hp_pred, seq_logits, hidden_states, damage_info
    
    def _get_per_timestep_features(self, frame_features):
        """
        Get per-timestep features for transformer.
        Simple approach: global pool each timestep.
        """
        B, T, C, H, W = frame_features.shape
        x = frame_features.view(B * T, C, H, W)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B*T, C]
        x = x.view(B, T, C)
        
        # Project to hidden_dim
        x = self.feature_proj(x)
        
        return x
    
    def apply_cooldown_mask(self, policy_logits, cooldowns, skill_to_action_map):
        """
        Apply cooldown masking to policy logits.
        
        Args:
            policy_logits: [B, action_dim]
            cooldowns: [B, 4] - 0=available, 1=on cooldown
            skill_to_action_map: dict mapping skill index (0-3) to action indices
            
        Returns:
            masked_logits: [B, action_dim]
        """
        masked = policy_logits.clone()
        
        for skill_idx, action_idx in skill_to_action_map.items():
            # Set logit to -inf for unavailable skills
            mask = cooldowns[:, skill_idx] > 0.5  # [B]
            masked[mask, action_idx] = float('-inf')
        
        return masked
    
    def get_action(self, frames, past_actions, cooldowns, skill_to_action_map=None, hidden_states=None):
        """
        Get action with cooldown masking for inference.
        
        Returns:
            action: int - selected action index
            policy: [action_dim] - masked probability distribution
            value: float - value estimate
            hp_pred: [30] - HP loss predictions
            seq_pred: [5] - predicted action sequence (argmax)
            hidden_states: Updated hidden states
            damage_info: dict with action_danger scores
        """
        with torch.no_grad():
            policy_logits, value, hp_pred, seq_logits, hidden_states, damage_info = self.forward(
                frames, past_actions, cooldowns, hidden_states
            )
            
            # Apply cooldown mask if provided
            if skill_to_action_map is not None:
                policy_logits = self.apply_cooldown_mask(policy_logits, cooldowns, skill_to_action_map)
            
            # Also penalize actions that lead to damage based on learned danger
            action_danger = damage_info['action_danger'].squeeze(0)  # [26]
            # Subtract danger from logits (make dangerous actions less likely)
            policy_logits = policy_logits - action_danger * 2.0
            
            # Softmax for probability
            policy = F.softmax(policy_logits, dim=-1)
            
            # Sample action
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            
            return (
                action.item(),
                policy.squeeze(0).cpu().numpy(),
                value.item(),
                hp_pred.squeeze(0).cpu().numpy(),
                seq_logits.squeeze(0).argmax(dim=-1).cpu().numpy(),
                hidden_states,
                {
                    'action_danger': action_danger.cpu().numpy(),
                    'frame_scores': damage_info['frame_scores'].squeeze(0).cpu().numpy()
                }
            )


# Skill to action mapping (default)
DEFAULT_SKILL_TO_ACTION = {
    0: 14,  # Skill 1 -> action 14
    1: 15,  # Skill 2 -> action 15
    2: 16,  # Skill 3 -> action 16
    3: 18,  # Skill 4 -> action 18
}
