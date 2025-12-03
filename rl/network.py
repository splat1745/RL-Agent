import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class TwoStreamNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=1024, lstm_hidden=768, vector_dim=151):
        super(TwoStreamNetwork, self).__init__()
        
        # --- 1. Full Frame Encoder (MobileNetV3 Large + Extra Layers) ---
        # Input: [Batch, 16, 160, 160] (4 frames * 4 channels RGBD)
        self.full_encoder = models.mobilenet_v3_large(weights=None)
        
        original_first_layer = self.full_encoder.features[0][0]
        self.full_encoder.features[0][0] = nn.Conv2d(
            in_channels=16, # Updated for RGBD stack
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False
        )
        
        self.full_encoder.classifier = nn.Identity()
        # MobileNetV3 Large output: 960. Add extra processing.
        self.full_extra = nn.Sequential(
            nn.Linear(960, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.full_feature_dim = 1024

        # --- 2. Fine Crop Encoder (Quadrupled Capacity) ---
        # Input: [Batch, 16, 128, 128] (4 frames * 4 channels RGBD)
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.crop_feature_dim = 512

        # --- 3. Optical Flow Encoder (Quadrupled Capacity) ---
        # Input: [Batch, 2, 160, 160]
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.flow_feature_dim = 128
        
        # --- 4. Vector Encoder (New) ---
        # Input: [Batch, vector_dim]
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.vector_feature_dim = 128
        
        # --- 5. Fusion & Temporal (Deep MLP + Stacked LSTM) ---
        self.fusion_dim = self.full_feature_dim + self.crop_feature_dim + self.flow_feature_dim + self.vector_feature_dim
        
        # Deep Fusion MLP
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        
        # Stacked LSTM (2 Layers, 768 Hidden)
        self.lstm = nn.LSTM(1024, 768, num_layers=2, batch_first=True)
        
        # --- 6. Multi-Branch Action Head with Combo Intention ---
        # "Add an action head with multi branching, temporal smoothing, and action chunking logic. Add a “combo intention embedding.”"
        
        # Combo Intention Embedding (Internal State)
        # We project the LSTM output to an "Intention" vector
        self.intention_net = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 384) # 384-dim Intention Embedding
        )
        
        # Multi-Branch Heads
        # Branch 1: Movement (W, A, S, D, Dashes, Turns)
        self.movement_head = nn.Sequential(
            nn.Linear(768 + 384, 512), # LSTM + Intention
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Branch 2: Combat (Attacks, Blocks, Combos)
        self.combat_head = nn.Sequential(
            nn.Linear(768 + 384, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Final Action Projection (Merging Branches)
        # We combine the branch outputs to produce the final logits
        self.final_action_layer = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # --- Auxiliary Heads (For Imitation Learning) ---
        # 1. Camera Motion Prediction (From Flow)
        # Predicts g_dx, g_dy (2 dims)
        self.aux_camera_head = nn.Sequential(
            nn.Linear(self.flow_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # 2. Enemy Position Prediction (From Full Frame)
        # Predicts enemy_dx, enemy_dy (2 dims)
        self.aux_enemy_head = nn.Sequential(
            nn.Linear(self.full_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        # 3. Forward Dynamics (Next State Prediction)
        # Predicts next vector_obs (151 dims) from LSTM state + Action Embedding
        # Note: We don't have action embedding here yet, so we predict from LSTM state (which implies action intent)
        self.aux_dynamics_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, vector_dim)
        )
        
        # Initialize Weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    param.data.fill_(0.0)
        
    def forward(self, full_frames, crop_frames, flow, vector_obs, hidden_state=None, seq_len=1, return_aux=False):
        """
        full_frames: [Batch*Seq, C, H, W]
        seq_len: Length of sequence (default 1 for inference)
        return_aux: If True, returns auxiliary predictions
        """
        # 1. Encode Full Frames
        x_full = self.full_encoder.features(full_frames)
        x_full = self.full_encoder.avgpool(x_full)
        x_full = torch.flatten(x_full, 1) # [Batch*Seq, 960]
        x_full = self.full_extra(x_full) # [Batch*Seq, 1024]
        
        # 2. Encode Crop
        x_crop = self.crop_encoder(crop_frames) # [Batch*Seq, 512]
        
        # 3. Encode Flow
        x_flow = self.flow_encoder(flow) # [Batch*Seq, 128]
        
        # 4. Encode Vector
        x_vector = self.vector_encoder(vector_obs) # [Batch*Seq, 128]
        
        # 5. Fusion
        fusion = torch.cat([x_full, x_crop, x_flow, x_vector], dim=1) # [Batch*Seq, Fusion_Dim]
        
        # Project
        fusion_proj = self.fusion_layer(fusion) # [Batch*Seq, 1024]
        
        # 6. LSTM
        # Reshape to [Batch, Seq, Features]
        batch_size = fusion_proj.size(0) // seq_len
        fusion_seq = fusion_proj.view(batch_size, seq_len, -1)
        
        lstm_out, new_hidden = self.lstm(fusion_seq, hidden_state) # [Batch, Seq, 768]
        
        # 7. Heads with Multi-Branching & Intention
        # Flatten back to [Batch*Seq, 768]
        lstm_out_flat = lstm_out.reshape(-1, 768)
        
        # Generate Intention
        intention = self.intention_net(lstm_out_flat) # [Batch*Seq, 128]
        
        # Concatenate LSTM output with Intention for branches
        branch_input = torch.cat([lstm_out_flat, intention], dim=1) # [Batch*Seq, 768+128]
        
        # Branch Processing
        move_feat = self.movement_head(branch_input) # [Batch*Seq, 128]
        combat_feat = self.combat_head(branch_input) # [Batch*Seq, 128]
        
        # Merge Branches
        merged_feat = torch.cat([move_feat, combat_feat], dim=1) # [Batch*Seq, 256]
        
        action_probs = self.final_action_layer(merged_feat)
        value = self.critic(lstm_out_flat)
        
        if return_aux:
            # Auxiliary Predictions
            pred_camera = self.aux_camera_head(x_flow) # [Batch*Seq, 2]
            pred_enemy = self.aux_enemy_head(x_full)   # [Batch*Seq, 2]
            pred_next_vec = self.aux_dynamics_head(lstm_out_flat) # [Batch*Seq, 151]
            
            return action_probs, value, new_hidden, intention, pred_camera, pred_enemy, pred_next_vec
        
        return action_probs, value, new_hidden, intention


