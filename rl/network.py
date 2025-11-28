import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class TwoStreamNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, lstm_hidden=128):
        super(TwoStreamNetwork, self).__init__()
        
        # --- 1. Full Frame Encoder (MobileNetV3 Small) ---
        # Input: [Batch, 12, 160, 160] (4 frames * 3 channels)
        # We modify the first layer to accept 12 channels instead of 3
        self.full_encoder = models.mobilenet_v3_small(weights=None)
        
        # Modify first conv layer for 12 channels
        original_first_layer = self.full_encoder.features[0][0]
        self.full_encoder.features[0][0] = nn.Conv2d(
            in_channels=12,
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False
        )
        
        # Remove classifier, keep feature extractor
        # MobileNetV3 Small output features: 576 channels at 5x5 (for 160x160 input) -> GlobalPool -> 576
        self.full_encoder.classifier = nn.Identity()
        self.full_feature_dim = 576 

        # --- 2. Fine Crop Encoder (Small CNN) ---
        # Input: [Batch, 12, 128, 128]
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.crop_feature_dim = 128

        # --- 3. Optical Flow Encoder (Small CNN) ---
        # Input: [Batch, 2, 160, 160] (Magnitude of flow for last 2 frames? Or 2 channels x 1 frame?)
        # User said: "flow = compute_optical_flow(last2 frames) # shape [2, H, W]"
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.flow_feature_dim = 32
        
        # --- 4. Fusion & Temporal ---
        self.fusion_dim = self.full_feature_dim + self.crop_feature_dim + self.flow_feature_dim
        
        # Fusion Layer (Linear Projection)
        # 1024 -> 256 -> 128 (LSTM)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        
        # --- 5. Heads ---
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
        
    def forward(self, full_frames, crop_frames, flow, hidden_state=None):
        """
        full_frames: [Batch, Seq, C, H, W] or [Batch, C_stack, H, W]
        If input is [Batch, C_stack, H, W], we treat it as Seq=1
        """
        # Assuming inputs are [Batch, Channels, H, W] for a single step
        # MobileNet expects [Batch, Channels, H, W]
        
        # 1. Encode Full Frames
        # MobileNetV3 features returns [Batch, 576, 1, 1] roughly before pooling?
        # The .features part returns spatial map. We need to pool it.
        # Actually, we replaced classifier with Identity, but MobileNetV3 structure is:
        # features -> avgpool -> classifier.
        # So we need to call features then avgpool.
        
        x_full = self.full_encoder.features(full_frames)
        x_full = self.full_encoder.avgpool(x_full)
        x_full = torch.flatten(x_full, 1) # [Batch, 576]
        
        # 2. Encode Crop
        x_crop = self.crop_encoder(crop_frames) # [Batch, 128]
        
        # 3. Encode Flow
        x_flow = self.flow_encoder(flow) # [Batch, 32]
        
        # 4. Fusion
        fusion = torch.cat([x_full, x_crop, x_flow], dim=1) # [Batch, Fusion_Dim]
        
        # Project
        fusion_proj = self.fusion_layer(fusion)
        
        # 5. LSTM
        # LSTM expects [Batch, Seq, Features]
        # We add Sequence dimension
        fusion_seq = fusion_proj.unsqueeze(1) 
        
        lstm_out, new_hidden = self.lstm(fusion_seq, hidden_state)
        
        # Take last output for heads
        last_out = lstm_out[:, -1, :]
        
        # 6. Heads
        action_probs = self.actor(last_out)
        value = self.critic(last_out)
        
        return action_probs, value, new_hidden

