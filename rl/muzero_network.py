import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class VisionEncoder(nn.Module):
    def __init__(self, latent_dim=512, input_channels=12): # 4 frames * 3 RGB
        super(VisionEncoder, self).__init__()
        
        # Use MobileNetV2 as base
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
        except:
            weights = None # Fallback
            
        self.mobilenet = models.mobilenet_v2(weights=weights)
        
        # Modify first layer to accept Stacked Frames (12 channels)
        # Standard MobileNetV2 first layer is Conv2d(3, 32, stride=2, padding=1)
        original_first = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_first.out_channels,
            kernel_size=original_first.kernel_size,
            stride=original_first.stride,
            padding=original_first.padding,
            bias=False
        )
        
        # Remove classifier
        self.mobilenet.classifier = nn.Identity()
        
        # MobileNetV2 last channel is 1280. 
        # We need to project to latent_dim (512).
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh() # Normalized latent space often helps dynamics
        )

    def forward(self, x):
        # x: [Batch, C, H, W]
        features = self.mobilenet.features(x)
        latent = self.projection(features)
        return latent

class DynamicsModel(nn.Module):
    def __init__(self, latent_dim=512, action_dim=26):
        super(DynamicsModel, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.Tanh() # Predict next latent
        )
        
    def forward(self, latent, action_one_hot):
        x = torch.cat([latent, action_one_hot], dim=1)
        next_latent = self.net(x)
        return next_latent

class MuZeroNetwork(nn.Module):
    def __init__(self, action_dim, latent_dim=512, lstm_hidden=512):
        super(MuZeroNetwork, self).__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.lstm_hidden = lstm_hidden
        
        # 1. Vision Encoder
        self.encoder = VisionEncoder(latent_dim=latent_dim)
        
        # 2. Dynamics Model
        self.dynamics = DynamicsModel(latent_dim=latent_dim, action_dim=action_dim)
        
        # 3. Temporal Backbone (LSTM)
        # Input: Latent (512) -> Hidden (512)
        self.lstm = nn.LSTM(latent_dim, lstm_hidden, num_layers=2, batch_first=True)
        
        # 4. Heads (From LSTM Hidden State)
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Scalar value
        )
        
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

    def forward(self, x, hidden_state=None):
        """
        Main Forward Pass (Inference/Training Policy)
        x: [Batch, Seq, C, H, W] or [Batch, C, H, W]
        """
        # Handle Sequence Dimension
        if x.dim() == 5: # [B, Seq, C, H, W]
            B, Seq, C, H, W = x.shape
            x = x.view(B * Seq, C, H, W)
            latents = self.encoder(x)
            latents = latents.view(B, Seq, -1)
        else:
            latents = self.encoder(x)
            latents = latents.unsqueeze(1) # [B, 1, Latent]
            
        # LSTM
        lstm_out, new_hidden = self.lstm(latents, hidden_state)
        
        # Heads
        # Flatten for heads
        B, Seq, _ = lstm_out.shape
        flat_out = lstm_out.reshape(B * Seq, -1)
        
        policy_logits = self.policy_head(flat_out)
        values = self.value_head(flat_out)
        
        # Reshape back if needed, or keep flat
        if Seq > 1:
            policy_logits = policy_logits.view(B, Seq, -1)
            values = values.view(B, Seq, 1)
            
        return policy_logits, values, new_hidden, latents

