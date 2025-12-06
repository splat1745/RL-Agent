"""Quick test for temporal network shapes."""
import torch
from rl.temporal_network import TemporalMuZeroNetwork

print("Testing TemporalMuZeroNetwork...")
net = TemporalMuZeroNetwork()

# Create test inputs
frames = torch.randn(2, 30, 3, 128, 128)
actions = torch.randint(0, 26, (2, 30))
cooldowns = torch.zeros(2, 4)

# Forward pass
policy, value, hp_pred, seq_pred, hidden, damage_info = net(frames, actions, cooldowns)

print(f"Policy: {policy.shape}")
print(f"Value: {value.shape}")
print(f"HP Pred: {hp_pred.shape}")
print(f"Seq Pred: {seq_pred.shape}")
print(f"Frame Scores: {damage_info['frame_scores'].shape}")
print(f"Action Danger: {damage_info['action_danger'].shape}")
print("SUCCESS!")
