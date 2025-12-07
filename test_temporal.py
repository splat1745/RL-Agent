w"""Quick test for temporal network shapes."""
import torch
from rl.temporal_network import TemporalMuZeroNetwork

def test_shapes():
    # Batch size 2, Sequence length 30, Action Dim 26
    B, T, C, H, W = 2, 30, 3, 128, 128
    
    frames = torch.randn(B, T, C, H, W).cuda()
    past_actions = torch.randint(0, 26, (B, T)).cuda()
    cooldowns = torch.zeros(B, 4).cuda()
    enemy_state = torch.randn(B, T, 6).cuda()  # [B, T, 6]
    
    # Initialize network (512 dim, 4 layers)
    net = TemporalMuZeroNetwork(action_dim=26, hidden_dim=512, seq_len=T).cuda()
    
    print("Testing TemporalMuZeroNetwork...")
    policy, value, hp_pred, seq_pred, _, damage_info = net(frames, past_actions, cooldowns, enemy_state)
    
    print(f"Policy: {policy.shape}")
    print(f"Value: {value.shape}")
    print(f"HP Pred: {hp_pred.shape}")
    print(f"Seq Pred: {seq_pred.shape}")
    print(f"Frame Scores: {damage_info['frame_scores'].shape}")
    print("SUCCESS!")

if __name__ == "__main__":
    test_shapes()
