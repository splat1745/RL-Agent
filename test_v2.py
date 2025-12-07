"""
Test script for V2 network architecture.
Validates shapes, forward pass, and loss computation.
"""

import torch
import numpy as np
import sys

def test_network_shapes():
    """Test that all tensor shapes are correct."""
    print("=" * 50)
    print("Testing Network V2 Shapes")
    print("=" * 50)
    
    from rl.network_v2 import PolicyNetworkV2, build_action_mask, DEFAULT_SKILL_TO_ACTION
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize network
    net = PolicyNetworkV2(action_dim=26, vector_dim=151, freeze_backbone=True).to(device)
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    
    # Test inputs
    batch_size = 2
    seq_len = 4
    
    full_frames = torch.randn(batch_size * seq_len, 3, 160, 160).to(device)
    crop_frames = torch.randn(batch_size * seq_len, 3, 128, 128).to(device)
    flow = torch.randn(batch_size * seq_len, 1, 160, 160).to(device)
    vector_obs = torch.randn(batch_size * seq_len, 151).to(device)
    
    print(f"\nInput shapes:")
    print(f"  full_frames: {full_frames.shape}")
    print(f"  crop_frames: {crop_frames.shape}")
    print(f"  flow: {flow.shape}")
    print(f"  vector_obs: {vector_obs.shape}")
    
    # Forward pass without aux
    action_probs, value, hidden = net(
        full_frames, crop_frames, flow, vector_obs,
        seq_len=seq_len, return_aux=False
    )
    
    print(f"\nOutput shapes (without aux):")
    print(f"  action_probs: {action_probs.shape}")
    print(f"  value: {value.shape}")
    print(f"  hidden[0]: {hidden[0].shape}")
    print(f"  hidden[1]: {hidden[1].shape}")
    
    assert action_probs.shape == (batch_size * seq_len, 26), "Action probs shape mismatch"
    assert value.shape == (batch_size * seq_len, 1), "Value shape mismatch"
    assert hidden[0].shape == (2, batch_size, 512), "Hidden h shape mismatch"
    assert hidden[1].shape == (2, batch_size, 512), "Hidden c shape mismatch"
    
    # Forward pass with aux
    action_probs, value, hidden, aux = net(
        full_frames, crop_frames, flow, vector_obs,
        seq_len=seq_len, return_aux=True
    )
    
    print(f"\nAuxiliary output shapes:")
    print(f"  dynamics.pos_mean: {aux['dynamics']['pos_mean'].shape}")
    print(f"  dynamics.pos_logvar: {aux['dynamics']['pos_logvar'].shape}")
    print(f"  dynamics.vel_mean: {aux['dynamics']['vel_mean'].shape}")
    print(f"  dynamics.damage_logit: {aux['dynamics']['damage_logit'].shape}")
    print(f"  dynamics.latent: {aux['dynamics']['latent'].shape}")
    print(f"  hp_logits: {aux['hp_logits'].shape}")
    print(f"  seq_logits: {aux['seq_logits'].shape}")
    
    assert aux['dynamics']['pos_mean'].shape == (batch_size * seq_len, 2)
    assert aux['dynamics']['latent'].shape == (batch_size * seq_len, 64)
    assert aux['hp_logits'].shape == (batch_size * seq_len, 3)
    assert aux['seq_logits'].shape == (batch_size * seq_len, 5, 26)
    
    # Test action masking
    cooldowns = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]).float()
    mask = build_action_mask(cooldowns, DEFAULT_SKILL_TO_ACTION, 26)
    print(f"\nAction mask shape: {mask.shape}")
    print(f"  Masked actions for batch 0: {[i for i in range(26) if not mask[0, i]]}")
    print(f"  Masked actions for batch 1: {[i for i in range(26) if not mask[1, i]]}")
    
    print("\n✓ All shape tests passed!")
    return True


def test_loss_computation():
    """Test loss computation."""
    print("\n" + "=" * 50)
    print("Testing Loss Computation")
    print("=" * 50)
    
    from rl.network_v2 import PolicyNetworkV2, PolicyNetworkV2Loss
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = PolicyNetworkV2(action_dim=26).to(device)
    loss_fn = PolicyNetworkV2Loss()
    
    batch_size = 4
    
    # Create real forward pass to get gradients
    full_frames = torch.randn(batch_size, 3, 160, 160, requires_grad=True).to(device)
    crop_frames = torch.randn(batch_size, 3, 128, 128, requires_grad=True).to(device)
    flow = torch.randn(batch_size, 1, 160, 160, requires_grad=True).to(device)
    vector_obs = torch.randn(batch_size, 151, requires_grad=True).to(device)
    
    action_probs, value, _, aux_outputs = net(
        full_frames, crop_frames, flow, vector_obs,
        seq_len=1, return_aux=True
    )
    aux_outputs['value'] = value
    
    # Mock targets
    old_log_probs = torch.randn(batch_size).to(device)
    actions = torch.randint(0, 26, (batch_size,)).to(device)
    advantages = torch.randn(batch_size).to(device)
    returns = torch.randn(batch_size).to(device)
    
    dynamics_targets = {
        'next_pos': torch.randn(batch_size, 2).to(device),
        'next_vel': torch.randn(batch_size, 2).to(device),
        'damage_occurred': torch.randint(0, 2, (batch_size, 1)).float().to(device)
    }
    
    hp_targets = torch.randint(0, 2, (batch_size, 3)).float().to(device)
    seq_targets = torch.randint(0, 26, (batch_size, 5)).to(device)
    
    # Compute loss
    losses = loss_fn.compute(
        net, old_log_probs, actions, advantages, returns,
        aux_outputs, dynamics_targets, hp_targets, seq_targets
    )
    
    print(f"Loss breakdown:")
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  Policy: {losses['policy'].item():.4f}")
    print(f"  Value: {losses['value'].item():.4f}")
    print(f"  Dynamics: {losses['dynamics'].item():.4f}")
    print(f"  HP: {losses['hp'].item():.4f}")
    print(f"  Seq: {losses['seq'].item():.4f}")
    print(f"  Entropy: {losses['entropy'].item():.4f}")
    
    # Check gradients flow
    losses['total'].backward()
    
    grad_count = sum(1 for p in net.parameters() if p.grad is not None)
    print(f"\nGradients computed for {grad_count} parameters")
    
    print("\n✓ Loss computation test passed!")
    return True


def test_agent_integration():
    """Test full agent workflow."""
    print("\n" + "=" * 50)
    print("Testing Agent V2 Integration")
    print("=" * 50)
    
    from rl.agent_v2 import PPOAgentV2
    
    agent = PPOAgentV2(action_dim=26, freeze_backbone=True)
    print(f"Agent initialized on {agent.device}")
    
    # Mock observation
    pixel_obs = {
        'full': np.random.rand(3, 160, 160).astype(np.float32),
        'crop': np.random.rand(3, 128, 128).astype(np.float32),
        'flow': np.random.rand(1, 160, 160).astype(np.float32)
    }
    vector_obs = np.random.rand(151).astype(np.float32)
    cooldowns = np.array([0, 1, 0, 0], dtype=np.float32)
    
    # Select action
    action, log_prob, value = agent.select_action(pixel_obs, vector_obs, cooldowns)
    
    print(f"\nAction selection:")
    print(f"  Action: {action}")
    print(f"  Log prob: {log_prob:.4f}")
    print(f"  Value: {value:.4f}")
    
    assert 0 <= action < 26, "Invalid action"
    
    # Test normalization is working
    vec_mean = agent.vec_normalizer.mean
    vec_var = agent.vec_normalizer.var
    print(f"\nVector normalizer stats updated:")
    print(f"  Mean range: [{vec_mean.min():.4f}, {vec_mean.max():.4f}]")
    print(f"  Var range: [{vec_var.min():.4f}, {vec_var.max():.4f}]")
    
    print("\n✓ Agent integration test passed!")
    return True


def test_feature_builder():
    """Test feature builder module."""
    print("\n" + "=" * 50)
    print("Testing Feature Builder")
    print("=" * 50)
    
    from utils.feature_builder import FeatureBuilder
    
    builder = FeatureBuilder(full_size=160, crop_size=128, vector_dim=151)
    
    # Mock frame and detections
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    det_dict = {
        'player': (320, 320, 50, 100, 0.9),  # cx, cy, w, h, conf
        'enemies': [(400, 300, 40, 80, 0.8)],
        'obstacles': [],
        'goal': None,
        'health': 0.75,
        'cooldowns': {'skill_0': 0, 'skill_1': 2.5, 'skill_2': 0, 'skill_3': 0}
    }
    
    # Build observations
    pixel_obs, vector_obs = builder.build(frame, det_dict, last_action=5, mouse_movement=(10, -5))
    
    print(f"Pixel observation shapes:")
    print(f"  full: {pixel_obs['full'].shape}")
    print(f"  crop: {pixel_obs['crop'].shape}")
    print(f"  flow: {pixel_obs['flow'].shape}")
    print(f"Vector observation shape: {vector_obs.shape}")
    
    assert pixel_obs['full'].shape == (3, 160, 160)
    assert pixel_obs['crop'].shape == (3, 128, 128)
    assert pixel_obs['flow'].shape == (1, 160, 160)
    assert vector_obs.shape == (151,)
    
    # Check normalization
    print(f"\nVector observation stats:")
    print(f"  Mean: {vector_obs.mean():.4f}")
    print(f"  Std: {vector_obs.std():.4f}")
    
    # Get Kalman states
    kalman = builder.get_kalman_states()
    print(f"\nKalman states:")
    print(f"  Player: x={kalman['player']['x']:.1f}, y={kalman['player']['y']:.1f}, conf={kalman['player']['confidence']:.2f}")
    print(f"  Enemy: x={kalman['enemy']['x']:.1f}, y={kalman['enemy']['y']:.1f}, conf={kalman['enemy']['confidence']:.2f}")
    
    print("\n✓ Feature builder test passed!")
    return True


def main():
    """Run all tests."""
    print("V2 Architecture Test Suite")
    print("=" * 50)
    
    tests = [
        ("Network Shapes", test_network_shapes),
        ("Loss Computation", test_loss_computation),
        ("Agent Integration", test_agent_integration),
        ("Feature Builder", test_feature_builder),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
