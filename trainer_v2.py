"""
Trainer V2 - Updated training pipeline with staged training.

Training Stages:
1. Stage 0: Data collection (50k sequences)
2. Stage 1: Supervised pretrain (100k steps) - encoder + dynamics + hp + seq
3. Stage 2: RL warmup (100k steps) - freeze early CNN, train policy/value
4. Stage 3: Hybrid training - unfreeze gradually, multi-objective PPO
5. Stage 4: Online loop - episode → upload → train → hot reload

Loss Weights (Initial):
- c_value = 0.5
- c_dynamics = 0.2
- c_hp = 0.2
- c_seq = 0.15
- c_entropy = 0.01
"""

import os
import time
import glob
import pickle
import argparse
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rl.agent_v2 import PPOAgentV2
from rl.network_v2 import RunningMeanStd
from rl.memory import Memory


class TrajectoryDataset(Dataset):
    """
    Dataset for loading trajectories with proper target extraction.
    """
    def __init__(self, data_dir, seq_len=64, device=None, max_files=None):
        self.data = []
        self.seq_len = seq_len
        self.device = device or torch.device('cpu')
        
        files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
        if max_files:
            files = files[:max_files]
            
        print(f"Loading {len(files)} trajectory files...")
        
        for idx, f_path in enumerate(files):
            try:
                with open(f_path, 'rb') as f:
                    session_data = pickle.load(f)
                    
                if not session_data:
                    continue
                    
                # Process into sequences
                for i in range(0, len(session_data) - seq_len - 5, seq_len):
                    seq = session_data[i:i + seq_len]
                    future = session_data[i + seq_len:i + seq_len + 5]
                    
                    if len(seq) == seq_len and len(future) >= 3:
                        processed = self._process_sequence(seq, future)
                        self.data.append(processed)
                        
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
                
            if (idx + 1) % 10 == 0:
                print(f"  Loaded {idx + 1}/{len(files)} files, {len(self.data)} sequences")
                
        print(f"Total: {len(self.data)} sequences")
        
    def _process_sequence(self, seq, future):
        """
        Process sequence and extract targets.
        
        Returns dict with:
        - full_frames: [seq_len, 3, 160, 160]
        - crop_frames: [seq_len, 3, 128, 128]
        - flow: [seq_len, 1, 160, 160]
        - vector_obs: [seq_len, vector_dim]
        - actions: [seq_len]
        - rewards: [seq_len]
        - dones: [seq_len]
        - dynamics_targets: dict
        - hp_targets: [seq_len, 3]
        - seq_targets: [seq_len, 5]
        """
        # Lists to collect
        full_list, crop_list, flow_list = [], [], []
        vector_list, action_list, reward_list, done_list = [], [], [], []
        
        for s in seq:
            # Handle different data formats
            if 'pixel_obs' in s:
                pixel = s['pixel_obs']
                
                # V2 format: RGB only
                if pixel['full'].shape[0] == 3:
                    full_list.append(pixel['full'])
                    crop_list.append(pixel['crop'])
                    flow_list.append(pixel['flow'])
                else:
                    # Old RGBD format - extract RGB only
                    full_list.append(pixel['full'][:3])
                    crop_list.append(pixel['crop'][:3])
                    # Old flow is [2, H, W], convert to magnitude
                    if pixel['flow'].shape[0] == 2:
                        mag = np.sqrt(pixel['flow'][0]**2 + pixel['flow'][1]**2)
                        flow_list.append(mag[np.newaxis, ...])
                    else:
                        flow_list.append(pixel['flow'])
            else:
                # Fallback for old format
                continue
                
            vector_list.append(s.get('vector_obs', np.zeros(151)))
            action_list.append(s.get('action', 0))
            reward_list.append(s.get('reward', 0.0))
            done_list.append(s.get('done', False))
            
        if len(full_list) != self.seq_len:
            return None
            
        # Stack
        full = np.stack(full_list).astype(np.float32)
        crop = np.stack(crop_list).astype(np.float32)
        flow = np.stack(flow_list).astype(np.float32)
        vector = np.stack(vector_list).astype(np.float32)
        actions = np.array(action_list, dtype=np.int64)
        rewards = np.array(reward_list, dtype=np.float32)
        dones = np.array(done_list, dtype=np.float32)
        
        # Extract dynamics targets (predict next enemy position/velocity)
        # Assume vector_obs has: [px, py, vx, vy, enemy_dx, enemy_dy, ...]
        # We predict enemy position and velocity for NEXT frame
        next_pos = np.zeros((self.seq_len, 2), dtype=np.float32)
        next_vel = np.zeros((self.seq_len, 2), dtype=np.float32)
        damage_occurred = np.zeros((self.seq_len, 1), dtype=np.float32)
        
        for i in range(self.seq_len - 1):
            # Next enemy position (indices 4, 5)
            if len(vector[i + 1]) > 5:
                next_pos[i] = vector[i + 1, 4:6]
            # Approximate velocity from position change
            if len(vector[i]) > 5 and len(vector[i + 1]) > 5:
                next_vel[i] = vector[i + 1, 4:6] - vector[i, 4:6]
            # Damage: check if health dropped
            # Assume health is in vector_obs somewhere (adjust index as needed)
            # For now, use reward as proxy (negative reward = damage)
            if rewards[i] < -0.1:
                damage_occurred[i] = 1.0
                
        # HP targets: did HP drop in next 1, 2, 3 frames?
        hp_targets = np.zeros((self.seq_len, 3), dtype=np.float32)
        for i in range(self.seq_len):
            for h in range(3):
                if i + h + 1 < self.seq_len and rewards[i + h + 1] < -0.1:
                    hp_targets[i, h] = 1.0
                    
        # Sequence targets: next 5 actions
        seq_targets = np.zeros((self.seq_len, 5), dtype=np.int64)
        for i in range(self.seq_len):
            for j in range(5):
                if i + j + 1 < self.seq_len:
                    seq_targets[i, j] = actions[i + j + 1]
                elif len(future) > j - (self.seq_len - i - 1):
                    future_idx = j - (self.seq_len - i - 1)
                    if future_idx < len(future):
                        seq_targets[i, j] = future[future_idx].get('action', 0)
                        
        return {
            'full': torch.from_numpy(full),
            'crop': torch.from_numpy(crop),
            'flow': torch.from_numpy(flow),
            'vector': torch.from_numpy(vector),
            'actions': torch.from_numpy(actions),
            'rewards': torch.from_numpy(rewards),
            'dones': torch.from_numpy(dones),
            'next_pos': torch.from_numpy(next_pos),
            'next_vel': torch.from_numpy(next_vel),
            'damage': torch.from_numpy(damage_occurred),
            'hp_targets': torch.from_numpy(hp_targets),
            'seq_targets': torch.from_numpy(seq_targets)
        }
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Custom collate that handles None entries."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
        
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        collated[key] = torch.stack([b[key] for b in batch])
    return collated


class TrainerV2:
    """
    Multi-stage trainer for PolicyNetworkV2.
    """
    def __init__(
        self,
        data_dir,
        model_path,
        device="cuda",
        batch_size=2,
        seq_len=64,
        accumulation_steps=16
    ):
        self.data_dir = data_dir
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.accumulation_steps = accumulation_steps
        
        print(f"TrainerV2 on {self.device}")
        
        # Initialize agent
        self.agent = PPOAgentV2(action_dim=26, freeze_backbone=True)
        if os.path.exists(model_path):
            print(f"Loading existing model: {model_path}")
            self.agent.load(model_path)
        self.agent.policy.to(self.device)
        
        # Training stage
        self.stage = 0
        self.total_steps = 0
        
    def pretrain_supervised(self, epochs=10, lr=1e-4):
        """
        Stage 1: Supervised pretraining of encoder + auxiliary heads.
        No policy RL, just prediction losses.
        """
        print("=" * 50)
        print("Stage 1: Supervised Pretraining")
        print("=" * 50)
        
        # Load dataset
        dataset = TrajectoryDataset(self.data_dir, seq_len=self.seq_len)
        if len(dataset) == 0:
            print("No data found!")
            return
            
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Freeze policy head, train only encoder + aux
        self.agent.policy.freeze_backbone()
        
        # Separate optimizer for pretraining
        pretrain_params = [
            p for n, p in self.agent.policy.named_parameters()
            if p.requires_grad and 'policy_head' not in n and 'value_head' not in n
        ]
        optimizer = torch.optim.AdamW(pretrain_params, lr=lr, weight_decay=1e-5)
        
        for epoch in range(epochs):
            epoch_losses = {
                'dynamics': [], 'hp': [], 'seq': []
            }
            
            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    continue
                    
                # Move to device
                full = batch['full'].to(self.device)
                crop = batch['crop'].to(self.device)
                flow = batch['flow'].to(self.device)
                vector = batch['vector'].to(self.device)
                
                B, S = full.shape[:2]
                
                # Flatten for network
                full_flat = full.view(B * S, *full.shape[2:])
                crop_flat = crop.view(B * S, *crop.shape[2:])
                flow_flat = flow.view(B * S, *flow.shape[2:])
                vector_flat = vector.view(B * S, -1)
                
                # Forward with aux
                _, _, _, aux = self.agent.policy(
                    full_flat, crop_flat, flow_flat, vector_flat,
                    seq_len=S, return_aux=True
                )
                
                # Dynamics loss
                dynamics_targets = {
                    'next_pos': batch['next_pos'].view(B * S, 2).to(self.device),
                    'next_vel': batch['next_vel'].view(B * S, 2).to(self.device),
                    'damage_occurred': batch['damage'].view(B * S, 1).to(self.device)
                }
                dyn_loss = self.agent.policy.dynamics_head.compute_loss(
                    aux['dynamics'], dynamics_targets
                )['total']
                
                # HP loss
                hp_targets = batch['hp_targets'].view(B * S, 3).to(self.device)
                hp_loss = self.agent.policy.hp_head.compute_loss(
                    aux['hp_logits'], hp_targets
                )
                
                # Sequence loss
                seq_targets = batch['seq_targets'].view(B * S, 5).to(self.device)
                seq_loss = self.agent.policy.seq_head.compute_loss(
                    aux['seq_logits'], seq_targets
                )
                
                # Total pretrain loss
                loss = dyn_loss + hp_loss + seq_loss
                
                # Backward with accumulation
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(pretrain_params, 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                epoch_losses['dynamics'].append(dyn_loss.item())
                epoch_losses['hp'].append(hp_loss.item())
                epoch_losses['seq'].append(seq_loss.item())
                
                self.total_steps += 1
                
            # Log epoch
            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Dyn: {avg['dynamics']:.4f} | "
                  f"HP: {avg['hp']:.4f} | "
                  f"Seq: {avg['seq']:.4f}")
            
            # Save
            self.agent.save(self.model_path)
            
        self.stage = 1
        print("Stage 1 complete.")
        
    def train_rl_warmup(self, epochs=10, lr=3e-4):
        """
        Stage 2: RL warmup with frozen early layers.
        Train policy + value only.
        """
        print("=" * 50)
        print("Stage 2: RL Warmup")
        print("=" * 50)
        
        # Keep backbone frozen, train policy/value
        # (Already frozen from Stage 1)
        
        # Training happens via agent.update() with trajectory data
        # This stage uses shorter rollouts (T=128)
        
        dataset = TrajectoryDataset(self.data_dir, seq_len=128)
        if len(dataset) == 0:
            print("No data found!")
            return
            
        loader = DataLoader(
            dataset,
            batch_size=1,  # Process one trajectory at a time
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        for epoch in range(epochs):
            epoch_metrics = []
            
            for batch in loader:
                if batch is None:
                    continue
                    
                # Create memory from batch
                memory = Memory()
                
                B, S = batch['full'].shape[:2]
                
                for i in range(S):
                    pixel_obs = {
                        'full': batch['full'][0, i].numpy(),
                        'crop': batch['crop'][0, i].numpy(),
                        'flow': batch['flow'][0, i].numpy()
                    }
                    vector_obs = batch['vector'][0, i].numpy()
                    
                    memory.states.append((pixel_obs, vector_obs))
                    memory.actions.append(batch['actions'][0, i].item())
                    memory.logprobs.append(0.0)  # Will be recomputed
                    memory.rewards.append(batch['rewards'][0, i].item())
                    memory.is_terminals.append(batch['dones'][0, i].item() > 0.5)
                    
                # Prepare auxiliary targets
                dynamics_targets = {
                    'next_pos': batch['next_pos'][0].numpy(),
                    'next_vel': batch['next_vel'][0].numpy(),
                    'damage_occurred': batch['damage'][0].numpy()
                }
                hp_targets = batch['hp_targets'][0].numpy()
                seq_targets = batch['seq_targets'][0].numpy()
                
                # Update
                metrics = self.agent.update(
                    memory, 
                    dynamics_targets=dynamics_targets,
                    hp_targets=hp_targets,
                    seq_targets=seq_targets
                )
                epoch_metrics.append(metrics)
                
            # Log
            if epoch_metrics:
                avg = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]}
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Total: {avg['total_loss']:.4f} | "
                      f"Policy: {avg['policy_loss']:.4f} | "
                      f"Value: {avg['value_loss']:.4f}")
                      
            self.agent.save(self.model_path)
            
        self.stage = 2
        print("Stage 2 complete.")
        
    def train_hybrid(self, epochs=20):
        """
        Stage 3: Hybrid training with gradual unfreezing.
        Full multi-objective PPO.
        """
        print("=" * 50)
        print("Stage 3: Hybrid Training")
        print("=" * 50)
        
        # Gradually unfreeze backbone
        self.agent.policy.unfreeze_last_layers(num_layers=2)
        
        # Similar to Stage 2 but with full losses
        dataset = TrajectoryDataset(self.data_dir, seq_len=self.seq_len)
        if len(dataset) == 0:
            print("No data found!")
            return
            
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Reduce entropy coefficient over time
        initial_entropy = 0.01
        final_entropy = 0.001
        
        for epoch in range(epochs):
            # Adjust entropy coefficient
            progress = epoch / epochs
            self.agent.loss_fn.c_entropy = initial_entropy * (1 - progress) + final_entropy * progress
            
            epoch_metrics = []
            
            for batch in loader:
                if batch is None:
                    continue
                    
                memory = Memory()
                B, S = batch['full'].shape[:2]
                
                for i in range(S):
                    pixel_obs = {
                        'full': batch['full'][0, i].numpy(),
                        'crop': batch['crop'][0, i].numpy(),
                        'flow': batch['flow'][0, i].numpy()
                    }
                    vector_obs = batch['vector'][0, i].numpy()
                    
                    memory.states.append((pixel_obs, vector_obs))
                    memory.actions.append(batch['actions'][0, i].item())
                    memory.logprobs.append(0.0)
                    memory.rewards.append(batch['rewards'][0, i].item())
                    memory.is_terminals.append(batch['dones'][0, i].item() > 0.5)
                    
                dynamics_targets = {
                    'next_pos': batch['next_pos'][0].numpy(),
                    'next_vel': batch['next_vel'][0].numpy(),
                    'damage_occurred': batch['damage'][0].numpy()
                }
                hp_targets = batch['hp_targets'][0].numpy()
                seq_targets = batch['seq_targets'][0].numpy()
                
                metrics = self.agent.update(
                    memory,
                    dynamics_targets=dynamics_targets,
                    hp_targets=hp_targets,
                    seq_targets=seq_targets
                )
                epoch_metrics.append(metrics)
                
            if epoch_metrics:
                avg = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]}
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Total: {avg['total_loss']:.4f} | "
                      f"Entropy: {avg['entropy']:.4f}")
                      
            # Unfreeze more layers halfway through
            if epoch == epochs // 2:
                self.agent.policy.unfreeze_last_layers(num_layers=3)
                print("Unfroze 3 layers")
                
            self.agent.save(self.model_path)
            
        self.stage = 3
        print("Stage 3 complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="D:/Auto-Farmer-Data/imitation_train")
    parser.add_argument("--model", type=str, default="ppo_v2.pth")
    parser.add_argument("--stage", type=int, default=1, help="1=pretrain, 2=warmup, 3=hybrid")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()
    
    trainer = TrainerV2(
        data_dir=args.data,
        model_path=args.model,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )
    
    if args.stage == 1:
        trainer.pretrain_supervised(epochs=args.epochs)
    elif args.stage == 2:
        trainer.train_rl_warmup(epochs=args.epochs)
    elif args.stage == 3:
        trainer.train_hybrid(epochs=args.epochs)
    else:
        # Run all stages
        trainer.pretrain_supervised(epochs=10)
        trainer.train_rl_warmup(epochs=10)
        trainer.train_hybrid(epochs=20)


if __name__ == "__main__":
    main()
