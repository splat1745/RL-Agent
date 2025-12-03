import os
import glob
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rl.agent import PPOAgent
import argparse
import gc

class ImitationDataset(Dataset):
    def __init__(self, data_dir, seq_len=64, device=None):
        self.data = []
        self.seq_len = seq_len
        self.device = device if device is not None else torch.device('cpu')
        self.gpu_full = False
        
        files = glob.glob(os.path.join(data_dir, "*.pkl"))
        print(f"Found {len(files)} data files.")
        
        total_steps = 0
        gpu_steps = 0
        
        for f_path in files:
            try:
                with open(f_path, 'rb') as f:
                    session_data = pickle.load(f)
                    if not session_data: continue
                    
                    # Split session into sequences
                    for i in range(0, len(session_data) - seq_len, seq_len):
                        seq = session_data[i : i + seq_len]
                        if len(seq) == seq_len:
                            # Process sequence
                            processed_seq = self._process_sequence(seq)
                            
                            # Try GPU
                            if not self.gpu_full:
                                # Check memory safety margin (leave ~30% free for training overhead)
                                if self.device.type == 'cuda':
                                    total = torch.cuda.get_device_properties(self.device).total_memory
                                    allocated = torch.cuda.memory_allocated(self.device)
                                    if allocated > total * 0.95:
                                        self.gpu_full = True
                                        print(f"GPU Memory threshold reached ({allocated/1024**3:.2f} GB). Switching to CPU.")
                                        self.data.append(processed_seq)
                                        continue

                                try:
                                    # Move all tensors in the dict to GPU
                                    gpu_seq = {k: v.to(self.device) for k, v in processed_seq.items()}
                                    self.data.append(gpu_seq)
                                    gpu_steps += 1
                                except Exception as e:
                                    if "out of memory" in str(e) or "OutOfMemoryError" in str(type(e)):
                                        print("GPU Memory Full (OOM). Switching to CPU RAM for remaining data.")
                                        self.gpu_full = True
                                        torch.cuda.empty_cache()
                                        # Append the CPU version
                                        self.data.append(processed_seq)
                                    else:
                                        raise e
                            else:
                                self.data.append(processed_seq)
                            
                            total_steps += 1
                    
                    # Explicitly delete to free RAM
                    del session_data
                    gc.collect()
                    
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
                
        print(f"Loaded {total_steps} sequences. {gpu_steps} on GPU, {total_steps - gpu_steps} on CPU.")

    def _process_sequence(self, seq):
        # Collate sequence data into tensors
        full = np.array([s['pixel_obs']['full'] for s in seq])
        crop = np.array([s['pixel_obs']['crop'] for s in seq])
        flow = np.array([s['pixel_obs']['flow'] for s in seq])
        vector = np.array([s['vector_obs'] for s in seq])
        actions = np.array([s['action'] for s in seq])
        
        return {
            'full': torch.FloatTensor(full),
            'crop': torch.FloatTensor(crop),
            'flow': torch.FloatTensor(flow),
            'vector': torch.FloatTensor(vector),
            'actions': torch.LongTensor(actions)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is already processed tensors (GPU or CPU)
        return self.data[idx]

def mixed_collate(batch):
    # batch is a list of dicts
    elem = batch[0]
    keys = elem.keys()
    
    collated = {}
    for key in keys:
        # Gather all items for this key
        items = [d[key] for d in batch]
        
        # Check if any are on CPU
        any_cpu = any(item.device.type == 'cpu' for item in items)
        
        if any_cpu:
            # Move all to CPU to stack
            stacked = torch.stack([item.cpu() for item in items])
        else:
            # All on GPU
            stacked = torch.stack(items)
            
        collated[key] = stacked
        
    return collated

def train(data_dir, output_model, epochs=10, batch_size=2, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # 1. Load Data
    # Pass device to dataset to enable GPU preloading
    dataset = ImitationDataset(data_dir, seq_len=32, device=device) 
    if len(dataset) == 0:
        print("No data found. Exiting.")
        return

    # Use custom collate to handle mixed GPU/CPU batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=mixed_collate)
    
    # 2. Initialize Agent
    agent = PPOAgent(action_dim=26, lr=lr)
    agent.policy.train()
    
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
    
    # 3. Training Loop
    for epoch in range(epochs):
        total_loss = 0
        total_aux_loss = 0
        batches = 0
        
        for batch in dataloader:
            # Move to device (if not already there)
            full = batch['full'].to(device)
            crop = batch['crop'].to(device)
            flow = batch['flow'].to(device)
            vector = batch['vector'].to(device)
            targets = batch['actions'].to(device)
            
            b, s, c, h, w = full.shape
            
            # Flatten Batch and Seq for CNN processing
            full_flat = full.view(-1, *full.shape[2:])
            crop_flat = crop.view(-1, *crop.shape[2:])
            flow_flat = flow.view(-1, *flow.shape[2:])
            vector_flat = vector.view(-1, *vector.shape[2:])
            targets_flat = targets.view(-1)
            
            # Initialize hidden state for this batch
            h0 = torch.zeros(2, b, 768).to(device)
            c0 = torch.zeros(2, b, 768).to(device)
            hidden = (h0, c0)
            
            optimizer.zero_grad()
            
            # Forward with Aux
            probs, _, _, _, pred_camera, pred_enemy, pred_next_vec = agent.policy(
                full_flat, crop_flat, flow_flat, vector_flat, 
                hidden_state=hidden, seq_len=s, return_aux=True
            )
            
            # Main Action Loss
            loss_action = criterion(probs, targets_flat)
            
            # --- Auxiliary Losses ---
            # 1. Camera Motion (Indices 25, 26 in vector_obs)
            # Target: vector_flat[:, 25:27]
            target_camera = vector_flat[:, 25:27]
            loss_camera = mse_loss(pred_camera, target_camera)
            
            # 2. Enemy Position (Indices 4, 5 in vector_obs)
            # Target: vector_flat[:, 4:6]
            target_enemy = vector_flat[:, 4:6]
            loss_enemy = mse_loss(pred_enemy, target_enemy)
            
            # 3. Forward Dynamics (Next State Prediction)
            # Target: vector_flat shifted by 1.
            # We predict state[t+1] from state[t].
            # Since we flattened [Batch, Seq], we need to be careful.
            # We can reshape back to [Batch, Seq, Dim] to shift.
            pred_next_vec_seq = pred_next_vec.view(b, s, -1)
            vector_seq = vector.view(b, s, -1)
            
            # Predict t+1 from t.
            # Pred[t] should match Vector[t+1]
            # We ignore the last prediction (for t=N) and last target (t=0)
            # Actually:
            # Input at t -> Output Pred[t] -> Target Vector[t+1]
            # So we compare Pred[:, :-1] with Vector[:, 1:]
            
            loss_dynamics = 0
            if s > 1:
                pred_steps = pred_next_vec_seq[:, :-1, :]
                target_steps = vector_seq[:, 1:, :]
                loss_dynamics = mse_loss(pred_steps, target_steps)
            
            # Total Loss
            # Weight auxiliary losses
            loss = loss_action + 0.5 * loss_camera + 0.5 * loss_enemy + 0.5 * loss_dynamics
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Handle scalar 0 for loss_dynamics if s <= 1
            dyn_val = loss_dynamics.item() if isinstance(loss_dynamics, torch.Tensor) else 0
            total_aux_loss += (loss_camera.item() + loss_enemy.item() + dyn_val)
            batches += 1
            
        avg_loss = total_loss / batches if batches > 0 else 0
        avg_aux = total_aux_loss / batches if batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} (Aux: {avg_aux:.4f})")
        
        # Save checkpoint
        agent.save(output_model)
        print(f"Saved model to {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/imitation", help="Path to imitation data")
    parser.add_argument("--out", type=str, default="ppo_agent_imitation.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    args = parser.parse_args()
    
    train(args.data, args.out, args.epochs)
