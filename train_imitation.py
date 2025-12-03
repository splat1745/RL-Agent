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
from collections import Counter

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
        # Decompress from uint8/float16 to float32 if needed
        
        full_list = []
        crop_list = []
        flow_list = []
        vector_list = []
        action_list = []
        
        for s in seq:
            # Decompress Full: uint8 [0-255] -> float32 [0.0-1.0]
            # If already float (old data), just cast
            f_raw = s['pixel_obs']['full']
            if f_raw.dtype == np.uint8:
                f = f_raw.astype(np.float32) / 255.0
            else:
                f = f_raw.astype(np.float32)
            full_list.append(f)
            
            # Decompress Crop: uint8 [0-255] -> float32 [0.0-1.0]
            c_raw = s['pixel_obs']['crop']
            if c_raw.dtype == np.uint8:
                c = c_raw.astype(np.float32) / 255.0
            else:
                c = c_raw.astype(np.float32)
            crop_list.append(c)
            
            # Decompress Flow: float16 -> float32
            fl = s['pixel_obs']['flow'].astype(np.float32)
            flow_list.append(fl)
            
            # Decompress Vector: float16 -> float32
            v = s['vector_obs'].astype(np.float32)
            vector_list.append(v)
            
            action_list.append(s['action'])
            
        full = np.array(full_list)
        crop = np.array(crop_list)
        flow = np.array(flow_list)
        vector = np.array(vector_list)
        actions = np.array(action_list)
        
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

def train(data_dir, output_model, epochs=10, batch_size=2, lr=1e-4, accumulation_steps=16, freeze=False, seq_len=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Optimize for fixed input sizes
    torch.backends.cudnn.benchmark = True
    
    # 1. Load Data
    # Pass device to dataset to enable GPU preloading
    full_dataset = ImitationDataset(data_dir, seq_len=seq_len, device=device) 
    if len(full_dataset) == 0:
        print("No data found. Exiting.")
        return

    # Split Train/Val (85% Train, 15% Val)
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {train_size} sequences, Validating on {val_size} sequences.")

    # Use custom collate to handle mixed GPU/CPU batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=mixed_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=mixed_collate)
    
    # 2. Initialize Agent
    agent = PPOAgent(action_dim=26, lr=lr)
    
    if freeze:
        agent.policy.freeze_backbone()
        
    agent.policy.train()
    
    # Calculate Class Weights (Use only training data to avoid leakage)
    print("Calculating class weights from training set...")
    all_actions = []
    # We need to iterate the subset. random_split returns a Subset object.
    # Subset.dataset is the original, Subset.indices are the indices.
    for idx in train_dataset.indices:
        seq = full_dataset.data[idx]
        actions = seq['actions'].cpu().numpy()
        all_actions.extend(actions)
        
    counter = Counter(all_actions)
    total_samples = len(all_actions)
    num_classes = 26
    
    print("Top 5 Actions:")
    for action, count in counter.most_common(5):
        print(f"  Action {action}: {count} ({count/total_samples*100:.2f}%)")
    
    weights = torch.ones(num_classes).to(device)
    for cls_id in range(num_classes):
        count = counter.get(cls_id, 0)
        if count > 0:
            # Inverse frequency
            weights[cls_id] = total_samples / (count * num_classes)
        else:
            weights[cls_id] = 1.0
            
    # Normalize weights
    weights = weights / weights.mean()
    print(f"Class Weights: {weights}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
    
    # Scheduler to reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # 3. Training Loop
    print(f"Starting training with Batch Size {batch_size} and Accumulation Steps {accumulation_steps} (Effective Batch: {batch_size * accumulation_steps})")
    
    for epoch in range(epochs):
        # --- TRAIN ---
        agent.policy.train()
        total_loss = 0
        total_aux_loss = 0
        train_correct = 0
        train_total = 0
        batches = 0
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
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
            
            # Mixed Precision Forward
            with torch.cuda.amp.autocast():
                # Forward with Aux
                probs, _, _, _, pred_camera, pred_enemy, pred_next_vec = agent.policy(
                    full_flat, crop_flat, flow_flat, vector_flat, 
                    hidden_state=hidden, seq_len=s, return_aux=True
                )
                
                # Main Action Loss
                loss_action = criterion(probs, targets_flat)
                
                # Accuracy Calc
                _, predicted = torch.max(probs, 1)
                
                # --- Auxiliary Losses ---
                target_camera = vector_flat[:, 25:27]
                loss_camera = mse_loss(pred_camera, target_camera)
                
                target_enemy = vector_flat[:, 4:6]
                loss_enemy = mse_loss(pred_enemy, target_enemy)
                
                pred_next_vec_seq = pred_next_vec.view(b, s, -1)
                vector_seq = vector.view(b, s, -1)
                
                loss_dynamics = 0
                if s > 1:
                    pred_steps = pred_next_vec_seq[:, :-1, :]
                    target_steps = vector_seq[:, 1:, :]
                    loss_dynamics = mse_loss(pred_steps, target_steps)
                
                # Total Loss
                loss = loss_action + 0.1 * loss_camera + 0.1 * loss_enemy + 0.1 * loss_dynamics
                
                # Normalize loss for accumulation
                loss = loss / accumulation_steps
            
            # Mixed Precision Backward
            scaler.scale(loss).backward()
            
            # Step Optimizer
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Stats (Move to CPU for logging to avoid sync)
            train_correct += (predicted == targets_flat).sum().item()
            train_total += targets_flat.size(0)
            total_loss += loss.item() * accumulation_steps 
            
            # Handle scalar 0 for loss_dynamics if s <= 1
            dyn_val = loss_dynamics.item() if isinstance(loss_dynamics, torch.Tensor) else 0
            total_aux_loss += (loss_camera.item() + loss_enemy.item() + dyn_val)
            batches += 1
            
        # Step for any remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        avg_loss = total_loss / batches if batches > 0 else 0
        avg_aux = total_aux_loss / batches if batches > 0 else 0
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        
        # --- VALIDATION ---
        agent.policy.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                full = batch['full'].to(device)
                crop = batch['crop'].to(device)
                flow = batch['flow'].to(device)
                vector = batch['vector'].to(device)
                targets = batch['actions'].to(device)
                
                b, s, c, h, w = full.shape
                
                full_flat = full.view(-1, *full.shape[2:])
                crop_flat = crop.view(-1, *crop.shape[2:])
                flow_flat = flow.view(-1, *flow.shape[2:])
                vector_flat = vector.view(-1, *vector.shape[2:])
                targets_flat = targets.view(-1)
                
                h0 = torch.zeros(2, b, 768).to(device)
                c0 = torch.zeros(2, b, 768).to(device)
                hidden = (h0, c0)
                
                probs, _, _, _ = agent.policy(
                    full_flat, crop_flat, flow_flat, vector_flat, 
                    hidden_state=hidden, seq_len=s, return_aux=False
                )
                
                loss = criterion(probs, targets_flat)
                val_loss += loss.item()
                
                _, predicted = torch.max(probs, 1)
                val_correct += (predicted == targets_flat).sum().item()
                val_total += targets_flat.size(0)
                val_batches += 1
                
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Step Scheduler
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (Aux: {avg_aux:.4f}) | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")
        
        # Save Latest
        agent.save(output_model)
        
        # Save Best
        # Check if accuracy improved OR if accuracy is same but loss improved
        if val_acc > best_val_acc or (val_acc == best_val_acc and avg_val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_name = output_model.replace(".pth", "_best.pth")
            agent.save(best_name)
            print(f"New Best Model! Saved to {best_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/imitation", help="Path to imitation data")
    parser.add_argument("--out", type=str, default="ppo_agent_imitation.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step")
    parser.add_argument("--accumulate", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--freeze", action="store_true", help="Freeze the MobileNet backbone")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length for LSTM")
    args = parser.parse_args()
    
    train(args.data, args.out, args.epochs, args.batch_size, accumulation_steps=args.accumulate, freeze=args.freeze, seq_len=args.seq_len)
