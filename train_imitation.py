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

class ImitationDataset(Dataset):
    def __init__(self, data_dir, seq_len=64):
        self.data = []
        self.seq_len = seq_len
        
        files = glob.glob(os.path.join(data_dir, "*.pkl"))
        print(f"Found {len(files)} data files.")
        
        for f_path in files:
            try:
                with open(f_path, 'rb') as f:
                    session_data = pickle.load(f)
                    # session_data is a list of dicts
                    if not session_data: continue
                    
                    # Split session into sequences
                    for i in range(0, len(session_data) - seq_len, seq_len):
                        seq = session_data[i : i + seq_len]
                        if len(seq) == seq_len:
                            self.data.append(seq)
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
                
        print(f"Loaded {len(self.data)} sequences of length {seq_len}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        
        # Collate sequence data
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

def train(data_dir, output_model, epochs=10, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # 1. Load Data
    dataset = ImitationDataset(data_dir, seq_len=32) # Smaller seq_len for memory
    if len(dataset) == 0:
        print("No data found. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 2. Initialize Agent
    # Action dim is 26 as per main.py
    agent = PPOAgent(action_dim=26, lr=lr)
    agent.policy.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
    
    # 3. Training Loop
    for epoch in range(epochs):
        total_loss = 0
        batches = 0
        
        for batch in dataloader:
            # Move to device
            # Shape: [Batch, Seq, ...]
            full = batch['full'].to(device)
            crop = batch['crop'].to(device)
            flow = batch['flow'].to(device)
            vector = batch['vector'].to(device)
            targets = batch['actions'].to(device)
            
            # Reshape for Network: [Batch * Seq, ...]
            # But wait, the network handles sequences if we pass seq_len?
            # Let's check agent.update logic.
            # In agent.update: 
            # full_batch = torch.stack(...) -> [Seq, ...] (Batch=1)
            # probs, ... = self.policy(..., seq_len=seq_len)
            
            # The TwoStreamNetwork likely expects flattened batch if seq_len is not provided,
            # or handles reshaping internally if seq_len is provided.
            # Let's assume we need to flatten the batch dimension into the sequence for the LSTM?
            # Actually, LSTM expects (Seq, Batch, Feat) or (Batch, Seq, Feat).
            # If TwoStreamNetwork handles seq_len, it probably expects inputs as (Batch*Seq, ...) 
            # and then reshapes internally before LSTM.
            
            b, s, c, h, w = full.shape
            
            # Flatten Batch and Seq for CNN processing
            full_flat = full.view(-1, *full.shape[2:])
            crop_flat = crop.view(-1, *crop.shape[2:])
            flow_flat = flow.view(-1, *flow.shape[2:])
            vector_flat = vector.view(-1, *vector.shape[2:])
            targets_flat = targets.view(-1)
            
            # Initialize hidden state for this batch
            # (num_layers, batch_size, hidden_dim)
            # Batch size here is 'b' (the number of sequences)
            h0 = torch.zeros(2, b, 768).to(device)
            c0 = torch.zeros(2, b, 768).to(device)
            hidden = (h0, c0)
            
            optimizer.zero_grad()
            
            # Forward
            # We pass seq_len=s so the network knows how to unflatten for LSTM
            probs, _, _, _ = agent.policy(full_flat, crop_flat, flow_flat, vector_flat, hidden_state=hidden, seq_len=s)
            
            # Probs shape: [Batch*Seq, ActionDim]
            loss = criterion(probs, targets_flat)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
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
