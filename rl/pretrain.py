"""
Stage 1: Supervised Pretraining
Refines the network using existing offline data before starting RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
from tqdm import tqdm

from rl.temporal_network import TemporalMuZeroNetwork

class OfflineDataset(Dataset):
    def __init__(self, data_path, seq_len=30):
        self.files = glob.glob(os.path.join(data_path, "*.npy")) # Assumption
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Placeholder loading - expects dictionary or arrays
        # This implementation assumes the data structure matches buffers
        data = np.load(self.files[idx], allow_pickle=True).item()
        return data

class Pretrainer:
    def __init__(self, data_path, save_path="checkpoints/stage1_pretrained.pt", 
                 device='cuda', action_dim=26, hidden_dim=512):
        self.device = device
        self.save_path = save_path
        
        self.network = TemporalMuZeroNetwork(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            seq_len=30,
            freeze_backbone=False # Fine-tune backbone too? Or keep frozen?
            # Usually pretrain = train everything on valid human data
        ).to(device)
        
        self.optimizer = optim.AdamW(self.network.parameters(), lr=1e-4)
        
        # Datasets
        # self.dataset = OfflineDataset(data_path)
        # self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        
    def train_epoch(self):
        self.network.train()
        total_loss = 0
        
        # Placeholder loop
        # for batch in self.dataloader:
        #    ...
        
        return total_loss
        
    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stage': 1
        }, self.save_path)
        print(f"Saved pretrained model to {self.save_path}")

if __name__ == "__main__":
    # Example usage
    trainer = Pretrainer("data/offline")
    # trainer.train_epoch()
    trainer.save() # Just save initialized model if no data
