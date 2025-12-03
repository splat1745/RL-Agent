import os
import time
import glob
import torch
import argparse
from rl.agent import PPOAgent
from train_imitation import ImitationDataset, mixed_collate
from torch.utils.data import DataLoader

def continuous_train(data_dir, model_path, device_name="cuda"):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Continuous Trainer started on {device}...")
    print(f"Watching {data_dir} for new data...")
    
    # Initialize Agent
    agent = PPOAgent(action_dim=26)
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        agent.load(model_path)
    agent.policy.to(device)
    agent.policy.train()
    
    processed_files = set()
    
    # Initial scan to mark existing files as processed (optional, or train on them first)
    # existing = glob.glob(os.path.join(data_dir, "*.pkl"))
    # for f in existing:
    #     processed_files.add(f)
    
    # Convert set to list for sampling
    processed_list = list(processed_files)
    import random
        
    while True:
        # 1. Find new files
        all_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        # Use set for fast lookup
        processed_set = set(processed_list)
        new_files = [f for f in all_files if f not in processed_set]
        
        if not new_files:
            time.sleep(5)
            continue
            
        print(f"Found {len(new_files)} new data files.")
        
        # --- Replay Logic ---
        # Mix new files with up to 5 random old files to prevent forgetting
        replay_files = []
        if processed_list:
            k = min(len(processed_list), 5)
            replay_files = random.sample(processed_list, k)
            print(f"Mixing in {len(replay_files)} old files for replay.")
            
        training_files = new_files + replay_files
        
        # 2. Load Data
        dataset = ImitationDataset(data_dir, seq_len=16, device=device, file_list=training_files)
        
        if len(dataset) == 0:
            print("Dataset is empty (possibly due to corrupted files). Skipping training round.")
            time.sleep(5)
            continue
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=mixed_collate)
        
        # 3. Train (Few epochs on the new data)
        optimizer = torch.optim.Adam(agent.policy.parameters(), lr=1e-4)
        
        print("Training...")
        for epoch in range(5): # Quick updates
            total_loss = 0
            for batch in loader:
                # ... (Simplified training loop) ...
                full = batch['full'].to(device)
                crop = batch['crop'].to(device)
                flow = batch['flow'].to(device)
                vector = batch['vector'].to(device)
                targets = batch['actions'].to(device)
                
                full_flat = full.view(-1, *full.shape[2:])
                crop_flat = crop.view(-1, *crop.shape[2:])
                flow_flat = flow.view(-1, *flow.shape[2:])
                vector_flat = vector.view(-1, *vector.shape[2:])
                targets_flat = targets.view(-1)
                
                optimizer.zero_grad()
                # Policy returns 4 values by default: probs, value, hidden, aux/intention
                probs, _, _, _ = agent.policy(full_flat, crop_flat, flow_flat, vector_flat, seq_len=batch['full'].shape[1])
                loss = torch.nn.functional.cross_entropy(probs, targets_flat)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"  Update Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")
            
        # 4. Save Model
        print(f"Saving updated model to {model_path}...")
        agent.save(model_path)
        
        # Mark as processed
        for f in new_files:
            processed_list.append(f)
            
        print("Waiting for more data...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/imitation")
    parser.add_argument("--model", type=str, default="ppo_agent_imitation.pth")
    args = parser.parse_args()
    
    continuous_train(args.data, args.model)