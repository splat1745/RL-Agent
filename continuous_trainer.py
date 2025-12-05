import os
# Set allocator config to reduce fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import glob
import torch
import argparse
import pickle
import numpy as np
from rl.agent import PPOAgent
from rl.memory import Memory

def train_rl_ppo(data_dir, model_path, device_name="cuda"):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"RL Trainer (PPO) started on {device}...")
    print(f"Watching {data_dir} for traj_*.pkl files...")
    
    # Initialize Agent
    agent = PPOAgent(action_dim=26)
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        agent.load(model_path)
    agent.policy.to(device)
    agent.policy.train()
    
    # Processed log
    processed_log_path = "processed_traj.txt"
    processed_files = set()
    if os.path.exists(processed_log_path):
        with open(processed_log_path, "r") as f:
            for line in f:
                processed_files.add(line.strip())
    
    print(f"Loaded {len(processed_files)} processed trajectories from history.")
    
    while True:
        # 1. Find new files
        all_files = glob.glob(os.path.join(data_dir, "traj_*.pkl"))
        # Sort by modification time to process oldest first
        all_files.sort(key=os.path.getmtime)
        
        new_files = [f for f in all_files if os.path.basename(f) not in processed_files]
        
        if not new_files:
            print("No new trajectories. Waiting...")
            time.sleep(5)
            continue
            
        print(f"Found {len(new_files)} new trajectories.")
        
        for f_path in new_files:
            try:
                print(f"Processing {os.path.basename(f_path)}...")
                
                # Load Pickle
                with open(f_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Reconstruct Memory
                memory = Memory()
                
                # Decompress States
                # Data is a list of tuples: (compressed_obs, vector_obs)
                # compressed_obs: {'full': uint8, ...}
                for i, (comp_obs, vec_obs) in enumerate(data['states']):
                    # Decompress
                    full = comp_obs['full'].astype(np.float32) / 255.0
                    crop = comp_obs['crop'].astype(np.float32) / 255.0
                    flow = comp_obs['flow'].astype(np.float32) # Already normalized/float16
                    
                    pixel_obs = {
                        'full': full,
                        'crop': crop,
                        'flow': flow
                    }
                    
                    # Add to memory
                    memory.states.append((pixel_obs, vec_obs.astype(np.float32)))
                    
                memory.actions = data['actions']
                memory.logprobs = data['logprobs']
                memory.rewards = data['rewards']
                memory.is_terminals = data['is_terminals']
                
                hit_history = data.get('hit_history', [])
                
                print(f"  Loaded {len(memory.states)} steps. Updating Agent...")
                
                # Update Agent
                agent.update(memory, hit_history=hit_history)
                
                # Save Model
                agent.save(model_path)
                print("  Model saved.")
                
                # Backup/Mark as processed
                processed_files.add(os.path.basename(f_path))
                with open(processed_log_path, "a") as log:
                    log.write(f"{os.path.basename(f_path)}\n")
                    
                # Optional: Delete file to save space? 
                # os.remove(f_path) 
                
            except Exception as e:
                print(f"Failed to process {f_path}: {e}")
                
        # Small sleep between batches
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/rl_train")
    parser.add_argument("--model", type=str, default="ppo_agent_imitation.pth")
    args = parser.parse_args()
    
    train_rl_ppo(args.data, args.model)