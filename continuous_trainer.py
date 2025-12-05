import os
# Set allocator config to reduce fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import glob
import torch
import argparse
import re
from rl.agent import PPOAgent
from train_imitation import ImitationDataset, mixed_collate
from torch.utils.data import DataLoader

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def continuous_train(data_dir, model_path, device_name="cuda"):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Continuous Trainer started on {device}...")
    print(f"Watching {data_dir} (Absolute: {os.path.abspath(data_dir)}) for *.pkl files...")
    
    # Initialize Agent
    agent = PPOAgent(action_dim=26)
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        agent.load(model_path)
    agent.policy.to(device)
    agent.policy.train()
    
    # Load processed files history
    processed_log_path = "processed_files.txt"
    processed_files = set()
    if os.path.exists(processed_log_path):
        with open(processed_log_path, "r") as f:
            for line in f:
                processed_files.add(line.strip())
    
    print(f"Loaded {len(processed_files)} processed files from history.")
    
    # Convert set to list for sampling
    processed_list = list(processed_files)
    import random
        
    while True:
        # 1. Find new files
        all_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        all_files.sort(key=natural_keys) # Sort naturally (data_1, data_2, ... data_10)
        
        if len(all_files) == 0:
             print(f"No .pkl files found in {data_dir}. Waiting...")
             time.sleep(5)
             continue

        # Use basenames for robust comparison (avoids path differences)
        processed_basenames = set()
        for p in processed_files:
            processed_basenames.add(os.path.basename(p))
            
        new_files = []
        for f in all_files:
            if os.path.basename(f) not in processed_basenames:
                new_files.append(f)
        
        if not new_files:
            print(f"No new files. Watching {len(all_files)} total files. (Last seen: {os.path.basename(all_files[-1]) if all_files else 'None'})")
            time.sleep(5)
            continue
            
        print(f"Found {len(new_files)} new data files: {[os.path.basename(f) for f in new_files]}")
        
        # --- Replay Logic ---
        # Mix new files with up to 5 random old files to prevent forgetting
        replay_files = []
        if processed_list:
            k = min(len(processed_list), 5)
            replay_files = random.sample(processed_list, k)
            print(f"Mixing in {len(replay_files)} old files for replay.")
            
        training_files = new_files + replay_files
        random.shuffle(training_files) # Shuffle order
        
        print(f"Starting Sequential Training on {len(training_files)} files...")
        
        successfully_processed = set()
        
        # 2. Chunked Training Loop
        # Process files in batches to fill GPU without crashing
        chunk_size = 5 
        for i in range(0, len(training_files), chunk_size):
            chunk = training_files[i : i + chunk_size]
            print(f"Processing Chunk {i//chunk_size + 1} ({len(chunk)} files)...")
            
            # Load Chunk
            try:
                # Try to load into GPU first (up to 70%), then spill to CPU
                dataset = ImitationDataset(data_dir, seq_len=16, device=device, file_list=chunk)
                
                # Track successful loads
                if hasattr(dataset, 'successfully_loaded_files'):
                    for f in dataset.successfully_loaded_files:
                        successfully_processed.add(os.path.abspath(f))
                        
            except Exception as e:
                print(f"Failed to load chunk: {e}")
                continue
                
            if len(dataset) == 0:
                print("  Empty dataset, skipping.")
                continue
            
            # Reduced batch size to 2 for stability
            loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=mixed_collate)
            
            # 3. Train (Few epochs per chunk)
            print(f"  Starting training on chunk with {len(dataset)} sequences...")
            optimizer = torch.optim.Adam(agent.policy.parameters(), lr=1e-4)
            accumulation_steps = 16
            
            for epoch in range(27): # Reduced epochs per file since we iterate many
                total_loss = 0
                batches = 0
                optimizer.zero_grad()
                
                for i, batch in enumerate(loader):
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
                    
                    # Policy returns 4 values: probs, value, hidden, aux/intention
                    probs, _, _, _ = agent.policy(full_flat, crop_flat, flow_flat, vector_flat, seq_len=batch['full'].shape[1])
                    loss = torch.nn.functional.cross_entropy(probs, targets_flat)
                    
                    # Normalize loss for accumulation
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    total_loss += loss.item() * accumulation_steps
                    batches += 1
                
                # Step for remaining gradients
                if batches % accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                if batches > 0:
                    print(f"  Epoch {epoch+1}: Loss {total_loss/batches:.4f}")
            
            # Cleanup Memory Immediately
            del dataset
            del loader
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        # 4. Save Model (After batch is done)
        print(f"Saving updated model to {model_path}...")
        agent.save(model_path)
        
        # Mark as processed
        with open(processed_log_path, "a") as f:
            for f_path in new_files:
                # Only mark if it was successfully loaded
                if os.path.abspath(f_path) in successfully_processed:
                    processed_files.add(f_path)
                    processed_list.append(f_path)
                    f.write(f"{f_path}\n")
                else:
                    print(f"Skipping mark for {os.path.basename(f_path)} (failed to load). Will retry next loop.")
            
        print("Waiting for more data...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/imitation")
    parser.add_argument("--model", type=str, default="ppo_agent_imitation.pth")
    args = parser.parse_args()
    
    continuous_train(args.data, args.model)