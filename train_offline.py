import os
import glob
import pickle
import numpy as np
import torch
import cv2
from rl.muzero_agent import MuZeroAgent
from tqdm import tqdm

def load_trajectory(filepath):
    """
    Loads a .pkl trajectory and adapts it for Vision-Only MuZero.
    Returns list of (frames, action, reward, next_frames, done, logprob)
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        memory = []
        states = data['states'] # List of (compressed_obs, vector_obs)
        actions = data['actions']
        rewards = data['rewards']
        logprobs = data['logprobs']
        
        # Pre-process frames: uint8 -> float32 0-1, Resize to 128x128
        # Stack: full frame only? Or full + flow?
        # Plan says: "96x96 or 128x128 RGB Frame Stack (4 frames)"
        # Current data: 'full' is 160x160.
        
        processed_frames = []
        
        for i, (comp_obs, _) in enumerate(states):
            # Decompress
            # comp_obs['full'] is (160, 160, 3)? No, check main.py compression
            # main.py: (pixel_obs['full'] * 255).astype(np.uint8)
            # pixel_obs['full'] comes from capture: (C, H, W) or (H, W, C)?
            # Torchvision models expect (C, H, W).
            # Let's assume stored as (C, H, W) based on previous code.
            
            full = comp_obs['full'].astype(np.float32) / 255.0 
            # Force print to debug
            # print(f"DEBUG: comp_obs['full'] shape: {full.shape}", flush=True)
            
            # Robust Resize Logic
            # Goal: Get to (12, 128, 128)
            
            if full.ndim == 3:
                channels = -1
                img_to_resize = None
                
                # Case 1: (H, W, C) where H=W=160
                if full.shape[0] == 160 and full.shape[1] == 160:
                    img_to_resize = full # Already (160, 160, C)
                    
                # Case 2: (C, H, W) where H=W=160
                elif full.shape[1] == 160 and full.shape[2] == 160:
                    img_to_resize = np.transpose(full, (1, 2, 0)) # -> (160, 160, C)
                    
                # Case 3: (H, C, W)? Unlikely but possible if reshape error
                else:
                    print(f"Skipping weird shape: {full.shape}", flush=True)
                    continue
                    
                # Perform Resize
                # cv2.resize takes (W, H)
                resized = cv2.resize(img_to_resize, (128, 128)) # -> (128, 128, C)
                
                # Transpose back to (C, H, W) for PyTorch
                resized = np.transpose(resized, (2, 0, 1))
                
                # Check Channels
                if resized.shape[0] == 3:
                    # Stack info was missing, simulate it
                    resized = np.concatenate([resized]*4, axis=0) # (12, 128, 128)
                elif resized.shape[0] == 4:
                     # (4, 128, 128) -> maybe RGBA? Or single channel stack? 
                     # MobileNet expects 12. 
                     # Replicate 3 times?
                     resized = np.concatenate([resized]*3, axis=0)
                elif resized.shape[0] == 12:
                    # Perfect
                    pass
                else:
                    print(f"Skipping bad channel count after resize: {resized.shape}", flush=True)
                    continue
                    
                processed_frames.append(resized)
                
            else:
                 print(f"Warning: ndim != 3 {full.shape}, skipping.", flush=True)
                 continue
            
        # Build Memory Struct
        # We need (s, a, r, s', done, logprob)
        for i in range(len(processed_frames) - 1):
            s = processed_frames[i]
            a = actions[i]
            r = rewards[i]
            s_next = processed_frames[i+1]
            logprob = logprobs[i]
            done = False # Trajectories usually stored as continuous segments
            
            memory.append((s, a, r, s_next, done, logprob))
            
        return memory
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def train_offline():
    data_dir = "D:\\Auto-Farmer-Data\\rl_train" # New RL data
    # fallback to imitation data if rl_train is empty
    if not os.path.exists(data_dir) or not glob.glob(os.path.join(data_dir, "*.pkl")):
        data_dir = "D:\\Auto-Farmer-Data\\imitation_train"
        print(f"Using imitation data from {data_dir}")
        
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    files.sort()
    
    agent = MuZeroAgent()
    
    BATCH_SIZE = 4 # Process whole files as batches, or chunks of files
    
    print(f"Found {len(files)} trajectory files.")
    
    for epoch in range(10):
        total_loss = 0
        p_loss_total = 0
        v_loss_total = 0
        d_loss_total = 0
        
        np.random.shuffle(files)
        
        pbar = tqdm(files)
        for fpath in pbar:
            memory = load_trajectory(fpath)
            if not memory: continue
            
            # Train on this trajectory
            # We can split it into mini-batches if too large for GPU VRAM
            # Trajectory is 1024 steps. 
            # 1024 * 12 * 128 * 128 * 4 bytes ~ 800MB. Should fit on 3050/5070 easily.
            
            l, pl, vl, dl = agent.update(memory)
            
            total_loss += l
            p_loss_total += pl
            v_loss_total += vl
            d_loss_total += dl
            
            pbar.set_description(f"Loss: {l:.4f} (P={pl:.2f} V={vl:.2f} D={dl:.2f})")
            
        avg_loss = total_loss / len(files)
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        
        agent.save("muzero_agent_offline.pth")
        print("Model saved.")

if __name__ == "__main__":
    train_offline()
