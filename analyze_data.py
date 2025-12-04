import glob
import pickle
import os
import numpy as np
from collections import Counter

def analyze_data(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    print(f"Found {len(files)} files.")
    
    all_actions = []
    
    for f_path in files:
        try:
            with open(f_path, 'rb') as f:
                try:
                    session_data = pickle.load(f)
                except (EOFError, pickle.UnpicklingError):
                    print(f"Skipping corrupted file: {f_path}")
                    continue
                    
                if not session_data: continue
                
                actions = [s['action'] for s in session_data]
                all_actions.extend(actions)
        except Exception as e:
            print(f"Error {f_path}: {e}")
            
    if not all_actions:
        print("No actions found.")
        return

    counter = Counter(all_actions)
    total = len(all_actions)
    
    print(f"Total Samples: {total}")
    print("Action Distribution:")
    
    # Sort by action ID
    for action_id in sorted(counter.keys()):
        count = counter[action_id]
        pct = (count / total) * 100
        print(f"Action {action_id}: {count} ({pct:.2f}%)")

    # Suggest Weights
    print("\nSuggested Class Weights (Inverse Frequency):")
    weights = {}
    for action_id, count in counter.items():
        weights[action_id] = total / (count * len(counter))
    
    for action_id in sorted(weights.keys()):
        print(f"Action {action_id}: {weights[action_id]:.4f}")

if __name__ == "__main__":
    analyze_data(r"D:\Auto-Farmer-Data\imitation_train")
