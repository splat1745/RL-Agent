import pickle
import os
import glob

data_dir = "D:\\Auto-Farmer-Data\\imitation_train"
files = glob.glob(os.path.join(data_dir, "*.pkl"))

if not files:
    print("No files found.")
else:
    fpath = files[0]
    print(f"Inspecting {fpath}...")
    try:
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            print(f"Keys in first element: {list(data[0].keys())}")
        else:
            print("Data is not a non-empty list.")
            
    except Exception as e:
        print(f"Error: {e}")
