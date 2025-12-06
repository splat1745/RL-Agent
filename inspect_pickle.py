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
        
        print(f"Type of data: {type(data)}")
        if isinstance(data, list):
            print(f"Length of data: {len(data)}")
            if len(data) > 0:
                print(f"First element type: {type(data[0])}")
                print(f"First element: {data[0]}")
        elif isinstance(data, dict):
            print(f"Keys: {data.keys()}")
        else:
            print(f"Data: {data}")
            
    except Exception as e:
        print(f"Error: {e}")
