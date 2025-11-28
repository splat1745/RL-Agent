import time
import numpy as np
import cv2
import torch
from detection.inference import init_perception

def run_benchmark(duration=10.0):
    print("Initializing Benchmark...")
    
    # Initialize Pipeline
    pipeline = init_perception()
    
    # Create Dummy Frames (Random noise to simulate texture for flow)
    # 1920x1080 is typical game resolution
    frames = []
    print("Generating dummy frames...")
    for i in range(60):
        # Random noise
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        # Add a moving rectangle to simulate tracking
        cv2.rectangle(frame, (100 + i*5, 100 + i*2), (200 + i*5, 300 + i*2), (0, 255, 0), -1)
        frames.append(frame)
        
    print(f"Starting Benchmark Loop for {duration} seconds...")
    
    start_time = time.time()
    frame_count = 0
    
    times = {
        "total": [],
        "get_obs": [],
        "pixel_obs": []
    }
    
    while time.time() - start_time < duration:
        frame_idx = frame_count % len(frames)
        frame = frames[frame_idx]
        
        t0 = time.time()
        
        # 1. Vector Obs
        obs = pipeline.get_obs(frame)
        t1 = time.time()
        
        # 2. Pixel Obs
        pixel_obs = pipeline.get_pixel_obs(frame)
        t2 = time.time()
        
        times["get_obs"].append(t1 - t0)
        times["pixel_obs"].append(t2 - t1)
        times["total"].append(t2 - t0)
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: FPS={1.0/(t2-t0):.2f}", end='\r')
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = frame_count / total_time
    
    print("\n\n--- Benchmark Results ---")
    print(f"Total Frames: {frame_count}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print("-" * 30)
    print(f"Avg get_obs (Detection + Logic): {np.mean(times['get_obs'])*1000:.2f} ms")
    print(f"Avg pixel_obs (Crop + Flow):     {np.mean(times['pixel_obs'])*1000:.2f} ms")
    print(f"Avg Total Latency:               {np.mean(times['total'])*1000:.2f} ms")
    print("-" * 30)
    
    # Resource Usage Hint
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    run_benchmark()
