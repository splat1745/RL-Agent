import cv2
import os
import argparse

def extract_frames(video_path, output_dir, interval=None, count=None):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames.")

    if count is not None:
        interval = max(1, total_frames // count)
        print(f"Targeting {count} frames. Calculated interval: {interval}")
    elif interval is None:
        interval = 30 # Default
        print(f"Using default interval: {interval}")
    else:
        print(f"Using specified interval: {interval}")

    frame_count = 0
    saved_count = 0
    
    print(f"Extracting frames from {video_path} to {output_dir}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % interval == 0:
            # Generate filename
            filename = f"frame_{saved_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            print(f"Saved {filename}", end='\r')
            
            if count is not None and saved_count >= count:
                break
            
        frame_count += 1
        
    cap.release()
    print(f"\nDone. Saved {saved_count} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for RoboFlow.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", default=r"T:\robo_flow", help="Output directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--interval", type=int, help="Save every Nth frame")
    group.add_argument("--count", type=int, help="Total number of frames to extract")
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output, args.interval, args.count)
