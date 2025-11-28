import cv2
import os
import argparse

def extract_frames(video_path, output_dir, target_fps=None):
    """
    Extracts frames from a video file at a specific FPS.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video: {video_path}")
    print(f"Original FPS: {video_fps:.2f}")
    print(f"Duration: {duration:.2f}s")
    
    if target_fps is None:
        target_fps = video_fps
        
    if target_fps > 30:
        print("Warning: Target FPS > 30. Clamping to 30 as per requirements.")
        target_fps = 30
        
    step = video_fps / target_fps
    
    print(f"Extracting at {target_fps} FPS (Step: {step:.2f} frames)...")
    
    current_frame = 0
    saved_count = 0
    next_save_frame = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame >= int(next_save_frame):
            # Save frame
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            next_save_frame += step
            
        current_frame += 1
        
        if current_frame % 100 == 0:
            print(f"Processed {current_frame}/{total_frames} frames...")

    cap.release()
    print(f"Done. Saved {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video at custom FPS")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", "-o", default="extracted_frames", help="Output directory")
    parser.add_argument("--fps", "-f", type=float, default=5.0, help="Target FPS to extract (max 30)")
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output, args.fps)
