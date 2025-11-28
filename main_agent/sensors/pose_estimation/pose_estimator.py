from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob

class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt"):
        print(f"Loading Pose Model: {model_path}")
        self.model = YOLO(model_path)
        
    def estimate(self, frame):
        """
        Runs pose estimation on a frame.
        Returns keypoints: [N, 17, 3] (x, y, conf)
        """
        results = self.model(frame, verbose=False)
        if results and results[0].keypoints is not None:
            # keypoints.data is [N, 17, 3] (x, y, conf)
            return results[0].keypoints.data.cpu().numpy()
        return None

def process_poses(frames_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    estimator = PoseEstimator()
    
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    print(f"Estimating poses for {len(frame_files)} frames...")
    
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        keypoints = estimator.estimate(frame)
        
        if keypoints is not None:
            base_name = os.path.basename(frame_file).replace(".png", "_pose.npy")
            np.save(os.path.join(output_dir, base_name), keypoints)
            
        if i % 100 == 0:
            print(f"Processed {i} frames...")

if __name__ == "__main__":
    # Example usage
    # process_poses("data/frames/session_1", "data/poses")
    pass
