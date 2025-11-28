import cv2
import os
import numpy as np
import glob
import torch
import torchvision
from torchvision.models.video import r3d_18, R3D_18_Weights

# Fix for OpenCV FFMPEG read attempts warning
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "20000"

def extract_frames(video_path, output_dir, fps=10):
    """
    Extracts frames from a video at a specific FPS.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Check if frames already exist
        if len(glob.glob(os.path.join(output_dir, "*.png"))) > 0:
            print(f"Frames already extracted for {video_path}. Skipping.")
            return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or np.isnan(video_fps):
        print(f"Warning: Could not read FPS for {video_path}, defaulting to 30.")
        video_fps = 30.0
        
    frame_interval = int(video_fps / fps)
    if frame_interval < 1: frame_interval = 1
    
    count = 0
    frame_id = 0
    
    print(f"Extracting frames from {video_path} at {fps} FPS...")
    
    # Try to read the first frame to check validity
    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame. Attempting to skip 10 frames...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read video {video_path} even after skipping. Skipping file.")
            cap.release()
            return
        else:
            print("Recovered by skipping frames.")
            count = 10

    while True:
        if count > 10: # If we didn't just read it above
            ret, frame = cap.read()
            if not ret:
                break
        
        if count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{frame_id:06d}.png")
            cv2.imwrite(frame_name, frame)
            frame_id += 1
            
        count += 1
        
    cap.release()
    print(f"Extracted {frame_id} frames to {output_dir}")

class VideoFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Loading R3D-18 model on {self.device}...")
        # Load pre-trained R3D-18 model
        weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=weights).to(self.device)
        self.model.eval()
        self.transform = weights.transforms()

    def extract_clip_features(self, clip_frames):
        """
        Extracts features from a clip (list of numpy arrays).
        clip_frames: List of (H, W, C) numpy arrays (BGR)
        """
        if not clip_frames:
            return None
            
        # Convert to tensor: (T, H, W, C) -> (T, C, H, W) -> (C, T, H, W)
        # TorchVision video models expect (B, C, T, H, W)
        
        # 1. Convert BGR to RGB
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip_frames]
        
        # 2. Stack and permute
        # shape: (T, H, W, C)
        tensor = torch.from_numpy(np.stack(frames_rgb))
        # shape: (T, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        
        # 3. Apply transforms (resizing, normalization)
        # The transform expects (T, C, H, W) or (C, T, H, W)?
        # R3D weights transform usually expects (C, T, H, W) uint8 [0-255]
        # But the error says "size of tensor a (16) must match size of tensor b (3) at non-singleton dimension 1"
        # This implies the transform expects (C, ...) at dim 1, but got 16 (T).
        # So input should be (..., C, T, H, W) or similar?
        
        # Let's check what we have:
        # tensor was (T, C, H, W) -> permuted to (C, T, H, W)
        # If transform expects (C, T, H, W), then dim 1 is T=16.
        # Wait, normalize usually works on C channel.
        # If tensor is (C, T, H, W), dim 0 is C=3, dim 1 is T=16.
        
        # The error "size of tensor a (16) must match size of tensor b (3) at non-singleton dimension 1"
        # suggests that 'a' has 16 at dim 1, and 'b' (mean/std) has 3.
        # If mean/std are broadcastable, they usually match C.
        # If the tensor is (C, T, H, W), then dim 0 is 3.
        # If the tensor is (T, C, H, W), then dim 1 is 3.
        
        # Let's look at the permute again.
        # tensor = tensor.permute(1, 0, 2, 3) # (C, T, H, W)
        
        # If the transform expects (T, C, H, W), then we shouldn't permute yet?
        # VideoClassification preset usually expects (T, C, H, W) or (C, T, H, W).
        # Let's try passing (T, C, H, W) directly.
        
        # Revert permute
        tensor = tensor.permute(1, 0, 2, 3) # Back to (T, C, H, W)
        
        # Actually, let's just print shape to be sure if we could.
        # But let's try standard (C, T, H, W) which is standard for 3D CNNs, 
        # BUT the transform might be doing something else.
        
        # The error says dim 1 is 16.
        # If we passed (C, T, H, W) -> (3, 16, H, W). Dim 1 is 16.
        # Normalization vector is size 3.
        # It tries to match dim 1. So it expects C at dim 1.
        # So it expects (B, C, T, H, W) or (T, C, H, W)?
        
        # If it expects C at dim 1, then (3, 16, H, W) has 16 at dim 1.
        # So it thinks 16 is the channel dimension? No, it thinks dim 1 is the channel dimension.
        # So we provided (3, 16, H, W).
        # It wants (..., 3, ...).
        
        # If we provide (T, C, H, W) -> (16, 3, H, W). Dim 1 is 3.
        # This matches!
        
        # So we should NOT permute to (C, T, H, W) before transform.
        # We should pass (T, C, H, W).
        
        # tensor is currently (C, T, H, W) because of line 104.
        # Let's undo that.
        tensor = tensor.permute(1, 0, 2, 3) # Back to (T, C, H, W)
        
        # Ensure it is uint8 (0-255)
        tensor = tensor.to(torch.uint8)
        
        tensor = self.transform(tensor)
        
        # Now tensor is likely (C, T, H, W) float normalized.
        
        # Add batch dimension: (1, C, T, H, W)
        input_tensor = tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # We want embeddings, not class logits.
            # R3D-18 structure: stem -> layer1 -> ... -> layer4 -> avgpool -> fc
            # We can hook or just replace fc?
            # Or just use the output if we want class probabilities (Action Recognition)
            # For "features", usually we want the output of avgpool.
            
            # Let's forward pass until avgpool
            x = self.model.stem(input_tensor)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            # Flatten
            features = x.flatten(1)
            
        return features.cpu().numpy()

def process_video_clips(frames_dir, output_dir, clip_length=16):
    """
    Groups frames into clips and extracts 3D features.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Check if already processed
    if len(glob.glob(os.path.join(output_dir, "*.npy"))) > 0:
        print(f"Clips already processed for {frames_dir}. Skipping.")
        return

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if len(frame_files) < clip_length:
        print(f"Not enough frames for a clip in {frames_dir}")
        return

    print(f"Processing {len(frame_files)} frames into clips of length {clip_length}...")
    
    extractor = VideoFeatureExtractor()
    
    clip_buffer = []
    clip_idx = 0
    
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is None: continue
        
        clip_buffer.append(frame)
        
        if len(clip_buffer) == clip_length:
            # Extract features
            features = extractor.extract_clip_features(clip_buffer)
            
            if features is not None:
                # Save
                save_path = os.path.join(output_dir, f"clip_{clip_idx:04d}.npy")
                np.save(save_path, features)
            
            # Sliding window? Or non-overlapping?
            # Prompt says "Input N-frame clips". Usually non-overlapping for dataset creation.
            # Let's do non-overlapping for now.
            clip_buffer = [] 
            clip_idx += 1
            
            if clip_idx % 10 == 0:
                print(f"Processed {clip_idx} clips...")

def main():
    # 1. Process Videos
    # Input videos (Check T: drive first, then local)
    video_dir = r"T:\Auto-Farmer-Data\raw_videos"
    if not os.path.exists(video_dir):
        video_dir = "data/raw_videos"

    # Output Directories (Use T: drive for massive storage)
    base_storage = r"T:\Auto-Farmer-Data"
    if not os.path.exists(base_storage):
        os.makedirs(base_storage)
        
    frames_base_dir = os.path.join(base_storage, "frames")
    preproc_dir = os.path.join(base_storage, "preproc")
    
    print(f"Input Videos: {video_dir}")
    print(f"Output Frames: {frames_base_dir}")
    print(f"Output Preproc: {preproc_dir}")
    
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        session_dir = os.path.join(frames_base_dir, video_name)
        session_preproc_dir = os.path.join(preproc_dir, video_name)
        
        # Step 3: Extract Frames
        extract_frames(video_path, session_dir, fps=15)
        
        # Step 4: Preprocessing (Video Clips -> 3D Features)
        process_video_clips(session_dir, session_preproc_dir, clip_length=16)

if __name__ == "__main__":
    main()
