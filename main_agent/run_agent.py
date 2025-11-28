import time
import cv2
import numpy as np
import torch

# Import sensors
from main_agent.sensors.object_detection.detector import ObjectDetector
from main_agent.sensors.pose_estimation.pose_estimator import PoseEstimator
from main_agent.sensors.combat_state.inference import CombatStateInference
from main_agent.utils.observation import ObservationBuilder

# Import capture (reusing existing)
from capture.capture import capture_service

def main():
    print("--- Advanced RL Agent (Step 15) ---")
    
    # 1. Initialize Sensors
    detector = ObjectDetector() # Step 15.2
    pose_estimator = PoseEstimator() # Step 15.3
    combat_inference = CombatStateInference() # Step 15.4
    obs_builder = ObservationBuilder() # Step 15.5
    
    # 2. Initialize Policy (Placeholder)
    # policy = torch.load("main_agent/policy/distilled_model.pth")
    
    # 3. Start Capture
    capture_service.select_window()
    capture_service.start()
    time.sleep(1)
    
    print("Starting Run Loop...")
    try:
        while True:
            # 1. Capture frame
            frame = capture_service.get_latest_frame()
            if frame is None:
                continue
                
            # 2. Run object detection
            det_dict = detector.detect(frame)
            
            # 3. Run pose estimation
            pose_data = pose_estimator.estimate(frame)
            
            # 4. Derive combat state
            combat_state = combat_inference.infer(pose_data)
            
            # 5. Build observation vector
            obs = obs_builder.build(det_dict, pose_data, combat_state)
            
            # 6. Feed into policy
            # action = policy(obs)
            action = "idle" # Placeholder
            
            # 7. Get action & 8. Send input
            # controller.execute(action)
            
            # Visualization
            cv2.putText(frame, f"State: {combat_state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Agent View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        capture_service.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
