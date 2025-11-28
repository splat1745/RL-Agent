from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolov11m.pt"):
        print(f"Loading Object Detector: {model_path}")
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        results = self.model(frame, verbose=False)
        # Parse results into a clean dictionary
        det_dict = {
            "enemies": [],
            "projectiles": []
        }
        
        if results:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                xywh = box.xywh[0].cpu().numpy()
                # Map classes based on your model
                # For now, assume everything is an enemy
                det_dict["enemies"].append(xywh)
                
        return det_dict
