from ultralytics import YOLO
import torch

def train_model():
    # 1. Configuration
    # Use 'yolo11n.pt' for speed, 'yolo11s.pt' for better accuracy
    model_name = "yolo11s.pt" 
    data_yaml = "datasets/data.yaml" # You get this file after labeling
    epochs = 50 # 50-100 is usually good for small datasets
    img_size = 640
    batch_size = 16 # Reduce if you run out of GPU memory
    
    print(f"Starting training with {model_name} on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")

    # 2. Load Model
    model = YOLO(model_name)

    # 3. Train
    # This will automatically download the model if not present
    # and save results to 'runs/detect/trainX'
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        plots=True,       # Save training plots
        half=True,        # Use FP16
        augment=True,     # Use data augmentation (flips, color changes)
        workers=4
    )
    
    print("Training complete!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print("Update your perception.py MODEL_PATH to point to this file.")

if __name__ == "__main__":
    # Ensure we have the data.yaml before running
    import os
    if not os.path.exists("datasets/data.yaml"):
        print("ERROR: 'datasets/data.yaml' not found.")
        print("Please follow the labeling steps to export your dataset first.")
    else:
        train_model()
