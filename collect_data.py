import cv2
import os
import time
import uuid
from capture import capture_service

# Configuration
SAVE_DIR = "datasets/raw_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print("--- Data Collection Tool ---")
    print(f"Images will be saved to: {os.path.abspath(SAVE_DIR)}")
    
    # 1. Select Window
    capture_service.select_window()
    capture_service.start()
    
    print("\nControls:")
    print("  [C] - Capture current frame")
    print("  [A] - Auto-capture mode (toggle 1 frame/sec)")
    print("  [Q] - Quit")
    
    # Create a resizable window and set a reasonable default size
    cv2.namedWindow("Data Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Data Collector", 1280, 720)
    
    auto_capture = False
    last_capture_time = time.time()
    count = 0
    
    try:
        while True:
            frame = capture_service.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Display
            cv2.imshow("Data Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Manual Capture
            if key == ord('c'):
                filename = f"{SAVE_DIR}/img_{uuid.uuid4().hex[:8]}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                count += 1
            
            # Toggle Auto Capture
            elif key == ord('a'):
                auto_capture = not auto_capture
                print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
            
            # Quit
            elif key == ord('q'):
                break
            
            # Auto Capture Logic
            if auto_capture:
                if time.time() - last_capture_time >= 1.0: # 1 second interval
                    filename = f"{SAVE_DIR}/auto_{uuid.uuid4().hex[:8]}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Auto-Saved: {filename}")
                    count += 1
                    last_capture_time = time.time()
                    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        capture_service.stop()
        cv2.destroyAllWindows()
        print(f"Session ended. Total images captured: {count}")

if __name__ == "__main__":
    main()
