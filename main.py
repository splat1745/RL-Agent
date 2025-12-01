import time
import sys
import argparse
import cv2
import numpy as np
import torch
from collections import deque
from capture.capture import capture_service
from detection.inference import init_perception
from rl.agent import PPOAgent
from rl.memory import Memory
from control.actions import get_action_name, ACTION_MAP
from control.keyboard_mouse import InputController
from utils.reward import calculate_reward
from utils.visualization import draw_detections
from utils.config_wizard import ConfigWizard, CONFIG_FILE

from utils.state import StateManager

from control.policy import DirectPolicy

def preview_detection(perception, capture_service):
    print("Starting Preview. Press 's' to start Agent, 'd' for Direct Policy, 'p' to quit.")
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    while True:
        loop_start = time.time()
        frame = capture_service.get_latest_frame()
        if frame is not None:
            # Detect
            det_dict = perception.detect(frame)
            # Preprocess for vis
            vis_frame = perception.preprocess(frame)
            if vis_frame is not None:
                draw_detections(vis_frame, det_dict)
                
                # FPS Calculation
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                cv2.putText(vis_frame, f"FPS: {fps}", (vis_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.imshow("Preview (Press 's' to Start)", vis_frame)
                
                # ROI Preview
                debug_imgs = []
                target_h = 300 # Increased from 100 for better visibility
                
                if hasattr(perception, 'debug_health') and perception.debug_health is not None:
                    # Resize health to fixed height?
                    h, w = perception.debug_health.shape[:2]
                    scale = target_h / h
                    dim = (int(w * scale), target_h)
                    resized = cv2.resize(perception.debug_health, dim)
                    debug_imgs.append(resized)
                    
                if hasattr(perception, 'debug_rois'):
                    for tid, img in perception.debug_rois.items():
                        h, w = img.shape[:2]
                        scale = target_h / h
                        dim = (int(w * scale), target_h)
                        resized = cv2.resize(img, dim)
                        debug_imgs.append(resized)
                    
                if debug_imgs:
                    # Stack
                    roi_vis = np.hstack(debug_imgs)
                    cv2.imshow("ROI Logic Preview", roi_vis)
                else:
                    # If no debug images, maybe clear the window or show black
                    # To avoid stale window
                    try:
                        cv2.destroyWindow("ROI Logic Preview")
                    except:
                        pass
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Starting Agent...")
            cv2.destroyWindow("Preview (Press 's' to Start)")
            try:
                cv2.destroyWindow("ROI Logic Preview")
            except:
                pass
            return "agent"
        elif key == ord('d'):
            print("Starting Direct Policy...")
            cv2.destroyWindow("Preview (Press 's' to Start)")
            try:
                cv2.destroyWindow("ROI Logic Preview")
            except:
                pass
            return "direct"
        elif key == ord('p'):
            print("Quitting...")
            capture_service.stop()
            exit()
            
        # No sleep for max FPS


import os
import threading
import queue

# Shared state for threading
frame_queue = queue.Queue(maxsize=1)
obs_queue = queue.Queue(maxsize=1)
status_queue = queue.Queue(maxsize=1) # (action_name, reward)
stop_event = threading.Event()

def direct_policy_loop(policy, controller, state_manager):
    """
    Runs the Direct Policy logic in a separate thread.
    """
    print("Direct Policy Thread Started.")
    time_step = 0
    
    while not stop_event.is_set():
        try:
            # Get latest observation (blocking with timeout)
            obs_data = obs_queue.get(timeout=0.1)
            pixel_obs, vector_obs = obs_data
        except queue.Empty:
            continue
            
        # Policy -> Action
        action_idx = policy.select_action(vector_obs)
        action_name = get_action_name(action_idx)
        
        # Debug Print
        print(f"Direct Step {time_step}: Action={action_name}")
        
        # Execute Action
        controller.execute(action_name, duration=0.1) # Faster updates for direct control
        
        # Calculate Reward (Just for logging)
        reward = calculate_reward(vector_obs, action_idx, state_manager)
        
        # Update Status
        if status_queue.full():
            try:
                status_queue.get_nowait()
            except queue.Empty:
                pass
        status_queue.put((action_name, reward, None))
        
        time_step += 1

def agent_loop(agent, memory, controller, state_manager, update_timestep, model_path):
    """
    Runs the RL Agent logic in a separate thread.
    """
    print("Agent Thread Started.")
    time_step = 0
    running_reward = 0
    hit_history_batch = []
    
    # No more obs_stack here, handled in inference.py
    
    while not stop_event.is_set():
        try:
            # Get latest observation (blocking with timeout)
            # Queue now returns (pixel_obs, vector_obs)
            obs_data = obs_queue.get(timeout=0.1)
            pixel_obs, vector_obs = obs_data
        except queue.Empty:
            continue
            
        # 3. Agent -> Action
        # pixel_obs is a dict {'full': ..., 'crop': ..., 'flow': ...}
        # vector_obs is np.array [36]
        action_idx, log_prob, intention = agent.select_action(pixel_obs, vector_obs)
        action_name = get_action_name(action_idx)
        
        # Auto-Lock Feature REMOVED
        # "Remove autolock from everything. direct policy should not be accessable to RL agent."
        
        # Debug Print
        print(f"Agent Step {time_step}: Action={action_name}")
        
        # 4. Execute Action (Blocking for duration)
        controller.execute(action_name, duration=0.3)
        
        # 5. Calculate Reward (Use vector_obs for logic)
        last_hit_len = len(state_manager.hit_history)
        reward = calculate_reward(vector_obs, action_idx, state_manager)
        was_hit = len(state_manager.hit_history) > last_hit_len
        hit_history_batch.append(was_hit)
        
        # Update Status for Main Thread
        if status_queue.full():
            try:
                status_queue.get_nowait()
            except queue.Empty:
                pass
        status_queue.put((action_name, reward, intention))
        
        # 6. Store in Memory
        # Store tuple of (pixel_obs, vector_obs)
        memory.states.append((pixel_obs, vector_obs))
        memory.actions.append(action_idx)
        memory.logprobs.append(log_prob)
        memory.rewards.append(reward)
        memory.is_terminals.append(False) # Never terminal for now
        
        time_step += 1
        running_reward += reward
        
        # 7. Update Policy
        if time_step % update_timestep == 0:
            print(f"Updating Policy... Avg Reward: {running_reward/update_timestep:.2f}")
            agent.update(memory, hit_history=hit_history_batch)
            memory.clear()
            hit_history_batch = []
            # Reset hidden state after update? Or keep it?
            # Usually reset for batch updates if not using TBPTT
            # agent.reset_hidden() # We now persist hidden state in agent.update logic
            running_reward = 0
            
            # Save Model
            print(f"Saving model to {model_path}...")
            agent.save(model_path)

def main():
    print("--- Auto-Farmer RL Agent ---")
    
    # Parse args with argparse
    parser = argparse.ArgumentParser(description="Auto-Farmer RL Agent")
    parser.add_argument("--setup", "-s", action="store_true", help="Force run the setup wizard")
    parser.add_argument("--model", "-m", type=str, default=None, 
                        help="Path to detection model (.pt for YOLO, .pth for RF-DETR)")
    args = parser.parse_args()
    
    force_setup = args.setup
    detection_model_path = args.model
    
    if detection_model_path:
        print(f"Using detection model: {detection_model_path}")

    # 1. Select Window
    # capture_service.select_window()
    print("Auto-selecting 'Roblox' window...")
    capture_service.window_title = "Roblox"
    
    # Start Capture early to check resolution
    print("Starting Screen Capture...")
    capture_service.start()
    time.sleep(1.0) # Warmup
    
    # Check Config
    wizard = ConfigWizard(capture_service)
    if force_setup or not wizard.validate_config():
        print("Starting setup wizard...")
        wizard.select_health_bar()
    
    # 2. Initialize Components
    print("Initializing Perception...")
    perception = init_perception(detection_model_path)
    
    print("Initializing Agent...")
    # Action Dim = 26 (Updated)
    # 0-14: Basic + Look X
    # 15: r_2
    # 16: g
    # 17: space (Jump)
    # 18: f (Block)
    # 19: m1 (Single Click)
    # 20: turn_left_micro
    # 21: turn_right_micro
    # 22: turn_left_small
    # 23: turn_right_small
    # 24: turn_left_large
    # 25: turn_right_large
    agent = PPOAgent(action_dim=26)
    
    # Load existing model
    model_path = "ppo_agent_pixel.pth" # Changed name to avoid conflict
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            # agent.load(model_path)
            print("Skipping load to reset behavior loop.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")
            
    memory = Memory()
    
    print("Initializing Controls...")
    controller = InputController()
    state_manager = StateManager()
    
    # Focus the game window
    print("Focusing Game Window...")
    capture_service.focus()
    time.sleep(0.3)
    
    # 4. Preview Mode
    mode = preview_detection(perception, capture_service)
    
    if mode == "agent":
        # 5. Start Agent Thread
        agent_thread = threading.Thread(target=agent_loop, args=(agent, memory, controller, state_manager, 10, model_path))
        agent_thread.daemon = True
        agent_thread.start()
    elif mode == "direct":
        # 5. Start Direct Policy Thread
        policy = DirectPolicy()
        agent_thread = threading.Thread(target=direct_policy_loop, args=(policy, controller, state_manager))
        agent_thread.daemon = True
        agent_thread.start()
    
    print("Starting Visualization Loop. Press 'p' to quit.")
    
    # Visualization State
    current_action = "idle"
    current_reward = 0.0
    current_intention = None
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while True:
            loop_start = time.time()
            # 1. Get Frame (Fast)
            frame = capture_service.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue
                
            # 2. Perception
            # Get mouse movement
            mdx, mdy = controller.get_recent_movement()
            
            # Get Vector Obs (for Reward) - This runs detection
            vector_obs = perception.get_obs(frame, last_action=current_action, mouse_movement=(mdx, mdy))
            
            # Get Pixel Obs (for Agent) - This uses cached detection
            pixel_obs = perception.get_pixel_obs(frame)
            
            if pixel_obs is None: continue
            
            # Feed Agent (Non-blocking put)
            if obs_queue.full():
                try:
                    obs_queue.get_nowait() # Discard old
                except queue.Empty:
                    pass
            obs_queue.put((pixel_obs, vector_obs))
            
            # Check for status updates from Agent
            try:
                current_action, current_reward, current_intention = status_queue.get_nowait()
            except queue.Empty:
                pass
            
            # 3. Visualization
            # Draw every frame or every few frames
            vis_frame = perception.preprocess(frame)
            if vis_frame is not None:
                if hasattr(perception, 'last_det'):
                    draw_detections(vis_frame, perception.last_det)
                
                # Draw status
                cv2.putText(vis_frame, f"Action: {current_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Reward: {current_reward:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw Intention Preview
                if current_intention is not None:
                    # Intention is [1, 384]
                    # Visualize as a small heatmap bar
                    intent_vis = current_intention.reshape(12, 32) # 12x32 grid
                    # Normalize for vis
                    intent_vis = (intent_vis - intent_vis.min()) / (intent_vis.max() - intent_vis.min() + 1e-6)
                    intent_vis = (intent_vis * 255).astype(np.uint8)
                    intent_vis = cv2.applyColorMap(intent_vis, cv2.COLORMAP_JET)
                    intent_vis = cv2.resize(intent_vis, (160, 80), interpolation=cv2.INTER_NEAREST)
                    
                    # Add Grid Lines
                    # Rows: 12
                    step_y = 80 / 12
                    for i in range(1, 12):
                        y = int(i * step_y)
                        cv2.line(intent_vis, (0, y), (160, y), (0,0,0), 1)
                    
                    # Cols: 32
                    step_x = 160 / 32
                    for i in range(1, 32):
                        x = int(i * step_x)
                        cv2.line(intent_vis, (x, 0), (x, 80), (0,0,0), 1)
                    
                    # Overlay on frame
                    h, w = vis_frame.shape[:2]
                    vis_frame[h-90:h-10, 10:170] = intent_vis
                    cv2.putText(vis_frame, "Intention Map", (10, h-95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # FPS Calculation
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                cv2.putText(vis_frame, f"FPS: {fps}", (vis_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow("Agent View (High FPS)", vis_frame)
                
                # Show Network Inputs
                net_vis = perception.visualize_obs()
                cv2.imshow("Network Inputs (Full, Crop, Flow)", net_vis)
                
                # ROI Preview (Agent Mode)
                debug_imgs = []
                target_h = 50 # Reduced from 300
                
                if hasattr(perception, 'debug_health') and perception.debug_health is not None:
                    # Explicitly cast shape to tuple to avoid type checker issues
                    shape = perception.debug_health.shape
                    h, w = shape[0], shape[1]
                    scale = target_h / h
                    dim = (int(w * scale), target_h)
                    resized = cv2.resize(perception.debug_health, dim)
                    debug_imgs.append(resized)
                    
                if hasattr(perception, 'debug_rois'):
                    for tid, img in perception.debug_rois.items():
                        shape = img.shape
                        h, w = shape[0], shape[1]
                        scale = target_h / h
                        dim = (int(w * scale), target_h)
                        resized = cv2.resize(img, dim)
                        debug_imgs.append(resized)
                    
                if debug_imgs:
                    roi_vis = np.hstack(debug_imgs)
                    cv2.imshow("ROI Logic Preview", roi_vis)
            
            if cv2.waitKey(1) & 0xFF == ord('p'):
                stop_event.set()
                break
            
            # No sleep for max FPS
            # elapsed = time.time() - loop_start
            # if elapsed < 0.016:
            #     time.sleep(0.016 - elapsed)

    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
    finally:
        capture_service.stop()
        cv2.destroyAllWindows()
        print("Cleaned up.")

if __name__ == "__main__":
    main()
