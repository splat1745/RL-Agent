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
from control.input_listener import InputListener
import datetime
import pickle


def preview_detection(perception, capture_service):
    print("Starting Preview. Press 's' to start Agent, 't' for Temporal Agent, 'd' for Direct Policy, 'i' for Imitation Mode, 'p' to quit.")
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
        elif key == ord('t'):
            print("Starting Temporal Agent (30-frame context)...")
            cv2.destroyWindow("Preview (Press 's' to Start)")
            try:
                cv2.destroyWindow("ROI Logic Preview")
            except:
                pass
            return "temporal"
        elif key == ord('d'):
            print("Starting Direct Policy...")
            cv2.destroyWindow("Preview (Press 's' to Start)")
            try:
                cv2.destroyWindow("ROI Logic Preview")
            except:
                pass
            return "direct"
        elif key == ord('i'):
            print("Starting Imitation Mode...")
            cv2.destroyWindow("Preview (Press 's' to Start)")
            try:
                cv2.destroyWindow("ROI Logic Preview")
            except:
                pass
            return "imitation"
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
            if len(obs_data) == 3:
                pixel_obs, vector_obs, _ = obs_data
            else:
                pixel_obs, vector_obs = obs_data
        except queue.Empty:
            continue
            
        # Policy -> Action
        action_idx = policy.select_action(vector_obs)
        action_name = get_action_name(action_idx)
        
        # Execute
        controller.execute(action_name, duration=0.1)
        time_step += 1

def agent_loop_muzero(agent, controller, state_manager, perception, cooldown_detector, save_interval, model_path):
    """
    MuZero online training loop.
    Handles frame stacking, action selection, transition storage, and periodic training.
    Uses perception for reward calculation and cooldown_detector for move availability.
    """
    from rl.rewards import RewardCalculator
    
    print("MuZero Agent Loop Started.")
    
    reward_calc = RewardCalculator()
    if cooldown_detector:
        reward_calc.cooldown_detector = cooldown_detector
    
    frame_stack = deque(maxlen=4)
    time_step = 0
    
    # Initial Frame Stack - wait for frames
    print("Filling frame buffer...")
    while len(frame_stack) < 4:
        frame = capture_service.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Preprocess: resize to 128x128, transpose to (C, H, W)
        resized = cv2.resize(frame, (128, 128))
        transposed = np.transpose(resized, (2, 0, 1))  # (3, 128, 128)
        frame_stack.append(transposed)
        time.sleep(0.05)
    
    print("Buffer filled. Starting interaction...")
    
    while not stop_event.is_set():
        try:
            loop_start = time.time()
            
            # 1. State: stack 4 frames -> (12, 128, 128)
            current_stack = np.concatenate(list(frame_stack), axis=0)  # (12, 128, 128)
            
            # 2. Act with epsilon-greedy exploration
            epsilon = max(0.1, 1.0 - time_step / 10000.0)  # Decay epsilon
            
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(0, agent.action_dim)
            else:
                action_idx = agent.select_action(current_stack)
            
            action_name = get_action_name(action_idx)
            
            # 3. Execute action
            controller.execute(action_name, duration=0.1)
            
            # 4. Observe next state
            time.sleep(0.05)  # Wait for effect
            frame = capture_service.get_latest_frame()
            if frame is None:
                continue
            
            # Run perception for reward calculation
            try:
                perception.detect(frame)  # Updates internal state (HP, detections)
            except:
                pass
            
            # Update cooldown status
            if cooldown_detector:
                cooldown_detector.update(frame)
            
            # Calculate reward using perception-based heuristics
            reward = reward_calc.calculate(perception, action_idx)
            
            # Preprocess next frame
            resized = cv2.resize(frame, (128, 128))
            next_frame_transposed = np.transpose(resized, (2, 0, 1))
            
            # Build next stack
            next_stack_list = list(frame_stack)[1:] + [next_frame_transposed]
            next_stack = np.concatenate(next_stack_list, axis=0)
            
            done = False
            
            # 5. Store transition
            agent.store_transition(current_stack, action_idx, reward, next_stack, done, 0.0)
            
            # 6. Train periodically
            if time_step > 100 and time_step % 64 == 0:
                try:
                    loss, pl, vl, dl = agent.update_with_buffer(batch_size=8)
                    if status_queue.full():
                        status_queue.get_nowait()
                    status_queue.put((f"Train L={loss:.2f}", reward, None))
                except Exception as e:
                    print(f"Training error: {e}")
            elif time_step % 10 == 0:
                if status_queue.full():
                    try:
                        status_queue.get_nowait()
                    except:
                        pass
                status_queue.put((f"Act {action_name}", reward, None))
            
            # Update frame stack
            frame_stack.append(next_frame_transposed)
            time_step += 1
            
            # Save periodically
            if time_step % save_interval == 0:
                agent.save(model_path)
                print(f"Online Model Saved at step {time_step}.")
                
        except Exception as e:
            print(f"Error in Agent Loop: {e}")
            time.sleep(0.5)
    
    print("MuZero Agent Loop Ended.")

def temporal_agent_loop(agent, controller, perception, cooldown_detector, save_interval, model_path, training_stage=2):
    """
    Temporal agent loop with 30-frame context and safety override.
    Uses multihead predictions for policy, HP danger, and action sequences.
    """
    print(f"Temporal Agent Loop Started. Training Stage: {training_stage}")
    
    # Import reward calculator
    from rl.rewards import RewardCalculator
    reward_calc = RewardCalculator()
    
    agent.set_stage(training_stage)
    agent.reset_episode()
    
    time_step = 0
    last_hp = 1.0
    train_interval = 64
    
    # Fill initial buffers
    print("Filling 30-frame buffer...")
    while not agent.is_ready():
        frame = capture_service.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        agent.observe(frame, reward=0.0, hp_delta=0.0)
        time.sleep(0.03)  # ~30fps
    
    print("Buffer filled. Starting interaction...")
    
    while not stop_event.is_set():
        try:
            # 1. Get current frame
            frame = capture_service.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # 2. Run perception
            try:
                perception.detect(frame)
            except:
                pass
            
            # 3. Get cooldown state
            cooldowns = np.array([0.0, 0.0, 0.0, 0.0])
            if cooldown_detector:
                cooldown_detector.update(frame)
                cooldowns = np.array([
                    1.0 if not cooldown_detector.is_move_available(i) else 0.0 
                    for i in range(4)
                ])
            
            # 4. Calculate epsilon based on training progress
            epsilon = max(0.05, 0.5 - agent.train_steps / 50000.0)
            
            # 5. Select action (with safety override)
            action_idx, action_info = agent.select_action(cooldowns, epsilon=epsilon)
            action_name = get_action_name(action_idx)
            
            # 6. Execute action
            controller.execute(action_name, duration=0.1)
            
            # 7. Wait and observe next frame
            time.sleep(0.03)
            next_frame = capture_service.get_latest_frame()
            if next_frame is None:
                continue
            
            # 8. Calculate reward
            try:
                perception.detect(next_frame)
            except:
                pass
            
            current_hp = getattr(perception, 'filtered_health', 1.0)
            hp_delta = current_hp - last_hp
            reward = reward_calc.calculate(perception, action_idx)
            last_hp = current_hp
            
            # 9. Observe for agent buffer
            agent.observe(next_frame, reward=reward, hp_delta=hp_delta)
            
            # 10. Store sequence periodically
            if time_step % 30 == 0:
                agent.store_sequence(cooldowns, done=False)
            
            # 11. Train
            if time_step > 100 and time_step % train_interval == 0:
                losses = agent.train_step(batch_size=8)
                if losses:
                    loss_str = " ".join([f"{k}:{v:.3f}" for k, v in losses.items()])
                    if status_queue.full():
                        try:
                            status_queue.get_nowait()
                        except:
                            pass
                    danger = action_info.get('danger', 0)
                    status_queue.put((f"Train {loss_str}", reward, danger))
            elif time_step % 10 == 0:
                if status_queue.full():
                    try:
                        status_queue.get_nowait()
                    except:
                        pass
                danger = action_info.get('danger', 0)
                status_queue.put((f"Act {action_name} D={danger:.2f}", reward, danger))
            
            time_step += 1
            
            # 12. Save periodically
            if time_step % save_interval == 0:
                agent.save(model_path)
                print(f"Temporal Model Saved at step {time_step}.")
            
            # Auto-shutdown for Stage 2
            if training_stage == 2 and agent.train_steps >= 102000:
                print("Stage 2 Training Complete (102k steps). Saving and stopping...")
                agent.save(model_path)
                stop_event.set()
                break
                
        except Exception as e:
            print(f"Error in Temporal Loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)
    
    print("Temporal Agent Loop Ended.")

def main():
    print("--- Auto-Farmer RL Agent ---")
    
    # Parse args with argparse
    parser = argparse.ArgumentParser(description="Auto-Farmer RL Agent")
    parser.add_argument("--setup", "-s", action="store_true", help="Force run the health bar setup wizard")
    parser.add_argument("--setup-cooldown", action="store_true", help="Run the cooldown bar setup wizard")
    parser.add_argument("--model", "-m", type=str, default=None, 
                        help="Path to detection model (.pt for YOLO, .pth for RF-DETR)")
    parser.add_argument("--agent-model", type=str, default=None, help="Path to pretrained RL agent model")
    parser.add_argument("--fast", action="store_true", help="Enable FP8 precision (Fast Mode)")
    parser.add_argument("--temporal", action="store_true", help="Use temporal agent (30-frame context)")
    parser.add_argument("--stage", type=int, default=2, choices=[2, 3, 4], help="Training stage (2=warmup, 3=hybrid, 4=safety_override)")
    args = parser.parse_args()
    
    force_setup = args.setup
    force_cooldown_setup = args.setup_cooldown
    detection_model_path = args.model
    agent_model_path = args.agent_model
    fast_mode = args.fast
    use_temporal = args.temporal
    training_stage = args.stage
    
    if detection_model_path:
        print(f"Using detection model: {detection_model_path}")

    # 1. Select Window
    # Set ROBLOX_WINDOW_TITLE env var to skip the interactive prompt.
    _window_arg = os.environ.get("ROBLOX_WINDOW_TITLE", "")
    if _window_arg:
        print(f"Using window title from env: '{_window_arg}'")
        capture_service.window_title = _window_arg
    else:
        capture_service.select_window()
    
    # Start Capture early to check resolution
    print("Starting Screen Capture...")
    capture_service.start()
    time.sleep(1.0) # Warmup
    
    # Check Config
    wizard = ConfigWizard(capture_service)
    if force_setup or not wizard.validate_config():
        print("Starting health bar setup wizard...")
        wizard.select_health_bar()
    
    # Cooldown Setup Wizard
    if force_cooldown_setup:
        print("Starting cooldown bar setup wizard...")
        wizard.select_cooldown_bars()
    
    # Load Cooldown Detector
    from utils.config_wizard import CooldownDetector
    cooldown_rois = ConfigWizard.load_cooldown_rois()
    cooldown_detector = CooldownDetector(cooldown_rois) if cooldown_rois else None
    if cooldown_detector:
        print(f"Loaded {len(cooldown_rois)} cooldown ROIs.")
    else:
        print("No cooldown ROIs configured. Use --setup-cooldown to configure.")
    
    # 2. Initialize Components
    print("Initializing Perception...")
    perception = init_perception(detection_model_path, fast_mode=fast_mode)
    
    print("Initializing Agent...")
    # Action Dim = 26 (Updated)
    # 0-14: Basic + Look X
    # 15: r_2
    # 16: g
    # 17: space (Jump)
    
    # Instantiate MuZero Agent
    from rl.muzero_agent import MuZeroAgent
    agent = MuZeroAgent()
    
    # Load Offline Weights if available
    model_path = agent_model_path if agent_model_path else "muzero_agent_offline.pth"
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            agent.load(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading agent model: {e}")
            print("Starting from scratch.")
    else:
        print("No offline model found. Starting from scratch.")
            
    print("Initializing Controls...")
    controller = InputController()
    state_manager = StateManager()
    
    # Focus the game window
    print("Focusing Game Window...")
    capture_service.focus()
    time.sleep(0.3)
    
    # Hot Reload State
    last_model_mtime = 0
    if agent_model_path and os.path.exists(agent_model_path):
        last_model_mtime = os.path.getmtime(agent_model_path)
    
    # 4. Preview Mode
    mode = preview_detection(perception, capture_service)
    
    if mode == "agent":
        # 5. Start Agent Thread with perception and cooldown detector for rewards
        agent_thread = threading.Thread(target=agent_loop_muzero, 
                                      args=(agent, controller, state_manager, perception, cooldown_detector, 1000, "muzero_agent_online.pth"))
        agent_thread.daemon = True
        agent_thread.start()
        
        # UI Loop
        while True:
            try:
                msg = status_queue.get(timeout=0.1)
                print(f"STATUS: {msg[0]} R={msg[1]}")
            except:
                pass
            # Check if thread died
            if not agent_thread.is_alive():
                print("Agent thread died.")
                break
            time.sleep(0.1)
    
    elif mode == "temporal":
        # Initialize Temporal Agent
        from rl.temporal_agent import TemporalAgent
        
        temporal_agent = TemporalAgent(
            action_dim=26,
            hidden_dim=384,
            seq_len=30,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            stage=training_stage
        )
        
        # Load existing model if available
        if os.path.exists("temporal_agent.pth"):
            temporal_agent.load("temporal_agent.pth")
        
        # Start Temporal Agent Thread
        agent_thread = threading.Thread(
            target=temporal_agent_loop,
            args=(temporal_agent, controller, perception, cooldown_detector, 1000, "temporal_agent.pth", training_stage)
        )
        agent_thread.daemon = True
        agent_thread.start()
        
        # UI Loop with danger indicator
        while True:
            try:
                msg = status_queue.get(timeout=0.1)
                danger = msg[2] if len(msg) > 2 else 0
                print(f"STATUS: {msg[0]} R={msg[1]:.3f} DANGER={danger:.2f}")
            except:
                pass
            if not agent_thread.is_alive():
                print("Temporal agent thread died.")
                break
            time.sleep(0.1)
            
    elif mode == "direct":
        # 5. Start Direct Policy Thread
        policy = DirectPolicy()
        agent_thread = threading.Thread(target=direct_policy_loop, args=(policy, controller, state_manager))
        agent_thread.daemon = True
        agent_thread.start()
    elif mode == "imitation":
        # 5. Start Imitation Thread
        agent_thread = threading.Thread(target=imitation_loop, args=(state_manager, controller))
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
            # Pass det_dict (perception.last_det) along with obs
            obs_queue.put((pixel_obs, vector_obs, perception.last_det))
            
            # Check for status updates from Agent
            try:
                current_action, current_reward, current_intention = status_queue.get_nowait()
            except queue.Empty:
                pass
            
            # --- Hot Reload Check ---
            if agent_model_path and os.path.exists(agent_model_path):
                try:
                    mtime = os.path.getmtime(agent_model_path)
                    if mtime > last_model_mtime:
                        print(f"New model detected! Reloading from {agent_model_path}...")
                        agent.load(agent_model_path)
                        last_model_mtime = mtime
                        print("Model reloaded successfully.")
                except Exception as e:
                    print(f"Hot reload failed: {e}")
            
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
