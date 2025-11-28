import cv2

def draw_detections(frame, det_dict):
    """
    Draws bounding boxes and labels on the frame.
    """
    # Colors for different classes (BGR)
    COLORS = {
        "player": (255, 0, 255),  # Purple
        "enemy": (255, 0, 255),   # Purple
        "item": (255, 255, 0),    # Cyan
        "obstacle": (0, 165, 255),# Orange
        "goal": (255, 0, 255)     # Magenta
    }
    
    # Draw Player
    if det_dict["player"] is not None:
        # Handle variable tuple length
        p_data = det_dict["player"]
        cx, cy, w, h = p_data[:4]
        conf = p_data[4] if len(p_data) > 4 else 0.0
        flash = p_data[5] if len(p_data) > 5 else 0.0
        ragdoll = p_data[6] if len(p_data) > 6 else 0.0
            
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        # Visual feedback for Flash (Damage)
        color = COLORS["player"]
        if flash > 0.2:
            color = (0, 0, 255) # Red flash
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"Player {conf:.2f}"
        if ragdoll > 0.5:
            label += " [RAGDOLL]"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw Enemies
    for item in det_dict["enemies"]:
        cx, cy, w, h = item[:4]
        conf = item[4] if len(item) > 4 else 0.0
        state = item[5] if len(item) > 5 else ""
        flash = item[6] if len(item) > 6 else 0.0
        ragdoll = item[7] if len(item) > 7 else 0.0
            
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        color = COLORS["enemy"]
        if flash > 0.2:
            color = (0, 255, 255) # Yellow flash for enemy hit
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"Enemy {conf:.2f}"
        if state and state != "Idle":
             label += f" [{state}]"
        if ragdoll > 0.5:
             label += " [RD]"
             
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw Items
    for (cx, cy, w, h) in det_dict["items"]:
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["item"], 2)
        cv2.putText(frame, "Item", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["item"], 2)

    # Draw Obstacles
    for (cx, cy, w, h) in det_dict["obstacles"]:
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["obstacle"], 2)
        cv2.putText(frame, "Obstacle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["obstacle"], 2)

    # Draw Goal
    if det_dict["goal"] is not None:
        cx, cy, w, h = det_dict["goal"]
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["goal"], 2)
        cv2.putText(frame, "Goal", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["goal"], 2)
        
    # Draw Health
    if "health" in det_dict:
        hp = det_dict["health"]
        cv2.putText(frame, f"HP: {hp*100:.1f}%", (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame
