# Define action mapping
# 0: Idle
# 1: W
# 2: A
# 3: S
# 4: D
# 5: Turn Left
# 6: Turn Right

ACTION_MAP = {
    0: "idle",
    1: "w",
    2: "a",
    3: "s",
    4: "d",
    5: "turn_left",
    6: "turn_right",
    7: "dash_back",  # Evasive Back (S + Q)
    8: "dash_left",  # Evasive Left (A + Q)
    9: "dash_right",  # Evasive Right (D + Q)
    10: "dash_forward"  # Forward (not evasive) (W + Q)
}

def get_action_name(action_id):
    return ACTION_MAP.get(action_id, "idle")
