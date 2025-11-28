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
    8: "click",      # Basic Attack
    9: "1",          # Move 1
    10: "2",         # Move 2
    11: "3",         # Move 3
    12: "4",         # Move 4
    13: "r",         # Special / Teleport (R + Empty)
    14: "r_1",       # Combo: R + 1 (Aim + Spam R)
    15: "r_2",       # Combo: R + 2 (Aim + 2)
    16: "dash_left", # Evasive Left (A + Q)
    17: "dash_right" # Evasive Right (D + Q)
}

def get_action_name(action_id):
    return ACTION_MAP.get(action_id, "idle")
