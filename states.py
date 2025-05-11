from utils import *
import time
import numpy as np

def check_session_end(model, data, start_time, egg_start_pos, 
                     exclude_lst=["arm_finger_left", "arm_finger_right"],
                     arm_parts_lst=["base_1", "arm_base", "arm_base_2",
                                   "arm_base_2_1", "arm_handle", "arm_handle_1"],
                     target_time_start=None):
    """Enhanced session termination with curriculum awareness"""
    # Timeout check
    if time.time() - start_time > 300 * 60:
        logging.info("Session ended due to time limit.")
        return True, -65

    # Egg distance check
    egg_id = get_body_id(model, "egg")
    if egg_id != -1:
        egg_pos = data.xpos[egg_id][:2]
        if np.linalg.norm(egg_pos - egg_start_pos) > 5:
            logging.info("Session ended: Egg too far from start.")
            return True, -65

    # Arm collision check
    for body_name in arm_parts_lst:
        if body_name not in exclude_lst and check_contact(model, data, "floor", body_name):
            logging.info(f"Session ended: {body_name} touched floor.")
            return True, -50

    # Target success check
    if egg_in_target(model, data):
        target_time_start = target_time_start or time.time()
        if time.time() - target_time_start > 10:
            logging.info("Session ended: Successful placement.")
            return True, 100
    else:
        target_time_start = None

    return False, 0


def reward_function(model, data, prev_dist, mode="all", *args, **kwargs):
    """
    Reward function with scaling per learning stage and distance between egg and arm_handle_1.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        prev_dist: Previous distance between the egg and the target.
        mode: "grasp", "hold", "transport", "final", or "all"

    Returns:
        Tuple of (reward, current_dist, arm_egg_dist)
    """
    reward = 0
    current_dist = 0
    arm_egg_dist = 0

    # Scaling factors per stage
    scale = {
        "grasp":      {"grasp": 1.0, "hold": 0.2, "transport": 0.0, "final": 0.0, "arm_egg": 1.0},
        "hold":       {"grasp": 0.2, "hold": 1.0, "transport": 0.2, "final": 0.0, "arm_egg": 0.2},
        "transport":  {"grasp": 0.0, "hold": 0.2, "transport": 1.0, "final": 0.2, "arm_egg": 0.1},
        "final":      {"grasp": 0.0, "hold": 0.2, "transport": 0.2, "final": 1.0, "arm_egg": 0.0},
        "all":        {"grasp": 1.0, "hold": 1.0, "transport": 1.0, "final": 1.0, "arm_egg": 0.5},
    }

    s = scale.get(mode, scale["all"])

    egg_id = get_body_id(model, "egg")
    target_id = get_body_id(model, "egg_base_target")
    arm_handle_id = get_body_id(model, "arm_handle_1")

    # 🥚 Grasp-related rewards
    if egg_at_the_start(model, data):
        reward += s["grasp"] * 0.1
    elif not egg_at_the_holding(model, data) and not egg_at_the_start(model, data):
        reward -= s["grasp"] * 0.1
    if egg_on_the_floor(model, data):
        reward -= s["grasp"] * 1.0

    # ✋ Hold rewards
    reward -= 0.01  # base step penalty
    if egg_at_the_holding(model, data):
        reward += s["hold"] * 0.2
        if egg_id != -1:
            reward += s["hold"] * 0.1 * data.xpos[egg_id][2]

    # 🚚 Transport rewards
    if egg_id != -1 and target_id != -1:
        current_dist = np.linalg.norm(data.xpos[egg_id][:2] - data.xpos[target_id][:2])
        reward += s["transport"] * 0.02 * (prev_dist - current_dist)

    # 🎯 Task completion
    if egg_in_target(model, data):
        reward += s["final"] * 100.0

    # 💞 Distance to arm_handle_1
    if egg_id != -1 and arm_handle_id != -1:
        arm_egg_dist = np.linalg.norm(data.xpos[egg_id] - data.xpos[arm_handle_id])
        reward -= s["arm_egg"] * 0.01 * arm_egg_dist  # the smaller the better!

    reward -= 0.001  # soft step penalty

    return reward, current_dist, arm_egg_dist
