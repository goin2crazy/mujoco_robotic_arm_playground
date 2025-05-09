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

def reward_function(model, data, prev_dist, mode="grasp", *args, **kwargs):
    """
    Mode-based reward function with curriculum learning
    
    Modes:
    - 'grasp': Focus on initial contact and pickup
    - 'lift': Focus on vertical elevation
    - 'transport': Focus on horizontal movement
    - 'final': Full task with all components
    """
    reward = 0
    egg_id = get_body_id(model, "egg")
    target_id = get_body_id(model, "egg_base_target")
    
    # Common metrics
    contact = egg_at_the_holding(model, data)
    egg_height = data.xpos[egg_id][2] if egg_id != -1 else 0
    current_dist = np.linalg.norm(data.xpos[egg_id][:2] - data.xpos[target_id][:2]) \
        if (egg_id != -1 and target_id != -1) else prev_dist

    # Mode-specific rewards
    if mode == "grasp":
        # Focus on initial contact and grip maintenance
        reward += 2.0 * contact  # Strong contact incentive
        reward -= 0.2 * (1 - contact)  # Penalize breaking contact
        reward += 0.05 * egg_height  # Small lift encouragement
        reward += 0.1 / (current_dist + 0.1)  # Inverse distance bonus
        
    elif mode == "lift":
        # Focus on vertical elevation and stability
        reward += 1.0 * contact  # Maintain contact
        reward += 0.5 * egg_height  # Direct height reward
        reward -= 0.1 * abs(data.qvel[model.joints]).mean()  # Velocity penalty
        reward -= 0.02 * current_dist  # Mild distance penalty
        
    elif mode == "transport":
        # Focus on horizontal movement and positioning
        reward += 0.5 * contact  # Maintain grip
        reward += 0.3 * egg_height  # Maintain height
        reward += 0.5 * (prev_dist - current_dist)  # Movement bonus
        reward -= 0.005 * np.linalg.norm(data.ctrl)  # Control effort penalty
        
    elif mode == "final":
        # Full task with balanced components
        reward += 1.0 * contact
        reward += 0.4 * egg_height
        reward += 0.7 * (prev_dist - current_dist)
        reward -= 0.01 * np.linalg.norm(data.qacc)  # Acceleration penalty
        reward -= 0.002 * np.linalg.norm(data.ctrl)  # Control effort
        
    # Common penalties
    if egg_on_the_floor(model, data):
        reward -= 2.0  # Strong floor penalty
    
    # Curriculum progression bonuses
    if mode != "final":
        # Encourage reaching next stage
        if egg_height > 0.5 and mode in ["grasp", "lift"]:
            reward += 1.0
        if current_dist < 0.5 and mode == "transport":
            reward += 2.0
            
    # Sparse rewards (only in final mode)
    if mode == "final":
        if egg_in_target(model, data):
            reward += 100.0  # Completion bonus
        elif egg_on_the_floor(model, data):
            reward -= 50.0  # Failure penalty

    return reward, current_dist