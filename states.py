from utils import *
import time
import numpy as np

def check_session_end(model, data, steps, 
                     exclude_lst=[],
                     arm_parts_lst=["base_1", "arm_base", "arm_base_2",
                                   "arm_base_2_1", "arm_handle", "arm_handle_1", 
                                   "arm_finger_left", "arm_finger_right",
                                   ],
                     max_steps=8000):
    """Enhanced session termination with curriculum awareness"""
    # Timeout check
    if steps > max_steps:
        logging.info("Session ended due to time limit.")
        return True, 0

    # Egg distance check
    # egg_id = get_body_id(model, "egg")
    # if egg_id != -1:
    #     egg_pos = data.xpos[egg_id][:2]
    #     if np.linalg.norm(egg_pos - egg_start_pos) > 5:
    #         logging.info("Session ended: Egg too far from start.")
    #         return True, -65

    # Arm collision check
    for body_name in arm_parts_lst:
        if body_name not in exclude_lst and check_contact(model, data, "floor", body_name):
            logging.info(f"Session ended: {body_name} touched floor.")
            return False, -2

    return False, 0

def improved_reward_function_reach_task(model, data, 
                             reach_to_body_pos
                             ): 
    
    # Get positions
    reach_to_body_pos = reach_to_body_pos
    grip_pos = (get_body_pos(model, data, "arm_finger_left") + 
               get_body_pos(model, data, "arm_finger_right")) / 2
    
    # 1. REACHING STAGE REWARDS
    grip_to_egg_dist = np.linalg.norm(grip_pos - reach_to_body_pos)
    reach_reward = 1.0 / (0.3 + grip_to_egg_dist)  # Encourage approach
    return {"reach": reach_reward, "reach_to_object_pos": reach_to_body_pos}

def improved_reward_function_grab_transport(model, 
                                            data, 
                                            action, 
                             ):
    # initialize the reward 
    reward = 0 

    # reaching reward 
    egg_pos = get_body_pos(model, data, "egg")
    reaching_reward = improved_reward_function_reach_task(model, data, egg_pos)['reach']
    reward += reaching_reward

    nest_pos = get_body_pos(model, data, "egg_base_target")
    nest_to_egg_dist = np.linalg.norm(nest_pos - egg_pos)
    egg_to_nest_reach_reward = 1.0 / (0.3 + nest_to_egg_dist)  # Encourage approach
    reward += egg_to_nest_reach_reward 

    # 2. GRASPING REWARDS (continuous)
    left_contact = contact_force(model, data, "egg", "arm_finger_left")
    right_contact = contact_force(model, data, "egg", "arm_finger_right")
    grip_force = (left_contact + right_contact)
    grasp_reward = np.clip(grip_force, 0, 10)  # Max 10 reward for firm grip
    reward += grasp_reward * 10
    
    # 5. GRIP MAINTENANCE (critical!)
    grip_stability = 1.0 / (0.1 + abs(left_contact - right_contact))
    reward += grip_stability

    # Action penalty (encourage smooth control)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward 
