from utils import *
import time
import numpy as np

def check_session_end(model, data, steps, 
                     exclude_lst=[],
                     arm_parts_lst=["base_1", "arm_base", "arm_base_2",
                                   "arm_base_2_1", "arm_handle", "arm_handle_1", 
                                   "arm_finger_left", "arm_finger_right",
                                   ],
                     max_steps=1000):
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
                             reach_to_body_name
                             ): 
    
    # Get positions
    reach_to_body_pos = get_body_pos(model, data, reach_to_body_name)
    grip_pos = (get_body_pos(model, data, "arm_finger_left") + 
               get_body_pos(model, data, "arm_finger_right")) / 2
    
    # 1. REACHING STAGE REWARDS
    grip_to_egg_dist = np.linalg.norm(grip_pos - reach_to_body_pos)
    reach_reward = 1.0 / (0.3 + grip_to_egg_dist)  # Encourage approach
    return {"reach": reach_reward, "reach_to_object_pos": reach_to_body_pos}

def improved_reward_function_grab_transport(model, data, 
                             old_target_dist, 
                             egg_initial_z):

    reach_reward_fn_output = improved_reward_function_reach_task
    reach_reward = reach_reward_fn_output['reach']
    egg_pos = reach_reward_fn_output['reach_to_object_pos']

    target_pos = get_body_pos(model, data, "egg_base_target")
    # 2. GRASPING REWARDS (continuous)
    left_contact = contact_force(model, data, "egg", "arm_finger_left")
    right_contact = contact_force(model, data, "egg", "arm_finger_right")
    grip_force = (left_contact + right_contact)
    grasp_reward = np.clip(grip_force, 0, 10)  # Max 10 reward for firm grip
    
    # 3. LIFTING REWARDS
    egg_height = egg_pos[2] - egg_initial_z
    lift_reward = 0
    if grip_force > 1.0:  # Only if grasped
        lift_reward = 5 * egg_height  # Encourage lifting
    
    # 4. TRANSPORT REWARDS
    egg_to_target_dist = np.linalg.norm(egg_pos - target_pos)
    transport_reward = 0
    if grip_force > 1.0:  # Only if grasped
        # Progress reward (velocity toward target)
        progress = old_target_dist - egg_to_target_dist
        transport_reward = 10 * progress
        
        # Success bonus
        if egg_to_target_dist < 0.3:
            transport_reward += 100
    
    # 5. GRIP MAINTENANCE (critical!)
    grip_stability = 1.0 / (0.1 + abs(left_contact - right_contact))
    
    # 6. TERMINAL CONDITIONS
    done = False
    if egg_pos[2] < egg_initial_z - 0.1:  # Egg dropped
        done = True
    elif egg_to_target_dist < 0.02:  # Success
        done = True
    
    # Combine rewards with stage weighting
    total_reward = (
        0.5 * reach_reward +
        2.0 * grasp_reward +
        1.0 * lift_reward +
        3.0 * transport_reward +
        0.5 * grip_stability
    )
    
    return total_reward, done, {
        "reach_reward": reach_reward,
        "target_dist": egg_to_target_dist
    }