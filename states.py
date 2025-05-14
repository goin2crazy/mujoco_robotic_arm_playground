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


def reward_function_grasp(model, data, prev_dist, mode="all", *args, **kwargs): 
    
    reward = 0
    current_dist = 0
    arm_egg_dist = 0

    egg_id = get_body_id(model, "egg")
    target_id = get_body_id(model, "egg_base_target")

    right_fing_id, left_fing_id = get_body_id(model, "arm_finger_right"), get_body_id(model, "arm_finger_left")

    distance_fingers_egg_right = sum([(egg_pos - fing_pos) ** 2 for egg_pos, fing_pos in 
                                zip(data.xpos[egg_id], data.xpos[right_fing_id])]) ** 0.5

    distance_fingers_egg_left = sum([(egg_pos - fing_pos) ** 2 for egg_pos, fing_pos in 
                                zip(data.xpos[egg_id], data.xpos[left_fing_id])]) ** 0.5

    reward -=distance_fingers_egg_right
    reward -=distance_fingers_egg_left

    # In python True in bool = 1 and False = 0
    reward -= 100 * egg_on_the_floor(model, data)

    reward += 20 * check_contact(model, data, "floor", "arm_finger_right")
    reward += 20 * check_contact(model, data, "floor", "arm_finger_left")
    return reward

def roughness_penalty(actions, max_value=1.0, min_value=-1.0, diff_weight=1.0, extreme_weight=1.0):
    """
    Calculates a penalty for "rough" actions in a sequence.

    The penalty is composed of two parts:
    1. Penalty for rapid changes: Sum of squared differences between consecutive actions.
       This penalizes jerky movements.
    2. Penalty for sustained extreme values: Counts occurrences of consecutive actions
       that are both at max_value or both at min_value. This penalizes the agent
       for "sticking" to the boundaries of its action space.

    Args:
        actions (list of float): A list of numerical action values.
        max_value (float, optional): The maximum possible value for an action.
                                     Defaults to 1.0.
        min_value (float, optional): The minimum possible value for an action.
                                     Defaults to -1.0.
        diff_weight (float, optional): A weight to scale the penalty from rapid changes.
                                       Defaults to 1.0.
        extreme_weight (float, optional): A weight to scale the penalty from sustained
                                          extreme values. Defaults to 1.0.

    Returns:
        float: A non-negative value representing the total roughness penalty.
               Higher values indicate rougher actions.
    """
    if not actions:
        return 0.0

    total_penalty = 0.0

    # --- 1. Penalty for rapid changes (jerkiness) ---
    # This is calculated as the sum of squared differences between consecutive actions.
    # It penalizes large jumps in action values.
    diff_penalty_sum = 0.0
    if len(actions) > 1:
        for i in range(len(actions) - 1):
            change = actions[i+1] - actions[i]
            diff_penalty_sum += change * change  # Square the change to emphasize larger jumps
    
    total_penalty += diff_weight * diff_penalty_sum

    # --- 2. Penalty for sustained extreme values ---
    # This penalizes sequences like [max_value, max_value] or [min_value, min_value].
    # It counts how many times an action remains at an extreme value for at least two consecutive steps.
    sustained_extreme_count = 0
    if len(actions) > 1:
        for i in range(len(actions) - 1):
            current_action = actions[i]
            next_action = actions[i+1]
            
            # Check if both current and next actions are at the maximum value
            if current_action == max_value and next_action == max_value:
                sustained_extreme_count += 1
            # Check if both current and next actions are at the minimum value
            elif current_action == min_value and next_action == min_value:
                sustained_extreme_count += 1
                
    total_penalty += extreme_weight * sustained_extreme_count
    
    return total_penalty

if __name__ == '__main__':
    # Example Usage:
    print("--- Example Test Cases ---")

    # Smooth actions, low penalty
    actions_smooth = [0.0, 0.1, 0.2, 0.3, 0.4]
    penalty_smooth = roughness_penalty(actions_smooth)
    print(f"Actions: {actions_smooth}, Penalty: {penalty_smooth:.4f}") # Expected: low

    # Rapidly changing actions, high diff penalty
    actions_rapid_change = [0.0, 1.0, -1.0, 1.0, 0.0]
    penalty_rapid_change = roughness_penalty(actions_rapid_change)
    print(f"Actions: {actions_rapid_change}, Penalty: {penalty_rapid_change:.4f}") # Expected: high

    # Sustained max value, high extreme penalty
    actions_sustained_max = [0.5, 1.0, 1.0, 1.0, 0.5]
    penalty_sustained_max = roughness_penalty(actions_sustained_max)
    print(f"Actions: {actions_sustained_max}, Penalty: {penalty_sustained_max:.4f}") # Expected: moderate to high

    # Sustained min value, high extreme penalty
    actions_sustained_min = [0.0, -1.0, -1.0, -1.0, -1.0, 0.0]
    penalty_sustained_min = roughness_penalty(actions_sustained_min)
    print(f"Actions: {actions_sustained_min}, Penalty: {penalty_sustained_min:.4f}") # Expected: moderate to high

    # Mixed: some rapid changes and some sustained extremes
    actions_mixed = [0.0, 0.8, 1.0, 1.0, 0.2, -0.7, -1.0, -1.0, -1.0]
    penalty_mixed = roughness_penalty(actions_mixed)
    print(f"Actions: {actions_mixed}, Penalty: {penalty_mixed:.4f}")

    # Single action, penalty should be 0
    actions_single = [0.5]
    penalty_single = roughness_penalty(actions_single)
    print(f"Actions: {actions_single}, Penalty: {penalty_single:.4f}") # Expected: 0.0

    actions_single_extreme = [1.0]
    penalty_single_extreme = roughness_penalty(actions_single_extreme)
    print(f"Actions: {actions_single_extreme}, Penalty: {penalty_single_extreme:.4f}") # Expected: 0.0

    # Empty actions, penalty should be 0
    actions_empty = []
    penalty_empty = roughness_penalty(actions_empty)
    print(f"Actions: {actions_empty}, Penalty: {penalty_empty:.4f}") # Expected: 0.0

    # Test with different weights
    actions_test_weights = [0.0, 1.0, 1.0] # 1 diff (0 to 1), 1 sustained (1,1)
    # (1-0)^2 = 1. sustained_extreme_count = 1
    penalty_weights_default = roughness_penalty(actions_test_weights) # diff_w=1, extreme_w=1. Penalty = 1*1 + 1*1 = 2
    penalty_weights_diff_high = roughness_penalty(actions_test_weights, diff_weight=5.0, extreme_weight=1.0) # P = 5*1 + 1*1 = 6
    penalty_weights_extreme_high = roughness_penalty(actions_test_weights, diff_weight=1.0, extreme_weight=5.0) # P = 1*1 + 5*1 = 6
    print(f"Actions: {actions_test_weights}")
    print(f"  Default weights: Penalty: {penalty_weights_default:.4f}")
    print(f"  High diff_weight: Penalty: {penalty_weights_diff_high:.4f}")
    print(f"  High extreme_weight: Penalty: {penalty_weights_extreme_high:.4f}")

