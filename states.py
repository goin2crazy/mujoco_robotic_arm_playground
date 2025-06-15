from utils import *
import time
import numpy as np

def check_session_end(model, data, steps, egg_start_pos, 
                     exclude_lst=[],
                     arm_parts_lst=["base_1", "arm_base", "arm_base_2",
                                   "arm_base_2_1", "arm_handle", "arm_handle_1", 
                                   "arm_finger_left", "arm_finger_right",
                                   ],
                     target_time_start=None):
    """Enhanced session termination with curriculum awareness"""
    # Timeout check
    if steps > 18_000:
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
            return False, -10

    # Target success check
    if egg_in_target(model, data):
        target_time_start = target_time_start or time.time()
        if time.time() - target_time_start > 10:
            logging.info("Session ended: Successful placement.")
            return True, 100
    else:
        target_time_start = None

    return False, 0

import numpy as np

def distance_penalty(model, data):
    """
    Calculates a penalty based on the distance between the fingers and the egg.

    Args:
        model: The MuJoCo model object.
        data: The MuJoCo data object containing simulation state (e.g., positions).

    Returns:
        float: The calculated distance penalty (negative value).
        float: The mean distance between fingers and the egg.
    """
    egg_id = get_body_id(model, "egg")
    right_fing_id, left_fing_id = get_body_id(model, "arm_finger_right"), get_body_id(model, "arm_finger_left")

    egg_pos = np.array(data.xpos[egg_id])
    right_fing_pos = np.array(data.xpos[right_fing_id])
    left_fing_pos = np.array(data.xpos[left_fing_id])

    distance_fingers_egg_right = np.linalg.norm(egg_pos - right_fing_pos)
    distance_fingers_egg_left = np.linalg.norm(egg_pos - left_fing_pos)

    distance_mean = (distance_fingers_egg_right + distance_fingers_egg_left) / 2
    return -distance_mean, distance_mean

def target_touching_reward(model, data):
    """
    Calculates a reward for the fingers touching the egg.

    Args:
        model: The MuJoCo model object.
        data: The MuJoJo data object containing simulation state.

    Returns:
        float: The calculated touching reward.
    """
    reward = 0.0
    reward += 20 * check_contact(model, data, "egg", "arm_finger_right")
    reward += 20 * check_contact(model, data, "egg", "arm_finger_left")

    return reward

def reward_function_grasp(model, data, old_distance, *args, **kwargs):
    """
    Calculates the reward for a grasping task, optimizing distance calculations
    using NumPy. Unused parameters have been removed.

    Args:
        model: The MuJoCo model object.
        data: The MuJoCo data object containing simulation state (e.g., positions).
        old_distance (float): The distance from the previous step, used for distance_reward.

    Returns:
        A tuple containing:
            - reward (float): The calculated reward.
            - distance_sum (float): The sum of distances between fingers and the egg.
    """
    reward = 0.0

    # Calculate distance penalty and get the current mean distance
    distance_pen, current_distance_mean = distance_penalty(model, data)
    reward += distance_pen

    # Calculate target touching reward
    reward += target_touching_reward(model, data)

    return reward, current_distance_mean

def adjust_reward_for_smoothness(
    total_reward: float,
    new_actions: np.ndarray,
    old_actions: np.ndarray,
    moving_smoothness: float = 5e-4,
    penalty_multiplier: float = 1.0
) -> float:
    """
    Adjusts the total reward for an RL robot by penalizing large changes in actions.

    This function helps to encourage smoother robot movements during training by
    reducing the reward if the squared difference between new and old actions
    exceeds a specified 'moving_smoothness' threshold.

    Args:
        total_reward (float): The original reward obtained from the environment.
        new_actions (np.ndarray): A NumPy array representing the robot's
                                  actions at the current timestep.
        old_actions (np.ndarray): A NumPy array representing the robot's
                                  actions at the previous timestep.
        moving_smoothness (float, optional): The threshold for the squared
                                             difference between new and old actions.
                                             If (new_action - old_action)**2
                                             is greater than this value, a penalty
                                             is applied. Defaults to 5e-4.
        penalty_multiplier (float, optional): A factor to multiply the
                                              sum of penalized squared differences
                                              by, controlling the severity of the
                                              penalty. Defaults to 100.0.

    Returns:
        float: The adjusted reward after applying the smoothness penalty.
               The adjusted reward will be total_reward - penalty.
    """
    if not isinstance(new_actions, np.ndarray) or not isinstance(old_actions, np.ndarray):
        print("Warning: new_actions and old_actions should be NumPy arrays.")
        return total_reward # Return original reward if types are incorrect

    if new_actions.shape != old_actions.shape:
        print("Warning: Shape mismatch between new_actions and old_actions. Returning original reward.")
        return total_reward # Return original reward if shapes don't match

    # Calculate the squared difference for each action component
    action_diff_squared = (new_actions - old_actions) ** 2

    # Identify where the squared difference exceeds the smoothness threshold
    # This creates a boolean array where True indicates a penalty condition
    penalized_elements = action_diff_squared > moving_smoothness

    # Calculate the total penalty. We sum up only the squared differences
    # that exceeded the threshold and then apply the penalty_multiplier.
    penalty = np.sum(action_diff_squared[penalized_elements]) * penalty_multiplier

    # Subtract the calculated penalty from the original total reward
    adjusted_reward = total_reward - penalty

    return adjusted_reward


# Later some new feature 
# --- End DataRecorder Class ---
def reward_function_grasp_v2(model, data, old_distance, *args, **kwargs):
    reward = 0.0

    # Calculate distance penalty and get the current mean distance
    _, current_distance_mean = distance_penalty(model, data)
    reward += (old_distance - current_distance_mean) * 1000

    # Calculate target touching reward
    reward += target_touching_reward(model, data)

    return reward, current_distance_mean
