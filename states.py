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

def reward_function_grasp(model, data):
    """
    Calculates the reward for a grasping task, optimizing distance calculations
    using NumPy. Unused parameters have been removed.

    Args:
        model: The MuJoCo model object.
        data: The MuJoCo data object containing simulation state (e.g., positions).

    Returns:
        A tuple containing:
            - reward (float): The calculated reward.
            - distance_sum (float): The sum of distances between fingers and the egg.
    """
    reward = 0.0
    # current_dist and arm_egg_dist were not used in the original logic and are now removed.

    # Assuming get_body_id, egg_on_the_floor, and check_contact are defined elsewhere
    # and work as expected.
    egg_id = get_body_id(model, "egg")
    # target_id is not used in the provided logic and is now removed.

    right_fing_id, left_fing_id = get_body_id(model, "arm_finger_right"), get_body_id(model, "arm_finger_left")

    # Convert positions to NumPy arrays for efficient vectorized operations
    egg_pos = np.array(data.xpos[egg_id])
    right_fing_pos = np.array(data.xpos[right_fing_id])
    left_fing_pos = np.array(data.xpos[left_fing_id])

    # Calculate Euclidean distances using numpy.linalg.norm
    # This is significantly faster than list comprehensions for larger arrays.
    distance_fingers_egg_right = np.linalg.norm(egg_pos - right_fing_pos)
    distance_fingers_egg_left = np.linalg.norm(egg_pos - left_fing_pos)

    distance_mean = (distance_fingers_egg_right + distance_fingers_egg_left)/2 

    # Original reward calculation logic
    reward = (-distance_mean)

    # In python True in bool = 1 and False = 0, this behavior is preserved.
    # Assuming egg_on_the_floor returns a boolean or 0/1.
    # reward -= 100 * egg_on_the_floor(model, data)

    # # Assuming check_contact returns a boolean or 0/1.
    reward += 20 * check_contact(model, data, "egg", "arm_finger_right")
    reward += 20 * check_contact(model, data, "egg", "arm_finger_left")

    return reward, distance_mean

# Note: The functions `get_body_id`, `egg_on_the_floor`, and `check_contact`
# are assumed to be defined elsewhere in your codebase.
import numpy as np

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

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Example 1: Actions are smooth ---")
    current_reward_1 = 10.0
    prev_actions_1 = np.array([0.1, 0.2, 0.3])
    curr_actions_1 = np.array([0.101, 0.201, 0.301]) # Small changes

    adjusted_reward_1 = adjust_reward_for_smoothness(
        current_reward_1,
        curr_actions_1,
        prev_actions_1,
        moving_smoothness=5e-4, # Default value
        penalty_multiplier=1
    )
    print(f"Original Reward: {current_reward_1}")
    print(f"Previous Actions: {prev_actions_1}")
    print(f"Current Actions: {curr_actions_1}")
    print(f"Adjusted Reward: {adjusted_reward_1}\n") # Should be close to original reward

    print("--- Example 2: Actions are not smooth (penalty applied) ---")
    current_reward_2 = 10.0
    prev_actions_2 = np.array([0.1, 0.2, 0.3])
    curr_actions_2 = np.array([0.5, 0.1, 0.8]) # Large changes

    adjusted_reward_2 = adjust_reward_for_smoothness(
        current_reward_2,
        curr_actions_2,
        prev_actions_2,
        moving_smoothness=5e-4, # Default value
        penalty_multiplier=1
    )
    print(f"Original Reward: {current_reward_2}")
    print(f"Previous Actions: {prev_actions_2}")
    print(f"Current Actions: {curr_actions_2}")
    print(f"Adjusted Reward: {adjusted_reward_2}\n") # Should be significantly lower

    print("--- Example 3: Adjusting penalty_multiplier ---")
    current_reward_3 = 10.0
    prev_actions_3 = np.array([0.1, 0.2, 0.3])
    curr_actions_3 = np.array([0.5, 0.1, 0.8]) # Large changes

    adjusted_reward_3 = adjust_reward_for_smoothness(
        current_reward_3,
        curr_actions_3,
        prev_actions_3,
        moving_smoothness=5e-4,
        penalty_multiplier=1 # Increased penalty
    )
    print(f"Original Reward: {current_reward_3}")
    print(f"Previous Actions: {prev_actions_3}")
    print(f"Current Actions: {curr_actions_3}")
    print(f"Adjusted Reward (higher penalty): {adjusted_reward_3}\n")

    print("--- Example 4: Edge case - all actions are zero ---")
    current_reward_4 = 5.0
    prev_actions_4 = np.array([0.0, 0.0])
    curr_actions_4 = np.array([0.0, 0.0])

    adjusted_reward_4 = adjust_reward_for_smoothness(
        current_reward_4,
        curr_actions_4,
        prev_actions_4
    )
    print(f"Original Reward: {current_reward_4}")
    print(f"Previous Actions: {prev_actions_4}")
    print(f"Current Actions: {curr_actions_4}")
    print(f"Adjusted Reward (no change): {adjusted_reward_4}\n")

    print("--- Example 5: Edge case - different array shapes ---")
    current_reward_5 = 10.0
    prev_actions_5 = np.array([0.1, 0.2])
    curr_actions_5 = np.array([0.1, 0.2, 0.3])

    adjusted_reward_5 = adjust_reward_for_smoothness(
        current_reward_5,
        curr_actions_5,
        prev_actions_5
    )
    print(f"Original Reward: {current_reward_5}")
    print(f"Previous Actions: {prev_actions_5}")
    print(f"Current Actions: {curr_actions_5}")
    print(f"Adjusted Reward (shape mismatch): {adjusted_reward_5}\n")

    print("--- Example 6: Edge case - non-numpy inputs ---")
    current_reward_6 = 10.0
    prev_actions_6 = [0.1, 0.2]
    curr_actions_6 = [0.1, 0.2, 0.3]

    adjusted_reward_6 = adjust_reward_for_smoothness(
        current_reward_6,
        curr_actions_6,
        prev_actions_6
    )
    print(f"Original Reward: {current_reward_6}")
    print(f"Previous Actions: {prev_actions_6}")
    print(f"Current Actions: {curr_actions_6}")
    print(f"Adjusted Reward (non-numpy inputs): {adjusted_reward_6}\n")


