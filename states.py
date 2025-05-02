from utils import * 
import time 


def check_session_end(model,
                      data,
                      start_time,
                      egg_start_pos,
                      exclude_lst: list = ["arm_finger_left", "arm_finger_right"],
                      arm_parts_lst: list = ["base_1",
                                            "arm_base",
                                            "arm_base_2",
                                            "arm_base_2_1",
                                            "arm_handle",
                                            "arm_handle_1"],
                      target_time_start = None):
    """
    Checks if the session should end based on time, egg position, and arm contact.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        start_time: The time when the session started.
        egg_start_pos: The initial position of the egg.
        target_time_start: The time when the egg first entered the target.  None if not in target.

    Returns:
        True if the session should end, False otherwise.
    """

    # Check if the session has been running for more than 5 minutes (300 seconds).
    if time.time() - start_time > 300 * 60:
        logging.info("Session ended due to time limit.")
        return True, -65 

    # Check if the egg is too far away from its starting position.
    egg_id = get_body_id(model, "egg")
    if egg_id != -1:  # check if the egg_id is valid
        egg_pos = data.xpos[egg_id][:2]  # Get the X and Y coordinates
        distance = np.linalg.norm(egg_pos - egg_start_pos)
        if distance > 5:  # If the egg is more than 5 meters away
            logging.info("Session ended: Egg is too far from start.")
            return True, -65 

    # Check if any children of the base (excluding fingers) are touching the floor.
    for body_name in arm_parts_lst:
        if body_name not in exclude_lst:  # Added base, shoulder, elbow, wrist
            if check_contact(model, data, "floor", body_name):
                logging.info(f"Session ended: {body_name} is touching the floor.")
                return True, -50

    # Check if the egg has been in the target for more than 10 seconds.
    if egg_in_target(model, data):
        if target_time_start is None:
            # The egg just entered the target, record the time.
            target_time_start = time.time()
        elif time.time() - target_time_start > 10:
            # The egg has been in the target for more than 10 seconds.
            logging.info("Session ended: Egg is in target for too long.")
            return True, 100
    else:
        # The egg is not in the target, reset the timer.
        target_time_start = None

    return False, 0 



def reward_function(model, data, prev_dist, *args, **kwargs):
    """
    Calculates the reward for the current state.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        prev_dist: Previous distance between the egg and the target.
        egg_start_pos: The initial position of the egg

    Returns:
        The reward for the current state.
    """
    reward = 0

    # Get body IDs
    egg_id = get_body_id(model, "egg")
    target_id = get_body_id(model, "egg_base_target")

    # 1. Initial Reward/Penalty
    if egg_at_the_start(model, data):
        reward += 0.1
    elif not egg_at_the_holding(model,data) and not egg_at_the_start(model,data):
        reward -= 0.1
    if egg_on_the_floor(model, data):
        reward -= 1.0

    # 2. Manipulation Rewards
    reward -= 0.01  # Small negative reward per step to encourage speed
    if egg_at_the_holding(model, data):
        reward += 0.2
        if egg_id != -1:
          reward += 0.1 * data.xpos[egg_id][2]  # Reward for lifting egg

    # 3. Navigation Reward
    if egg_id != -1 and target_id != -1:
        current_dist = np.linalg.norm(data.xpos[egg_id][:2] - data.xpos[target_id][:2])
        reward += 0.02 * (prev_dist - current_dist)

    # 4. Task Completion Reward
    if egg_in_target(model, data):
        reward += 100.0
    reward -= 0.001 # Small penalty per step

    return reward, current_dist
