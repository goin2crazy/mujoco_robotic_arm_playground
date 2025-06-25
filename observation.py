import numpy as np
# Assuming 'utils.py' contains get_body_id and the task state functions:
# egg_at_the_start, egg_on_the_floor, egg_at_the_holding, egg_in_target
from utils import (get_body_id, 
                   get_body_pos)
# If get_body_id or other functions are not in utils, adjust the import accordingly.

def get_observation_reach_task(model, data, body_to_reach_name, joint_ids=[5, 6, 7, 8, 9, 10]):
    observation = {}

    reach_body_id = get_body_id(model, body_to_reach_name)
    finger_left_id = get_body_id(model, "arm_finger_left")
    finger_right_id = get_body_id(model, "arm_finger_right")

    reach_body_position_valid = False
    if reach_body_id != -1:
        reach_body_position = data.xpos[reach_body_id].copy()
        reach_body_position_valid = True

    # # *IT WOULD BE MORE USEFULL TO USE THE DISTANCE BETWEEN BODY_TO_REACH AND FINGERS (Implemented as vector components)
    if reach_body_position_valid and finger_left_id != -1:
        finger_left_position = data.xpos[finger_left_id].copy()
        vec_reach_body_to_finger_left = reach_body_position - finger_left_position # Calculate vector difference
        observation['vec_reach_body_to_finger_left'] = vec_reach_body_to_finger_left
    else:
        observation['vec_reach_body_to_finger_left'] = [0.0, 0.0, 0.0] # Default vector if body_to_reach or finger is missing

    # # 5. Vector from body_to_reach to right finger (dx, dy, dz)
    if reach_body_position_valid and finger_right_id != -1:
        finger_right_position = data.xpos[finger_right_id].copy()
        vec_reach_body_to_finger_right = reach_body_position - finger_right_position # Calculate vector difference
        observation['vec_reach_body_to_finger_right'] = vec_reach_body_to_finger_right
    else:
        observation['vec_reach_body_to_finger_right'] = [0.0, 0.0, 0.0] # Default vector if body_to_reach or finger is missing

    # 8. Arm joint velocities
    # Ensure the slice [:7] correctly corresponds to your arm's joints in qvel
    
    arm_joint_velocities = data.qvel[joint_ids].copy()
    observation['arm_joints_velocities'] = (arm_joint_velocities)
    
    arm_joint_angles = data.qpos[joint_ids].copy()
    observation['arm_joints_angles'] = (arm_joint_angles)
    return observation
    

def get_observation_grab_transport_task(model, data):
    """
    Returns the observation vector for the RL agent.

    The observation includes information about the egg, fingers, arm, target,
    and contact forces.
    # """

    # # Get body IDs using the provided utility function
    # target_id = get_body_id(model, "egg_base_target")

    # # Initialize an empty list to hold the observation values.
    # observation = []

    # # --- Retrieve egg position for distance calculations ---
    # # We need egg_position even if it's not directly in the observation vector,
    # # to calculate distances to fingers.
    # egg_position_valid = False
    # egg_position = np.array([0.0, 0.0, 0.0]) # Default placeholder
    # if egg_id != -1:
    #     egg_position = data.xpos[egg_id].copy()
    #     egg_position_valid = True

    # arm_joint_angles = data.qpos[[7, 8, 9, 10, 11, 12]].copy()
    # observation.extend(arm_joint_angles)

    # # 8. Arm joint velocities
    # # Ensure the slice [:7] correctly corresponds to your arm's joints in qvel
    # arm_joint_velocities = data.qvel[[7, 8, 9, 10, 11, 12]].copy()
    # observation.extend(arm_joint_velocities)

    # # 9. Distance between egg and target (in XY plane) - Scalar
    # if egg_position_valid and target_id != -1:
    #     target_position = data.xpos[target_id].copy()
    #     # Calculate distance only in the XY plane
    #     dist_egg_to_target_xy = egg_position - target_position
    #     observation.extend(dist_egg_to_target_xy)
    # else:
    #     observation.extend([0.0, 0, 0]) # Default distance
    # return np.array(observation, dtype=np.float64)
    ...