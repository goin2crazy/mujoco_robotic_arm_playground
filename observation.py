import mujoco
import numpy as np
# Assuming 'utils.py' contains get_body_id and the task state functions:
# egg_at_the_start, egg_on_the_floor, egg_at_the_holding, egg_in_target
from utils import (get_body_id, 
                   get_body_pos)
# If get_body_id or other functions are not in utils, adjust the import accordingly.

def get_observation_reach_task(model, data, reach_to_body_pos, joint_ids=[6, 7, 8, 9, 10]):
    """
    Calculates and returns observations as a list.

    Args:
        model: The MuJoCo model object.
        data: The MuJoCo data object.
        body_to_reach_name (str): The name of the body to reach.
        joint_ids (list): A list of joint IDs for the arm.

    Returns:
        list: A list containing the calculated observations.
    """
    observation = []

    finger_left_id = get_body_id(model, "arm_finger_left")
    finger_right_id = get_body_id(model, "arm_finger_right")

    reach_body_position = reach_to_body_pos
    reach_body_position_valid = True

    # Vector from body_to_reach to left finger (dx, dy, dz)
    if reach_body_position_valid and finger_left_id != -1:
        finger_left_position = data.xpos[finger_left_id].copy()
        vec_reach_body_to_finger_left = reach_body_position - finger_left_position
        observation.extend(vec_reach_body_to_finger_left.tolist()) # Convert numpy array to list and extend
    else:
        observation.extend([0.0, 0.0, 0.0])

    # Vector from body_to_reach to right finger (dx, dy, dz)
    if reach_body_position_valid and finger_right_id != -1:
        finger_right_position = data.xpos[finger_right_id].copy()
        vec_reach_body_to_finger_right = reach_body_position - finger_right_position
        observation.extend(vec_reach_body_to_finger_right.tolist()) # Convert numpy array to list and extend
    else:
        observation.extend([0.0, 0.0, 0.0])

    # Arm joint velocities
    arm_joint_velocities = data.qvel[joint_ids].copy()
    observation.extend(arm_joint_velocities.tolist()) # Convert numpy array to list and extend

    # Arm joint angles
    arm_joint_angles = data.qpos[joint_ids].copy()
    observation.extend(arm_joint_angles.tolist()) # Convert numpy array to list and extend

    return np.array(observation, dtype=np.float16)
    

def get_observation_grab_transport_task(model, data, target_body_name:str='egg', ):
    # Lets init the observations list 
    obs = []

    # first lets add there the basic obsrevations, because there we are always reaching to propably egg 
    target_position = get_body_pos(model, data, target_body_name)
    reach_obs = get_observation_reach_task(model=model, data=data, reach_to_body_pos=target_position)
    obs.extend(reach_obs)

    # lets add the distance from left finger to right finger to let agent see how much it is open 
    left_finger_pos = get_body_pos(model, data, "arm_finger_left")
    right_finger_pos = get_body_pos(model, data, "arm_finger_right")
    fingers_dist = np.linalg.norm(left_finger_pos -right_finger_pos)
    obs.append(fingers_dist)

    # gripper center (bese, roboarm fingers located on)
    gripper_centre = "arm_handle_1"
    ee_id = get_body_id(model, gripper_centre)
    ee_pos = data.xpos[ee_id]
    ee_vel = data.qvel[ee_id]  # Linear velocity
    obs.append(ee_vel)

    # nest (target place to put the egg)
    nest = "egg_base_target"
    nest_pos = get_body_pos(model, data, nest)

    obs.extend(nest_pos - ee_pos)
    obs.extend(nest_pos - target_position)
    
    # convert the obs list to array 
    try: 
        obs = np.array(obs)
    except: 
        print(obs)
        exit()

    return obs