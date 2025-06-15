import numpy as np
# Assuming 'utils.py' contains get_body_id and the task state functions:
# egg_at_the_start, egg_on_the_floor, egg_at_the_holding, egg_in_target
from utils import get_body_id, egg_at_the_start, egg_on_the_floor, egg_at_the_holding, egg_in_target
# If get_body_id or other functions are not in utils, adjust the import accordingly.

def get_observation(model, data):
    """
    Returns the observation vector for the RL agent.

    The observation includes information about the egg, fingers, arm, target,
    and contact forces.
    """
    # Get body IDs using the provided utility function
    egg_id = get_body_id(model, "egg")
    finger_left_id = get_body_id(model, "arm_finger_left")
    finger_right_id = get_body_id(model, "arm_finger_right")
    target_id = get_body_id(model, "egg_base_target")

    # Initialize an empty list to hold the observation values.
    observation = []

    # --- Retrieve egg position for distance calculations ---
    # We need egg_position even if it's not directly in the observation vector,
    # to calculate distances to fingers.
    egg_position_valid = False
    egg_position = np.array([0.0, 0.0, 0.0]) # Default placeholder
    if egg_id != -1:
        egg_position = data.xpos[egg_id].copy()
        egg_position_valid = True

    # 1. Egg position (x, y, z) - User marked as *NOT SO USEFULL
    # if egg_id != -1:
    #     # egg_position is already fetched above if egg_id is valid
    #     observation.extend(egg_position)
    # else:
    #     observation.extend([0, 0, 0])  # Default if egg ID not found

    # 2. Egg velocity (linear: x_dot, y_dot, z_dot)
    # if egg_id != -1:
        # Assuming egg_id is a direct index or get_body_id maps to an index
        # suitable for qvel structure where each body has 6 DoFs (3 lin, 3 ang vel).
        # This extracts the linear velocity part.
        # egg_velocity = data.qvel[model.body_jntadr[egg_id]:model.body_jntadr[egg_id] + 3].copy()
        # observation.extend(egg_velocity)
    # else:
    #     observation.extend([0.0, 0.0, 0.0]) # Default velocity

    # 3. Egg orientation (quaternion: w, x, y, z)
    # if egg_id != -1:
    #     egg_orientation = data.xquat[egg_id].copy()
    #     observation.extend(egg_orientation)
    # else:
    #     observation.extend([1.0, 0.0, 0.0, 0.0]) # Default identity quaternion

    # 4. Vector from egg to left finger (dx, dy, dz)
    # *IT WOULD BE MORE USEFULL TO USE THE DISTANCE BETWEEN EGG AND FINGERS (Implemented as vector components)
    if egg_position_valid and finger_left_id != -1:
        finger_left_position = data.xpos[finger_left_id].copy()
        vec_egg_to_finger_left = egg_position - finger_left_position # Calculate vector difference
        observation.extend(vec_egg_to_finger_left)
    else:
        observation.extend([0.0, 0.0, 0.0]) # Default vector if egg or finger is missing

    # 5. Vector from egg to right finger (dx, dy, dz)
    if egg_position_valid and finger_right_id != -1:
        finger_right_position = data.xpos[finger_right_id].copy()
        vec_egg_to_finger_right = egg_position - finger_right_position # Calculate vector difference
        observation.extend(vec_egg_to_finger_right)
    else:
        observation.extend([0.0, 0.0, 0.0]) # Default vector if egg or finger is missing

    # 6. Finger velocities (linear: left_finger_vel, right_finger_vel)
    # if finger_left_id != -1:
    #     finger_left_velocity = data.qvel[model.body_jntadr[finger_left_id]:model.body_jntadr[finger_left_id] + 3].copy()
    #     observation.extend(finger_left_velocity)
    # else:
    #     observation.extend([0.0, 0.0, 0.0]) # Default velocity

    # if finger_right_id != -1:
    #     finger_right_velocity = data.qvel[model.body_jntadr[finger_right_id]:model.body_jntadr[finger_right_id] + 3].copy()
    #     observation.extend(finger_right_velocity)
    # else:
    #     observation.extend([0.0, 0.0, 0.0]) # Default velocity

    # 7. Arm joint angles (e.g., 7 DoF arm)
    # Ensure the slice [:7] correctly corresponds to your arm's joints in qpos
    # num_arm_joints = 6 # Example, adjust if different
    arm_joint_angles = data.qpos[[7, 8, 9, 10, 11, 12]].copy()
    observation.extend(arm_joint_angles)

    # 8. Arm joint velocities
    # Ensure the slice [:7] correctly corresponds to your arm's joints in qvel
    arm_joint_velocities = data.qvel[[7, 8, 9, 10, 11, 12]].copy()
    observation.extend(arm_joint_velocities)

    # 9. Distance between egg and target (in XY plane) - Scalar
    if egg_position_valid and target_id != -1:
        target_position = data.xpos[target_id].copy()
        # Calculate distance only in the XY plane
        dist_egg_to_target_xy = egg_position - target_position
        observation.extend(dist_egg_to_target_xy)
    else:
        observation.extend([0.0, 0, 0]) # Default distance

    # 10. Binary values for task states (assuming these functions are in utils)
    # observation.extend([
    #     float(egg_at_the_start(model, data)),
    #     float(egg_on_the_floor(model, data)),
    #     float(egg_at_the_holding(model, data)),
    #     float(egg_in_target(model, data))
    # ])
    
    # 11. Sum of contact forces on the egg - Scalar
    # This sums the magnitudes of all contact forces involving the egg.
    # sum_egg_contact_force_magnitude = 0.0
    # if egg_id != -1: # Only calculate if egg exists
    #     for i in range(data.ncon):
    #         contact = data.contact[i]
    #         # Check if the egg is one of the bodies in contact
    #         # geom_bodyid maps geom ID to body ID
    #         body_id_geom1 = model.geom_bodyid[contact.geom1]
    #         body_id_geom2 = model.geom_bodyid[contact.geom2]

    #         if body_id_geom1 == egg_id or body_id_geom2 == egg_id:
    #             # contact.frame is the 6D force-torque vector in the contact frame.
    #             # We are interested in the magnitude of the force (first 3 components).
    #             force_vector = np.array([contact.frame[0], contact.frame[1], contact.frame[2]])
    #             force_magnitude = np.linalg.norm(force_vector)
    #             sum_egg_contact_force_magnitude += force_magnitude
    # observation.append(sum_egg_contact_force_magnitude)

    return np.array(observation, dtype=np.float64)

