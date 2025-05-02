from utils import * 


def get_observation(model, data):
    """
    Returns the observation vector.
    """
    # Get body IDs
    egg_id = get_body_id(model, "egg")
    finger_left_id = get_body_id(model, "arm_finger_left")
    finger_right_id = get_body_id(model, "arm_finger_right")
    target_id = get_body_id(model, "egg_base_target")

    # Initialize an empty list to hold the observation values.
    observation = []

    # 1. Egg position (x, y, z)
    if egg_id != -1:
        egg_position = data.xpos[egg_id].copy()
        observation.extend(egg_position)
    else:
        observation.extend([0, 0, 0])  # Default if egg ID not found

    # 2. Egg velocity (x, y, z)
    if egg_id != -1:
        egg_velocity = data.qvel[egg_id * 6:egg_id * 6 + 3].copy() # Linear velocity
        observation.extend(egg_velocity)
    else:
        observation.extend([0, 0, 0])

    # 3. Egg orientation (quaternion)
    if egg_id != -1:
        egg_orientation = data.xquat[egg_id].copy()
        observation.extend(egg_orientation)
    else:
        observation.extend([1, 0, 0, 0])  # Default quaternion

    # 4. Finger positions (left and right)
    if finger_left_id != -1:
        finger_left_position = data.xpos[finger_left_id].copy()
        observation.extend(finger_left_position)
    else:
        observation.extend([0, 0, 0])

    if finger_right_id != -1:
        finger_right_position = data.xpos[finger_right_id].copy()
        observation.extend(finger_right_position)
    else:
        observation.extend([0, 0, 0])

    # 5. Finger velocities (left and right)
    if finger_left_id != -1:
        finger_left_velocity = data.qvel[finger_left_id * 6:finger_left_id * 6 + 3].copy()
        observation.extend(finger_left_velocity)
    else:
        observation.extend([0, 0, 0])

    if finger_right_id != -1:
        finger_right_velocity = data.qvel[finger_right_id * 6:finger_right_id * 6 + 3].copy()
        observation.extend(finger_right_velocity)
    else:
        observation.extend([0, 0, 0])

    # 6. Arm joint angles
    arm_joint_angles = data.qpos[:7].copy()  # Assuming 7 joints, adjust as needed.
    observation.extend(arm_joint_angles)

    # 7. Arm joint velocities
    arm_joint_velocities = data.qvel[:7].copy()
    observation.extend(arm_joint_velocities)

    # 8. Target position
    if target_id != -1:
        target_position = data.xpos[target_id].copy()
        observation.extend(target_position)
    else:
        observation.extend([0, 0, 0])

    # 9. Distance between egg and target
    if egg_id != -1 and target_id != -1:
        dist_egg_to_target = np.linalg.norm(data.xpos[egg_id][:2] - data.xpos[target_id][:2])
        observation.append(dist_egg_to_target)
    else:
        observation.append(0)

    # 10. Binary values for task states
    observation.extend([
        egg_at_the_start(model, data),
        egg_on_the_floor(model, data),
        egg_at_the_holding(model, data),
        egg_in_target(model, data)
    ])
    
    # contact forces.
    contact_forces = []
    for i in range(data.ncon):
        contact = data.contact[i]
        body_id_1_from_contact = model.geom_bodyid[contact.geom1]
        body_id_2_from_contact = model.geom_bodyid[contact.geom2]
        if body_id_1_from_contact == egg_id or body_id_2_from_contact == egg_id:
            # Sum the forces for this contact
            force = np.sqrt(contact.frame[0]**2 + contact.frame[1]**2 + contact.frame[2]**2)
            contact_forces.append(force)
    if len(contact_forces) > 0:
        observation.append(sum(contact_forces))
    else:
        observation.append(0)

    return np.array(observation, dtype=np.float64) # Ensure the data type is float64

