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
    if egg_id != -1:
        # Assuming egg_id is a direct index or get_body_id maps to an index
        # suitable for qvel structure where each body has 6 DoFs (3 lin, 3 ang vel).
        # This extracts the linear velocity part.
        egg_velocity = data.qvel[model.body_jntadr[egg_id]:model.body_jntadr[egg_id] + 3].copy()
        observation.extend(egg_velocity)
    else:
        observation.extend([0.0, 0.0, 0.0]) # Default velocity

    # 3. Egg orientation (quaternion: w, x, y, z)
    if egg_id != -1:
        egg_orientation = data.xquat[egg_id].copy()
        observation.extend(egg_orientation)
    else:
        observation.extend([1.0, 0.0, 0.0, 0.0]) # Default identity quaternion

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
    if finger_left_id != -1:
        finger_left_velocity = data.qvel[model.body_jntadr[finger_left_id]:model.body_jntadr[finger_left_id] + 3].copy()
        observation.extend(finger_left_velocity)
    else:
        observation.extend([0.0, 0.0, 0.0]) # Default velocity

    if finger_right_id != -1:
        finger_right_velocity = data.qvel[model.body_jntadr[finger_right_id]:model.body_jntadr[finger_right_id] + 3].copy()
        observation.extend(finger_right_velocity)
    else:
        observation.extend([0.0, 0.0, 0.0]) # Default velocity

    # 7. Arm joint angles (e.g., 7 DoF arm)
    # Ensure the slice [:7] correctly corresponds to your arm's joints in qpos
    num_arm_joints = 7 # Example, adjust if different
    arm_joint_angles = data.qpos[:num_arm_joints].copy()
    observation.extend(arm_joint_angles)

    # 8. Arm joint velocities
    # Ensure the slice [:7] correctly corresponds to your arm's joints in qvel
    arm_joint_velocities = data.qvel[:num_arm_joints].copy()
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
    observation.extend([
        float(egg_at_the_start(model, data)),
        float(egg_on_the_floor(model, data)),
        float(egg_at_the_holding(model, data)),
        float(egg_in_target(model, data))
    ])
    
    # 11. Sum of contact forces on the egg - Scalar
    # This sums the magnitudes of all contact forces involving the egg.
    sum_egg_contact_force_magnitude = 0.0
    if egg_id != -1: # Only calculate if egg exists
        for i in range(data.ncon):
            contact = data.contact[i]
            # Check if the egg is one of the bodies in contact
            # geom_bodyid maps geom ID to body ID
            body_id_geom1 = model.geom_bodyid[contact.geom1]
            body_id_geom2 = model.geom_bodyid[contact.geom2]

            if body_id_geom1 == egg_id or body_id_geom2 == egg_id:
                # contact.frame is the 6D force-torque vector in the contact frame.
                # We are interested in the magnitude of the force (first 3 components).
                force_vector = np.array([contact.frame[0], contact.frame[1], contact.frame[2]])
                force_magnitude = np.linalg.norm(force_vector)
                sum_egg_contact_force_magnitude += force_magnitude
    observation.append(sum_egg_contact_force_magnitude)

    return np.array(observation, dtype=np.float64)


if __name__ == '__main__':
    # This is a placeholder for testing.
    # To run this, you would need mock 'model' and 'data' objects
    # and the 'utils' functions.
    print("get_observation function defined.")
    print("To test, you'd need to mock MuJoCo model, data, and utils.")

    # Example of how you might mock (very simplified):
    class MockModel:
        def __init__(self):
            # Example: body_jntadr[0] is start for body 0, body_jntadr[1] for body 1, etc.
            # For free bodies (like egg, fingers if they are free), each typically has 1 joint with 6 DoFs (3 trans, 3 rot).
            # If they are part of a kinematic chain, their DoFs might be different or part of a larger joint.
            # This mock assumes body_jntadr points to the start of the 6 DoFs for these bodies in qvel.
            self.body_jntadr = np.array([0, 6, 12, 18]) # qvel index for egg, fingerL, fingerR, target
            self.geom_bodyid = np.array([0, 0, 1, 1, 2, 3]) # Example geom to body mapping

    class MockData:
        def __init__(self):
            self.xpos = np.array([
                [0.0, 0.0, 0.5],  # Egg (ID 0)
                [0.1, 0.1, 0.2],  # Finger Left (ID 1)
                [-0.1, 0.1, 0.2], # Finger Right (ID 2)
                [0.0, 0.0, 0.0]   # Target (ID 3)
            ])
            self.qvel = np.zeros(24 + 7) # Placeholder for velocities (e.g. 4 bodies * 6 DoF + 7 arm DoF)
                                      # Arm qvels are assumed to be the first 7 elements.
            self.qvel[0:7] = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007] # Arm joint velocities
            # Velocities for bodies (using body_jntadr from MockModel)
            # Egg (ID 0, starts at qvel[model.body_jntadr[0]=0] in this example, but arm is first, so let's adjust)
            # For this mock, let's assume arm joints are first 7, then body DoFs follow.
            # So, if model.body_jntadr[0] for egg is, say, 7 (after arm joints).
            # Let's redefine mock_model.body_jntadr for clarity with arm joints.
            # Let arm_dof = 7.
            # body_jntadr for egg (body 0) = arm_dof = 7
            # body_jntadr for finger L (body 1) = arm_dof + 6 = 13
            # body_jntadr for finger R (body 2) = arm_dof + 12 = 19
            # body_jntadr for target (body 3) = arm_dof + 18 = 25
            # Total qvel size = arm_dof + 4*6 = 7 + 24 = 31
            # Redefine qvel size for this structure
            self.qvel = np.zeros(31)
            self.qvel[0:7] = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007] # Arm joint velocities

            # Egg linear velocity (at index model.body_jntadr[0] which is 7 for this mock)
            self.qvel[7:10] = [0.01, 0.0, -0.01]
            # Finger L linear velocity (at index model.body_jntadr[1] which is 13)
            self.qvel[13:16] = [0.0, 0.01, 0.0]
            # Finger R linear velocity (at index model.body_jntadr[2] which is 19)
            self.qvel[19:22] = [0.0, -0.01, 0.0]

            self.xquat = np.array([
                [1.0, 0.0, 0.0, 0.0], # Egg orientation
                [1.0, 0.0, 0.0, 0.0], # Finger L
                [1.0, 0.0, 0.0, 0.0], # Finger R
                [1.0, 0.0, 0.0, 0.0]  # Target
            ])
            self.qpos = np.zeros(7) # Arm joint angles (first 7 of qpos)
            self.ncon = 1 # Number of contacts
            self.contact = [MockContact()]

    class MockContact:
        def __init__(self):
            self.geom1 = 0 # Geom ID of first body in contact (e.g., part of egg, maps to body 0)
            self.geom2 = 2 # Geom ID of second body in contact (e.g., part of finger L, maps to body 1)
            self.frame = np.array([0.1, 0.2, 0.0, 0, 0, 0]) # Force-torque vector [fx,fy,fz, tx,ty,tz]


    # Mock utility functions (these would normally be in utils.py)
    def get_body_id(model, name):
        if name == "egg": return 0
        if name == "arm_finger_left": return 1
        if name == "arm_finger_right": return 2
        if name == "egg_base_target": return 3
        return -1

    def egg_at_the_start(model, data): return True
    def egg_on_the_floor(model, data): return False
    def egg_at_the_holding(model, data): return False
    def egg_in_target(model, data): return False

    # --- Test Run ---
    mock_model = MockModel()
    # Adjust mock_model.body_jntadr for the scenario where arm qvels are first
    mock_model.body_jntadr = np.array([7, 13, 19, 25]) # Start indices in qvel after 7 arm DoFs

    mock_data = MockData()


    print("\n--- Mock Test Run (with Vector Distances) ---")
    try:
        obs = get_observation(mock_model, mock_data)
        print(f"Observation vector (length {len(obs)}):")
        print(obs)

        # Expected components based on mock data and logic:
        # 2. Egg velocity: [0.01, 0.0, -0.01] (3 elements)
        # 3. Egg orientation: [1.0, 0.0, 0.0, 0.0] (4 elements)
        # 4. Vec egg to L finger: egg_pos - finger_L_pos = [0,0,0.5] - [0.1,0.1,0.2] = [-0.1, -0.1, 0.3] (3 elements)
        # 5. Vec egg to R finger: egg_pos - finger_R_pos = [0,0,0.5] - [-0.1,0.1,0.2] = [0.1, -0.1, 0.3] (3 elements)
        # 6. Finger L vel: [0.0, 0.01, 0.0], Finger R vel: [0.0, -0.01, 0.0] (6 elements)
        # 7. Arm joint angles: data.qpos[:7] = [0,0,0,0,0,0,0] (7 elements)
        # 8. Arm joint velocities: data.qvel[:7] = [0.001 to 0.007] (7 elements)
        # 9. Dist egg to target XY (scalar): norm([0,0] - [0,0]) = 0 (1 element)
        # 10. Binary states: [1.0, 0.0, 0.0, 0.0] (4 elements)
        # 11. Contact force sum (scalar): norm([0.1, 0.2, 0.0]) approx 0.2236 (1 element)
        # Total length: 3 + 4 + 3 + 3 + 6 + 7 + 7 + 1 + 4 + 1 = 39

        print(f"\nExpected total length: 39, Actual length: {len(obs)}")
        print("\nBreakdown of expected values (approximate for forces):")
        print(f"Egg vel (3): {obs[0:3]}")
        print(f"Egg quat (4): {obs[3:7]}")
        print(f"Vec egg-Lfinger (3): {obs[7:10]}")
        print(f"Vec egg-Rfinger (3): {obs[10:13]}")
        print(f"Finger L vel (3): {obs[13:16]}, Finger R vel (3): {obs[16:19]}")
        print(f"Arm angles (7): {obs[19:26]}")
        print(f"Arm vels (7): {obs[26:33]}")
        print(f"Dist egg-target XY (1): {obs[33]:.4f}")
        print(f"Binary states (4): {obs[34:38]}")
        print(f"Contact force sum (1): {obs[38]:.4f}")


    except ImportError as e:
        print(f"Import error: {e}. Ensure 'utils.py' is available or mock its functions.")
    except AttributeError as e:
        print(f"AttributeError: {e}. Mock objects might be missing attributes from a real MuJoCo model/data (check body_jntadr, geom_bodyid, etc.).")
    except IndexError as e:
        print(f"IndexError: {e}. Check array sizes and indexing in mock data or function logic, especially for qvel and body_jntadr.")
    except Exception as e:
        print(f"An error occurred during the mock test: {e}")

