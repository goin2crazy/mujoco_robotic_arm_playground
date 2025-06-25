import mujoco
import numpy as np
import cv2
import os
import logging  # Import the logging module

def load_model_and_data(xml_path="egg.xml"):
    """
    Loads the MuJoCo model from an XML file and creates the data object.
    This function sets up the scene.
    """
    # Check if the XML file exists
    if not os.path.exists(xml_path):
        # Use logging for error reporting
        logging.error(f"XML file not found at {xml_path}")
        raise FileNotFoundError(f"Error: XML file not found at {xml_path}")

    # Load the model from the XML file
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        logging.error(f"Error loading MuJoCo model: {e}")
        raise

    # Create the data object, which holds the state of the model
    data = mujoco.MjData(model)
    return model, data

def get_body_id(model, body_name):
    """
    Gets the ID of a body given its name.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        logging.warning(f"Body name '{body_name}' not found in the model.")
    return body_id

def check_contact(model, data, body_name1, body_name2):
    """
    Checks if two bodies are in contact.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        body_name1: Name of the first body.
        body_name2: Name of the second body.

    Returns:
        True if the bodies are in contact, False otherwise.
    """
    body_id1 = get_body_id(model, body_name1)
    body_id2 = get_body_id(model, body_name2)

    if body_id1 == -1 or body_id2 == -1:
        return False  # Return false if either body ID is invalid

    for i in range(data.ncon):
        contact = data.contact[i]
        # Changed the condition to use the body IDs.  Much more robust.
        body_id_1_from_contact = model.geom_bodyid[contact.geom1]
        body_id_2_from_contact = model.geom_bodyid[contact.geom2]
        if (body_id_1_from_contact == body_id1 and body_id_2_from_contact == body_id2) or \
                (body_id_1_from_contact == body_id2 and body_id_2_from_contact == body_id1):
            return True
    return False

    
def get_body_pos(model, data, body_name): 
            
    obj_id = get_body_id(model, body_name)
    obj_pos = np.array(data.xpos[obj_id])
    return obj_pos 

def contact_force(model, data, body1, body2):
    for contact in data.contact:
        body1_id = get_body_id(model, body1)
        body2_id = get_body_id(model, body2)
        if (contact.geom1 == body1_id and contact.geom2 == body2_id) or \
           (contact.geom1 == body2_id and contact.geom2 == body1_id):
            return contact.force
    return 0.0

def read_recorded_trajectory(filepath):
    """
    Reads a single .npz trajectory file and prints its contents.

    Args:
        filepath (str): The full path to the .npz file (e.g., "recorded_data/trajectory_20250604-120113.npz").
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    try:
        # np.load loads the .npz file.
        # allow_pickle=True is necessary if your 'infos' array contains Python objects (like dictionaries).
        # Without it, you might get an error.
        data = np.load(filepath, allow_pickle=True) 

        print(f"\n--- Reading data from: {filepath} ---")
        print(f"Available keys in the file: {list(data.keys())}\n")

        # Access each array by its key
        observations = data["observations"]
        controls = data["controls"]
        rewards = data["rewards"]
        dones = data["dones"]
        infos = data["infos"]

        print(f"Total steps recorded: {len(observations)}")
        print(f"Shape of observations array: {observations.shape}")
        print(f"Shape of controls array: {controls.shape}")
        print(f"Shape of rewards array: {rewards.shape}")
        print(f"Shape of dones array: {dones.shape}")
        # infos might have a shape like (N,) if they are individual dictionaries
        print(f"Shape of infos array: {infos.shape}") 

        print("\n--- First 5 steps (or fewer if trajectory is short) ---")
        num_steps_to_show = min(5, len(observations))

        for i in range(num_steps_to_show):
            print(f"\n--- Step {i+1} ---")
            print(f"Observation (first 5 elements): {observations[i][:5]}") # Print first few elements
            print(f"Control values: {controls[i]}")
            print(f"Reward: {rewards[i]:.4f}")
            print(f"Terminated: {dones[i]}")
            print(f"Info: {infos[i]}") # This will likely be a dictionary

        # It's good practice to close the loaded file object
        data.close()
        return data 

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    # --- IMPORTANT: Set the correct path to your .npz file ---
    # Example: If you saved a file named 'trajectory_20250604-120113.npz'
    # in a folder called 'recorded_data' in the same directory as this script.
    
    # 1. Option A: Specify a known file path
    # Make sure this path exists and points to one of your saved files.
    # You'll need to replace 'YOUR_TIMESTAMP_HERE' with an actual timestamp from your files.
    # For example: "recorded_data/trajectory_20250604-120934.npz"
    # sample_filepath = "recorded_data/trajectory_YOUR_TIMESTAMP_HERE.npz" 
    
    # 2. Option B: Find the latest file in the 'recorded_data' directory
    recorded_data_dir = "recorded_data"
    
    if not os.path.exists(recorded_data_dir):
        print(f"Error: Directory '{recorded_data_dir}' not found. Please ensure you have recorded data first.")
    else:
        npz_files = [f for f in os.listdir(recorded_data_dir) if f.endswith('.npz')]
        if not npz_files:
            print(f"No .npz files found in '{recorded_data_dir}'. Please record some data first.")
        else:
            # Sort files by modification time to get the latest one
            latest_file = max(npz_files, key=lambda f: os.path.getmtime(os.path.join(recorded_data_dir, f)))
            sample_filepath = os.path.join(recorded_data_dir, latest_file)
            print(f"Found latest recorded file: {sample_filepath}")
            read_recorded_trajectory(sample_filepath)