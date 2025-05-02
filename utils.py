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

def egg_at_the_start(model, data):
    """
    Checks if the egg_base_start body is touching the egg body.
    """
    return check_contact(model, data, "egg_base_start", "egg")

def egg_on_the_floor(model, data):
    """
    Checks if the egg body is touching the floor body.
    """
    return check_contact(model, data, "egg", "floor")

def egg_at_the_holding(model, data):
    """
    Checks if only arm_finger_right and arm_finger_left are touching the egg.
    """
    egg_id = get_body_id(model, "egg")
    finger_left_id = get_body_id(model, "arm_finger_left")
    finger_right_id = get_body_id(model, "arm_finger_right")
    if egg_id == -1 or finger_left_id == -1 or finger_right_id == -1:
        return False

    touching_finger_left = check_contact(model, data, "egg", "arm_finger_right")
    touching_finger_right = check_contact(model, data, "egg", "arm_finger_left")

    # Check that only the fingers are touching the egg.
    num_contacts = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        body_id_1_from_contact = model.geom_bodyid[contact.geom1]
        body_id_2_from_contact = model.geom_bodyid[contact.geom2]
        if body_id_1_from_contact == egg_id or body_id_2_from_contact == egg_id:
            if body_id_1_from_contact == finger_left_id or body_id_2_from_contact == finger_left_id:
                num_contacts += 1
            elif body_id_1_from_contact == finger_right_id or body_id_2_from_contact == finger_right_id:
                num_contacts += 1

    return touching_finger_left and touching_finger_right and num_contacts == 2

def egg_in_target(model, data):
    """
    Checks if the egg body is touching the egg_base_target.
    """
    return check_contact(model, data, "egg", "egg_base_target")