import mujoco
import numpy as np
import cv2
import os
import time  # Import the time module
import logging  # Import the logging module

from utils import * 
from observation import get_observation
from states import * 

def visualize_mujoco(model, data):
    """
    Visualizes the MuJoCo model using OpenCV.
    """
    # Initialize the renderer.
    with mujoco.Renderer(model) as renderer:
        start_time = time.time()
        egg_start_pos = data.xpos[get_body_id(model, "egg")][:2].copy()  # stores the initial xy position of the egg.
        if np.any(np.isnan(egg_start_pos)):
            egg_start_pos = np.array([0, 0])

        egg_dist_to_target = 99999
        # Main simulation loop
        while True:
            # Simulate the model.
            try:
                mujoco.mj_step(model, data)
            except Exception as e:
                logging.error(f"Error in mujoco.mj_step: {e}")
                break  # Exit the loop on error

            # Update and render the scene.
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            img = renderer.render()  # Get the rendered image

            # Convert the image to a format OpenCV can use (BGR)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Display the image using OpenCV
            cv2.imshow("MuJoCo Scene", img_bgr)
            cv2.waitKey(1)  # 1 millisecond delay for real-time update

            # Get and print the observation vector.
            observation = get_observation(model, data)
            rewards, egg_dist_to_target = reward_function_grasp(model, data, prev_dist=egg_dist_to_target) 

            logging.info(f"Observation: {observation}")

            # Print contact information.  Consider using logging here as well.
            logging.info(f"Egg at the start: {egg_at_the_start(model, data)}")
            logging.info(f"Egg on the floor: {egg_on_the_floor(model, data)}")
            logging.info(f"Egg at the holding: {egg_at_the_holding(model, data)}")
            logging.info(f"Egg in target: {egg_in_target(model, data)}")

            # Check for session end.
            done, addictional_reward = check_session_end(model, data, start_time, egg_start_pos)
            rewards += addictional_reward

            logging.info(f"Current rewards: {rewards}")

            if done: 
                break 
            
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        model, data = load_model_and_data("egg_final.xml")
        visualize_mujoco(model, data)
    except Exception as e:
        logging.critical(f"An error occurred: {e}")  # Log the error and exit
