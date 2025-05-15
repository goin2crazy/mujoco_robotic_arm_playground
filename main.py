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
    Visualizes the MuJoCo model using OpenCV and allows control with keyboard.
    """
    with mujoco.Renderer(model) as renderer:
        start_time = time.time()
        egg_start_pos = data.xpos[get_body_id(model, "egg")][:2].copy()
        if np.any(np.isnan(egg_start_pos)):
            egg_start_pos = np.array([0, 0])

        egg_dist_to_target = 99999

        # Set initial control value
        control_increment = 0.01  # How much it increases/decreases per keypress
        min_control = -1.0
        max_control = 1.0

        reward_counter = 0 

        while True:
            try:
                mujoco.mj_step(model, data)
            except Exception as e:
                logging.error(f"Error in mujoco.mj_step: {e}")
                break

            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("MuJoCo Scene", img_bgr)

            # 🧠 Keyboard interaction
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):  # Decrease control
                data.ctrl = [max(min_control, data.ctrl[0] - control_increment)] + [0] * (len(data.ctrl)-1)
            elif key == ord('d'):  # Increase control
                data.ctrl = [min(max_control, data.ctrl[0] + control_increment)]+ [0] * (len(data.ctrl)-1)
            elif key == ord('q'):  # Exit manually with Q
                logging.info("Manual exit requested.")
                break

            # 📝 Observation + Reward logic
            observation = get_observation(model, data)
            rewards, egg_dist_to_target = reward_function_grasp(model, data, prev_dist=egg_dist_to_target)

            # logging.info(f"Observation: {observation}")
            # logging.info(f"data.ctrl[0]: {data.ctrl[0]}")
            # logging.info(f"Egg at the start: {egg_at_the_start(model, data)}")
            # logging.info(f"Egg on the floor: {egg_on_the_floor(model, data)}")
            # logging.info(f"Egg at the holding: {egg_at_the_holding(model, data)}")
            # logging.info(f"Egg in target: {egg_in_target(model, data)}")

            done, additional_reward = check_session_end(model, data, start_time, egg_start_pos)
            rewards += additional_reward


            reward_counter +=1 
            if reward_counter > 100: 
                reward_counter = 0 
                logging.info(f"Current rewards: {rewards}")

            if done:
                logging.info("Episode finished.")
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
