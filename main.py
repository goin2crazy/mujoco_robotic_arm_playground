import numpy as np
import cv2
import logging
from robo_env import MujocoRobotArmEnv # Assuming your class is in this file

def visualize_mujoco_env(env):
    """
    Visualizes the MuJoCo robot arm environment using OpenCV and allows control with keyboard.
    """
    logging.info("Starting MuJoCo environment visualization.")

    # Reset the environment to get the initial observation and info
    observation, info = env.reset()
    logging.info(f"Initial observation shape: {observation.shape}")

    # Set initial control value
    # We will apply actions as increments to the current control values,
    # similar to how it's handled in the environment's step method.
    control_increment_rate = 0.01  # This will be multiplied by the action [-1, 1]
    
    # Get the number of actuators from the action space shape
    num_actuators = env.action_space.shape[0]
    current_action = np.zeros(num_actuators, dtype=np.float32)

    while True:
        # Render the environment
        img = env.render(mode="rgb_array")
        if img is not None:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("MuJoCo Scene", img_bgr)
        else:
            logging.warning("Rendering returned None. Is the renderer initialized?")
            break

        # Keyboard interaction for control
        key = cv2.waitKey(1) & 0xFF
        action_to_take = np.zeros(num_actuators, dtype=np.float32)

        if key == ord('a'):  # Decrease first actuator control
            action_to_take[0] = -1.0
        elif key == ord('d'):  # Increase first actuator control
            action_to_take[0] = 1.0
        elif key == ord('q'):  # Exit manually with Q
            logging.info("Manual exit requested. Closing environment.")
            break
        
        # Step the environment with the determined action
        # The environment's step method handles the application of action to data.ctrl
        observation, reward, terminated, truncated, info = env.step(action_to_take)
        print(reward)

        if terminated:
            logging.info(f"Episode finished. Final reward: {reward}")
            # You might want to reset the environment here to start a new episode
            # or break if you only want one episode per run.
            observation, info = env.reset()
            logging.info("Environment reset for a new episode.")


        # Logging information
        # The info dictionary from the environment already contains the relevant states
        logging.debug(f"Observation shape: {observation.shape}")
        logging.debug(f"Current reward: {reward}")
        logging.debug(f"Egg at start: {info['egg_at_start']}")
        logging.debug(f"Egg on floor: {info['egg_on_floor']}")
        logging.debug(f"Egg at holding: {info['egg_at_holding']}")
        logging.debug(f"Egg in target: {info['egg_in_target']}")
        logging.debug(f"Simulation time: {info['time']:.3f}")

    cv2.destroyAllWindows()
    env.close() # Ensure the environment resources are properly released


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define the path to your MuJoCo model
    model_xml_path = "egg_final.xml"

    try:
        # Create an instance of your custom environment
        env = MujocoRobotArmEnv(model_path=model_xml_path, 
                                moving_rate=0.01, 
                                roughness_penalty_scale=0.01)
        # Run the visualization loop
        visualize_mujoco_env(env)

    except Exception as e:
        logging.critical(f"An error occurred during environment initialization or visualization: {e}")