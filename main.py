import numpy as np
import cv2
import logging
import os # Import os for path manipulation
import datetime # Import datetime for unique filenames

# Make sure MujocoRobotArmEnv is importable from your current directory or path
from robo_env import MujocoRobotArmEnv 

# --- New DataRecorder Class ---
class DataRecorder:
    def __init__(self, save_dir="recorded_data"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True) # Ensure the directory exists
        self.reset()

    def reset(self):
        """Resets the stored data for a new recording session/episode."""
        self.observations = []
        self.controls = [] # Corresponds to dummy_action at each step
        self.rewards = []
        self.dones = [] # For termination status
        self.infos = [] # For additional info from the environment

    def record_step(self, observation, control, reward, terminated, info):
        """Records data for a single simulation step."""
        self.observations.append(observation.copy()) # .copy() is important for NumPy arrays
        self.controls.append(control.copy())
        self.rewards.append(reward)
        self.dones.append(terminated)
        self.infos.append(info.copy()) # .copy() if info dict contains mutable objects

    def save_data(self, filename=None):
        """Saves all recorded data to a .npz file."""
        if not self.observations:
            logging.warning("No data recorded to save.")
            return

        if filename is None:
            # Generate a unique filename using timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"trajectory_{timestamp}.npz"
        
        filepath = os.path.join(self.save_dir, filename)

        # Convert lists to NumPy arrays for efficient saving
        data_to_save = {
            "observations": np.array(self.observations),
            "controls": np.array(self.controls),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            # Flatten info dictionaries or choose specific keys if they are complex
            # For simplicity, we'll convert them to an object array.
            "infos": np.array(self.infos, dtype=object) 
        }

        try:
            np.savez_compressed(filepath, **data_to_save)
            logging.info(f"Data saved successfully to {filepath}")
            self.reset() # Reset recorder after saving
        except Exception as e:
            logging.error(f"Error saving data to {filepath}: {e}")

# --- End DataRecorder Class ---


def visualize_mujoco_env(env):
    """
    Visualizes the MuJoCo robot arm environment using OpenCV and allows control with keyboard.
    Directly manipulates dummy_action for smoother, immediate control feedback.
    Also records trajectory data.
    """
    logging.info("Starting MuJoCo environment visualization.")

    # Initialize the data recorder
    recorder = DataRecorder()

    # Reset the environment to get the initial observation and info
    observation, info = env.reset()
    logging.info(f"Initial observation shape: {observation.shape}")

    num_actuators = env.action_space.shape[0]
    
    # Define control parameters for smoothness
    smoothness_coef = 0.3 
    control_change_amount = 1

    # Use the actual limits from the environment's model
    # It's generally safer to get these directly from the model if available
    # rather than hardcoding them, as they might vary with different models.
    # If your model indeed has fixed -1 to 1 limits for all 5 actuators,
    # then your hardcoded min_ctrl/max_ctrl lists are fine.
    # Example using model limits:
    # min_ctrl = env.model.actuator_ctrlrange[:, 0] 
    # max_ctrl = env.model.actuator_ctrlrange[:, 1]
    min_ctrl = [-1] * 5 
    max_ctrl = [1] * 5

    dummy_action = env.data.ctrl
    while True:
        # Render the environment
        img = env.render(mode="rgb_array")
        if img is not None:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("MuJoCo Scene", img_bgr)
        else:
            logging.warning("Rendering returned None. Is the renderer initialized?")
            break

        # Get keyboard input
        key = cv2.waitKey(1) & 0xFF

        # --- Direct manipulation of dummy_action with smoothness ---
        # Joint 0: p1 (A <-> D)
        if key == ord('a'):  # Decrease p1
            dummy_action[0] = np.clip(dummy_action[0] - (smoothness_coef * control_change_amount), min_ctrl[0], max_ctrl[0])
        elif key == ord('d'):  # Increase p1
            dummy_action[0] = np.clip(dummy_action[0] + (smoothness_coef * control_change_amount), min_ctrl[0], max_ctrl[0])
        
        # Joint 1: p2_arm (W <-> S)
        elif key == ord('w'):  # Decrease p2_arm
            dummy_action[1] = np.clip(dummy_action[1] - (smoothness_coef * control_change_amount), min_ctrl[1], max_ctrl[1])
        elif key == ord('s'):  # Increase p2_arm
            dummy_action[1] = np.clip(dummy_action[1] + (smoothness_coef * control_change_amount), min_ctrl[1], max_ctrl[1])

        # Joint 2: p1_arm2 (E <-> F)
        elif key == ord('e'):  # Decrease p1_arm2
            dummy_action[2] = np.clip(dummy_action[2] - (smoothness_coef * control_change_amount), min_ctrl[2], max_ctrl[2])
        elif key == ord('f'):  # Increase p1_arm2
            dummy_action[2] = np.clip(dummy_action[2] + (smoothness_coef * control_change_amount), min_ctrl[2], max_ctrl[2])

        # Joints 3 and 4: fingers (C <-> V)
        # Assuming actuators 3 and 4 control the fingers
        elif key == ord('c'):  # Decrease fingers
            if num_actuators > 3: 
                dummy_action[3] = np.clip(dummy_action[3] - (smoothness_coef * control_change_amount), min_ctrl[3], max_ctrl[3])
            if num_actuators > 4: # If there's a 5th actuator for the other finger
                dummy_action[4] = np.clip(dummy_action[4] - (smoothness_coef * control_change_amount), min_ctrl[4], max_ctrl[4])
        elif key == ord('v'):  # Increase fingers
            if num_actuators > 3:
                dummy_action[3] = np.clip(dummy_action[3] + (smoothness_coef * control_change_amount), min_ctrl[3], max_ctrl[3])
            if num_actuators > 4:
                dummy_action[4] = np.clip(dummy_action[4] + (smoothness_coef * control_change_amount), min_ctrl[4], max_ctrl[4])

        # --- New: Keybind for Saving Data ---
        elif key == ord('g'): # Press 'S' to save the current trajectory
            logging.info("Saving recorded data...")
            recorder.save_data()
            # The recorder automatically resets after saving, so it's ready for a new trajectory.
            continue # Skip stepping the environment for this frame, as 's' is a command

        # Exit manually with Q
        elif key == ord('q'):  
            logging.info("Manual exit requested. Closing environment.")
            # Optionally save unsaved data before exiting
            if recorder.observations: # Check if there's any data pending
                logging.info("Saving remaining data before exit.")
                recorder.save_data(filename="partial_trajectory_on_exit_2.npz")
            break
        
        # Step the environment with a dummy action (all zeros)
        # The control values (dummy_action) have already been set manually above.
        # The environment's step method will now use these pre-set control values
        # when it calls mujoco.mj_step(), and then calculate rewards, observations, etc.
        observation, reward, terminated, truncated, info = env.step(dummy_action)

        # --- Record data after each step ---
        recorder.record_step(observation, dummy_action, reward, terminated, info)

        if terminated:
            logging.info(f"Episode finished. Final reward: {reward}")
            # Reset the environment to start a new episode
            observation, info = env.reset()
            logging.info("Environment reset for a new episode. Recorder also reset.")
            recorder.reset() # Also reset the recorder for a new episode

        # Logging information (can be set to DEBUG level for less verbose output)
        logging.debug(f"Observation shape: {observation.shape}")
        logging.debug(f"Current dummy_action: {dummy_action}") 
        logging.debug(f"Current reward: {reward}")
        
        print(f"Current reward: {reward} Current actions: {dummy_action} ") # Keeping your print for direct feedback

    cv2.destroyAllWindows()
    env.close() # Ensure the environment resources are properly released


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define the path to your MuJoCo model
    model_xml_path = "egg_final.xml"

    try:
        # Create an instance of your custom environment
        env = MujocoRobotArmEnv(model_path=model_xml_path, roughness_penalty_scale=100, moving_rate=5e-4) 
        
        # Run the visualization loop
        visualize_mujoco_env(env)

    except Exception as e:
        logging.critical(f"An error occurred during environment initialization or visualization: {e}")