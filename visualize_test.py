import numpy as np
import cv2
import logging
import os # Import os for path manipulation
import datetime # Import datetime for unique filenames

# Import also libs for simulating the the vectorized enviroment where agent was trained
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.ppo import PPO 
# Make sure MujocoRobotArmEnv is importable from your current directory or path
from robo_env import MujocoRobotArmEnv, MujocoRobotArmEnv_Vanilla

from states import *

def visualize_mujoco_env(env, agent_model):
    """
    Visualizes the MuJoCo robot arm environment using OpenCV and allows control with keyboard.
    Directly manipulates dummy_action for smoother, immediate control feedback.
    Also records trajectory data.
    """
    logging.info("Starting MuJoCo environment visualization.")

    # Initialize the data
    # Reset the environment to get the initial observation and info
    observation, info = env.reset()
    logging.info(f"Initial observation shape: {observation.shape}")

    vec_env = make_vec_env(lambda: env, n_envs=1)
    vec_env = VecNormalize(vec_env, training=False, norm_obs=True, norm_reward=True,)

    while True:
        # Render the environment
        img = env.render(mode="rgb_array")
        if img is not None:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("MuJoCo Scene", img_bgr)
        else:
            logging.warning("Rendering returned None. Is the renderer initialized?")
            break

        
        obs = vec_env.normalize_obs(observation)
        dummy_action, _ = agent_model.predict(obs)
        # The control values (dummy_action) have already been set manually above.
        # The environment's step method will now use these pre-set control values
        # when it calls mujoco.mj_step(), and then calculate rewards, observations, etc.
        observation, reward, terminated, truncated, info = env.step(dummy_action)

        if terminated:
            logging.info(f"Episode finished. Final reward: {reward}")
            # Reset the environment to start a new episode
            observation, info = env.reset()
            logging.info("Environment reset for a new episode. Recorder also reset.")

        # Logging information (can be set to DEBUG level for less verbose output)
        logging.debug(f"Observation shape: {observation.shape}")
        logging.debug(f"Current dummy_action: {dummy_action}") 
        logging.debug(f"Current reward: {reward}")
        
        print(f"Current reward: {reward} Current actions: {dummy_action} ") # Keeping your print for direct feedback

    cv2.destroyAllWindows()
    env.close() # Ensure the environment resources are properly released

def load_model(checkpoint_path): 
    return PPO.load(checkpoint_path)

if __name__ == "__main__": 
    model = load_model("runs\V 32 6m\ppo_5m_arch_advanced.zip")

    env = MujocoRobotArmEnv("egg_final.xml", 
                           roughness_penalty_scale=10,
                           moving_rate=5e-4, 
                          additional_reward_scale=0.1,
                          reward_fn = reward_function_grasp_v2)
    visualize_mujoco_env(env, model)