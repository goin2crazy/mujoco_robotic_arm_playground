
import random
from pprint import pprint
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import mujoco 
import logging 
from observation import * 
from states import *

class MujocoRobotArmEnvReachTask(gym.Env):
    """
    A Gymnasium environment for controlling a MuJoCo robot arm.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 model_path="your_model_reach.xml", 
                 moving_rate=1, 
                 min_move=-1, 
                 max_move=1, 
                 additional_reward_scale=1, 
                 reach_to_object_name = 'body_to_reach', 
                 ):  # Add model_path
        """
        Initializes the environment.
        Args:
            model_path (str): Path to the MuJoCo XML model file.
        """
        super().__init__()
        self.model_path = model_path
        self.load_mujoco_env()
        
        #initialize the main mujoco env parameters 
        self.data = mujoco.MjData(self.model)
        self.steps_made_in_episode = 0       

        #initialize the main calculation fns 
        self.get_observation = lambda model, data: get_observation_reach_task(model=model, data=data, body_to_reach_name=reach_to_object_name)
        self.main_reward_fn = lambda model, data: improved_reward_function_reach_task(model, data, reach_to_body_name=reach_to_object_name)

        self.additional_reward_scale = additional_reward_scale


        self.action_space = spaces.Box(low=-1, high=1, shape=self.data.ctrl.shape, dtype=np.float32) # Placeholder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=np.array(self.get_observation(self.model, self.data).values()).shape, dtype=np.float64) # Placeholder

        self.renderer = None  # Initialize renderer lazily
        self.moving_rate = moving_rate
        self.min_move = min_move
        self.max_move=max_move

    def load_mujoco_env(self): 
        with open(self.model_path, 'r', encoding="utf-8") as f: 
            xlm_str = f.read() 

        # Randomize x in range 0 to 0.8
        x = random.uniform(0, 0.8)

        # Randomize z in range -0.5 to 0.8
        z = random.uniform(-0.5, 0.8)

        # Randomize y in range -0.3 to 0.8
        y = random.uniform(-0.2, 0.8)

        randomized_xlm_string = (xlm_str
                                .replace("{reach_body_pos_x}", f"{x:.3f}")
                                .replace("{reach_body_pos_y}", f"{y:.3f}")
                                .replace("{reach_body_pos_z}", f"{z:.3f}"))
        # Load the new model from the XML string
        self.model = mujoco.MjModel.from_xml_string(randomized_xlm_string)
        # A new model requires a new data instance
        self.data = mujoco.MjData(self.model)

    def step(self, action):
        """
        Simulates one step of the environment.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Apply the action (e.g., set joint torques).
        assert action.shape == self.action_space.shape # Important: Check shape

        try:
            diff = action - self.data.ctrl  

            # Maximum change per step
            alpha = self.moving_rate

            # Clamp each component of diff to [-alpha, +alpha]
            capped = np.clip(diff, -alpha, alpha)  

            # Apply the capped update
            self.data.ctrl[:] = np.clip(self.data.ctrl + capped, self.min_move, self.max_move)

            mujoco.mj_step(self.model, self.data)
        except Exception as e:
            logging.error(f"Error in mj_step: {e}")
            # Handle the error appropriately (e.g., set terminated/truncated, return a default observation)
            observation = np.zeros(self.observation_space.shape)  # Or some other safe value
            return observation, 0, True, False, {"error": str(e)}

        # Get the observation.
        observation_dict = self.get_observation(self.model, self.data)
        observation = np.array(observation_dict.values())

        reward = 0
        # Get reward and termination from unified function
        reward_fn_output = self.main_reward_fn(
            self.model, self.data,
        )
        reach_reward = reward_fn_output['reach']

        # Check for the end of the session.
        terminated, additional_reward = check_session_end(self.model, self.data, steps = self.steps_made_in_episode) #TODO start time
        
        if terminated:
            self.steps_made_in_episode = 0
        else: 
            self.steps_made_in_episode+=1 

        reward += reach_reward
        reward += additional_reward * self.additional_reward_scale

        info = {
            **observation_dict
        }

        # Important:  Return a valid tuple, even if there's an error.
        return observation, reward, terminated, False, info  # truncated is always false

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional reset options.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed, options=options)  # Handle seed

        if random.random() < 0.05:
            self.load_mujoco_env()
        else: 
            # Reset MuJoCo state.
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data) # Forward simulation to ensure initial state is correct

        # Get the initial observation.
        observation_dict = self.get_observation(self.model, self.data)
        observation = np.array(observation_dict.values())

        info = {**observation_dict}  # Add any relevant info here

        return observation, info

    def render(self, mode="human"):
        """
        Renders the environment.

        Args:
            mode (str): The rendering mode ("human" or "rgb_array").

        Returns:
            np.ndarray or None: The rendered image if mode is "rgb_array", None otherwise.
        """
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {mode}")

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)

        # Update the scene.
        mujoco.mj_forward(self.model, self.data) #  Make sure data is consistent before rendering
        self.renderer.update_scene(self.data)

        if mode == "human":
            self.renderer.render()  # Render to the default GLFW window
            return None
        elif mode == "rgb_array":
            img = self.renderer.render()
            return img  # Return the raw image data
        else:
            return None

    def close(self):
        """
        Closes the environment and releases resources.
        """
        if self.renderer:
            self.renderer.close()
        self.renderer = None
        # No need to close self.model or self.data, they don't have close() methods in mujoco

    def __del__(self):
        # Ensure resources are cleaned up.  Important for preventing memory leaks.
        self.close()

