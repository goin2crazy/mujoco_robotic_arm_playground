from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import mujoco 
import logging 
from observation import * 
from states import *

class MujocoRobotArmEnv(gym.Env):
    """
    A Gymnasium environment for controlling a MuJoCo robot arm.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, model_path="your_model.xml", moving_rate=1e-3):  # Add model_path
        """
        Initializes the environment.
        Args:
            model_path (str): Path to the MuJoCo XML model file.
        """
        super().__init__()

        # Load the MuJoCo model.
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            raise ValueError(f"Error loading MuJoCo model from {model_path}: {e}")

        self.data = mujoco.MjData(self.model)

        # Define action and observation spaces.  CRITICAL.
        # Example:  Action space is the joint torque limits.
        # low = self.model.actuator_ctrlrange[:, 0]
        # high = self.model.actuator_ctrlrange[:, 1]
        # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Example: Observation space is joint positions and velocities
        # num_joints = self.model.njnt
        # obs_dim = num_joints * 2  # For positions and velocities
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=self.data.ctrl.shape, dtype=np.float32) # Placeholder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=get_observation(self.model, self.data).shape, dtype=np.float64) # Placeholder

        self.renderer = None  # Initialize renderer lazily

        self._time = 0  # Track the current time in the simulation.
        self.egg_start_pos = None # store the initial position of the egg
        self.current_dist = 9999
        self.moving_rate = moving_rate

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
            self.data.ctrl[:] = self.data.ctrl + (action * self.moving_rate)  # Apply action (replace with your control logic)
            mujoco.mj_step(self.model, self.data)
        except Exception as e:
            logging.error(f"Error in mj_step: {e}")
            # Handle the error appropriately (e.g., set terminated/truncated, return a default observation)
            observation = np.zeros(self.observation_space.shape)  # Or some other safe value
            return observation, 0, True, False, {"error": str(e)}

        self._time = self.data.time # update the time

        # Get the observation.
        observation = get_observation(self.model, self.data)

        # Calculate the reward.
        reward, self.current_dist = reward_function(self.model, self.data, prev_dist=self.current_dist, mode='grasp') #TODO prev_dist

        # Penalty for motions roughness
        roughtness_penalty = roughness_penalty(action)
        reward += roughtness_penalty

        # Check for the end of the session.
        terminated, additional_reward = check_session_end(self.model, self.data, time.time(), self.egg_start_pos) #TODO start time

        reward += additional_reward

        info = {
            "egg_at_start": egg_at_the_start(self.model, self.data),
            "egg_on_floor": egg_on_the_floor(self.model, self.data),
            "egg_at_holding": egg_at_the_holding(self.model, self.data),
            "egg_in_target": egg_in_target(self.model, self.data),
            "time": self._time,
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

        # Reset MuJoCo state.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) # Forward simulation to ensure initial state is correct

        self._time = 0  # Reset time
        self.egg_start_pos = self.data.xpos[get_body_id(self.model, "egg")][:2].copy()  # stores the initial xy position of the egg.
        if np.any(np.isnan(self.egg_start_pos)):
            self.egg_start_pos = np.array([0, 0])

        # Get the initial observation.
        observation = get_observation(self.model, self.data)

        info = {}  # Add any relevant info here

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


roboenv_1 = MujocoRobotArmEnv("egg_final.xml")
check_env(roboenv_1)