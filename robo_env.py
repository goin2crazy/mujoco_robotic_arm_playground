
import random
from pprint import pprint
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import mujoco 
import logging 
from observation import * 
from inference import ReachingAgent
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
        self.target_body_name = reach_to_object_name
        self.load_mujoco_env()

                # smooth target position changing 
        # The current target position, which will be smoothly interpolated
        # Initialized to a default position
        self.current_target_position = [0.0, 0.0, 0.0] 
        
        # The next random target position that the current_target_position will move towards
        self.next_random_target_position = [0.0, 0.0, 0.0] 
        
        # The position from which the current transition started
        self.start_transition_position = [0.0, 0.0, 0.0]
        
        # Flag to indicate if a smoothing transition is currently active
        self.transition_active = False
        
        # Counter for steps within the current transition phase
        self.steps_in_transition = 0
        
        # The number of steps over which the smoothing transition should occur
        self.transition_duration = 100 # Example: transition over 100 steps
        
        #initialize the main mujoco env parameters 
        self.data = mujoco.MjData(self.model)
        self.steps_made_in_episode = 0       

        #initialize the main calculation fns 
        self.get_observation = lambda model, data, target_position: get_observation_reach_task(model=model, data=data, reach_to_body_pos=target_position)
        self.main_reward_fn = lambda model, data, target_position: improved_reward_function_reach_task(model, data, reach_to_body_pos=target_position)

        self.additional_reward_scale = additional_reward_scale


        self.action_space = spaces.Box(low=-1, high=1, shape=self.data.ctrl.shape, dtype=np.float32) # Placeholder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.get_observation(self.model, self.data, target_position=self.get_target_position()).shape, dtype=np.float64) # Placeholder

        self.renderer = None  # Initialize renderer lazily
        self.moving_rate = moving_rate
        self.min_move = min_move
        self.max_move=max_move



    def get_target_position(self):
            """
            Calculates and returns the target position. 
            
            Every 450 steps, a new random target is set, and the current target 
            smoothly interpolates towards this new random target over `transition_duration` steps.
            """

            # Check if it's time to generate a new random target position
            # This condition ensures a new target is set only when the previous transition (if any) is complete
            # or at the very beginning (steps_made_in_episode == 0)
            if self.steps_made_in_episode % 450 == 0 and not self.transition_active:
                # Generate new random values for x, y, and z components
                # x: from -1 to 1
                random_x = random.uniform(-1.0, 1.0)
                # y: from -0.2 to 0.3
                random_y = random.uniform(-0.2, 0.3)
                # z: from -1 to 1
                random_z = random.uniform(-1.0, 1.0)

                # Set the new random target
                self.next_random_target_position = [random_x, random_z, random_y]
                
                # Record the starting position for the current transition
                self.start_transition_position = list(self.current_target_position) # Make a copy
                
                # Activate the transition flag and reset the transition step counter
                self.transition_active = True
                self.steps_in_transition = 0
                
                
            # If a transition is active, perform linear interpolation
            if self.transition_active:
                # Increment the steps within the current transition
                self.steps_in_transition += 1
                
                # Calculate the interpolation factor (alpha)
                # This value goes from 0 to 1 over the transition duration
                alpha = min(1.0, self.steps_in_transition / self.transition_duration)

                # Perform linear interpolation for each component (x, y, z)
                # current_value = start_value * (1 - alpha) + end_value * alpha
                self.current_target_position[0] = self.start_transition_position[0] * (1 - alpha) + self.next_random_target_position[0] * alpha
                self.current_target_position[1] = self.start_transition_position[1] * (1 - alpha) + self.next_random_target_position[1] * alpha
                self.current_target_position[2] = self.start_transition_position[2] * (1 - alpha) + self.next_random_target_position[2] * alpha

                # If the transition is complete, deactivate the flag
                if self.steps_in_transition >= self.transition_duration:
                    self.transition_active = False
                    # Ensure the target snaps exactly to the final position to avoid floating point errors
                    self.current_target_position = list(self.next_random_target_position)

            # Increment the total steps made in the episode
            self.steps_made_in_episode += 1
            
            # Return the current (potentially interpolated) target position
            return self.current_target_position        

    def load_mujoco_env(self): 
        with open(self.model_path, 'r', encoding="utf-8") as f: 
            xlm_str = f.read() 

        # Load the new model from the XML string
        self.model = mujoco.MjModel.from_xml_string(xlm_str)
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
        observation= self.get_observation(self.model, self.data, target_position=self.get_target_position())

        reward = 0
        # Get reward and termination from unified function
        reward_fn_output = self.main_reward_fn(
            self.model, self.data,
            target_position=self.get_target_position(), 
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
        observation = self.get_observation(self.model, self.data, target_position=self.get_target_position() )

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

class MujocoRobotArmEnvTransportTask(gym.Env):
    """
    A Gymnasium environment for controlling a MuJoCo robot arm.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 model_path="your_egg_roboarm_env.xml", 
                 reaching_agent_model_path = "checkpoint.zip", 
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

        # lets initialize the reaching part 
        self.reaching = ReachingAgent(reaching_agent_model_path) 

        self.get_observation = lambda model, data: get_observation_grab_transport_task(model=model, data=data, target_body_name="egg")
        self.main_reward_fn = lambda model, data, action: improved_reward_function_grab_transport(model=model, data=data, action=action)

        #initialize the main calculation fns 
        self.action_space = spaces.Box(low=-1, high=1, shape=self.data.ctrl.shape, dtype=np.float32) # Placeholder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.get_observation(self.model, self.data).shape, dtype=np.float64) # Placeholder

        self.renderer = None  # Initialize renderer lazily
        self.moving_rate = moving_rate
        self.min_move = min_move
        self.max_move=max_move

    def load_mujoco_env(self): 
        with open(self.model_path, 'r', encoding="utf-8") as f: 
            xlm_str = f.read() 

        # Load the new model from the XML string
        self.model = mujoco.MjModel.from_xml_string(xlm_str)
        # A new model requires a new data instance
        self.data = mujoco.MjData(self.model)

    def env_step(self, predicted_action): 
        # x, y, z position for reacher agent to move 
        position_to_move = predicted_action[:-2]

        local_indicator = 'base'
        local_indicator_pos = get_body_pos(self.model, self.data, local_indicator)

        # So there we make that we gonna train the main agent
        # to predict x, y, z locally to roboarms base, which gonna add more flexibility to in future 
        position_to_move += local_indicator_pos
        action = self.reaching(self.model, self.data, position_to_move)
        # there is dealing with raw actions

        # Apply the capped update
        self.data.ctrl = [*action[:-2], 0, 0]

        #something with local position of roboarm 

        mujoco.mj_step(self.model, self.data)

    def step(self, action):
        """
        Simulates one step of the environment.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

    
        self.env_step(action)

        # Get the observation.
        observation= self.get_observation(self.model, self.data)

        reward = 0
        # Get reward and termination from unified function
        reach_reward = self.main_reward_fn(
            self.model, self.data,
            action, 
        )

        # Check for the end of the session.
        terminated, additional_reward = check_session_end(self.model, self.data, steps = self.steps_made_in_episode) #TODO start time
        
        if terminated:
            self.steps_made_in_episode = 0
        else: 
            self.steps_made_in_episode+=1 

        reward += reach_reward
        reward += additional_reward 

        info = {

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
        observation = self.get_observation(self.model, self.data )

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

