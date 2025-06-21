from pprint import pprint
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

    def __init__(self, 
                 model_path="your_model.xml", 
                 moving_rate=5e-3, 
                 additional_reward_scale=1, 
                 ):  # Add model_path
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

        self.steps_made_in_episode = 0         

        self.additional_reward_scale = additional_reward_scale
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
            # OLD ACTIONS FORMULA - DOESNT WORKS WELL!!! ESPECIALLY FOR DEMOS 
            # self.data.ctrl[:] = self.data.ctrl + (action * (self.moving_rate ** 0.5))  

            # So if we imagine that action = data.ctrl + some movement 
            # The with math simple laws we ger that 
            # some little movement = action - data.ctrl 
            # So the smooth this little movement we apply the moving_rate 
            # And it all goes data.ctrl = action = data.ctrl + little_movement = data.ctrl + (action- data.ctrl) * moving_rate 
            # But it can be kinda bed
            # So its better to change to the 
            # data.ctrl = action = data.ctrl + little_movement = data.ctrl + min((action- data.ctrl), moving_rate) 
            # Compute per-joint difference between desired action and current control

            # But since there is also might be negative little moovement, its better to use clip, which work on both sides
            diff = action - self.data.ctrl  

            # Maximum change per step
            alpha = self.moving_rate

            # Clamp each component of diff to [-alpha, +alpha]
            capped = np.clip(diff, -alpha, alpha)  

            # Apply the capped update
            self.data.ctrl[:] = self.data.ctrl + capped

            mujoco.mj_step(self.model, self.data)
        except Exception as e:
            logging.error(f"Error in mj_step: {e}")
            # Handle the error appropriately (e.g., set terminated/truncated, return a default observation)
            observation = np.zeros(self.observation_space.shape)  # Or some other safe value
            return observation, 0, True, False, {"error": str(e)}

        self._time = self.data.time # update the time

        # Get the observation.
        observation = get_observation(self.model, self.data)

        reward = 0
        # Get reward and termination from unified function
        total_reward, terminated, info = improved_reward_function(
            self.model, self.data,
            old_target_dist=self.last_target_dist, 
            egg_initial_z=self.egg_initial_z, 
        )

        self.last_target_dist = info['target_dist']
        self.last_action = action

        # Check for the end of the session.
        terminated, additional_reward = check_session_end(self.model, self.data, self.steps_made_in_episode, self.egg_start_pos) #TODO start time
        
        if terminated:
            self.steps_made_in_episode = 0
        else: 
            self.steps_made_in_episode+=1 

        reward += total_reward
        reward += additional_reward * self.additional_reward_scale

        info = {
            "egg_at_start": egg_at_the_start(self.model, self.data),
            "egg_on_floor": egg_on_the_floor(self.model, self.data),
            "egg_at_holding": egg_at_the_holding(self.model, self.data),
            "egg_to_target": info['target_dist'],
            "egg_to_grip": info['grip_dist'],
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
        self.egg_start_pos = get_body_pos(self.model, self.data, body_name="egg")

        # Get the initial observation.
        observation = get_observation(self.model, self.data)
        self.last_action = self.data.ctrl
        self.last_dist=0
        self.egg_initial_z = get_body_pos(model=self.model, data = self.data, body_name="egg")[2]

        self.last_target_dist = np.linalg.norm(self.egg_start_pos - get_body_pos(model = self.model, 
                                                                                data= self.data,
                                                                                body_name= 'egg_base_target'))
        

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


class RoboArmJointsController_Vanilla(): 
    # So main purpose of this class in to return the controls for mujoco data 
    # Lets make the "convert_control" fn main here 
    # and for fingers and arm parts make the different processing methods
    
    def __init__(self, 
                 arm_parts=None, 
                 fingers_parts=None, 
                 moving_rate= 0.03, 
                 min_ctrl=-1, 
                 max_ctrl=1, 
                 ): 
        """
        Arguments:

            arm_parts: 
                - its dictionary of pair joint_name-its_id_in_controls, can be be infinite count, no matter, 
                - Default: None, if value doesnt changes it turns into
                    arm_parts = {
                    'p1': 0, 
                    ...}

            fingers_parts: 
                - Can be only two 
                - format should be the same as arm_parts 
         
        """
        # So first lets just define which propability applies to which 
        # 0. p1: A <-> D 
        # 1. p2_arm: W <-> S
        # 2. p1_arm2: E <-> F
        # 3 and 4. fingers: C <-> V 
        # To code: 
        if arm_parts == None: 
            arm_parts = {
                'p1': 0, 
                'p2_arm': 1, 
                'p1_arm2': 2    
                         }
            
        if fingers_parts == None: 
            fingers_parts = {
                'arm_finger_right': 3, 
                'arm_finger_left': 4, 
                         }

        self.moving_rate = moving_rate
        self.arm_p = arm_parts
        self.fingers_p = fingers_parts 

        self.min_ctrl = min_ctrl
        self.max_ctrl = max_ctrl 

        self.setup_propabilities() 

    def control_the_fingers(self, dummy_action, direction:int): 
        # Direction parameter here is like 
        # If 1 - fingers close 
        # If -1 - fingers open 
        # I just hope it add a little bit of understanding 
        first_finger_id, second_finger_id = list(self.fingers_p.values())

        dummy_action[first_finger_id] = np.clip(dummy_action[first_finger_id]
                                                - (self.moving_rate), self.min_ctrl, self.max_ctrl)*direction

        dummy_action[second_finger_id] = np.clip(dummy_action[second_finger_id]
                                                + (self.moving_rate), self.min_ctrl, self.max_ctrl)*direction
        return dummy_action
    
    def control_the_arm_parts(self, dummy_action, direction:int, joint_name:str): 
        joint_id = self.arm_p[joint_name]
        print("CONTROL JOINT OF ARM", joint_id, joint_name)

        dummy_action[joint_id] = np.clip(dummy_action[joint_id]
                                                + (self.moving_rate), self.min_ctrl, self.max_ctrl)*direction
        return dummy_action

    def setup_propabilities(self): 

        arm_parts_move = []
        for k in list(self.arm_p.keys()): 
            item = (k, (lambda dummy_action, joint_name_k: self.control_the_arm_parts(dummy_action, direction=1, joint_name=joint_name_k )))
            arm_parts_move.append(item)
        
        arm_parts_move_opposite = []
        for k in list(self.arm_p.keys()): 
            item = (k, (lambda dummy_action, joint_name_k: self.control_the_arm_parts(dummy_action, direction=-1, joint_name=joint_name_k )))
            arm_parts_move_opposite.append(item)


        self.probs = [
            # Initialize the arm parts 
            # In the finnal all we need its just put the data.ctrl instead of dummy_action, and no ifs needed
            *arm_parts_move, 

            # Ther is action into opposite direction for arm parts 
            *arm_parts_move_opposite, 

            # Now lets deal with robots fingers 
            ("fingers", (lambda dummy_action, joint_name_non: self.control_the_fingers(dummy_action=dummy_action, direction=1))), 

            # The opposite direction to close fingers
            ("fingers", (lambda dummy_action, joint_name_non: self.control_the_fingers(dummy_action=dummy_action, direction=-1))), 

            ("stop", (lambda dummy_action, joint_name_non: dummy_action)), 
        ]

        pprint(self.probs)    

    def convert_control(self, propabilities, model_data): 
        choosen_move = np.argmax(propabilities)

        moving_fn_item = self.probs[choosen_move]
        print(moving_fn_item)
        key, fn = moving_fn_item
        return fn(model_data, key)
    
# --- Test Script ---
if __name__ == "__main__":
    print("--- Starting RoboArmJointsController_Vanilla Test ---")

    # Initialize the controller
    controller = RoboArmJointsController_Vanilla(
        arm_parts={'shoulder': 0, 'elbow': 1, 'wrist': 2},
        fingers_parts={'gripper_left': 3, 'gripper_right': 4},
        moving_rate=0.1, # Increased moving rate for clearer test results
        min_ctrl=-1.0,
        max_ctrl=1.0
    )
    # Initial model data (control array) - all zeros for easy observation
    # The size of this array should match the highest joint ID + 1
    initial_model_data = np.array([0.0] * (max(list(controller.arm_p.values()) + list(controller.fingers_p.values())) + 1))
    print(f"\nInitial model_data: {initial_model_data}")

    # Test each possible action by setting a high probability for its index
    num_actions = len(controller.probs)
    print(f"Total number of defined actions: {num_actions}")

    action_descriptions = [
        "Shoulder positive (+)", "Elbow positive (+)", "Wrist positive (+)",
        "Shoulder negative (-)", "Elbow negative (-)", "Wrist negative (-)",
        "Fingers close", "Fingers open", "No change"
    ]

    for i in range(num_actions):
        # Create probabilities array to select the i-th action
        probabilities = np.zeros(num_actions)
        probabilities[i] = 1.0 # Set a high probability for the current action

        # Ensure we're testing with a fresh copy of the initial data
        current_model_data = initial_model_data.copy()

        # Get the new control data after applying the chosen action
        new_model_data = controller.convert_control(probabilities, current_model_data)

        # Print results
        print(f"\n--- Testing Action {i}: {action_descriptions[i] if i < len(action_descriptions) else 'Unknown Action'} ---")
        print(f"  Input model_data (before action): {current_model_data}")
        print(f"  Action chosen (argmax of probabilities): {np.argmax(probabilities)}")
        print(f"  Output model_data (after action): {new_model_data}")

    print("\n--- RoboArmJointsController_Vanilla Test Complete ---")
    print("All assertions passed (if any were enabled). Review the output above.")


class MujocoRobotArmEnv_Vanilla(gym.Env):
    """
    A Gymnasium environment for controlling a MuJoCo robot arm.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 model_path="your_model.xml", 
                 moving_rate=5e-3,
                 reward_fn = None, 
                 reward_fn_scale =1, 
                 additional_reward_scale=1, 
                 ):  # Add model_path
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
        self.reward_fn = reward_function_grasp if reward_fn ==None else reward_fn

        self.steps_made_in_episode = 0         
        self.reward_fn_scale = reward_fn_scale 
        self.additional_reward_scale = additional_reward_scale
        
        self.controller = RoboArmJointsController_Vanilla(None, None, moving_rate=moving_rate)

        self.action_space = spaces.Box(low=0, high=1, shape=(len(self.controller.probs), ), dtype=np.float32) # Placeholder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=get_observation(self.model, self.data).shape, dtype=np.float64) # Placeholder

        self.renderer = None  # Initialize renderer lazily

        self._time = 0  # Track the current time in the simulation.
        self.egg_start_pos = None # store the initial position of the egg
        self.moving_rate = moving_rate

    def step(self, action_propabilities):
        """
        Simulates one step of the environment.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Apply the action (e.g., set joint torques).
        assert action_propabilities.shape == self.action_space.shape # Important: Check shape

        try:
            # Apply the capped update
            action = self.controller.convert_control(action_propabilities, self.data.ctrl)
            self.data.ctrl[:] = action

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
        reward, self.last_dist = self.reward_fn(self.model, self.data, self.last_dist)
        reward = reward * self.reward_fn_scale

        self.last_action = action

        # Check for the end of the session.
        terminated, additional_reward = check_session_end(self.model, self.data, self.steps_made_in_episode, self.egg_start_pos) #TODO start time
        
        if terminated:
            self.steps_made_in_episode = 0
        else: 
            self.steps_made_in_episode+=1 

        reward += additional_reward * self.additional_reward_scale

        info = {
            # "egg_at_start": egg_at_the_start(self.model, self.data),
            # "egg_on_floor": egg_on_the_floor(self.model, self.data),
            # "egg_at_holding": egg_at_the_holding(self.model, self.data),
            # "egg_in_target": egg_in_target(self.model, self.data),
            # "time": self._time,
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
        self.last_action = self.data.ctrl
        self.last_dist=0

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
        # if self.renderer:
        #     self.renderer.close()
        self.renderer = None
        # No need to close self.model or self.data, they don't have close() methods in mujoco

    def __del__(self):
        # Ensure resources are cleaned up.  Important for preventing memory leaks.
        self.close()
