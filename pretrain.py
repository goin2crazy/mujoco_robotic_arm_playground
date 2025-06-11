import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from stable_baselines3.common.policies import BasePolicy

import os
import numpy as np
from typing import Dict, Optional, List

def read_recorded_trajectory(filepath):
    """
    Reads a single .npz trajectory file.
    """
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    
def load_multiple_trajectories(directory_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Loads all .npz trajectory files from a specified directory and combines them
    into a single dataset.

    Args:
        directory_path (str): The full path to the directory containing the .npz files.

    Returns:
        A single dictionary containing all demonstration data concatenated together,
        or None if the directory is not found or contains no valid files.
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return None

    # Find all files in the directory that end with .npz
    npz_files = [f for f in os.listdir(directory_path) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"Warning: No .npz files found in '{directory_path}'.")
        return None

    print(f"Found {len(npz_files)} trajectory files to load.")

    # Initialize lists to hold the data from each file
    all_observations: List[np.ndarray] = []
    all_controls: List[np.ndarray] = []
    all_rewards: List[np.ndarray] = []
    all_dones: List[np.ndarray] = []
    all_infos: List[np.ndarray] = []

    # Loop through each file and load its contents
    for file_name in npz_files:
        filepath = os.path.join(directory_path, file_name)
        try:
            # Here we assume read_recorded_trajectory just returns the loaded data
            data = np.load(filepath, allow_pickle=True)
            
            # Append the data arrays to our lists
            all_observations.append(data["observations"])
            all_controls.append(data["controls"])
            all_rewards.append(data["rewards"])
            all_dones.append(data["dones"])
            all_infos.append(data["infos"])
            print(f"  - Loaded {len(data['observations'])} steps from {file_name}")

        except Exception as e:
            print(f"Error loading or processing file {filepath}: {e}")

    if not all_observations:
        print("Error: Could not load valid data from any files.")
        return None

    # Concatenate all the lists of arrays into single large arrays
    combined_demonstrations = {
        "observations": np.concatenate(all_observations, axis=0),
        "controls": np.concatenate(all_controls, axis=0),
        "rewards": np.concatenate(all_rewards, axis=0),
        "dones": np.concatenate(all_dones, axis=0),
        "infos": np.concatenate(all_infos, axis=0),
    }
    
    total_steps = len(combined_demonstrations["observations"])
    print(f"\nSuccessfully combined all trajectories. Total steps available for training: {total_steps}")
    
    return combined_demonstrations

def behavioral_cloning_pretraining(
    student_policy: BasePolicy,
    env: gym.Env,
    demonstrations: dict,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
):
    """
    Pre-trains the policy using behavioral cloning (BC).

    This function is updated based on robust practices for handling different
    action spaces in Stable Baselines3.

    :param student_policy: The policy to be trained.
    :param env: The Gymnasium environment.
    :param demonstrations: A dictionary containing 'observations' and 'controls' (actions).
    :param epochs: The number of training epochs.
    :param learning_rate: The learning rate for the optimizer.
    :param batch_size: The batch size for training.
    """
    print("\n--- Starting Behavioral Cloning Pre-training ---")
    
    device = student_policy.device
    
    # --- 1. Create DataLoader from demonstrations ---
    obs = th.tensor(demonstrations['observations'], dtype=th.float32)
    # 'controls' from your file are the 'actions' for the policy
    actions = th.tensor(demonstrations['controls'], dtype=th.float32)

    dataset = TensorDataset(obs, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 2. Select the appropriate loss function based on the action space ---
    if isinstance(env.action_space, gym.spaces.Box):
        # Continuous action space
        criterion = nn.MSELoss()
        print("Using MSELoss for continuous action space.")
    else:
        # Discrete action space
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss for discrete action space.")
        
    # --- 3. Setup optimizer ---
    optimizer = optim.Adam(student_policy.parameters(), lr=learning_rate)
    
    # --- 4. Training loop ---
    student_policy.train() # Set the policy to training mode

    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()
            
            # --- Get action predictions from the policy ---
            if isinstance(env.action_space, gym.spaces.Box):
                # For PPO/A2C with continuous actions, the policy forward pass
                # returns (actions, values, log_probs). We only need the actions.
                action_prediction, _, _ = student_policy(batch_obs)
            else:
                # For discrete actions, we get the distribution and then the logits
                # to compare against the expert actions (which are class indices).
                dist = student_policy.get_distribution(batch_obs)
                action_prediction = dist.distribution.logits
                # Target for CrossEntropyLoss should be long type
                batch_actions = batch_actions.long()

            loss = criterion(action_prediction, batch_actions)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], BC Loss: {avg_loss:.6f}")

    print("--- Behavioral Cloning Finished ---\n")