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

    
def behavioral_cloning_with_critic(
    student_policy: BasePolicy,
    demonstrations: dict,
    gamma: float = 0.99,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 64,
    value_coef: float = 0.5,     # weight of critic loss
):
    device = student_policy.device

    # Prepare data
    obs = th.tensor(demonstrations['observations'], dtype=th.float32).to(device)
    actions = th.tensor(demonstrations['controls'], dtype=th.float32).to(device)
    rewards = th.tensor(demonstrations['rewards'], dtype=th.float32).to(device)

    # 1) Compute discounted returns for each time‑step
    returns = []
    discounted_sum = 0
    for r in rewards.flip(0):
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns = th.tensor(returns, dtype=th.float32).unsqueeze(-1).to(device)

    dataset = TensorDataset(obs, actions, returns)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Losses & optimizer
    actor_criterion  = nn.MSELoss()  # or CrossEntropyLoss for discrete
    critic_criterion = nn.MSELoss()
    optimizer = optim.Adam(student_policy.parameters(), lr=lr)

    student_policy.train()
    for epoch in range(epochs):
        total_actor_loss = 0
        total_critic_loss = 0

        for b_obs, b_acts, b_returns in loader:
            optimizer.zero_grad()

            # Forward pass: actor + critic
            # For SB3 policies: forward returns (actions, values, log_probs)
            act_pred, value_pred, _ = student_policy(b_obs)

            # 2) Actor loss (behavioral cloning)
            actor_loss = actor_criterion(act_pred, b_acts)

            # 3) Critic loss: fit value_pred to returns
            critic_loss = critic_criterion(value_pred, b_returns)

            # 4) Combined loss
            loss = actor_loss + value_coef * critic_loss
            loss.backward()
            optimizer.step()

            total_actor_loss  += actor_loss.item()
            total_critic_loss += critic_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Actor Loss: {total_actor_loss/len(loader):.5f}, "
                  f"Critic Loss: {total_critic_loss/len(loader):.5f}")
    print("✅ BC + Critic pretraining complete!\n")
