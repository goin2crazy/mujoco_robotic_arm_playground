import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import ReplayBuffer

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

def behavioral_cloning_pretraining(policy, demonstrations, learning_rate=1e-3, epochs=50, batch_size=64):
    """
    Pre-trains the policy using behavioral cloning.
    """
    print("\n--- Starting Behavioral Cloning Pre-training ---")
    
    # Extract observations and actions from demonstrations
    obs = torch.tensor(demonstrations['observations'], dtype=torch.float32)
    # 'controls' from your file are the 'actions' for the policy
    actions = torch.tensor(demonstrations['controls'], dtype=torch.float32)

    dataset = TensorDataset(obs, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_actions in dataloader:
            # Get the actions predicted by the policy
            # Note: For PPO's actor-critic, we train the actor part
            pred_actions, _ = policy.predict(batch_obs, deterministic=True)
            
            loss = loss_fn(torch.tensor(pred_actions, dtype=torch.float32), batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    print("--- Behavioral Cloning Finished ---\n")

def main():
    # --- 1. Load Demonstrations ---
    recorded_data_dir = "recorded_data"
    if not os.path.exists(recorded_data_dir):
        print(f"Error: Directory '{recorded_data_dir}' not found.")
        return

    npz_files = [f for f in os.listdir(recorded_data_dir) if f.endswith('.npz')]
    if not npz_files:
        print(f"No .npz files found in '{recorded_data_dir}'.")
        return

    latest_file = max(npz_files, key=lambda f: os.path.getmtime(os.path.join(recorded_data_dir, f)))
    sample_filepath = os.path.join(recorded_data_dir, latest_file)
    
    print(f"Loading demonstrations from: {sample_filepath}")
    demonstrations = read_recorded_trajectory(sample_filepath)
    if demonstrations is None:
        return

    # --- 2. Environment Setup ---
    # IMPORTANT: Replace 'CartPole-v1' with your actual environment name
    env_name = 'CartPole-v1' 
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    # --- 3. PPO Model Initialization ---
    # The policy will be pre-trained, then fine-tuned with PPO
    model = PPO('MlpPolicy', env, verbose=1)

    # --- 4. Behavioral Cloning Pre-training ---
    behavioral_cloning_pretraining(model.policy, demonstrations)

    # --- 5. Add Demonstrations to PPO's Replay Buffer ---
    # This ensures the agent continues to learn from the expert data
    replay_buffer = model.rollout_buffer
    
    # Manually add demonstration data to the buffer
    # Note: `infos` might need special handling if it contains complex objects
    replay_buffer.add(
        demonstrations['observations'],
        demonstrations['controls'],
        demonstrations['rewards'],
        np.array([False] * len(demonstrations['dones'])), # 'episode_start'
        demonstrations['dones'],
        [{} for _ in range(len(demonstrations['dones']))] # 'infos'
    )
    
    print(f"\nAdded {len(demonstrations['observations'])} expert transitions to the PPO replay buffer.")

    # --- 6. PPO Fine-tuning ---
    print("\n--- Starting PPO Fine-tuning ---")
    # The total_timesteps here are for interaction with the environment
    # The model will use both its own experience and the pre-loaded expert data for updates
    model.learn(total_timesteps=25000)
    print("--- PPO Training Finished ---\n")

    # --- 7. Evaluate the Trained Agent ---
    print("\n--- Evaluating Trained Agent ---")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # --- 8. Save the Final Model ---
    model.save("ppo_trained_with_demonstrations")
    print("\n--- Model Saved ---")

if __name__ == "__main__":
    from stable_baselines3.common.evaluation import evaluate_policy
    main()