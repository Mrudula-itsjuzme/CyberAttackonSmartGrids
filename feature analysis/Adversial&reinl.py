from stable_baselines3 import PPO
import numpy as np
import gym
from gym import spaces
import pandas as pd
import matplotlib.pyplot as plt

class CustomCyberEnv(gym.Env):
    def __init__(self, data):
        super(CustomCyberEnv, self).__init__()
        self.data = data
        self.current_step = 0
        
        # Define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(data.shape[1],), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(data.shape[1],), dtype=np.float32)

    def step(self, action):
        reward = -np.abs(self.data[self.current_step] - action).mean()  # Reward inversely proportional to error
        done = self.current_step >= len(self.data) - 1
        self.current_step += 1
        next_state = self.data[self.current_step % len(self.data)]
        return next_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Data: {self.data[self.current_step]}")

if __name__ == "__main__":
    print("Running Script 2: Adversarial and Reinforcement Learning")

    # Load your dataset
    dataset_path = "your_dataset.csv"  # Replace with your actual dataset path
    print("Loading dataset...")
    data = pd.read_csv(dataset_path)

    # Preprocess dataset (handling missing values and scaling)
    print("Preprocessing dataset...")
    data = data.select_dtypes(include=[np.number])  # Select numeric columns only
    data = data.fillna(data.mean())  # Replace missing values with column mean
    data = (data - data.min()) / (data.max() - data.min())  # Min-max normalization
    data = data.to_numpy()  # Convert to NumPy array

    # Create environment
    print("Setting up reinforcement learning environment...")
    env = CustomCyberEnv(data)

    # Initialize and train PPO model
    print("Training PPO model...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Evaluate model
    print("Evaluating trained PPO model...")
    state = env.reset()
    rewards = []
    actions = []

    for _ in range(len(data)):
        action, _ = model.predict(state)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break

    print("Training completed. Plotting results...")

    # Plot rewards over time
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Rewards")
    plt.title("Rewards Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

    # Plot actions for the first feature
    plt.figure(figsize=(10, 6))
    plt.plot([a[0] for a in actions], label="Actions (First Feature)")
    plt.title("Actions Over Time (First Feature)")
    plt.xlabel("Steps")
    plt.ylabel("Action Value")
    plt.legend()
    plt.show()

    print("Script 2 completed: Adversarial and Reinforcement Learning")
