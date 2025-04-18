import gymnasium as gym
from stable_baselines3 import PPO
import os
import warnings
from typing import Callable
import numpy as np
from .sky import sky  # Import the curried sky function

# Paths for saving and loading the model
MODEL_PATHS = [
    "ppo_three_body",           # Current path
    "model/ppo_three_body",     # Docker volume path
    "pretrained/ppo_three_body" # Path for pre-committed model
]

# Custom objects to handle version compatibility
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        """
        return progress_remaining * initial_value
    return func

def train_rl_model(env_name: str, model_path: str = "model/ppo_three_body"):
    """
    Train a new RL model in the specified environment.
    """
    print("Training new model...")
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)  # Increase timesteps for better training

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
    model.save(model_path)
    print(f"Model trained and saved at {model_path}")
    return model

def load_rl_model(env_name: str, model_paths: list = MODEL_PATHS):
    """
    This function is designed to work with a custom reinforcement learning (RL) environment inspired by the "Three-Body Problem" novel. 
    The environment models a society's survival and advancement under the influence of three suns. The emperor can issue commands 
    to "dry up" (pause societal progress to avoid extinction) or "hydrate" (resume societal progress). The goal is to maximize 
    societal advancement while avoiding extinction.
    Environment Details:
    - **State Space**: The state space represents the current environmental and societal conditions. This may include:
        - The positions or visibility of the three suns.
        - The current societal advancement level.
        - Whether the society is currently "dried up" or "hydrated".
    - **Action Space**: The emperor has two possible actions:
        1. "Dry up" - Pause societal progress to avoid extinction.
        2. "Hydrate" - Resume societal progress to advance the society.
    - **Reward Function**:
        - Positive rewards are given for advancing the society.
        - Negative rewards (or penalties) are applied if the society goes extinct due to all three suns being visible while hydrated.
        - A balance is required between advancing the society and avoiding extinction.
    Parameters:
    - env_name (str): The name of the environment to load or train the RL model for.
    Returns:
    - PPO: A pre-trained or newly trained Proximal Policy Optimization (PPO) model tailored for the specified environment.
    
    Load a pre-trained RL model if available, otherwise train a new one.
    """
    # Define custom objects to handle compatibility issues
    custom_objects = {
        "learning_rate": 0.0003,
        "lr_schedule": linear_schedule(0.0003),
        "clip_range": linear_schedule(0.2)
    }
    
    # Suppress specific warnings related to deserialization
    warnings.filterwarnings("ignore", message=".*Could not deserialize object.*")
    
    # Try to load from any of the possible locations
    for model_path in model_paths:
        if os.path.exists(f"{model_path}.zip"):
            try:
                print(f"Loading model from {model_path}")
                return PPO.load(model_path, custom_objects=custom_objects)
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
    
    print("Model not found in any location, training new model...")
    return train_rl_model(env_name)

def predict(model, state):
    """
    Use the trained model to predict the next action.
    """
    action, _ = model.predict(state)
    return int(action)

if __name__ == "__main__":
    # Define the custom environment name
    ENV_NAME = "ThreeBodyEnv-v0"  # Replace with your custom Gym environment
    
    # Load or train the RL model
    model = load_rl_model(ENV_NAME)
    
    # Example usage: simulate the environment
    env = gym.make(ENV_NAME)
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = predict(model, state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()  # Visualize the environment (if supported)
    
    print(f"Simulation finished with total reward: {total_reward}")