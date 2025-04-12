#!/usr/bin/env python3
"""
Script to train the RL model locally and save it to the pretrained directory.
This allows the model to be committed to the repository and used in deployment
without requiring training on startup.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from typing import Callable

# Custom objects to handle version compatibility
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end)
        """
        return progress_remaining * initial_value
    return func

def train_and_save_model():
    # Create pretrained directory if it doesn't exist
    os.makedirs("pretrained", exist_ok=True)
    
    print("Training new model locally...")
    env = gym.make("CartPole-v1")
    
    # Using fixed hyperparameters for reproducibility
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        clip_range=0.2
    )
    model.learn(total_timesteps=10000)
    
    # Save the model to the pretrained directory
    model_path = "pretrained/ppo_cartpole"
    model.save(model_path)
    print(f"Model trained and saved to {model_path}.zip")
    
    # Quick validation that the model works
    state = env.reset()[0]
    action, _ = model.predict(state)
    print(f"Model validation - Prediction for initial state: {action}")
    print("\nThis model can now be committed to your repository and will be used in deployment.")
    print("The application will load this pre-trained model instead of training on startup.")

if __name__ == "__main__":
    train_and_save_model()