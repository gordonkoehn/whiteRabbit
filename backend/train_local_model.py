#!/usr/bin/env python3
"""
Script to train the RL model locally and save it to the pretrained directory.
This allows the model to be committed to the repository and used in deployment
without requiring training on startup.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO

def train_and_save_model():
    # Create pretrained directory if it doesn't exist
    os.makedirs("pretrained", exist_ok=True)
    
    print("Training new model locally...")
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    
    # Save the model to the pretrained directory
    model_path = "pretrained/ppo_cartpole"
    model.save(model_path)
    print(f"Model trained and saved to {model_path}.zip")
    
    # Quick validation that the model works
    state = env.reset()[0]
    action, _ = model.predict(state)
    print(f"Model validation - Prediction for initial state: {action}")

if __name__ == "__main__":
    train_and_save_model()