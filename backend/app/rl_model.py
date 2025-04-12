import gymnasium as gym
from stable_baselines3 import PPO
import os

def train_rl_model():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_cartpole")
    return model

def load_rl_model():
    model_path = "ppo_cartpole"
    if os.path.exists(f"{model_path}.zip"):
        try:
            return PPO.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
    print("Model not found, training new model...")
    return train_rl_model()

def predict(model, state):
    action, _ = model.predict(state)
    return action