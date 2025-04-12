import gymnasium as gym
from stable_baselines3 import PPO
import os

MODEL_PATHS = [
    "ppo_cartpole",           # Current path (for backward compatibility)
    "model/ppo_cartpole",     # Docker volume path
    "pretrained/ppo_cartpole" # Path for pre-committed model
]

def train_rl_model():
    print("Training new model...")
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname("model/ppo_cartpole") if os.path.dirname("model/ppo_cartpole") else ".", exist_ok=True)
    model.save("model/ppo_cartpole")
    return model

def load_rl_model():
    # Try to load from any of the possible locations
    for model_path in MODEL_PATHS:
        if os.path.exists(f"{model_path}.zip"):
            try:
                print(f"Loading model from {model_path}")
                return PPO.load(model_path)
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
    
    print("Model not found in any location, training new model...")
    return train_rl_model()

def predict(model, state):
    action, _ = model.predict(state)
    return int(action)