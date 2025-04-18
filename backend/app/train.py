import gymnasium as gym
from stable_baselines3 import PPO
import os

def train_model(env_name: str, model_path: str = "models/ppo_model"):
    """
    Train a PPO model in the specified environment and save it.
    """
    print(f"Training PPO model on environment: {env_name}")
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)  # Adjust timesteps as needed

    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    ENV_NAME = "ThreeBodyEnv-v0"  # Our custom Gym environment
    MODEL_PATH = "models/ppo_cartpole"
    train_model(ENV_NAME, MODEL_PATH)