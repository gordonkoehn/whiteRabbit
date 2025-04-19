import os
import sys
from app.environments import register_all_envs

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rl_model import train_rl_model, predict

def test_predict():

    register_all_envs()
    
    model = train_rl_model(env_name="environments/ThreeBodyEnv-v0")
    state = [0, 0, 0, 0]
    action = predict(model, state)
    assert isinstance(action, int)