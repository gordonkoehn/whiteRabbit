from gymnasium.envs.registration import register

register(
    id="ThreeBodyEnv-v0",
    entry_point="app.environments.three_body_env:ThreeBodyEnv",  # Correct module path
)