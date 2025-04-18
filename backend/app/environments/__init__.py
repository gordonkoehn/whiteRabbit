from gymnasium.envs.registration import register

register(
    id="ThreeBodyEnv-v0",
    entry_point="environments.three_body_env:ThreeBodyEnv",
)