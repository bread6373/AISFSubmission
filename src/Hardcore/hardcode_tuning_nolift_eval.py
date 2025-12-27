import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1. Create a single environment with 'human' render mode
env_id = "BipedalWalkerHardcore-v3"
env = gym.make(env_id, render_mode="human")

# 2. Wrap it and load your normalization stats
# IMPORTANT: Use the stats from your Hardcore training
env = DummyVecEnv([lambda: env])
env = VecNormalize.load("vec_normalize_nolift.pkl", env)

# No training updates during evaluation!
env.training = False
env.norm_reward = False 

# 3. Load the model
model = PPO.load("ppo_hardcore_nolift", env=env)

# 4. Run the loop
obs = env.reset()
for _ in range(2000):
    # deterministic=True is critical for evaluation
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()