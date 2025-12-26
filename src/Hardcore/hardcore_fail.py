import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create a vectorized environment (accelerates training)
env_id = "BipedalWalkerHardcore-v3"
env = make_vec_env(env_id, n_envs=4)

# 2. Define the PPO model
# These hyperparameters are tuned for continuous robotic control
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    device="auto" # Uses GPU if available
)

# 3. Train the agent
print("Starting training...")
model.learn(total_timesteps=1_000_000)

# 4. Save the trained model
model.save("ppo_bipedal_walker_hardcore")

# 5. Evaluate the agent
eval_env = gym.make(env_id, render_mode="human")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

eval_env.close()