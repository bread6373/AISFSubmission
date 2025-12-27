import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

class LidarNudgeWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        lidar_data = obs[14:24]
        min_dist = np.min(lidar_data)
        velocity_x = obs[2]
   
        # If we are about to hit something fast, penalize.
        if min_dist < 0.15 and velocity_x > 0.4:
            reward -= 0.2
        
        # We only give the bonus if an obstacle is VERY close (min_dist < 0.1)
        # AND we reduce the bonus amount so it doesn't outweigh energy costs.
        if min_dist < 0.1:
            # Check if knees (indices 10 and 13) are lifted
            if obs[10] > 0.5 or obs[13] > 0.5:
                reward += 0.02
        
        if min_dist < 0.1 and (obs[10] > 0.5 or obs[13] > 0.5):
            if velocity_x > 0.6: 
                reward += 0.1
            
        return obs, reward, terminated, truncated, info

def make_env():
    env = gym.make("BipedalWalkerHardcore-v3")
    env = LidarNudgeWrapper(env)
    return env

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # 1. Parallelize (Crucial for a fresh start to see many obstacle types)
    num_cpu = 12 
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    
    # 2. Fresh Normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Model Configuration
    model = PPO(
        policy="MlpPolicy",
        env=env,
        device=device,
        learning_rate=2e-4,   # Standard LR for starting from scratch
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.03,        # PRO TIP: Higher entropy encourages exploration
        verbose=1,
        tensorboard_log="./ppo_hardcore_lidar/", # tensorboard --logdir ./ppo_hardcore_lidar/
        policy_kwargs=dict(net_arch=dict(pi=[400, 300], vf=[400, 300]))
    )

    # 4. Training (Prepare for a long run)
    # Without transfer learning, expect to need 10M - 20M steps
    print("Starting training from scratch...")
    model.learn(total_timesteps=5_000_000)

    # 5. Save
    model.save("ppo_hardcore_lidar")
    env.save("vec_normalize_lidar.pkl")

    print("Training Complete!")