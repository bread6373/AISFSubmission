import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

class LidarNudgeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # We track the previous min_dist to detect when we've cleared a hurdle
        self.prev_min_dist = 1.0 

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_min_dist = np.min(obs[14:24])
        velocity_x = obs[2]
        
        # If we were just near an obstacle (prev < 0.2) and now it's clear (current > 0.5)
        # AND we are moving forward, give a massive 'Success' reward.
        if self.prev_min_dist < 0.2 and current_min_dist > 0.5:
            if velocity_x > 0.2:
                reward += 2.0  # Significant bonus for successfully passing an obstacle
        
        # 2. SAFETY: Penalty for rushing at obstacles
        if current_min_dist < 0.15 and velocity_x > 0.4:
            reward -= 0.2
        
        # 3. ACTIVE NAVIGATION: Bonus for lifting knees while speed is high
        if current_min_dist < 0.1:
            if obs[6] > 0.5 or obs[11] > 0.5:
                reward += 0.02
                if velocity_x > 0.6: 
                    reward += 0.1
        
        # 4. FORM CONTROL: Anti-scoot and Uprightness
        if abs(obs[6]) > 0.8 and abs(obs[11]) > 0.8:
            reward -= 0.05  
        if abs(obs[0]) > 0.3: 
            reward -= 0.02

        # Update tracking variable for the next step
        self.prev_min_dist = current_min_dist
        
        # Encourage faster horizontal speed
        if abs(obs[2]) > 0.7:
            reward += 0.3

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
        tensorboard_log="./ppo_hardcore_scratch/",
        policy_kwargs=dict(net_arch=dict(pi=[400, 300], vf=[400, 300]))
    )

    # 4. Training (Prepare for a long run)
    # Without transfer learning, expect to need 10M - 20M steps
    print("Starting training from scratch...")
    model.learn(total_timesteps=5_000_000)

    # 5. Save
    model.save("ppo_hardcore_knee")
    env.save("vec_normalize_knee.pkl")

    print("Training Complete!")