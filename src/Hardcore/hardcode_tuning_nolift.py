import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

class LidarNudgeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_min_dist = 1.0 

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        forward_lidar = obs[17:24]
        current_min_dist = np.min(forward_lidar)
        
        knee_lifting = (obs[6] > 0.5 or obs[11] > 0.5)
        is_moving_forward = obs[2] > 0.2
        
        # A. OBSTACLE MODE: (Wall/Stump detected)
        if current_min_dist < 0.25:
            if knee_lifting:
                reward += 0.08  # Encouraged to lift
            if obs[2] > 0.6:
                reward += 0.1   # Momentum bonus
                
        # B. STABILITY MODE: (Flat ground)
        else:      
            # Vertical Smoothing: Penalty for bouncing (obs[3] is Y-velocity)
            reward -= 0.02 * abs(obs[3])

        # --- 3. CORE PERFORMANCE ---
        # Speed reward (only if upright)
        if obs[2] > 0.8 and abs(obs[0]) < 0.2:
            reward += 0.3
            
        # Clearance Bonus
        if self.prev_min_dist < 0.2 and current_min_dist > 0.5:
            if is_moving_forward:
                reward += 1.5 

        # --- 4. POSTURE ---
        if abs(obs[0]) > 0.3: # Body Tilt
            reward -= 0.05
        if abs(obs[6]) > 0.8 and abs(obs[11]) > 0.8: # Deep Scoot
            reward -= 0.1

        self.prev_min_dist = current_min_dist
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
    model.learn(total_timesteps=10_000_000)

    # 5. Save
    model.save("ppo_hardcore_nolift")
    env.save("vec_normalize_nolift.pkl")

    print("Training Complete!")