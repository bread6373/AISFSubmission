import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

def make_env():
    return gym.make("BipedalWalkerHardcore-v3")

if __name__ == "__main__":
    # 1. Parallelize (Crucial for a fresh start to see many obstacle types)
    num_cpu = 12 
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    
    # 2. Fresh Normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Model Configuration
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,   # Standard LR for starting from scratch
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.02,        # PRO TIP: Higher entropy encourages exploration
        verbose=1,
        tensorboard_log="./ppo_hardcore_scratch/",
        policy_kwargs=dict(net_arch=dict(pi=[400, 300], vf=[400, 300]))
    )

    # 4. Training (Prepare for a long run)
    # Without transfer learning, expect to need 10M - 20M steps
    print("Starting training from scratch...")
    model.learn(total_timesteps=5_000_000)

    # 5. Save
    model.save("ppo_hardcore")
    env.save("vec_normalize.pkl")

    print("Training Complete!")

    #pip install swig#pip install "gymnasium[box2d]"
