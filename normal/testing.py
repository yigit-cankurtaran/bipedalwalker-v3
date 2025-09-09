import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
import os

def test(model_path="models/best_model.zip", norm_path="models/vec_normalize.pkl", ep_count=5):
    base_env = gym.make("BipedalWalker-v3", render_mode="human")
    env = Monitor(base_env)
    env = DummyVecEnv([lambda: env])

    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.norm_reward = False
        env.training = False
    else:
        print("normalization path doesn't exist, run training")
    
    model = PPO.load(model_path)

    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=ep_count,
                                 render=True, return_episode_rewards=True)

    for i in range(ep_count):
        print(f"ep {i+1}'s reward is {rewards[i]}")

if __name__ == "__main__":
    test()
