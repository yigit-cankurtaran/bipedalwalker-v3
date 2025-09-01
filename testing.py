import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

def test():
    model_path=""
    ep_count = 5
    try:
        model_path = "models/best_model.zip"
    except FileNotFoundError:
        print("model file not found, run training")

    env = gym.make("BipedalWalker-v3", render_mode="human", hardcore=True)
    env = Monitor(env)
    model = PPO.load(model_path)

    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=ep_count,
                                 render=True, return_episode_rewards=True)

    for i in range(ep_count):
        print(f"ep {i+1}'s reward is {rewards[i]}")

if __name__ == "__main__":
    test()
