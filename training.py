import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import os

def train():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    timesteps = 100_000
    
    # TODO: vecnormalize these
    train_env = make_vec_env("BipedalWalker-v3", 4, env_kwargs={"hardcore":True})
    train_env = VecNormalize(train_env)

    eval_env = DummyVecEnv([lambda: Monitor(gym.make("BipedalWalker-v3", hardcore=True))])
    eval_env = VecNormalize(eval_env, training=False)
    
    eval_callback = EvalCallback(
        eval_env,
        log_path="logs",
        best_model_save_path="models",
        n_eval_episodes=10,
        eval_freq=2500
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        # the rest are gonna be default for now
        # pretty horrible results 
    )
    
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True
    )

    train_env.save("models/vec_normalize.pkl")
    model.save("models/latest_model.zip")

    # in case we need the train func elswehere
    return train_env, model

if __name__ == "__main__":
    train()

