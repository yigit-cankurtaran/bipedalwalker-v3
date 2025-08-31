import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
import os

log_path = os.makedirs("logs", exist_ok=True)
model_path = os.makedirs("models", exist_ok=True)

# TODO: vecnormalize these
train_env = make_vec_env("BipedalWalker-v3", 4, env_kwargs={"hardcore":True})
eval_env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3", hardcore=True)])

eval_callback = EvalCallback(
    eval_env,
    log_path,
    best_model_save_path=model_path
)

model = PPO(
    "MlpPolicy",
    train_env,
    # the rest are gonna be default for now
)


