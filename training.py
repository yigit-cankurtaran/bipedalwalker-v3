import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import os

def linear_decay(init_val):
    def func(progress_remaining):
        return init_val * progress_remaining

    return func

def train():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    timesteps = 2_000_000
    
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

    # syncing normalization stats with eval and train
    # not really a problem for simpler and more forgiving envs
    # but for hardcore we need to fix this
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    model = PPO(
        "MlpPolicy",
        train_env,
        gae_lambda=0.9, # lower lambda can be better for locomotion
        gamma=0.999, # need future good rewards
        ent_coef=0.05, # need a higher exploration
        learning_rate=linear_decay(3e-3),
        max_grad_norm=1.0
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

