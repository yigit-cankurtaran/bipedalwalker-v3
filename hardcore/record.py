"""Record deterministic rollouts from the saved hardcore PPO policy."""

import argparse
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from reward_shaper import SimpleBipedalRewardShaper


ROOT = Path(__file__).resolve().parents[1]


def record(episodes: int, seed: int, output: Path, model_path: Path, norm_path: Path):
    """Run the policy in RGB mode and pipe frames directly to FFmpeg."""
    # VecNormalize stores the observation statistics used during training.
    stats_env = DummyVecEnv(
        [lambda: SimpleBipedalRewardShaper(gym.make("BipedalWalker-v3", hardcore=True))]
    )
    normalizer = VecNormalize.load(str(norm_path), stats_env)
    normalizer.training = False
    normalizer.norm_reward = False

    model = PPO.load(str(model_path))
    env = SimpleBipedalRewardShaper(
        gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    frame = env.render()
    height, width, _ = frame.shape
    ffmpeg = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "50",
        "-i",
        "-",
        "-vf",
        "fps=30,scale=1620:1080:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x111827,setsar=1",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    encoder = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE)
    episode_rewards = []
    try:
        for episode in range(episodes):
            if episode:
                obs, _ = env.reset(seed=seed + episode)
                frame = env.render()
            total_reward = 0.0
            steps = 0
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())

            while True:
                normalized_obs = np.clip(
                    (obs - normalizer.obs_rms.mean)
                    / np.sqrt(normalizer.obs_rms.var + normalizer.epsilon),
                    -normalizer.clip_obs,
                    normalizer.clip_obs,
                ).astype(np.float32)
                action, _ = model.predict(normalized_obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                encoder.stdin.write(np.ascontiguousarray(env.render()).tobytes())
                if terminated or truncated:
                    episode_rewards.append(total_reward)
                    print(
                        f"Episode {episode + 1}/{episodes}: "
                        f"reward={total_reward:.1f}, steps={steps}"
                    )
                    break
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        return_code = encoder.wait()
        env.close()
        normalizer.close()

    if return_code:
        raise subprocess.CalledProcessError(return_code, ffmpeg)
    print(f"Saved {episodes} episodes to {output}")
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/hardcore_rollouts.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "hardcore/models/best_model.zip")
    parser.add_argument("--normalization", type=Path, default=ROOT / "hardcore/models/vec_normalize.pkl")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    record(args.episodes, args.seed, args.output, args.model, args.normalization)


if __name__ == "__main__":
    main()
