import numpy as np
from gymnasium import Wrapper


class SimpleBipedalRewardShaper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # stability bonus
        hull_angle = obs[0]  # Hull angle in radians
        stability_bonus = 0.1 * np.exp(-3 * abs(hull_angle))

        # speed reward to encourage forward movement
        horizontal_velocity = obs[2]  # normalized horizontal velocity
        speed_reward = np.clip(horizontal_velocity, 0, 1)

        # 3. Leg coordination - penalize both feet down (inefficient gait)
        right_foot_contact = obs[8]
        left_foot_contact = obs[13]
        both_feet_penalty = (
            -0.15 if (right_foot_contact > 0.5 and left_foot_contact > 0.5) else 0
        )

        # 4. Vertical stability - penalize bouncing
        # TODO: remove this and see how the training goes
        vertical_velocity = obs[3]  # Normalized vertical velocity
        vertical_penalty = -0.3 * abs(vertical_velocity)

        shaped_reward = (
            reward
            + stability_bonus
            + speed_reward
            + both_feet_penalty
            + vertical_penalty
        )

        info["reward_components"] = {
            "original": reward,
            "stability": stability_bonus,
            "speed": speed_reward,
            "coordination": both_feet_penalty,
            "vertical": vertical_penalty,
        }

        return obs, shaped_reward, terminated, truncated, info
