import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

class BipedalRewardShaper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = None
        self.step_count = 0
        self.contact_history = []
        self.prev_hull_angle = 0
        self.prev_hull_vel = 0
        
    def reset(self, **kwargs):
        self.prev_action = None
        self.step_count = 0
        self.contact_history = []
        self.prev_hull_angle = 0
        self.prev_hull_vel = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract state information
        hull_angle = obs[2]
        hull_vel_x = obs[3] 
        hull_vel_y = obs[4]
        joint_angles = obs[6:10]
        joint_speeds = obs[10:14]
        leg_contacts = obs[14:16]
        lidar = obs[16:]
        
        shaped_reward = self._compute_shaped_reward(
            obs, action, reward, hull_angle, hull_vel_x, hull_vel_y,
            joint_angles, joint_speeds, leg_contacts, lidar
        )
        
        self.prev_action = action
        self.prev_hull_angle = hull_angle
        self.prev_hull_vel = hull_vel_x
        self.step_count += 1
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _compute_shaped_reward(self, obs, action, base_reward, hull_angle, 
                              hull_vel_x, hull_vel_y, joint_angles, joint_speeds,
                              leg_contacts, lidar):
        
        # Start with base reward (forward progress)
        shaped_reward = base_reward
        
        # 1. Stability rewards
        stability_reward = self._stability_reward(hull_angle, hull_vel_y)
        shaped_reward += 0.3 * stability_reward
        
        # 2. Symmetric leg usage
        symmetry_reward = self._leg_symmetry_reward(joint_angles, joint_speeds)
        shaped_reward += 0.2 * symmetry_reward
        
        # 3. Proper gait pattern (alternating contacts)
        gait_reward = self._gait_pattern_reward(leg_contacts)
        shaped_reward += 0.25 * gait_reward
        
        # 4. Joint smoothness (penalize jerky movements)
        smoothness_reward = self._joint_smoothness_reward(action)
        shaped_reward += 0.15 * smoothness_reward
        
        # 5. Obstacle navigation
        obstacle_reward = self._obstacle_navigation_reward(leg_contacts, lidar, hull_vel_x)
        shaped_reward += 0.1 * obstacle_reward
        
        return shaped_reward
    
    def _stability_reward(self, hull_angle, hull_vel_y):
        # Reward staying upright and not bouncing too much
        upright_reward = np.exp(-5 * abs(hull_angle))  # Penalize tilting
        stable_y_vel = np.exp(-10 * abs(hull_vel_y))   # Penalize vertical bouncing
        return upright_reward + 0.5 * stable_y_vel
    
    def _leg_symmetry_reward(self, joint_angles, joint_speeds):
        # Compare left and right leg joint positions and speeds
        left_leg = joint_angles[:2]   # hip, knee
        right_leg = joint_angles[2:4] # hip, knee
        left_speeds = joint_speeds[:2]
        right_speeds = joint_speeds[2:4]
        
        # Penalize large differences in leg usage
        angle_diff = np.mean(np.abs(left_leg - right_leg))
        speed_diff = np.mean(np.abs(left_speeds - right_speeds))
        
        # Small differences are good, large differences are bad
        symmetry_score = np.exp(-2 * (angle_diff + speed_diff))
        return symmetry_score
    
    def _gait_pattern_reward(self, leg_contacts):
        # Track contact history for gait analysis
        self.contact_history.append(leg_contacts.copy())
        if len(self.contact_history) > 20:  # Keep last 20 steps
            self.contact_history.pop(0)
        
        if len(self.contact_history) < 10:
            return 0
        
        # Look for alternating contact pattern
        contacts = np.array(self.contact_history[-10:])  # Last 10 steps
        left_contacts = contacts[:, 0]
        right_contacts = contacts[:, 1]
        
        # Reward periods where only one foot is in contact
        single_contact_steps = np.sum((left_contacts > 0) != (right_contacts > 0))
        alternation_score = single_contact_steps / 10.0
        
        # Penalize double support or no contact
        both_contact = np.sum((left_contacts > 0) & (right_contacts > 0))
        no_contact = np.sum((left_contacts == 0) & (right_contacts == 0))
        penalty = (both_contact + 2 * no_contact) / 10.0
        
        return alternation_score - penalty
    
    def _joint_smoothness_reward(self, action):
        if self.prev_action is None:
            return 0
        
        # Penalize large changes in action (jerky movements)
        action_diff = np.mean(np.abs(action - self.prev_action))
        smoothness = np.exp(-5 * action_diff)
        return smoothness
    
    def _obstacle_navigation_reward(self, leg_contacts, lidar, hull_vel_x):
        # Detect obstacles using lidar
        front_lidar = lidar[5:15]  # Front-facing sensors
        obstacle_detected = np.any(front_lidar < 0.5)  # Obstacle within 0.5 units
        
        if not obstacle_detected:
            return 0
        
        # If obstacle detected, reward lifting feet
        foot_lift_reward = 0
        if leg_contacts[0] == 0:  # Left foot lifted
            foot_lift_reward += 0.5
        if leg_contacts[1] == 0:  # Right foot lifted
            foot_lift_reward += 0.5
            
        # Extra reward for maintaining forward velocity near obstacles
        vel_bonus = min(hull_vel_x / 2.0, 1.0) if hull_vel_x > 0 else 0
        
        return foot_lift_reward + vel_bonus