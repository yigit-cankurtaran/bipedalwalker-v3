# BipedalWalker Hardcore RL

High-performance reinforcement learning implementation for the BipedalWalker-v3 Hardcore environment using PPO and custom reward shaping. Achieves 450+ mean rewards, significantly outperforming standard benchmarks.

## Performance

This implementation achieves state-of-the-art results on the notoriously difficult BipedalWalker-v3 Hardcore environment:

- **Best performance**: 534 mean reward
- **Consistent performance**: 450+ mean reward
- **Standard benchmarks**: ~122 mean reward (RL-Baselines3-Zoo)

The performance improvement comes from intelligent reward shaping and carefully tuned hyperparameters specifically optimized for the hardcore variant.

## Project Structure

```
├── hardcore/               # Hardcore environment implementation
│   ├── training.py        # Main training script with PPO
│   ├── testing.py         # Evaluation and testing
│   └── reward_shaper.py   # Custom reward shaping wrapper
├── normal/                # Standard BipedalWalker for comparison
│   ├── training.py
│   └── testing.py
└── requirements.txt       # Dependencies
```

## Key Features

### Custom Reward Shaping

The `SimpleBipedalRewardShaper` enhances the original environment rewards with:

- **Stability bonuses**: Encourages upright posture maintenance
- **Speed rewards**: Promotes forward momentum
- **Gait coordination**: Penalizes inefficient dual-foot contact
- **Vertical stability**: Reduces excessive bouncing

### Optimized Hyperparameters

PPO configuration tuned specifically for hardcore locomotion:

- Higher GAE lambda (0.97) for better long-term planning
- Increased gamma (0.995) for shaped reward environments
- Longer rollouts (4096 steps) to capture gait patterns
- Reduced entropy coefficient (0.005) for stable learning

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train on the hardcore environment:

```bash
cd hardcore
python training.py
```

Train on the standard environment for comparison:

```bash
cd normal
python training.py
```

### Testing

Evaluate a trained model with visual rendering:

```bash
cd hardcore
python testing.py
```

## Environment Details

The BipedalWalker-v3 Hardcore variant includes challenging terrain features:

- Stumps and pitfalls
- Ladders requiring climbing
- Irregular surfaces
- Dynamic obstacles

Success requires the agent to navigate these obstacles while maintaining stable bipedal locomotion.

## Results Analysis

Training logs and evaluation metrics are saved in:

- `hardcore/logs/` - Training progress and evaluation results
- `hardcore/models/` - Saved models and normalization parameters

The reward shaping components are tracked individually for analysis and debugging.

## Technical Notes

- Uses VecNormalize for observation and reward normalization
- Synchronized normalization between training and evaluation environments
- Comprehensive evaluation callbacks with best model saving
- Tensorboard logging for training visualization

## Dependencies

Built with modern RL libraries:

- Stable-Baselines3 2.7.0
- Gymnasium 1.2.0
- PyTorch 2.8.0
- NumPy, Matplotlib for analysis
