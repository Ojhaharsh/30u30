# Data Directory

This directory stores training logs and saved model checkpoints.

## Contents

| File | What It Is |
|------|-----------|
| `training_log_*.csv` | Per-iteration stats: reward, clip_fraction, entropy, value_loss |
| `checkpoint_*.pt` | Saved model weights (actor-critic network state dict) |

## Log Format

Each CSV has columns:
```
iteration, mean_reward, clip_fraction, entropy, value_loss, policy_loss
```

## Notes

- Logs are written by `train_minimal.py` when `--save_log` is passed.
- Checkpoints are saved every 50 iterations by default.
- The `visualization.py` script can read these logs to plot learning curves.
