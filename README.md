# dronerl

Reinforcement learning for quadrotor control. A drone learns to fly to a target
position using PPO, trained in a fast vectorized simulator.

## Overview

The environment ([`environment.py`](environment.py)) is a native Stable-Baselines3
`VecEnv` that simulates many drones at once in batched NumPy. Dynamics use a
**collective-thrust + body-rate (CTBR)** command model: the policy commands a
thrust and roll/pitch/yaw rates, which pass through a first-order actuator lag
(ARX) before a rigid-body integration step in NED coordinates (quaternion
attitude).

- **Action** (4): `[thrust, p, q, r]`, each in `[-1, 1]`.
- **Observation** (13): `[pos_err(3), roll/pitch/yaw(3), vel(3), body_rates(3), thrust(1)]`.
- **Reward**: negative tracking error (position, speed, rate), with a fixed
  penalty for hitting the ground or leaving the arena.

## Files

| File | Purpose |
|------|---------|
| [`environment.py`](environment.py) | Vectorized CTBR drone `VecEnv`. |
| [`train.py`](train.py) | PPO training loop; saves checkpoints and TensorBoard logs. |
| [`analyze.ipynb`](analyze.ipynb) | Plots training history and a test rollout (position, velocity, actions, top-down trajectory). |

## Setup

```bash
pip install stable-baselines3 torch gymnasium numpy tensorboard matplotlib jupyter
```

## Usage

Train:

```bash
python train.py
```

Checkpoints are written to `models/<run>/` and TensorBoard logs to `logs/<run>/`.
Monitor training:

```bash
tensorboard --logdir logs
```

Analyze a trained model — open [`analyze.ipynb`](analyze.ipynb) and run all cells.
It loads the most recent checkpoint and plots the training curve and one
deterministic rollout.
