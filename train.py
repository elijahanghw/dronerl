import os
from time import time
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from environment import DroneEnv

# --- Output directories ----------------------------------------------------
models_dir = "models"
logs_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# --- Environment -----------------------------------------------------------
env = DroneEnv(num_envs=64)
env = VecMonitor(env)   # records episode reward/length for TensorBoard

# --- Model -----------------------------------------------------------------
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=[dict(pi=[32, 32], vf=[32, 32])],
    log_std_init=0,
)
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    tensorboard_log=logs_dir,
    n_steps=1000,            # rollout buffer = n_steps * num_envs = 64000
    batch_size=6400,         # must divide 64000 (64000 / 6400 = 10 minibatches)
    n_epochs=10,
    gamma=0.999,
)


def train(model, log_name, n=int(10_000_000)):
    start_time = time()
    # Save every 10 policy rollouts.
    TIMESTEPS = model.n_steps * env.num_envs * 10
    save_dir = os.path.join(models_dir, log_name)
    os.makedirs(save_dir, exist_ok=True)

    # Reset timesteps for the 1st loop.
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, tb_log_name=log_name)
    model.save(os.path.join(save_dir, str(model.num_timesteps)))

    while model.num_timesteps < n:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name)
        model.save(os.path.join(save_dir, str(model.num_timesteps)))
        print(f"Time elapsed: {(time() - start_time):.2f}s")
        print(f"Model saved at {os.path.join(save_dir, str(model.num_timesteps))}")


if __name__ == "__main__":
    log_name = datetime.now().strftime("PPO_%Y%m%d_%H%M%S")
    train(model, log_name)
