"""A batched (vectorized) drone environment for Stable-Baselines3.

This is a native SB3 ``VecEnv``: it simulates ``num_envs`` drones at once in
batched NumPy and exposes the ``VecEnv`` API (``step_async``/``step_wait``,
4-tuple returns, automatic sub-env reset). Do NOT wrap it in ``DummyVecEnv`` /
``make_vec_env`` -- it already is a vectorized env, so pass it straight to the
algorithm (e.g. ``PPO("MlpPolicy", DroneEnv())``).

Dynamics: collective-thrust + body-rate (CTBR) command, pushed through a
per-channel first-order lag with optional transport delay (ARX model), then
integrated as a rigid body in NED coordinates with a quaternion attitude.

State layout (per env), stored in ``self.states`` with shape (num_envs, 14):

    0:3    pos      x, y, z            (NED, z positive down)
    3:7    quat     qw, qx, qy, qz     (body <- world attitude)
    7:10   vel      vx, vy, vz
    10:13  rates    p, q, r            (achieved body rates)
    13     c        achieved collective thrust (in g; c=1 -> hover)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Batched quaternion helpers. All operate on leading batch dim N.
# Quaternion convention: [w, x, y, z].
# ---------------------------------------------------------------------------
def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N, 4) unit quaternions -> (N, 3, 3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two batches of quaternions, (N, 4)."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def _quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    """(N, 3) rotation vectors (axis * angle) -> (N, 4) quaternions."""
    angle = np.linalg.norm(v, axis=1, keepdims=True)          # (N, 1)
    small = angle < 1e-8
    # For small angles, sin(a/2)/a -> 1/2; avoid divide-by-zero.
    sin_half_over_angle = np.where(small, 0.5, np.sin(0.5 * angle) / np.where(small, 1.0, angle))
    xyz = v * sin_half_over_angle
    w = np.cos(0.5 * angle)
    return np.concatenate([w, xyz], axis=1)


def _quat_to_euler(q: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions -> (N, 3) roll/pitch/yaw (ZYX intrinsic), radians."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))      # roll
    theta = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))         # pitch
    psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))      # yaw
    return np.stack([phi, theta, psi], axis=1)


def _euler_to_quat(phi: np.ndarray, theta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Roll/pitch/yaw (N,) -> (N, 4) quaternions (ZYX intrinsic)."""
    cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
    cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
    cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=1)


class DroneEnv(VecEnv):
    """Vectorized CTBR drone environment (native SB3 VecEnv)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, num_envs: int = 64, render_mode: str | None = None):
        # --- Action space: [thrust, p, q, r] command, all in [-1, 1] -------
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # --- Observation space (per env): pos_err(3) + euler(3) + vel(3)
        #     + rates(3) + thrust(1) = 13 ------------------------------------
        # Attitude is observed as roll/pitch/yaw, though the state integrates a
        # quaternion internally.
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # VecEnv.observation_space / action_space are the *single-env* spaces;
        # the base class stacks to (num_envs, ...) internally.
        super().__init__(num_envs, observation_space, action_space)
        self.render_mode = render_mode

        self.target = np.array([0.0, 0.0, -1.5], dtype=np.float32)

        # --- Sim / command config ----------------------------------------
        self.dt = 0.01
        self.max_episode_steps = 1000
        self.rate_max_pq = 10.0     # rad/s, roll/pitch rate command scale
        self.rate_max_r = 5.0       # rad/s, yaw rate command scale

        # ARX first-order lag per channel, internal order [p, q, r, c].
        # a = exp(-dt/tau), b = 1 - a  (unit DC gain). Tune the time constants.
        tau = np.array([0.03, 0.03, 0.05, 0.08], dtype=np.float32)  # [p,q,r,c]
        self._arx_a = np.exp(-self.dt / tau)
        self._arx_b = 1.0 - self._arx_a

        # Transport delay (integer steps) per channel; 0 = no delay.
        self._delays = np.array([0, 0, 0, 0], dtype=int)
        self._dmax = int(self._delays.max()) + 1
        # Command ring buffer, [p,q,r,c]; row -1 is most recent.
        self._cmd_buf = np.zeros((num_envs, self._dmax, 4), dtype=np.float32)

        # --- Episode bookkeeping -----------------------------------------
        self.states = np.zeros((num_envs, 14), dtype=np.float32)
        self.step_counts = np.zeros(num_envs, dtype=int)
        self.actions = np.zeros((num_envs, 4), dtype=np.float32)

        self.reset()

    # ----------------------------------------------------------------------
    # VecEnv core API
    # ----------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._reset_idx(np.ones(self.num_envs, dtype=bool))
        return self._get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        self.states = self._dynamics(self.states, self.actions)
        self.step_counts += 1

        rewards, terminated = self._reward_and_termination(self.states)
        truncated = self.step_counts >= self.max_episode_steps
        dones = terminated | truncated

        # Build infos; record terminal obs + truncation flag before auto-reset.
        obs = self._get_obs()
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        done_idx = np.nonzero(dones)[0]
        for i in done_idx:
            infos[i]["terminal_observation"] = obs[i].copy()
            if truncated[i] and not terminated[i]:
                infos[i]["TimeLimit.truncated"] = True

        # VecEnv contract: auto-reset done sub-envs and return the *fresh* obs.
        if done_idx.size > 0:
            self._reset_idx(dones)
            obs = self._get_obs()

        return obs, rewards.astype(np.float32), dones, infos

    # ----------------------------------------------------------------------
    # Dynamics (batched)
    # ----------------------------------------------------------------------
    def _dynamics(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        n = states.shape[0]

        # --- Command: tanh-style action -> physical CTBR ------------------
        # Action channel order [thrust, p, q, r]; internal ARX order [p,q,r,c].
        U = np.clip(actions, -1.0, 1.0)
        c_cmd = 1.0 + U[:, 0]                       # hover at U[:,0]=0
        p_cmd = U[:, 1] * self.rate_max_pq
        q_cmd = U[:, 2] * self.rate_max_pq
        r_cmd = U[:, 3] * self.rate_max_r
        u = np.stack([p_cmd, q_cmd, r_cmd, c_cmd], axis=1)   # (n, 4)

        # --- Per-channel first-order lag + delay (ARX) --------------------
        # Delayed command per channel: d==0 uses the current command, else the
        # command issued d steps ago from the ring buffer (row -1 = newest).
        u_delayed = np.empty_like(u)
        for i in range(4):
            d = self._delays[i]
            u_delayed[:, i] = u[:, i] if d == 0 else self._cmd_buf[:, self._dmax - d, i]

        x_prev = np.concatenate([states[:, 10:13], states[:, 13:14]], axis=1)  # [p,q,r,c]
        x_new = self._arx_a[None] * x_prev + self._arx_b[None] * u_delayed
        rates_n = x_new[:, :3]
        c_n = x_new[:, 3]
        # Push the current command, drop the oldest.
        self._cmd_buf = np.concatenate([self._cmd_buf[:, 1:], u[:, None, :]], axis=1)

        # --- Rigid-body dynamics (NED) ------------------------------------
        pos, quat, vel = states[:, 0:3], states[:, 3:7], states[:, 7:10]
        R = _quat_to_rotmat(quat)                                       # (n,3,3)
        body_thrust = np.zeros((n, 3), dtype=states.dtype)
        body_thrust[:, 2] = -c_n * GRAVITY                             # along body -z
        accel = np.array([0.0, 0.0, GRAVITY]) + np.einsum("nij,nj->ni", R, body_thrust)

        vel_n = vel + self.dt * accel
        pos_n = pos + self.dt * vel_n                                   # semi-implicit
        dq = _quat_from_rotvec(rates_n * self.dt)
        quat_n = _quat_mul(quat, dq)
        quat_n /= np.linalg.norm(quat_n, axis=1, keepdims=True) + 1e-12

        return np.concatenate(
            [pos_n, quat_n, vel_n, rates_n, c_n[:, None]], axis=1
        ).astype(np.float32)

    # ----------------------------------------------------------------------
    # Reward / termination (batched)
    # ----------------------------------------------------------------------
    def _reward_and_termination(self, states: np.ndarray):
        pos = states[:, 0:3]
        pos_err = np.linalg.norm(pos - self.target, axis=1)
        speed = np.linalg.norm(states[:, 7:10], axis=1)
        rate = np.linalg.norm(states[:, 10:13], axis=1)

        rewards = (-pos_err - 0.1 * speed - 0.01 * rate) * self.dt

        ground_collision = pos[:, 2] >= 0.0                            # z down in NED
        rewards[ground_collision] = -10.0

        out_of_bounds = np.any(np.abs(pos[:, 0:2]) > 5.0, axis=1)      # grid edges
        out_of_bounds |= np.any(np.abs(states[:, 10:13]) > 1000.0, axis=1)  # numeric blow-up
        rewards[out_of_bounds] = - 10

        success = pos_err <= 0.2 
        success &= speed <= 0.1
        rewards[success] = 20.0

        terminated = ground_collision | out_of_bounds | success
        return rewards, terminated

    # ----------------------------------------------------------------------
    # Reset / observation helpers
    # ----------------------------------------------------------------------
    def _reset_idx(self, mask: np.ndarray) -> None:
        """Reinitialize the sub-envs selected by boolean ``mask``."""
        k = int(mask.sum())
        if k == 0:
            return

        rng = np.random
        x0 = rng.uniform(-3.0, -2.5, size=k)
        y0 = rng.uniform(-1.0, 1.0, size=k)
        z0 = rng.uniform(-1.8, -1.2, size=k)
        vel0 = rng.uniform(-0.5, 0.5, size=(k, 3))

        phi0 = rng.uniform(-np.pi / 9, np.pi / 9, size=k)
        theta0 = rng.uniform(-np.pi / 9, np.pi / 9, size=k)
        psi0 = rng.uniform(-np.pi / 9, np.pi / 9, size=k)
        quat0 = _euler_to_quat(phi0, theta0, psi0)

        rates0 = rng.uniform(-0.1, 0.1, size=(k, 3))

        self.states[mask, 0:3] = np.stack([x0, y0, z0], axis=1)
        self.states[mask, 3:7] = quat0
        self.states[mask, 7:10] = vel0
        self.states[mask, 10:13] = rates0
        self.states[mask, 13] = 1.0                                    # start at hover
        self.step_counts[mask] = 0

        # Reset command history to the hover command ([p,q,r,c] = [0,0,0,1]).
        self._cmd_buf[mask] = 0.0
        self._cmd_buf[mask, :, 3] = 1.0

    def _get_obs(self) -> np.ndarray:
        pos_err = self.states[:, 0:3] - self.target
        euler = _quat_to_euler(self.states[:, 3:7])
        obs = np.concatenate(
            [pos_err, euler, self.states[:, 7:10],
             self.states[:, 10:13], self.states[:, 13:14]],
            axis=1,
        )
        return obs.astype(np.float32)

    # ----------------------------------------------------------------------
    # VecEnv boilerplate (required abstract methods)
    # ----------------------------------------------------------------------
    def close(self) -> None:
        pass

    def render(self, mode: str | None = None):
        if (mode or self.render_mode) == "rgb_array":
            # TODO: render a frame of env 0. Placeholder black image.
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self, attr_name) for _ in self._idx(indices)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._idx(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._idx(indices)]

    def _idx(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices


# ---------------------------------------------------------------------------
# Quick sanity check + training stub.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = DroneEnv(num_envs=8)

    obs = env.reset()
    assert obs.shape == (8, env.observation_space.shape[0])
    for _ in range(5):
        actions = np.stack([env.action_space.sample() for _ in range(env.num_envs)])
        obs, rewards, dones, infos = env.step(actions)
    print("step() OK -> obs", obs.shape, "rewards", rewards.shape, "dones", dones.shape)

    # Example training run (uncomment to use):
    # from stable_baselines3 import PPO
    #
    # model = PPO("MlpPolicy", DroneEnv(num_envs=64), verbose=1)
    # model.learn(total_timesteps=1_000_000)
    # model.save("drone_ppo")
