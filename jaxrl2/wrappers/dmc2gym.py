# Taken from
# https://github.com/denisyarats/dmc2gym
# and modified to exclude duplicated code.

import copy
from typing import OrderedDict

import dm_env
import gym
import numpy as np
from gym import spaces


def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict) or isinstance(spec, dict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(
            low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=spec.dtype
        )
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float("-inf")
            high = float("inf")
        else:
            raise ValueError()

        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    else:
        raise NotImplementedError


def dmc_obs2gym_obs(obs):
    if isinstance(obs, OrderedDict) or isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = dmc_obs2gym_obs(v)
        return obs
    else:
        return np.asarray(obs)


class DMC2GYM(gym.core.Env):
    def __init__(self, env: dm_env.Environment):
        self._env = env

        self.action_space = dmc_spec2gym_space(self._env.action_spec())

        self.observation_space = dmc_spec2gym_space(self._env.observation_spec())

        self.viewer = None

    def _get_viewer(self):
        if self.viewer is None:
            from gym.envs.mujoco.mujoco_rendering import Viewer

            self.viewer = Viewer(
                self._env.physics.model.ptr, self._env.physics.data.ptr
            )
        return self.viewer

    def __getattr__(self, name):
        return getattr(self._env, name)

    def seed(self, seed: int):
        if hasattr(self._env, "random_state"):
            self._env.random_state.seed(seed)
        else:
            self._env.task.random.seed(seed)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        obs = time_step.observation

        info = {}
        if done and time_step.discount == 1.0:
            info["TimeLimit.truncated"] = True

        return dmc_obs2gym_obs(obs), reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = time_step.observation
        return dmc_obs2gym_obs(obs)

    def render(
        self, mode="rgb_array", height: int = 84, width: int = 84, camera_id: int = 0
    ):
        assert mode in ["human", "rgb_array"], (
            "only support rgb_array and human mode, given %s" % mode
        )
        if mode == "rgb_array":
            return self._env.physics.render(
                height=height, width=width, camera_id=camera_id
            )
        elif mode == "human":
            self._get_viewer().render()
