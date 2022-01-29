import collections

import gym
import numpy as np
from gym.spaces import Box


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self._num_stack = num_stack

        assert "pixels" in self.observation_space.spaces
        pixel_obs_spaces = self.observation_space.spaces["pixels"]

        self._env_dim = pixel_obs_spaces.shape[-1]

        low = np.repeat(pixel_obs_spaces.low[..., np.newaxis], num_stack, axis=-1)
        high = np.repeat(pixel_obs_spaces.high[..., np.newaxis], num_stack, axis=-1)
        new_pixel_obs_spaces = Box(low=low, high=high, dtype=pixel_obs_spaces.dtype)
        self.observation_space.spaces["pixels"] = new_pixel_obs_spaces

        self._frames = collections.deque(maxlen=num_stack)

    def reset(self):
        obs = self.env.reset()
        for i in range(self._num_stack):
            self._frames.append(obs["pixels"])
        obs["pixels"] = self.frames
        return obs

    @property
    def frames(self):
        return np.stack(self._frames, axis=-1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs["pixels"])
        obs["pixels"] = self.frames
        return obs, reward, done, info
