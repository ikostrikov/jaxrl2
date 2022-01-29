import collections
import copy
from typing import Iterable, Optional

import gym
import jax
import numpy as np
from flax.core import frozen_dict
from gym.spaces import Box

from jaxrl2.data.dataset import DatasetDict, _sample
from jaxrl2.data.replay_buffer import ReplayBuffer


class MemoryEfficientReplayBuffer(ReplayBuffer):
    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, capacity: int
    ):

        pixel_obs_space = observation_space.spaces["pixels"]
        self._num_stack = pixel_obs_space.shape[-1]
        self._unstacked_dim_size = pixel_obs_space.shape[-2]
        low = pixel_obs_space.low[..., 0]
        high = pixel_obs_space.high[..., 0]
        unstacked_pixel_obs_space = Box(low=low, high=high, dtype=pixel_obs_space.dtype)
        observation_space = copy.deepcopy(observation_space)
        observation_space.spaces["pixels"] = unstacked_pixel_obs_space

        self._first = True
        self._is_correct_index = np.full(capacity, False, dtype=bool)

        next_observation_space_dict = copy.deepcopy(observation_space.spaces)
        next_observation_space_dict.pop("pixels")
        next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space=next_observation_space,
        )

    def insert(self, data_dict: DatasetDict):
        if self._insert_index == 0 and self._capacity == len(self) and not self._first:
            indxs = np.arange(len(self) - self._num_stack, len(self))
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._is_correct_index[self._insert_index] = False
                super().insert(element)

        data_dict = data_dict.copy()
        data_dict["observations"] = data_dict["observations"].copy()
        data_dict["next_observations"] = data_dict["next_observations"].copy()

        obs_pixels = data_dict["observations"].pop("pixels")
        next_obs_pixels = data_dict["next_observations"].pop("pixels")

        if self._first:
            for i in range(self._num_stack):
                data_dict["observations"]["pixels"] = obs_pixels[..., i]
                self._is_correct_index[self._insert_index] = False
                super().insert(data_dict)

        data_dict["observations"]["pixels"] = next_obs_pixels[..., -1]

        self._first = data_dict["dones"]

        self._is_correct_index[self._insert_index] = True
        super().insert(data_dict)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._is_correct_index[indx] = False

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        include_pixels: bool = True,
    ) -> frozen_dict.FrozenDict:

        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

            for i in range(batch_size):
                while not self._is_correct_index[indx[i]]:
                    if hasattr(self.np_random, "integers"):
                        indx[i] = self.np_random.integers(len(self))
                    else:
                        indx[i] = self.np_random.randint(len(self))
        else:
            raise NotImplementedError()

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys

        keys = list(keys)
        keys.remove("observations")

        batch = super().sample(batch_size, keys, indx)
        batch = batch.unfreeze()

        obs_keys = self.dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        obs_keys.remove("pixels")

        batch["observations"] = {}
        for k in obs_keys:
            batch["observations"][k] = _sample(
                self.dataset_dict["observations"][k], indx
            )

        obs_pixels = self.dataset_dict["observations"]["pixels"]
        obs_pixels = np.lib.stride_tricks.sliding_window_view(
            obs_pixels, self._num_stack + 1, axis=0
        )
        obs_pixels = obs_pixels[indx - self._num_stack]

        if include_pixels:
            batch["observations"]["pixels"] = obs_pixels[..., :-1]
            if "next_observations" in keys:
                batch["next_observations"]["pixels"] = obs_pixels[..., 1:]
        else:
            batch["observations"]["pixels"] = obs_pixels

        return frozen_dict.freeze(batch)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
