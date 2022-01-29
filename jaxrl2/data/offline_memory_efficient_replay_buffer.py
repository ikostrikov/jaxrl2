import collections
import glob

import gym
import jax
import numpy as np

from jaxrl2.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer


class OfflineMemoryEfficientReplayBuffer(MemoryEfficientReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        include_pixels: bool = True,
        data_url: str = "",
        replay: int = 10,
    ):

        super().__init__(observation_space, action_space, capacity, include_pixels)

        self._data_url = data_url
        self._replay = replay
        self.load()

    def load(self):
        while self._size < self._capacity:
            self.load_episode()

    def load_episode(self):
        file = self.np_random.choice(glob.glob(self._data_url + "*.pkl"))
        episode = np.load(file, allow_pickle=True)
        frames = collections.deque(maxlen=self._num_stack)
        for _ in range(self._num_stack):
            frames.append(
                np.concatenate(
                    [
                        episode["camera_0_rgb"][0],
                        episode["camera_1_rgb"][0],
                        episode["camera_gripper_rgb"][0],
                    ],
                    axis=-1,
                )
            )
        for t in range(episode["reward"].shape[0] - 1):
            transition = dict()
            transition["observations"] = dict()
            transition["observations"]["pixels"] = np.stack(frames, axis=-1)
            transition["observations"]["ee_forces"] = episode["ee_forces"][t]
            transition["observations"]["ee_qp"] = episode["ee_qp"][t]
            transition["observations"]["robot_qp"] = episode["robot_qp"][t]

            transition["actions"] = episode["action"][t + 1]
            transition["rewards"] = episode["reward"][t + 1]

            frames.append(
                np.concatenate(
                    [
                        episode["camera_0_rgb"][t + 1],
                        episode["camera_1_rgb"][t + 1],
                        episode["camera_gripper_rgb"][t + 1],
                    ],
                    axis=-1,
                )
            )

            transition["next_observations"] = dict()
            transition["next_observations"]["pixels"] = np.stack(frames, axis=-1)
            transition["next_observations"]["ee_forces"] = episode["ee_forces"][t + 1]
            transition["next_observations"]["ee_qp"] = episode["ee_qp"][t + 1]
            transition["next_observations"]["robot_qp"] = episode["robot_qp"][t + 1]

            transition["masks"] = 0.0
            transition["dones"] = 0.0

            self.insert(transition)

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
            for _ in range(self._replay):
                yield queue.popleft()
                enqueue(1)
            self.load_episode()
