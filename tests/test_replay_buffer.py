import gym
import numpy as np

import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.data import MemoryEfficientReplayBuffer, ReplayBuffer
from jaxrl2.wrappers import wrap_pixels

BATCH_SIZE = 3

CAPACITY = 10


def test_replay_buffer():
    env = gym.make("HalfCheetah-v3")

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, CAPACITY)

    for i in range(2 * CAPACITY):
        obs = env.observation_space.sample()
        act = env.action_space.sample()

        replay_buffer.insert(
            dict(
                observations=obs,
                actions=act,
                rewards=1.0,
                next_observations=obs,
                masks=1.0,
                dones=False,
            )
        )

    replay_buffer.sample(BATCH_SIZE)


def test_replay_buffer_dict_obs():
    env = gym.make("cheetah-run-v0")
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, CAPACITY)

    for i in range(2 * CAPACITY):
        obs = env.observation_space.sample()
        act = env.action_space.sample()
        replay_buffer.insert(
            dict(
                observations=obs,
                actions=act,
                rewards=1.0,
                next_observations=obs,
                masks=1.0,
                dones=False,
            )
        )
    replay_buffer.sample(BATCH_SIZE)


def test_efficient_replay_buffer():
    env = gym.make("cheetah-run-v0")
    env = wrap_pixels(env, action_repeat=2, image_size=2)
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, 20
    )

    obs = env.reset()
    obs["pixels"][:] = 1
    for i in range(20):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        next_obs["pixels"][:] = i + 2
        reward = i + 2

        replay_buffer.insert(
            dict(
                observations=obs,
                actions=action,
                rewards=reward,
                masks=1.0,
                dones=done,
                next_observations=next_obs,
            )
        )
        obs = next_obs

    for i in range(20):
        batch = replay_buffer.sample(4, include_pixels=False)
        obs = batch["observations"]["pixels"][..., :-1]
        next_obs = batch["observations"]["pixels"][..., 1:]
        reward = batch["rewards"]

        assert np.all(next_obs - obs >= 0)
        assert np.all(next_obs - obs <= 1)
        assert np.allclose(next_obs[..., -1], np.reshape(reward, [-1, 1, 1, 1]))


def test_efficient_replay_buffer_with_observation():
    env = gym.make("cheetah-run-v0")
    env = wrap_pixels(env, action_repeat=2, image_size=2)
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, 20
    )

    obs = env.reset()
    obs["pixels"][:] = 1
    for i in range(20):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        next_obs["pixels"][:] = i + 2
        reward = i + 2

        replay_buffer.insert(
            dict(
                observations=obs,
                actions=action,
                rewards=reward,
                masks=1.0,
                dones=done,
                next_observations=next_obs,
            )
        )
        obs = next_obs

    for i in range(20):
        batch = replay_buffer.sample(4)
        obs = batch["observations"]["pixels"]
        next_obs = batch["next_observations"]["pixels"]
        reward = batch["rewards"]

        assert np.all(next_obs - obs >= 0)
        assert np.all(next_obs - obs <= 1)
        assert np.allclose(next_obs[..., -1], np.reshape(reward, [-1, 1, 1, 1]))
