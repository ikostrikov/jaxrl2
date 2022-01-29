import gym
import numpy as np
import pytest

from jaxrl2.data import Dataset
from jaxrl2.data.d4rl_dataset import D4RLDataset

DATASET_LEN = 10
DATASET_DICT = {
    "obs": {
        "goal": np.zeros((DATASET_LEN, 1)),
        "states": {
            "position": np.zeros((DATASET_LEN, 3)),
            "velocity": np.zeros((DATASET_LEN, 3)),
        },
    },
    "action": np.ones((DATASET_LEN, 1)),
}

BAD_LEN_DATASET_DICT = {
    "obs": {"goal": np.zeros((DATASET_LEN, 1)), "start": np.zeros((DATASET_LEN, 1))},
    "action": np.ones((1, 1)),
}

BAD_TYPE_DATASET_DICT = {
    "obs": {"goal": np.zeros((DATASET_LEN, 1)), "start": np.zeros((DATASET_LEN, 1))},
    "action": 1,
}

BATCH_SIZE = 3

KEYS = ["action"]


def test_init():
    with pytest.raises(AssertionError):
        dataset = Dataset(BAD_LEN_DATASET_DICT)


def test_type():
    with pytest.raises(TypeError):
        dataset = Dataset(BAD_TYPE_DATASET_DICT)


def test_length():
    dataset = Dataset(DATASET_DICT)

    assert len(dataset) == DATASET_LEN


def test_sample():
    dataset = Dataset(DATASET_DICT)
    batch = dataset.sample(BATCH_SIZE)
    assert len(batch["action"]) == BATCH_SIZE


def test_sample_keys():
    dataset = Dataset(DATASET_DICT)
    batch = dataset.sample(BATCH_SIZE, keys=KEYS)
    assert list(batch.keys()) == KEYS


def test_d4rl_dataset():
    env = gym.make("halfcheetah-expert-v2")
    dataset = D4RLDataset(env)

    batch = dataset.sample(BATCH_SIZE)


def test_split():
    dataset = Dataset(DATASET_DICT)
    train_dataset, test_dataset = dataset.split(0.6)
    assert len(train_dataset) == 6
    assert len(test_dataset) == 4
