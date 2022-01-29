import subprocess

import jax
import numpy as np

from jaxrl2 import networks


def test_d4rl_tanh_normal_run():
    """Check that a D4RL gym run completes with no errors."""
    command = [
        "python",
        "examples/train_offline.py",
        "--config",
        "examples/configs/offline_config.py:bc",
        "--max_steps",
        "110",
        "--eval_interval",
        "100",
        "--batch_size",
        "32",
        "--eval_episodes",
        "1",
        "--config.model_config.distr",
        "tanh_normal",
        "--config.model_config.hidden_dims",
        "(32, 32)",
    ]
    subprocess.run(command, check=True)


def test_d4rl_ar_mog_run():
    """Check that a D4RL gym run completes with no errors."""
    command = [
        "python",
        "examples/train_offline.py",
        "--config",
        "examples/configs/offline_config.py:bc",
        "--max_steps",
        "110",
        "--eval_interval",
        "100",
        "--batch_size",
        "32",
        "--eval_episodes",
        "1",
        "--config.model_config.distr",
        "ar",
        "--config.model_config.hidden_dims",
        "(32, 32)",
    ]
    subprocess.run(command, check=True)


def test_ar_sampling():
    actor_def = networks.ARPolicy((32, 32), 5, dropout_rate=None)
    obs = np.zeros((2, 7), dtype=np.float32)
    dist, _ = actor_def.init_with_output(jax.random.PRNGKey(42), obs)
    x = dist.sample(seed=jax.random.PRNGKey(42))
    assert x.shape == (2, 5)
