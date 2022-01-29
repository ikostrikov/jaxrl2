import subprocess

import jaxrl2.extra_envs.dm_control_suite


def test_iql():
    """Check that a D4RL gym run completes with no errors."""
    command = [
        "python",
        "examples/train_offline.py",
        "--config",
        "examples/configs/offline_config.py:iql_antmaze",
        "--env_name",
        "antmaze-large-play-v2",
        "--max_steps",
        "110",
        "--batch_size",
        "32",
        "--eval_interval",
        "100",
        "--eval_episodes",
        "1",
        "--config.model_config.hidden_dims",
        "(32, 32)",
    ]
    subprocess.run(command, check=True)
