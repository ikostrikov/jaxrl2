import subprocess

import jaxrl2.extra_envs.dm_control_suite


def test_online_dmc():
    """Check that a D4RL gym run completes with no errors."""
    command = [
        "python",
        "examples/train_online.py",
        "--env_name",
        "cheetah-run-v0",
        "--config",
        "examples/configs/sac_default.py",
        "--max_steps",
        "110",
        "--start_training",
        "50",
        "--batch_size",
        "32",
        "--eval_interval",
        "100",
        "--eval_episodes",
        "1",
        "--config.hidden_dims",
        "(32, 32)",
    ]
    subprocess.run(command, check=True)


def test_drq_dmc():
    """Check that a D4RL gym run completes with no errors."""
    command = [
        "python",
        "examples/train_online_pixels.py",
        "--env_name",
        "cheetah-run-v0",
        "--config",
        "examples/configs/drq_default.py",
        "--max_steps",
        "210",
        "--action_repeat",
        "2",
        "--start_training",
        "90",
        "--eval_interval",
        "100",
        "--batch_size",
        "32",
        "--eval_episodes",
        "1",
        "--config.hidden_dims",
        "(32, 32)",
        "--config.cnn_features",
        "(2, 4, 8, 16)",
    ]
    subprocess.run(command, check=True)
