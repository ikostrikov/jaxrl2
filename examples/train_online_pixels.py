#! /usr/bin/env python
import os
import pickle

import gym
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import DrQLearner
from jaxrl2.data import MemoryEfficientReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_pixels

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", None, "Number of training steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")
config_flags.DEFINE_config_file(
    "config",
    "configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup-v0": 8,
    "reacher-easy-v0": 4,
    "cheetah-run-v0": 4,
    "finger-spi-n-0": 2,
    "ball_in_cup-catch-v0": 4,
    "walker-walk-v0": 2,
}


def main(_):
    wandb.init(project="jaxrl2_online_pixels")
    wandb.config.update(FLAGS)

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    def wrap(env):
        if "quadruped" in FLAGS.env_name:
            camera_id = 2
        else:
            camera_id = 0
        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            image_size=FLAGS.image_size,
            num_stack=FLAGS.num_stack,
            camera_id=camera_id,
        )

    env = gym.make(FLAGS.env_name)
    env = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap(eval_env)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = DrQLearner(
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs
    )
    replay_buffer_size = FLAGS.replay_buffer_size or FLAGS.max_steps // action_repeat
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size
    )
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size, "include_pixels": False}
    )

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i * action_repeat)

        if i >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i * action_repeat)

        if i % FLAGS.eval_interval == 0:
            if FLAGS.save_buffer:
                dataset_folder = os.path.join("datasets")
                os.makedirs("datasets", exist_ok=True)
                dataset_file = os.path.join(dataset_folder, f"{FLAGS.env_name}")
                with open(dataset_file, "wb") as f:
                    pickle.dump(replay_buffer, f)

            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i * action_repeat)


if __name__ == "__main__":
    app.run(main)
