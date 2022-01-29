#! /usr/bin/env python
import gym
import jax
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl2.agents import BCLearner, IQLLearner
from jaxrl2.data import D4RLDataset
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_float("filter_percentile", None, "Take top N% trajectories.")
flags.DEFINE_float(
    "filter_threshold", None, "Take trajectories with returns above the threshold."
)
config_flags.DEFINE_config_file(
    "config",
    "configs/offline_config.py:bc",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    wandb.init(project="jaxrl2_offline")
    wandb.config.update(FLAGS)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env)
    env.seed(FLAGS.seed)

    dataset = D4RLDataset(env)
    if FLAGS.filter_percentile is not None or FLAGS.filter_threshold is not None:
        dataset.filter(
            percentile=FLAGS.filter_percentile, threshold=FLAGS.filter_threshold
        )
    dataset.seed(FLAGS.seed)

    if "antmaze" in FLAGS.env_name:
        dataset.dataset_dict["rewards"] *= 100
    elif FLAGS.env_name.split("-")[0] in ["hopper", "halfcheetah", "walker2d"]:
        dataset.normalize_returns(scaling=1000)

    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = FLAGS.max_steps
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs
    )

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        batch = dataset.sample(FLAGS.batch_size)
        info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            info = jax.device_get(info)
            wandb.log(info, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, env, num_episodes=FLAGS.eval_episodes)
            eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)


if __name__ == "__main__":
    app.run(main)
