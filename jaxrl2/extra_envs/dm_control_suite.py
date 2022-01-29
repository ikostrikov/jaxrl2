import gym
from dm_control import suite
from gym.envs.registration import register

from jaxrl2.wrappers import DMC2GYM


def create_dm_control_env(domain_name: str, task_name: str) -> gym.Env:
    env = suite.load(domain_name=domain_name, task_name=task_name)
    return DMC2GYM(env)


create_dm_control_env.metadata = DMC2GYM.metadata

for (domain_name, task_name) in suite.ALL_TASKS:
    register(
        id=f"{domain_name}-{task_name}-v0",
        entry_point="jaxrl2.extra_envs.dm_control_suite:create_dm_control_env",
        max_episode_steps=1000,
        kwargs=dict(domain_name=domain_name, task_name=task_name),
    )
