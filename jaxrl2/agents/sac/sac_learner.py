"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.sac.actor_updater import update_actor
from jaxrl2.agents.sac.critic_updater import update_critic
from jaxrl2.agents.sac.temperature import Temperature
from jaxrl2.agents.sac.temperature_updater import update_temperature
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


@functools.partial(jax.jit, static_argnames=("backup_entropy", "critic_reduction"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    batch: FrozenDict,
    discount: float,
    tau: float,
    target_entropy: float,
    backup_entropy: bool,
    critic_reduction: str,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:

    rng, key = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(
        key,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        discount,
        backup_entropy=backup_entropy,
        critic_reduction=critic_reduction,
    )
    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = update_temperature(
        temp, actor_info["entropy"], target_entropy
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


class SACLearner(Agent):
    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_reduction: str = "min",
        init_temperature: float = 1.0,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if np.all(action_space.low == -1) and np.all(action_space.high == 1):
            low = None
            high = None
        else:
            low = action_space.low
            high = action_space.high

        actor_def = NormalTanhPolicy(hidden_dims, action_dim, low=low, high=high)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = StateActionEnsemble(hidden_dims, num_qs=2)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        self._rng = rng

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic_params,
            new_temp,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy,
            self.critic_reduction,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        self._temp = new_temp

        return info
