"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.drq.augmentations import batched_random_crop
from jaxrl2.agents.sac.actor_updater import update_actor
from jaxrl2.agents.sac.critic_updater import update_critic
from jaxrl2.agents.sac.temperature import Temperature
from jaxrl2.agents.sac.temperature_updater import update_temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import D4PGEncoder, ResNetV2Encoder
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


# Helps to minimize CPU to GPU transfer.
def _unpack(batch):
    # Assuming that if next_observation is missing, it's combined with observation:
    obs_pixels = batch["observations"]["pixels"][..., :-1]
    next_obs_pixels = batch["observations"]["pixels"][..., 1:]

    obs = batch["observations"].copy(add_or_replace={"pixels": obs_pixels})
    next_obs = batch["next_observations"].copy(
        add_or_replace={"pixels": next_obs_pixels}
    )

    batch = batch.copy(
        add_or_replace={"observations": obs, "next_observations": next_obs}
    )

    return batch


def _share_encoder(source, target):
    # Use critic conv layers in actor:
    new_params = target.params.copy(
        add_or_replace={"encoder": source.params["encoder"]}
    )
    return target.replace(params=new_params)


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
    batch = _unpack(batch)
    actor = _share_encoder(source=critic, target=actor)

    rng, key = jax.random.split(rng)
    aug_pixels = batched_random_crop(key, batch["observations"]["pixels"])
    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch = batch.copy(add_or_replace={"observations": observations})

    rng, key = jax.random.split(rng)
    aug_next_pixels = batched_random_crop(key, batch["next_observations"]["pixels"])
    next_observations = batch["next_observations"].copy(
        add_or_replace={"pixels": aug_next_pixels}
    )
    batch = batch.copy(add_or_replace={"next_observations": next_observations})

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
    new_actor, actor_info = update_actor(key, actor, critic, temp, batch)
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


class DrQLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_reduction: str = "min",
        init_temperature: float = 1.0,
        encoder: str = "d4pg",
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if encoder == "d4pg":
            encoder_def = D4PGEncoder(
                cnn_features, cnn_filters, cnn_strides, cnn_padding
            )
        elif encoder == "resnet":
            encoder_def = ResNetV2Encoder((2, 2, 2, 2))

        policy_def = NormalTanhPolicy(hidden_dims, action_dim)
        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            stop_gradient=True,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = StateActionEnsemble(hidden_dims, num_qs=2)
        critic_def = PixelMultiplexer(
            encoder=encoder_def, network=critic_def, latent_dim=latent_dim
        )
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
