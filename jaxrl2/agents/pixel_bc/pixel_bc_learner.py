"""Implementations of algorithms for continuous control."""

from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.bc.actor_updater import log_prob_update
from jaxrl2.agents.drq.augmentations import batched_random_crop
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import D4PGEncoder, ResNetV2Encoder
from jaxrl2.networks.normal_policy import UnitStdNormalPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer
from jaxrl2.types import Params, PRNGKey


@jax.jit
def _update_jit(
    rng: PRNGKey, actor: TrainState, batch: TrainState
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    rng, key = jax.random.split(rng)
    aug_pixels = batched_random_crop(key, batch["observations"]["pixels"])
    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch = batch.copy(add_or_replace={"observations": observations})

    rng, new_actor, actor_info = log_prob_update(rng, actor, batch)

    return rng, new_actor, actor_info


class PixelBCLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        dropout_rate: Optional[float] = None,
        encoder: str = "d4pg",
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        if encoder == "d4pg":
            encoder_def = D4PGEncoder(
                cnn_features, cnn_filters, cnn_strides, cnn_padding
            )
        elif encoder == "resnet":
            encoder_def = ResNetV2Encoder((2, 2, 2, 2))

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        policy_def = UnitStdNormalPolicy(
            hidden_dims, action_dim, dropout_rate=dropout_rate
        )
        actor_def = PixelMultiplexer(
            encoder=encoder_def, network=policy_def, latent_dim=latent_dim
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        self._rng = rng
        self._actor = actor

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, info = _update_jit(self._rng, self._actor, batch)

        self._rng = new_rng
        self._actor = new_actor

        return info
