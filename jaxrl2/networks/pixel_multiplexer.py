from typing import Dict, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init


class PixelMultiplexer(nn.Module):
    encoder: nn.Module
    network: nn.Module
    latent_dim: int
    stop_gradient: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        assert (
            len(observations.keys()) <= 2
        ), "Can include only pixels and states fields."

        x = self.encoder(observations["pixels"])

        if self.stop_gradient:
            # We do not update conv layers with policy gradients.
            x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        if "states" in observations:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(
                observations["states"]
            )
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)

        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)
