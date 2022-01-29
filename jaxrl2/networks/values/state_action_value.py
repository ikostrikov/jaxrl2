from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.mlp import MLP


class StateActionValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        inputs = {"states": observations, "actions": actions}
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(
            inputs, training=training
        )
        return jnp.squeeze(critic, -1)
