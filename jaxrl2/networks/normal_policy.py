from typing import Callable, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init


class UnitStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    apply_tanh: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, training: bool = False
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
            dropout_rate=self.dropout_rate,
            activations=self.activations,
        )(observations, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.apply_tanh:
            means = nn.tanh(means)

        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.ones_like(means)
        )
