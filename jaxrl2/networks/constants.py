import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)
