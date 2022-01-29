import enum
from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init


class MaskType(enum.Enum):
    input = 1
    hidden = 2
    output = 3


@jax.util.cache()
def get_mask(
    input_dim: int, output_dim: int, randvar_dim: int, mask_type: MaskType
) -> jnp.DeviceArray:
    """
    Create a mask for MADE.

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf

    Args:
        input_dim: Dimensionality of the inputs.
        output_dim: Dimensionality of the outputs.
        rand_var_dim: Dimensionality of the random variable.
        mask_type: MaskType.

    Returns:
        A mask.
    """
    if mask_type == MaskType.input:
        in_degrees = jnp.arange(input_dim) % randvar_dim
    else:
        in_degrees = jnp.arange(input_dim) % (randvar_dim - 1)

    if mask_type == MaskType.output:
        out_degrees = jnp.arange(output_dim) % randvar_dim - 1
    else:
        out_degrees = jnp.arange(output_dim) % (randvar_dim - 1)

    in_degrees = jnp.expand_dims(in_degrees, 0)
    out_degrees = jnp.expand_dims(out_degrees, -1)
    return (out_degrees >= in_degrees).astype(jnp.float32).transpose()


class MaskedDense(nn.Dense):
    event_size: int = 1
    mask_type: MaskType = MaskType.hidden
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        kernel = jnp.asarray(kernel, self.dtype)

        mask = get_mask(*kernel.shape, self.event_size, self.mask_type)
        kernel = kernel * mask

        y = jax.lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y


class MaskedMLP(nn.Module):
    features: Sequence[int]
    activate_final: bool = False
    dropout_rate: Optional[float] = 0.1

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, conds: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        x = inputs
        x_conds = conds
        for i, feat in enumerate(self.features):
            if i == 0:
                mask_type = MaskType.input
            elif i + 1 < len(self.features):
                mask_type = MaskType.hidden
            else:
                mask_type = MaskType.output

            should_activate = i + 1 < len(self.features) or self.activate_final
            x = MaskedDense(
                feat,
                event_size=inputs.shape[-1],
                mask_type=mask_type,
                kernel_init=default_init(jnp.sqrt(2) if should_activate else 1e-4),
            )(x)
            x_conds = nn.Dense(
                feat, kernel_init=default_init(jnp.sqrt(2) if should_activate else 1e-4)
            )(x_conds)
            x = x + x_conds
            if should_activate:
                x = nn.relu(x)
                x_conds = nn.relu(x_conds)
                if self.dropout_rate is not None:
                    x_conds = jnp.broadcast_to(x_conds, x.shape)
                    x_tmp = jnp.stack([x_conds, x])
                    x_tmp = nn.Dropout(rate=self.dropout_rate)(
                        x_tmp, deterministic=not training
                    )
                    x_conds, x = x_tmp
        return x
