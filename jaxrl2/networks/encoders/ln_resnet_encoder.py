# Based on:
# https://github.com/google/flax/blob/main/examples/imagenet/models.py
# and
# https://github.com/google-research/big_transfer/blob/master/bit_jax/models.py
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any


class ResNetV2Block(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.norm()(x)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides)(residual)

        return residual + y


class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


class ResNetV2Encoder(nn.Module):
    """ResNetV2."""

    stage_sizes: Tuple[int]
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)

        x = x.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        if x.shape[-2] == 224:
            x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)])(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        else:
            x = conv(self.num_filters, (3, 3))(x)

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = ResNetV2Block(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)

        x = norm()(x)
        x = self.act(x)
        return x.reshape((*x.shape[:-3], -1))
