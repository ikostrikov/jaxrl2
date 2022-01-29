from typing import Callable, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl2.networks.mlp import MLP
from jaxrl2.types import PRNGKey


def cont2disc(values: jnp.ndarray, n: int) -> jnp.ndarray:
    values = (values + 1) / 2
    values = values * n
    values = jnp.floor(values)
    return jnp.clip(values, 0, n - 1)


def disc2cont(values: jnp.ndarray, n: int) -> jnp.ndarray:
    values = values + 0.5
    values = values / n
    return values * 2 - 1


def quantize(values: jnp.ndarray, n: int) -> jnp.ndarray:
    values = cont2disc(values, n)
    return disc2cont(values, n)


class AR(distrax.Distribution):
    def __init__(
        self,
        param_fn: Callable[..., Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        batch_shape: Tuple[int],
        event_dim: int,
        num_classes: int,
        beam_size: int = 20,
    ):
        self._param_fn = param_fn
        self._event_dim = event_dim
        self._event_dtype = jnp.float32
        self._batch_shape = batch_shape
        self._num_classes = num_classes
        self._beam_size = beam_size

    def _sample_n(self, key: PRNGKey, n: int) -> jnp.ndarray:
        keys = jax.random.split(key, self._event_dim)

        samples = jnp.zeros((n, *self._batch_shape, 0), self._event_dtype)

        # TODO: Consider rewriting it with nn.scan.
        for i in range(self._event_dim):
            dist = self._distr_fn(samples)
            dim_samples = dist.sample(seed=keys[i])
            dim_samples = disc2cont(dim_samples, self._num_classes)
            samples = jnp.concatenate([samples, dim_samples[..., jnp.newaxis]], axis=-1)

        return samples

    def log_prob(self, values: jnp.ndarray) -> jnp.ndarray:
        targets = cont2disc(values, self._num_classes)
        targets = targets.astype(jnp.int32)

        log_probs = []
        for i in range(values.shape[-1]):
            log_probs.append(self._distr_fn(values[..., :i]).log_prob(targets[..., i]))

        return jnp.stack(log_probs, axis=-1).sum(axis=-1)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self._event_dim,)

    def _distr_fn(self, samples: jnp.ndarray) -> distrax.Distribution:
        logits = self._param_fn(samples)
        return distrax.Categorical(logits=logits)

        # return distrax.Transformed(dist, distrax.Tanh())

    def mode(self):
        # return self.sample(seed=jax.random.PRNGKey(42))
        beam = jnp.zeros((self._beam_size, *self._batch_shape, 0), self._event_dtype)
        log_probs = 0

        # TODO: Consider rewriting it with nn.scan.
        for i in range(self._event_dim):
            dist = self._distr_fn(beam)

            mean = jnp.arange(self._num_classes)
            mean = disc2cont(mean, self._num_classes)
            mean = jnp.broadcast_to(mean, (*beam.shape[:-1], self._num_classes))
            mean = jnp.moveaxis(mean, -1, 0)

            candidate = jnp.repeat(beam[jnp.newaxis], self._num_classes, axis=0)
            candidate = jnp.concatenate([candidate, mean[..., jnp.newaxis]], axis=-1)

            mean = cont2disc(mean, self._num_classes)
            log_probs = log_probs + dist.log_prob(mean)

            log_probs = jnp.reshape(log_probs, (-1, *self._batch_shape))
            indx = jnp.argsort(log_probs, axis=0)[-self._beam_size :]

            candidate = jnp.reshape(candidate, (-1, *self._batch_shape, i + 1))
            beam = jnp.take_along_axis(
                candidate, jnp.expand_dims(indx, axis=-1), axis=0
            )
            log_probs = jnp.take_along_axis(log_probs, indx, axis=0)

        return beam[-1]


class ARPolicy(nn.Module):
    features: Sequence[int]
    action_dim: int
    num_components: int = 100
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self, states: jnp.ndarray, training: bool = False
    ) -> distrax.Distribution:
        is_initializing = len(self.variables) == 0

        mlps = []

        for i in range(self.action_dim):
            mlps.append(
                MLP(
                    (*self.features, self.num_components),
                    dropout_rate=self.dropout_rate,
                    scale_final=1e-4,
                )
            )

        if is_initializing:
            for i in range(self.action_dim):
                actions = jnp.zeros((*states.shape[:-1], i), states.dtype)
                inputs = {"actions": actions, "states": states}
                mlps[i](inputs)

        def param_fn(actions: Optional[jnp.ndarray] = None) -> distrax.Distribution:
            new_state_shape = (*actions.shape[:-1], states.shape[-1])
            states_ = jnp.broadcast_to(states, new_state_shape)

            actions = quantize(actions, self.num_components)

            inputs = {"actions": actions, "states": states_}
            indx = actions.shape[-1]

            return mlps[indx](inputs, training=training)

        return AR(param_fn, states.shape[:-1], self.action_dim, self.num_components)
