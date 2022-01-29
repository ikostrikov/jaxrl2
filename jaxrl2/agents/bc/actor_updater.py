from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def log_prob_update(
    rng: PRNGKey, actor: TrainState, batch: FrozenDict
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist = actor.apply_fn(
            {"params": actor_params},
            batch["observations"],
            training=True,
            rngs={"dropout": key},
        )
        log_probs = dist.log_prob(batch["actions"])
        actor_loss = -log_probs.mean()
        return actor_loss, {"bc_loss": actor_loss}

    grads, info = jax.grad(loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return rng, new_actor, info
