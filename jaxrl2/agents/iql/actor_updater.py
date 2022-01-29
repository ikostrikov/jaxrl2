from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def update_actor(
    key: PRNGKey,
    actor: TrainState,
    target_critic: TrainState,
    value: TrainState,
    batch: FrozenDict,
    A_scaling: float,
    critic_reduction: str,
) -> Tuple[TrainState, Dict[str, float]]:
    v = value.apply_fn({"params": value.params}, batch["observations"])

    qs = target_critic.apply_fn(
        {"params": target_critic.params}, batch["observations"], batch["actions"]
    )
    if critic_reduction == "min":
        q = qs.min(axis=0)
    elif critic_reduction == "mean":
        q = qs.mean(axis=0)
    else:
        raise NotImplemented()
    exp_a = jnp.exp((q - v) * A_scaling)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist = actor.apply_fn(
            {"params": actor_params},
            batch["observations"],
            training=True,
            rngs={"dropout": key},
        )
        log_probs = dist.log_prob(batch["actions"])
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {"actor_loss": actor_loss, "adv": q - v}

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
