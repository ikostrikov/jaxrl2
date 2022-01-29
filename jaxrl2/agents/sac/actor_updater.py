from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_actor(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
) -> Tuple[TrainState, Dict[str, float]]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        qs = critic.apply_fn({"params": critic.params}, batch["observations"], actions)
        q = qs.mean(axis=0)
        actor_loss = (log_probs * temp.apply_fn({"params": temp.params}) - q).mean()
        return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
