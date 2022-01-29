import jax

from jaxrl2.types import Params


def soft_target_update(
    critic_params: Params, target_critic_params: Params, tau: float
) -> Params:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic_params, target_critic_params
    )

    return new_target_params
