from typing import Callable
import jax
import jax.numpy as jnp
import jaxtyping


def newton(
    x: jax.Array,
    residual: Callable[[jax.Array], jax.Array],
    hessian: Callable[[jax.Array], jax.Array],
    aux: jaxtyping.PyTree,
) -> tuple[jax.Array, None]:
    return x - jnp.linalg.solve(hessian(x), residual(x)), None
