from typing import Any, Callable
import jax
import jax.numpy as jnp


def newton(
    x: jax.Array,
    residual: Callable[[jax.Array], jax.Array],
    hessian: Callable[[jax.Array], jax.Array],
    aux: Any = None,
) -> tuple[jax.Array, None]:
    return x - jnp.linalg.solve(hessian(x), residual(x)), None
