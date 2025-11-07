import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def map_node_to_dof(n: ArrayLike) -> jax.Array:
    return 3 * jnp.asarray(n)[..., None] + jnp.arange(3)


def parallel_transport(u: jax.Array, t1: jax.Array, t2: jax.Array) -> jax.Array:
    def parallel_case(args: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        u, _, _ = args
        return u  # When parallel, no transport needed

    def non_parallel_case(args: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        u, t1, t2 = args
        b = jnp.cross(t1, t2)
        b_unit = b / jnp.linalg.norm(b)
        # Gram-Schmidt orthogonalization
        b_unit = b_unit - jnp.dot(b_unit, t1) * t1
        b_unit = b_unit / jnp.linalg.norm(b_unit)
        b_unit = b_unit - jnp.dot(b_unit, t2) * t2
        b_unit = b_unit / jnp.linalg.norm(b_unit)
        n1 = jnp.cross(t1, b_unit)
        n2 = jnp.cross(t2, b_unit)
        transported = (
            jnp.dot(u, t1) * t2 + jnp.dot(u, n1) * n2 + jnp.dot(u, b_unit) * b_unit
        )
        return transported

    # lax.cond necessary to separate parallel from non parallel gradient
    b = jnp.cross(t1, t2)
    b_norm = jnp.linalg.norm(b)
    return jax.lax.cond(b_norm < 1e-12, parallel_case, non_parallel_case, (u, t1, t2))


def signed_angle(u: jax.Array, v: jax.Array, n: ArrayLike) -> jax.Array:
    w = jnp.cross(u, v)
    dot_uv = jnp.dot(u, v)
    signed_sin = jnp.dot(w, n)
    angle = jnp.atan2(signed_sin, dot_uv)
    return angle


def rotate_axis_angle(u: jax.Array, v: jax.Array, theta: ArrayLike) -> jax.Array:
    return (
        jnp.cos(theta) * u
        + jnp.sin(theta) * jnp.cross(v, u)
        + jnp.dot(v, u) * (1 - jnp.cos(theta)) * v
    )
