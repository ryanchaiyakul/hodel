from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
import jaxtyping

from ..state import StaticState


@register_dataclass
@dataclass(frozen=True)
class Stencil:
    nat_strain: jax.Array  # shape determined by subclass

    def get_strain(self, state: StaticState) -> jax.Array: ...
    def get_energy(self, state: StaticState, Theta: jaxtyping.PyTree) -> jax.Array:
        """General energy function. To differentiate w.r.t. `q`, enclose the update like below

        ```python
        def func(q, state0, top, Theta):
            state = state.update(q, top)
            return triplet.get_energy(state, Theta)

        # Now we properly autodiff through m1, m2, and ref_twist
        E, grad_E = jax.value_grad(func, 0)(q, state0, top, Theta)
        ```

        Args:
            state (StaticState): StaticState object.
            Theta (jaxtyping.PyTree): Parameters for get_K(del_strain, Theta).

        Returns:
            jax.Array:
        """
        del_strain = self.get_strain(state) - self.nat_strain
        return self._core_energy_func(del_strain, Theta)

    def _core_energy_func(
        self, del_strain: jax.Array, Theta: jaxtyping.PyTree
    ) -> jax.Array:
        return 0.5 * del_strain.T @ self.get_K(del_strain, Theta) @ del_strain

    def get_K(self, del_strain: jax.Array, Theta: jaxtyping.PyTree) -> jax.Array: ...

    @staticmethod
    def get_stretch_strain(n0: jax.Array, n1: jax.Array, l_k: jax.Array) -> jax.Array:
        edge = n1 - n0
        edge_len = jnp.linalg.norm(edge)
        return jnp.array([edge_len / l_k - 1.0])

    @staticmethod
    def get_bend_strain(
        n0: jax.Array,
        n1: jax.Array,
        n2: jax.Array,
        m1e: jax.Array,
        m2e: jax.Array,
        m1f: jax.Array,
        m2f: jax.Array,
    ) -> jax.Array:
        ee = n1 - n0
        ef = n2 - n1
        norm_e = jnp.linalg.norm(ee)
        norm_f = jnp.linalg.norm(ef)
        te = ee / norm_e
        tf = ef / norm_f
        chi = 1.0 + jnp.sum(te * tf)
        kb = 2.0 * jnp.cross(te, tf) / chi
        kappa1 = 0.5 * jnp.sum(kb * (m2e + m2f))
        kappa2 = -0.5 * jnp.sum(kb * (m1e + m1f))
        return jnp.array([kappa1, kappa2])

    @staticmethod
    def get_twist_strain(
        theta_e: jax.Array, theta_f: jax.Array, ref_twist: jax.Array
    ) -> jax.Array:
        return theta_f - theta_e + ref_twist  # ref_twist has a dimension

    @staticmethod
    def get_hinge_strain(
        n0: jax.Array, n1: jax.Array, n2: jax.Array, n3: jax.Array
    ) -> jax.Array:
        e0 = n1 - n0
        e1 = n2 - n0
        e2 = n3 - n0

        # face normals
        u = jnp.cross(e0, e1)
        v = jnp.cross(e2, e0)
        u_norm = u / jnp.linalg.norm(u)
        v_norm = v / jnp.linalg.norm(v)

        # atan2 from sin/cos for autodiff stability
        cos_theta = jnp.dot(u_norm, v_norm)
        sin_theta = jnp.dot(e0 / jnp.linalg.norm(e0), jnp.cross(u_norm, v_norm))
        theta = jnp.atan2(sin_theta, cos_theta)
        return jnp.array([theta])
