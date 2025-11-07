from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
import jaxtyping

from .state import StaticState


@register_dataclass
@dataclass(frozen=True)
class BaseTriplet:
    """Base 3-node triplet."""

    node_dofs: jax.Array  # [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]]
    edge_dofs: jax.Array  # [e0, e1]
    edge_signs: jax.Array  # [-1/+1, -1/+1]
    l_k: jax.Array  # [l_k0, l_k1]
    ref_index: jax.Array  # [i]
    nat_strain: jax.Array  # [ε, κ₁, κ₂, τ]

    @jax.jit
    def get_strain(self, state: StaticState) -> jax.Array:
        """General energy function. To differentiate w.r.t. `q`, enclose the update like below

        ```python
        def func(q, state0, top):
            state = state.update(q, top)
            return triplet.get_strain(state)

        # Now we properly autodiff through m1, m2, and ref_twist
        # Note strain is not a scalar!
        grad_Eps = jax.jacobian(func, 0)(q, state0, top)
        ```

        Args:
            state (StaticState): StaticState object.

        Returns:
            jax.Array:
        """
        return self._static_get_strain(
            self.node_dofs,
            self.edge_dofs,
            self.edge_signs,
            self.l_k,
            self.ref_index,
            state,
        )

    @jax.jit
    def get_energy(
        self,
        state: StaticState,
        Theta: jaxtyping.PyTree = None,
    ) -> jax.Array:
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
            Theta (jaxtyping.PyTree): Parameters for get_K().

        Returns:
            jax.Array:
        """
        del_strain = self.get_strain(state) - self.nat_strain
        return self._core_energy_func(del_strain, Theta)

    @jax.jit
    def get_K(self, del_strain: jax.Array, Theta: jaxtyping.PyTree) -> jax.Array: ...

    @staticmethod
    @jax.jit
    def _static_get_strain(
        node_dofs: jax.Array,
        edge_dofs: jax.Array,
        edge_signs: jax.Array,
        l_k: jax.Array,
        ref_index: jax.Array,
        state: StaticState,
    ) -> jax.Array:
        n0, n1, n2 = state.q[node_dofs]
        m1e, m2e, m1f, m2f = BaseTriplet._get_material_directors(
            edge_dofs, edge_signs, state
        )
        theta_e, theta_f = BaseTriplet._get_thetas(edge_dofs, edge_signs, state)
        eps0 = BaseTriplet.get_stretch_strain(n0, n1, l_k[0])
        eps1 = BaseTriplet.get_stretch_strain(n1, n2, l_k[1])
        kappa = BaseTriplet.get_bend_strain(n0, n1, n2, m1e, m2e, m1f, m2f)
        tau = BaseTriplet.get_twist_strain(theta_e, theta_f, state.ref_twist[ref_index])
        return jnp.concat([eps0, eps1, kappa, tau])

    @jax.jit
    def _core_energy_func(
        self, del_strain: jax.Array, Theta: jaxtyping.PyTree
    ) -> jax.Array:
        return 0.5 * del_strain.T @ self.get_K(del_strain, Theta) @ del_strain

    @staticmethod
    @jax.jit
    def _get_material_directors(
        edge_dofs: jax.Array, edge_signs: jax.Array, state: StaticState
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sign correct m1,m2."""
        m1e = state.m1[edge_dofs[0]]
        m2e = state.m2[edge_dofs[0]] * edge_signs[0]
        m1f = state.m1[edge_dofs[1]]
        m2f = state.m2[edge_dofs[1]] * edge_signs[1]
        return m1e, m2e, m1f, m2f

    @staticmethod
    @jax.jit
    def _get_thetas(
        edge_dofs: jax.Array, edge_signs: jax.Array, state: StaticState
    ) -> tuple[jax.Array, jax.Array]:
        """Sign correct theta_e, theta_f."""
        theta_e = state.q[edge_dofs[0]] * edge_signs[0]
        theta_f = state.q[edge_dofs[1]] * edge_signs[1]
        return theta_e, theta_f

    @staticmethod
    @jax.jit
    def get_stretch_strain(n0: jax.Array, n1: jax.Array, l_k: jax.Array) -> jax.Array:
        edge = n1 - n0
        edge_len = jnp.linalg.norm(edge)
        return jnp.array([edge_len / l_k - 1.0])

    @staticmethod
    @jax.jit
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
    @jax.jit
    def get_twist_strain(
        theta_e: jax.Array, theta_f: jax.Array, ref_twist: jax.Array
    ) -> jax.Array:
        return theta_f - theta_e + ref_twist


@register_dataclass
@dataclass(frozen=True)
class DERTriplet(BaseTriplet):
    """DER with constant diagonal stiffness matrix."""

    K: jax.Array  # diagonal: [EA1, EA2, EI1, EI2, GJ]

    @classmethod
    def init(
        cls,
        node_dofs: jax.Array,
        edge_dofs: jax.Array,
        edge_signs: jax.Array,
        l_k: jax.Array,
        ref_index: jax.Array,
        EA: jax.Array,
        EI: jax.Array,
        GJ: jax.Array,
        state: StaticState,
    ) -> DERTriplet:
        diag = jnp.concat(
            [
                EA * l_k,  # l_k is [l_k0, l_k1]
                EI / jnp.mean(l_k),
                GJ / jnp.mean(l_k),
            ],
        )
        K = jnp.diag(diag)
        nat_strain = cls._static_get_strain(
            node_dofs, edge_dofs, edge_signs, l_k, ref_index, state
        )
        return cls(node_dofs, edge_dofs, edge_signs, l_k, ref_index, nat_strain, K)

    @jax.jit
    def get_K(self, del_strain: jax.Array, Theta: jaxtyping.PyTree) -> jax.Array:
        return self.K


@register_dataclass
@dataclass(frozen=True)
class ParametrizedDERTriplet(BaseTriplet):
    """DER with constant diagonal stiffness matrix where K is passed as Theta."""

    @classmethod
    def init(
        cls,
        node_dofs: jax.Array,
        edge_dofs: jax.Array,
        edge_signs: jax.Array,
        l_k: jax.Array,
        ref_index: jax.Array,
        state: StaticState,
    ) -> ParametrizedDERTriplet:
        nat_strain = cls._static_get_strain(
            node_dofs, edge_dofs, edge_signs, l_k, ref_index, state
        )
        return cls(node_dofs, edge_dofs, edge_signs, l_k, ref_index, nat_strain)

    @jax.jit
    def get_K(self, del_strain: jax.Array, Theta: jaxtyping.PyTree) -> jax.Array:
        inv_l_k = 1 / self.l_k
        v_k = 1 / jnp.mean(self.l_k)  # voronoi length
        return jnp.diag(Theta * jnp.array([inv_l_k[0], inv_l_k[1], v_k, v_k, v_k]))


@register_dataclass
@dataclass(frozen=True)
class Triplet(BaseTriplet):
    """BaseTriplet with natural strain initialization with `init(...)`."""

    @classmethod
    def init(
        cls,
        node_dofs: jax.Array,
        edge_dofs: jax.Array,
        edge_signs: jax.Array,
        l_k: jax.Array,
        ref_index: jax.Array,
        state: StaticState,
    ) -> Triplet:
        nat_strain = cls._static_get_strain(
            node_dofs, edge_dofs, edge_signs, l_k, ref_index, state
        )
        return cls(node_dofs, edge_dofs, edge_signs, l_k, ref_index, nat_strain)
