from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
import jaxtyping

from ..state import StaticState
from .stencil import Stencil


@register_dataclass
@dataclass(frozen=True)
class BaseTriplet(Stencil):
    """Base 3-node triplet."""

    node_dofs: jax.Array  # [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]]
    edge_dofs: jax.Array  # [θ0, θ1]
    dir_dofs: jax.Array  # [e0, e1]
    edge_signs: jax.Array  # [-1/+1, -1/+1]
    l_k: jax.Array  # [l_k0, l_k1]
    ref_index: jax.Array  # [i]

    def get_strain(self, state: StaticState) -> jax.Array:
        return self._static_get_strain(
            self.node_dofs,
            self.edge_dofs,
            self.dir_dofs,
            self.edge_signs,
            self.l_k,
            self.ref_index,
            state,
        )

    @staticmethod
    def _static_get_strain(
        node_dofs: jax.Array,
        edge_dofs: jax.Array,
        dir_dofs: jax.Array,
        edge_signs: jax.Array,
        l_k: jax.Array,
        ref_index: jax.Array,
        state: StaticState,
    ) -> jax.Array:
        n0, n1, n2 = state.q[node_dofs]
        m1e, m2e, m1f, m2f = BaseTriplet._get_material_directors(
            dir_dofs, edge_signs, state
        )
        theta_e, theta_f = BaseTriplet._get_thetas(edge_dofs, edge_signs, state)
        eps0 = BaseTriplet.get_stretch_strain(n0, n1, l_k[0])
        eps1 = BaseTriplet.get_stretch_strain(n1, n2, l_k[1])
        kappa = BaseTriplet.get_bend_strain(n0, n1, n2, m1e, m2e, m1f, m2f)
        tau = BaseTriplet.get_twist_strain(theta_e, theta_f, state.ref_twist[ref_index])
        return jnp.concat([eps0, eps1, kappa, tau])

    @staticmethod
    def _get_material_directors(
        dir_dofs: jax.Array, edge_signs: jax.Array, state: StaticState
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sign correct m1,m2."""
        m1e = state.m1[dir_dofs[0]]
        m2e = state.m2[dir_dofs[0]] * edge_signs[0]
        m1f = state.m1[dir_dofs[1]]
        m2f = state.m2[dir_dofs[1]] * edge_signs[1]
        return m1e, m2e, m1f, m2f

    @staticmethod
    def _get_thetas(
        edge_dofs: jax.Array, edge_signs: jax.Array, state: StaticState
    ) -> tuple[jax.Array, jax.Array]:
        """Sign correct theta_e, theta_f."""
        theta_e = state.q[edge_dofs[0]] * edge_signs[0]
        theta_f = state.q[edge_dofs[1]] * edge_signs[1]
        return theta_e, theta_f


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
        dir_dofs: jax.Array,
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
            node_dofs, edge_dofs, dir_dofs, edge_signs, l_k, ref_index, state
        )
        return cls(
            nat_strain=nat_strain,
            node_dofs=node_dofs,
            edge_dofs=edge_dofs,
            dir_dofs=dir_dofs,
            edge_signs=edge_signs,
            l_k=l_k,
            ref_index=ref_index,
            K=K,
        )

    def get_K(self, del_strain: jax.Array, Theta: jaxtyping.PyTree) -> jax.Array:
        return self.K


@register_dataclass
@dataclass(frozen=True)
class Triplet(BaseTriplet):
    """BaseTriplet with natural strain initialization with `init(...)`."""

    @classmethod
    def init(
        cls,
        node_dofs: jax.Array,
        edge_dofs: jax.Array,
        dir_dofs: jax.Array,
        edge_signs: jax.Array,
        l_k: jax.Array,
        ref_index: jax.Array,
        state: StaticState,
    ) -> Triplet:
        nat_strain = cls._static_get_strain(
            node_dofs, edge_dofs, dir_dofs, edge_signs, l_k, ref_index, state
        )
        return cls(
            nat_strain=nat_strain,
            node_dofs=node_dofs,
            edge_dofs=edge_dofs,
            dir_dofs=dir_dofs,
            edge_signs=edge_signs,
            l_k=l_k,
            ref_index=ref_index,
        )


@register_dataclass
@dataclass(frozen=True)
class ParametrizedDERTriplet(Triplet):
    """DER with constant diagonal stiffness matrix where [EA1, EA2, EI1, EI2, GJ] is passed as Theta."""

    def get_K(self, del_strain: jax.Array, Theta: jaxtyping.PyTree) -> jax.Array:
        l_k = self.l_k
        inv_v_k = 1 / jnp.mean(self.l_k)  # voronoi length
        return jnp.diag(Theta * jnp.array([l_k[0], l_k[1], inv_v_k, inv_v_k, inv_v_k]))
