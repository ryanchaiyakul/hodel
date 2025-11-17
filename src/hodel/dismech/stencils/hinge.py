from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
import jaxtyping

from .stencil import Stencil
from ..state import StaticState


@register_dataclass
@dataclass(frozen=True)
class BaseHinge(Stencil):
    """4-node hinge."""

    node_dofs: jax.Array  # [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
    l_k: jax.Array  # [l_k0, l_k1, l_k2, l_k3, l_k4]

    def get_strain(self, state: StaticState) -> jax.Array:
        return self._static_get_strain(self.node_dofs, self.l_k, state)

    @staticmethod
    def _static_get_strain(
        node_dofs: jax.Array, l_k: jax.Array, state: StaticState
    ) -> jax.Array:
        """[ε_0, ε_1, ε_2, ε_3, ε_4, ψ]"""
        n0, n1, n2, n3 = state.q[node_dofs]
        eps0 = BaseHinge.get_stretch_strain(n0, n1, l_k[0])
        eps1 = BaseHinge.get_stretch_strain(n0, n2, l_k[1])
        eps2 = BaseHinge.get_stretch_strain(n0, n3, l_k[2])
        eps3 = BaseHinge.get_stretch_strain(n1, n2, l_k[3])
        eps4 = BaseHinge.get_stretch_strain(n1, n3, l_k[4])
        psi = BaseHinge.get_hinge_strain(n0, n1, n2, n3)
        return jnp.concat([eps0, eps1, eps2, eps3, eps4, psi])


@register_dataclass
@dataclass(frozen=True)
class DESHinge(BaseHinge):
    """4-node hinge."""

    K: jax.Array  # [ks_0, ks_1, ks_2, ks_3, ks_4, kb]

    @classmethod
    def init(
        cls,
        node_dofs: jax.Array,
        l_k: jax.Array,
        ks: jax.Array,
        kb: jax.Array,
        state: StaticState,
    ) -> DESHinge:
        diag = jnp.concat([ks * l_k, kb])
        K = jnp.diag(diag)
        nat_strain = cls._static_get_strain(node_dofs, l_k, state)
        return cls(
            nat_strain=nat_strain,
            node_dofs=node_dofs,
            l_k=l_k,
            K=K,
        )

    def get_K(self, del_strain: jax.Array, Theta: jaxtyping.PyTree) -> jax.Array:
        return self.K


@register_dataclass
@dataclass(frozen=True)
class Hinge(BaseHinge):
    @classmethod
    def init(
        cls,
        node_dofs: jax.Array,
        l_k: jax.Array,
        state: StaticState,
    ) -> Hinge:
        nat_strain = cls._static_get_strain(node_dofs, l_k, state)
        return cls(
            nat_strain=nat_strain,
            node_dofs=node_dofs,
            l_k=l_k,
        )
