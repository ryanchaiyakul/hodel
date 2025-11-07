from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from .geometry import Geometry
from .util import map_node_to_dof


@register_dataclass
@dataclass(frozen=True)
class Connectivity:
    """Connectivity between DOFs."""

    edge_node_dofs: jax.Array  # [[x0, y0, z0], [x1, y1, z1]]
    edge_dofs: jax.Array  # [n1, n2]
    triplet_edge_dofs: jax.Array  # [e0, e1]
    triplet_signs: jax.Array  # [-1/+1, -1/+1]

    @classmethod
    def init(
        cls,
        nodes: jax.Array,
        edges: jax.Array,
        triplets: jax.Array,
        triplet_signs: jax.Array,
    ) -> Connectivity:
        n_nodes = nodes.shape[0] * 3
        return Connectivity(
            edge_node_dofs=map_node_to_dof(edges)
            if edges.size
            else jnp.empty((0, 2, 3), dtype=edges.dtype),
            edge_dofs=jnp.arange(
                n_nodes,
                n_nodes + edges.shape[0],
            ),
            triplet_edge_dofs=triplets[:, [1, 3]] + n_nodes,
            triplet_signs=triplet_signs,
        )

    @classmethod
    def from_geo(cls, geo: Geometry) -> Connectivity:
        """Backwards compatibility with PyDiSMech Geometry class.

        Args:
            geo (Geometry): PyDiSMech Geometry object.

        Returns:
            Connectivity:
        """
        return Connectivity.init(
            jnp.asarray(geo.nodes, dtype=jnp.int32),
            jnp.asarray(geo.edges, dtype=jnp.int32),
            jnp.asarray(geo.bend_twist_springs, dtype=jnp.int32),
            jnp.asarray(geo.bend_twist_signs, dtype=jnp.int32),
        )
