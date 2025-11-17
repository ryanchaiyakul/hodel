from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from .util import map_node_to_dof


@register_dataclass
@dataclass(frozen=True)
class Connectivity:
    """Connectivity between DOFs."""

    edge_node_dofs: jax.Array  # [[[x0, y0, z0], [x1, y1, z1]], ...]
    edge_dofs: jax.Array  # [θ, ...]
    triplet_dir_dofs: jax.Array  # [[e0, e1], ...]
    triplet_edge_dofs: jax.Array  # [[θ0, θ1], ...]
    triplet_signs: jax.Array  # [[-1/+1, -1/+1], ...]
    hinge_dofs: (
        jax.Array
    )  # [[[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], ...]

    @classmethod
    def init(
        cls,
        nodes: jax.Array,
        rod_edges: jax.Array,
        triplets: jax.Array,
        triplet_signs: jax.Array,
        hinges: jax.Array,
    ) -> Connectivity:
        n_nodes = nodes.shape[0] * 3
        if rod_edges.size:
            edge_node_dofs = map_node_to_dof(rod_edges)
            edge_dofs = jnp.arange(n_nodes, n_nodes + rod_edges.shape[0])
        else:
            edge_node_dofs = jnp.empty((0, 2, 3), dtype=rod_edges.dtype)
            edge_dofs = jnp.empty(0, dtype=rod_edges.dtype)

        if triplets.size:
            triplet_dir_dofs = triplets[:, [1, 3]]
            triplet_edge_dofs = triplets[:, [1, 3]] + n_nodes
        else:
            triplet_dir_dofs = jnp.empty((0, 2), dtype=rod_edges.dtype)
            triplet_edge_dofs = jnp.empty((0, 2), dtype=rod_edges.dtype)

        if hinges.size:
            hinge_dofs = map_node_to_dof(hinges)
        else:
            hinge_dofs = jnp.empty((0, 2, 3), dtype=rod_edges.dtype)

        return Connectivity(
            edge_node_dofs=edge_node_dofs,
            edge_dofs=edge_dofs,
            triplet_dir_dofs=triplet_dir_dofs,
            triplet_edge_dofs=triplet_edge_dofs,
            triplet_signs=triplet_signs,
            hinge_dofs=hinge_dofs,
        )
