from .state import StaticState, DynamicState
from .connectivity import Connectivity
from .triplet import DERTriplet, ParametrizedDERTriplet, Triplet
from .util import map_node_to_dof
from .animate import animate
from .legacy import Mesh, Geometry, Material, SimParams, get_rod_stiffness, get_mass

import jax
import jax.numpy as jnp


def from_legacy(mesh: Mesh, geom: Geometry, mat: Material, sim_params: SimParams):
    n_nodes = mesh.nodes.shape[0]

    # Connectivity
    top = Connectivity.init(
        jnp.asarray(mesh.nodes, dtype=jnp.int32),
        jnp.asarray(mesh.edges, dtype=jnp.int32),
        jnp.asarray(mesh.bend_twist_springs, dtype=jnp.int32),
        jnp.asarray(mesh.bend_twist_signs, dtype=jnp.int32),
    )

    # State
    q = jnp.concat(
        (
            jnp.asarray(mesh.nodes, dtype=jnp.float32).flatten(),
            jnp.zeros(mesh.edges.shape[0]),
        )
    )
    state = StaticState.init(q, top)

    # Triplet
    node_dofs = map_node_to_dof(
        jnp.asarray(mesh.bend_twist_springs[:, [0, 2, 4]], dtype=jnp.int32)
    )

    # Stiffness
    EA, EI1, EI2, GJ = get_rod_stiffness(geom, mat)
    n_triplets = node_dofs.shape[0]
    EA = jnp.repeat(jnp.array([[EA]]), n_triplets, axis=0)
    EI = jnp.repeat(jnp.array([[EI1, EI2]]), n_triplets, axis=0)
    GJ = jnp.repeat(jnp.array([[GJ]]), n_triplets, axis=0)

    # natural length
    l0 = jnp.linalg.norm(state.q[node_dofs[:, 1]] - state.q[node_dofs[:, 0]], axis=1)
    l1 = jnp.linalg.norm(state.q[node_dofs[:, 2]] - state.q[node_dofs[:, 1]], axis=1)
    l_k = jnp.stack([l0, l1], axis=1)

    triplets = jax.vmap(DERTriplet.init, (0, 0, 0, 0, 0, 0, 0, 0, 0, None))(
        node_dofs,
        top.triplet_edge_dofs,
        top.triplet_dir_dofs,
        top.triplet_signs,
        l_k,
        jnp.arange(node_dofs.shape[0])[..., None],  # FIXME: not always be in order
        EA,
        EI,
        GJ,
        state,
    )
    return top, state, triplets
