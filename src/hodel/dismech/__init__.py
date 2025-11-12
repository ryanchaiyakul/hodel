from .state import StaticState
from .connectivity import Connectivity
from .triplet import DERTriplet, ParametrizedDERTriplet, Triplet
from .util import map_node_to_dof
from .animate import animate
from .legacy import Mesh, Geometry, Material, SimParams, get_rod_stiffness, get_mass

import jax
import jax.numpy as jnp


__all__ = [
    "StaticState",
    "Connectivity",
    "DERTriplet",
    "ParametrizedDERTriplet",
    "Triplet",
    "map_node_to_dof",
    "animate",
    "Mesh",
    "Geometry",
    "Material",
    "SimParams",
]


def from_legacy(
    mesh: Mesh, geom: Geometry, mat: Material
) -> tuple[Connectivity, StaticState, jax.Array, DERTriplet]:
    """Get DER triplet from legacy classes.

    Args:
        mesh (Mesh): PyDiSMech Mesh object.
        geom (Geometry):  PyDiSMech Geometry object.
        mat (Material):  PyDiSMech Material object.

    Returns:
        tuple[Connectivity, StaticState, jax.Array, DERTriplet]
    """
    top = Connectivity.init(
        jnp.asarray(mesh.nodes, dtype=jnp.int32),
        jnp.asarray(mesh.edges, dtype=jnp.int32),
        jnp.asarray(mesh.bend_twist_springs, dtype=jnp.int32),
        jnp.asarray(mesh.bend_twist_signs, dtype=jnp.int32),
    )
    q = jnp.concat(
        (
            jnp.asarray(mesh.nodes, dtype=jnp.float32).flatten(),
            jnp.zeros(mesh.edges.shape[0]),
        )
    )
    state = StaticState.init(q, top)

    ref_len = jnp.linalg.norm(
        mesh.nodes[mesh.edges[:, 0]] - mesh.nodes[mesh.edges[:, 1]], axis=1
    )
    weights = 0.5 * ref_len
    n_nodes = mesh.nodes.shape[0]
    v_ref_len = jnp.zeros(n_nodes)
    v_ref_len = v_ref_len.at[mesh.edges[:, 0]].add(weights)
    v_ref_len = v_ref_len.at[mesh.edges[:, 1]].add(weights)
    mass = get_mass(v_ref_len, ref_len, geom, mat)

    node_dofs = map_node_to_dof(
        jnp.asarray(mesh.bend_twist_springs[:, [0, 2, 4]], dtype=jnp.int32)
    )

    EA, EI1, EI2, GJ = get_rod_stiffness(geom, mat)
    n_triplets = node_dofs.shape[0]
    batch_EA = jnp.repeat(jnp.array([[EA]]), n_triplets, axis=0)
    batch_EI = jnp.repeat(jnp.array([[EI1, EI2]]), n_triplets, axis=0)
    batch_GJ = jnp.repeat(jnp.array([[GJ]]), n_triplets, axis=0)

    l0 = jnp.linalg.norm(state.q[node_dofs[:, 1]] - state.q[node_dofs[:, 0]], axis=1)
    l1 = jnp.linalg.norm(state.q[node_dofs[:, 2]] - state.q[node_dofs[:, 1]], axis=1)
    l_k = jnp.stack([l0, l1], axis=1)

    triplets = jax.vmap(DERTriplet.init, (0, 0, 0, 0, 0, 0, 0, 0, 0, None))(
        node_dofs,
        top.triplet_edge_dofs,
        top.triplet_dir_dofs,
        top.triplet_signs,
        l_k,
        jnp.arange(node_dofs.shape[0])[..., None],
        batch_EA,
        batch_EI,
        batch_GJ,
        state,
    )
    return top, state, mass, triplets


def from_legacy_custom(
    mesh: Mesh,
    geom: Geometry,
    mat: Material,
    cls: type[Triplet] = ParametrizedDERTriplet,
) -> tuple[Connectivity, StaticState, jax.Array, jax.Array, Triplet]:
    """Get Triplet from legacy classes (do not override init).

    Args:
        mesh (Mesh): PyDiSMech Mesh object.
        geom (Geometry):  PyDiSMech Geometry object.
        mat (Material):  PyDiSMech Material object.
        cls (type[Triplet], optional): Class to initialize from. Defaults to ParametrizedDERTriplet.

    Returns:
        tuple[Connectivity, StaticState, jax.Array, Triplet]:
    """
    top = Connectivity.init(
        jnp.asarray(mesh.nodes, dtype=jnp.int32),
        jnp.asarray(mesh.edges, dtype=jnp.int32),
        jnp.asarray(mesh.bend_twist_springs, dtype=jnp.int32),
        jnp.asarray(mesh.bend_twist_signs, dtype=jnp.int32),
    )
    q = jnp.concat(
        (
            jnp.asarray(mesh.nodes, dtype=jnp.float32).flatten(),
            jnp.zeros(mesh.edges.shape[0]),
        )
    )
    state = StaticState.init(q, top)

    ref_len = jnp.linalg.norm(
        mesh.nodes[mesh.edges[:, 0]] - mesh.nodes[mesh.edges[:, 1]], axis=1
    )
    weights = 0.5 * ref_len
    n_nodes = mesh.nodes.shape[0]
    v_ref_len = jnp.zeros(n_nodes)
    v_ref_len = v_ref_len.at[mesh.edges[:, 0]].add(weights)
    v_ref_len = v_ref_len.at[mesh.edges[:, 1]].add(weights)
    mass = get_mass(v_ref_len, ref_len, geom, mat)

    EA, EI1, EI2, GJ = get_rod_stiffness(geom, mat)
    theta = jnp.array([EA, EA, EI1, EI2, GJ])

    node_dofs = map_node_to_dof(
        jnp.asarray(mesh.bend_twist_springs[:, [0, 2, 4]], dtype=jnp.int32)
    )

    l0 = jnp.linalg.norm(state.q[node_dofs[:, 1]] - state.q[node_dofs[:, 0]], axis=1)
    l1 = jnp.linalg.norm(state.q[node_dofs[:, 2]] - state.q[node_dofs[:, 1]], axis=1)
    l_k = jnp.stack([l0, l1], axis=1)

    triplets = jax.vmap(cls.init, (0, 0, 0, 0, 0, 0, None))(
        node_dofs,
        top.triplet_edge_dofs,
        top.triplet_dir_dofs,
        top.triplet_signs,
        l_k,
        jnp.arange(node_dofs.shape[0])[..., None],
        state,
    )
    return top, state, mass, theta, triplets
