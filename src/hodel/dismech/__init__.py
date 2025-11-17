from .state import StaticState
from .connectivity import Connectivity
from .stencils import DERTriplet, ParametrizedDERTriplet, Triplet, DESHinge, Hinge
from .util import map_node_to_dof
from .animate import animate
from .legacy import (
    Mesh,
    Geometry,
    Material,
    SimParams,
    get_rod_stiffness,
    get_shell_stiffness,
    get_mass,
)

import jax
import jax.numpy as jnp


__all__ = [
    "StaticState",
    "Connectivity",
    "DESHinge",
    "DERTriplet",
    "ParametrizedDERTriplet",
    "Hinge",
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
) -> tuple[Connectivity, StaticState, jax.Array, DERTriplet | None, DESHinge | None]:
    """Get DER/DES from legacy classes.

    Args:
        mesh (Mesh): PyDiSMech Mesh object.
        geom (Geometry):  PyDiSMech Geometry object.
        mat (Material):  PyDiSMech Material object.

    Returns:
        tuple[Connectivity, StaticState, jax.Array, DERTriplet | None, DESHinge | None]
    """
    top = Connectivity.init(
        jnp.asarray(mesh.nodes, dtype=jnp.int32),
        jnp.asarray(mesh.rod_edges, dtype=jnp.int32),
        jnp.asarray(mesh.bend_twist_springs, dtype=jnp.int32),
        jnp.asarray(mesh.bend_twist_signs, dtype=jnp.int32),
        jnp.asarray(mesh.hinges, dtype=jnp.int32),
    )
    q = jnp.concat(
        (
            jnp.asarray(mesh.nodes, dtype=jnp.float32).flatten(),
            jnp.zeros(mesh.rod_edges.shape[0]),
        )
    )
    state = StaticState.init(q, top)

    mass = get_mass(mesh, geom, mat)

    triplets = hinges = None
    if mesh.bend_twist_springs.size:
        node_dofs = map_node_to_dof(
            jnp.asarray(mesh.bend_twist_springs[:, [0, 2, 4]], dtype=jnp.int32)
        )

        EA, EI1, EI2, GJ = get_rod_stiffness(geom, mat)
        n_triplets = node_dofs.shape[0]
        batch_EA = jnp.repeat(jnp.array([[EA]]), n_triplets, axis=0)
        batch_EI = jnp.repeat(jnp.array([[EI1, EI2]]), n_triplets, axis=0)
        batch_GJ = jnp.repeat(jnp.array([[GJ]]), n_triplets, axis=0)

        l0 = jnp.linalg.norm(
            state.q[node_dofs[:, 1]] - state.q[node_dofs[:, 0]], axis=1
        )
        l1 = jnp.linalg.norm(
            state.q[node_dofs[:, 2]] - state.q[node_dofs[:, 1]], axis=1
        )
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

    if mesh.hinges.size:
        l_k = jnp.stack(
            [
                jnp.linalg.norm(
                    state.q[top.hinge_dofs[:, 1]] - state.q[top.hinge_dofs[:, 0]],
                    axis=1,
                ),
                jnp.linalg.norm(
                    state.q[top.hinge_dofs[:, 2]] - state.q[top.hinge_dofs[:, 0]],
                    axis=1,
                ),
                jnp.linalg.norm(
                    state.q[top.hinge_dofs[:, 3]] - state.q[top.hinge_dofs[:, 0]],
                    axis=1,
                ),
                jnp.linalg.norm(
                    state.q[top.hinge_dofs[:, 2]] - state.q[top.hinge_dofs[:, 1]],
                    axis=1,
                ),
                jnp.linalg.norm(
                    state.q[top.hinge_dofs[:, 3]] - state.q[top.hinge_dofs[:, 1]],
                    axis=1,
                ),
            ],
            axis=1,
        )
        n_hinges = l_k.shape[0]
        ks, kb = get_shell_stiffness(geom, mat)
        ks = jnp.repeat(jnp.array([[ks]]), n_hinges, axis=0)
        kb = jnp.repeat(jnp.array([[kb]]), n_hinges, axis=0)
        hinges = jax.vmap(DESHinge.init, (0, 0, 0, 0, None))(
            top.hinge_dofs, l_k, ks, kb, state
        )
    return top, state, mass, triplets, hinges


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
        jnp.asarray(mesh.hinges, dtype=jnp.float32),
    )
    q = jnp.concat(
        (
            jnp.asarray(mesh.nodes, dtype=jnp.float32).flatten(),
            jnp.zeros(mesh.edges.shape[0]),
        )
    )
    state = StaticState.init(q, top)

    mass = get_mass(mesh, geom, mat)

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
