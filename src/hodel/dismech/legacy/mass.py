import jax
import jax.numpy as jnp

from .params import Geometry, Material
from .mesh import Mesh


def get_mass(
    mesh: Mesh,
    geom: Geometry,
    mat: Material,
) -> jax.Array:
    ref_len = jnp.linalg.norm(
        mesh.nodes[mesh.edges[:, 0]] - mesh.nodes[mesh.edges[:, 1]], axis=1
    )
    weights = 0.5 * ref_len
    n_nodes = mesh.nodes.shape[0]
    v_ref_len = jnp.zeros(n_nodes)
    v_ref_len = v_ref_len.at[mesh.edges[:, 0]].add(weights)
    v_ref_len = v_ref_len.at[mesh.edges[:, 1]].add(weights)

    if mesh.rod_edges.size:
        rod_ref_len = jnp.linalg.norm(
            mesh.nodes[mesh.rod_edges[:, 0]] - mesh.nodes[mesh.rod_edges[:, 1]], axis=1
        )
    else:
        rod_ref_len = jnp.empty((0, 3))

    if mesh.face_nodes.size:
        v1 = mesh.nodes[mesh.face_nodes[:, 1]] - mesh.nodes[mesh.face_nodes[:, 0]]
        v2 = mesh.nodes[mesh.face_nodes[:, 2]] - mesh.nodes[mesh.face_nodes[:, 1]]
        face_area = 0.5 * jnp.linalg.norm(jnp.cross(v1, v2), axis=1)
    else:
        face_area = jnp.empty((0, 3))

    A = geom.axs if geom.axs else jnp.pi * geom.rod_r0**2
    n_nodes = v_ref_len.shape[0]
    n_edges = rod_ref_len.shape[0]
    n_faces = face_area.shape[0]
    mass = jnp.zeros(n_nodes * 3 + n_edges)

    # Node contributions
    if n_nodes:
        dm_nodes = v_ref_len * A * mat.density
        node_dofs = jnp.arange(3 * n_nodes).reshape(-1, 3)  # x,y,z same
        mass = mass.at[node_dofs].add(dm_nodes[:, None])

    # Edge contributions (moment of inertia)
    if n_edges:
        factor = geom.jxs / geom.axs if geom.jxs and geom.axs else geom.rod_r0**2 / 2
        edge_mass = rod_ref_len * A * mat.density * factor
        edge_dofs = 3 * n_nodes + jnp.arange(n_edges)
        mass = mass.at[edge_dofs].set(edge_mass)

    # Shell face contributions
    if n_faces:
        m_shell = mat.density * face_area * geom.shell_h
        dof_indices = (3 * mesh.face_nodes[:, :, None] + jnp.arange(3)).reshape(-1)
        mass = mass.at[dof_indices].add(jnp.repeat(m_shell / 3, 9))

    return mass
