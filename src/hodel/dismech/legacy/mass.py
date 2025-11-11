import jax
import jax.numpy as jnp

from .params import Geometry, Material


def get_mass(
    voronoi_ref_len: jnp.ndarray, ref_len: jnp.ndarray, geom: Geometry, mat: Material
) -> jax.Array:
    A = geom.axs if geom.axs else jnp.pi * geom.rod_r0**2
    n_nodes = voronoi_ref_len.shape[0]
    n_edges = ref_len.shape[0]
    mass = jnp.zeros(n_nodes * 3 + n_edges)

    # Node contributions
    if n_nodes:
        dm_nodes = voronoi_ref_len * A * mat.density
        node_dofs = jnp.arange(3 * n_nodes).reshape(-1, 3)  # x,y,z same
        mass = mass.at[node_dofs].add(dm_nodes[:, None])

    # Edge contributions (moment of inertia)
    if n_edges:
        factor = geom.jxs / geom.axs if geom.jxs and geom.axs else geom.rod_r0**2 / 2
        edge_mass = ref_len * A * mat.density * factor
        edge_dofs = 3 * n_nodes + jnp.arange(n_edges)
        mass = mass.at[edge_dofs].set(edge_mass)

    """# Shell face contributions
    if self.__n_faces:
    faces = self.__face_nodes_shell
    v1 = self.__nodes[faces[:, 1]] - self.__nodes[faces[:, 0]]
    v2 = self.__nodes[faces[:, 2]] - self.__nodes[faces[:, 1]]
    areas = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
    m_shell = material.density * areas * geom.shell_h
    dof_indices = (3 * faces[:, :, None] + np.arange(3)).reshape(-1)
    np.add.at(mass, dof_indices, np.repeat(m_shell / 3, 9))"""

    return mass
