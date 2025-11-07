from __future__ import annotations
from typing import Self
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from .connectivity import Connectivity
from .geometry import Geometry
from .util import parallel_transport, signed_angle, rotate_axis_angle


@register_dataclass
@dataclass(frozen=True)
class StaticState:
    """Position DOFs + directors."""

    q: jax.Array
    a1: jax.Array
    a2: jax.Array
    m1: jax.Array
    m2: jax.Array
    ref_twist: jax.Array

    @classmethod
    def init(cls, q0: jax.Array, top: Connectivity) -> Self:
        """Initialize directors with q0.

        Args:
            q (jax.Array): DOFs.
            top (Connectivity): Connectivity between DOFs.

        Returns:
            Self:
        """
        t = cls.get_tangent(q0, top)
        a1, a2 = cls.get_space_parallel(t, top)
        m1, m2 = cls.get_material_directors(q0, a1, a2, top)
        ref_twist = cls.get_reference_twist(
            a1, t, jnp.zeros(top.triplet_signs.shape[0]), top
        )
        return cls(
            q=q0,
            a1=a1,
            a2=a2,
            m1=m1,
            m2=m2,
            ref_twist=ref_twist,
        )

    @classmethod
    def from_geo(cls, geo: Geometry, top: Connectivity) -> Self:
        """Backwards compatibility with PyDiSMech Geometry class.
        Generate `top` from `Connectivity.from_geo(geo)`.

        Args:
            geo (Geometry): PyDiSMech Geometry object.
            top (Connectivity): Connectivity between DOFs.

        Returns:
            Self:
        """
        q = jnp.concat(
            (
                jnp.asarray(geo.nodes, dtype=jnp.float32).flatten(),
                jnp.zeros(geo.edges.shape[0]),
            )
        )
        return cls.init(q, top)

    @jax.jit
    def update(self, q: jax.Array, top: Connectivity) -> Self:
        """Update q + directors with new q.

        Args:
            q (jax.Array): DOFs.
            top (Connectivity): Connectivity between DOFs.

        Returns:
            Self:
        """
        t0 = self.get_tangent(self.q, top)
        t = self.get_tangent(q, top)
        a1, a2 = self.get_time_parallel(t0, self.a1, t)
        m1, m2 = self.get_material_directors(q, a1, a2, top)
        ref_twist = self.get_reference_twist(a1, t, self.ref_twist, top)
        return type(self)(
            q=q,
            a1=a1,
            a2=a2,
            m1=m1,
            m2=m2,
            ref_twist=ref_twist,
        )

    @staticmethod
    @jax.jit
    def get_space_parallel(
        t: jax.Array, top: Connectivity
    ) -> tuple[jax.Array, jax.Array]:
        """Parallel transport from tangent to tangent.

        Args:
            t (jax.Array): Tangents.
            top (Connectivity): Connectivity between DOFs.

        Returns:
            tuple[jax.Array, jax.Array]: a1, a2
        """
        n_edges = top.edge_node_dofs.shape[0]
        tangent_padded = jnp.concatenate(
            [
                t,
                jnp.array([[1.0, 0.0, 0.0]]),  # temp for 0 tangent
            ],
            axis=0,
        )
        # Compute initial a1, a2
        a1_init = jnp.cross(tangent_padded[0], jnp.array([0.0, 1.0, 0.0]))
        a1_alt = jnp.cross(tangent_padded[0], jnp.array([0.0, 0.0, -1.0]))
        a1_init = jnp.where(jnp.linalg.norm(a1_init) < 1e-12, a1_alt, a1_init)
        a1_init = a1_init / jnp.maximum(jnp.linalg.norm(a1_init), 1e-12)
        a2_init = jnp.cross(tangent_padded[0], a1_init)

        def scan_func(
            carry: jax.Array, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            a1_prev = carry
            t_prev, t_curr = inputs

            a1_curr = parallel_transport(a1_prev, t_prev, t_curr)
            a1_curr = a1_curr - jnp.dot(a1_curr, t_curr) * t_curr
            a1_curr = a1_curr / jnp.maximum(jnp.linalg.norm(a1_curr), 1e-12)
            a2_curr = jnp.cross(t_curr, a1_curr)

            return a1_curr, (a1_curr, a2_curr)

        # Scan over pairs
        _, (a1_rest, a2_rest) = jax.lax.scan(scan_func, a1_init, (t[:-1], t[1:]))
        a1 = jnp.concatenate([a1_init[None], a1_rest], axis=0)[:n_edges]
        a2 = jnp.concatenate([a2_init[None], a2_rest], axis=0)[:n_edges]
        return a1, a2

    @staticmethod
    @jax.jit
    def get_time_parallel(
        t0: jax.Array, a10: jax.Array, t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Parallel transport from one time step to the next.

        Args:
            t0 (jax.Array): Prior edge tangents.
            a10 (jax.Array): Previous 1st reference director.
            t (jax.Array): Current edge tangents.

        Returns:
            tuple[jax.Array, jax.Array]: a1, a2
        """
        a1 = jax.vmap(parallel_transport)(a10, t0, t)
        a2 = jnp.cross(t, a1)
        return a1, a2

    @staticmethod
    @jax.jit
    def get_material_directors(
        q: jax.Array, a1: jax.Array, a2: jax.Array, top: Connectivity
    ) -> tuple[jax.Array, jax.Array]:
        """Rotate reference frame (a1, a2) by signed angle along the shared director.

        Args:
            q (jax.Array): DOFs.
            a1 (jax.Array): 1st reference director.
            a2 (jax.Array): 2nd reference director.
            top (Connectivity): Connectivity between DOFs.

        Returns:
            tuple[jax.Array, jax.Array]: m1, m2
        """
        theta = q[top.edge_dofs]
        cos_theta = jnp.cos(theta)[:, None]
        sin_theta = jnp.sin(theta)[:, None]
        m1 = cos_theta * a1 + sin_theta * a2
        m2 = -sin_theta * a1 + cos_theta * a2
        return m1, m2

    # TODO: Comment
    @staticmethod
    @jax.jit
    def get_reference_twist(
        a1: jax.Array, tangent: jax.Array, ref_twist: jax.Array, top: Connectivity
    ) -> jax.Array:
        def func(u: jax.Array, t: jax.Array, r: jax.Array) -> jax.Array:
            ut = parallel_transport(u[0], t[0], t[1])
            ut = rotate_axis_angle(ut, t[1], r)
            return r + signed_angle(ut, u[1], t[1])

        ts = tangent[top.triplet_edge_dofs] * top.triplet_signs[..., None]
        us = a1[top.triplet_edge_dofs]
        return jax.vmap(func)(ts, us, ref_twist)

    @staticmethod
    @jax.jit
    def get_tangent(q: jax.Array, top: Connectivity) -> jax.Array:
        """Get edge tangents."""

        def func(pos: jax.Array) -> jax.Array:
            de = pos[1] - pos[0]
            return de / jnp.maximum(jnp.linalg.norm(de), 1e-12)

        positions = q[top.edge_node_dofs]
        return jax.vmap(func)(positions)


@register_dataclass
@dataclass(frozen=True)
class DynamicState(StaticState):
    """StaticState + velocity (u) & acceleration (a)."""

    u: jax.Array
    a: jax.Array

    @classmethod
    def init(cls, q0: jax.Array, top: Connectivity) -> Self:
        """Initialize directors with q0 at rest (u=a=0).

        Args:
            q0 (jax.Array): DOFs.
            top (Connectivity): Connectivity between DOFs.

        Returns:
            Self:
        """
        t = cls.get_tangent(q0, top)
        a1, a2 = cls.get_space_parallel(t, top)
        m1, m2 = cls.get_material_directors(q0, a1, a2, top)
        ref_twist = cls.get_reference_twist(
            a1, t, jnp.zeros(top.triplet_signs.shape[0]), top
        )
        return cls(
            q=q0,
            u=jnp.zeros_like(q0),
            a=jnp.zeros_like(q0),
            a1=a1,
            a2=a2,
            m1=m1,
            m2=m2,
            ref_twist=ref_twist,
        )

    @jax.jit
    def update(
        self, q: jax.Array, u: jax.Array, a: jax.Array, top: Connectivity
    ) -> Self:
        """Update q + directors with new q, u and a.
        Calculate new u/a based on your respective integration scheme.

        Args:
            q (jax.Array): DOFs.
            u (jax.Array): velocity of DOFs.
            a (jax.Array): acceleration of DOFs.
            top (Connectivity): Connectivity between DOFs.

        Returns:
            Self:
        """
        t0 = self.get_tangent(self.q, top)
        t = self.get_tangent(q, top)
        a1, a2 = self.get_time_parallel(t0, self.a1, t, top)
        m1, m2 = self.get_material_directors(q, a1, a2, top)
        ref_twist = self.get_reference_twist(a1, t, self.ref_twist, top)
        return type(self)(
            q=q,
            u=u,
            a=a,
            a1=a1,
            a2=a2,
            m1=m1,
            m2=m2,
            ref_twist=ref_twist,
        )
