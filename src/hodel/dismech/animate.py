import jax
import numpy as np
import plotly.graph_objects as go


from .connectivity import Connectivity


def animate(t: jax.Array, qs: jax.Array, conn: Connectivity, fix_axes: bool = True):
    def get_edge_go(q: np.ndarray, edge_node_dofs: np.ndarray) -> go.Scatter3d:
        q_nodes = q[edge_node_dofs]
        x_edges, y_edges, z_edges = [], [], []
        for e in q_nodes:
            x_edges += [e[0, 0], e[1, 0], None]
            y_edges += [e[0, 1], e[1, 1], None]
            z_edges += [e[0, 2], e[1, 2], None]
        return go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode="lines+markers",
            line=dict(color="black", width=4),
            marker=dict(size=6, color="red"),
            name="edges",
        )

    def get_triangles_go(q: np.ndarray, hinge_node_dofs: np.ndarray) -> go.Mesh3d:
        # q_nodes shape: (N, 4, 3)
        q_nodes = q[hinge_node_dofs]

        # Build triangles: (N,2,3,3)
        tris = np.stack(
            [
                q_nodes[:, [0, 1, 2], :],  # tri 1
                q_nodes[:, [0, 1, 3], :],  # tri 2
            ],
            axis=1,
        )

        # Reshape to (2N, 3, 3) then (2N*3, 3)
        tris_flat = tris.reshape(-1, 3, 3)

        # All vertices
        vertices = tris_flat.reshape(-1, 3)
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        # Define triangle connectivity: each triangle uses 3 consecutive vertices
        n_tris = tris_flat.shape[0]
        i = np.arange(0, n_tris * 3, 3)  # [0, 3, 6, 9, ...]
        j = np.arange(1, n_tris * 3, 3)  # [1, 4, 7, 10, ...]
        k = np.arange(2, n_tris * 3, 3)  # [2, 5, 8, 11, ...]

        return go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            opacity=0.75,
            name="triangles",
        )

    # Avoid cpu <-> gpu overhead
    edge_node_dofs = np.array(conn.edge_node_dofs)
    hinge_node_dofs = np.array(conn.hinge_dofs)
    q_numpy = np.array(qs)
    t_numpy = np.round(np.array(t), 3)

    def get_data(q: np.ndarray):
        if edge_node_dofs.size and hinge_node_dofs.size:
            return get_edge_go(q, edge_node_dofs), get_triangles_go(q, hinge_node_dofs)
        elif hinge_node_dofs.size:
            return get_triangles_go(q, hinge_node_dofs)
        elif edge_node_dofs.size:
            return get_edge_go(q, edge_node_dofs)
        else:
            raise ValueError("Cannot animate an empty scene!")

    frames = [
        go.Frame(
            data=get_data(q),
            name=str(t),
        )
        for q, t in zip(q_numpy, t_numpy)
    ]

    scene_config: dict[str, str | dict] = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
    }

    if fix_axes:
        padding = 0.05

        if conn.edge_node_dofs.size:
            q_nodes = q_numpy[:, : conn.edge_dofs[0]].reshape(-1, 3)
        else:
            q_nodes = q_numpy.reshape(-1, 3)

        x_vals = q_nodes[:, 0]
        y_vals = q_nodes[:, 1]
        z_vals = q_nodes[:, 2]

        x_range = np.max(x_vals) - np.min(x_vals)
        y_range = np.max(y_vals) - np.min(y_vals)
        z_range = np.max(z_vals) - np.min(z_vals)

        scene_config["xaxis"] = dict(
            range=[
                np.min(x_vals) - padding * x_range,
                np.max(x_vals) + padding * x_range,
            ],
            title="X",
        )
        scene_config["yaxis"] = dict(
            range=[
                np.min(y_vals) - padding * y_range,
                np.max(y_vals) + padding * y_range,
            ],
            title="Y",
        )
        scene_config["zaxis"] = dict(
            range=[
                np.min(z_vals) - padding * z_range,
                np.max(z_vals) + padding * z_range,
            ],
            title="Z",
        )

    layout = go.Layout(
        title="Dismech-JAX",
        scene=scene_config,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(t)],
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(t),
                        "method": "animate",
                    }
                    for t in t_numpy
                ],
                "transition": {"duration": 0},
                "x": 0.1,
                "y": 0,
                "currentvalue": {"prefix": "λ = "},
            }
        ],
    )

    fig = go.Figure(
        data=get_data(q_numpy[0]),
        frames=frames,
        layout=layout,
    )

    return fig
