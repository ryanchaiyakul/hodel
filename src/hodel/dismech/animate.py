import jax
import numpy as np
import plotly.graph_objects as go


from .connectivity import Connectivity


def animate(t: jax.Array, qs: jax.Array, conn: Connectivity, fix_axes: bool = True):
    def get_edge_go(q: np.ndarray, edge_node_dofs: np.ndarray) -> go.Scatter3d:
        q0_edge_nodes = q[edge_node_dofs]
        x_edges, y_edges, z_edges = [], [], []
        for e in q0_edge_nodes:
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

    # Avoid cpu <-> gpu overhead
    edge_node_dofs = np.array(conn.edge_node_dofs)
    q_numpy = np.array(qs)
    t_numpy = np.round(np.array(t), 3)

    frames = [
        go.Frame(data=get_edge_go(q, edge_node_dofs), name=str(t))
        for q, t in zip(q_numpy, t_numpy)
    ]

    scene_config: dict[str, str | dict] = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
    }

    if fix_axes:
        padding = 0.05

        q_nodes = q_numpy[:, edge_node_dofs]
        x_vals = q_nodes[:, :, :, 0]
        y_vals = q_nodes[:, :, :, 1]
        z_vals = q_nodes[:, :, :, 2]

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
                "currentvalue": {"prefix": "t = "},
            }
        ],
    )

    fig = go.Figure(
        data=get_edge_go(q_numpy[0], edge_node_dofs),
        frames=frames,
        layout=layout,
    )

    return fig