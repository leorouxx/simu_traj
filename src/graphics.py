import gc
from itertools import cycle

import numpy as np
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.core import State

### Practicly all of the below code is vibe coded


# Util
def get_object_name_by_id(obj_id) -> str:
    if obj_id == -1:
        return "Ground Truth"
    for obj in gc.get_objects():
        if id(obj) == obj_id:
            return obj.__class__.__name__
    return ""


# Color wheel
color_cycle = cycle(px.colors.qualitative.Plotly)


def plot_trajectories_3d(
    gt_history: list[State],
    est_history: dict[int, list[State]] | None = None,
    knots_positions: np.ndarray | None = None,
    knots_rotations: list[R]| None = None,
    show_orientations: bool = True,
    step_orientation: int = 50
):
    """
    Affiche la trajectoire 3D GT, l'estimation, ainsi que les knots de position et d'orientation.

    :param gt_history: list of states of the Ground Truth 
    :param est_history: _Optional_ list of states of the Estimated State
    :param knots_positions: _Optional_ list of the positions of
    :param knots_rotations:
    :param show_orientations:
    :param step_orientation:
    """
    fig = go.Figure()

    # 1. Trajectoire Ground Truth (Ligne continue)
    gt_pos = np.array([s.position for s in gt_history])
    fig.add_trace(go.Scatter3d(
        x=gt_pos[:, 0], y=gt_pos[:, 1], z=gt_pos[:, 2],
        mode='lines',
        name='Vérité Terrain (GT)',
        line=dict(color='black', width=4)
    ))

    # 2. Polygone de contrôle (Ligne pointillée reliant les knots de position)
    if knots_positions is not None:
        fig.add_trace(go.Scatter3d(
            x=knots_positions[:, 0], y=knots_positions[:, 1], z=knots_positions[:, 2],
            mode='lines+markers+text',
            name='Polygone de contrôle (Knots)',
            line=dict(color='orange', width=1, dash='dot'),
            marker=dict(size=6, color='gold', symbol='diamond'),
            text=[f"K{i}" for i in range(len(knots_positions))],
            textposition="top center"
        ))

    # 3. Orientations des Knots (Repères RGB aux points de contrôle) <-- NOUVEAU
    if knots_positions is not None and knots_rotations is not None:
        assert len(knots_positions) == len(knots_rotations)
        scale_knots = 0.35  # Légèrement plus grand pour bien les différencier
        
        for i, (p, rot) in enumerate(zip(knots_positions, knots_rotations)):
            R_mat = rot.as_matrix()
            colors = ['darkred', 'darkgreen', 'darkblue']
            
            for j in range(3):
                axis_end = p + R_mat[:, j] * scale_knots
                fig.add_trace(go.Scatter3d(
                    x=[p[0], axis_end[0]],
                    y=[p[1], axis_end[1]],
                    z=[p[2], axis_end[2]],
                    mode='lines',
                    line=dict(color=colors[j], width=6), # Trait épais pour les knots
                    name=f'Orient. Knot K{i}',
                    showlegend=False
                ))

    # 4. Trajectoire Estimée (si fournie)
    if est_history:
        for estimator_id in est_history:
            estimator_name = get_object_name_by_id(estimator_id) 
            est_pos = np.array([s.position for s in est_history[estimator_id]])
            fig.add_trace(go.Scatter3d(
                x=est_pos[:, 0], y=est_pos[:, 1], z=est_pos[:, 2],
                mode='lines',
                name=f'Estimation ({estimator_name})',
                line=dict(color=next(color_cycle), width=3, dash='dash')
            ))

    # 5. Trièdres d'orientation sur la trajectoire GT
    if show_orientations:
        scale_gt = 0.2
        for i in range(0, len(gt_history), step_orientation):
            state = gt_history[i]
            p = state.position
            R_mat = state.rotation.as_matrix()

            colors = ['red', 'green', 'blue']
            for j in range(3):
                axis_end = p + R_mat[:, j] * scale_gt
                fig.add_trace(go.Scatter3d(
                    x=[p[0], axis_end[0]],
                    y=[p[1], axis_end[1]],
                    z=[p[2], axis_end[2]],
                    mode='lines',
                    line=dict(color=colors[j], width=2),
                    showlegend=False
                ))

    fig.update_layout(
        title="Trajectoire SE(3) : GT vs Knots (Position & Orientation)",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_sensor_error_3d(gt_history: list[State], est_history: dict[str, list[State]]):
    """
    Affiche l'erreur de position 3D sous forme de vecteurs reliant 
    chaque point estimé à sa position réelle, pour plusieurs sources d'estimation.
    """
    fig = go.Figure()

    for name, history in est_history.items():
        step = max(1, len(history) // 100)
        x_lines, y_lines, z_lines = [], [], []

        for i in range(0, min(len(gt_history), len(history)), step):
            p_gt = gt_history[i].position
            p_est = history[i].position
            
            x_lines.extend([p_gt[0], p_est[0], None])
            y_lines.extend([p_gt[1], p_est[1], None])
            z_lines.extend([p_gt[2], p_est[2], None])

        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            name=f'Écart - {name}',
            line=dict(width=2)
        ))

    fig.update_layout(
        title="Visualisation des écarts de position (Erreur 3D)",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode='data'
        )
    )
    fig.show()



# Debugging
def plot_state_comparison(gt_history, states_dict: dict[str, list[State]]):
    """
    states_dict: {id_1: [State, ...], id_2: [State, ...]}
    """
    fig = make_subplots(
        rows=3, cols=3, 
        shared_xaxes=True,
        subplot_titles=(
            "Pos X", "Pos Y", "Pos Z",
            "Vel X", "Vel Y", "Vel Z",
            "Gyro X", "Gyro Y", "Gyro Z"
        )
    )
    states_dict[-1] = gt_history

    colors = ["#2ca02c","#d62728", "#1f77b4", "#b41f91", "#ec8f1d", "#120aee"]

    for (id_, states), color in zip(states_dict.items(), colors):
        name = get_object_name_by_id(id_)
        t = [s.timestamp for s in states]
        pos = np.array([s.position for s in states])
        vel = np.array([s.velocity_world for s in states])
        omega = np.array([s.angular_velocity_body for s in states])

        for col, data in enumerate([pos, vel, omega]):
            for axis in range(3):
                row = axis + 1
                fig.add_trace(
                    go.Scatter(
                        x=t, y=data[:, axis], 
                        name=f"{name}", 
                        legendgroup=name,
                        showlegend=(row == 1 and col == 0),
                        line=dict(color=color)
                    ),
                    row=row, col=col+1
                )

    fig.update_layout(height=800, title_text="Comparaison Numérique des États")
    fig.show()