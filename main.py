
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.geometry import KnotList, SplineTrajectorySE3
from src.core import Simulator
import src.sensors as sens
import src.estimators as est
import src.fields as field
from src.data_models import GoupedPointCloud
from src.graphics import plot_trajectories_3d, plot_sensor_error_3d, plot_state_comparison


SIM_DT = 0.001 # s


if __name__ == "__main__":

    ### Trajectory definition

    ### Complex trajectory
    knots = KnotList(
        timestamps = np.linspace(0, 10, 8),
        positions= np.array([
            [0, 0, 0], [1, 2, 0.5], [3, 2, 1], [5, 0, 2],
            [6, -2, 1], [6, -1, 0.5], [8, 1, 0], [10, 0, 0]
        ]),
        rotations=[R.from_euler('xyz', [0.1 * i, 0.05 * i, 0.2 * i]) for i in range(8)]
    )

    ### Straight line
    # knots = KnotList(
    #     timestamps=np.linspace(0, 30, 8),
    #     positions=np.linspace([0, 0, 0], [10, 0, 0], 8),
    #     rotations=[R.identity() for _ in range(8)]
    # )
    spline_gt = SplineTrajectorySE3(knots)


    ### Sensor(s) definitions

    sim = Simulator(spline_gt, dt=SIM_DT)

    ideal_imu = sens.IMU(rate=50.0)
    sim.add_sensor(ideal_imu)

    point_cloud = GoupedPointCloud(
        positions = np.array(
            [
            [  0.0,   0.0,   0.0],
            [0.002,   0.0,   0.0],
            [  0.0, 0.002,   0.0],
            [0.002, 0.002,   0.0], 
            [  0.0,   0.0, 0.002],
            [0.002,   0.0, 0.002],
            [  0.0, 0.002, 0.002],
            [0.002, 0.002, 0.002]
            ]
        ),
        groups = [
            [0, 1, 2, 4],
            [1, 0, 3, 5],
            [2, 0, 3, 6],
            [3, 1, 2, 7],
            [4, 0, 5, 6],
            [5, 4, 1, 7],
            [6, 4, 7, 2],
            [7, 5, 3, 6], 
        ]
    )

    ideal_magnetometer = sens.MagnetometerArray(
        rate = 50.0,
        point_cloud = point_cloud,
        field = field.linear_mag_field_example
    )
    sim.add_sensor(ideal_magnetometer)

    ### Estimator(s) definition

    estimators = [
        est.DeadReckoningIMUEstimator(ideal_imu),
        est.MagneticKinematicEstimator(ideal_magnetometer, interp_type="first"),
        # est.MagneticKinematicEstimator(ideal_magnetometer, interp_type="baryc")
    ]

    sim.add_estimator(estimators)

    ### Simulation
    print("Exécution de la simulation :")
    t_start = knots.timestamps[0]
    t_end   = knots.timestamps[-1]
    gt_hist, est_hist = sim.run(t_start, t_end)

    print(
        f"Simulation terminée.",
        f"Durée de la simu          : {t_end - t_start:.2f} s",
        f"Nombre de pas de la simu  : {len(gt_hist)}",
        sep="\n\t"
    )

    for estimator in estimators:
        print(f"Position finale estimée ({estimator.__class__.__name__}): {est_hist[id(estimator)][-1].position}") if est_hist else 0

    # Visaluazition
    plot_trajectories_3d(
        gt_hist, 
        est_hist, 
        knots_positions=knots.positions,
        knots_rotations=knots.rotations,
        show_orientations=True, 
        step_orientation=int(1/(SIM_DT))
    )

    plot_sensor_error_3d(gt_hist, est_hist)
