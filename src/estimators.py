from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.data_models import SensorMeasurement, State, GoupedPointCloud
import src.sensors as sens
import src.math_utils as m

class Estimator(ABC):
    
    def initialize(self, initial_state: State):
        """(optional if stateless) Initialize the state before simulation"""
        pass

    def predict(self, dt: float) -> None:
        """(optional) predict the state after an elapsed time dt since last prediction"""
        pass

    @abstractmethod
    def update(self, measurement: SensorMeasurement) -> None:
        """updates the estimated step from sensor measurement"""
        pass

    @abstractmethod
    def get_estimated_state(self) -> State | None:
        pass


class DeadReckoningIMUEstimator(Estimator):
    """Simple Integration Estimator"""

    def __init__(self, sensor: sens.IMU):
        if not isinstance(sensor, sens.IMU):
            raise TypeError("IMU estimator requires IMU sensor data")
        
        self.target_sensor_id = id(sensor)

        self.current_state: State | None = None
        self.last_imu_time: float | None = None
        self.gravity = np.array([0.0, 0.0, -9.81])

    def initialize(self, initial_state: State):
        self.current_state = initial_state
        self.last_imu_time = initial_state.timestamp

    def update(self, measurement: sens.IMUMeasurement) -> None:
        if (
            measurement.sensor_id != self.target_sensor_id or 
            self.current_state is None
        ): return

        t = measurement.timestamp
        dt = t - self.last_imu_time
        if dt <= 0:
            return

        acc_b = measurement.acceleration
        gyr_b = measurement.angular_rate

        # Intégration orientation
        rot_step = R.from_rotvec(gyr_b * dt)
        rot_half = R.from_rotvec(gyr_b * 0.5 * dt)

        R_curr = self.current_state.rotation
        R_next = R_curr * rot_step
        R_half = R_curr * rot_half

        # Intégration accélération
        a_w = R_half.apply(acc_b) + self.gravity
        v_w_curr = self.current_state.velocity_world
        v_w_next = self.current_state.velocity_world + a_w * dt

        p_next = self.current_state.position + 0.5 * (v_w_curr + v_w_next) * dt

        self.current_state = State(
            timestamp=t,
            position=p_next,
            rotation=R_next,
            velocity_world=v_w_next,
            acceleration_world=a_w,
            angular_velocity_body=gyr_b
        )
        self.last_imu_time = t

    def get_estimated_state(self) -> State | None:
        return self.current_state


class MagneticKinematicEstimator(Estimator):
    """
    Magnetic Gradient based Estimator.
    Resolve the least squares problem
    """

    def __init__(self, sensor: sens.MagnetometerArray, interp_type: str = "first"):
        if not isinstance(sensor, sens.MagnetometerArray):
            raise TypeError("MagneticKinematicEstimator requires MagnetometerArray sensor data")
        
        self.target_sensor_id = id(sensor)
        self.point_cloud = sensor.point_cloud

        if interp_type == "first":
            self.rs = np.array([self.point_cloud.positions[group[0]] for group in self.point_cloud.groups])
        elif interp_type == "baryc":
            self.rs = np.array([np.mean(self.point_cloud.get_group(i), axis=0) for i in range(self.point_cloud.nb_groups)])
        else:
            raise ValueError("interp_type should be either first or baryc")

        self.current_state: State | None = None
        self.last_time: float | None = None
        self.last_B = np.empty(shape= (self.point_cloud.nb_groups, 3))

    def initialize(self, initial_state: State):
        self.current_state = initial_state

    def update(self, measurement: sens.MagneticArrayMeasurement) -> None:
        if (
            measurement.sensor_id != self.target_sensor_id or 
            self.current_state is None
        ): return

        t = measurement.timestamp

        A_blocks = []
        b_blocks = []

        curr_B = np.empty_like(self.last_B)
        curr_grad_B = np.empty(shape = (len(curr_B), 3, 3))


        if self.last_time is None:
            for i, group in enumerate(self.point_cloud.groups):
                positions = self.point_cloud.get_group(i)
                raw_measurements = measurement.magnetic_fields[group]
                est_point = self.rs[i]
    
                curr_B[i], _ = m.compute_gradient(est_point, positions, raw_measurements)

            self.last_time = t
            self.last_B = curr_B

            return

        
        dt = t - self.last_time
        assert dt > 0

        for i, group in enumerate(self.point_cloud.groups):
            positions = self.point_cloud.get_group(i)
            raw_measurements = measurement.magnetic_fields[group]
            est_point = self.rs[i]

            curr_B[i], curr_grad_B[i] = m.compute_gradient(est_point, positions, raw_measurements)

            B_dot = (curr_B[i] - self.last_B[i])/dt

            rot_term = m.skew(curr_B[i]) - curr_grad_B[i] @ m.skew(est_point)
            M_i = np.hstack((curr_grad_B[i], rot_term))
            
            A_blocks.append(M_i)
            b_blocks.append(B_dot)

        A = np.vstack(A_blocks)         # (3*n, 6)
        b = np.concatenate(b_blocks)    # (3*n, )
 
        # least square solving: Y = [v_c, omega]^T
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms < 1e-12] = 1.0

        A_scaled = A / col_norms

        Y_scaled, _, _, _ = np.linalg.lstsq(A_scaled, b, rcond=None)

        Y = Y_scaled / col_norms
        v     = Y[0:3]
        omega = Y[3:6]

        # Kinematic Integration (Dead Reckoning)
        # As of know, speed and omega are expressed in the body frame
        R_curr = self.current_state.rotation
        R_next = R_curr * R.from_rotvec(omega * dt)
        
        # Trapezoid integration
        v_world_curr = R_curr.apply(v)
        v_world_next = R_next.apply(v)
        p_next = self.current_state.position + 0.5 * (v_world_curr + v_world_next) * dt

        self.current_state = State(
            timestamp=t,
            position=p_next,
            rotation=R_next,
            velocity_world=v_world_curr,
            acceleration_world=np.zeros(3),
            angular_velocity_body=omega
        )
        self.last_time = t
        self.last_B = curr_B

    def get_estimated_state(self) -> State | None:
        return self.current_state
