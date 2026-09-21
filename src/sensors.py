from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.data_models import State, SensorMeasurement, GoupedPointCloud
from src.fields import MagneticField

### ABSTRACT CLASS ###

class Sensor(ABC):
    rate: float
    time_ratio: Fraction | None

    def __init__(self, rate: float):
        self.rate = rate
        self.time_ratio = None

    def attach_simulation_dt(self, sim_dt):
        self.time_ratio = Fraction(sim_dt) * Fraction(self.rate)

    def should_sample(self, step: int) -> bool:
        """
        Given a step number (as an integer) tells wether or not to sample.
        """
        if self.time_ratio:
            current_samples = (step * self.time_ratio.numerator) // self.time_ratio.denominator
            prev_samples = ((step - 1) * self.time_ratio.numerator) // self.time_ratio.denominator
            return current_samples > prev_samples

        raise RuntimeError("no simulation was attached to the sensor")


    @abstractmethod
    def measure(self, gt: State) -> SensorMeasurement:
        """Generate a measurement from Ground Truth."""
        pass


### IMU IMPLEMS ###

@dataclass
class IMUMeasurement(SensorMeasurement):
    """IMU-specific measurement"""
    acceleration: np.ndarray
    angular_rate: np.ndarray


class IMU(Sensor):
    """Ideal inertial IMU (i.e. no noise)"""

    def __init__(self, rate: float):
        super().__init__(rate)
        self.gravity_world = np.array([0.0, 0.0, -9.81])


    def measure(self, gt: State) -> IMUMeasurement:
        acc_ideal = gt.rotation.inv().apply(gt.acceleration_world - self.gravity_world)
        gyr_ideal = gt.angular_velocity_body

        return IMUMeasurement(
            timestamp=gt.timestamp,
            sensor_id=id(self),
            acceleration=acc_ideal,
            angular_rate=gyr_ideal
        )

class RealIMU(IMU):
    """Noisy inertial IMU (i.e. no noise)"""

    def __init__(self, rate: float, accel_noise: float, gyro_noise: float):
        super().__init__(rate)
        self.accel_std = accel_noise
        self.gyro_std = gyro_noise
        self.accel_bias = np.array([0.05, -0.02, 0.01])
        self.gyro_bias = np.array([0.001, -0.002, 0.001])
        self.gravity_world = np.array([0.0, 0.0, -9.81])

    def measure(self, gt: State) -> IMUMeasurement:
        ideal_measurement = super().measure(gt)

        # accelerometer measure inner forces : R_w_b^T * (a_w - g_w)
        accel_ideal = ideal_measurement.accelerometer
        accel_noisy = accel_ideal + self.accel_bias + np.random.normal(0, self.accel_std, 3)

        # gyroscope
        gyro_ideal = ideal_measurement.accelerometer
        gyro_noisy = gyro_ideal + self.gyro_bias + np.random.normal(0, self.gyro_std, 3)

        return IMUMeasurement(
            timestamp=gt.timestamp,
            sensor_id=id(self),
            acceleration=accel_noisy,
            angular_rate=gyro_noisy
        )

### MAGNETIC IMPLEM ###

@dataclass
class MagneticMeasurement(SensorMeasurement):
    magnetic_field: np.ndarray       #(3,)     magnetic field


@dataclass
class MagneticArrayMeasurement(SensorMeasurement):
    """
    Magnetometer array measurement
    """
    r: np.ndarray                    # (N,3)     local position vectors 
    magnetic_fields: np.ndarray      # (N,3)     magnetic fields
    magnetic_grads: np.ndarray       # (N,3,3)   magnetic fields gradient

    def __post_init__(self) -> None:
        assert len(self.r) == len(self.magnetic_fields) == len(self.magnetic_grads)



class Magnetometer(Sensor):
    """Ideal Magnetometer (i.e. no noise)"""

    def __init__(self, rate: float, field: MagneticField):
        super().__init__(rate)
        self.field = field


    def measure(self, gt: State) -> MagneticMeasurement:
        return MagneticMeasurement(
            timestamp=gt.timestamp,
            sensor_id=id(self),
            magnetic_field = self.field(gt).value,
        )

class MagnetometerArray(Sensor):
    """Ideal Magnetometer Array (i.e. no noise)"""
    
    def __init__(self, rate: float, point_cloud: GoupedPointCloud , field: MagneticField):
        super().__init__(rate)
        self.field = field
        self.point_cloud = point_cloud
        self.rs = self.point_cloud.positions


    def measure(self, gt: State) -> MagneticArrayMeasurement:
        B_list = []
        grad_B_list = []
        
        for r_i in self.rs:
            pos_i_w = gt.position + gt.rotation.apply(r_i)
            omega_w = gt.rotation.apply(gt.angular_velocity_body)
            r_w = gt.rotation.apply(r_i)
            v_i_w = gt.velocity_world + np.cross(omega_w, r_w)
            
            # Custom state for this specific point
            # Note : field evaluation typically needs time, pos and rot,
            # but for the sake of rigor, this was 
            state_i = State(
                timestamp=gt.timestamp,
                position=pos_i_w,
                rotation=gt.rotation,
                velocity_world=v_i_w,
                acceleration_world=gt.acceleration_world,
                angular_velocity_body=gt.angular_velocity_body
            )
            
            # field evalution
            measurement = self.field(state_i)
            B_list.append(measurement.value)
            grad_B_list.append(measurement.grad)


        return MagneticArrayMeasurement(
            timestamp = gt.timestamp,
            sensor_id = id(self),
            r = self.rs,
            magnetic_fields = np.array(B_list),
            magnetic_grads = np.array(grad_B_list)
        )


### MIMU (or MARG) ###


@dataclass
class MIMUMeasurement(SensorMeasurement):
    acceleration: np.ndarray
    angular_rate: np.ndarray
    magnetic_field: np.ndarray                   #(3,)  magnetic field


@dataclass
class MIMUArrayMeasurement(SensorMeasurement):
    """
    Magnetometer array measurement
    """
    r: np.ndarray       # (N,3)     local position vectors 
    B: np.ndarray       # (N,3)     magnetic fields
    grad_B: np.ndarray  # (N,3,3)   magnetic fields gradient

    def __post_init__(self) -> None:
        assert len(self.r) == len(self.B)