import numpy as np
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass, field

@dataclass
class State:
    """Body State at time t"""
    timestamp: float

    position: np.ndarray                # Vector3 (World)
    rotation: R                         # Rotation SO(3) (Body -> World)

    velocity_world: np.ndarray          # Vector3 (World)
    acceleration_world: np.ndarray      # Vector3 (World)
    angular_velocity_body: np.ndarray   # Vector3 (Body)

    def __post_init__(self) -> None:
        assert (
            len(self.position) 
            == len(self.velocity_world) 
            == len(self.acceleration_world) 
            == len(self.angular_velocity_body)
            == 3 
        )
        assert isinstance(self.rotation, R)
        self.timestamp = float(self.timestamp)
    
    @property
    def velocity_body(self) -> np.ndarray:
        return self.rotation.inv().apply(self.velocity_world)


@dataclass
class SensorMeasurement:
    """Generic measurement given by a sensor"""
    timestamp: float
    sensor_id: int


@dataclass
class GoupedPointCloud:
    """
    Grouped point cloud (N,3) 
    (meaning user can define groups of points as list of indices)
    """
    positions: np.ndarray
    groups: list[list[int]]
    nb_groups: int = field(init=False)
    __scaled: bool = field(init=False)

    def __post_init__(self):
        for group in self.groups:
            assert all(0 <= i < len(self.positions) for i in group)
        self.nb_groups = len(self.groups)
        self.__scaled = False
    
    def get_group(self, i: int) -> np.ndarray:
        return self.positions[self.groups[i]]

    def scale(self, coeff: int|float|np.ndarray) -> None:
        """
        scale (up or down) positions by a controlled factor.
        """
        if self.__scaled:
            print("Warning, point cloud got scaled twice (might be unwanted)")

        self.positions *= coeff
        self.__scaled = True