from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

import numpy as np

from src.data_models import State



class Field(ABC):
    """Generic Analytic field"""

    @abstractmethod
    def __call__(self, state: State) -> Any:
        """Evaluate the field at position, orientation and time described by state"""
        pass

    def __repr__(self):
        return self.__class__.__name__


### Implementations 

### Magnetic Fields

@dataclass
class MagneticEvaluation:
    value: np.ndarray
    grad: np.ndarray

class MagneticField(Field):
    """Generic Magnetic field"""

    def __call__(self, state: State) -> MagneticEvaluation:
        """
        Evaluate the field at position and time described by state
        returns a MagneticEvaluation, containing field and gradient
        """

        B_w = self.get_B_w(state)
        grad_B_w = self.get_grad_B_w(state)

        R_mat = state.rotation.as_matrix()
        R_inv = state.rotation.inv().as_matrix()
        
        return MagneticEvaluation(
            value = R_inv @ B_w ,
            grad = R_inv @ grad_B_w @ R_mat
        )

    @abstractmethod
    def get_B_w(self, state: State) -> np.ndarray:
        pass

    @abstractmethod
    def get_grad_B_w(self, state: State) -> np.ndarray:
        pass


class LinearMagneticField(MagneticField):
    """Divergence and curl free linear field"""

    def __init__(self, J: np.ndarray) -> None:
        self.J_B = J

    def get_B_w(self, state: State) -> np.ndarray:
        return self.J_B @ state.position

    def get_grad_B_w(self, state: State) -> np.ndarray:
        return self.J_B


linear_mag_field_example = LinearMagneticField(np.array([[1.0, 2.0, 3.0], [2.0, -1.0, 4.0], [3.0, 4.0, 0.0]]))



class QuadraticMagneticField(MagneticField):
    """Divergence and curl free Quadratic field"""

    def get_B_w(self, state: State) -> np.ndarray:
        x, y, z = state.position
        x = x + 4
        y = y + 4
        z = z + 4
        return np.array([3 * x**2 - 3 * y**2 - 6 * x * z, -6 * x * y, 3 * z**2 - 3 * x**2])

    def get_grad_B_w(self, state: State) -> np.ndarray:
        x, y, z = state.position
        return np.array([
            [6 * x - 6 * z, -6 * y, -6 * x],
            [-6 * y, -6 * x, 0.0],
            [-6 * x, 0.0, 6 * z],
        ])