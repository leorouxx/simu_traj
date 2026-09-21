from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, RotationSpline
from scipy.interpolate import CubicSpline


from src.data_models import State


class KnotList:
    timestamps: np.ndarray
    positions: np.ndarray
    rotations: list[R]


    def __init__(self, timestamps: np.ndarray, positions: np.ndarray, rotations: list[R]):
        """
        :param timestamps: 1D array of timestamps for each control point (N,)
        :param positions: control points positions (N, 3)
        :param rotations: control points orientations (N)
        """
        assert len(timestamps) == len(positions) == len(rotations) >= 2
        self.timestamps = np.asarray(timestamps)
        self.positions = np.asarray(positions)
        self.rotations = np.asarray(rotations)

    @classmethod
    def from_yaml(cls, file_path) -> KnotList:
        raise NotImplementedError()


class SplineTrajectorySE3:
    """
    Continuous trajectory on R^3 x SO(3)
    C2 smooth Spline defined by a KnotList.
    """

    def __init__(self, knotlist: KnotList):
        """
        :param knotlist: KnotList instance that describe the trajetory 
        """
        self.ts = knotlist.timestamps
        
        # cubic spline for translation
        self.pos_spline = CubicSpline(self.ts, knotlist.positions, axis=0)
        self.vel_spline = self.pos_spline.derivative(1)
        self.acc_spline = self.pos_spline.derivative(2)
        
        # cubic spline for rotation
        quats = np.array([rot.as_quat() for rot in knotlist.rotations])
        rotations_obj = R.from_quat(quats)
        self.rot_spline = RotationSpline(self.ts, rotations_obj)


    def get_state(self, t: float) -> State:
        """
        Returns the full analytical state at time t.
        """
        p_curr = self.pos_spline(t)
        v_w = self.vel_spline(t)
        a_w = self.acc_spline(t)

        R_curr: R = self.rot_spline(t)
        omega_w = self.rot_spline(t, 1)
        
        # projection in the body frame
        omega_b = R_curr.inv().apply(omega_w)

        return State(
            timestamp               = t,
            position                = p_curr,
            rotation                = R_curr,
            velocity_world          = v_w,
            acceleration_world      = a_w,
            angular_velocity_body   = omega_b
        )
