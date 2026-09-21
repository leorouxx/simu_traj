import numpy as np

def skew(v: np.ndarray) -> np.ndarray:
    """Get the skew-symmetric matrix associated to v"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def get_symm_traceless(M: np.ndarray) -> np.ndarray:
    """
    Given a 3x3 real matrix
    returns 1/2 (A^T + A) - I tr(A)/3
    """
    if M.shape != (3,3): raise ValueError("Matrix should be 3x3")
    A_symm = (M + M.T)/2
    corr = np.identity(3, dtype=np.float32) * M.trace()/3
    return A_symm - corr


def compute_gradient(
        target_location: np.ndarray, 
        measure_locations: np.ndarray, 
        measures: np.ndarray,
        normalize: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    given k measure, we linearly approximate the value and the gradient at the target location X_0:

    $$
    B_i approx= B(X_0) + gradB(X_0) . r_i 
    $$

    locations should be expressed in the same referential, with the same origin.

    :param target_location: (3,) point (X_0) where to compute the gradient 
    :param measure_locations: (k,3) points where the samples are acquired 
    :param measures: (k,3) measures values to interpolate

    :return (B, grad_B): a tuple (3,) and (3,3) containing the estimations of value and the gradient.
    """

    n = len(measure_locations)
    r_is = measure_locations - target_location

    A = np.zeros((3 * n, 12))
    b = measures.flatten()

    d_norm = np.linalg.norm(r_is)
    if (np.isclose(d_norm, 0)) or (not normalize):
        d_norm = 1

    for i in range(n):
        dr = r_is[i]/d_norm
        
        A[3*i : 3*(i+1), 0:3] = np.eye(3)
        
        A[3*i,     3:6 ]  = dr  # Ligne Bx
        A[3*i + 1, 6:9 ]  = dr  # Ligne By
        A[3*i + 2, 9:12]  = dr  # Ligne Bz

    X_est, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    B_est = X_est[0:3]
    grad_B_est = X_est[3:12].reshape((3, 3))/d_norm

    return B_est, grad_B_est


def compute_lpvf_gradient(
        target_location: np.ndarray, 
        measure_locations: np.ndarray, 
        measures: np.ndarray,
        normalize: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    given k measure, we linearly approximate the value and the gradient at the target location X_0:

    $$
    B_i approx= B(X_0) + gradB(X_0) . r_i 
    $$

    locations should be expressed in the same referential, with the same origin.
    In addition, the measured filed is assumed to be divergence and curl free (i.e a Laplacian Vector Field)

    :param target_location: (3,) point (X_0) where to compute the gradient 
    :param measure_locations: (k,3) points where the samples are acquired 
    :param measures: (k,3) measures values to interpolate

    :return (B, grad_B): a tuple (3,) and (3,3) containing the estimations of value and the gradient.
    """

    n = len(measure_locations)
    r_is = measure_locations - target_location

    A = np.zeros((3 * n, 8))
    b = measures.flatten()

    d_norm = np.mean(np.abs(r_is))
    if (np.isclose(d_norm, 0)) or (not normalize):
        d_norm = 1

    for i in range(n):
        dr = r_is[i]/d_norm
        
        A[3*i : 3*(i+1), 0:3] = np.eye(3)
            
        A[3*i,     3:8] = ( dr[0], dr[1], dr[2],     0,     0)
        A[3*i + 1, 3:8] = (     0, dr[0],     0, dr[1], dr[2])
        A[3*i + 2, 3:8] = (-dr[2],     0, dr[0],-dr[2], dr[1])

    X_est, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    B_est = X_est[0:3]
    gb_el = X_est[3:8]/d_norm

    grad_B_est = np.array([
        [gb_el[0], gb_el[1], gb_el[2]],
        [gb_el[1], gb_el[3], gb_el[4]],
        [gb_el[2], gb_el[4], -gb_el[0] - gb_el[3]]
    ])

    return B_est, grad_B_est

