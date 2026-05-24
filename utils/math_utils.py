"""Shared rotation and vector math."""

import numpy as np
from scipy.spatial.transform import Rotation


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 3×3 body-to-world rotation matrix from ZYX Euler angles.

    Args:
        roll, pitch, yaw: Angles in radians.

    Returns:
        R (3×3) such that v_world = R @ v_body.
    """
    return Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [w, x, y, z] to ZYX Euler angles [roll, pitch, yaw].

    Args:
        q: Shape (4,), must be unit-normalised.

    Returns:
        Shape (3,) array [roll, pitch, yaw] in radians.
    """
    # scipy uses scalar-last convention [x, y, z, w]
    w, x, y, z = q / np.linalg.norm(q)
    angles = Rotation.from_quat([x, y, z, w]).as_euler('ZYX')
    return np.array([angles[2], angles[1], angles[0]])  # reorder to [roll, pitch, yaw]


def normalize_angle(angle: float) -> float:
    """Wrap angle to (−π, π].

    Args:
        angle: Input angle in radians.

    Returns:
        Wrapped angle in (−π, π].
    """
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix of vector v.

    Args:
        v: Shape (3,).

    Returns:
        [v]× such that [v]× @ u == np.cross(v, u).
    """
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])
