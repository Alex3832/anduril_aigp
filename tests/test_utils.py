"""Unit tests for shared math utilities."""

import numpy as np
import pytest

from utils.math_utils import (
    euler_to_rotation_matrix,
    quaternion_to_euler,
    normalize_angle,
    skew_symmetric,
)


class TestEulerToRotationMatrix:
    def test_identity_at_zero_angles(self):
        R = euler_to_rotation_matrix(0, 0, 0)
        assert np.allclose(R, np.eye(3))

    def test_orthogonality(self):
        R = euler_to_rotation_matrix(0.1, 0.2, 0.3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        R = euler_to_rotation_matrix(0.5, -0.3, 1.2)
        assert np.isclose(np.linalg.det(R), 1.0)


class TestQuaternionToEuler:
    def test_identity_quaternion(self):
        euler = quaternion_to_euler(np.array([1.0, 0.0, 0.0, 0.0]))
        assert np.allclose(euler, [0, 0, 0], atol=1e-10)

    def test_round_trip(self):
        """Convert Euler → R → q → Euler and check consistency."""
        roll, pitch, yaw = 0.2, -0.1, 1.0
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        # Build quaternion from rotation matrix
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        w = 0.5 * np.sqrt(max(0.0, 1 + trace))
        x = (R[2, 1] - R[1, 2]) / (4 * w) if w > 1e-6 else 0.0
        y = (R[0, 2] - R[2, 0]) / (4 * w) if w > 1e-6 else 0.0
        z = (R[1, 0] - R[0, 1]) / (4 * w) if w > 1e-6 else 0.0
        q = np.array([w, x, y, z])
        euler = quaternion_to_euler(q)
        assert np.allclose(euler, [roll, pitch, yaw], atol=1e-6)


class TestNormalizeAngle:
    @pytest.mark.parametrize("angle,expected", [
        (0.0, 0.0),
        (np.pi, -np.pi),
        (3 * np.pi, -np.pi),
        (-np.pi / 2, -np.pi / 2),
    ])
    def test_wraps_correctly(self, angle, expected):
        assert normalize_angle(angle) == pytest.approx(expected, abs=1e-10)


class TestSkewSymmetric:
    def test_cross_product_equivalence(self):
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([4.0, 5.0, 6.0])
        assert np.allclose(skew_symmetric(v) @ u, np.cross(v, u))

    def test_antisymmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        S = skew_symmetric(v)
        assert np.allclose(S, -S.T)
