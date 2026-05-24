"""Unit tests for the perception package."""

import numpy as np
import pytest
import cv2

from perception.gate_detector import GateDetector, GateObservation
from perception.camera import CameraIntrinsics, body_to_camera_rotation
import config


class TestGateDetector:
    def _detector(self):
        return GateDetector(
            hsv_lower=(0, 120, 70),
            hsv_upper=(10, 255, 255),
            min_area_px=100,
        )

    def test_returns_none_on_blank_frame(self):
        det = self._detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert det.detect(frame) is None

    def test_detects_red_blob(self):
        det = self._detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[200:280, 300:400] = (0, 0, 255)
        obs = det.detect(frame)
        assert obs is not None
        assert isinstance(obs, GateObservation)
        assert obs.area_px > 0

    def test_below_min_area_returns_none(self):
        det = GateDetector(hsv_lower=(0, 120, 70), hsv_upper=(10, 255, 255),
                           min_area_px=100_000)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[200:210, 300:310] = (0, 0, 255)
        assert det.detect(frame) is None


class TestCameraIntrinsics:
    def test_from_spec_matches_vadr_ts_002(self):
        intr = CameraIntrinsics.from_spec()
        assert intr.fx == 320.0
        assert intr.fy == 320.0
        assert intr.cx == 320.0
        assert intr.cy == 180.0

    def test_matrix_shape(self):
        intr = CameraIntrinsics.from_spec()
        assert intr.matrix.shape == (3, 3)

    def test_principal_point_ray_is_optical_axis(self):
        intr = CameraIntrinsics.from_spec()
        ray = intr.pixel_to_ray(intr.cx, intr.cy)
        assert np.allclose(ray, [0, 0, 1], atol=1e-6)

    def test_pixel_to_ray_unit_length(self):
        intr = CameraIntrinsics.from_spec()
        ray = intr.pixel_to_ray(100.0, 50.0)
        assert np.isclose(np.linalg.norm(ray), 1.0)


class TestBodyToCameraRotation:
    def test_is_rotation_matrix(self):
        R = body_to_camera_rotation()
        assert R.shape == (3, 3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_body_forward_maps_to_camera_forward_and_up(self):
        """Body +X (forward) should project mostly onto camera +Z (into scene)
        with a small -Y component (upward in OpenCV = negative, because camera is
        tilted 20° up so body-forward is slightly above the image centre)."""
        R = body_to_camera_rotation()
        body_forward = np.array([1.0, 0.0, 0.0])
        cam_vec = R @ body_forward
        # Z component (into scene) should be the largest
        assert cam_vec[2] > 0
        assert cam_vec[2] == pytest.approx(np.cos(np.deg2rad(config.CAMERA_TILT_UP_DEG)),
                                           abs=1e-6)
