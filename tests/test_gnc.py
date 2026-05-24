"""Unit tests for the GNC package."""

import numpy as np
import pytest

from gnc.state_estimator import DroneState, StateEstimator
from gnc.guidance import GuidanceSystem, Waypoint


class TestDroneState:
    def test_default_fields_are_zero(self):
        s = DroneState()
        assert np.allclose(s.position, 0)
        assert np.allclose(s.attitude, 0)

    def test_fields_are_independent_across_instances(self):
        a = DroneState()
        b = DroneState()
        a.position[0] = 99.0
        assert b.position[0] == 0.0


class TestStateEstimator:
    def _make_attitude_msg(self, roll=0.1, pitch=0.2, yaw=0.3,
                           rr=0.0, pr=0.0, yr=0.0, t_ms=1000):
        class Msg:
            def get_type(self): return "ATTITUDE"
            time_boot_ms = t_ms
        m = Msg()
        m.roll, m.pitch, m.yaw = roll, pitch, yaw
        m.rollspeed, m.pitchspeed, m.yawspeed = rr, pr, yr
        return m

    def _make_imu_msg(self, ax=0.0, ay=0.0, az=9.81,
                      gx=0.0, gy=0.0, gz=0.0, t_us=1_000_000):
        class Msg:
            def get_type(self): return "HIGHRES_IMU"
        m = Msg()
        m.xacc,  m.yacc,  m.zacc  = ax, ay, az
        m.xgyro, m.ygyro, m.zgyro = gx, gy, gz
        m.time_usec = t_us
        return m

    def test_attitude_update(self):
        est = StateEstimator()
        est.update(self._make_attitude_msg(roll=0.1, pitch=0.2, yaw=0.3))
        assert est.state.attitude == pytest.approx([0.1, 0.2, 0.3])

    def test_imu_stores_accel_and_gyro(self):
        est = StateEstimator()
        est.update(self._make_imu_msg(ax=1.0, ay=2.0, az=9.81,
                                      gx=0.1, gy=0.2, gz=0.3))
        assert est.state.accel_body == pytest.approx([1.0, 2.0, 9.81])
        assert est.state.ang_vel    == pytest.approx([0.1, 0.2, 0.3])

    def test_imu_no_integration_on_first_message(self):
        # First message has no previous timestamp so position should stay zero
        est = StateEstimator()
        est.update(self._make_imu_msg(ax=5.0, ay=5.0, az=9.81, t_us=0))
        assert np.allclose(est.state.position, 0)
        assert np.allclose(est.state.velocity, 0)

    def test_imu_integrates_velocity_and_position(self):
        est = StateEstimator()
        # Drone level (attitude = 0), so R = identity.
        # accel = [0, 0, 9.81] → accel_ned after gravity removal = [0, 0, 0]
        # Apply a pure X acceleration instead to get a clean result.
        # az = 9.81 cancels gravity; ax = 2.0 gives net 2 m/s² north.
        est.update(self._make_imu_msg(ax=2.0, ay=0.0, az=9.81, t_us=0))
        dt_us = 500_000  # 0.5 s
        est.update(self._make_imu_msg(ax=2.0, ay=0.0, az=9.81, t_us=dt_us))
        dt = 0.5
        assert est.state.velocity[0] == pytest.approx(2.0 * dt, abs=1e-6)
        assert est.state.position[0] == pytest.approx(0.5 * 2.0 * dt ** 2, abs=1e-6)

    def test_imu_integrates_attitude_from_gyro(self):
        est = StateEstimator()
        est.update(self._make_imu_msg(gz=1.0, t_us=0))           # 1 rad/s yaw rate
        est.update(self._make_imu_msg(gz=1.0, t_us=1_000_000))   # dt = 1 s
        assert est.state.attitude[2] == pytest.approx(1.0, abs=1e-6)

    def test_imu_ignores_negative_dt(self):
        est = StateEstimator()
        est.update(self._make_imu_msg(ax=5.0, t_us=1_000_000))
        # Send a message with earlier timestamp — should be ignored
        est.update(self._make_imu_msg(ax=5.0, t_us=500_000))
        assert np.allclose(est.state.velocity, 0)

    def test_state_returns_copy(self):
        est = StateEstimator()
        est.update(self._make_attitude_msg())
        s1 = est.state
        s1.attitude[0] = 999.0
        s2 = est.state
        assert s2.attitude[0] != 999.0

    def test_unknown_message_type_ignored(self):
        est = StateEstimator()
        class UnknownMsg:
            def get_type(self): return "SOMETHING_ELSE"
        est.update(UnknownMsg())
        assert np.allclose(est.state.attitude, 0)

    def test_timestamp_updated_from_attitude(self):
        est = StateEstimator()
        est.update(self._make_attitude_msg(t_ms=5000))
        assert est.state.timestamp == pytest.approx(5.0)


class TestGuidanceSystem:
    def _waypoints(self):
        return [
            Waypoint(position=np.array([1.0, 0.0, -1.0])),
            Waypoint(position=np.array([2.0, 0.0, -1.0])),
            Waypoint(position=np.array([3.0, 0.0, -1.0])),
        ]

    def test_starts_at_first_waypoint(self):
        g = GuidanceSystem(self._waypoints())
        assert g.current_waypoint.position[0] == pytest.approx(1.0)

    def test_advance_moves_to_next(self):
        g = GuidanceSystem(self._waypoints())
        g.advance()
        assert g.current_waypoint.position[0] == pytest.approx(2.0)

    def test_complete_after_all_advanced(self):
        wps = self._waypoints()
        g = GuidanceSystem(wps)
        for _ in wps:
            g.advance()
        assert g.complete
        assert g.current_waypoint is None

    def test_remaining_count(self):
        g = GuidanceSystem(self._waypoints())
        assert g.remaining() == 3
        g.advance()
        assert g.remaining() == 2
