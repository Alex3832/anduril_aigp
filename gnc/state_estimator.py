"""Drone state container and telemetry-driven state updater.

The simulator provides filtered attitude and raw IMU data via MAVLink.
Position and velocity are dead-reckoned by integrating IMU accelerometer
readings (no GPS or LOCAL_POSITION_NED available).
"""

from dataclasses import dataclass, field

import numpy as np

from utils.math_utils import euler_to_rotation_matrix

# Gravity vector in NED world frame (down = +Z). Subtracted after rotating
# body-frame accel to world frame so integration doesn't accumulate 9.81 m/s².
_GRAVITY_NED = np.array([0.0, 0.0, 9.81])


@dataclass
class DroneState:
    """Snapshot of the drone's estimated state at a single point in time."""
    # N , E, D  = X, Y, Z
    position:   np.ndarray = field(default_factory=lambda: np.zeros(3))  # NED (m),   IMU-integrated
    velocity:   np.ndarray = field(default_factory=lambda: np.zeros(3))  # NED (m/s), IMU-integrated
    attitude:   np.ndarray = field(default_factory=lambda: np.zeros(3))  # [roll, pitch, yaw] rad
    ang_vel:    np.ndarray = field(default_factory=lambda: np.zeros(3))  # [p, q, r] rad/s
    accel_body: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body frame (m/s²)
    accel_ned:  np.ndarray = field(default_factory=lambda: np.zeros(3))  # NED world frame (m/s²), gravity removed
    timestamp:  float = 0.0                                               # seconds


class StateEstimator:
    """Updates DroneState from raw MAVLink messages.

    Call update(msg) from a background thread whenever a new message arrives.
    Read state from the main thread (under a lock — see main.py).

    Handles:
        ATTITUDE    → attitude (filtered roll/pitch/yaw), ang_vel
        HIGHRES_IMU → accel, ang_vel, and integrates position/velocity/attitude
    """

    def __init__(self) -> None:
        self._state = DroneState()
        self._last_imu_time_us = None  # None until first IMU message received

    @property
    def state(self) -> DroneState:
        """Return a copy of the current state (copy avoids partial-read races)."""
        s = self._state
        return DroneState(
            position=s.position.copy(),
            velocity=s.velocity.copy(),
            attitude=s.attitude.copy(),
            ang_vel=s.ang_vel.copy(),
            accel_body=s.accel_body.copy(),
            accel_ned=s.accel_ned.copy(),
            timestamp=s.timestamp,
        )

    def update(self, msg) -> None:
        """Dispatch a MAVLink message to the appropriate handler.

        Args:
            msg: Any MAVLink message object returned by recv_match().
        """
        msg_type = msg.get_type()
        if msg_type == "ATTITUDE":
            self._handle_attitude(msg)
        elif msg_type == "HIGHRES_IMU":
            self._handle_imu(msg)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _handle_attitude(self, msg) -> None:
        # Filtered attitude from the flight controller — overrides gyro integration
        # whenever a new ATTITUDE message arrives.
        self._state.attitude  = np.array([msg.roll, msg.pitch, msg.yaw])
        self._state.ang_vel   = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
        self._state.timestamp = msg.time_boot_ms / 1000.0

    def _handle_imu(self, msg) -> None:
        accel_body = np.array([msg.xacc,  msg.yacc,  msg.zacc]) # x, y, z
        gyro       = np.array([msg.xgyro, msg.ygyro, msg.zgyro])
        time_us    = msg.time_usec # micro sec
        # If therd's a change in recorded time
        if self._last_imu_time_us is not None:
            dt = (time_us - self._last_imu_time_us) / 1e6 # Calc change in time

            # Guard against bad dt: negative (out-of-order msg)
            if 0 < dt:
                roll, pitch, yaw = self._state.attitude
                R = euler_to_rotation_matrix(roll, pitch, yaw)

                # Rotate body-frame accel to NED world frame, remove gravity
                accel_ned = R @ accel_body - _GRAVITY_NED

                # Linear kinematics
                # delta_x = v*t + 0.5*a*t^2
                self._state.position += self._state.velocity * dt + 0.5 * accel_ned * dt ** 2
                # delta_v = a*t
                self._state.velocity += accel_ned * dt

                # Angular kinematics
                # delta_theta = omega*t
                self._state.attitude += gyro * dt

                self._state.accel_ned = accel_ned

        self._last_imu_time_us  = time_us
        self._state.accel_body  = accel_body
        self._state.ang_vel     = gyro
