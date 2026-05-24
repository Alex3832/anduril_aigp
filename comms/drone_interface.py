"""Abstract drone interface — defines the contract all backends must satisfy.

Supported control messages per VADR-TS-002 §4.3:
  Client → Simulator: SET_POSITION_TARGET_LOCAL_NED, SET_ATTITUDE_TARGET
  Client → Simulator: HEARTBEAT  (§5.2: client must maintain at ≥2 Hz)
  Simulator → Client: HEARTBEAT, ATTITUDE, HIGHRES_IMU, TIMESYNC
"""

import abc
from typing import Optional

import numpy as np


class DroneInterface(abc.ABC):
    """Protocol that all drone backends must satisfy.

    Concrete implementations can target SITL via MAVLink or a stub for testing.
    """

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish connection and wait for the simulator's HEARTBEAT."""

    @abc.abstractmethod
    def send_heartbeat(self) -> None:
        """Send one client HEARTBEAT to the simulator.

        Per §5.2, the client is responsible for maintaining heartbeat messages.
        Callers must invoke this at least MIN_HEARTBEAT_HZ (2 Hz) times per second.
        Recommended pattern: call from a dedicated timer thread.
        """

    # ------------------------------------------------------------------
    # Vehicle commands (stubs — implement as needed)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def arm(self) -> None:
        """Arm motors."""

    @abc.abstractmethod
    def disarm(self) -> None:
        """Disarm motors."""

    @abc.abstractmethod
    def takeoff(self, altitude_m: float) -> None:
        """Command a vertical takeoff to the specified altitude (m, positive up)."""

    @abc.abstractmethod
    def land(self) -> None:
        """Command a landing at the current position."""

    # ------------------------------------------------------------------
    # Control commands
    #
    # Currently assuming PX4 like implemntation of SET_POSITION_TARGET_LOCAL_NED and SET_ATTITUDE_TARGET.
    # Waiting on spec clarification
    #
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def set_position(self, position_ned: np.ndarray, yaw_rad: float) -> None:
        """Send SET_POSITION_TARGET_LOCAL_NED with position only.

        The autopilot's position controller determines the velocity needed to
        reach the target. Slowest to respond.

        Args:
            position_ned: Desired position (m) in NED frame, shape (3,).
            yaw_rad:      Desired heading (rad).
        """

    @abc.abstractmethod
    def set_velocity(self, velocity_ned: np.ndarray, yaw_rad: float) -> None:
        """Send SET_POSITION_TARGET_LOCAL_NED with velocity only.

        Bypasses the position controller entirely and commands the velocity
        controller directly.

        Args:
            velocity_ned: Desired velocity (m/s) in NED frame, shape (3,).
            yaw_rad:      Desired heading (rad).
        """

    @abc.abstractmethod
    def set_position_velocity(self, position_ned: np.ndarray,
                              velocity_ned: np.ndarray, yaw_rad: float) -> None:
        """Send SET_POSITION_TARGET_LOCAL_NED with position + velocity feedforward.

        The position controller runs normally; the velocity is added to its
        output as a feedforward term so the autopilot doesn't have to discover
        the expected motion from position error alone.

        Args:
            position_ned: Desired position (m) in NED frame, shape (3,).
            velocity_ned: Feedforward velocity (m/s) in NED frame, shape (3,).
            yaw_rad:      Desired heading (rad).
        """

    @abc.abstractmethod
    def set_position_velocity_acceleration(self, position_ned: np.ndarray,
                                           velocity_ned: np.ndarray,
                                           acceleration_ned: np.ndarray,
                                           yaw_rad: float) -> None:
        """Send SET_POSITION_TARGET_LOCAL_NED with position + velocity + acceleration feedforward.

        Adds acceleration feedforward on top of position+velocity control.
        Most responsive; useful when following a pre-planned trajectory where
        the velocity derivative is known.

        Args:
            position_ned:     Desired position (m) in NED frame, shape (3,).
            velocity_ned:     Feedforward velocity (m/s) in NED frame, shape (3,).
            acceleration_ned: Feedforward acceleration (m/s²) in NED frame, shape (3,).
            yaw_rad:          Desired heading (rad).
        """

    # PLACEHOLDER: type_mask combinations for SET_ATTITUDE_TARGET are educated guesses
    # based on PX4/ArduPilot docs. Verify against simulator behaviour and update when
    # a future tech spec clarifies the supported combinations.

    @abc.abstractmethod
    def set_attitude_thrust(self, quaternion: np.ndarray, thrust: float) -> None:
        """Send SET_ATTITUDE_TARGET with attitude + thrust; body rates ignored.

        The autopilot's attitude controller works to achieve the commanded
        orientation. Most commonly supported combination.

        Args:
            quaternion: Desired attitude as unit quaternion [w, x, y, z], shape (4,).
            thrust:     Normalised collective thrust in [0, 1].
        """

    @abc.abstractmethod
    def set_rates_thrust(self, body_rates: np.ndarray, thrust: float) -> None:
        """Send SET_ATTITUDE_TARGET with body rates + thrust; attitude ignored.

        Bypasses the attitude controller and commands the rate controller
        directly. Lower latency, higher risk.

        Args:
            body_rates: Desired [roll, pitch, yaw] rates (rad/s) in body frame, shape (3,).
            thrust:     Normalised collective thrust in [0, 1].
        """

    @abc.abstractmethod
    def set_attitude_rates_thrust(self, quaternion: np.ndarray,
                                  body_rates: np.ndarray, thrust: float) -> None:
        """Send SET_ATTITUDE_TARGET with attitude + body rates + thrust.

        Body rates act as feedforward into the rate controller on top of
        the attitude controller output.

        Args:
            quaternion: Desired attitude as unit quaternion [w, x, y, z], shape (4,).
            body_rates: Feedforward [roll, pitch, yaw] rates (rad/s) in body frame, shape (3,).
            thrust:     Normalised collective thrust in [0, 1].
        """

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_imu(self) -> Optional[np.ndarray]:
        """Return latest HIGHRES_IMU reading as [ax, ay, az, gx, gy, gz] or None."""

    @abc.abstractmethod
    def get_attitude(self) -> Optional[np.ndarray]:
        """Return latest ATTITUDE reading as [roll, pitch, yaw, rollspeed, pitchspeed, yawspeed] or None."""
