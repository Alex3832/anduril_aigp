"""MAVLink implementation of DroneInterface."""

import time
from typing import Optional

import numpy as np
from pymavlink import mavutil

from comms.drone_interface import DroneInterface


class MavlinkBackend(DroneInterface):
    """Thin adapter that maps DroneInterface calls to MAVLink messages.

    Thread safety: send_heartbeat() is safe to call from a timer thread while
    other methods run on the main thread, because pymavlink's send path uses
    its own internal lock.
    """

    def __init__(self, connection_string: str) -> None:
        self._conn_str = connection_string
        self._connection = None  # set by connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._connection = mavutil.mavlink_connection(self._conn_str)
        self._connection.wait_heartbeat()

    def send_heartbeat(self) -> None:
        """Send a client HEARTBEAT to the simulator (must be called ≥2 Hz)."""
        assert self._connection is not None, "Call connect() first"
        # Mavlink docs: Send heartbeat from a MAVLink application.
        self._connection.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,# Type of the MAV (quadrotor, helicopter, etc.) (type:uint8_t, values:MAV_TYPE)
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,      # Autopilot type / class. (type:uint8_t, values:MAV_AUTOPILOT)
            0,                                          # base_mode : System mode bitmap. (type:uint8_t, values:MAV_MODE_FLAG)
            0,                                          # custom_mode : A bitfield for use for autopilot-specific flags (type:uint32_t)
            0,                                          # system_status : System status flag. (type:uint8_t, values:MAV_STATE)
        )

    # ------------------------------------------------------------------
    # Vehicle commands (stubs — implement as needed)
    # ------------------------------------------------------------------

    def arm(self) -> None:
        raise NotImplementedError

    def disarm(self) -> None:
        raise NotImplementedError

    def takeoff(self, altitude_m: float) -> None:
        raise NotImplementedError

    def land(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Control commands
    #
    # Currently assuming PX4 like implemntation of SET_POSITION_TARGET_LOCAL_NED and SET_ATTITUDE_TARGET.
    # Waiting on spec clarification
    #
    # ------------------------------------------------------------------

    def _send_command_target(self, type_mask: int,
                              pos: np.ndarray, vel: np.ndarray,
                              acc: np.ndarray, yaw_rad: float) -> None:
        self._connection.mav.set_position_target_local_ned_send(
            int(time.time() * 1000),
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            acc[0], acc[1], acc[2],
            yaw_rad, 0,                 # yaw_rate always ignored
        )

    def set_position(self, position_ned: np.ndarray, yaw_rad: float) -> None:
        assert self._connection is not None, "Call connect() first"
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self._send_command_target(type_mask, position_ned,
                                   np.zeros(3), np.zeros(3), yaw_rad)

    def set_velocity(self, velocity_ned: np.ndarray, yaw_rad: float) -> None:
        assert self._connection is not None, "Call connect() first"
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self._send_command_target(type_mask, np.zeros(3),
                                   velocity_ned, np.zeros(3), yaw_rad)

    def set_position_velocity(self, position_ned: np.ndarray,
                              velocity_ned: np.ndarray, yaw_rad: float) -> None:
        assert self._connection is not None, "Call connect() first"
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self._send_command_target(type_mask, position_ned,
                                   velocity_ned, np.zeros(3), yaw_rad)

    def set_position_velocity_acceleration(self, position_ned: np.ndarray,
                                           velocity_ned: np.ndarray,
                                           acceleration_ned: np.ndarray,
                                           yaw_rad: float) -> None:
        assert self._connection is not None, "Call connect() first"
        type_mask = mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        self._send_command_target(type_mask, position_ned,
                                   velocity_ned, acceleration_ned, yaw_rad)

    # PLACEHOLDER: type_mask combinations for SET_ATTITUDE_TARGET are educated guesses
    # based on PX4/ArduPilot docs. Verify against simulator behaviour and update when
    # a future tech spec clarifies the supported combinations.

    def _send_attitude_target(self, type_mask: int, quaternion: np.ndarray,
                              body_rates: np.ndarray, thrust: float) -> None:
        q = quaternion / np.linalg.norm(quaternion)
        self._connection.mav.set_attitude_target_send(
            int(time.time() * 1000),
            self._connection.target_system,
            self._connection.target_component,
            type_mask,
            [q[0], q[1], q[2], q[3]],
            body_rates[0], body_rates[1], body_rates[2],
            float(np.clip(thrust, 0.0, 1.0)),
        )

    def set_attitude_thrust(self, quaternion: np.ndarray, thrust: float) -> None:
        assert self._connection is not None, "Call connect() first"
        # 0b00000111: ignore body roll/pitch/yaw rates, use attitude + thrust
        self._send_attitude_target(0b00000111, quaternion, np.zeros(3), thrust)

    def set_rates_thrust(self, body_rates: np.ndarray, thrust: float) -> None:
        assert self._connection is not None, "Call connect() first"
        # 0b10000000: ignore attitude quaternion, use body rates + thrust
        self._send_attitude_target(0b10000000, np.array([1.0, 0, 0, 0]), body_rates, thrust)

    def set_attitude_rates_thrust(self, quaternion: np.ndarray,
                                  body_rates: np.ndarray, thrust: float) -> None:
        assert self._connection is not None, "Call connect() first"
        # 0b00000000: nothing ignored, attitude + rates as feedforward + thrust
        self._send_attitude_target(0b00000000, quaternion, body_rates, thrust)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_imu(self) -> Optional[np.ndarray]:
        assert self._connection is not None, "Call connect() first"
        msg = self._connection.recv_match(type='HIGHRES_IMU', blocking=True, timeout=1.0)
        if not msg or msg.get_type() == "BAD_DATA":
            return None
        return np.array([msg.xacc, msg.yacc, msg.zacc,
                         msg.xgyro, msg.ygyro, msg.zgyro])

    def get_attitude(self) -> Optional[np.ndarray]:
        assert self._connection is not None, "Call connect() first"
        msg = self._connection.recv_match(type='ATTITUDE', blocking=True, timeout=1.0)
        if not msg or msg.get_type() == "BAD_DATA":
            return None
        return np.array([msg.roll, msg.pitch, msg.yaw,
                         msg.rollspeed, msg.pitchspeed, msg.yawspeed])
