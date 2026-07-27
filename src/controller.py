import math
import time

from pymavlink import mavutil
from scipy.spatial.transform import Rotation

import config
from data_logger import LogRow, RunLogger, NAN

# --------------------------------------------------------------------------------------
# RESET COMMAND
MAVLINK_CMD_SIM_RESET = 31000

# --------------------------------------------------------------------------------------
# PID CONTROLLER
# --------------------------------------------------------------------------------------
class PID:
    """Generic PID with anti-windup integral clamping and output saturation."""
    # Initialize parameters for PID
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit if integral_limit is not None else output_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error, now=None):
        now = time.time() if now is None else now
        dt = 0.0 if self._prev_time is None else now - self._prev_time
        self._prev_time = now

        derivative = 0.0 if dt <= 0.0 else (error - self._prev_error) / dt
        self._prev_error = error

        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.output_limit is not None:
            output = max(-self.output_limit, min(self.output_limit, output))
        return output

# --------------------------------------------------------------------------------------
# MOTOR CONTROLS
# --------------------------------------------------------------------------------------

MOTOR_FRONT_LEFT = 0
MOTOR_FRONT_RIGHT = 1
MOTOR_BACK_LEFT = 0
MOTOR_BACK_RIGHT = 0

def update_motor_control(mavlink_conn, system_boot_ms):
    motor_rpms = [MOTOR_FRONT_LEFT, MOTOR_FRONT_RIGHT, MOTOR_BACK_LEFT, MOTOR_BACK_RIGHT, 0, 0, 0, 0]
    mavlink_conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        0,
        motor_rpms
    )

# --------------------------------------------------------------------------------------
# ATTITUDE CONTROLS
# --------------------------------------------------------------------------------------
ATTITUDE_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE |
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE |
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
)

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list:
    """Aerospace ZYX (yaw-pitch-roll) Euler angles [rad] -> MAVLink w,x,y,z quaternion."""
    qx, qy, qz, qw = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_quat()
    return [qw, qx, qy, qz]

def update_attitude_flight_control(mavlink_conn, system_boot_ms, q, thrust):
    now_ms = int(time.time() * 1000)

    """
    Sets a desired vehicle attitude + thrust. Used by an external controller to
    command the vehicle (manual controller or other system).

    time_boot_ms              : Timestamp (time since system boot). [ms] (type:uint32_t)
    target_system             : System ID (type:uint8_t)
    target_component          : Component ID (type:uint8_t)
    type_mask                 : Bitmap to indicate which dimensions should be ignored by the vehicle. (type:uint8_t, values:ATTITUDE_TARGET_TYPEMASK)
    q                         : Attitude quaternion (w, x, y, z order, zero-rotation is 1, 0, 0, 0) (type:float)
    body_roll_rate            : Body roll rate [rad/s] (type:float) - ignored, rate loop closed onboard
    body_pitch_rate           : Body pitch rate [rad/s] (type:float) - ignored, rate loop closed onboard
    body_yaw_rate             : Body yaw rate [rad/s] (type:float) - ignored, rate loop closed onboard
    thrust                    : Collective thrust, normalized to 0 .. 1 (-1 .. 1 for vehicles capable of reverse trust) (type:float)
    """
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        ATTITUDE_MASK,
        q,
        0.0,
        0.0,
        0.0,
        thrust
    )

# --------------------------------------------------------------------------------------
# POSITION CONTROLS
# --------------------------------------------------------------------------------------
VELOCITY_POSITION_MASK = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |

        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |

        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

def update_position_flight_control_test(mavlink_conn, system_boot_ms):
    now_ms = int(time.time() * 1000)

    """
    Sets a desired vehicle position in a local north-east-down coordinate
    frame. Used by an external controller to command the vehicle
    (manual controller or other system).

    time_boot_ms              : Timestamp (time since system boot). [ms] (type:uint32_t)
    target_system             : System ID (type:uint8_t)
    target_component          : Component ID (type:uint8_t)
    coordinate_frame          : Valid options are: MAV_FRAME_LOCAL_NED = 1, MAV_FRAME_LOCAL_OFFSET_NED = 7, MAV_FRAME_BODY_NED = 8, MAV_FRAME_BODY_OFFSET_NED = 9 (type:uint8_t, values:MAV_FRAME)
    type_mask                 : Bitmap to indicate which dimensions should be ignored by the vehicle. (type:uint16_t, values:POSITION_TARGET_TYPEMASK)
    x                         : X Position in NED frame [m] (type:float)
    y                         : Y Position in NED frame [m] (type:float)
    z                         : Z Position in NED frame (note, altitude is negative in NED) [m] (type:float)
    vx                        : X velocity in NED frame [m/s] (type:float)
    vy                        : Y velocity in NED frame [m/s] (type:float)
    vz                        : Z velocity in NED frame [m/s] (type:float)
    afx                       : X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afy                       : Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afz                       : Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    yaw                       : yaw setpoint [rad] (type:float)
    yaw_rate                  : yaw rate setpoint [rad/s] (type:float)
    """
    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        VELOCITY_POSITION_MASK,
        0.0, 0, 0.0,    # ignored position NED
        2.0, 0.0, 0.0,  # Vel - 2 m/s forward
        0.0, 0, 0.0,    # ignored acceleration
        0,              # ignored yaw
        0.0             # ignored yaw rate
    )

def update_position_flight_control(mavlink_conn, system_boot_ms, vx, vy, vz):
    now_ms = int(time.time() * 1000)

    """
    Sets a desired vehicle position in a local north-east-down coordinate
    frame. Used by an external controller to command the vehicle
    (manual controller or other system).

    time_boot_ms              : Timestamp (time since system boot). [ms] (type:uint32_t)
    target_system             : System ID (type:uint8_t)
    target_component          : Component ID (type:uint8_t)
    coordinate_frame          : Valid options are: MAV_FRAME_LOCAL_NED = 1, MAV_FRAME_LOCAL_OFFSET_NED = 7, MAV_FRAME_BODY_NED = 8, MAV_FRAME_BODY_OFFSET_NED = 9 (type:uint8_t, values:MAV_FRAME)
    type_mask                 : Bitmap to indicate which dimensions should be ignored by the vehicle. (type:uint16_t, values:POSITION_TARGET_TYPEMASK)
    x                         : X Position in NED frame [m] (type:float)
    y                         : Y Position in NED frame [m] (type:float)
    z                         : Z Position in NED frame (note, altitude is negative in NED) [m] (type:float)
    vx                        : X velocity in NED frame [m/s] (type:float)
    vy                        : Y velocity in NED frame [m/s] (type:float)
    vz                        : Z velocity in NED frame [m/s] (type:float)
    afx                       : X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afy                       : Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afz                       : Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    yaw                       : yaw setpoint [rad] (type:float)
    yaw_rate                  : yaw rate setpoint [rad/s] (type:float)
    """

    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        VELOCITY_POSITION_MASK,
        0.0, 0, 0.0,    # ignored position NED
        vx, vy, vz,  # Vel
        0.0, 0, 0.0,    # ignored acceleration
        0,              # ignored yaw
        0.0             # ignored yaw rate
    )

# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------

CONTROL_HZ = config.CONTROL_HZ

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.target_ned = None
        self.target_yaw = math.pi

        self.arm_time = None
        self.logger = None

        # Outer loop: NED position error [m] -> desired NED velocity setpoint [m/s]
        self.pos_pid_x = PID(**vars(config.POS_X))
        self.pos_pid_y = PID(**vars(config.POS_Y))
        self.pos_pid_z = PID(**vars(config.POS_Z))

        # Middle loop: NED velocity error [m/s] -> desired roll/pitch angle [rad] + thrust [0..1]
        self.vel_pid_roll   = PID(**vars(config.VEL_ROLL))
        self.vel_pid_pitch  = PID(**vars(config.VEL_PITCH))
        self.vel_pid_thrust = PID(**vars(config.VEL_THRUST))

    def update(self):
        # send automated targets to sim flight controller
        # update_attitude_flight_control(self.sim_conn, self.system_boot_ms)
        # alternatively one of
        # update_position_flight_control(self.sim_conn, self.system_boot_ms, 0.0, 0.0, 0.0)  # test constant velocity command
        # update_motor_control(self.sim_conn, self.system_boot_ms)

        # self.position_control_pid()
        self.goto_ned()

        time.sleep(1.0 / CONTROL_HZ)
# sp refers to set point
    def attitude_command(self, roll_sp, pitch_sp, yaw_sp, thrust_sp):
        """Convert the commanded roll/pitch/yaw directly into a quaternion attitude target,
        sent as an ATTITUDE_TARGET. The vehicle's onboard controller closes the rate loop."""
        q = euler_to_quaternion(roll_sp, pitch_sp, yaw_sp)
        update_attitude_flight_control(self.sim_conn, self.system_boot_ms, q, thrust_sp)

    def position_control_pid(self):
        P_gain = 0.5
        I_gain = 0.0
        D_gain = 0.0

        if 'pos_x' in self.data:
            print(f"Current position: x={self.data['pos_x']:.2f}, y={self.data['pos_y']:.2f}, z={self.data['pos_z']:.2f}", flush=True)  

        if 'active_gate_index' not in self.data or 'gates' not in self.data:
            print("No gate information received yet, skipping position control update...", flush=True)
            return

        x_error = self.data['gates'][self.data['active_gate_index']]['position_ned_x'] - self.data['pos_x']
        y_error = self.data['gates'][self.data['active_gate_index']]['position_ned_y'] - self.data['pos_y']
        z_error = self.data['gates'][self.data['active_gate_index']]['position_ned_z'] - self.data['pos_z']

        V_x = P_gain * x_error
        V_y = P_gain * y_error
        V_z = P_gain * z_error

        update_position_flight_control(self.sim_conn, self.system_boot_ms, V_x, V_y, V_z)

    # -------------------------------
    # Fly to an arbitrary NED position (cascaded PID: position -> velocity -> attitude quaternion)
    # -------------------------------
    def set_target_ned(self, x, y, z, yaw=math.pi):
        """Assign an arbitrary local NED target (metres, z negative = up) + heading [rad] to fly to."""
        self.target_ned = (x, y, z)
        self.target_yaw = yaw

    def position_pid_step(self, now):
        """Outer loop: NED position error [m] -> desired NED velocity setpoint [m/s]."""
        x_error = self.data['gates'][self.data['active_gate_index']]['position_ned_x'] - self.data['pos_x']
        y_error = self.data['gates'][self.data['active_gate_index']]['position_ned_y'] - self.data['pos_y']
        z_error = self.data['gates'][self.data['active_gate_index']]['position_ned_z'] - self.data['pos_z']

        vx_sp = self.pos_pid_x.update(x_error, now)
        vy_sp = self.pos_pid_y.update(y_error, now)
        vz_sp = self.pos_pid_z.update(z_error, now)
        return vx_sp, vy_sp, vz_sp

    def velocity_pid_step(self, vx_sp, vy_sp, vz_sp, now):
        """
        Middle loop: NED velocity error [m/s] -> desired roll/pitch angle [rad] + thrust [0..1].
        A multirotor accelerates by tilting: pitching nose-down accelerates it North,
        rolling right accelerates it East. Climbing (more negative vz) needs thrust above hover.
        """
        vx_error = vx_sp - self.data.get('vel_x', 0.0)
        vy_error = vy_sp - self.data.get('vel_y', 0.0)
        vz_error = vz_sp - self.data.get('vel_z', 0.0)

        pitch_sp = -self.vel_pid_pitch.update(vx_error, now)   # need +North accel -> nose-down (negative) pitch
        roll_sp  = -self.vel_pid_roll.update(vy_error, now)    # need +East accel  -> bank right (positive) roll
        thrust_sp = (config.HOVER_THRUST - self.vel_pid_thrust.update(vz_error, now)) # could cause gimbal lock  # +Down vel error -> reduce thrust
        # (1/ math.cos(self.data['pitch'])) * 
        thrust_sp = max(0.0, min(1.0, thrust_sp))
    
        return roll_sp, pitch_sp, thrust_sp

    def goto_ned(self, position_tolerance_m=0.25):
        """
        Cascaded PID flight to self.target_ned:
          position error -> velocity setpoint -> roll/pitch/thrust setpoint -> quaternion attitude target
        Call every control tick (e.g. from update()). Returns True once within tolerance.
        """
        if self.target_ned is None:
            return False

        if 'pos_x' not in self.data:
            print("No position estimate received yet, skipping goto_ned update...", flush=True)
            return False

        now = time.time()

        x_error = self.target_ned[0] - self.data['pos_x']
        y_error = self.target_ned[1] - self.data['pos_y']
        z_error = self.target_ned[2] - self.data['pos_z']
        distance = math.sqrt(x_error ** 2 + y_error ** 2 + z_error ** 2)

        vx_sp, vy_sp, vz_sp = self.position_pid_step(now)
        if distance <= position_tolerance_m:
            vx_sp = vy_sp = vz_sp = 0.0

        roll_sp, pitch_sp, thrust_sp = self.velocity_pid_step(vx_sp, vy_sp, vz_sp, now)
        self.attitude_command(roll_sp, pitch_sp, self.target_yaw, thrust_sp)

        if self.logger is not None:
            self._log_tick(now, vx_sp, vy_sp, vz_sp, roll_sp, pitch_sp, thrust_sp)

        return distance <= position_tolerance_m

    def _log_tick(self, now, vx_sp, vy_sp, vz_sp, roll_sp, pitch_sp, thrust_sp):
        """Log commanded (setpoint) vs. actual (measured) values for every quantity in the cascade."""
        d = self.data
        row = LogRow(
            t=now - self.arm_time,

            x_cmd=self.data['gates'][self.data['active_gate_index']]['position_ned_x'], x_act=d.get('pos_x', NAN),
            y_cmd=self.data['gates'][self.data['active_gate_index']]['position_ned_y'], y_act=d.get('pos_y', NAN),
            z_cmd=self.data['gates'][self.data['active_gate_index']]['position_ned_z'], z_act=d.get('pos_z', NAN),

            vx_cmd=vx_sp, vx_act=d.get('vel_x', NAN),
            vy_cmd=vy_sp, vy_act=d.get('vel_y', NAN),
            vz_cmd=vz_sp, vz_act=d.get('vel_z', NAN),

            roll_cmd=roll_sp, roll_act=d.get('roll', NAN),
            pitch_cmd=pitch_sp, pitch_act=d.get('pitch', NAN),
            yaw_cmd=self.target_yaw, yaw_act=d.get('yaw', NAN),
            thrust_cmd=thrust_sp,

            roll_rate_act=d.get('rollspeed', NAN),
            pitch_rate_act=d.get('pitchspeed', NAN),
            yaw_rate_act=d.get('yawspeed', NAN),
        )
        self.logger.log(row)

    # -------------------------------
    # Arm the drone
    # -------------------------------
    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,  # arm
            0, 0, 0, 0, 0, 0
        )
        self.arm_time = time.time()
        self.logger = RunLogger()
        print(f"Logging run to {self.logger.path}", flush=True)

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,  # confirmation
            0, 0, 0, 0, 0, 0, 0
        )


