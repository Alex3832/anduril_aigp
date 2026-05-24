"""Single source of truth for all tunable parameters and PID gains.

Constants marked [SPEC §X.Y] are fixed by VADR-TS-002 and must not be changed
without a corresponding spec update. All others are tunable.
"""

from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


# ---------------------------------------------------------------------------
# Spec-fixed constants (VADR-TS-002)
# ---------------------------------------------------------------------------

# Camera intrinsics — pinhole model, zero distortion  [SPEC §3.8]
CAMERA_RESOLUTION: tuple = (640, 360)   # px (width, height)
CAMERA_FX: float = 320.0               # px
CAMERA_FY: float = 320.0               # px
CAMERA_CX: float = 320.0               # px
CAMERA_CY: float = 180.0               # px
CAMERA_TILT_UP_DEG: float = 20.0       # camera pitched upward from body forward axis

# Vision stream transport  [SPEC §4.6]
VISION_UDP_PORT: int = 5600
VISION_FPS: int = 30                   # Hz

# Gate geometry  [SPEC §3.7]
GATE_OUTER_M: float = 2.700            # outer square side (m)
GATE_INNER_M: float = 1.500            # passable inner square side (m)
GATE_DEPTH_M: float = 0.260            # gate thickness (m)

# Drone chassis  [SPEC §3.6]
DRONE_WIDTH_M:  float = 0.280
DRONE_LENGTH_M: float = 0.280
DRONE_HEIGHT_M: float = 0.160

# Timing constraints  [SPEC §4.4]
PHYSICS_HZ: int = 120                  # simulator physics update rate
MAX_COMMAND_HZ: int = 100              # client must not exceed this
MIN_HEARTBEAT_HZ: int = 2              # client must send heartbeat at least this often

# Qualification run limit  [SPEC §8.3]
MAX_RUN_DURATION_S: float = 480.0      # 8 minutes

# MAVLink connection
MAVLINK_CONNECTION: str = "udpin:localhost:14540" # (or udp:localhost:14540, 127.0.0.1:14540,etc.) ; MAVLink API listening for SITL connection via UDP (see https://mavlink.io/en/mavgen_python/)

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# --- Position control (outer loop) ---
POS_X = PIDGains(kp=1.2, ki=0.0, kd=0.4)
POS_Y = PIDGains(kp=1.2, ki=0.0, kd=0.4)
POS_Z = PIDGains(kp=1.5, ki=0.05, kd=0.3)

# --- Velocity control (inner loop) ---
VEL_X = PIDGains(kp=2.0, ki=0.1, kd=0.1)
VEL_Y = PIDGains(kp=2.0, ki=0.1, kd=0.1)
VEL_Z = PIDGains(kp=2.5, ki=0.2, kd=0.05)

# --- Attitude control ---
ROLL  = PIDGains(kp=6.0, ki=0.0, kd=0.3)
PITCH = PIDGains(kp=6.0, ki=0.0, kd=0.3)
YAW   = PIDGains(kp=4.0, ki=0.0, kd=0.2)

# --- State estimator noise ---
IMU_ACCEL_NOISE_STD: float = 0.05    # m/s²
IMU_GYRO_NOISE_STD:  float = 0.002   # rad/s
CAMERA_POSE_NOISE_STD: float = 0.02  # m

# --- Gate detection (tune from sim imagery) ---
GATE_HSV_LOWER = (0, 120, 70)        # HSV lower bound for gate colour
GATE_HSV_UPPER = (10, 255, 255)      # HSV upper bound
MIN_GATE_AREA_PX: int = 500

# --- Mission ---
WAYPOINT_ARRIVAL_RADIUS_M: float = 0.3
MAX_SPEED_MPS: float = 8.0

