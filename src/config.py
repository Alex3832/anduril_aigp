from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------------------
# CONTROL LOOP
# --------------------------------------------------------------------------------------
CONTROL_HZ = 250

# --------------------------------------------------------------------------------------
# DATA LOGGING
# --------------------------------------------------------------------------------------
LOG_DIR = "logs"

# --------------------------------------------------------------------------------------
# PID GAINS
# --------------------------------------------------------------------------------------
@dataclass
class PIDGains:
    """kp, ki, kd plus optional output/integral saturation limits for a single PID loop."""
    kp: float
    ki: float
    kd: float
    output_limit: Optional[float] = None
    integral_limit: Optional[float] = None


# Outer loop: NED position error [m] -> desired NED velocity setpoint [m/s]
POS_X = PIDGains(kp=0.8, ki=0.0, kd=0.1, output_limit=3.0)
POS_Y = PIDGains(kp=0.8, ki=0.0, kd=0.1, output_limit=3.0)
POS_Z = PIDGains(kp=0.8, ki=0.0, kd=0.1, output_limit=.25)

# Middle loop: NED velocity error [m/s] -> desired roll/pitch angle [rad] + thrust [0..1]
VEL_ROLL = PIDGains(kp=0.12, ki=0.02, kd=0.01, output_limit=0.35)
VEL_PITCH = PIDGains(kp=0.12, ki=0.02, kd=0.01, output_limit=0.35)
VEL_THRUST = PIDGains(kp=0.20, ki=0.1, kd=0.1, output_limit=1, integral_limit=.9)
