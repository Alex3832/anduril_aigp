"""Flight controller — to be designed by the team."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ControlOutput:
    """Command sent to the drone interface each control tick."""

    velocity: np.ndarray  # NED (m/s), shape (3,)
    yaw: float            # rad


class Controller:
    """Converts state + guidance setpoint into a ControlOutput.

    Implementation TBD — team to decide on control method
    (PID, MPC, pure pursuit, etc.) before filling this in.
    """

    def compute(self, state, setpoint) -> ControlOutput:
        raise NotImplementedError
