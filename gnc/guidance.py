"""Guidance system — to be designed by the team."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Waypoint:
    """A target position the drone should pass through."""

    position: np.ndarray       # NED (m), shape (3,)
    yaw: float = 0.0           # rad


class GuidanceSystem:
    """Manages the ordered gate sequence and emits position setpoints.

    Implementation TBD — team to decide on guidance method
    (waypoint following, trajectory tracking, etc.) before filling this in.
    """

    def __init__(self, waypoints: List[Waypoint]) -> None:
        self._waypoints = waypoints
        self._index: int = 0

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if self.complete:
            return None
        return self._waypoints[self._index]

    @property
    def complete(self) -> bool:
        return self._index >= len(self._waypoints)

    def advance(self) -> None:
        self._index += 1

    def remaining(self) -> int:
        return max(0, len(self._waypoints) - self._index)
