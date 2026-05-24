"""Run performance tracking — gate crossing times, speeds, and lap time.

This is the objective the AI optimises against. A lower lap_time_s with all
gates crossed is a better run.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GateCrossing:
    """Record of a single gate passage."""

    gate_index: int
    wall_time_s: float   # monotonic clock time of crossing
    speed_mps: float     # drone speed at crossing


class PerformanceTracker:
    """Records gate crossings during a run and derives performance metrics.

    Usage:
        tracker = PerformanceTracker(n_gates=10)
        tracker.start_run()
        ...
        tracker.record_crossing(gate_index=0, speed_mps=current_speed)
        ...
        print(tracker.lap_time_s)   # None until all gates crossed
    """

    def __init__(self, n_gates: int) -> None:
        """
        Args:
            n_gates: Total number of gates on the course (including finish).
        """
        self._n_gates = n_gates
        self._crossings: List[GateCrossing] = []
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """Reset all state and record the run start time."""
        self._crossings.clear()
        self._start_time = time.monotonic()

    def record_crossing(self, gate_index: int, speed_mps: float) -> None:
        """Record that the drone passed through a gate.

        Args:
            gate_index: Zero-based index of the gate that was crossed.
            speed_mps:  Drone speed at the moment of crossing (m/s).
        """
        self._crossings.append(GateCrossing(
            gate_index=gate_index,
            wall_time_s=time.monotonic(),
            speed_mps=speed_mps,
        ))

    # ------------------------------------------------------------------
    # Metrics (read by the AI as part of AgentObservation)
    # ------------------------------------------------------------------

    @property
    def lap_time_s(self) -> Optional[float]:
        """Total elapsed time from start to final gate, or None if incomplete."""
        if self._start_time is None or len(self._crossings) < self._n_gates:
            return None
        return self._crossings[-1].wall_time_s - self._start_time

    @property
    def gates_crossed(self) -> int:
        return len(self._crossings)

    def last_gate_speed(self) -> float:
        """Speed at the most recently crossed gate, or 0.0 if none yet."""
        if not self._crossings:
            return 0.0
        return self._crossings[-1].speed_mps

    def last_segment_time_s(self) -> float:
        """Wall time between the two most recent gate crossings, or 0.0."""
        if len(self._crossings) < 2:
            return 0.0
        return self._crossings[-1].wall_time_s - self._crossings[-2].wall_time_s

    def average_gate_speed(self) -> float:
        """Mean speed across all recorded gate crossings so far."""
        if not self._crossings:
            return 0.0
        return float(sum(c.speed_mps for c in self._crossings) / len(self._crossings))

    @property
    def complete(self) -> bool:
        return len(self._crossings) >= self._n_gates
