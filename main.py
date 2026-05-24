"""Main entry point — connects to the simulator and runs the control loop.

Thread model:
    main thread      — control loop (reads state, sends commands)
    telemetry thread — receives MAVLink messages, updates StateEstimator
    heartbeat thread — sends HEARTBEAT to sim at MIN_HEARTBEAT_HZ (≥2 Hz per §4.4)

DroneState is protected by state_lock. Always acquire the lock before reading
or writing state to avoid partial-update races between the telemetry thread and
the control loop.
"""

import threading
import time

import config
from gnc.state_estimator import StateEstimator, DroneState
from comms.mavlink_backend import MavlinkBackend
from utils.logger import get_logger

log = get_logger(__name__)


def telemetry_loop(
    backend: MavlinkBackend,
    estimator: StateEstimator,
    lock: threading.Lock,
    stop: threading.Event,
) -> None:
    """Receive a HIGHRES_IMU MAVLink message and push it into the state estimator."""
    assert backend._connection is not None
    conn = backend._connection

    while not stop.is_set():
        msg = conn.recv_match(type='HIGHRES_IMU', blocking=True, timeout=1.0)
        if not msg:
            continue
        if msg.get_type() == "BAD_DATA":
            continue
        with lock:
            estimator.update(msg)


def heartbeat_loop(
    backend: MavlinkBackend,
    stop: threading.Event,
) -> None:
    """Send HEARTBEAT to the simulator at MIN_HEARTBEAT_HZ."""
    interval = 1.0 / config.MIN_HEARTBEAT_HZ
    while not stop.is_set():
        backend.send_heartbeat()
        time.sleep(interval)


def main() -> None:
    backend = MavlinkBackend(config.MAVLINK_CONNECTION)
    backend.connect()
    log.info("Connected to simulator")

    estimator = StateEstimator()
    state_lock = threading.Lock()
    stop_event = threading.Event()

    telem_thread = threading.Thread(
        target=telemetry_loop,
        args=(backend, estimator, state_lock, stop_event),
        daemon=True,
        name="telemetry",
    )
    hb_thread = threading.Thread(
        target=heartbeat_loop,
        args=(backend, stop_event),
        daemon=True,
        name="heartbeat",
    )

    telem_thread.start()
    hb_thread.start()
    log.info("Telemetry and heartbeat threads started")

    dt = 1.0 / config.MAX_COMMAND_HZ

    try:
        while True:
            loop_start = time.monotonic()

            with state_lock:
                state: DroneState = estimator.state

            # --- Guidance and control go here ---
            # setpoint = guidance.current_waypoint
            # cmd = controller.compute(state, setpoint)
            # backend.set_velocity(cmd.velocity, cmd.yaw)

            log.debug(
                "att=[%.2f, %.2f, %.2f]",
                state.attitude[0], state.attitude[1], state.attitude[2],
            )

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        log.info("Shutting down")
        stop_event.set()


if __name__ == "__main__":
    main()
