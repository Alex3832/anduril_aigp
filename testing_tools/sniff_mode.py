"""
Standalone sniffer that continuously reports simulation mode from the AIGP simulator.

Reports:
  - Flight mode and arm state from HEARTBEAT (1 Hz)
  - Race status transitions from ENCAPSULATED_DATA

Run this instead of (not alongside) the main drone stack:
    python sniff_mode.py [--host 127.0.0.1] [--port 14550]
"""

import argparse
import struct
from datetime import datetime

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
RACE_STATUS_FMT = "<BQqqIq"

# ArduCopter custom flight modes
ARDUPILOT_MODES = {
    0:  "STABILIZE",
    1:  "ACRO",
    2:  "ALT_HOLD",
    3:  "AUTO",
    4:  "GUIDED",
    5:  "LOITER",
    6:  "RTL",
    7:  "CIRCLE",
    9:  "LAND",
    11: "DRIFT",
    13: "SPORT",
    16: "POSHOLD",
    17: "BRAKE",
    20: "GUIDED_NOGPS",
}

MAV_STATES = {
    0: "UNINIT",
    1: "BOOT",
    2: "CALIBRATING",
    3: "STANDBY",
    4: "ACTIVE",
    5: "CRITICAL",
    6: "EMERGENCY",
    7: "POWEROFF",
    8: "FLIGHT_TERMINATION",
}


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def decode_flight_mode(base_mode: int, custom_mode: int) -> str:
    armed = bool(base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    custom_enabled = bool(base_mode & mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED)

    if custom_enabled:
        mode_name = ARDUPILOT_MODES.get(custom_mode, f"CUSTOM({custom_mode})")
    else:
        mode_name = f"base_mode=0x{base_mode:02X}"

    arm_str = "ARMED" if armed else "DISARMED"
    return f"{mode_name}  [{arm_str}]"


def decode_race_status(raw: bytes) -> dict:
    _, sim_boot_ms, race_start_ms, race_finish_ns, active_gate, last_gate_time = struct.unpack_from(
        RACE_STATUS_FMT, raw
    )
    race_started = race_start_ms >= 0
    race_finished = race_finish_ns >= 0

    if not race_started:
        phase = "PRE-RACE"
    elif race_finished:
        phase = "FINISHED"
    else:
        phase = "RACING"

    return {
        "phase": phase,
        "sim_boot_ms": sim_boot_ms,
        "race_start_ms": race_start_ms,
        "race_finish_ns": race_finish_ns,
        "active_gate": active_gate,
        "last_gate_time": last_gate_time,
    }


def format_race_line(rs: dict) -> str:
    phase = rs["phase"]
    gate = rs["active_gate"]
    sim_ms = rs["sim_boot_ms"]

    if phase == "RACING":
        elapsed_ms = sim_ms - rs["race_start_ms"]
        return (
            f"RACE:{phase}  gate={gate}"
            f"  elapsed={elapsed_ms / 1000:.3f}s"
            f"  last_gate={rs['last_gate_time'] / 1e9:.3f}s"
        )
    elif phase == "FINISHED":
        return (
            f"RACE:{phase}  gate={gate}"
            f"  race_time={rs['race_finish_ns'] / 1e9:.3f}s"
        )
    else:
        return f"RACE:{phase}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sniff AIGP simulator mode continuously")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14550)
    args = parser.parse_args()

    conn_str = f"udpin:{args.host}:{args.port}"
    print(f"Connecting to {conn_str} ...")
    conn = mavutil.mavlink_connection(conn_str)
    print("Waiting for heartbeat...")
    conn.wait_heartbeat()
    print(f"Heartbeat from system {conn.target_system}. Sniffing mode...\n")

    last_flight_mode_str = None
    last_sys_state = None
    last_race_phase = None

    while True:
        try:
            msg = conn.recv_match(blocking=True)
        except KeyboardInterrupt:
            print("\nStopped.")
            break

        if msg is None or msg.get_type() == "BAD_DATA":
            continue

        msg_type = msg.get_type()

        if msg_type == "HEARTBEAT":
            flight_mode_str = decode_flight_mode(msg.base_mode, msg.custom_mode)
            sys_state = MAV_STATES.get(msg.system_status, f"STATE({msg.system_status})")

            mode_changed = flight_mode_str != last_flight_mode_str
            state_changed = sys_state != last_sys_state

            if mode_changed or state_changed:
                tag = " [CHANGED]" if (last_flight_mode_str is not None) else " [INIT]"
                print(f"[{ts()}] MODE  {flight_mode_str}  sys={sys_state}{tag}")
                last_flight_mode_str = flight_mode_str
                last_sys_state = sys_state
            else:
                print(f"[{ts()}] MODE  {flight_mode_str}  sys={sys_state}")

        elif msg_type == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if raw[0] != ENCAPSULATED_RACE_STATUS_MSG_ID:
                continue

            rs = decode_race_status(raw)
            race_line = format_race_line(rs)

            if rs["phase"] != last_race_phase:
                print(f"[{ts()}] {race_line}  [CHANGED]")
                last_race_phase = rs["phase"]
            else:
                print(f"[{ts()}] {race_line}")


if __name__ == "__main__":
    main()
