"""
Standalone sniffer for MAVLink ENCAPSULATED_DATA messages from the AIGP simulator.

Run this instead of (not alongside) the main drone stack:
    python sniff_encapsulated.py [--host 127.0.0.1] [--port 14550]

Prints race status updates as they arrive and dumps the full gate table
when track data is fully reassembled.
"""

import argparse
import struct
import time
from datetime import datetime

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID  = 2

RACE_STATUS_FMT = "<BQqqIq"
TRACK_CHUNK_HDR_FMT = "<BH"
GATES_HDR_FMT = "<H"
GATE_FMT = "<Hfffffffff"
GATE_SZ = struct.calcsize(GATE_FMT)


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def print_race_status(raw: bytes) -> None:
    _, sim_boot_ms, race_start_ms, race_finish_ns, active_gate, last_gate_time = struct.unpack_from(
        RACE_STATUS_FMT, raw
    )
    race_running = race_finish_ns < 0
    race_started = race_start_ms >= 0

    elapsed_s = ""
    if race_started and race_running:
        elapsed_ms = sim_boot_ms - race_start_ms
        elapsed_s = f"  elapsed={elapsed_ms/1000:.3f}s"

    finish_s = ""
    if not race_running:
        finish_s = f"  FINISHED race_time={race_finish_ns/1e9:.3f}s"

    print(
        f"[{ts()}] RACE_STATUS  sim={sim_boot_ms}ms"
        f"  gate={active_gate}"
        f"  last_gate_time={last_gate_time/1e9:.3f}s"
        f"{elapsed_s}{finish_s}"
    )


def print_track_data(payload: bytes) -> None:
    num_gates, = struct.unpack_from(GATES_HDR_FMT, payload)
    payload = payload[struct.calcsize(GATES_HDR_FMT):]

    print(f"\n[{ts()}] TRACK_DATA  {num_gates} gates")
    print(f"  {'ID':>3}  {'pos_N':>8} {'pos_E':>8} {'pos_D':>8}  "
          f"{'qw':>7} {'qx':>7} {'qy':>7} {'qz':>7}  "
          f"{'W':>6} {'H':>6}")
    print("  " + "-" * 90)

    for i in range(num_gates):
        gate_id, nx, ey, dz, qw, qx, qy, qz, w, h = struct.unpack_from(GATE_FMT, payload)
        payload = payload[GATE_SZ:]
        print(f"  {gate_id:>3}  {nx:>8.3f} {ey:>8.3f} {dz:>8.3f}  "
              f"{qw:>7.4f} {qx:>7.4f} {qy:>7.4f} {qz:>7.4f}  "
              f"{w:>6.2f} {h:>6.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sniff AIGP simulator ENCAPSULATED_DATA messages")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14550)
    args = parser.parse_args()

    conn_str = f"udpin:{args.host}:{args.port}"
    print(f"Connecting to {conn_str} ...")
    conn = mavutil.mavlink_connection(conn_str)
    print("Waiting for heartbeat...")
    conn.wait_heartbeat()
    print(f"Heartbeat received from system {conn.target_system}. Sniffing...\n")

    track_chunks: dict = {}
    expected_chunks: dict = {}

    while True:
        try:
            msg = conn.recv_match(blocking=True)
        except KeyboardInterrupt:
            print("\nStopped.")
            break

        if msg is None or msg.get_type() == "BAD_DATA":
            continue

        msg_type = msg.get_type()

        if msg_type == "DATA_TRANSMISSION_HANDSHAKE":
            transfer_id = msg.width
            track_chunks[transfer_id] = {}
            expected_chunks[transfer_id] = msg.packets
            print(f"[{ts()}] HANDSHAKE  transfer_id={transfer_id}  expecting {msg.packets} chunks")

        elif msg_type == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            data_type = raw[0]

            if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
                print_race_status(raw)

            elif data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
                _, transfer_id = struct.unpack_from(TRACK_CHUNK_HDR_FMT, raw)
                if transfer_id not in expected_chunks:
                    continue
                chunk_payload = raw[struct.calcsize(TRACK_CHUNK_HDR_FMT):]
                track_chunks[transfer_id][msg.seqnr] = chunk_payload

                received = len(track_chunks[transfer_id])
                total = expected_chunks[transfer_id]
                print(f"[{ts()}] TRACK_CHUNK  transfer={transfer_id}  chunk {msg.seqnr}  ({received}/{total})")

                if received == total:
                    full = b"".join(
                        track_chunks[transfer_id][i] for i in range(total)
                    )
                    del track_chunks[transfer_id]
                    del expected_chunks[transfer_id]
                    print_track_data(full)


if __name__ == "__main__":
    main()
