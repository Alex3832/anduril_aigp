"""Camera intrinsics and UDP vision-stream receiver.

The simulator does NOT expose a V4L2/DirectShow device. Frames are pushed as
chunked JPEG packets over UDP on port 5600 using the binary format defined in
VADR-TS-002 §4.6. VisionStreamReceiver implements that protocol.
"""

import socket
import struct
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

# Binary header layout (little-endian, 24 bytes total)  [SPEC §4.6]
_HEADER_FMT = "<IHHIIQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # == 24


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters (no lens distortion)."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_spec(cls) -> "CameraIntrinsics":
        """Return the intrinsics specified in VADR-TS-002 §3.8."""
        return cls(
            fx=config.CAMERA_FX,
            fy=config.CAMERA_FY,
            cx=config.CAMERA_CX,
            cy=config.CAMERA_CY,
        )

    @property
    def matrix(self) -> np.ndarray:
        """3×3 camera matrix K."""
        return np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ],
        ], dtype=float)

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Back-project pixel (u, v) to a unit ray in the OpenCV camera frame.

        No distortion correction needed per spec (§3.8).

        Returns:
            Normalised (3,) vector [x, y, 1]-normalised in camera frame.
        """
        ray = np.array([(u - self.cx) / self.fx,
                        (v - self.cy) / self.fy,
                        1.0])
        return ray / np.linalg.norm(ray)


def body_to_camera_rotation() -> np.ndarray:
    """Return the 3×3 rotation matrix from body-NED frame to OpenCV camera frame.

    Spec (§3.8):
    - Body NED: X forward, Y right, Z down.
    - Camera is at the same origin, tilted 20° upward (pitched nose-up).
    - Body-to-IMU is identity.

    Step 1 — rotate body NED by -20° around Y (nose up) to get camera-NED frame:
        Ry(-20°) = [[cos20,  0, -sin20],
                    [0,      1,  0    ],
                    [sin20,  0,  cos20]]

    Step 2 — reorder axes from NED camera to OpenCV convention
              (X_cv = right = Y_cam_NED, Y_cv = down = Z_cam_NED, Z_cv = fwd = X_cam_NED):
        R_ned_to_cv = [[0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0]]

    Returns:
        R (3×3) such that p_cv = R @ p_body_NED.
    """
    tilt = np.deg2rad(config.CAMERA_TILT_UP_DEG)
    c, s = np.cos(tilt), np.sin(tilt)

    R_body_to_cam_ned = np.array([
        [ c,  0, -s],
        [ 0,  1,  0],
        [ s,  0,  c],
    ])

    R_ned_to_cv = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])

    return R_ned_to_cv @ R_body_to_cam_ned


@dataclass
class _FrameBuffer:
    """Accumulates chunks for a single frame_id."""
    total_chunks: int
    jpeg_size: int
    sim_time_ns: int
    chunks: Dict[int, bytes]

    def complete(self) -> bool:
        return len(self.chunks) == self.total_chunks

    def assemble(self) -> bytes:
        return b"".join(self.chunks[i] for i in range(self.total_chunks))


class VisionStreamReceiver:
    """Receives the simulator's chunked UDP JPEG stream and exposes decoded frames.

    The simulator sends each camera frame split across one or more UDP packets.
    Each packet carries a 24-byte metadata header followed by a JPEG slice.
    Frames are reassembled from chunks and decoded to BGR numpy arrays.

    Usage:
        with VisionStreamReceiver() as cam:
            while True:
                result = cam.read()
                if result is not None:
                    frame, sim_time_ns = result
                    # process frame ...
    """

    def __init__(self, port: int = config.VISION_UDP_PORT,
                 buffer_size: int = 65536) -> None:
        self._port = port
        self._buf_size = buffer_size
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, int]] = None  # (frame_bgr, sim_time_ns)
        self._running = False
        self._frame_buffers: Dict[int, _FrameBuffer] = {}

    def start(self) -> None:
        """Open the socket and begin receiving frames in a background thread."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()

    def read(self) -> Optional[Tuple[np.ndarray, int]]:
        """Return the most recently completed (frame_bgr, sim_time_ns), or None."""
        with self._lock:
            return self._latest

    def __enter__(self) -> "VisionStreamReceiver":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                data = self._sock.recv(self._buf_size)
            except socket.timeout:
                continue

            if len(data) < _HEADER_SIZE:
                continue

            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = (
                struct.unpack_from(_HEADER_FMT, data, 0)
            )
            payload = data[_HEADER_SIZE: _HEADER_SIZE + payload_size]

            if frame_id not in self._frame_buffers:
                self._frame_buffers[frame_id] = _FrameBuffer(
                    total_chunks=total_chunks,
                    jpeg_size=jpeg_size,
                    sim_time_ns=sim_time_ns,
                    chunks={},
                )
            buf = self._frame_buffers[frame_id]
            buf.chunks[chunk_id] = payload

            if buf.complete():
                jpeg_bytes = buf.assemble()
                self._frame_buffers.pop(frame_id, None)
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    with self._lock:
                        self._latest = (frame, buf.sim_time_ns)
