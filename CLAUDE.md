# Drone Racing GNC — Claude Context

## Project Overview
Autonomous drone racing, GPS-denied, camera + IMU only.
Competition: Anduril AI Grand Prix

## Stack
- Python 3.11
- NumPy, SciPy for math
- OpenCV for perception
- MavLink for communication

## Repo Structure


## Conventions
- Type hints on all public functions
- Dataclasses for all inter-module data structures
- PID gains and tunable params live in config.py only
- Run tests from repo root: pytest tests/

