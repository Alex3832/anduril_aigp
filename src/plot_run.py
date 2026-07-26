"""
Plot commanded vs. actual values for a logged run.

Usage:
    python plot_run.py [path/to/run_YYYYmmdd_HHMMSS.csv]

If no path is given, the most recently modified CSV in config.LOG_DIR is used.
"""
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt

import config

# (title, cmd column or None, act column) - rates have no cmd column: the rate loop
# is closed onboard the vehicle, not commanded by this code.
QUANTITIES = [
    ("x [m]", "x_cmd", "x_act"),
    ("y [m]", "y_cmd", "y_act"),
    ("z [m]", "z_cmd", "z_act"),
    ("vx [m/s]", "vx_cmd", "vx_act"),
    ("vy [m/s]", "vy_cmd", "vy_act"),
    ("vz [m/s]", "vz_cmd", "vz_act"),
    ("roll [rad]", "roll_cmd", "roll_act"),
    ("pitch [rad]", "pitch_cmd", "pitch_act"),
    ("yaw [rad]", "yaw_cmd", "yaw_act"),
    ("roll_rate [rad/s]", None, "roll_rate_act"),
    ("pitch_rate [rad/s]", None, "pitch_rate_act"),
    ("yaw_rate [rad/s]", None, "yaw_rate_act"),
]


def find_latest_run(log_dir=config.LOG_DIR):
    runs = glob.glob(os.path.join(log_dir, "run_*.csv"))
    if not runs:
        raise FileNotFoundError(f"No run_*.csv files found in {log_dir}")
    return max(runs, key=os.path.getmtime)


def load_run(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k: float(v) for k, v in row.items()} for row in reader]
    return rows


def plot_run(path):
    rows = load_run(path)
    t = [row["t"] for row in rows]

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    fig.suptitle(os.path.basename(path))

    for ax, (title, cmd_col, act_col) in zip(axes.flat, QUANTITIES):
        if cmd_col is not None:
            ax.plot(t, [row[cmd_col] for row in rows], label="commanded", linewidth=1.5)
        ax.plot(t, [row[act_col] for row in rows], label="actual", linewidth=1.0, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("t [s since arm]")
        ax.legend(fontsize="small")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Plotting {run_path}", flush=True)
    plot_run(run_path)
