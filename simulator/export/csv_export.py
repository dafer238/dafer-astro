import os
import csv
import numpy as np
from simulator.sim.trajectory import TrajectoryData
from simulator.core.constants import R_EARTH
from simulator.core.conversions import state_to_coe, StateVector


def export_trajectory_csv(traj: TrajectoryData, filepath: str | None = None,
                          stride: int = 10) -> str:
    if filepath is None:
        os.makedirs("exports", exist_ok=True)
        filepath = os.path.join("exports", "trajectory.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_s", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms",
            "r_km", "v_kms", "alt_km", "a_km", "e", "i_deg", "raan_deg",
            "omega_deg", "theta_deg"
        ])
        for i in range(0, len(traj.t), stride):
            r, v = traj.r[i], traj.v[i]
            r_mag = np.linalg.norm(r)
            v_mag = np.linalg.norm(v)
            alt = r_mag - R_EARTH
            coe = state_to_coe(StateVector(r=r, v=v))
            writer.writerow([
                f"{traj.t[i]:.3f}",
                f"{r[0]:.6f}", f"{r[1]:.6f}", f"{r[2]:.6f}",
                f"{v[0]:.8f}", f"{v[1]:.8f}", f"{v[2]:.8f}",
                f"{r_mag:.6f}", f"{v_mag:.8f}", f"{alt:.4f}",
                f"{coe.a:.4f}", f"{coe.e:.8f}", f"{coe.i:.6f}",
                f"{coe.raan:.6f}", f"{coe.omega:.6f}", f"{coe.theta:.4f}",
            ])

    return filepath
