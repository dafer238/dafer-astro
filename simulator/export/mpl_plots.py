import os
import numpy as np
import matplotlib.pyplot as plt
from simulator.sim.trajectory import TrajectoryData
from simulator.core.constants import R_EARTH, MU_EARTH
from simulator.core.conversions import specific_energy, state_to_coe, StateVector


def export_plots(traj: TrajectoryData, filepath: str | None = None) -> str:
    if filepath is None:
        os.makedirs("exports", exist_ok=True)
        filepath = os.path.join("exports", "orbit_plots.png")

    t_min = traj.t / 60.0
    alt = np.linalg.norm(traj.r, axis=1) - R_EARTH
    vel = np.linalg.norm(traj.v, axis=1)

    stride = max(1, len(traj.t) // 500)
    idx = np.arange(0, len(traj.t), stride)
    eps_arr = np.array([specific_energy(traj.r[i], traj.v[i], MU_EARTH) for i in idx])

    fig = plt.figure(figsize=(14, 10), facecolor="#0a0a1a")

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_facecolor("#111122")
    ax1.plot(traj.r[:, 0], traj.r[:, 1], traj.r[:, 2],
             color="#00bfff", lw=0.8, alpha=0.8)
    ax1.scatter(*traj.r[0], color="lime", s=50, zorder=5)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    ax1.plot_surface(
        R_EARTH * np.outer(np.cos(u), np.sin(v)),
        R_EARTH * np.outer(np.sin(u), np.sin(v)),
        R_EARTH * np.outer(np.ones(20), np.cos(v)),
        color="deepskyblue", alpha=0.15, linewidth=0
    )
    lim = np.max(np.abs(traj.r)) * 1.1
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_zlim(-lim, lim)
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlabel("X (km)", color="white", fontsize=8)
    ax1.set_ylabel("Y (km)", color="white", fontsize=8)
    ax1.set_zlabel("Z (km)", color="white", fontsize=8)
    ax1.set_title("3D Orbit — ECI", color="white", fontsize=10)
    ax1.tick_params(colors="gray", labelsize=6)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#111122")
    ax2.plot(t_min, alt, color="#00bfff", lw=1)
    ax2.set_xlabel("Time (min)", color="gray")
    ax2.set_ylabel("Altitude (km)", color="gray")
    ax2.set_title("Altitude vs Time", color="white", fontsize=10)
    ax2.tick_params(colors="gray")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#111122")
    ax3.plot(t_min, vel, color="#ff6b35", lw=1)
    ax3.set_xlabel("Time (min)", color="gray")
    ax3.set_ylabel("Velocity (km/s)", color="gray")
    ax3.set_title("Velocity vs Time", color="white", fontsize=10)
    ax3.tick_params(colors="gray")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#111122")
    ax4.plot(traj.t[idx] / 60.0, eps_arr, color="#7fff00", lw=1)
    ax4.set_xlabel("Time (min)", color="gray")
    ax4.set_ylabel("Energy (km2/s2)", color="gray")
    ax4.set_title("Specific Energy", color="white", fontsize=10)
    ax4.tick_params(colors="gray")

    plt.tight_layout()
    plt.savefig(filepath, dpi=140, facecolor="#0a0a1a", bbox_inches="tight")
    plt.close(fig)

    return filepath
