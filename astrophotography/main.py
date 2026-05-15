"""
main.py — Entry point for the Equatorial Tracker astrophotography tool.

Usage:
    python main.py          Launch the GUI control panel (default)
    python main.py --cli    Run in CLI mode (print results to terminal)
    python main.py --plot   Run in CLI mode and show the 3D plot
"""

import argparse
import pathlib

import yaml


def load_config() -> dict:
    """Load configuration from the YAML file."""
    config_path = pathlib.Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_cli(show_plot: bool = False) -> None:
    """Run the tool in command-line mode."""
    from export import export_servo_to_csv, export_servo_to_json
    from tracker import (
        compute_exposure_table,
        compute_polar_alignment,
        correct_azimuth_for_magnetic,
        generate_servo_tracking_data,
        get_magnetic_declination,
        simulate_tracking_drift,
    )

    config = load_config()
    obs = config["observer"]
    star = config["target_star"]
    cam = config["camera"]
    fl_cfg = config["focal_lengths"]
    mag_cfg = config["magnetic_declination"]
    sim_cfg = config["alignment_simulation"]
    servo_cfg = config["servo"]

    # --- Polar Alignment ---
    alignment = compute_polar_alignment(
        lat=obs["latitude"],
        lon=obs["longitude"],
        elevation=obs["elevation"],
        star_ra=star["ra"],
        star_dec=star["dec"],
    )
    alt = alignment["altitude"]
    az = alignment["azimuth"]

    print("\n=== EQUATORIAL TRACKER CONFIGURATION ===")
    print(f"Latitude: {obs['latitude']:.4f}°")
    print(f"Longitude: {obs['longitude']:.4f}°")
    print(f"Elevation: {obs['elevation']} m")
    print(f"\nCelestial pole altitude: {alt:.2f}°")
    print(f"Celestial pole azimuth (true N): {az:.2f}°")
    print(f"Computed at: {alignment['time_utc']}")

    # --- Magnetic Declination ---
    if mag_cfg["mode"] == "auto":
        mag_dec = get_magnetic_declination(
            lat=obs["latitude"],
            lon=obs["longitude"],
            elevation_m=obs["elevation"],
        )
    else:
        mag_dec = mag_cfg["manual_value_deg"]

    print("\n--- Magnetic Declination Correction ---")
    if mag_dec is not None:
        direction = "East" if mag_dec >= 0 else "West"
        compass_az = correct_azimuth_for_magnetic(az, mag_dec)
        print(f"Magnetic declination: {mag_dec:+.2f}° ({direction})")
        print(f"Compass reading to polar axis: {compass_az:.2f}° (magnetic)")
    else:
        compass_az = az
        print("Magnetic declination: N/A (install pyIGRF for automatic calculation)")
        print(f"Using true azimuth: {az:.2f}°")

    # --- Physical Adjustments ---
    print("\n>>> TRACKER PHYSICAL ADJUSTMENTS:")
    print(f"  Tilt polar axis to: {alt:.2f}° from horizontal")
    print(
        f"  Orient compass to: {compass_az:.2f}°"
        f" ({'magnetic' if mag_dec else 'true N'})"
    )

    # --- Exposure Table ---
    focal_lengths = list(
        range(fl_cfg["min_mm"], fl_cfg["max_mm"] + 1, fl_cfg["step_mm"])
    )
    exposure_table = compute_exposure_table(
        focal_lengths=focal_lengths,
        crop_factor=cam["crop_factor"],
        pixel_pitch_um=cam["pixel_pitch_um"],
        aperture=cam["aperture"],
    )

    print(
        f"\n--- Maximum Exposure Table (crop={cam['crop_factor']}, "
        f"f/{cam['aperture']}, {cam['pixel_pitch_um']}µm) ---"
    )
    print(f"{'Focal (mm)':<12} {'500 Rule (s)':<14} {'NPF Rule (s)':<14}")
    print("-" * 40)
    for entry in exposure_table:
        npf_str = (
            f"{entry['max_exposure_npf']:.2f}" if entry["max_exposure_npf"] else "N/A"
        )
        print(
            f"{entry['focal_length_mm']:<12} {entry['max_exposure_500']:<14.2f} {npf_str:<14}"
        )

    # --- Drift Simulation ---
    import numpy as np

    drift = simulate_tracking_drift(
        misalignment_alt_arcsec=sim_cfg["misalignment_alt_arcsec"],
        misalignment_az_arcsec=sim_cfg["misalignment_az_arcsec"],
        observer_lat_deg=obs["latitude"],
        target_dec_deg=sim_cfg["target_declination_deg"],
        duration_sec=sim_cfg["simulation_duration_sec"],
        time_step_sec=sim_cfg["time_step_sec"],
    )
    max_drift = np.max(drift["total_drift_arcsec"])
    print("\n--- Alignment Error Simulation ---")
    print(f"  Alt misalignment: {sim_cfg['misalignment_alt_arcsec']} arcsec")
    print(f"  Az misalignment: {sim_cfg['misalignment_az_arcsec']} arcsec")
    print(f"  Target declination: {sim_cfg['target_declination_deg']}°")
    print(f"  Duration: {sim_cfg['simulation_duration_sec']} sec")
    print(f"  Max total drift: {max_drift:.2f} arcsec ({max_drift / 60:.3f} arcmin)")

    # --- Servo Data Export ---
    servo_data = generate_servo_tracking_data(
        duration_sec=servo_cfg["duration_sec"],
        time_step_sec=servo_cfg["time_step_sec"],
        sidereal_rate=True,
    )

    output_dir = pathlib.Path(servo_cfg["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = servo_cfg["output_format"]
    metadata = {
        "observer_latitude": obs["latitude"],
        "observer_longitude": obs["longitude"],
        "polar_altitude_deg": alt,
        "polar_azimuth_deg": az,
        "magnetic_declination_deg": mag_dec,
        "compass_azimuth_deg": compass_az,
    }

    if fmt == "json":
        filepath = output_dir / "servo_tracking.json"
        export_servo_to_json(servo_data, filepath, metadata=metadata)
    else:
        filepath = output_dir / "servo_tracking.csv"
        export_servo_to_csv(servo_data, filepath, metadata=metadata)

    print("\n--- Servo Export ---")
    print(f"  Format: {fmt.upper()}")
    print(
        f"  Duration: {servo_cfg['duration_sec']} sec ({servo_cfg['duration_sec'] / 3600:.2f} hours)"
    )
    print(f"  Time step: {servo_cfg['time_step_sec']} sec")
    print(f"  Data points: {len(servo_data['time_sec'])}")
    print(f"  Saved to: {filepath.resolve()}")

    # --- 3D Plot (optional) ---
    if show_plot:
        _show_3d_plot(alt, az, obs["latitude"], star["name"])


def _show_3d_plot(alt: float, az: float, lat: float, star_name: str) -> None:
    """Display the 3D polar alignment plot."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- Earth globe with correct rotation axis ---
    globe_radius = 0.3
    earth_axis_alt = np.radians(lat)
    earth_ax_y = np.cos(earth_axis_alt)
    earth_ax_z = np.sin(earth_axis_alt)
    earth_axis = np.array([0.0, earth_ax_y, earth_ax_z])
    earth_axis = earth_axis / np.linalg.norm(earth_axis)

    def rotation_matrix_from_z_to(target):
        z = np.array([0.0, 0.0, 1.0])
        t = target / np.linalg.norm(target)
        v = np.cross(z, t)
        s = np.linalg.norm(v)
        c = np.dot(z, t)
        if s < 1e-10:
            return np.eye(3) if c > 0 else -np.eye(3)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    R = rotation_matrix_from_z_to(earth_axis)

    u_s = np.linspace(0, 2 * np.pi, 40)
    v_s = np.linspace(0, np.pi, 20)
    x_s = globe_radius * np.outer(np.cos(u_s), np.sin(v_s))
    y_s = globe_radius * np.outer(np.sin(u_s), np.sin(v_s))
    z_s = globe_radius * np.outer(np.ones(np.size(u_s)), np.cos(v_s))

    x_rot = R[0, 0] * x_s + R[0, 1] * y_s + R[0, 2] * z_s
    y_rot = R[1, 0] * x_s + R[1, 1] * y_s + R[1, 2] * z_s
    z_rot = R[2, 0] * x_s + R[2, 1] * y_s + R[2, 2] * z_s

    ax.plot_surface(
        x_rot,
        y_rot,
        z_rot,
        color="lightblue",
        alpha=0.25,
        edgecolor="steelblue",
        linewidth=0.15,
    )

    # Latitude rings
    for ring_lat_deg in [0, 30, -30, 60, -60]:
        ring_lat = np.radians(ring_lat_deg)
        theta = np.linspace(0, 2 * np.pi, 60)
        rx = globe_radius * np.cos(ring_lat) * np.cos(theta)
        ry = globe_radius * np.cos(ring_lat) * np.sin(theta)
        rz = globe_radius * np.sin(ring_lat) * np.ones_like(theta)
        rx_r = R[0, 0] * rx + R[0, 1] * ry + R[0, 2] * rz
        ry_r = R[1, 0] * rx + R[1, 1] * ry + R[1, 2] * rz
        rz_r = R[2, 0] * rx + R[2, 1] * ry + R[2, 2] * rz
        lw = 1.0 if ring_lat_deg == 0 else 0.4
        color = "darkblue" if ring_lat_deg == 0 else "steelblue"
        ax.plot(rx_r, ry_r, rz_r, color=color, linewidth=lw, alpha=0.6)

    # Earth rotation axis
    ax.quiver(
        0,
        0,
        0,
        earth_axis[0] * 0.9,
        earth_axis[1] * 0.9,
        earth_axis[2] * 0.9,
        color="blue",
        arrow_length_ratio=0.08,
        linewidth=2,
        label=f"Earth rotation axis (lat={lat:.2f}\u00b0)",
    )

    # Zenith
    ax.quiver(
        0,
        0,
        0,
        0,
        0,
        0.9,
        color="cyan",
        arrow_length_ratio=0.08,
        linewidth=1.5,
        label="Zenith (local vertical)",
    )

    # Tracker polar axis
    x_polar = np.cos(np.radians(alt)) * np.sin(np.radians(az))
    y_polar = np.cos(np.radians(alt)) * np.cos(np.radians(az))
    z_polar = np.sin(np.radians(alt))
    ax.quiver(
        0,
        0,
        0,
        x_polar,
        y_polar,
        z_polar,
        color="red",
        arrow_length_ratio=0.08,
        linewidth=2,
        label=f"Tracker polar axis ({star_name})",
    )

    # North direction
    ax.quiver(
        0,
        0,
        0,
        0,
        0.9,
        0,
        color="green",
        arrow_length_ratio=0.08,
        linewidth=1.5,
        label="North (horizon)",
    )

    # Altitude angle arc
    arc_pts = 30
    alt_angles = np.linspace(0, np.radians(alt), arc_pts)
    arc_x = 0.6 * np.cos(alt_angles) * np.sin(np.radians(az))
    arc_y = 0.6 * np.cos(alt_angles) * np.cos(np.radians(az))
    arc_z = 0.6 * np.sin(alt_angles)
    ax.plot(arc_x, arc_y, arc_z, color="orange", linewidth=2.5)
    mid_alt_r = np.radians(alt) / 2
    ax.text(
        0.72 * np.cos(mid_alt_r) * np.sin(np.radians(az)),
        0.72 * np.cos(mid_alt_r) * np.cos(np.radians(az)),
        0.72 * np.sin(mid_alt_r),
        f"Alt = {alt:.2f}\u00b0",
        color="orange",
        fontsize=10,
        fontweight="bold",
    )

    # Azimuth angle arc
    az_angles = np.linspace(0, np.radians(az), arc_pts)
    ax.plot(
        0.5 * np.sin(az_angles),
        0.5 * np.cos(az_angles),
        np.zeros(arc_pts),
        color="purple",
        linewidth=2.5,
    )
    mid_az_r = np.radians(az) / 2
    ax.text(
        0.6 * np.sin(mid_az_r),
        0.6 * np.cos(mid_az_r),
        -0.05,
        f"Az = {az:.2f}\u00b0",
        color="purple",
        fontsize=10,
        fontweight="bold",
    )

    # Zenith-to-axis angle
    za_angles = np.linspace(0, np.radians(90 - lat), arc_pts)
    ax.plot(
        np.zeros(arc_pts),
        0.45 * np.sin(za_angles),
        0.45 * np.cos(za_angles),
        color="blue",
        linewidth=1.5,
        linestyle="--",
    )
    mid_za = np.radians(90 - lat) / 2
    ax.text(
        0,
        0.48 * np.sin(mid_za),
        0.48 * np.cos(mid_za),
        f"90\u00b0-lat = {90 - lat:.2f}\u00b0",
        color="blue",
        fontsize=8,
    )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X (East)")
    ax.set_ylabel("Y (North)")
    ax.set_zlabel("Z (Zenith)")
    ax.set_title("Polar Axis Alignment \u2014 Local Horizontal Coordinates")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def run_gui() -> None:
    """Launch the GUI control panel."""
    from gui import TrackerGUI

    app = TrackerGUI()
    app.run()


def main() -> None:
    """Parse arguments and dispatch to GUI or CLI."""
    parser = argparse.ArgumentParser(
        description="Equatorial Tracker — Astrophotography alignment and tracking tool"
    )
    parser.add_argument(
        "--cli", action="store_true", help="Run in command-line mode (no GUI)"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show 3D alignment plot (implies --cli)"
    )
    args = parser.parse_args()

    if args.cli or args.plot:
        run_cli(show_plot=args.plot)
    else:
        run_gui()


if __name__ == "__main__":
    main()
