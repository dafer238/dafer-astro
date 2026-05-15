"""
gui.py — Simple tkinter control panel for the equatorial tracker tool.

Provides a dashboard-style GUI integrating:
    - Polar alignment display with magnetic declination correction
    - Maximum exposure table for configurable focal lengths
    - Tracking drift simulation with plot
    - Servo data export (JSON/CSV)
    - 3D polar alignment visualization
"""

import pathlib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from export import (
    export_alignment_report,
    export_exposure_table_csv,
    export_servo_to_csv,
    export_servo_to_json,
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tracker import (
    compute_exposure_table,
    compute_polar_alignment,
    correct_azimuth_for_magnetic,
    generate_servo_tracking_data,
    get_magnetic_declination,
    simulate_tracking_drift,
)

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def load_config() -> dict[str, Any]:
    """Load configuration from the YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class TrackerGUI:
    """Main GUI application for the equatorial tracker control panel."""

    def __init__(self) -> None:
        self.config = load_config()
        self.root = tk.Tk()
        self.root.title("Equatorial Tracker — Control Panel")
        self.root.geometry("1000x720")
        self.root.minsize(900, 650)

        # Data holders
        self.alignment_data: dict[str, Any] = {}
        self.mag_declination: float | None = None
        self.exposure_table: list[dict] = []
        self.drift_data: dict[str, np.ndarray] = {}
        self.servo_data: dict[str, Any] = {}

        self._build_ui()
        self._run_calculations()

    def _build_ui(self) -> None:
        """Construct the UI layout."""
        # Top bar with a refresh button
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(fill=tk.X)
        ttk.Label(
            top_frame,
            text="Equatorial Tracker Control Panel",
            font=("Segoe UI", 14, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(top_frame, text="Refresh", command=self._run_calculations).pack(
            side=tk.RIGHT
        )
        ttk.Button(top_frame, text="3D Plot", command=self._show_3d_plot).pack(
            side=tk.RIGHT, padx=5
        )

        # Notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Alignment
        self.alignment_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.alignment_frame, text="Polar Alignment")
        self._build_alignment_tab()

        # Tab 2: Exposure Table
        self.exposure_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.exposure_frame, text="Exposure Table")
        self._build_exposure_tab()

        # Tab 3: Drift Simulation
        self.drift_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.drift_frame, text="Drift Simulation")
        self._build_drift_tab()

        # Tab 4: Servo Export
        self.servo_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.servo_frame, text="Servo Export")
        self._build_servo_tab()

    # ------------------------------------------------------------------
    # Alignment Tab
    # ------------------------------------------------------------------
    def _build_alignment_tab(self) -> None:
        """Build the polar alignment display tab."""
        frame = self.alignment_frame

        # Observer info
        obs_group = ttk.LabelFrame(frame, text="Observer Location", padding=10)
        obs_group.pack(fill=tk.X, pady=5)

        cfg = self.config["observer"]
        self.lbl_lat = ttk.Label(obs_group, text=f"Latitude: {cfg['latitude']:.4f}°")
        self.lbl_lat.grid(row=0, column=0, sticky=tk.W, padx=10)
        self.lbl_lon = ttk.Label(obs_group, text=f"Longitude: {cfg['longitude']:.4f}°")
        self.lbl_lon.grid(row=0, column=1, sticky=tk.W, padx=10)
        self.lbl_elev = ttk.Label(obs_group, text=f"Elevation: {cfg['elevation']} m")
        self.lbl_elev.grid(row=0, column=2, sticky=tk.W, padx=10)

        # Polar alignment results
        align_group = ttk.LabelFrame(frame, text="Polar Alignment", padding=10)
        align_group.pack(fill=tk.X, pady=5)

        self.lbl_pole_alt = ttk.Label(align_group, text="Celestial pole altitude: —")
        self.lbl_pole_alt.grid(row=0, column=0, sticky=tk.W, padx=10, pady=3)
        self.lbl_pole_az = ttk.Label(align_group, text="Celestial pole azimuth: —")
        self.lbl_pole_az.grid(row=1, column=0, sticky=tk.W, padx=10, pady=3)
        self.lbl_time = ttk.Label(align_group, text="Computed at: —")
        self.lbl_time.grid(row=2, column=0, sticky=tk.W, padx=10, pady=3)

        # Magnetic correction
        mag_group = ttk.LabelFrame(
            frame, text="Magnetic Declination Correction", padding=10
        )
        mag_group.pack(fill=tk.X, pady=5)

        self.lbl_mag_dec = ttk.Label(mag_group, text="Magnetic declination: —")
        self.lbl_mag_dec.grid(row=0, column=0, sticky=tk.W, padx=10, pady=3)
        self.lbl_compass_az = ttk.Label(
            mag_group, text="Compass reading (magnetic N): —"
        )
        self.lbl_compass_az.grid(row=1, column=0, sticky=tk.W, padx=10, pady=3)
        self.lbl_mag_note = ttk.Label(
            mag_group,
            text="Note: Point your compass to this reading to align with true celestial pole.",
            foreground="gray",
        )
        self.lbl_mag_note.grid(row=2, column=0, sticky=tk.W, padx=10, pady=3)

        # Physical adjustments summary
        adj_group = ttk.LabelFrame(frame, text="Physical Adjustments", padding=10)
        adj_group.pack(fill=tk.X, pady=5)

        self.lbl_tilt = ttk.Label(adj_group, text="Tilt polar axis to: —")
        self.lbl_tilt.grid(row=0, column=0, sticky=tk.W, padx=10, pady=3)
        self.lbl_orient = ttk.Label(adj_group, text="Orient compass to: —")
        self.lbl_orient.grid(row=1, column=0, sticky=tk.W, padx=10, pady=3)

        # Export button
        ttk.Button(
            frame, text="Export Alignment Report (JSON)", command=self._export_alignment
        ).pack(pady=10)

    # ------------------------------------------------------------------
    # Exposure Tab
    # ------------------------------------------------------------------
    def _build_exposure_tab(self) -> None:
        """Build the exposure table tab."""
        frame = self.exposure_frame

        # Camera info
        cam_group = ttk.LabelFrame(frame, text="Camera Settings", padding=10)
        cam_group.pack(fill=tk.X, pady=5)

        cam = self.config["camera"]
        ttk.Label(cam_group, text=f"Crop factor: {cam['crop_factor']}").grid(
            row=0, column=0, sticky=tk.W, padx=10
        )
        ttk.Label(cam_group, text=f"Pixel pitch: {cam['pixel_pitch_um']} \u03bcm").grid(
            row=0, column=1, sticky=tk.W, padx=10
        )
        ttk.Label(cam_group, text=f"Aperture: f/{cam['aperture']}").grid(
            row=0, column=2, sticky=tk.W, padx=10
        )

        fl_cfg = self.config["focal_lengths"]
        ttk.Label(
            cam_group,
            text=f"Focal range: {fl_cfg['min_mm']}–{fl_cfg['max_mm']} mm "
            f"(step {fl_cfg['step_mm']})",
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=10, pady=3)

        # Treeview table
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("focal_length", "rule_500", "npf_rule")
        self.exposure_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=15
        )
        self.exposure_tree.heading("focal_length", text="Focal Length (mm)")
        self.exposure_tree.heading("rule_500", text="500 Rule (sec)")
        self.exposure_tree.heading("npf_rule", text="NPF Rule (sec)")
        self.exposure_tree.column("focal_length", width=120, anchor=tk.CENTER)
        self.exposure_tree.column("rule_500", width=120, anchor=tk.CENTER)
        self.exposure_tree.column("npf_rule", width=120, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self.exposure_tree.yview
        )
        self.exposure_tree.configure(yscrollcommand=scrollbar.set)
        self.exposure_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(
            frame, text="Export Exposure Table (CSV)", command=self._export_exposure_csv
        ).pack(pady=5)

    # ------------------------------------------------------------------
    # Drift Simulation Tab
    # ------------------------------------------------------------------
    def _build_drift_tab(self) -> None:
        """Build the drift simulation tab."""
        frame = self.drift_frame

        # Parameters
        param_group = ttk.LabelFrame(frame, text="Simulation Parameters", padding=10)
        param_group.pack(fill=tk.X, pady=5)

        sim = self.config["alignment_simulation"]
        self.var_mis_alt = tk.DoubleVar(value=sim["misalignment_alt_arcsec"])
        self.var_mis_az = tk.DoubleVar(value=sim["misalignment_az_arcsec"])
        self.var_target_dec = tk.DoubleVar(value=sim["target_declination_deg"])
        self.var_duration = tk.DoubleVar(value=sim["simulation_duration_sec"])

        ttk.Label(param_group, text="Alt misalignment (arcsec):").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )
        ttk.Entry(param_group, textvariable=self.var_mis_alt, width=10).grid(
            row=0, column=1, padx=5
        )
        ttk.Label(param_group, text="Az misalignment (arcsec):").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )
        ttk.Entry(param_group, textvariable=self.var_mis_az, width=10).grid(
            row=0, column=3, padx=5
        )
        ttk.Label(param_group, text="Target Dec (°):").grid(
            row=1, column=0, sticky=tk.W, padx=5
        )
        ttk.Entry(param_group, textvariable=self.var_target_dec, width=10).grid(
            row=1, column=1, padx=5
        )
        ttk.Label(param_group, text="Duration (sec):").grid(
            row=1, column=2, sticky=tk.W, padx=5
        )
        ttk.Entry(param_group, textvariable=self.var_duration, width=10).grid(
            row=1, column=3, padx=5
        )

        ttk.Button(
            param_group, text="Run Simulation", command=self._run_drift_simulation
        ).grid(row=2, column=0, columnspan=4, pady=10)

        # Matplotlib figure embedded in tkinter
        self.drift_fig, self.drift_ax = plt.subplots(figsize=(8, 3.5))
        self.drift_canvas = FigureCanvasTkAgg(self.drift_fig, master=frame)
        self.drift_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

        self.lbl_max_drift = ttk.Label(
            frame, text="Max total drift: —", font=("Segoe UI", 10, "bold")
        )
        self.lbl_max_drift.pack(pady=3)

    # ------------------------------------------------------------------
    # Servo Export Tab
    # ------------------------------------------------------------------
    def _build_servo_tab(self) -> None:
        """Build the servo export tab."""
        frame = self.servo_frame

        info_group = ttk.LabelFrame(frame, text="Servo Tracking Settings", padding=10)
        info_group.pack(fill=tk.X, pady=5)

        servo_cfg = self.config["servo"]
        self.var_servo_duration = tk.DoubleVar(value=servo_cfg["duration_sec"])
        self.var_servo_step = tk.DoubleVar(value=servo_cfg["time_step_sec"])
        self.var_servo_format = tk.StringVar(value=servo_cfg["output_format"])

        ttk.Label(info_group, text="Duration (sec):").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )
        ttk.Entry(info_group, textvariable=self.var_servo_duration, width=10).grid(
            row=0, column=1, padx=5
        )
        ttk.Label(info_group, text="Time step (sec):").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )
        ttk.Entry(info_group, textvariable=self.var_servo_step, width=10).grid(
            row=0, column=3, padx=5
        )
        ttk.Label(info_group, text="Format:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Combobox(
            info_group,
            textvariable=self.var_servo_format,
            values=["json", "csv"],
            state="readonly",
            width=8,
        ).grid(row=1, column=1, padx=5)

        # Info display
        self.servo_info_text = tk.Text(
            frame, height=10, width=80, state=tk.DISABLED, wrap=tk.WORD
        )
        self.servo_info_text.pack(fill=tk.BOTH, expand=True, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            btn_frame, text="Generate Servo Data", command=self._generate_servo
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            btn_frame, text="Export Servo Data", command=self._export_servo
        ).pack(side=tk.LEFT, padx=5)

    # ------------------------------------------------------------------
    # Calculation logic
    # ------------------------------------------------------------------
    def _run_calculations(self) -> None:
        """Run all calculations and update the UI."""
        cfg = self.config

        # Polar alignment
        obs = cfg["observer"]
        star = cfg["target_star"]
        self.alignment_data = compute_polar_alignment(
            lat=obs["latitude"],
            lon=obs["longitude"],
            elevation=obs["elevation"],
            star_ra=star["ra"],
            star_dec=star["dec"],
        )

        alt = self.alignment_data["altitude"]
        az = self.alignment_data["azimuth"]
        time_utc = self.alignment_data["time_utc"]

        # Magnetic declination
        mag_cfg = cfg["magnetic_declination"]
        if mag_cfg["mode"] == "auto":
            self.mag_declination = get_magnetic_declination(
                lat=obs["latitude"],
                lon=obs["longitude"],
                elevation_m=obs["elevation"],
            )
        else:
            self.mag_declination = mag_cfg["manual_value_deg"]

        compass_az = None
        if self.mag_declination is not None:
            compass_az = correct_azimuth_for_magnetic(az, self.mag_declination)

        # Update alignment tab labels
        self.lbl_pole_alt.config(text=f"Celestial pole altitude: {alt:.2f}°")
        self.lbl_pole_az.config(text=f"Celestial pole azimuth (true N): {az:.2f}°")
        self.lbl_time.config(text=f"Computed at: {time_utc}")

        if self.mag_declination is not None:
            self.lbl_mag_dec.config(
                text=f"Magnetic declination: {self.mag_declination:+.2f}° "
                f"({'East' if self.mag_declination >= 0 else 'West'})"
            )
        else:
            self.lbl_mag_dec.config(text="Magnetic declination: N/A (install pyIGRF)")

        if compass_az is not None:
            self.lbl_compass_az.config(
                text=f"Compass reading (magnetic N): {compass_az:.2f}°"
            )
            self.lbl_orient.config(
                text=f"Orient compass to: {compass_az:.2f}° (magnetic)"
            )
        else:
            self.lbl_compass_az.config(text="Compass reading: N/A")
            self.lbl_orient.config(
                text=f"Orient towards: {az:.2f}° (true N, no mag correction)"
            )

        self.lbl_tilt.config(text=f"Tilt polar axis to: {alt:.2f}° from horizontal")

        # Exposure table
        fl_cfg = cfg["focal_lengths"]
        focal_lengths = list(
            range(fl_cfg["min_mm"], fl_cfg["max_mm"] + 1, fl_cfg["step_mm"])
        )

        cam = cfg["camera"]
        self.exposure_table = compute_exposure_table(
            focal_lengths=focal_lengths,
            crop_factor=cam["crop_factor"],
            pixel_pitch_um=cam["pixel_pitch_um"],
            aperture=cam["aperture"],
        )

        # Update exposure treeview
        for row in self.exposure_tree.get_children():
            self.exposure_tree.delete(row)
        for entry in self.exposure_table:
            npf_str = (
                f"{entry['max_exposure_npf']:.2f}" if entry["max_exposure_npf"] else "—"
            )
            self.exposure_tree.insert(
                "",
                tk.END,
                values=(
                    f"{entry['focal_length_mm']}",
                    f"{entry['max_exposure_500']:.2f}",
                    npf_str,
                ),
            )

        # Run drift simulation with default params
        self._run_drift_simulation()

    def _run_drift_simulation(self) -> None:
        """Run drift simulation and update the plot."""
        obs = self.config["observer"]
        try:
            self.drift_data = simulate_tracking_drift(
                misalignment_alt_arcsec=self.var_mis_alt.get(),
                misalignment_az_arcsec=self.var_mis_az.get(),
                observer_lat_deg=obs["latitude"],
                target_dec_deg=self.var_target_dec.get(),
                duration_sec=self.var_duration.get(),
                time_step_sec=self.config["alignment_simulation"]["time_step_sec"],
            )
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
            return

        # Update plot
        ax = self.drift_ax
        ax.clear()
        t_min = self.drift_data["time_sec"] / 60.0
        ax.plot(
            t_min, self.drift_data["drift_dec_arcsec"], label="Dec drift", color="blue"
        )
        ax.plot(
            t_min, self.drift_data["drift_ra_arcsec"], label="RA drift", color="green"
        )
        ax.plot(
            t_min,
            self.drift_data["total_drift_arcsec"],
            label="Total drift",
            color="red",
            linewidth=2,
        )
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Drift (arcseconds)")
        ax.set_title("Tracking Drift from Polar Misalignment")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        self.drift_fig.tight_layout()
        self.drift_canvas.draw()

        max_drift = np.max(self.drift_data["total_drift_arcsec"])
        self.lbl_max_drift.config(
            text=f"Max total drift: {max_drift:.2f} arcsec "
            f"({max_drift / 60:.3f} arcmin)"
        )

    def _generate_servo(self) -> None:
        """Generate servo tracking data and display summary."""
        try:
            self.servo_data = generate_servo_tracking_data(
                duration_sec=self.var_servo_duration.get(),
                time_step_sec=self.var_servo_step.get(),
                sidereal_rate=True,
            )
        except Exception as e:
            messagebox.showerror("Servo Generation Error", str(e))
            return

        # Display summary
        n_points = len(self.servo_data["time_sec"])
        total_angle = self.servo_data["angle_deg"][-1]
        step_angle = self.servo_data["step_angle_deg"]
        avg_velocity = np.mean(self.servo_data["angular_velocity_deg_per_sec"])

        info = (
            f"Servo Tracking Profile Generated\n"
            f"{'=' * 40}\n"
            f"Duration: {self.var_servo_duration.get():.2f} sec "
            f"({self.var_servo_duration.get() / 3600:.2f} hours)\n"
            f"Time step: {self.var_servo_step.get():.2f} sec\n"
            f"Data points: {n_points}\n"
            f"Total rotation: {total_angle:.4f}°\n"
            f"Step angle: {step_angle:.6f}° per step\n"
            f"Avg velocity: {avg_velocity:.6f} °/sec\n"
            f"Sidereal rate: {avg_velocity * 3600:.4f} °/hour\n"
            f"\nReady to export as {self.var_servo_format.get().upper()}."
        )

        self.servo_info_text.config(state=tk.NORMAL)
        self.servo_info_text.delete("1.0", tk.END)
        self.servo_info_text.insert("1.0", info)
        self.servo_info_text.config(state=tk.DISABLED)

    def _export_servo(self) -> None:
        """Export servo data to file."""
        if not self.servo_data:
            messagebox.showwarning("No Data", "Generate servo data first.")
            return

        fmt = self.var_servo_format.get()
        ext = ".json" if fmt == "json" else ".csv"
        filepath = filedialog.asksaveasfilename(
            title="Save Servo Tracking Data",
            defaultextension=ext,
            filetypes=[(f"{fmt.upper()} files", f"*{ext}"), ("All files", "*.*")],
            initialfile=f"servo_tracking{ext}",
        )
        if not filepath:
            return

        metadata = {
            "observer_latitude": self.config["observer"]["latitude"],
            "observer_longitude": self.config["observer"]["longitude"],
            "polar_altitude_deg": self.alignment_data.get("altitude"),
            "polar_azimuth_deg": self.alignment_data.get("azimuth"),
            "magnetic_declination_deg": self.mag_declination,
            "sidereal_rate_deg_per_sec": float(
                self.servo_data["step_angle_deg"] / self.var_servo_step.get()
            )
            if self.var_servo_step.get() > 0
            else 0,
        }

        try:
            if fmt == "json":
                export_servo_to_json(self.servo_data, filepath, metadata=metadata)
            else:
                export_servo_to_csv(self.servo_data, filepath, metadata=metadata)
            messagebox.showinfo("Export Complete", f"Saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _export_alignment(self) -> None:
        """Export alignment report."""
        filepath = filedialog.asksaveasfilename(
            title="Save Alignment Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="alignment_report.json",
        )
        if not filepath:
            return

        obs = self.config["observer"]
        report_data = {
            "polar_altitude_deg": self.alignment_data.get("altitude"),
            "polar_azimuth_deg": self.alignment_data.get("azimuth"),
            "time_utc": self.alignment_data.get("time_utc"),
            "latitude": obs["latitude"],
            "longitude": obs["longitude"],
            "elevation_m": obs["elevation"],
            "magnetic_declination_deg": self.mag_declination,
            "compass_azimuth_deg": correct_azimuth_for_magnetic(
                self.alignment_data.get("azimuth", 0), self.mag_declination or 0.0
            ),
            "exposure_table": self.exposure_table,
        }

        try:
            export_alignment_report(report_data, filepath)
            messagebox.showinfo("Export Complete", f"Saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _export_exposure_csv(self) -> None:
        """Export exposure table to CSV."""
        filepath = filedialog.asksaveasfilename(
            title="Save Exposure Table",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="exposure_table.csv",
        )
        if not filepath:
            return

        try:
            export_exposure_table_csv(self.exposure_table, filepath)
            messagebox.showinfo("Export Complete", f"Saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _show_3d_plot(self) -> None:
        """Open the 3D polar alignment visualization in a new window."""
        alt = self.alignment_data.get("altitude", 0)
        az = self.alignment_data.get("azimuth", 0)
        lat = self.config["observer"]["latitude"]
        star_name = self.config["target_star"]["name"]

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        # --- Earth globe with correct rotation axis ---
        globe_radius = 0.3
        earth_axis_alt = np.radians(lat)
        earth_ax_x = 0.0
        earth_ax_y = np.cos(earth_axis_alt)
        earth_ax_z = np.sin(earth_axis_alt)
        earth_axis = np.array([earth_ax_x, earth_ax_y, earth_ax_z])
        earth_axis = earth_axis / np.linalg.norm(earth_axis)

        # Rotation matrix to align sphere poles with Earth axis
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
        mid_alt = np.radians(alt) / 2
        ax.text(
            0.72 * np.cos(mid_alt) * np.sin(np.radians(az)),
            0.72 * np.cos(mid_alt) * np.cos(np.radians(az)),
            0.72 * np.sin(mid_alt),
            f"Alt = {alt:.2f}\u00b0",
            color="orange",
            fontsize=10,
            fontweight="bold",
        )

        # Azimuth angle arc
        az_angles = np.linspace(0, np.radians(az), arc_pts)
        arc_az_x = 0.5 * np.sin(az_angles)
        arc_az_y = 0.5 * np.cos(az_angles)
        arc_az_z = np.zeros(arc_pts)
        ax.plot(arc_az_x, arc_az_y, arc_az_z, color="purple", linewidth=2.5)
        mid_az = np.radians(az) / 2
        ax.text(
            0.6 * np.sin(mid_az),
            0.6 * np.cos(mid_az),
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

    def run(self) -> None:
        """Start the tkinter event loop."""
        self.root.mainloop()
