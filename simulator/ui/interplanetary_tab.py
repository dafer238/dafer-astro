"""
Interplanetary / multi-body tab for the orbital mechanics simulator.
Provides a macro-level view for transfer orbits between planets, Earth-Moon transfers,
and multi-spacecraft simulation with solar system context.
"""

from __future__ import annotations
import dearpygui.dearpygui as dpg
import numpy as np
import time
from datetime import datetime, timezone

from simulator.core.ephemeris import (
    get_solar_system_state,
    datetime_to_jd,
    planet_ecliptic_position,
    _PLANET_MU,
    _PLANET_RADIUS_KM,
    _PLANET_COLORS,
)
from simulator.core.spacecraft import Spacecraft, MultiBodyTrajectory
from simulator.core.state import OrbitalElements, StateVector
from simulator.core.conversions import coe_to_state, orbital_period
from simulator.core.constants import AU, MU_EARTH, R_EARTH
from simulator.sim.multibody import MultiBodyScenario, MultiBodyEngine
from simulator.sim.scenario import PerturbationConfig
from simulator.render.viewport3d import Viewport3D
from simulator.render.projection import perspective_matrix, project_points
from simulator.sim.trajectory import TrajectoryData


# Default spacecraft colors for up to 8 craft
_SC_COLORS = [
    (0, 255, 0),
    (255, 80, 80),
    (80, 180, 255),
    (255, 255, 0),
    (255, 128, 0),
    (200, 80, 255),
    (0, 255, 200),
    (255, 200, 200),
]


class InterplanetaryTab:
    """Interplanetary transfer and multi-body simulation tab."""

    def __init__(self):
        self.engine = MultiBodyEngine()
        self.viewport: Viewport3D | None = None
        self.result: MultiBodyTrajectory | None = None
        self._spacecraft_list: list[dict] = []  # [{name, color, central_body, coe_or_state}]
        self._console_lines: list[str] = []
        self._playing = False
        self._play_speed = 100.0
        self._current_time = 0.0
        self._last_frame_time = 0.0
        self._view_scale = "solar_system"  # solar_system, earth_system, planet

    def build(self, parent):
        """Build the interplanetary tab UI inside the given parent."""
        with dpg.group(horizontal=True, parent=parent, tag="ipt_content_group"):
            self._build_left_panel()
            self._build_center_panel()
            self._build_right_panel()

    def _build_left_panel(self):
        with dpg.child_window(width=300, tag="ipt_left_panel"):
            dpg.add_text("INTERPLANETARY CONFIG", color=(255, 180, 0))
            dpg.add_separator()

            # Epoch selection
            dpg.add_text("Simulation Epoch:", color=(150, 150, 150))
            with dpg.group(horizontal=True):
                dpg.add_input_int(label="Y", tag="ipt_year", default_value=2024, width=60)
                dpg.add_input_int(
                    label="M",
                    tag="ipt_month",
                    default_value=6,
                    width=40,
                    min_value=1,
                    max_value=12,
                    min_clamped=True,
                    max_clamped=True,
                )
                dpg.add_input_int(
                    label="D",
                    tag="ipt_day",
                    default_value=15,
                    width=40,
                    min_value=1,
                    max_value=31,
                    min_clamped=True,
                    max_clamped=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="H",
                    tag="ipt_hour",
                    default_value=12,
                    width=40,
                    min_value=0,
                    max_value=23,
                    min_clamped=True,
                    max_clamped=True,
                )
                dpg.add_input_int(
                    label="Min",
                    tag="ipt_minute",
                    default_value=0,
                    width=40,
                    min_value=0,
                    max_value=59,
                    min_clamped=True,
                    max_clamped=True,
                )

            dpg.add_spacer(height=4)
            dpg.add_input_float(
                label="Duration (days)",
                tag="ipt_duration_days",
                default_value=180.0,
                width=140,
                step=10,
                format="%.1f",
            )

            dpg.add_spacer(height=6)
            dpg.add_text("Central Body:", color=(150, 150, 150))
            dpg.add_combo(
                ["Sun", "Earth", "Mars", "Jupiter"],
                default_value="Earth",
                tag="ipt_central_body",
                width=140,
            )

            dpg.add_spacer(height=4)
            dpg.add_text("Perturbing Bodies:", color=(150, 150, 150))
            dpg.add_checkbox(label="Moon", tag="ipt_pert_moon", default_value=True)
            dpg.add_checkbox(label="Sun", tag="ipt_pert_sun", default_value=True)
            dpg.add_checkbox(label="Jupiter", tag="ipt_pert_jupiter", default_value=False)
            dpg.add_checkbox(label="J2 (Earth)", tag="ipt_pert_j2", default_value=True)

            dpg.add_spacer(height=8)
            dpg.add_text("SPACECRAFT", color=(0, 255, 200))
            dpg.add_separator()

            # Spacecraft definition
            dpg.add_input_text(label="Name", tag="ipt_sc_name", default_value="Craft-1", width=140)
            dpg.add_combo(
                ["Green", "Red", "Blue", "Yellow", "Orange", "Purple", "Cyan", "White"],
                default_value="Green",
                tag="ipt_sc_color",
                width=140,
            )

            dpg.add_spacer(height=2)
            dpg.add_text("Initial orbit (around central body):", color=(150, 150, 150))
            dpg.add_input_float(
                label="a (km)",
                tag="ipt_sc_a",
                default_value=6778.0,
                width=140,
                step=100,
                format="%.1f",
            )
            dpg.add_input_float(
                label="e",
                tag="ipt_sc_e",
                default_value=0.0001,
                width=140,
                step=0.001,
                format="%.5f",
            )
            dpg.add_slider_float(
                label="i (deg)",
                tag="ipt_sc_i",
                default_value=28.5,
                min_value=0,
                max_value=180,
                width=140,
                format="%.2f",
            )
            dpg.add_slider_float(
                label="RAAN",
                tag="ipt_sc_raan",
                default_value=0,
                min_value=0,
                max_value=360,
                width=140,
                format="%.2f",
            )
            dpg.add_slider_float(
                label="omega",
                tag="ipt_sc_omega",
                default_value=0,
                min_value=0,
                max_value=360,
                width=140,
                format="%.2f",
            )
            dpg.add_slider_float(
                label="theta",
                tag="ipt_sc_theta",
                default_value=0,
                min_value=0,
                max_value=360,
                width=140,
                format="%.2f",
            )

            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add SC", callback=self._add_spacecraft, width=80)
                dpg.add_button(label="Clear All", callback=self._clear_spacecraft, width=80)

            dpg.add_spacer(height=2)
            dpg.add_text("Presets:", color=(150, 150, 150))
            with dpg.group(horizontal=True):
                dpg.add_button(label="ISS+Hubble", callback=self._preset_iss_hubble, width=90)
                dpg.add_button(label="GTO", callback=self._preset_gto, width=50)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Moon Transfer", callback=self._preset_moon_transfer, width=105
                )
                dpg.add_button(label="LEO pair", callback=self._preset_leo_pair, width=75)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Mars Hohmann", callback=self._preset_mars_transfer, width=105)
                dpg.add_button(label="Venus", callback=self._preset_venus_transfer, width=60)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="GPS Const.", callback=self._preset_gps_constellation, width=90
                )
                dpg.add_button(label="Asteroid", callback=self._preset_asteroid, width=70)

            dpg.add_spacer(height=4)
            dpg.add_input_text(
                tag="ipt_sc_list_text",
                multiline=True,
                readonly=True,
                height=100,
                width=-1,
                default_value="No spacecraft defined.",
            )

            dpg.add_spacer(height=8)
            dpg.add_button(
                label="SIMULATE",
                callback=self._on_simulate,
                width=-1,
                height=30,
                tag="ipt_simulate_btn",
            )

    def _build_center_panel(self):
        with dpg.child_window(
            width=-1, tag="ipt_center_panel", no_scrollbar=True, no_scroll_with_mouse=True
        ):
            dpg.add_text("MULTI-BODY VIEWPORT", color=(255, 180, 0))
            dpg.add_separator()

            self.viewport = Viewport3D("ipt_viewport_drawlist", width=1100, height=520)
            self.viewport.create("ipt_center_panel")

            dpg.add_spacer(height=4)
            dpg.add_text("View Presets:", color=(150, 150, 150))
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="ISO", callback=lambda: self.viewport.set_view_isometric(), width=45
                )
                dpg.add_button(label="XY", callback=lambda: self.viewport.set_view_xy(), width=45)
                dpg.add_button(label="XZ", callback=lambda: self.viewport.set_view_xz(), width=45)
                dpg.add_button(label="Earth", callback=self._view_earth, width=55)
                dpg.add_button(label="Moon", callback=self._view_moon_scale, width=55)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Solar System", callback=self._view_solar_system, width=95)
                dpg.add_button(label="Mars", callback=self._view_mars, width=50)
                dpg.add_button(label="Jupiter", callback=self._view_jupiter, width=60)

            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Play", tag="ipt_play_btn", callback=self._toggle_play, width=50
                )
                dpg.add_button(label="Reset", callback=self._reset_time, width=50)
                dpg.add_slider_float(
                    label="##ipt_time",
                    tag="ipt_time_slider",
                    default_value=0,
                    min_value=0,
                    max_value=1,
                    width=300,
                    callback=self._on_time_scrub,
                    format="%.3f",
                )
                dpg.add_combo(
                    ["1x", "10x", "100x", "500x", "1000x", "5000x"],
                    default_value="100x",
                    tag="ipt_speed_combo",
                    callback=self._on_speed_change,
                    width=70,
                )

            dpg.add_text("t = 0.0 s", tag="ipt_time_display", color=(150, 150, 150))

    def _build_right_panel(self):
        with dpg.child_window(width=200, tag="ipt_right_panel"):
            dpg.add_text("MULTI-BODY TELEMETRY", color=(0, 255, 200))
            dpg.add_separator()
            dpg.add_input_text(
                tag="ipt_telemetry_text",
                multiline=True,
                readonly=True,
                height=200,
                width=-1,
                default_value="No simulation running.",
            )

            dpg.add_spacer(height=8)
            dpg.add_text("RELATIVE DISTANCES", color=(255, 200, 80))
            dpg.add_separator()
            dpg.add_input_text(
                tag="ipt_distances_text",
                multiline=True,
                readonly=True,
                height=120,
                width=-1,
                default_value="---",
            )

            dpg.add_spacer(height=8)
            dpg.add_text("SOLAR SYSTEM", color=(255, 223, 0))
            dpg.add_separator()
            dpg.add_button(label="Show Planets", callback=self._show_planet_info, width=-1)
            dpg.add_input_text(
                tag="ipt_planets_text",
                multiline=True,
                readonly=True,
                height=160,
                width=-1,
                default_value="Click 'Show Planets'",
            )

            dpg.add_spacer(height=8)
            dpg.add_text("CONSOLE", color=(150, 150, 150))
            dpg.add_separator()
            dpg.add_input_text(
                tag="ipt_console_text",
                multiline=True,
                readonly=True,
                height=80,
                width=-1,
                default_value="",
            )

    def _log(self, msg: str):
        self._console_lines.append(f"> {msg}")
        if len(self._console_lines) > 30:
            self._console_lines = self._console_lines[-30:]
        if dpg.does_item_exist("ipt_console_text"):
            dpg.set_value("ipt_console_text", "\n".join(self._console_lines))

    def _get_epoch(self) -> datetime:
        return datetime(
            year=dpg.get_value("ipt_year"),
            month=dpg.get_value("ipt_month"),
            day=dpg.get_value("ipt_day"),
            hour=dpg.get_value("ipt_hour"),
            minute=dpg.get_value("ipt_minute"),
            tzinfo=timezone.utc,
        )

    def _color_name_to_rgb(self, name: str) -> tuple[int, int, int]:
        mapping = {
            "Green": (0, 255, 0),
            "Red": (255, 80, 80),
            "Blue": (80, 180, 255),
            "Yellow": (255, 255, 0),
            "Orange": (255, 128, 0),
            "Purple": (200, 80, 255),
            "Cyan": (0, 255, 200),
            "White": (255, 255, 255),
        }
        return mapping.get(name, (255, 255, 255))

    def _add_spacecraft(self):
        name = dpg.get_value("ipt_sc_name")
        color = self._color_name_to_rgb(dpg.get_value("ipt_sc_color"))
        coe = OrbitalElements(
            a=dpg.get_value("ipt_sc_a"),
            e=dpg.get_value("ipt_sc_e"),
            i=dpg.get_value("ipt_sc_i"),
            raan=dpg.get_value("ipt_sc_raan"),
            omega=dpg.get_value("ipt_sc_omega"),
            theta=dpg.get_value("ipt_sc_theta"),
        )
        # Check for duplicate names
        for sc in self._spacecraft_list:
            if sc["name"] == name:
                name = name + f"_{len(self._spacecraft_list) + 1}"
                break

        self._spacecraft_list.append({"name": name, "color": color, "coe": coe})
        self._refresh_sc_list()
        self._log(f"Added spacecraft: {name}")

        # Auto-increment name
        base = "Craft-"
        dpg.set_value("ipt_sc_name", f"{base}{len(self._spacecraft_list) + 1}")

    def _clear_spacecraft(self):
        self._spacecraft_list.clear()
        self._refresh_sc_list()
        self._log("Cleared all spacecraft.")

    def _refresh_sc_list(self):
        if not self._spacecraft_list:
            dpg.set_value("ipt_sc_list_text", "No spacecraft defined.")
            return
        lines = []
        for i, sc in enumerate(self._spacecraft_list, 1):
            coe = sc["coe"]
            lines.append(f"{i}. {sc['name']} a={coe.a:.0f} e={coe.e:.4f} i={coe.i:.1f}")
        dpg.set_value("ipt_sc_list_text", "\n".join(lines))

    def _on_simulate(self):
        if not self._spacecraft_list:
            self._log("Add at least one spacecraft first.")
            return

        epoch = self._get_epoch()
        duration_days = dpg.get_value("ipt_duration_days")
        central_body = dpg.get_value("ipt_central_body")

        # Build perturbing bodies list
        perturbers = []
        if dpg.get_value("ipt_pert_moon"):
            perturbers.append("Moon")
        if dpg.get_value("ipt_pert_sun") and central_body != "Sun":
            perturbers.append("Sun")
        if dpg.get_value("ipt_pert_jupiter") and central_body != "Jupiter":
            perturbers.append("Jupiter")

        # Build perturbation config
        pert_cfg = PerturbationConfig(
            j2_enabled=dpg.get_value("ipt_pert_j2") and central_body == "Earth",
            drag_enabled=False,
        )

        # Build spacecraft objects
        spacecraft = []
        for sc_data in self._spacecraft_list:
            sc = Spacecraft(
                name=sc_data["name"],
                color=sc_data["color"],
                initial_coe=sc_data["coe"],
            )
            spacecraft.append(sc)

        scenario = MultiBodyScenario(
            spacecraft=spacecraft,
            central_body=central_body,
            perturbing_bodies=perturbers,
            epoch=epoch,
            duration_seconds=duration_days * 86400.0,
            perturbations=pert_cfg,
        )

        self._log(f"Simulating {len(spacecraft)} craft, {duration_days:.0f} days...")
        self.result = self.engine.compute(scenario)

        if self.result is None:
            self._log(f"ERROR: {self.engine.error}")
            return

        self._log(f"Done! {len(self.result.t)} time points.")

        # Set trajectories on viewport
        trajs = []
        for sc_name, sc_color in zip(self.result.spacecraft_names, self.result.spacecraft_colors):
            traj = self.result.get_trajectory(sc_name)
            trajs.append((traj, sc_color, sc_name))

        # Use first spacecraft's trajectory as the primary
        first_traj = trajs[0][0]
        body_colors = {
            "Earth": (28, 88, 170),
            "Moon": (140, 140, 140),
            "Mars": (180, 80, 50),
            "Sun": (255, 200, 0),
            "Jupiter": (180, 150, 100),
        }
        bc = body_colors.get(central_body, (100, 100, 200))
        self.viewport.set_trajectory(
            first_traj,
            _PLANET_RADIUS_KM.get(central_body, R_EARTH),
            body_name=central_body,
            body_color=bc,
        )
        self.viewport.set_multi_trajectories(trajs[1:] if len(trajs) > 1 else [])

        # Render celestial bodies in viewport
        self._update_celestial_bodies_in_viewport(epoch, central_body)

        self._current_time = 0.0
        dpg.configure_item("ipt_time_slider", max_value=self.result.t[-1])
        self._update_telemetry()

    def _toggle_play(self):
        if self.result is None:
            return
        self._playing = not self._playing
        dpg.configure_item("ipt_play_btn", label="Pause" if self._playing else "Play")

    def _reset_time(self):
        self._current_time = 0.0
        dpg.set_value("ipt_time_slider", 0.0)
        if self.viewport:
            self.viewport.set_time(0.0)
        self._update_telemetry()

    def _on_time_scrub(self, sender, app_data):
        self._current_time = app_data
        if self.viewport:
            self.viewport.set_time(self._current_time)
        self._update_telemetry()

    def _on_speed_change(self, sender, app_data):
        speed_map = {"1x": 1, "10x": 10, "100x": 100, "500x": 500, "1000x": 1000, "5000x": 5000}
        self._play_speed = speed_map.get(app_data, 100)

    def _update_telemetry(self):
        if self.result is None:
            return

        t = self._current_time
        lines = []
        central_radius = _PLANET_RADIUS_KM.get(dpg.get_value("ipt_central_body"), R_EARTH)
        for name in self.result.spacecraft_names:
            traj = self.result.get_trajectory(name)
            r, v = traj.interpolate(t)
            alt = np.linalg.norm(r) - central_radius
            vel = np.linalg.norm(v)
            lines.append(f"{name}:")
            lines.append(f"  Alt: {alt:.1f} km  V: {vel:.4f} km/s")

        dpg.set_value("ipt_telemetry_text", "\n".join(lines))

        # Relative distances
        names = self.result.spacecraft_names
        if len(names) >= 2:
            dist_lines = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ti = self.result.get_trajectory(names[i])
                    tj = self.result.get_trajectory(names[j])
                    ri, _ = ti.interpolate(t)
                    rj, _ = tj.interpolate(t)
                    d = np.linalg.norm(ri - rj)
                    dist_lines.append(f"{names[i]}<->{names[j]}: {d:.2f} km")
            dpg.set_value("ipt_distances_text", "\n".join(dist_lines))

        # Time display
        days = t / 86400.0
        dpg.set_value("ipt_time_display", f"t = {t:.0f} s ({days:.2f} days)")

    def _show_planet_info(self):
        epoch = self._get_epoch()
        jd = datetime_to_jd(epoch)
        ss = get_solar_system_state(jd)
        lines = [f"Epoch: {epoch.strftime('%Y-%m-%d %H:%M')} UTC"]
        for name in ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]:
            dist_au = np.linalg.norm(ss[name]["position"]) / AU
            lines.append(f"{name:10s} {dist_au:.4f} AU")
        moon_dist = np.linalg.norm(ss["Moon"]["position"] - ss["Earth"]["position"])
        lines.append(f"{'Moon':10s} {moon_dist:.0f} km from Earth")
        dpg.set_value("ipt_planets_text", "\n".join(lines))

    def _view_earth(self):
        """Zoom to Earth-orbit scale, centered on origin."""
        if self.viewport:
            self.viewport.camera.target = np.zeros(3)
            self.viewport.camera.distance = 25000.0
            self.viewport.set_view_isometric()

    def _view_moon_scale(self):
        """Zoom to Moon — center camera on Moon's position."""
        if self.viewport:
            # Get Moon position from ephemeris at current epoch
            try:
                from simulator.core.ephemeris import moon_geocentric_position, datetime_to_jd

                epoch = self._get_epoch()
                jd = datetime_to_jd(epoch)
                moon_pos = moon_geocentric_position(jd)
                self.viewport.camera.target = moon_pos
                self.viewport.camera.distance = 50000.0  # Close-up of Moon
            except Exception:
                self.viewport.camera.target = np.array([384400.0, 0.0, 0.0])
                self.viewport.camera.distance = 50000.0
            self.viewport.camera._dirty = True
            self.viewport._needs_redraw = True

    def _view_solar_system(self):
        """Zoom out to solar system scale, show all planets around Sun."""
        if self.viewport:
            self.viewport.camera.target = np.zeros(3)
            self.viewport.camera.distance = 3e9  # ~20 AU
            self.viewport.set_view_xy()
            # Load planet positions/orbits around Sun
            self._update_celestial_bodies_in_viewport(self._get_epoch(), "Sun")

    def _view_mars(self):
        """Mars orbit scale."""
        if self.viewport:
            self.viewport.camera.target = np.zeros(3)
            self.viewport.camera.distance = 4e8  # Mars orbit ~228M km
            self.viewport.set_view_xy()
            self._update_celestial_bodies_in_viewport(self._get_epoch(), "Sun")

    def _view_jupiter(self):
        """Jupiter orbit scale."""
        if self.viewport:
            self.viewport.camera.target = np.zeros(3)
            self.viewport.camera.distance = 1.2e9  # Jupiter orbit ~778M km
            self.viewport.set_view_xy()
            self._update_celestial_bodies_in_viewport(self._get_epoch(), "Sun")

    def _update_celestial_bodies_in_viewport(self, epoch, central_body_name: str):
        """Set visible celestial bodies relative to central body."""
        jd = datetime_to_jd(epoch)
        central_pos = planet_ecliptic_position(central_body_name, jd)

        bodies_to_show = []
        orbit_paths = []
        # Show relevant bodies depending on central body
        if central_body_name == "Earth":
            # Show Moon
            from simulator.core.ephemeris import moon_geocentric_position

            moon_pos = moon_geocentric_position(jd)
            bodies_to_show.append((moon_pos, 1737.4, (200, 200, 200), "Moon"))
            # Moon orbit path (sample over ~27 days)
            moon_pts = []
            for dt_days in np.linspace(0, 27.3, 120):
                mp = moon_geocentric_position(jd + dt_days)
                moon_pts.append(mp)
            orbit_paths.append((np.array(moon_pts), (200, 200, 200), "Moon orbit"))
        elif central_body_name == "Sun":
            # Show planets with orbit paths
            planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]
            orbital_periods_days = {
                "Mercury": 88,
                "Venus": 225,
                "Earth": 365,
                "Mars": 687,
                "Jupiter": 4333,
                "Saturn": 10759,
            }
            for pname in planets:
                pos = planet_ecliptic_position(pname, jd) - central_pos
                bodies_to_show.append((pos, _PLANET_RADIUS_KM[pname], _PLANET_COLORS[pname], pname))
                # Compute orbit path
                period = orbital_periods_days[pname]
                pts = []
                for dt_days in np.linspace(0, period, 180):
                    p = planet_ecliptic_position(pname, jd + dt_days) - central_pos
                    pts.append(p)
                orbit_paths.append((np.array(pts), _PLANET_COLORS[pname], pname))
        else:
            # Show Sun relative to this body
            sun_pos = -central_pos  # Sun at origin minus central body pos
            bodies_to_show.append((sun_pos, 695700.0, (255, 223, 0), "Sun"))

        self.viewport.set_extra_bodies(bodies_to_show)
        self.viewport.set_orbit_paths(orbit_paths)

    def frame_update(self, dt_frame: float):
        """Called each frame from the main app."""
        if self._playing and self.result is not None:
            self._current_time += dt_frame * self._play_speed
            if self._current_time > self.result.t[-1]:
                self._current_time = 0.0
            dpg.set_value("ipt_time_slider", self._current_time)
            if self.viewport:
                self.viewport.set_time(self._current_time)
            self._update_telemetry()

        if self.viewport:
            self.viewport.render()

    # --- Presets ---

    def _preset_iss_hubble(self):
        self._spacecraft_list = [
            {
                "name": "ISS",
                "color": (0, 255, 0),
                "coe": OrbitalElements(a=6779, e=0.0001, i=51.6, raan=0, omega=0, theta=0),
            },
            {
                "name": "Hubble",
                "color": (255, 80, 80),
                "coe": OrbitalElements(a=6917, e=0.0003, i=28.5, raan=45, omega=90, theta=180),
            },
        ]
        self._refresh_sc_list()
        self._log("Loaded ISS + Hubble preset")

    def _preset_gto(self):
        self._spacecraft_list = [
            {
                "name": "GTO-Sat",
                "color": (255, 255, 0),
                "coe": OrbitalElements(a=24400, e=0.73, i=7.0, raan=0, omega=180, theta=0),
            },
        ]
        self._refresh_sc_list()
        self._log("Loaded GTO preset")

    def _preset_moon_transfer(self):
        """Trans-lunar injection orbit."""
        self._spacecraft_list = [
            {
                "name": "TLI-Craft",
                "color": (80, 180, 255),
                "coe": OrbitalElements(a=200000, e=0.966, i=28.5, raan=0, omega=0, theta=0),
            },
        ]
        dpg.set_value("ipt_duration_days", 5.0)
        dpg.set_value("ipt_pert_moon", True)
        dpg.set_value("ipt_pert_sun", True)
        self._refresh_sc_list()
        self._log("Loaded Moon transfer preset")

    def _preset_leo_pair(self):
        self._spacecraft_list = [
            {
                "name": "Chaser",
                "color": (0, 255, 0),
                "coe": OrbitalElements(a=6778, e=0.0002, i=51.6, raan=30, omega=0, theta=0),
            },
            {
                "name": "Target",
                "color": (255, 80, 80),
                "coe": OrbitalElements(a=6778, e=0.0001, i=51.6, raan=30, omega=0, theta=5),
            },
        ]
        dpg.set_value("ipt_duration_days", 0.0625)  # ~1.5 hours
        self._refresh_sc_list()
        self._log("Loaded LEO rendezvous pair preset")

    def _preset_mars_transfer(self):
        """Earth-Mars Hohmann transfer orbit (heliocentric)."""
        self._spacecraft_list = [
            {
                "name": "Mars-Probe",
                "color": (0, 200, 255),
                "coe": OrbitalElements(
                    a=(AU + 1.524 * AU) / 2,  # Hohmann semi-major axis
                    e=(1.524 * AU - AU) / (1.524 * AU + AU),
                    i=1.85,
                    raan=49.6,
                    omega=286.5,
                    theta=0,
                ),
            },
        ]
        dpg.set_value("ipt_central_body", "Sun")
        dpg.set_value("ipt_duration_days", 260.0)  # ~8.5 months transfer
        self._refresh_sc_list()
        self._log("Loaded Mars Hohmann transfer (heliocentric)")

    def _preset_venus_transfer(self):
        """Earth-Venus Hohmann transfer orbit (heliocentric)."""
        r_venus = 0.723 * AU
        self._spacecraft_list = [
            {
                "name": "Venus-Probe",
                "color": (255, 200, 50),
                "coe": OrbitalElements(
                    a=(AU + r_venus) / 2,
                    e=(AU - r_venus) / (AU + r_venus),
                    i=3.4,
                    raan=76.7,
                    omega=55.0,
                    theta=180,
                ),
            },
        ]
        dpg.set_value("ipt_central_body", "Sun")
        dpg.set_value("ipt_duration_days", 146.0)  # ~4.8 months
        self._refresh_sc_list()
        self._log("Loaded Venus Hohmann transfer (heliocentric)")

    def _preset_gps_constellation(self):
        """GPS-like constellation: 4 spacecraft in 2 planes."""
        a_gps = R_EARTH + 20200.0
        self._spacecraft_list = [
            {
                "name": "GPS-A1",
                "color": (0, 255, 100),
                "coe": OrbitalElements(a=a_gps, e=0.0, i=55.0, raan=0, omega=0, theta=0),
            },
            {
                "name": "GPS-A2",
                "color": (0, 200, 100),
                "coe": OrbitalElements(a=a_gps, e=0.0, i=55.0, raan=0, omega=0, theta=90),
            },
            {
                "name": "GPS-B1",
                "color": (100, 100, 255),
                "coe": OrbitalElements(a=a_gps, e=0.0, i=55.0, raan=60, omega=0, theta=45),
            },
            {
                "name": "GPS-B2",
                "color": (80, 80, 220),
                "coe": OrbitalElements(a=a_gps, e=0.0, i=55.0, raan=60, omega=0, theta=135),
            },
        ]
        dpg.set_value("ipt_central_body", "Earth")
        dpg.set_value("ipt_duration_days", 1.0)  # 1 day
        self._refresh_sc_list()
        self._log("Loaded GPS constellation (4 SC, 2 planes)")

    def _preset_asteroid(self):
        """Near-Earth asteroid intercept (heliocentric, highly eccentric)."""
        self._spacecraft_list = [
            {
                "name": "Interceptor",
                "color": (255, 100, 0),
                "coe": OrbitalElements(
                    a=1.2 * AU,
                    e=0.35,
                    i=12.0,
                    raan=140,
                    omega=60,
                    theta=0,
                ),
            },
        ]
        dpg.set_value("ipt_central_body", "Sun")
        dpg.set_value("ipt_duration_days", 400.0)
        self._refresh_sc_list()
        self._log("Loaded asteroid intercept orbit (heliocentric)")
