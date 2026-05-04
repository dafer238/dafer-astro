import dearpygui.dearpygui as dpg
import numpy as np
import time
import os
import json

from simulator.ui.theme import create_theme, create_button_themes
from simulator.render.viewport3d import Viewport3D
from simulator.sim.engine import SimulationEngine
from simulator.sim.scenario import Scenario, PerturbationConfig, ManeuverEvent
from simulator.sim.trajectory import TrajectoryData
from simulator.core.state import OrbitalElements
from simulator.core.bodies import CelestialBody
from simulator.core.constants import R_EARTH, MU_EARTH
from simulator.core.conversions import (
    orbital_period,
    specific_energy,
    spec_ang_mom,
    state_to_coe,
    StateVector,
    vis_viva,
)
from simulator.physics.maneuvers import hohmann_dv, bielliptic_dv, plane_change_dv, combined_dv
from simulator.physics.perturbations import j2_raan_rate, j2_argp_rate
from simulator.physics.propulsion import ENGINES, engine_mdot
from simulator.core.conversions import coe_to_state
from simulator.ui.interplanetary_tab import InterplanetaryTab
from simulator.core.spacecraft import Spacecraft, MultiBodyTrajectory
from simulator.sim.multibody import MultiBodyScenario, MultiBodyEngine
from simulator.core.ephemeris import _PLANET_MU, _PLANET_RADIUS_KM


class App:
    def __init__(self):
        self.engine = SimulationEngine()
        self.compare_engine = SimulationEngine()
        self.preview_engine = SimulationEngine()
        self.viewport3d: Viewport3D | None = None
        self.trajectory: TrajectoryData | None = None
        self.compare_trajectory: TrajectoryData | None = None
        self.preview_trajectory: TrajectoryData | None = None
        self.playing = False
        self.play_speed = 1.0
        self.current_time = 0.0
        self.last_frame_time = 0.0
        self._post_burn_period_hint = orbital_period(6779.0, MU_EARTH)
        self._preview_enabled = True
        self._console_lines: list[str] = []
        self._maneuver_plan: list[ManeuverEvent] = []
        self._active_sc_name: str | None = None
        self._current_file_path: str | None = None
        self._left_panel_width = 280
        self._right_panel_width = 300
        self._bottom_panel_height = 140
        self._interplanetary_tab = InterplanetaryTab()
        self._active_tab = "orbital"
        self._tab1_spacecraft: list[dict] = []  # [{name, color, coe}]
        self._tab1_multi_result: MultiBodyTrajectory | None = None
        self._last_maneuver_template: str | None = None

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Orbital Mechanics Simulator", width=1400, height=900)

        theme = create_theme()
        dpg.bind_theme(theme)

        self._btn_themes = create_button_themes()

        self._build_ui()
        self._apply_button_themes()

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Key handler for Escape to clear burn preview
        with dpg.handler_registry():
            dpg.add_key_release_handler(key=dpg.mvKey_Escape, callback=self._on_escape)

        self.last_frame_time = time.time()
        self._log("Simulator ready. Set orbital parameters and click [Compute].")

        while dpg.is_dearpygui_running():
            self._frame_update()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def _build_ui(self):
        with dpg.window(
            tag="main_window",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_scrollbar=True,
            no_scroll_with_mouse=True,
        ):
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="New", callback=self._file_new)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Open...", callback=self._file_open)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Save", callback=self._file_save)
                    dpg.add_menu_item(label="Save As...", callback=self._file_save_as)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Exit", callback=self._file_exit)
                with dpg.menu(label="Edit"):
                    dpg.add_menu_item(label="Clear All Burns", callback=self._clear_burns)
                    dpg.add_menu_item(label="Clear Preview", callback=self._clear_preview_only)
                with dpg.menu(label="About"):
                    dpg.add_menu_item(label="About Orbital Simulator", callback=self._show_about)

            with dpg.tab_bar(tag="main_tab_bar", callback=self._on_tab_change):
                with dpg.tab(label="Orbital / Rendezvous", tag="tab_orbital"):
                    with dpg.child_window(tag="top_panel", border=False, no_scrollbar=True):
                        with dpg.group(horizontal=True):
                            self._build_left_panel()
                            self._build_center_panel()
                            self._build_right_panel()
                    self._build_bottom_panel()

                with dpg.tab(label="Interplanetary / Multi-Body", tag="tab_interplanetary"):
                    with dpg.child_window(tag="ipt_main_panel", border=False):
                        self._interplanetary_tab.build("ipt_main_panel")

        dpg.set_primary_window("main_window", True)

    def _apply_button_themes(self):
        """Apply color themes to categorized buttons."""
        t = self._btn_themes
        # Green: add actions
        for tag in ("btn_add_sc", "btn_add_burn"):
            dpg.bind_item_theme(tag, t["green"])
        # Red: delete/clear actions
        for tag in ("btn_delete_sc", "btn_clear_sc", "btn_clear_burns", "btn_clear_preview", "btn_remove_burn"):
            dpg.bind_item_theme(tag, t["red"])
        # Yellow: update/modify
        dpg.bind_item_theme("btn_update_sc", t["yellow"])
        # White: compute
        dpg.bind_item_theme("compute_btn", t["white"])

    def _on_tab_change(self, sender, app_data):
        """Track which tab is active for frame updates."""
        if app_data == dpg.get_item_children("main_tab_bar", 1)[0]:
            self._active_tab = "orbital"
        else:
            self._active_tab = "interplanetary"

    def _build_left_panel(self):
        with dpg.child_window(width=260, tag="left_panel"):
            dpg.add_text("ORBIT CONFIGURATION", color=(0, 191, 255))
            dpg.add_separator()

            dpg.add_text("Presets:", color=(150, 150, 150))
            with dpg.group(horizontal=True):
                dpg.add_button(label="ISS", callback=self._preset_iss, width=55)
                dpg.add_button(label="GEO", callback=self._preset_geo, width=55)
                dpg.add_button(label="Molniya", callback=self._preset_molniya, width=70)
            with dpg.group(horizontal=True):
                dpg.add_button(label="SSO", callback=self._preset_sso, width=55)
                dpg.add_button(label="HEO", callback=self._preset_heo, width=55)
                dpg.add_button(label="GPS", callback=self._preset_gps, width=55)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Tundra", callback=self._preset_tundra, width=60)
                dpg.add_button(label="Lunar", callback=self._preset_lunar, width=55)
                dpg.add_button(label="GTO", callback=self._preset_gto, width=55)

            dpg.add_spacer(height=6)
            dpg.add_text("Central Body:", color=(150, 150, 150))
            dpg.add_combo(
                ["Earth", "Moon", "Mars", "Sun", "Jupiter"],
                default_value="Earth",
                tag="central_body_select",
                width=140,
            )

            dpg.add_spacer(height=6)
            dpg.add_text("SPACECRAFT", color=(0, 255, 200))
            dpg.add_separator()
            dpg.add_input_text(label="Name", tag="sc_name_input", default_value="SC-1", width=100)
            dpg.add_combo(
                ["Cyan", "Green", "Red", "Yellow", "Orange", "Purple", "White"],
                default_value="Cyan",
                tag="sc_color_input",
                width=100,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add SC", callback=self._add_sc_tab1, width=70, tag="btn_add_sc")
                dpg.add_button(label="Update SC", callback=self._update_sc_tab1, width=75, tag="btn_update_sc")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Delete SC", callback=self._delete_sc_tab1, width=75, tag="btn_delete_sc")
                dpg.add_button(label="Clear SCs", callback=self._clear_sc_tab1, width=70, tag="btn_clear_sc")
            dpg.add_input_text(
                tag="sc_list_display",
                multiline=True,
                readonly=True,
                height=55,
                width=-1,
                default_value="Default single spacecraft.",
            )
            dpg.add_text("", tag="closest_approach_text", color=(255, 200, 80), wrap=240)

            dpg.add_spacer(height=6)
            dpg.add_text("Classical Orbital Elements:", color=(150, 150, 150))

            dpg.add_input_float(
                label="a (km)",
                tag="coe_a",
                default_value=6779.0,
                width=140,
                step=10.0,
                format="%.1f",
                callback=self._on_coe_change,
            )
            dpg.add_input_float(
                label="e",
                tag="coe_e",
                default_value=0.0001,
                width=140,
                step=0.001,
                format="%.5f",
                callback=self._on_coe_change,
            )
            dpg.add_slider_float(
                label="i (deg)",
                tag="coe_i",
                default_value=51.6,
                min_value=0.0,
                max_value=180.0,
                width=140,
                format="%.2f",
                callback=self._on_coe_change,
            )
            dpg.add_slider_float(
                label="RAAN (deg)",
                tag="coe_raan",
                default_value=0.0,
                min_value=0.0,
                max_value=360.0,
                width=140,
                format="%.2f",
                callback=self._on_coe_change,
            )
            dpg.add_slider_float(
                label="omega (deg)",
                tag="coe_omega",
                default_value=0.0,
                min_value=0.0,
                max_value=360.0,
                width=140,
                format="%.2f",
                callback=self._on_coe_change,
            )
            dpg.add_slider_float(
                label="theta (deg)",
                tag="coe_theta",
                default_value=0.0,
                min_value=0.0,
                max_value=360.0,
                width=140,
                format="%.2f",
                callback=self._on_coe_change,
            )

            dpg.add_spacer(height=4)
            dpg.add_input_float(
                label="Orbits",
                tag="n_orbits",
                default_value=3.0,
                width=140,
                step=1.0,
                format="%.1f",
            )

            dpg.add_spacer(height=6)
            dpg.add_button(
                label="COMPUTE", callback=self._on_compute, width=-1, height=28, tag="compute_btn"
            )

            dpg.add_spacer(height=4)
            dpg.add_text("", tag="orbit_info", color=(127, 255, 0))

            dpg.add_spacer(height=10)
            dpg.add_text("PERTURBATIONS", color=(255, 107, 53))
            dpg.add_separator()

            dpg.add_checkbox(label="J2 oblateness", tag="pert_j2", default_value=True)
            dpg.add_checkbox(label="Atmospheric drag", tag="pert_drag", default_value=False)
            dpg.add_checkbox(label="Moon gravity", tag="pert_moon", default_value=False)
            dpg.add_checkbox(label="Sun gravity", tag="pert_sun", default_value=False)
            dpg.add_slider_float(
                label="Cd",
                tag="drag_cd",
                default_value=2.2,
                min_value=1.5,
                max_value=3.5,
                width=140,
                format="%.2f",
            )
            dpg.add_input_float(
                label="B (km2/kg)",
                tag="drag_B",
                default_value=5.6e-9,
                width=140,
                format="%.2e",
                step=0,
                step_fast=0,
            )

            dpg.add_spacer(height=6)
            dpg.add_text("", tag="pert_info", color=(150, 150, 150))

            dpg.add_spacer(height=10)
            dpg.add_text("MANEUVERS", color=(255, 255, 0))
            dpg.add_separator()
            dpg.add_text("Target alt (km):", color=(150, 150, 150))
            dpg.add_input_float(
                label="##target_alt",
                tag="target_alt",
                default_value=35786.0,
                width=140,
                format="%.1f",
                callback=self._on_maneuver_param_change,
            )
            dpg.add_input_float(
                label="Plane dI (deg)",
                tag="plane_change_deg",
                default_value=5.0,
                width=140,
                step=0.5,
                format="%.2f",
                callback=self._on_maneuver_param_change,
            )
            dpg.add_combo(
                ["Periapsis", "Apoapsis", "Ascending Node", "Descending Node"],
                default_value="Periapsis",
                tag="maneuver_anchor",
                width=140,
                callback=self._on_maneuver_param_change,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Hohmann", callback=self._calc_hohmann, width=80)
                dpg.add_button(label="Bi-elliptic", callback=self._calc_bielliptic, width=90)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Plane-Only", callback=self._template_plane_change, width=80)
                dpg.add_button(
                    label="Transfer+Plane", callback=self._template_transfer_plane, width=110
                )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Circularize", callback=self._template_circularize, width=85)
                dpg.add_button(label="De-orbit", callback=self._template_deorbit, width=70)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Phasing", callback=self._template_phasing, width=70)
                dpg.add_button(label="Rendezvous", callback=self._template_rendezvous, width=85)
            dpg.add_checkbox(label="Compare vs no-burn", tag="compare_mode", default_value=True)
            dpg.add_text("", tag="maneuver_info", color=(200, 200, 200), wrap=240)

            dpg.add_spacer(height=8)
            dpg.add_text("BURN TIMELINE", color=(255, 200, 80))
            dpg.add_separator()
            dpg.add_input_float(
                label="dv (km/s)",
                tag="burn_dv",
                default_value=0.050,
                min_value=0.0,
                width=170,
                step=0.01,
                format="%.3f",
                callback=self._on_burn_param_change,
            )
            dpg.add_combo(
                ["At current time", "Periapsis", "Apoapsis", "Ascending Node", "Descending Node"],
                default_value="At current time",
                tag="burn_anchor",
                width=170,
                callback=self._on_burn_param_change,
            )
            dpg.add_combo(
                ["prograde", "retrograde", "normal", "antinormal", "radial_out", "radial_in"],
                default_value="prograde",
                tag="burn_dir",
                width=170,
                callback=self._on_burn_param_change,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add Burn", callback=self._add_burn, width=80, tag="btn_add_burn")
                dpg.add_button(label="Clear", callback=self._clear_burns, width=70, tag="btn_clear_burns")
                dpg.add_button(label="Clear Preview", callback=self._clear_preview_only, width=90, tag="btn_clear_preview")
            dpg.add_input_int(
                label="Remove # ",
                tag="burn_remove_idx",
                default_value=1,
                min_value=1,
                width=70,
                min_clamped=True,
            )
            dpg.add_button(label="Remove Burn", callback=self._remove_burn, width=110, tag="btn_remove_burn")
            dpg.add_input_text(
                tag="burn_plan_text",
                multiline=True,
                readonly=True,
                height=92,
                width=-1,
                default_value="No planned burns.",
            )

    def _build_center_panel(self):
        with dpg.child_window(
            width=-1,
            tag="center_panel",
            no_scrollbar=True,
            no_scroll_with_mouse=True,
        ):
            # Top toolbar: Active SC on left, view presets on right
            with dpg.group(horizontal=True):
                dpg.add_text("Active:", color=(200, 200, 200))
                dpg.add_combo(
                    [],
                    default_value="",
                    tag="active_sc_select",
                    width=100,
                    callback=self._on_active_sc_change,
                )
                dpg.add_spacer(width=20)
                dpg.add_button(label="Fit", callback=self._fit_view, width=40)
                dpg.add_spacer(width=12)
                dpg.add_button(
                    label="ISO", callback=lambda: self.viewport3d.set_view_isometric(), width=38
                )
                dpg.add_button(label="XY", callback=lambda: self.viewport3d.set_view_xy(), width=32)
                dpg.add_button(label="XZ", callback=lambda: self.viewport3d.set_view_xz(), width=32)
                dpg.add_button(label="YZ", callback=lambda: self.viewport3d.set_view_yz(), width=32)
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="Orbit",
                    callback=lambda: self.viewport3d.set_view_orbit_normal(),
                    width=45,
                )
                dpg.add_button(
                    label="Track",
                    callback=lambda: self.viewport3d.set_view_along_velocity(),
                    width=45,
                )
                dpg.add_button(
                    label="Nadir", callback=lambda: self.viewport3d.set_view_nadir(), width=45
                )

            dpg.add_separator()

            self.viewport3d = Viewport3D("viewport_drawlist", width=780, height=540)
            self.viewport3d.create("center_panel")
            self.viewport3d.set_orbit_selected_callback(self._on_orbit_clicked)

            dpg.add_spacer(height=2)

            with dpg.group(horizontal=True, tag="timeline_group"):
                dpg.add_button(label="Play", tag="play_btn", callback=self._toggle_play, width=50)
                dpg.add_button(label="Reset", callback=self._reset_time, width=50)
                dpg.add_slider_float(
                    label="##time_slider",
                    tag="time_slider",
                    default_value=0.0,
                    min_value=0.0,
                    max_value=1.0,
                    width=-1,
                    callback=self._on_time_scrub,
                    format="%.3f",
                )
                dpg.add_combo(
                    [
                        "1x",
                        "5x",
                        "10x",
                        "50x",
                        "100x",
                        "500x",
                        "1000x",
                        "5000x",
                        "10000x",
                        "50000x",
                    ],
                    default_value="10x",
                    tag="speed_combo",
                    callback=self._on_speed_change,
                    width=70,
                )

            dpg.add_text("t = 0.0 s  |  0.0 / 0.0 s", tag="time_display", color=(150, 150, 150))

    def _build_right_panel(self):
        with dpg.child_window(width=260, tag="right_panel"):
            dpg.add_text("TELEMETRY", color=(127, 255, 0))
            dpg.add_separator()

            for label, tag in [
                ("Altitude:", "tel_alt"),
                ("Velocity:", "tel_vel"),
                ("Period:", "tel_period"),
                ("Energy:", "tel_energy"),
                ("a:", "tel_a"),
                ("e:", "tel_e"),
                ("i:", "tel_i"),
                ("RAAN:", "tel_raan"),
                ("omega:", "tel_omega"),
                ("theta:", "tel_theta"),
            ]:
                with dpg.group(horizontal=True):
                    dpg.add_text(label, color=(150, 150, 150))
                    dpg.add_text("---", tag=tag)

            dpg.add_spacer(height=10)
            dpg.add_text("2D PLOTS", color=(255, 107, 53))
            dpg.add_separator()

            with dpg.plot(label="Altitude", height=160, width=-1, tag="plot_alt"):
                dpg.add_plot_axis(dpg.mvXAxis, label="t (min)", tag="alt_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="Alt (km)", tag="alt_y")

            with dpg.plot(label="Velocity", height=160, width=-1, tag="plot_vel"):
                dpg.add_plot_axis(dpg.mvXAxis, label="t (min)", tag="vel_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="v (km/s)", tag="vel_y")

            dpg.add_spacer(height=10)
            dpg.add_text("EXPORT", color=(0, 191, 255))
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="CSV", callback=self._export_csv, width=60)
                dpg.add_button(label="Plots (PNG)", callback=self._export_plots, width=90)
            dpg.add_text("", tag="export_status", color=(150, 150, 150))

            dpg.add_spacer(height=8)
            dpg.add_text("BURN EVENTS", color=(255, 200, 80))
            dpg.add_separator()
            dpg.add_input_text(
                tag="burn_events_text",
                multiline=True,
                readonly=True,
                height=92,
                width=-1,
                default_value="No executed burns.",
            )

            dpg.add_spacer(height=8)
            dpg.add_text("COMPARE", color=(160, 210, 255))
            dpg.add_separator()
            dpg.add_input_text(
                tag="compare_text",
                multiline=True,
                readonly=True,
                height=66,
                width=-1,
                default_value="Compare mode idle.",
            )

    def _build_bottom_panel(self):
        with dpg.child_window(tag="bottom_panel", border=False, no_scrollbar=True):
            dpg.add_text("CONSOLE", color=(150, 150, 150))
            dpg.add_separator()
            dpg.add_input_text(
                tag="console_text",
                multiline=True,
                readonly=True,
                height=80,
                width=-1,
                default_value="",
            )

    def _log(self, msg: str):
        self._console_lines.append(f"> {msg}")
        if len(self._console_lines) > 50:
            self._console_lines = self._console_lines[-50:]
        dpg.set_value("console_text", "\n".join(self._console_lines))

    def _frame_update(self):
        now = time.time()
        dt_frame = now - self.last_frame_time
        self.last_frame_time = now
        self._update_layout()

        if (
            self._preview_enabled
            and self.preview_engine.is_complete
            and self.preview_trajectory is None
        ):
            preview_result = self.preview_engine.result
            if preview_result is not None:
                self.preview_trajectory = preview_result
                self.viewport3d.set_preview_trajectory(preview_result)

        if self.compare_engine.is_complete and self.compare_trajectory is None:
            compare_result = self.compare_engine.result
            if compare_result is not None:
                self.compare_trajectory = compare_result
                self.viewport3d.set_reference_trajectory(compare_result)
                self._update_compare_metrics()
                self._log(f"Compare baseline ready: {compare_result.n_points} points")

        # Sync compare/reference visibility with checkbox
        if self.viewport3d:
            self.viewport3d.set_show_reference(bool(dpg.get_value("compare_mode")))

        if self.engine.is_complete and self.trajectory is None:
            result = self.engine.result
            if result is not None:
                self.trajectory = result
                self.viewport3d.set_trajectory(
                    result,
                    self._get_central_body().radius,
                    body_name=self._get_central_body().name,
                    body_color=self._get_body_color(self._get_central_body().name),
                )
                # Set satellite color based on active SC
                active_name = dpg.get_value("active_sc_select")
                for sc in self._tab1_spacecraft:
                    if sc["name"] == active_name:
                        self.viewport3d.set_satellite_color(sc["color"])
                        break
                else:
                    self.viewport3d.set_satellite_color((0, 191, 255))
                self.viewport3d.clear_transfers()
                burn_events = self.engine.burn_events
                if len(burn_events) > 0:
                    burn_points = np.array([ev["r"] for ev in burn_events])
                    self.viewport3d.set_burn_markers(burn_points)
                else:
                    self.viewport3d.clear_burn_markers()
                self._update_burn_events_display(burn_events)
                self._update_compare_metrics()
                self.current_time = result.t[0]
                self._update_plots()
                self._update_orbit_info()
                val = self.engine.validation
                if val:
                    self._log(
                        f"Conservation: dE/E={val['energy_drift']:.2e}  "
                        f"dh/h={val['angular_momentum_drift']:.2e}"
                    )
                self._log(f"Done: {result.n_points} points, duration={result.duration:.1f}s")
                dpg.configure_item("time_slider", max_value=result.duration)
                # Store trajectory for active SC and refresh all multi-trajectories
                active_name = dpg.get_value("active_sc_select")
                if active_name:
                    for sc in self._tab1_spacecraft:
                        if sc["name"] == active_name:
                            sc["trajectory"] = result
                            sc["burn_events"] = list(self.engine.burn_events)
                            break
                # Rebuild all multi-trajectories with latest data
                self._refresh_all_multi_trajectories()
                # Ensure viewport selection matches active SC
                if active_name and self.viewport3d:
                    self.viewport3d._selected_orbit = active_name
                # Now compute closest approach (after storing burn trajectory)
                if self._tab1_multi_result is not None:
                    self._compute_closest_approach_deferred()
                # Fit view after load
                if getattr(self, "_fit_after_compute", False):
                    self._fit_after_compute = False
                    self._fit_view()
            elif self.engine.error:
                self._log(f"ERROR: {self.engine.error}")

        if self.playing and self.trajectory is not None:
            self.current_time += dt_frame * self.play_speed
            if self.current_time > self.trajectory.t[-1]:
                self.current_time = self.trajectory.t[0]
            dpg.set_value("time_slider", self.current_time)
            self.viewport3d.set_time(self.current_time)
            self._update_telemetry()
            self._update_time_display()
            self._update_plot_cursor()

        if self.viewport3d:
            self.viewport3d.render()

        # Update interplanetary tab
        self._interplanetary_tab.frame_update(dt_frame)

    def _sc_color_name_to_rgb(self, name: str) -> tuple[int, int, int]:
        mapping = {
            "Cyan": (0, 191, 255),
            "Green": (0, 255, 0),
            "Red": (255, 80, 80),
            "Yellow": (255, 255, 0),
            "Orange": (255, 128, 0),
            "Purple": (200, 80, 255),
            "White": (255, 255, 255),
        }
        return mapping.get(name, (0, 191, 255))

    def _add_sc_tab1(self):
        name = dpg.get_value("sc_name_input")
        color = self._sc_color_name_to_rgb(dpg.get_value("sc_color_input"))
        a = dpg.get_value("coe_a")
        e = dpg.get_value("coe_e")
        i = dpg.get_value("coe_i")
        raan = dpg.get_value("coe_raan")
        omega = dpg.get_value("coe_omega")
        theta = dpg.get_value("coe_theta")
        coe = OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, theta=theta)

        # Check duplicate name
        for sc in self._tab1_spacecraft:
            if sc["name"] == name:
                name = name + f"_{len(self._tab1_spacecraft) + 1}"
                break

        self._tab1_spacecraft.append(
            {"name": name, "color": color, "coe": coe, "maneuvers": [], "trajectory": None}
        )
        self._refresh_tab1_sc_list()
        # Auto-select the newly added SC
        dpg.set_value("active_sc_select", name)
        self._active_sc_name = name
        # Clear preview when adding a new SC
        self._preview_enabled = False
        self.viewport3d.clear_preview_trajectory()
        self.viewport3d.clear_transfers()
        self.viewport3d.set_burn_cursor(None)
        # Auto-cycle name and color
        color_cycle = ["Cyan", "Green", "Red", "Yellow", "Orange", "Purple", "White"]
        next_idx = len(self._tab1_spacecraft) % len(color_cycle)
        dpg.set_value("sc_name_input", f"SC-{len(self._tab1_spacecraft) + 1}")
        dpg.set_value("sc_color_input", color_cycle[next_idx])
        self._log(f"Added spacecraft: {name}")

    def _update_sc_tab1(self):
        """Update the selected SC's COE from current slider values."""
        name = dpg.get_value("active_sc_select")
        if not name:
            self._log("No SC selected to update.")
            return
        a = dpg.get_value("coe_a")
        e = dpg.get_value("coe_e")
        i = dpg.get_value("coe_i")
        raan = dpg.get_value("coe_raan")
        omega = dpg.get_value("coe_omega")
        theta = dpg.get_value("coe_theta")
        coe = OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, theta=theta)
        for sc in self._tab1_spacecraft:
            if sc["name"] == name:
                sc["coe"] = coe
                break
        self._refresh_tab1_sc_list()
        self._log(f"Updated {name}: a={a:.0f} e={e:.4f}")

    def _clear_sc_tab1(self):
        self._tab1_spacecraft.clear()
        self._tab1_multi_result = None
        self._refresh_tab1_sc_list()
        self.viewport3d.clear_multi_trajectories()
        self.viewport3d.clear_closest_approach_line()
        # Reset name and color
        dpg.set_value("sc_name_input", "SC-1")
        dpg.set_value("sc_color_input", "Cyan")
        self._log("Cleared extra spacecraft.")

    def _delete_sc_tab1(self):
        """Delete the currently selected spacecraft."""
        name = dpg.get_value("active_sc_select")
        if not name:
            self._log("No SC selected to delete.")
            return
        self._tab1_spacecraft = [sc for sc in self._tab1_spacecraft if sc["name"] != name]
        self._tab1_multi_result = None
        self.viewport3d.clear_multi_trajectories()
        self.viewport3d.clear_closest_approach_line()
        self._refresh_tab1_sc_list()
        # Cycle back name and color
        color_cycle = ["Cyan", "Green", "Red", "Yellow", "Orange", "Purple", "White"]
        next_idx = len(self._tab1_spacecraft) % len(color_cycle)
        dpg.set_value("sc_name_input", f"SC-{len(self._tab1_spacecraft) + 1}")
        dpg.set_value("sc_color_input", color_cycle[next_idx])
        self._log(f"Deleted spacecraft: {name}")

    def _refresh_tab1_sc_list(self):
        if not self._tab1_spacecraft:
            dpg.set_value("sc_list_display", "Default single spacecraft.")
            dpg.configure_item("active_sc_select", items=[], default_value="")
            dpg.set_value("closest_approach_text", "")
            return
        lines = []
        names = []
        for i, sc in enumerate(self._tab1_spacecraft, 1):
            coe = sc["coe"]
            lines.append(f"{i}. {sc['name']} a={coe.a:.0f}")
            names.append(sc["name"])
        dpg.set_value("sc_list_display", "\n".join(lines))
        # Preserve current selection if still valid
        current = dpg.get_value("active_sc_select")
        if current not in names:
            current = names[0] if names else ""
        dpg.configure_item("active_sc_select", items=names, default_value=current)

    def _on_active_sc_change(self, sender=None, app_data=None, user_data=None):
        """When user selects a different SC, load its COE into the sliders."""
        name = dpg.get_value("active_sc_select")
        self._select_sc_by_name(name)

    def _on_orbit_clicked(self, name: str):
        """Called when user clicks an orbit in the 3D viewport."""
        dpg.set_value("active_sc_select", name)
        self._select_sc_by_name(name)

    def _select_sc_by_name(self, name: str):
        """Select a SC by name: load COE, maneuvers, swap trajectory, update telemetry."""
        # Save current maneuvers to previously active SC
        prev_name = getattr(self, "_active_sc_name", None)
        if prev_name:
            for sc in self._tab1_spacecraft:
                if sc["name"] == prev_name:
                    sc["maneuvers"] = list(self._maneuver_plan)
                    break
        self._active_sc_name = name

        for sc in self._tab1_spacecraft:
            if sc["name"] == name:
                coe = sc["coe"]
                dpg.set_value("coe_a", coe.a)
                dpg.set_value("coe_e", coe.e)
                dpg.set_value("coe_i", coe.i)
                dpg.set_value("coe_raan", coe.raan)
                dpg.set_value("coe_omega", coe.omega)
                dpg.set_value("coe_theta", coe.theta)
                self._on_coe_change()
                # Load this SC's maneuvers
                self._maneuver_plan = list(sc.get("maneuvers", []))
                self._refresh_burn_plan_display()
                # Use stored trajectory if available, else multi_result, else engine
                traj = sc.get("trajectory")
                if traj is None and self._tab1_multi_result is not None:
                    traj = self._tab1_multi_result.get_trajectory(name)
                if traj is not None:
                    self.trajectory = traj
                    self.viewport3d.set_trajectory(
                        traj,
                        self._get_central_body().radius,
                        body_name=self._get_central_body().name,
                        body_color=self._get_body_color(self._get_central_body().name),
                    )
                    self.current_time = min(self.current_time, traj.t[-1])
                    dpg.configure_item("time_slider", max_value=traj.duration)
                    dpg.set_value("time_slider", self.current_time)
                elif self.trajectory is not None:
                    self.viewport3d.set_trajectory(
                        self.trajectory,
                        self._get_central_body().radius,
                        body_name=self._get_central_body().name,
                        body_color=self._get_body_color(self._get_central_body().name),
                    )
                if self.trajectory is not None:
                    self._update_telemetry()
                    self._update_plots()
                    self._update_time_display()
                # Restore burn events display for this SC
                burn_events = sc.get("burn_events", [])
                self._update_burn_events_display(burn_events)
                if burn_events:
                    burn_points = np.array([ev["r"] for ev in burn_events])
                    self.viewport3d.set_burn_markers(burn_points)
                else:
                    self.viewport3d.clear_burn_markers()
                # Clear compare data for this SC
                self.compare_trajectory = None
                if self.viewport3d:
                    self.viewport3d.set_reference_trajectory(None)
                self._update_compare_metrics()
                # Clear burn preview
                self._preview_enabled = False
                if self.viewport3d:
                    self.viewport3d.clear_preview_trajectory()
                    self.viewport3d.set_burn_cursor(None)
                # Set satellite color
                self.viewport3d.set_satellite_color(sc["color"])
                # Highlight in viewport
                if self.viewport3d:
                    self.viewport3d._selected_orbit = name
                    self.viewport3d._needs_redraw = True
                # Refresh multi-trajectories so all SC show their latest (burned) paths
                self._refresh_all_multi_trajectories()
                break

    def _get_central_body(self) -> CelestialBody:
        """Get the selected central body."""
        body_name = dpg.get_value("central_body_select")
        if body_name == "Earth":
            return CelestialBody.earth()
        elif body_name == "Moon":
            return CelestialBody.moon()
        elif body_name == "Sun":
            return CelestialBody.sun()
        else:
            # Generic body from ephemeris data
            mu = _PLANET_MU.get(body_name, MU_EARTH)
            radius = _PLANET_RADIUS_KM.get(body_name, R_EARTH)
            return CelestialBody(name=body_name, mu=mu, radius=radius)

    def _get_body_color(self, name: str) -> tuple[int, int, int]:
        colors = {
            "Earth": (28, 88, 170),
            "Moon": (140, 140, 140),
            "Mars": (180, 80, 50),
            "Sun": (255, 200, 0),
            "Jupiter": (180, 150, 100),
            "Venus": (200, 180, 80),
            "Saturn": (200, 180, 120),
            "Mercury": (130, 130, 130),
        }
        return colors.get(name, (100, 100, 200))

    def _on_compute(self):
        a = dpg.get_value("coe_a")
        e = dpg.get_value("coe_e")
        i = dpg.get_value("coe_i")
        raan = dpg.get_value("coe_raan")
        omega = dpg.get_value("coe_omega")
        theta = dpg.get_value("coe_theta")
        n_orb = dpg.get_value("n_orbits")
        n_orb_eff = self._effective_n_orbits(a, n_orb)
        if n_orb_eff > n_orb + 1e-6:
            dpg.set_value("n_orbits", n_orb_eff)
            n_orb = n_orb_eff
            self._log(f"Auto-extended simulated orbits to {n_orb:.2f} so all burns are visible.")

        coe = OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, theta=theta)

        # Ensure active SC's stored COE matches sliders (user may have tweaked without Update)
        active_name = dpg.get_value("active_sc_select")
        if active_name:
            for sc in self._tab1_spacecraft:
                if sc["name"] == active_name:
                    sc["coe"] = coe
                    break

        pert = PerturbationConfig(
            j2_enabled=dpg.get_value("pert_j2"),
            drag_enabled=dpg.get_value("pert_drag"),
            drag_cd=dpg.get_value("drag_cd"),
            drag_ballistic_coeff=dpg.get_value("drag_B"),
            third_body_moon=dpg.get_value("pert_moon"),
            third_body_sun=dpg.get_value("pert_sun"),
        )

        scenario = Scenario(
            initial_coe=coe,
            central_body=self._get_central_body(),
            perturbations=pert,
            n_orbits=n_orb,
            maneuvers=self._build_absolute_maneuvers(a, n_orb),
        )

        compare_enabled = bool(dpg.get_value("compare_mode"))
        compare_scenario = None
        if compare_enabled and len(scenario.maneuvers) > 0:
            compare_scenario = Scenario(
                initial_coe=coe,
                central_body=self._get_central_body(),
                perturbations=pert,
                n_orbits=n_orb,
                maneuvers=[],
            )

        self._preview_enabled = False
        self.trajectory = None
        self.compare_trajectory = None
        self.preview_trajectory = None
        self.viewport3d.clear_transfers()
        self.viewport3d.clear_preview_trajectory()
        self.viewport3d.clear_preview_orbit()
        self.viewport3d.clear_reference_trajectory()
        self.viewport3d.clear_burn_markers()
        self.viewport3d.clear_multi_trajectories()
        self.viewport3d.set_burn_cursor(None)
        self._tab1_multi_result = None
        self._update_burn_events_display([])
        dpg.set_value("compare_text", "Compare mode idle.")
        dpg.set_value("closest_approach_text", "")
        self.playing = False
        dpg.set_value("play_btn", "Play")
        self.engine.compute(scenario)
        if compare_scenario is not None:
            self.compare_engine.compute(compare_scenario)
            self._log("Computing baseline no-burn trajectory for comparison...")

        # Compute additional spacecraft trajectories
        if self._tab1_spacecraft:
            self._compute_additional_spacecraft(scenario)

        self._log(
            f"Computing: a={a:.1f} e={e:.4f} i={i:.1f} n={n_orb:.0f} orbits, burns={len(scenario.maneuvers)}"
        )

    def _compute_additional_spacecraft(self, primary_scenario: Scenario):
        """Compute trajectories for additional spacecraft and display them."""
        from datetime import datetime, timezone

        central = self._get_central_body()
        # Use duration that covers all SC orbits
        T_primary = orbital_period(primary_scenario.initial_coe.a, central.mu)
        max_T = T_primary
        for sc_data in self._tab1_spacecraft:
            T_sc = orbital_period(sc_data["coe"].a, central.mu)
            if T_sc > max_T:
                max_T = T_sc
        duration = max_T * primary_scenario.n_orbits

        # Build perturbing bodies
        perturbers = []
        cfg = primary_scenario.perturbations
        if cfg.third_body_moon:
            perturbers.append("Moon")
        if cfg.third_body_sun:
            perturbers.append("Sun")

        active_name = dpg.get_value("active_sc_select")
        spacecraft = []
        for sc_data in self._tab1_spacecraft:
            spacecraft.append(
                Spacecraft(
                    name=sc_data["name"],
                    color=sc_data["color"],
                    initial_coe=sc_data["coe"],
                )
            )

        epoch = getattr(primary_scenario, "epoch", datetime(2024, 1, 1, tzinfo=timezone.utc))
        scenario = MultiBodyScenario(
            spacecraft=spacecraft,
            central_body=central.name,
            perturbing_bodies=perturbers,
            epoch=epoch,
            duration_seconds=duration,
            perturbations=cfg,
        )

        engine = MultiBodyEngine()
        result = engine.compute(scenario)
        if result is None:
            self._log(f"Additional SC error: {engine.error}")
            return

        self._tab1_multi_result = result

        # Set multi trajectories on viewport
        trajs = []
        for sc_name, sc_color in zip(result.spacecraft_names, result.spacecraft_colors):
            traj = result.get_trajectory(sc_name)
            trajs.append((traj, sc_color, sc_name))
        self.viewport3d.set_multi_trajectories(trajs)
        # Set active SC as selected in viewport
        active_name = dpg.get_value("active_sc_select")
        if active_name:
            self.viewport3d._selected_orbit = active_name

        # Compute closest approach between primary and each additional SC
        self._compute_closest_approach(primary_scenario)
        self._log(f"Computed {len(spacecraft)} additional spacecraft.")

    def _update_multi_traj_for_active(self, active_name: str, result):
        """Replace multi-trajectory entries for all SC that have stored burn trajectories."""
        if not self.viewport3d:
            return
        new_multi = []
        for orbit_pts, c, name, _t in self.viewport3d._multi_trajectories:
            # Check if this SC has a stored trajectory (with burns)
            sc_traj = None
            sc_color = None
            for sc in self._tab1_spacecraft:
                if sc["name"] == name:
                    if name == active_name:
                        sc_traj = result
                    else:
                        sc_traj = sc.get("trajectory")
                    sc_color = sc["color"]
                    break
            if sc_traj is not None:
                n = 3000
                if sc_traj.n_points <= n:
                    pts, t_arr = sc_traj.r, sc_traj.t
                else:
                    idx = np.linspace(0, sc_traj.n_points - 1, n, dtype=int)
                    pts, t_arr = sc_traj.r[idx], sc_traj.t[idx]
                new_multi.append((pts, (*sc_color, 220), name, t_arr))
            else:
                new_multi.append((orbit_pts, c, name, _t))
        self.viewport3d._multi_trajectories = new_multi
        self.viewport3d._needs_redraw = True

    def _refresh_all_multi_trajectories(self):
        """Rebuild multi-trajectories from stored burn trajectories for all SC."""
        if not self.viewport3d or not self._tab1_spacecraft:
            return
        new_multi = []
        for sc in self._tab1_spacecraft:
            name = sc["name"]
            color = sc["color"]
            # Prefer stored trajectory (with burns), else multi_result, else skip
            traj = sc.get("trajectory")
            if traj is None and self._tab1_multi_result is not None:
                traj = self._tab1_multi_result.get_trajectory(name)
            if traj is not None:
                n = 3000
                if traj.n_points <= n:
                    pts, t_arr = traj.r, traj.t
                else:
                    idx = np.linspace(0, traj.n_points - 1, n, dtype=int)
                    pts, t_arr = traj.r[idx], traj.t[idx]
                new_multi.append((pts, (*color, 220), name, t_arr))
        self.viewport3d._multi_trajectories = new_multi
        self.viewport3d._needs_redraw = True

    def _compute_closest_approach_deferred(self):
        """Compute closest approach after primary result is ready."""
        self._compute_closest_approach(None)

    def _compute_closest_approach(self, primary_scenario=None):
        """Compute closest approach between spacecraft pairs."""
        if self._tab1_multi_result is None:
            return

        names = self._tab1_multi_result.spacecraft_names

        def _get_best_traj(sc_name):
            """Get the best available trajectory for a SC (burned > original)."""
            for sc in self._tab1_spacecraft:
                if sc["name"] == sc_name and sc.get("trajectory") is not None:
                    return sc["trajectory"]
            return self._tab1_multi_result.get_trajectory(sc_name)

        if len(names) < 2:
            # Compare primary trajectory vs single additional SC
            if self.trajectory is None or len(names) < 1:
                return
            primary_result = self.trajectory
            other_traj = _get_best_traj(names[0])
            if other_traj is None:
                return
            t_max = min(primary_result.t[-1], other_traj.t[-1])
            t_samples = np.linspace(0, t_max, 2000)
            min_dist, min_t, min_r1, min_r2 = np.inf, 0.0, None, None
            for t in t_samples:
                r1, _ = primary_result.interpolate(t)
                r2, _ = other_traj.interpolate(t)
                d = np.linalg.norm(r1 - r2)
                if d < min_dist:
                    min_dist, min_t = d, t
                    min_r1, min_r2 = r1.copy(), r2.copy()
            dpg.set_value(
                "closest_approach_text", f"Primary↔{names[0]}: {min_dist:.2f} km @ t={min_t:.0f}s"
            )
            if min_r1 is not None:
                self.viewport3d.set_closest_approach_line(min_r1, min_r2)
            return

        # Multiple SC: compute pairwise closest approach
        lines = []
        best_dist = np.inf
        best_pa = None
        best_pb = None
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                traj_a = _get_best_traj(names[i])
                traj_b = _get_best_traj(names[j])
                if traj_a is None or traj_b is None:
                    continue
                t_max = min(traj_a.t[-1], traj_b.t[-1])
                t_samples = np.linspace(0, t_max, 2000)
                min_dist, min_t, min_r1, min_r2 = np.inf, 0.0, None, None
                for t in t_samples:
                    r1, _ = traj_a.interpolate(t)
                    r2, _ = traj_b.interpolate(t)
                    d = np.linalg.norm(r1 - r2)
                    if d < min_dist:
                        min_dist, min_t = d, t
                        min_r1, min_r2 = r1.copy(), r2.copy()
                lines.append(f"{names[i]}↔{names[j]}: {min_dist:.2f} km @ t={min_t:.0f}s")
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_pa, best_pb = min_r1, min_r2

        dpg.set_value("closest_approach_text", "\n".join(lines))
        if best_pa is not None:
            self.viewport3d.set_closest_approach_line(best_pa, best_pb)

    def _update_layout(self):
        vw = max(640, dpg.get_viewport_client_width())
        vh = max(480, dpg.get_viewport_client_height())

        dpg.configure_item("main_window", pos=(0, 0), width=vw, height=vh)

        # Account for tab bar height (~30px)
        tab_bar_h = 30
        available_h = vh - tab_bar_h

        bottom_h = min(max(110, int(available_h * 0.18)), 200)
        top_h = max(220, available_h - bottom_h - 8)

        left_w = min(self._left_panel_width, max(220, int(vw * 0.25)))
        right_w = min(self._right_panel_width, max(240, int(vw * 0.28)))
        remaining_w = vw - left_w - right_w - 30
        if remaining_w < 320:
            shortage = 320 - remaining_w
            right_reduce = min(shortage, max(0, right_w - 220))
            right_w -= right_reduce
            shortage -= right_reduce
            left_reduce = min(shortage, max(0, left_w - 200))
            left_w -= left_reduce
        center_w = max(320, vw - left_w - right_w - 30)

        dpg.configure_item("top_panel", width=vw - 12, height=top_h - 4)
        dpg.configure_item("left_panel", width=left_w, height=top_h - 8)
        dpg.configure_item("center_panel", width=center_w, height=top_h - 8)
        dpg.configure_item("right_panel", width=right_w, height=top_h - 8)

        dpg.configure_item("bottom_panel", width=vw - 12, height=bottom_h - 6)

        console_h = max(64, bottom_h - 48)
        dpg.configure_item("console_text", height=console_h)

        viewport_w = max(320, center_w - 16)
        viewport_h = max(260, top_h - 130)
        if self.viewport3d is not None:
            self.viewport3d.resize(viewport_w, viewport_h)

        slider_w = max(180, viewport_w - 220)
        dpg.configure_item("time_slider", width=slider_w)

    def _toggle_play(self):
        if self.trajectory is None:
            return
        self.playing = not self.playing
        dpg.configure_item("play_btn", label="Pause" if self.playing else "Play")

    def _reset_time(self):
        if self.trajectory is None:
            return
        self.current_time = self.trajectory.t[0]
        dpg.set_value("time_slider", 0.0)
        self.viewport3d.set_time(self.current_time)
        self._update_telemetry()
        self._update_time_display()
        self._update_plot_cursor()

    def _on_time_scrub(self, sender, app_data):
        if self.trajectory is None:
            return
        self.current_time = app_data
        self.viewport3d.set_time(self.current_time)
        self._update_telemetry()
        self._update_time_display()
        self._update_plot_cursor()
        # Update burn cursor if anchor is "At current time"
        if dpg.does_item_exist("burn_anchor") and str(dpg.get_value("burn_anchor")) == "At current time":
            self._update_burn_cursor()
            # Also update burn preview if dv > 0
            dv = float(dpg.get_value("burn_dv")) if dpg.does_item_exist("burn_dv") else 0.0
            if dv > 0:
                self._on_burn_param_change()

    def _on_speed_change(self, sender, app_data):
        speed_map = {
            "1x": 1,
            "5x": 5,
            "10x": 10,
            "50x": 50,
            "100x": 100,
            "500x": 500,
            "1000x": 1000,
            "5000x": 5000,
            "10000x": 10000,
            "50000x": 50000,
        }
        self.play_speed = speed_map.get(app_data, 10)

    def _on_coe_change(self, sender=None, app_data=None, user_data=None):
        """Live preview: compute analytical orbit from current COE sliders."""
        try:
            a = dpg.get_value("coe_a")
            e = dpg.get_value("coe_e")
            i_deg = dpg.get_value("coe_i")
            raan_deg = dpg.get_value("coe_raan")
            omega_deg = dpg.get_value("coe_omega")
            if a <= 0 or e < 0 or e >= 1.0:
                self.viewport3d.clear_preview_orbit()
                return
            body = self._get_central_body()
            # Generate points around the orbit analytically
            thetas = np.linspace(0, 2 * np.pi, 200)
            i = np.radians(i_deg)
            raan = np.radians(raan_deg)
            omega = np.radians(omega_deg)
            p = a * (1 - e**2)
            positions = []
            for th in thetas:
                r_mag = p / (1 + e * np.cos(th))
                # Perifocal coords
                x_pf = r_mag * np.cos(th)
                y_pf = r_mag * np.sin(th)
                # Rotation to inertial
                cos_O, sin_O = np.cos(raan), np.sin(raan)
                cos_i, sin_i = np.cos(i), np.sin(i)
                cos_w, sin_w = np.cos(omega), np.sin(omega)
                x = (cos_O * cos_w - sin_O * sin_w * cos_i) * x_pf + (
                    -cos_O * sin_w - sin_O * cos_w * cos_i
                ) * y_pf
                y = (sin_O * cos_w + cos_O * sin_w * cos_i) * x_pf + (
                    -sin_O * sin_w + cos_O * cos_w * cos_i
                ) * y_pf
                z = (sin_w * sin_i) * x_pf + (cos_w * sin_i) * y_pf
                positions.append([x, y, z])
            self.viewport3d.set_preview_orbit(np.array(positions), color=(100, 255, 100))
        except Exception:
            self.viewport3d.clear_preview_orbit()

    def _update_telemetry(self):
        if self.trajectory is None:
            return
        r, v = self.trajectory.interpolate(self.current_time)
        body = self._get_central_body()
        alt = np.linalg.norm(r) - body.radius
        vel = np.linalg.norm(v)
        eps = specific_energy(r, v, body.mu)
        coe = state_to_coe(StateVector(r=r, v=v), body.mu)
        T = orbital_period(coe.a, body.mu)

        dpg.set_value("tel_alt", f"{alt:.2f} km")
        dpg.set_value("tel_vel", f"{vel:.4f} km/s")
        dpg.set_value("tel_period", f"{T / 60:.2f} min")
        dpg.set_value("tel_energy", f"{eps:.4f} km2/s2")
        dpg.set_value("tel_a", f"{coe.a:.2f} km")
        dpg.set_value("tel_e", f"{coe.e:.6f}")
        dpg.set_value("tel_i", f"{coe.i:.4f} deg")
        dpg.set_value("tel_raan", f"{coe.raan:.4f} deg")
        dpg.set_value("tel_omega", f"{coe.omega:.4f} deg")
        dpg.set_value("tel_theta", f"{coe.theta:.2f} deg")

    def _seconds_to_human(self, secs: float) -> str:
        """Convert seconds to human-readable format: Xy Xm Xd HH:MM."""
        days_total = secs / 86400.0
        years = int(days_total // 365.25)
        rem = days_total - years * 365.25
        months = int(rem // 30.44)
        rem -= months * 30.44
        days = int(rem)
        hours = int((secs % 86400) // 3600)
        minutes = int((secs % 3600) // 60)
        parts = []
        if years > 0:
            parts.append(f"{years}y")
        if months > 0:
            parts.append(f"{months}m")
        if days > 0 or not parts:
            parts.append(f"{days}d")
        parts.append(f"{hours:02d}:{minutes:02d}")
        return " ".join(parts)

    def _update_time_display(self):
        if self.trajectory is None:
            return
        dur = self.trajectory.duration
        t_human = self._seconds_to_human(self.current_time)
        dur_human = self._seconds_to_human(dur)
        dpg.set_value(
            "time_display",
            f"t = {self.current_time:.1f}s ({t_human})  |  {dur:.1f}s ({dur_human})",
        )

    def _update_orbit_info(self):
        a = dpg.get_value("coe_a")
        e = dpg.get_value("coe_e")
        i = dpg.get_value("coe_i")
        T = orbital_period(a, MU_EARTH)
        rp = a * (1 - e)
        ra = a * (1 + e)
        dpg.set_value(
            "orbit_info", f"T={T / 60:.2f}min  rp={rp - R_EARTH:.0f}km  ra={ra - R_EARTH:.0f}km"
        )

        if dpg.get_value("pert_j2"):
            dr = j2_raan_rate(a, e, i)
            dw = j2_argp_rate(a, e, i)
            dpg.set_value("pert_info", f"J2: dRAAN={dr:.3f} d/day  dw={dw:.3f} d/day")

    def _update_plots(self):
        if self.trajectory is None:
            return
        t_min = self.trajectory.t / 60.0
        alt = np.linalg.norm(self.trajectory.r, axis=1) - R_EARTH
        vel = np.linalg.norm(self.trajectory.v, axis=1)

        if dpg.does_item_exist("alt_series"):
            dpg.delete_item("alt_series")
        if dpg.does_item_exist("vel_series"):
            dpg.delete_item("vel_series")

        dpg.add_line_series(t_min.tolist(), alt.tolist(), parent="alt_y", tag="alt_series")
        dpg.add_line_series(t_min.tolist(), vel.tolist(), parent="vel_y", tag="vel_series")

        dpg.fit_axis_data("alt_x")
        dpg.fit_axis_data("alt_y")
        dpg.fit_axis_data("vel_x")
        dpg.fit_axis_data("vel_y")
        self._update_plot_cursor()

    def _update_plot_cursor(self):
        """Update vertical time cursor on 2D plots."""
        if self.trajectory is None:
            return
        t_cursor = self.current_time / 60.0  # minutes
        for tag in ("alt_cursor", "vel_cursor"):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        dpg.add_inf_line_series([t_cursor], parent="alt_y", tag="alt_cursor")
        dpg.add_inf_line_series([t_cursor], parent="vel_y", tag="vel_cursor")

    def _on_maneuver_param_change(self, sender=None, app_data=None, user_data=None):
        """Live-update maneuver preview when target_alt, plane_change_deg, or anchor changes."""
        if self._last_maneuver_template is None:
            return
        # Re-run the last template to update preview dynamically
        template_map = {
            "hohmann": self._calc_hohmann,
            "bielliptic": self._calc_bielliptic,
            "plane_change": self._template_plane_change,
            "transfer_plane": self._template_transfer_plane,
            "circularize": self._template_circularize,
            "deorbit": self._template_deorbit,
            "phasing": self._template_phasing,
            "rendezvous": self._template_rendezvous,
        }
        fn = template_map.get(self._last_maneuver_template)
        if fn:
            fn()

    def _get_maneuver_start_time(self) -> float:
        """Get the fractional start time based on the maneuver_anchor dropdown."""
        anchor = str(dpg.get_value("maneuver_anchor"))
        if anchor == "Periapsis":
            return 0.0
        elif anchor == "Apoapsis":
            return 0.5
        elif anchor == "Ascending Node":
            omega = float(dpg.get_value("coe_omega"))
            theta0 = float(dpg.get_value("coe_theta"))
            target = (-omega) % 360.0
            frac = ((target - theta0) % 360.0) / 360.0
            if frac < 1e-6:
                frac = 1e-4
            return frac
        elif anchor == "Descending Node":
            omega = float(dpg.get_value("coe_omega"))
            theta0 = float(dpg.get_value("coe_theta"))
            target = (180.0 - omega) % 360.0
            frac = ((target - theta0) % 360.0) / 360.0
            if frac < 1e-6:
                frac = 1e-4
            return frac
        return 0.0

    def _calc_hohmann(self):
        self._last_maneuver_template = "hohmann"
        a = dpg.get_value("coe_a")
        e = dpg.get_value("coe_e")
        r1 = a * (1 - e) if e > 0.01 else a
        target_alt = dpg.get_value("target_alt")
        r2 = R_EARTH + target_alt
        result = hohmann_dv(r1, r2)
        self._post_burn_period_hint = orbital_period(r2, MU_EARTH)
        transfer = self._build_transfer_arc(result["r1"], result["r2"])
        if transfer is not None:
            self.viewport3d.clear_transfers()
            self.viewport3d.add_transfer_trajectory(transfer, color=(255, 220, 30, 240))
        burn_dir = "prograde" if r2 >= r1 else "retrograde"
        t_start = self._get_maneuver_start_time()
        self._set_maneuver_plan(
            [
                ManeuverEvent(time=t_start, dv_magnitude=result["dv1"], direction=burn_dir),
                ManeuverEvent(
                    time=result["dt_transfer"], dv_magnitude=result["dv2"], direction=burn_dir
                ),
            ]
        )
        msg = (
            f"Hohmann: dv1={result['dv1']:.4f} dv2={result['dv2']:.4f}\n"
            f"Total={result['dv_total']:.4f} km/s\n"
            f"Transit={result['dt_transfer'] / 3600:.2f} h"
        )
        dpg.set_value("maneuver_info", msg)
        self._log(f"Hohmann: dv_total={result['dv_total']:.4f} km/s (burn plan loaded)")

    def _calc_bielliptic(self):
        self._last_maneuver_template = "bielliptic"
        a = dpg.get_value("coe_a")
        e = dpg.get_value("coe_e")
        r1 = a * (1 - e) if e > 0.01 else a
        target_alt = dpg.get_value("target_alt")
        r2 = R_EARTH + target_alt
        r_b = max(r1, r2) * 2.5
        result = bielliptic_dv(r1, r2, r_b)
        self._post_burn_period_hint = orbital_period(r2, MU_EARTH)
        a1 = 0.5 * (r1 + r_b)
        a2 = 0.5 * (r2 + r_b)
        dt1 = orbital_period(a1, MU_EARTH) / 2.0
        dt2 = orbital_period(a2, MU_EARTH) / 2.0
        transfer_1 = self._build_transfer_arc(r1, r_b)
        transfer_2 = self._build_transfer_arc(r_b, r2)
        self.viewport3d.clear_transfers()
        if transfer_1 is not None:
            self.viewport3d.add_transfer_trajectory(transfer_1, color=(255, 210, 40, 230))
        if transfer_2 is not None:
            self.viewport3d.add_transfer_trajectory(transfer_2, color=(255, 140, 40, 230))
        t_start = self._get_maneuver_start_time()
        self._set_maneuver_plan(
            [
                ManeuverEvent(time=t_start, dv_magnitude=result["dv1"], direction="prograde"),
                ManeuverEvent(time=dt1, dv_magnitude=result["dv2"], direction="prograde"),
                ManeuverEvent(time=dt1 + dt2, dv_magnitude=result["dv3"], direction="retrograde"),
            ]
        )
        msg = (
            f"Bi-elliptic: dv1={result['dv1']:.4f}\n"
            f"dv2={result['dv2']:.4f} dv3={result['dv3']:.4f}\n"
            f"Total={result['dv_total']:.4f} km/s"
        )
        dpg.set_value("maneuver_info", msg)
        self._log(f"Bi-elliptic: dv_total={result['dv_total']:.4f} km/s (burn plan loaded)")

    def _template_plane_change(self):
        self._last_maneuver_template = "plane_change"
        a = dpg.get_value("coe_a")
        e = dpg.get_value("coe_e")
        delta_i = abs(float(dpg.get_value("plane_change_deg")))
        r = a * (1 - e * e)
        v_orb = np.sqrt(MU_EARTH / max(1.0, r))
        dv = plane_change_dv(v_orb, delta_i)
        self._post_burn_period_hint = orbital_period(a, MU_EARTH)
        t_start = self._get_maneuver_start_time()
        self._set_maneuver_plan([ManeuverEvent(time=t_start, dv_magnitude=dv, direction="normal")])
        dpg.set_value(
            "maneuver_info",
            f"Plane change template\ndI={delta_i:.2f} deg\ndv={dv:.4f} km/s",
        )
        self._log(f"Plane-change template loaded: dI={delta_i:.2f} dv={dv:.4f} km/s")

    def _template_transfer_plane(self):
        self._last_maneuver_template = "transfer_plane"
        a = dpg.get_value("coe_a")
        e = dpg.get_value("coe_e")
        r1 = a * (1 - e) if e > 0.01 else a
        target_alt = dpg.get_value("target_alt")
        delta_i = abs(float(dpg.get_value("plane_change_deg")))
        r2 = R_EARTH + target_alt

        result = hohmann_dv(r1, r2)
        self._post_burn_period_hint = orbital_period(r2, MU_EARTH)
        a_t = 0.5 * (r1 + r2)
        v_transfer_at_r2 = vis_viva(r2, a_t, MU_EARTH)
        v_circ_r2 = np.sqrt(MU_EARTH / r2)
        dv2_combined = combined_dv(v_transfer_at_r2, v_circ_r2, delta_i)
        burn_dir = "prograde" if r2 >= r1 else "retrograde"

        t_start = self._get_maneuver_start_time()
        self._set_maneuver_plan(
            [
                ManeuverEvent(time=t_start, dv_magnitude=result["dv1"], direction=burn_dir),
                ManeuverEvent(
                    time=result["dt_transfer"], dv_magnitude=dv2_combined, direction=burn_dir
                ),
            ]
        )
        dpg.set_value(
            "maneuver_info",
            f"Transfer+plane template\ndv1={result['dv1']:.4f} km/s\ndv2*={dv2_combined:.4f} km/s",
        )
        self._log(
            f"Transfer+plane template loaded: dv1={result['dv1']:.4f}, dv2*={dv2_combined:.4f} km/s"
        )

    def _build_absolute_maneuvers(self, a: float, n_orbits: float) -> list[ManeuverEvent]:
        single_orbit = orbital_period(a, MU_EARTH)
        total_duration = single_orbit * n_orbits
        burns: list[ManeuverEvent] = []
        for event in self._maneuver_plan:
            t = float(event.time)
            # Convention: values <= 50 are fractions of one initial orbit period
            # Values > 50 are absolute seconds (from templates like Hohmann)
            if t <= 50.0:
                t_abs = t * single_orbit
            else:
                t_abs = t
            if 0.0 <= t_abs <= total_duration:
                burns.append(
                    ManeuverEvent(
                        time=t_abs,
                        dv_magnitude=float(event.dv_magnitude),
                        direction=str(event.direction),
                    )
                )
        burns.sort(key=lambda b: b.time)
        return burns

    def _effective_n_orbits(self, a: float, n_orbits: float) -> float:
        if len(self._maneuver_plan) == 0:
            return n_orbits

        orbit_time = orbital_period(a, MU_EARTH)
        total_duration = orbit_time * n_orbits

        # Find time of last burn in absolute seconds
        max_time = 0.0
        for event in self._maneuver_plan:
            t = float(event.time)
            if t <= 50.0:
                t = t * orbit_time
            max_time = max(max_time, t)

        # Estimate post-burn orbit period by propagating through all burns
        e0 = float(dpg.get_value("coe_e"))
        i0 = float(dpg.get_value("coe_i"))
        raan0 = float(dpg.get_value("coe_raan"))
        omega0 = float(dpg.get_value("coe_omega"))
        theta0 = float(dpg.get_value("coe_theta"))
        coe = OrbitalElements(a=a, e=e0, i=i0, raan=raan0, omega=omega0, theta=theta0)
        sv = coe_to_state(coe, MU_EARTH)
        v_vec = np.array(sv.v)
        r_vec = np.array(sv.r)

        sorted_burns = sorted(self._maneuver_plan, key=lambda b: b.time)
        last_t = 0.0
        for burn in sorted_burns:
            bv = float(burn.time)
            bt = bv * orbit_time if bv <= 50.0 else bv
            dt = bt - last_t
            if dt > 0:
                r_mag = np.linalg.norm(r_vec)
                a_cur = 1.0 / (2.0 / r_mag - np.dot(v_vec, v_vec) / MU_EARTH)
                if a_cur > 0:
                    n_m = np.sqrt(MU_EARTH / a_cur**3)
                    cur_coe = state_to_coe(StateVector(r=r_vec, v=v_vec), MU_EARTH)
                    new_th = (cur_coe.theta + np.degrees(n_m * dt)) % 360.0
                    adv_coe = OrbitalElements(
                        a=cur_coe.a,
                        e=cur_coe.e,
                        i=cur_coe.i,
                        raan=cur_coe.raan,
                        omega=cur_coe.omega,
                        theta=new_th,
                    )
                    sv2 = coe_to_state(adv_coe, MU_EARTH)
                    r_vec = np.array(sv2.r)
                    v_vec = np.array(sv2.v)
            # Apply burn
            dv_mag = float(burn.dv_magnitude)
            direction = str(burn.direction)
            r_hat = r_vec / np.linalg.norm(r_vec)
            v_hat = v_vec / np.linalg.norm(v_vec)
            h_vec = np.cross(r_vec, v_vec)
            h_hat = h_vec / np.linalg.norm(h_vec)
            if direction == "prograde":
                dv_vec = v_hat * dv_mag
            elif direction == "retrograde":
                dv_vec = -v_hat * dv_mag
            elif direction == "normal":
                dv_vec = h_hat * dv_mag
            elif direction == "antinormal":
                dv_vec = -h_hat * dv_mag
            elif direction == "radial_out":
                dv_vec = r_hat * dv_mag
            elif direction == "radial_in":
                dv_vec = -r_hat * dv_mag
            else:
                dv_vec = v_hat * dv_mag
            v_vec = v_vec + dv_vec
            last_t = bt

        # Compute post-burn semi-major axis
        r_mag = np.linalg.norm(r_vec)
        v_mag = np.linalg.norm(v_vec)
        post_a = 1.0 / (2.0 / r_mag - v_mag**2 / MU_EARTH)
        if post_a > 0:
            post_period = orbital_period(post_a, MU_EARTH)
        else:
            post_period = orbit_time  # hyperbolic fallback

        # Ensure at least 1.5 full post-burn orbits after last burn
        desired_duration = max(total_duration, max_time + 2.5 * post_period)
        return max(n_orbits, desired_duration / orbit_time)

    def _on_burn_param_change(self, sender=None, app_data=None, user_data=None):
        """Live update: show preview of current burn params without committing."""
        self._update_burn_cursor()
        # Build a temporary maneuver plan with current slider values + existing burns
        dv = float(dpg.get_value("burn_dv"))
        if dv <= 0.0:
            # No burn to preview, just show existing plan
            if self._maneuver_plan:
                self._request_maneuver_preview()
            return
        direction = str(dpg.get_value("burn_dir"))
        anchor = str(dpg.get_value("burn_anchor"))
        t_frac = self._get_burn_time_frac(anchor)

        # Temporarily add this burn to the plan for preview
        temp_event = ManeuverEvent(time=t_frac, dv_magnitude=dv, direction=direction)
        original_plan = self._maneuver_plan.copy()
        self._maneuver_plan = sorted(original_plan + [temp_event], key=lambda e: e.time)
        self._request_maneuver_preview()
        self._maneuver_plan = original_plan

    def _request_maneuver_preview(self):
        if self.viewport3d is None:
            return

        self._preview_enabled = True
        self.preview_trajectory = None
        self.viewport3d.clear_preview_trajectory()

        if len(self._maneuver_plan) == 0:
            return

        a = float(dpg.get_value("coe_a"))
        e = float(dpg.get_value("coe_e"))
        i = float(dpg.get_value("coe_i"))
        raan = float(dpg.get_value("coe_raan"))
        omega = float(dpg.get_value("coe_omega"))
        theta = float(dpg.get_value("coe_theta"))
        n_orb = float(dpg.get_value("n_orbits"))
        n_orb = self._effective_n_orbits(a, n_orb)

        coe = OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, theta=theta)
        pert = PerturbationConfig(
            j2_enabled=dpg.get_value("pert_j2"),
            drag_enabled=dpg.get_value("pert_drag"),
            drag_cd=dpg.get_value("drag_cd"),
            drag_ballistic_coeff=dpg.get_value("drag_B"),
            third_body_moon=dpg.get_value("pert_moon"),
            third_body_sun=dpg.get_value("pert_sun"),
        )
        scenario = Scenario(
            initial_coe=coe,
            central_body=self._get_central_body(),
            perturbations=pert,
            n_orbits=n_orb,
            maneuvers=self._build_absolute_maneuvers(a, n_orb),
        )
        self.preview_engine.compute(scenario)

    def _set_maneuver_plan(self, events: list[ManeuverEvent]):
        self._maneuver_plan = sorted(events, key=lambda e: e.time)
        # Save to active SC
        active_name = dpg.get_value("active_sc_select")
        for sc in self._tab1_spacecraft:
            if sc["name"] == active_name:
                sc["maneuvers"] = list(self._maneuver_plan)
                break
        self._refresh_burn_plan_display()
        self._request_maneuver_preview()

    def _add_burn(self):
        dv = float(dpg.get_value("burn_dv"))
        direction = str(dpg.get_value("burn_dir"))
        anchor = str(dpg.get_value("burn_anchor"))
        if dv <= 0.0:
            self._log("Burn ignored: dv must be > 0.")
            return

        t_frac = self._get_burn_time_frac(anchor)

        self._maneuver_plan.append(
            ManeuverEvent(
                time=t_frac,
                dv_magnitude=dv,
                direction=direction,
            )
        )
        self._maneuver_plan.sort(key=lambda e: e.time)
        self._refresh_burn_plan_display()
        # Clear preview after committing a burn
        self._preview_enabled = False
        self.viewport3d.clear_preview_trajectory()
        self.viewport3d.clear_transfers()
        self.viewport3d.set_burn_cursor(None)
        # Save burns to active SC
        active_name = dpg.get_value("active_sc_select")
        for sc in self._tab1_spacecraft:
            if sc["name"] == active_name:
                sc["maneuvers"] = list(self._maneuver_plan)
                break
        self._log(
            f"Added burn: t={t_frac * 100.0:.1f}% ({anchor}) dv={dv:.3f} km/s dir={direction}"
        )
        # Auto-compute to show result
        self._on_compute()

    def _update_burn_cursor(self, sender=None, app_data=None):
        """Update the burn cursor marker on the 3D viewport."""
        if self.viewport3d is None:
            return
        try:
            anchor = str(dpg.get_value("burn_anchor"))
            direction = str(dpg.get_value("burn_dir"))

            # For "At current time", use exact trajectory position to match SC dot
            if anchor == "At current time" and self.trajectory is not None and len(self.trajectory.t) > 0:
                idx = int(np.searchsorted(self.trajectory.t, self.current_time))
                idx = min(idx, len(self.trajectory.r) - 1)
                r_vec = np.array(self.trajectory.r[idx])
                v_vec = np.array(self.trajectory.v[idx])
                pos = r_vec
                # Compute direction vector for arrow
                r_hat = r_vec / np.linalg.norm(r_vec)
                v_hat = v_vec / np.linalg.norm(v_vec)
                h_vec = np.cross(r_vec, v_vec)
                h_hat = h_vec / np.linalg.norm(h_vec)
                dir_map = {
                    "prograde": v_hat,
                    "retrograde": -v_hat,
                    "normal": h_hat,
                    "antinormal": -h_hat,
                    "radial_out": r_hat,
                    "radial_in": -r_hat,
                }
                dir_vec = dir_map.get(direction, v_hat)
                self.viewport3d.set_burn_cursor(pos, dir_vec)
                return

            t_frac = self._get_burn_time_frac(anchor)

            # Get position on orbit at t_frac, accounting for prior burns
            a = float(dpg.get_value("coe_a"))
            e = float(dpg.get_value("coe_e"))
            i = float(dpg.get_value("coe_i"))
            raan = float(dpg.get_value("coe_raan"))
            omega = float(dpg.get_value("coe_omega"))
            theta0 = float(dpg.get_value("coe_theta"))

            initial_period = orbital_period(a, MU_EARTH)
            t_abs = t_frac * initial_period

            # Start from initial state
            coe = OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, theta=theta0)
            sv = coe_to_state(coe, MU_EARTH)
            r_vec = np.array(sv.r)
            v_vec = np.array(sv.v)

            # Propagate through existing burns before this time
            last_t = 0.0
            sorted_burns = sorted(self._maneuver_plan, key=lambda b: b.time)
            for burn in sorted_burns:
                bv = float(burn.time)
                bt = bv * initial_period if bv <= 50.0 else bv
                if bt >= t_abs:
                    break
                dt = bt - last_t
                if dt > 0:
                    r_mag = np.linalg.norm(r_vec)
                    a_cur = 1.0 / (2.0 / r_mag - np.dot(v_vec, v_vec) / MU_EARTH)
                    if a_cur > 0:
                        n_m = np.sqrt(MU_EARTH / a_cur**3)
                        cur_coe = state_to_coe(StateVector(r=r_vec, v=v_vec), MU_EARTH)
                        new_th = (cur_coe.theta + np.degrees(n_m * dt)) % 360.0
                        adv_coe = OrbitalElements(
                            a=cur_coe.a,
                            e=cur_coe.e,
                            i=cur_coe.i,
                            raan=cur_coe.raan,
                            omega=cur_coe.omega,
                            theta=new_th,
                        )
                        sv2 = coe_to_state(adv_coe, MU_EARTH)
                        r_vec = np.array(sv2.r)
                        v_vec = np.array(sv2.v)
                # Apply burn
                dv_m = float(burn.dv_magnitude)
                d = str(burn.direction)
                rh = r_vec / np.linalg.norm(r_vec)
                vh = v_vec / np.linalg.norm(v_vec)
                hv = np.cross(r_vec, v_vec)
                hh = hv / np.linalg.norm(hv)
                if d == "prograde":
                    dvv = vh * dv_m
                elif d == "retrograde":
                    dvv = -vh * dv_m
                elif d == "normal":
                    dvv = hh * dv_m
                elif d == "antinormal":
                    dvv = -hh * dv_m
                elif d == "radial_out":
                    dvv = rh * dv_m
                elif d == "radial_in":
                    dvv = -rh * dv_m
                else:
                    dvv = vh * dv_m
                v_vec = v_vec + dvv
                last_t = bt

            # Propagate remaining time to burn point
            dt_remaining = t_abs - last_t
            if dt_remaining > 0:
                r_mag = np.linalg.norm(r_vec)
                a_cur = 1.0 / (2.0 / r_mag - np.dot(v_vec, v_vec) / MU_EARTH)
                if a_cur > 0:
                    n_m = np.sqrt(MU_EARTH / a_cur**3)
                    cur_coe = state_to_coe(StateVector(r=r_vec, v=v_vec), MU_EARTH)
                    new_th = (cur_coe.theta + np.degrees(n_m * dt_remaining)) % 360.0
                    adv_coe = OrbitalElements(
                        a=cur_coe.a,
                        e=cur_coe.e,
                        i=cur_coe.i,
                        raan=cur_coe.raan,
                        omega=cur_coe.omega,
                        theta=new_th,
                    )
                    sv2 = coe_to_state(adv_coe, MU_EARTH)
                    r_vec = np.array(sv2.r)
                    v_vec = np.array(sv2.v)

            pos = r_vec

            # Compute direction vector for arrow
            r_hat = r_vec / np.linalg.norm(r_vec)
            v_hat = v_vec / np.linalg.norm(v_vec)
            h_vec = np.cross(r_vec, v_vec)
            h_hat = h_vec / np.linalg.norm(h_vec)
            if direction == "prograde":
                d_hat = v_hat
            elif direction == "retrograde":
                d_hat = -v_hat
            elif direction == "normal":
                d_hat = h_hat
            elif direction == "antinormal":
                d_hat = -h_hat
            elif direction == "radial_out":
                d_hat = r_hat
            elif direction == "radial_in":
                d_hat = -r_hat
            else:
                d_hat = v_hat

            # Arrow length in viewport units (fixed visual size)
            arrow_len = np.linalg.norm(pos) * 0.15
            dir_vec = d_hat * arrow_len

            self.viewport3d.set_burn_cursor(pos, dir_vec)
        except Exception:
            self.viewport3d.set_burn_cursor(None)

    def _clear_burns(self):
        self._maneuver_plan.clear()
        self._refresh_burn_plan_display()
        self._request_maneuver_preview()
        self._log("Cleared planned burns.")

    def _clear_preview_only(self):
        self._preview_enabled = False
        self.viewport3d.clear_preview_trajectory()
        self.viewport3d.clear_transfers()
        self.viewport3d.set_burn_cursor(None)
        self._log("Cleared preview overlays.")

    def _on_escape(self, sender=None, app_data=None):
        """Escape key: clear burn preview and deselect all orbits."""
        self._clear_preview_only()
        # Deselect orbit
        if self.viewport3d:
            self.viewport3d._selected_orbit = None
            self.viewport3d._needs_redraw = True
        self.trajectory = None
        self._active_sc_name = None
        dpg.set_value("active_sc_select", "")
        # Clear telemetry and plots
        for tag in ("tel_alt", "tel_vel", "tel_period", "tel_energy", "tel_a", "tel_e", "tel_i", "tel_raan", "tel_omega", "tel_theta"):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, "—")
        for tag in ("alt_series", "vel_series", "alt_cursor", "vel_cursor"):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self._update_burn_events_display([])

    def _fit_view(self):
        if self.viewport3d:
            self.viewport3d.fit_to_content()

    def _remove_burn(self):
        idx = int(dpg.get_value("burn_remove_idx")) - 1
        if 0 <= idx < len(self._maneuver_plan):
            removed = self._maneuver_plan.pop(idx)
            self._refresh_burn_plan_display()
            self._request_maneuver_preview()
            self._log(f"Removed burn #{idx + 1}: dv={removed.dv_magnitude:.4f} {removed.direction}")
        else:
            self._log(f"No burn #{idx + 1} to remove.")

    def _refresh_burn_plan_display(self):
        if len(self._maneuver_plan) == 0:
            dpg.set_value("burn_plan_text", "No planned burns.")
            return

        lines = []
        for idx, burn in enumerate(self._maneuver_plan, start=1):
            if burn.time <= 50.0:
                when = f"{burn.time * 100.0:5.1f}%"
            else:
                when = f"{burn.time:7.1f}s"
            lines.append(f"{idx:02d}. t={when}  dv={burn.dv_magnitude:.4f}  {burn.direction}")
        dpg.set_value("burn_plan_text", "\n".join(lines))

    def _get_burn_time_frac(self, anchor: str) -> float:
        """Get burn time as fraction of initial orbit period based on anchor mode."""
        if anchor == "At current time":
            a = float(dpg.get_value("coe_a"))
            period = orbital_period(a, MU_EARTH)
            if period > 0:
                return self.current_time / period
            return 0.0
        return self._burn_time_from_anchor(anchor)

    def _burn_time_from_anchor(self, anchor: str) -> float:
        """Compute the fractional time (0-1 of one initial orbit) for an anchored burn.

        For chained burns, propagates through previous burns to find the
        correct anchor point on the post-burn orbit.
        """
        a0 = float(dpg.get_value("coe_a"))
        e0 = float(dpg.get_value("coe_e"))
        i0 = float(dpg.get_value("coe_i"))
        raan0 = float(dpg.get_value("coe_raan"))
        omega0 = float(dpg.get_value("coe_omega"))
        theta0 = float(dpg.get_value("coe_theta"))

        initial_period = orbital_period(a0, MU_EARTH)

        if len(self._maneuver_plan) == 0:
            # First burn: compute fraction based on initial orbit
            target_theta = self._anchor_to_theta(anchor, omega0)
            dtheta = (target_theta - theta0) % 360.0
            frac = dtheta / 360.0
            if frac < 1e-6:
                frac = 1e-4
            return frac

        # Chained burn: propagate through previous burns to get post-burn state,
        # then find target anchor on the resulting orbit.
        coe = OrbitalElements(a=a0, e=e0, i=i0, raan=raan0, omega=omega0, theta=theta0)
        sv = coe_to_state(coe, MU_EARTH)
        r_vec = np.array(sv.r)
        v_vec = np.array(sv.v)

        last_t_abs = 0.0  # absolute time in seconds
        sorted_burns = sorted(self._maneuver_plan, key=lambda b: b.time)
        for existing_burn in sorted_burns:
            t_val = float(existing_burn.time)
            t_abs = t_val * initial_period if t_val <= 50.0 else t_val
            # Propagate Keplerian to this time
            dt = t_abs - last_t_abs
            if dt > 0:
                r_mag = np.linalg.norm(r_vec)
                a_cur = 1.0 / (2.0 / r_mag - np.dot(v_vec, v_vec) / MU_EARTH)
                if a_cur > 0:
                    n_motion = np.sqrt(MU_EARTH / a_cur**3)
                    dM = n_motion * dt
                    cur_coe = state_to_coe(StateVector(r=r_vec, v=v_vec), MU_EARTH)
                    new_theta = (cur_coe.theta + np.degrees(dM)) % 360.0
                    advanced_coe = OrbitalElements(
                        a=cur_coe.a,
                        e=cur_coe.e,
                        i=cur_coe.i,
                        raan=cur_coe.raan,
                        omega=cur_coe.omega,
                        theta=new_theta,
                    )
                    sv2 = coe_to_state(advanced_coe, MU_EARTH)
                    r_vec = np.array(sv2.r)
                    v_vec = np.array(sv2.v)

            # Apply burn
            dv_mag = float(existing_burn.dv_magnitude)
            direction = str(existing_burn.direction)
            r_hat = r_vec / np.linalg.norm(r_vec)
            v_hat = v_vec / np.linalg.norm(v_vec)
            h_vec = np.cross(r_vec, v_vec)
            h_hat = h_vec / np.linalg.norm(h_vec)
            if direction == "prograde":
                dv_vec = v_hat * dv_mag
            elif direction == "retrograde":
                dv_vec = -v_hat * dv_mag
            elif direction == "normal":
                dv_vec = h_hat * dv_mag
            elif direction == "antinormal":
                dv_vec = -h_hat * dv_mag
            elif direction == "radial_out":
                dv_vec = r_hat * dv_mag
            elif direction == "radial_in":
                dv_vec = -r_hat * dv_mag
            else:
                dv_vec = v_hat * dv_mag
            v_vec = v_vec + dv_vec
            last_t_abs = t_abs

        # Now r_vec, v_vec is the post-burn state. Find new COE.
        post_coe = state_to_coe(StateVector(r=r_vec, v=v_vec), MU_EARTH)
        target_theta = self._anchor_to_theta(anchor, post_coe.omega)

        # Time from last burn to reach target theta on post-burn orbit
        dtheta = (target_theta - post_coe.theta) % 360.0
        if dtheta < 1e-6:
            dtheta = 0.001

        post_a = post_coe.a if post_coe.a > 0 else a0
        post_period = orbital_period(post_a, MU_EARTH)
        dt_to_target = (dtheta / 360.0) * post_period
        t_target_abs = last_t_abs + dt_to_target
        # Return as fraction of initial orbit period
        frac = t_target_abs / initial_period
        return frac

    def _anchor_to_theta(self, anchor: str, omega: float) -> float:
        if anchor == "Periapsis":
            return 0.0
        elif anchor == "Apoapsis":
            return 180.0
        elif anchor == "Ascending Node":
            return (-omega) % 360.0
        elif anchor == "Descending Node":
            return (180.0 - omega) % 360.0
        return 0.0

    def _update_burn_events_display(self, burn_events: list[dict]):
        if len(burn_events) == 0:
            dpg.set_value("burn_events_text", "No executed burns.")
            return

        rows = []
        dv_total = 0.0
        for idx, ev in enumerate(burn_events, start=1):
            dv_total += ev["dv"]
            rows.append(f"{idx:02d}. t={ev['time']:.1f}s dv={ev['dv']:.4f} {ev['direction']}")
            rows.append(f"    a={ev['a']:.1f}km e={ev['e']:.5f} i={ev['i']:.2f}deg")

        rows.append(f"Total dv={dv_total:.4f} km/s")
        dpg.set_value("burn_events_text", "\n".join(rows))

    def _update_compare_metrics(self):
        if self.trajectory is None or self.compare_trajectory is None:
            if dpg.get_value("compare_mode"):
                dpg.set_value("compare_text", "Waiting for both trajectories...")
            return

        r_main = self.trajectory.r[-1]
        v_main = self.trajectory.v[-1]
        r_ref = self.compare_trajectory.r[-1]
        v_ref = self.compare_trajectory.v[-1]

        alt_main = np.linalg.norm(r_main) - R_EARTH
        alt_ref = np.linalg.norm(r_ref) - R_EARTH
        d_alt = alt_main - alt_ref
        d_speed = np.linalg.norm(v_main) - np.linalg.norm(v_ref)
        miss = np.linalg.norm(r_main - r_ref)

        dpg.set_value(
            "compare_text",
            f"Final dAlt={d_alt:+.2f} km\n"
            f"Final dV={d_speed:+.5f} km/s\n"
            f"Position miss={miss:.2f} km",
        )

    def _build_transfer_arc(
        self, r_start: float, r_end: float, n_points: int = 420
    ) -> TrajectoryData | None:
        if n_points < 8:
            n_points = 8

        i = dpg.get_value("coe_i")
        raan = dpg.get_value("coe_raan")
        omega = dpg.get_value("coe_omega")
        a_t = (r_start + r_end) / 2.0
        e_t = abs(r_end - r_start) / (r_start + r_end)

        if r_end >= r_start:
            theta_0 = 0.0
            theta_1 = 180.0
        else:
            theta_0 = 180.0
            theta_1 = 360.0

        theta = np.linspace(theta_0, theta_1, n_points)
        t = np.linspace(0.0, orbital_period(a_t, MU_EARTH) / 2.0, n_points)

        r_hist = np.zeros((n_points, 3))
        v_hist = np.zeros((n_points, 3))
        for idx, th in enumerate(theta):
            sv = coe_to_state(OrbitalElements(a=a_t, e=e_t, i=i, raan=raan, omega=omega, theta=th))
            r_hist[idx] = sv.r
            v_hist[idx] = sv.v

        return TrajectoryData(t=t, r=r_hist, v=v_hist)

    def _export_csv(self):
        if self.trajectory is None:
            self._log("No trajectory to export.")
            return
        from simulator.export.csv_export import export_trajectory_csv

        path = export_trajectory_csv(self.trajectory)
        dpg.set_value("export_status", f"Saved: {os.path.basename(path)}")
        self._log(f"Exported CSV: {path}")

    def _export_plots(self):
        if self.trajectory is None:
            self._log("No trajectory to export.")
            return
        from simulator.export.mpl_plots import export_plots

        path = export_plots(self.trajectory)
        dpg.set_value("export_status", f"Saved: {os.path.basename(path)}")
        self._log(f"Exported plots: {path}")

    def _preset_iss(self):
        dpg.set_value("coe_a", 6779.0)
        dpg.set_value("coe_e", 0.0001)
        dpg.set_value("coe_i", 51.6)
        dpg.set_value("coe_raan", 0.0)
        dpg.set_value("coe_omega", 0.0)
        dpg.set_value("coe_theta", 0.0)
        dpg.set_value("n_orbits", 3.0)
        self._on_coe_change()

    def _preset_geo(self):
        dpg.set_value("coe_a", 42157.0)
        dpg.set_value("coe_e", 0.0001)
        dpg.set_value("coe_i", 0.0)
        dpg.set_value("coe_raan", 0.0)
        dpg.set_value("coe_omega", 0.0)
        dpg.set_value("coe_theta", 0.0)
        dpg.set_value("n_orbits", 1.0)
        self._on_coe_change()

    def _preset_molniya(self):
        rp = R_EARTH + 600.0
        ra = R_EARTH + 39000.0
        a = (rp + ra) / 2.0
        e = (ra - rp) / (ra + rp)
        dpg.set_value("coe_a", a)
        dpg.set_value("coe_e", e)
        dpg.set_value("coe_i", 63.4)
        dpg.set_value("coe_raan", 0.0)
        dpg.set_value("coe_omega", 270.0)
        dpg.set_value("coe_theta", 0.0)
        dpg.set_value("n_orbits", 2.0)
        self._on_coe_change()

    def _preset_sso(self):
        dpg.set_value("coe_a", 6971.0)
        dpg.set_value("coe_e", 0.0001)
        dpg.set_value("coe_i", 97.8)
        dpg.set_value("coe_raan", 0.0)
        dpg.set_value("coe_omega", 0.0)
        dpg.set_value("coe_theta", 0.0)
        dpg.set_value("n_orbits", 5.0)
        self._on_coe_change()

    def _preset_heo(self):
        rp = R_EARTH + 300.0
        ra = R_EARTH + 60000.0
        a = (rp + ra) / 2.0
        e = (ra - rp) / (ra + rp)
        dpg.set_value("coe_a", a)
        dpg.set_value("coe_e", e)
        dpg.set_value("coe_i", 28.5)
        dpg.set_value("coe_raan", 0.0)
        dpg.set_value("coe_omega", 0.0)
        dpg.set_value("coe_theta", 0.0)
        dpg.set_value("n_orbits", 1.0)
        self._on_coe_change()

    def _preset_gps(self):
        """GPS constellation: MEO at 20,200 km altitude, 55° inclination."""
        a = R_EARTH + 20200.0
        dpg.set_value("coe_a", a)
        dpg.set_value("coe_e", 0.0)
        dpg.set_value("coe_i", 55.0)
        dpg.set_value("coe_raan", 0.0)
        dpg.set_value("coe_omega", 0.0)
        dpg.set_value("coe_theta", 0.0)
        dpg.set_value("n_orbits", 2.0)
        self._on_coe_change()

    def _preset_tundra(self):
        """Tundra orbit: 24h period, e=0.268, i=63.4° (Russian comms)."""
        a = 42164.0  # ~24h period
        e = 0.268
        dpg.set_value("coe_a", a)
        dpg.set_value("coe_e", e)
        dpg.set_value("coe_i", 63.4)
        dpg.set_value("coe_raan", 90.0)
        dpg.set_value("coe_omega", 270.0)
        dpg.set_value("coe_theta", 0.0)
        dpg.set_value("n_orbits", 1.0)
        self._on_coe_change()

    def _preset_lunar(self):
        """Trans-Lunar Injection: GTO-like orbit reaching Moon distance."""
        rp = R_EARTH + 185.0
        ra = 384400.0  # Moon distance
        a = (rp + ra) / 2.0
        e = (ra - rp) / (ra + rp)
        dpg.set_value("coe_a", a)
        dpg.set_value("coe_e", e)
        dpg.set_value("coe_i", 28.5)
        dpg.set_value("coe_raan", 0.0)
        dpg.set_value("coe_omega", 0.0)
        dpg.set_value("coe_theta", 0.0)
        dpg.set_value("n_orbits", 1.0)
        self._on_coe_change()

    def _preset_gto(self):
        """Geostationary Transfer Orbit: 185km x 35,786km."""
        rp = R_EARTH + 185.0
        ra = R_EARTH + 35786.0
        a = (rp + ra) / 2.0
        e = (ra - rp) / (ra + rp)
        dpg.set_value("coe_a", a)
        dpg.set_value("coe_e", e)
        dpg.set_value("coe_i", 28.5)
        dpg.set_value("coe_raan", 0.0)
        dpg.set_value("coe_omega", 180.0)
        dpg.set_value("coe_theta", 0.0)
        dpg.set_value("n_orbits", 2.0)
        self._on_coe_change()

    def _template_circularize(self):
        """Add a prograde burn at apoapsis to circularize the orbit."""
        self._last_maneuver_template = "circularize"
        e = float(dpg.get_value("coe_e"))
        a = float(dpg.get_value("coe_a"))
        if e < 0.001:
            self._log("Orbit is already circular.")
            return
        body = self._get_central_body()
        ra = a * (1 + e)
        # Velocity at apoapsis on current orbit
        v_apo = np.sqrt(body.mu * (2.0 / ra - 1.0 / a))
        # Velocity for circular orbit at ra
        v_circ = np.sqrt(body.mu / ra)
        dv = v_circ - v_apo
        t_start = self._get_maneuver_start_time()
        self._set_maneuver_plan(
            [ManeuverEvent(time=t_start, dv_magnitude=abs(dv), direction="prograde")]
        )
        dpg.set_value(
            "maneuver_info",
            f"Circularize at apoapsis: Δv = {abs(dv):.4f} km/s\nFinal orbit: {ra - body.radius:.0f} km circular",
        )
        self._log(f"Circularize: dv={abs(dv):.4f} km/s at apoapsis")

    def _template_deorbit(self):
        """Add a retrograde burn to lower periapsis to ~80 km (re-entry)."""
        self._last_maneuver_template = "deorbit"
        a = float(dpg.get_value("coe_a"))
        e = float(dpg.get_value("coe_e"))
        body = self._get_central_body()
        ra = a * (1 + e)
        rp_target = body.radius + 80.0  # 80 km altitude for re-entry
        a_new = (ra + rp_target) / 2.0
        # Current velocity at apoapsis
        v_apo = np.sqrt(body.mu * (2.0 / ra - 1.0 / a))
        # Required velocity at apoapsis for new orbit
        v_new = np.sqrt(body.mu * (2.0 / ra - 1.0 / a_new))
        dv = v_apo - v_new
        t_start = self._get_maneuver_start_time()
        self._set_maneuver_plan([ManeuverEvent(time=t_start, dv_magnitude=dv, direction="retrograde")])
        dpg.set_value(
            "maneuver_info",
            f"De-orbit burn: Δv = {dv:.4f} km/s\nNew periapsis: {rp_target - body.radius:.0f} km (re-entry)",
        )
        self._log(f"De-orbit: dv={dv:.4f} km/s retrograde at apoapsis")

    def _template_phasing(self):
        """Phasing maneuver: adjust period to drift ahead/behind by 30°."""
        self._last_maneuver_template = "phasing"
        a = float(dpg.get_value("coe_a"))
        body = self._get_central_body()
        T = orbital_period(a, body.mu)
        # Drift 30° in one orbit → change period by T * (30/360)
        dT = T * (30.0 / 360.0)
        T_new = T - dT  # Shorter period to catch up
        a_new = (body.mu * (T_new / (2 * np.pi)) ** 2) ** (1.0 / 3.0)
        # Hohmann-like burn to phasing orbit
        v1 = np.sqrt(body.mu / a)
        v_transfer = np.sqrt(body.mu * (2.0 / a - 1.0 / a_new))
        dv = abs(v_transfer - v1)
        t_start = self._get_maneuver_start_time()
        self._set_maneuver_plan(
            [
                ManeuverEvent(time=t_start, dv_magnitude=dv, direction="retrograde"),
                ManeuverEvent(time=0.5, dv_magnitude=dv, direction="prograde"),
            ]
        )
        dpg.set_value("maneuver_info", f"Phasing: 2 burns × {dv:.4f} km/s\nDrift: 30° in 1 orbit")
        self._log(f"Phasing maneuver: dv={dv:.4f} km/s each")

    def _template_rendezvous(self):
        """Co-planar rendezvous: phasing + Hohmann to target altitude."""
        self._last_maneuver_template = "rendezvous"
        a = float(dpg.get_value("coe_a"))
        target_alt = float(dpg.get_value("target_alt"))
        body = self._get_central_body()
        r_target = body.radius + target_alt
        result = hohmann_dv(a, r_target)
        dv1 = result["dv1"]
        dv2 = result["dv2"]
        t_start = self._get_maneuver_start_time()
        # Add a phasing component
        self._set_maneuver_plan(
            [
                ManeuverEvent(time=t_start, dv_magnitude=dv1, direction="prograde"),
                ManeuverEvent(time=0.75, dv_magnitude=dv2, direction="prograde"),
            ]
        )
        dpg.set_value(
            "maneuver_info",
            f"Rendezvous via Hohmann:\nBurn 1: {dv1:.4f} km/s\nBurn 2: {dv2:.4f} km/s\n"
            f"Total Δv: {dv1 + dv2:.4f} km/s",
        )
        self._log(f"Rendezvous: Hohmann to {target_alt:.0f} km")

    # ─── File Menu ─────────────────────────────────────────────────────────

    def _file_new(self):
        """Reset to empty project, with save prompt if needed."""
        if self._tab1_spacecraft:
            self._pending_action = "new"
            self._show_save_prompt()
            return
        self._do_file_new()

    def _show_save_prompt(self):
        """Show a modal asking to save before destructive action."""
        if dpg.does_item_exist("save_prompt_window"):
            dpg.delete_item("save_prompt_window")
        with dpg.window(label="Save Changes?", tag="save_prompt_window", modal=True, width=320, height=120, no_resize=True):
            dpg.add_text("You have unsaved changes. Save before continuing?")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self._save_prompt_save, width=80)
                dpg.add_button(label="Don't Save", callback=self._save_prompt_discard, width=100)
                dpg.add_button(label="Cancel", callback=self._save_prompt_cancel, width=80)

    def _save_prompt_save(self):
        dpg.delete_item("save_prompt_window")
        self._file_save()
        self._execute_pending_action()

    def _save_prompt_discard(self):
        dpg.delete_item("save_prompt_window")
        self._execute_pending_action()

    def _save_prompt_cancel(self):
        dpg.delete_item("save_prompt_window")
        self._pending_action = None

    def _execute_pending_action(self):
        action = getattr(self, "_pending_action", None)
        self._pending_action = None
        if action == "new":
            self._do_file_new()
        elif action == "open":
            self._do_file_open()

    def _do_file_new(self):
        """Reset to empty project."""
        self._tab1_spacecraft.clear()
        self._maneuver_plan.clear()
        self.trajectory = None
        self.compare_trajectory = None
        self._tab1_multi_result = None
        self._active_sc_name = None
        self._current_file_path = None
        # Reset engines to prevent stale results from being picked up
        self.engine = SimulationEngine()
        self.compare_engine = SimulationEngine()
        self.preview_engine = SimulationEngine()
        self.preview_trajectory = None
        if self.viewport3d:
            self.viewport3d.set_multi_trajectories([])
            self.viewport3d.clear_burn_markers()
            self.viewport3d.clear_preview_trajectory()
            self.viewport3d.clear_preview_orbit()
            self.viewport3d.set_burn_cursor(None)
            self.viewport3d.set_reference_trajectory(None)
            self.viewport3d.clear_closest_approach_line()
            self.viewport3d.clear_orbit_paths()
            self.viewport3d._orbit_points = None
            self.viewport3d._trajectory = None
            self.viewport3d._selected_orbit = None
            self.viewport3d._needs_redraw = True
        self._refresh_burn_plan_display()
        self._update_burn_events_display([])
        dpg.configure_item("active_sc_select", items=[])
        dpg.set_value("active_sc_select", "")
        dpg.set_value("time_slider", 0.0)
        self.current_time = 0.0
        self._update_compare_metrics()
        # Clear telemetry
        for tag in ("tel_alt", "tel_vel", "tel_period", "tel_energy", "tel_a", "tel_e", "tel_i", "tel_raan", "tel_omega", "tel_theta"):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, "—")
        # Clear 2D plots
        for tag in ("alt_series", "vel_series", "alt_cursor", "vel_cursor"):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        for ax in ("alt_x", "alt_y", "vel_x", "vel_y"):
            if dpg.does_item_exist(ax):
                dpg.fit_axis_data(ax)
        self._log("New project created.")

    def _file_open(self):
        """Open a project file."""
        if self._tab1_spacecraft:
            self._pending_action = "open"
            self._show_save_prompt()
            return
        self._do_file_open()

    def _do_file_open(self):
        dpg.add_file_dialog(
            label="Open Project",
            callback=self._file_open_callback,
            cancel_callback=lambda *a: None,
            width=600,
            height=400,
            default_path=os.getcwd(),
            modal=True,
        )
        dpg.add_file_extension(".json", parent=dpg.last_container())

    def _file_open_callback(self, sender, app_data):
        file_path = app_data.get("file_path_name", "")
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self._load_project_data(data)
            self._current_file_path = file_path
            self._log(f"Loaded: {os.path.basename(file_path)}")
        except Exception as ex:
            self._log(f"Load error: {ex}")

    def _file_save(self):
        """Save to current file, or Save As if no file yet."""
        if getattr(self, "_current_file_path", None):
            self._save_to_file(self._current_file_path)
        else:
            self._file_save_as()

    def _file_save_as(self):
        """Save project to a new file."""
        dpg.add_file_dialog(
            label="Save Project As",
            callback=self._file_save_as_callback,
            cancel_callback=lambda *a: None,
            width=600,
            height=400,
            default_path=os.getcwd(),
            modal=True,
            directory_selector=False,
        )
        dpg.add_file_extension(".json", parent=dpg.last_container())

    def _file_save_as_callback(self, sender, app_data):
        file_path = app_data.get("file_path_name", "")
        if not file_path:
            return
        if not file_path.endswith(".json"):
            file_path += ".json"
        self._save_to_file(file_path)
        self._current_file_path = file_path

    def _save_to_file(self, file_path: str):
        """Serialize project state to JSON."""
        data = {
            "central_body": dpg.get_value("central_body_select") if dpg.does_item_exist("central_body_select") else "Earth",
            "spacecraft": [],
        }
        for sc in self._tab1_spacecraft:
            coe = sc["coe"]
            sc_data = {
                "name": sc["name"],
                "color": list(sc["color"]),
                "coe": {"a": coe.a, "e": coe.e, "i": coe.i, "raan": coe.raan, "omega": coe.omega, "theta": coe.theta},
                "maneuvers": [
                    {"time": m.time, "dv_magnitude": m.dv_magnitude, "direction": m.direction}
                    for m in sc.get("maneuvers", [])
                ],
            }
            data["spacecraft"].append(sc_data)
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            self._log(f"Saved: {os.path.basename(file_path)}")
        except Exception as ex:
            self._log(f"Save error: {ex}")

    def _load_project_data(self, data: dict):
        """Load project from parsed JSON data."""
        self._file_new()
        if "central_body" in data and dpg.does_item_exist("central_body_select"):
            dpg.set_value("central_body_select", data["central_body"])
        for sc_data in data.get("spacecraft", []):
            name = sc_data["name"]
            color = tuple(sc_data.get("color", [0, 191, 255]))
            coe_d = sc_data["coe"]
            coe = OrbitalElements(
                a=coe_d["a"], e=coe_d["e"], i=coe_d["i"],
                raan=coe_d["raan"], omega=coe_d["omega"], theta=coe_d["theta"],
            )
            maneuvers = [
                ManeuverEvent(time=m["time"], dv_magnitude=m["dv_magnitude"], direction=m["direction"])
                for m in sc_data.get("maneuvers", [])
            ]
            sc = {"name": name, "color": color, "coe": coe, "maneuvers": maneuvers, "trajectory": None, "burn_events": []}
            self._tab1_spacecraft.append(sc)
        # Update UI
        names = [sc["name"] for sc in self._tab1_spacecraft]
        dpg.configure_item("active_sc_select", items=names)
        if names:
            dpg.set_value("active_sc_select", names[0])
            self._select_sc_by_name(names[0])
            self._fit_after_compute = True
            self._on_compute()

    def _file_exit(self):
        dpg.stop_dearpygui()

    def _show_about(self):
        if dpg.does_item_exist("about_window"):
            dpg.delete_item("about_window")
        with dpg.window(label="About", tag="about_window", modal=True, width=350, height=150, no_resize=True):
            dpg.add_text("Orbital Mechanics Simulator")
            dpg.add_text("A DearPyGui-based orbital simulation tool.")
            dpg.add_text("Version 1.0")
            dpg.add_separator()
            dpg.add_button(label="Close", callback=lambda: dpg.delete_item("about_window"), width=80)


def run():
    app = App()
    app.run()
