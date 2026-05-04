import numpy as np
import dearpygui.dearpygui as dpg

from simulator.render.camera import Camera
from simulator.render.projection import perspective_matrix, project_points
from simulator.sim.trajectory import TrajectoryData
from simulator.core.constants import R_EARTH


CONTINENT_OUTLINES = [
    # Europe + North Africa
    [
        (40, -10),
        (44, 0),
        (48, 10),
        (52, 8),
        (55, 20),
        (60, 30),
        (64, 40),
        (68, 50),
        (66, 58),
        (58, 62),
        (52, 56),
        (46, 44),
        (42, 30),
        (36, 20),
        (34, 10),
        (36, 0),
        (40, -10),
    ],
    # Africa
    [
        (34, -18),
        (36, -10),
        (34, -2),
        (30, 5),
        (25, 10),
        (18, 14),
        (10, 20),
        (4, 14),
        (-2, 10),
        (-8, 14),
        (-14, 18),
        (-22, 18),
        (-30, 20),
        (-34, 12),
        (-30, 4),
        (-24, -2),
        (-18, -8),
        (-8, -2),
        (2, -8),
        (14, -14),
        (22, -16),
        (30, -14),
        (34, -18),
    ],
    # North America
    [
        (50, -165),
        (58, -150),
        (65, -140),
        (72, -120),
        (70, -100),
        (62, -90),
        (55, -80),
        (48, -70),
        (40, -65),
        (32, -75),
        (28, -88),
        (24, -100),
        (28, -114),
        (34, -126),
        (40, -136),
        (46, -150),
        (50, -165),
    ],
    # South America
    [
        (12, -80),
        (8, -74),
        (2, -66),
        (-6, -58),
        (-14, -54),
        (-22, -56),
        (-32, -60),
        (-42, -66),
        (-50, -72),
        (-54, -66),
        (-48, -56),
        (-38, -50),
        (-26, -46),
        (-14, -44),
        (-4, -48),
        (6, -56),
        (12, -68),
        (12, -80),
    ],
    # Greenland
    [
        (78, -74),
        (82, -60),
        (80, -42),
        (74, -30),
        (68, -36),
        (64, -48),
        (66, -60),
        (72, -70),
        (78, -74),
    ],
    # Australia
    [
        (-10, 112),
        (-16, 116),
        (-22, 124),
        (-28, 132),
        (-34, 142),
        (-38, 150),
        (-34, 154),
        (-28, 152),
        (-22, 146),
        (-16, 138),
        (-12, 128),
        (-10, 120),
        (-10, 112),
    ],
    # Asia
    [
        (62, 98),
        (58, 84),
        (52, 72),
        (46, 62),
        (40, 54),
        (32, 48),
        (24, 44),
        (16, 48),
        (8, 54),
        (2, 62),
        (0, 74),
        (4, 88),
        (10, 100),
        (16, 112),
        (22, 124),
        (28, 134),
        (36, 136),
        (44, 128),
        (50, 118),
        (56, 108),
        (62, 98),
    ],
    # Antarctica (simplified)
    [
        (-62, -180),
        (-66, -150),
        (-70, -120),
        (-72, -90),
        (-74, -60),
        (-75, -30),
        (-74, 0),
        (-75, 30),
        (-74, 60),
        (-72, 90),
        (-70, 120),
        (-66, 150),
        (-62, 180),
        (-62, -180),
    ],
]


def _latlon_to_xyz(lat_deg: float, lon_deg: float, radius: float) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.array([x, y, z])


class Viewport3D:
    def __init__(self, tag: str, width: int = 800, height: int = 600):
        self.tag = tag
        self.width = width
        self.height = height
        self.camera = Camera(distance=25000.0)
        self._trajectory: TrajectoryData | None = None
        self._transfer_trajectories: list[tuple[TrajectoryData, tuple[int, int, int, int]]] = []
        self._preview_points: np.ndarray | None = None
        self._reference_points: np.ndarray | None = None
        self._orbit_points: np.ndarray | None = None
        self._burn_points: np.ndarray | None = None
        self._burn_cursor_pos: np.ndarray | None = None
        self._burn_cursor_dir: np.ndarray | None = None
        self._body_radius = R_EARTH
        self._current_time = 0.0
        self._dragging = False
        self._last_mouse = (0.0, 0.0)
        self._needs_redraw = True

    def set_trajectory(self, traj: TrajectoryData, body_radius: float = R_EARTH):
        self._trajectory = traj
        self._body_radius = body_radius
        self._orbit_points, _ = traj.downsample(3000)
        self._current_time = traj.t[0]
        self._needs_redraw = True

    def add_transfer_trajectory(
        self, traj: TrajectoryData, color: tuple[int, int, int, int] = (255, 255, 0, 200)
    ):
        self._transfer_trajectories.append((traj, color))
        self._needs_redraw = True

    def clear_transfers(self):
        self._transfer_trajectories.clear()
        self._needs_redraw = True

    def set_preview_trajectory(self, traj: TrajectoryData | None):
        if traj is None:
            self._preview_points = None
        else:
            self._preview_points, _ = traj.downsample(2200)
        self._needs_redraw = True

    def clear_preview_trajectory(self):
        self._preview_points = None
        self._needs_redraw = True

    def set_reference_trajectory(self, traj: TrajectoryData | None):
        if traj is None:
            self._reference_points = None
        else:
            self._reference_points, _ = traj.downsample(2200)
        self._needs_redraw = True

    def clear_reference_trajectory(self):
        self._reference_points = None
        self._needs_redraw = True

    def set_burn_markers(self, points: np.ndarray | None):
        self._burn_points = points
        self._needs_redraw = True

    def clear_burn_markers(self):
        self._burn_points = None
        self._needs_redraw = True

    def set_burn_cursor(self, position: np.ndarray | None, direction_vec: np.ndarray | None = None):
        """Set a single point + optional direction arrow showing where the next burn will be placed."""
        self._burn_cursor_pos = position
        self._burn_cursor_dir = direction_vec
        self._needs_redraw = True

    def set_time(self, t: float):
        self._current_time = t
        self._needs_redraw = True

    def resize(self, width: int, height: int):
        width = max(100, width)
        height = max(100, height)
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            dpg.configure_item(self.tag, width=self.width, height=self.height)
            self._needs_redraw = True

    def create(self, parent):
        dpg.add_drawlist(width=self.width, height=self.height, tag=self.tag, parent=parent)
        with dpg.handler_registry(tag="viewport_handlers"):
            dpg.add_mouse_click_handler(button=0, callback=self._on_click)
            dpg.add_mouse_release_handler(button=0, callback=self._on_release)
            dpg.add_mouse_drag_handler(button=0, callback=self._on_drag)
            dpg.add_mouse_wheel_handler(callback=self._on_wheel)

    def _is_mouse_over_drawlist(self) -> bool:
        if not dpg.does_item_exist(self.tag):
            return False
        try:
            x, y = dpg.get_mouse_pos(local=False)
            x0, y0 = dpg.get_item_rect_min(self.tag)
            x1, y1 = dpg.get_item_rect_max(self.tag)
            return x0 <= x <= x1 and y0 <= y <= y1
        except Exception:
            return False

    def _on_click(self, sender, app_data):
        if self._is_mouse_over_drawlist():
            self._dragging = True
            self._last_mouse = dpg.get_mouse_pos(local=False)

    def _on_release(self, sender, app_data):
        self._dragging = False

    def _on_drag(self, sender, app_data):
        if not self._dragging:
            return
        if not self._is_mouse_over_drawlist():
            self._dragging = False
            return
        mx, my = dpg.get_mouse_pos(local=False)
        dx = mx - self._last_mouse[0]
        dy = my - self._last_mouse[1]
        self._last_mouse = (mx, my)
        self.camera.rotate(-dx * 0.005, -dy * 0.005)
        self._needs_redraw = True

    def _on_wheel(self, sender, app_data):
        if not self._is_mouse_over_drawlist():
            return
        self.camera.zoom(0.85 if app_data > 0 else 1.18)
        self._needs_redraw = True

    def set_view_xy(self):
        self.camera.azimuth = 0.0
        self.camera.elevation = np.pi / 2 - 0.001
        self._needs_redraw = True

    def set_view_xz(self):
        self.camera.azimuth = 0.0
        self.camera.elevation = 0.0
        self._needs_redraw = True

    def set_view_yz(self):
        self.camera.azimuth = np.pi / 2
        self.camera.elevation = 0.0
        self._needs_redraw = True

    def set_view_isometric(self):
        self.camera.azimuth = 0.6
        self.camera.elevation = 0.5
        self._needs_redraw = True

    def set_view_orbit_normal(self):
        if self._trajectory is None:
            self.set_view_isometric()
            return
        h = np.cross(self._trajectory.r[0], self._trajectory.v[0])
        h = h / np.linalg.norm(h)
        self.camera.azimuth = np.arctan2(h[1], h[0])
        self.camera.elevation = np.arcsin(np.clip(h[2], -1, 1))
        self._needs_redraw = True

    def set_view_along_velocity(self):
        if self._trajectory is None:
            self.set_view_isometric()
            return
        v_hat = self._trajectory.v[0] / np.linalg.norm(self._trajectory.v[0])
        self.camera.azimuth = np.arctan2(v_hat[1], v_hat[0]) + np.pi
        self.camera.elevation = np.arcsin(np.clip(v_hat[2], -1, 1))
        self._needs_redraw = True

    def set_view_nadir(self):
        if self._trajectory is None:
            self.set_view_isometric()
            return
        r_hat = self._trajectory.r[0] / np.linalg.norm(self._trajectory.r[0])
        self.camera.azimuth = np.arctan2(r_hat[1], r_hat[0]) + np.pi
        self.camera.elevation = np.arcsin(np.clip(r_hat[2], -1, 1))
        self._needs_redraw = True

    def render(self):
        if not self._needs_redraw:
            return
        self._needs_redraw = False

        dpg.delete_item(self.tag, children_only=True)
        view = self.camera.view_matrix()
        proj = perspective_matrix(45.0, self.width / max(1, self.height), 100.0, 200000.0)

        self._draw_earth(view, proj)
        self._draw_continents(view, proj)
        self._draw_reference(view, proj)
        self._draw_orbit(view, proj)
        self._draw_orbit_annotations(view, proj)
        self._draw_preview(view, proj)
        self._draw_transfers(view, proj)
        self._draw_burn_markers(view, proj)
        self._draw_burn_cursor(view, proj)
        self._draw_satellite(view, proj)
        self._draw_axes(view, proj)

    def _draw_earth(self, view, proj):
        center = project_points(np.zeros((1, 3)), view, proj, self.width, self.height)
        if center[0, 0] < -5000:
            return
        cx, cy = center[0]

        # Compute radius_px by projecting an actual point on the sphere's limb.
        # Use a point perpendicular to the view direction at body_radius distance.
        eye = self.camera.eye_position
        eye_hat = eye / np.linalg.norm(eye)
        # Find a vector perpendicular to eye direction
        up = np.array([0.0, 0.0, 1.0])
        perp = np.cross(eye_hat, up)
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(eye_hat, np.array([1.0, 0.0, 0.0]))
        perp = perp / np.linalg.norm(perp)
        limb_point = perp * self._body_radius
        limb_sp = project_points(limb_point.reshape(1, 3), view, proj, self.width, self.height)
        if limb_sp[0, 0] > -5000:
            radius_px = np.sqrt((limb_sp[0, 0] - cx) ** 2 + (limb_sp[0, 1] - cy) ** 2)
        else:
            scale = self._body_radius / self.camera.distance
            radius_px = max(10, scale * min(self.width, self.height) * 0.8)

        radius_px = max(10, radius_px)

        dpg.draw_circle(
            (cx, cy),
            radius_px,
            parent=self.tag,
            color=(28, 88, 170, 255),
            fill=(15, 68, 150, 255),
            thickness=2.0,
        )

        # Lat/lon line overlays (no additional filled spheres).
        for i in range(1, 8):
            lat = -90 + i * (180 / 8)
            frac = np.cos(np.radians(lat))
            r = radius_px * frac
            y_off = radius_px * np.sin(np.radians(lat))
            dpg.draw_circle(
                (cx, cy - y_off * 0.7),
                r * 0.95,
                parent=self.tag,
                color=(20, 60, 115, 72),
                thickness=0.55,
            )

        for i in range(8):
            lon = i * (360 / 8)
            pts = []
            for lat in range(-90, 91, 10):
                p3d = _latlon_to_xyz(lat, lon, self._body_radius)
                sp = project_points(p3d.reshape(1, 3), view, proj, self.width, self.height)
                if sp[0, 0] > -5000 and np.dot(p3d, self.camera.eye_position) > 0:
                    pts.append((float(sp[0, 0]), float(sp[0, 1])))
                else:
                    if len(pts) > 1:
                        dpg.draw_polyline(
                            pts, parent=self.tag, color=(20, 60, 115, 50), thickness=0.45
                        )
                    pts = []
            if len(pts) > 1:
                dpg.draw_polyline(pts, parent=self.tag, color=(20, 60, 115, 50), thickness=0.45)

    def _draw_continents(self, view, proj):
        scale = self._body_radius / self.camera.distance
        if max(10, scale * min(self.width, self.height) * 0.8) < 15:
            return
        eye = self.camera.eye_position

        for poly in CONTINENT_OUTLINES:
            pts_3d = np.array([_latlon_to_xyz(lat, lon, self._body_radius) for lat, lon in poly])
            screen = project_points(pts_3d, view, proj, self.width, self.height)
            valid = screen[:, 0] > -5000
            visible_pts = []

            for idx in range(len(pts_3d)):
                if not valid[idx]:
                    continue
                if np.dot(pts_3d[idx], eye) > 0:
                    visible_pts.append((float(screen[idx, 0]), float(screen[idx, 1])))
                else:
                    if len(visible_pts) > 2:
                        dpg.draw_polyline(
                            visible_pts, parent=self.tag, color=(96, 193, 120, 230), thickness=1.0
                        )
                    visible_pts = []

            if len(visible_pts) > 1:
                dpg.draw_polyline(
                    visible_pts, parent=self.tag, color=(96, 193, 120, 230), thickness=1.0
                )

    def _draw_reference(self, view, proj):
        if self._reference_points is None:
            return
        screen = project_points(self._reference_points, view, proj, self.width, self.height)
        valid = screen[:, 0] > -5000
        pts = []
        for i in range(len(screen)):
            if valid[i]:
                pts.append((float(screen[i, 0]), float(screen[i, 1])))
            else:
                if len(pts) > 1:
                    dpg.draw_polyline(
                        pts, parent=self.tag, color=(170, 170, 190, 150), thickness=1.1
                    )
                pts = []
        if len(pts) > 1:
            dpg.draw_polyline(pts, parent=self.tag, color=(170, 170, 190, 150), thickness=1.1)

    def _draw_orbit(self, view, proj):
        if self._orbit_points is None:
            return
        screen = project_points(self._orbit_points, view, proj, self.width, self.height)
        valid = screen[:, 0] > -5000
        pts = []
        for i in range(len(screen)):
            if valid[i]:
                pts.append((float(screen[i, 0]), float(screen[i, 1])))
            else:
                if len(pts) > 1:
                    dpg.draw_polyline(pts, parent=self.tag, color=(0, 191, 255, 220), thickness=1.5)
                pts = []
        if len(pts) > 1:
            dpg.draw_polyline(pts, parent=self.tag, color=(0, 191, 255, 220), thickness=1.5)

    def _draw_orbit_annotations(self, view, proj):
        if self._orbit_points is None or len(self._orbit_points) < 8:
            return

        # Use last third of trajectory (final orbit after burns) for annotations.
        n = len(self._orbit_points)
        start = max(0, n - n // 3)
        r = self._orbit_points[start:]
        if len(r) < 4:
            r = self._orbit_points

        r_mag = np.linalg.norm(r, axis=1)
        i_rp = int(np.argmin(r_mag))
        i_ra = int(np.argmax(r_mag))
        rp = r[i_rp]
        ra = r[i_ra]

        # Major axis
        major = np.vstack([rp, ra])
        major_sp = project_points(major, view, proj, self.width, self.height)
        if major_sp[0, 0] > -5000 and major_sp[1, 0] > -5000:
            dpg.draw_line(
                (float(major_sp[0, 0]), float(major_sp[0, 1])),
                (float(major_sp[1, 0]), float(major_sp[1, 1])),
                parent=self.tag,
                color=(180, 180, 220, 70),
                thickness=0.8,
            )

        # Minor axis estimate from orbital plane and major direction.
        t_idx = min(max(1, i_rp), len(r) - 2)
        tangent = r[t_idx + 1] - r[t_idx - 1]
        h = np.cross(rp, tangent)
        h_n = np.linalg.norm(h)
        if h_n > 1e-10:
            h_hat = h / h_n
            major_dir = ra - rp
            major_n = np.linalg.norm(major_dir)
            if major_n > 1e-10:
                major_dir = major_dir / major_n
                minor_dir = np.cross(h_hat, major_dir)
                minor_n = np.linalg.norm(minor_dir)
                if minor_n > 1e-10:
                    minor_dir = minor_dir / minor_n
                    center = 0.5 * (rp + ra)
                    proj_minor = (r - center) @ minor_dir
                    b = float(np.max(np.abs(proj_minor)))
                    p1 = center + minor_dir * b
                    p2 = center - minor_dir * b
                    minor_sp = project_points(
                        np.vstack([p1, p2]), view, proj, self.width, self.height
                    )
                    if minor_sp[0, 0] > -5000 and minor_sp[1, 0] > -5000:
                        dpg.draw_line(
                            (float(minor_sp[0, 0]), float(minor_sp[0, 1])),
                            (float(minor_sp[1, 0]), float(minor_sp[1, 1])),
                            parent=self.tag,
                            color=(180, 180, 220, 55),
                            thickness=0.7,
                        )

        # Ascending/descending node markers (z crossing).
        found_asc = False
        found_desc = False
        asc_screen = None
        desc_screen = None
        for idx in range(1, len(r)):
            z0 = r[idx - 1, 2]
            z1 = r[idx, 2]
            if z0 == z1 or z0 * z1 > 0:
                continue
            alpha = abs(z0) / (abs(z0) + abs(z1))
            node = r[idx - 1] * (1 - alpha) + r[idx] * alpha
            node_sp = project_points(node.reshape(1, 3), view, proj, self.width, self.height)
            if node_sp[0, 0] > -5000:
                is_ascending = z1 > z0
                if is_ascending and not found_asc:
                    col = (120, 220, 170, 170)
                    dpg.draw_circle(
                        (float(node_sp[0, 0]), float(node_sp[0, 1])),
                        2.5,
                        parent=self.tag,
                        color=col,
                        fill=col,
                        thickness=1,
                    )
                    dpg.draw_text(
                        (float(node_sp[0, 0]) + 5, float(node_sp[0, 1]) - 4),
                        "AN",
                        parent=self.tag,
                        color=(120, 220, 170, 160),
                        size=10,
                    )
                    asc_screen = (float(node_sp[0, 0]), float(node_sp[0, 1]))
                    found_asc = True
                elif not is_ascending and not found_desc:
                    col = (220, 170, 120, 170)
                    dpg.draw_circle(
                        (float(node_sp[0, 0]), float(node_sp[0, 1])),
                        2.5,
                        parent=self.tag,
                        color=col,
                        fill=col,
                        thickness=1,
                    )
                    dpg.draw_text(
                        (float(node_sp[0, 0]) + 5, float(node_sp[0, 1]) - 4),
                        "DN",
                        parent=self.tag,
                        color=(220, 170, 120, 160),
                        size=10,
                    )
                    desc_screen = (float(node_sp[0, 0]), float(node_sp[0, 1]))
                    found_desc = True
            if found_asc and found_desc:
                break

        # Line of nodes
        if asc_screen is not None and desc_screen is not None:
            dpg.draw_line(
                asc_screen,
                desc_screen,
                parent=self.tag,
                color=(170, 200, 180, 80),
                thickness=0.7,
            )

        # Periapsis / apoapsis subtle markers + altitude annotations.
        for label, p in (("rp", rp), ("ra", ra)):
            alt = np.linalg.norm(p) - self._body_radius
            sp = project_points(p.reshape(1, 3), view, proj, self.width, self.height)
            if sp[0, 0] < -5000:
                continue
            px, py = float(sp[0, 0]), float(sp[0, 1])
            dpg.draw_circle(
                (px, py),
                2.8,
                parent=self.tag,
                color=(220, 220, 220, 160),
                fill=(220, 220, 220, 120),
                thickness=1,
            )
            dpg.draw_text(
                (px + 6, py - 5),
                f"{label} {alt:.0f} km",
                parent=self.tag,
                color=(200, 200, 210, 145),
                size=11,
            )

    def _draw_transfers(self, view, proj):
        for traj, color in self._transfer_trajectories:
            r_down, _ = traj.downsample(1200)
            screen = project_points(r_down, view, proj, self.width, self.height)
            valid = screen[:, 0] > -5000
            pts = []
            for i in range(len(screen)):
                if valid[i]:
                    pts.append((float(screen[i, 0]), float(screen[i, 1])))
                else:
                    if len(pts) > 1:
                        dpg.draw_polyline(pts, parent=self.tag, color=color, thickness=2.0)
                    pts = []
            if len(pts) > 1:
                dpg.draw_polyline(pts, parent=self.tag, color=color, thickness=2.0)

    def _draw_preview(self, view, proj):
        if self._preview_points is None:
            return
        screen = project_points(self._preview_points, view, proj, self.width, self.height)
        valid = screen[:, 0] > -5000
        pts = []
        for i in range(len(screen)):
            if valid[i]:
                pts.append((float(screen[i, 0]), float(screen[i, 1])))
            else:
                if len(pts) > 1:
                    dpg.draw_polyline(
                        pts, parent=self.tag, color=(255, 180, 60, 185), thickness=1.6
                    )
                pts = []
        if len(pts) > 1:
            dpg.draw_polyline(pts, parent=self.tag, color=(255, 180, 60, 185), thickness=1.6)

    def _draw_burn_markers(self, view, proj):
        if self._burn_points is None or len(self._burn_points) == 0:
            return
        screen = project_points(self._burn_points, view, proj, self.width, self.height)
        for px, py in screen:
            if px < -5000:
                continue
            dpg.draw_circle(
                (float(px), float(py)),
                4.5,
                parent=self.tag,
                color=(255, 90, 50, 255),
                fill=(255, 90, 50, 210),
                thickness=1,
            )
            dpg.draw_circle(
                (float(px), float(py)), 8.5, parent=self.tag, color=(255, 140, 80, 96), thickness=1
            )

    def _draw_burn_cursor(self, view, proj):
        if self._burn_cursor_pos is None:
            return
        sp = project_points(
            self._burn_cursor_pos.reshape(1, 3), view, proj, self.width, self.height
        )
        if sp[0, 0] < -5000:
            return
        cx, cy = float(sp[0, 0]), float(sp[0, 1])
        # Pulsing diamond marker
        dpg.draw_circle(
            (cx, cy),
            5,
            parent=self.tag,
            color=(255, 255, 100, 255),
            fill=(255, 255, 50, 180),
            thickness=1.5,
        )
        dpg.draw_text((cx + 8, cy - 5), "BURN", parent=self.tag, color=(255, 255, 100, 200), size=9)

        # Direction arrow
        if self._burn_cursor_dir is not None:
            tip_3d = self._burn_cursor_pos + self._burn_cursor_dir
            tip_sp = project_points(tip_3d.reshape(1, 3), view, proj, self.width, self.height)
            if tip_sp[0, 0] > -5000:
                tx, ty = float(tip_sp[0, 0]), float(tip_sp[0, 1])
                dpg.draw_arrow(
                    (tx, ty),
                    (cx, cy),
                    parent=self.tag,
                    color=(255, 200, 50, 220),
                    thickness=2,
                    size=6,
                )

    def _draw_satellite(self, view, proj):
        if self._trajectory is None:
            return
        r, _ = self._trajectory.interpolate(self._current_time)
        screen = project_points(r.reshape(1, 3), view, proj, self.width, self.height)
        if screen[0, 0] < -5000:
            return
        cx, cy = screen[0]
        dpg.draw_circle(
            (cx, cy),
            5,
            parent=self.tag,
            color=(255, 255, 0, 255),
            fill=(255, 255, 0, 220),
            thickness=1,
        )
        dpg.draw_circle((cx, cy), 9, parent=self.tag, color=(255, 255, 0, 80), thickness=1)

    def _draw_axes(self, view, proj):
        length = self._body_radius * 1.3
        origin_screen = project_points(np.zeros((3, 3)), view, proj, self.width, self.height)
        tip_screen = project_points(
            np.array([[length, 0, 0], [0, length, 0], [0, 0, length]]),
            view,
            proj,
            self.width,
            self.height,
        )
        colors = [(200, 60, 60, 120), (60, 200, 60, 120), (60, 60, 200, 120)]
        labels = ["X", "Y", "Z"]
        for i in range(3):
            if origin_screen[i, 0] > -5000 and tip_screen[i, 0] > -5000:
                dpg.draw_line(
                    (origin_screen[i, 0], origin_screen[i, 1]),
                    (tip_screen[i, 0], tip_screen[i, 1]),
                    parent=self.tag,
                    color=colors[i],
                    thickness=1,
                )
                dpg.draw_text(
                    (tip_screen[i, 0] + 4, tip_screen[i, 1] - 4),
                    labels[i],
                    parent=self.tag,
                    color=colors[i],
                    size=11,
                )
