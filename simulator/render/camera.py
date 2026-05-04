import numpy as np


class Camera:
    def __init__(self, distance: float = 20000.0, azimuth: float = 0.4,
                 elevation: float = 0.5):
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.target = np.zeros(3)
        self._dirty = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self):
        self._dirty = False

    def rotate(self, d_azimuth: float, d_elevation: float):
        self.azimuth += d_azimuth
        self.elevation = np.clip(self.elevation + d_elevation, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        self._dirty = True

    def zoom(self, factor: float):
        self.distance = max(1000.0, self.distance * factor)
        self._dirty = True

    def pan(self, dx: float, dy: float):
        """Pan the camera target in screen-aligned directions."""
        eye = self.eye_position
        forward = self.target - eye
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-10:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
            r_norm = np.linalg.norm(right)
        right = right / r_norm
        up = np.cross(right, forward)
        scale = self.distance * 0.001
        self.target += right * (-dx * scale) + up * (dy * scale)
        self._dirty = True

    def reset(self):
        """Reset camera to default position."""
        self.distance = 20000.0
        self.azimuth = 0.4
        self.elevation = 0.5
        self.target = np.zeros(3)
        self._dirty = True

    @property
    def eye_position(self) -> np.ndarray:
        x = self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
        z = self.distance * np.sin(self.elevation)
        return self.target + np.array([x, y, z])

    def view_matrix(self) -> np.ndarray:
        eye = self.eye_position
        forward = self.target - eye
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-10:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
            r_norm = np.linalg.norm(right)
        right = right / r_norm
        up = np.cross(right, forward)

        R = np.eye(4)
        R[0, :3] = right
        R[1, :3] = up
        R[2, :3] = -forward

        T = np.eye(4)
        T[:3, 3] = -eye

        return R @ T
