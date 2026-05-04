from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from simulator.core.state import OrbitalElements
from simulator.core.conversions import state_to_coe, StateVector


class TrajectoryData:
    def __init__(self, t: np.ndarray, r: np.ndarray, v: np.ndarray,
                 m: np.ndarray | None = None):
        self.t = t
        self.r = r
        self.v = v
        self.m = m
        self._interp_r = CubicSpline(t, r)
        self._interp_v = CubicSpline(t, v)

    @property
    def duration(self) -> float:
        return self.t[-1] - self.t[0]

    @property
    def n_points(self) -> int:
        return len(self.t)

    def interpolate(self, t_query: float) -> tuple[np.ndarray, np.ndarray]:
        t_clamped = np.clip(t_query, self.t[0], self.t[-1])
        return self._interp_r(t_clamped), self._interp_v(t_clamped)

    def coe_at(self, t_query: float) -> OrbitalElements:
        r, v = self.interpolate(t_query)
        return state_to_coe(StateVector(r=r, v=v))

    def altitude_at(self, t_query: float, body_radius: float) -> float:
        r, _ = self.interpolate(t_query)
        return np.linalg.norm(r) - body_radius

    def downsample(self, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        if self.n_points <= n_points:
            return self.r, self.v
        idx = np.linspace(0, self.n_points - 1, n_points, dtype=int)
        return self.r[idx], self.v[idx]

    def altitude_range(self, body_radius: float) -> tuple[float, float]:
        alt = np.linalg.norm(self.r, axis=1) - body_radius
        return float(alt.min()), float(alt.max())
