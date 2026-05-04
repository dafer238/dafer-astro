"""
Multi-body spacecraft definition and n-body propagation support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from simulator.core.state import OrbitalElements, StateVector


@dataclass
class Spacecraft:
    """A named spacecraft with color for visualization."""

    name: str
    color: tuple[int, int, int]
    initial_state: StateVector | None = None
    initial_coe: OrbitalElements | None = None
    mass: float | None = None

    def get_y0(self, mu: float) -> np.ndarray:
        """Get initial state vector as array."""
        if self.initial_state is not None:
            return self.initial_state.to_array()
        if self.initial_coe is not None:
            from simulator.core.conversions import coe_to_state

            sv = coe_to_state(self.initial_coe, mu)
            if self.mass is not None:
                return np.concatenate([sv.r, sv.v, [self.mass]])
            return np.concatenate([sv.r, sv.v])
        raise ValueError(f"Spacecraft '{self.name}' has no initial state defined")


@dataclass
class MultiBodyTrajectory:
    """Holds trajectories for multiple spacecraft."""

    spacecraft_names: list[str]
    spacecraft_colors: list[tuple[int, int, int]]
    t: np.ndarray
    positions: dict[str, np.ndarray]  # name -> (N, 3) position array
    velocities: dict[str, np.ndarray]  # name -> (N, 3) velocity array

    def get_trajectory(self, name: str):
        """Get TrajectoryData-like object for a named spacecraft."""
        from simulator.sim.trajectory import TrajectoryData

        return TrajectoryData(
            t=self.t,
            r=self.positions[name],
            v=self.velocities[name],
        )
