"""
Multi-body scenario and simulation engine for propagating multiple spacecraft.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
from scipy.integrate import solve_ivp

from simulator.core.spacecraft import Spacecraft, MultiBodyTrajectory
from simulator.core.bodies import CelestialBody
from simulator.core.ephemeris import datetime_to_jd, _PLANET_MU
from simulator.core.conversions import coe_to_state, orbital_period
from simulator.core.constants import R_EARTH
from simulator.physics.nbody import build_multi_spacecraft_eom, build_nbody_eom
from simulator.physics.perturbations import j2_accel, drag_accel
from simulator.sim.scenario import PerturbationConfig, ManeuverEvent
from simulator.sim.trajectory import TrajectoryData


@dataclass
class MultiBodyScenario:
    """Scenario for simulating multiple spacecraft."""

    spacecraft: list[Spacecraft]
    central_body: str = "Earth"
    perturbing_bodies: list[str] = field(default_factory=lambda: ["Moon", "Sun"])
    epoch: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    duration_seconds: float = 86400.0  # 1 day default
    perturbations: PerturbationConfig = field(default_factory=PerturbationConfig)
    maneuvers: dict[str, list[ManeuverEvent]] = field(default_factory=dict)
    integrator_method: str = "DOP853"
    rtol: float = 1e-10
    atol: float = 1e-12


class MultiBodyEngine:
    """Engine for multi-spacecraft simulation."""

    def __init__(self):
        self._result: MultiBodyTrajectory | None = None
        self._error: str | None = None

    @property
    def result(self) -> MultiBodyTrajectory | None:
        return self._result

    @property
    def error(self) -> str | None:
        return self._error

    def compute(self, scenario: MultiBodyScenario) -> MultiBodyTrajectory | None:
        """Run the simulation synchronously and return result."""
        self._result = None
        self._error = None
        try:
            self._result = self._run(scenario)
            return self._result
        except Exception as e:
            self._error = str(e)
            return None

    def _run(self, scenario: MultiBodyScenario) -> MultiBodyTrajectory:
        n_sc = len(scenario.spacecraft)
        epoch_jd = datetime_to_jd(scenario.epoch)
        mu_central = _PLANET_MU[scenario.central_body]

        # Build initial state vector (all spacecraft concatenated)
        y0_parts = []
        for sc in scenario.spacecraft:
            y0_6 = sc.get_y0(mu_central)[:6]  # Only position/velocity for now
            y0_parts.append(y0_6)
        y0 = np.concatenate(y0_parts)

        # Build perturbation functions per spacecraft
        perts_per_sc = []
        for sc in scenario.spacecraft:
            sc_perts = []
            cfg = scenario.perturbations
            if cfg.j2_enabled and scenario.central_body == "Earth":

                def _j2(t, r, v, mu=mu_central):
                    return j2_accel(r, mu)

                sc_perts.append(_j2)
            if cfg.drag_enabled and scenario.central_body == "Earth":
                cd = cfg.drag_cd
                B = cfg.drag_ballistic_coeff

                def _drag(t, r, v, cd=cd, B=B):
                    return drag_accel(r, v, cd, B)

                sc_perts.append(_drag)
            perts_per_sc.append(sc_perts)

        # Build EOM
        eom = build_multi_spacecraft_eom(
            n_spacecraft=n_sc,
            central_body_name=scenario.central_body,
            perturbing_bodies=scenario.perturbing_bodies,
            epoch_jd=epoch_jd,
            perturbations_per_sc=perts_per_sc,
        )

        t_span = (0.0, scenario.duration_seconds)
        n_eval = max(2000, int(scenario.duration_seconds / 30.0))
        t_eval = np.linspace(0, scenario.duration_seconds, n_eval)

        # Reentry event for Earth-orbiting
        events = []
        if scenario.central_body == "Earth":
            for i in range(n_sc):

                def make_event(idx):
                    def reentry(t, y, idx=idx):
                        r = y[idx * 6 : idx * 6 + 3]
                        return np.linalg.norm(r) - (R_EARTH + 80.0)

                    reentry.terminal = False  # Don't stop for one craft
                    reentry.direction = -1
                    return reentry

                events.append(make_event(i))

        sol = solve_ivp(
            eom,
            t_span,
            y0,
            method=scenario.integrator_method,
            rtol=scenario.rtol,
            atol=scenario.atol,
            t_eval=t_eval,
            events=events if events else None,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        # Extract per-spacecraft trajectories
        positions = {}
        velocities = {}
        for i, sc in enumerate(scenario.spacecraft):
            idx = i * 6
            positions[sc.name] = sol.y[idx : idx + 3].T
            velocities[sc.name] = sol.y[idx + 3 : idx + 6].T

        return MultiBodyTrajectory(
            spacecraft_names=[sc.name for sc in scenario.spacecraft],
            spacecraft_colors=[sc.color for sc in scenario.spacecraft],
            t=sol.t,
            positions=positions,
            velocities=velocities,
        )
