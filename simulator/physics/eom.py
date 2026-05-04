from __future__ import annotations
from typing import Callable
import numpy as np
from simulator.core.constants import MU_EARTH
from simulator.core.bodies import CelestialBody
from simulator.physics.perturbations import j2_accel, drag_accel, third_body_accel


class EOMBuilder:
    def __init__(self, body: CelestialBody):
        self._mu = body.mu
        self._perturbations: list[Callable] = []

    def add_perturbation(self, func: Callable) -> EOMBuilder:
        self._perturbations.append(func)
        return self

    def add_j2(self, j2: float, Re: float) -> EOMBuilder:
        def _j2(t, r, v):
            return j2_accel(r, self._mu, j2, Re)
        self._perturbations.append(_j2)
        return self

    def add_drag(self, Cd: float = 2.2, B: float = 5.6e-9) -> EOMBuilder:
        def _drag(t, r, v):
            return drag_accel(r, v, Cd, B)
        self._perturbations.append(_drag)
        return self

    def add_third_body_ephemeris(
        self, body_name: str, central_body_name: str, epoch_jd: float
    ) -> EOMBuilder:
        """Add third-body perturbation using real ephemeris positions."""
        from simulator.core.ephemeris import planet_ecliptic_position, _PLANET_MU

        mu_body = _PLANET_MU[body_name]

        def _third_body(t, r, v):
            jd_now = epoch_jd + t / 86400.0
            central_helio = planet_ecliptic_position(central_body_name, jd_now)
            body_helio = planet_ecliptic_position(body_name, jd_now)
            r_body = body_helio - central_helio
            return third_body_accel(r, r_body, mu_body)

        self._perturbations.append(_third_body)
        return self

    def build(self) -> Callable[[float, np.ndarray], np.ndarray]:
        mu = self._mu
        perturbations = self._perturbations.copy()

        def eom(t: float, y: np.ndarray) -> np.ndarray:
            r = y[:3]
            v = y[3:6]
            r_mag = np.linalg.norm(r)
            a_total = -(mu / r_mag**3) * r
            for p_func in perturbations:
                a_total = a_total + p_func(t, r, v)
            return np.concatenate([v, a_total])

        return eom

    def build_7dof(self, F_kN: float, mdot_kgs: float,
                   thrust_dir_fn: Callable) -> Callable[[float, np.ndarray], np.ndarray]:
        mu = self._mu
        perturbations = self._perturbations.copy()

        def eom(t: float, y: np.ndarray) -> np.ndarray:
            r, v, m = y[:3], y[3:6], y[6]
            r_mag = np.linalg.norm(r)
            a_grav = -(mu / r_mag**3) * r
            for p_func in perturbations:
                a_grav = a_grav + p_func(t, r, v)
            a_thrust = (F_kN / m) * thrust_dir_fn(t, y)
            return np.concatenate([v, a_grav + a_thrust, [-mdot_kgs]])

        return eom
