"""
N-body equations of motion for multiple spacecraft orbiting in a system
with gravitational influence from all solar system bodies.
"""

from __future__ import annotations
from typing import Callable
import numpy as np
from simulator.core.ephemeris import (
    planet_ecliptic_position,
    moon_geocentric_position,
    datetime_to_jd,
    jd_to_centuries,
    _PLANET_MU,
)


def build_nbody_eom(
    central_body_name: str,
    perturbing_bodies: list[str],
    epoch_jd: float,
    perturbations: list[Callable] | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Build equations of motion for a spacecraft orbiting `central_body_name`,
    with gravitational perturbations from `perturbing_bodies`.

    The state vector y = [x, y, z, vx, vy, vz] is in the central-body-centered
    ecliptic frame.

    Args:
        central_body_name: Name of the central body (e.g., "Earth")
        perturbing_bodies: List of perturbing body names (e.g., ["Moon", "Sun", "Jupiter"])
        epoch_jd: Julian Date of simulation start (t=0 corresponds to this JD)
        perturbations: Additional perturbation functions (J2, drag, etc.)
    """
    mu_central = _PLANET_MU[central_body_name]
    perturber_mus = [(name, _PLANET_MU[name]) for name in perturbing_bodies]
    extra_perts = perturbations or []

    def eom(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:6]
        r_mag = np.linalg.norm(r)

        # Central body gravity
        a_total = -(mu_central / r_mag**3) * r

        # Third-body perturbations
        jd_now = epoch_jd + t / 86400.0  # t is in seconds

        # Get central body's heliocentric position
        central_helio = planet_ecliptic_position(central_body_name, jd_now)

        for body_name, mu_body in perturber_mus:
            # Get perturbing body heliocentric position
            body_helio = planet_ecliptic_position(body_name, jd_now)
            # Position of perturbing body relative to central body
            r_body = body_helio - central_helio

            # Third-body acceleration (indirect + direct)
            r_rel = r_body - r
            d = np.linalg.norm(r_rel)
            d_body = np.linalg.norm(r_body)
            if d > 1e-3 and d_body > 1e-3:
                a_total += mu_body * (r_rel / d**3 - r_body / d_body**3)

        # Additional perturbations (J2, drag, etc.)
        for p_func in extra_perts:
            a_total += p_func(t, r, v)

        return np.concatenate([v, a_total])

    return eom


def build_multi_spacecraft_eom(
    n_spacecraft: int,
    central_body_name: str,
    perturbing_bodies: list[str],
    epoch_jd: float,
    perturbations_per_sc: list[list[Callable]] | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Build coupled EOM for multiple spacecraft.
    State vector: [sc1_r, sc1_v, sc2_r, sc2_v, ...] each 6-element.

    Spacecraft do not gravitationally influence each other (negligible mass),
    but all experience the same gravitational field from celestial bodies.
    """
    mu_central = _PLANET_MU[central_body_name]
    perturber_mus = [(name, _PLANET_MU[name]) for name in perturbing_bodies]
    perts = perturbations_per_sc or [[] for _ in range(n_spacecraft)]

    def eom(t: float, y: np.ndarray) -> np.ndarray:
        dy = np.zeros_like(y)

        # Compute third-body positions once per timestep
        jd_now = epoch_jd + t / 86400.0
        central_helio = planet_ecliptic_position(central_body_name, jd_now)
        body_positions = []
        for body_name, mu_body in perturber_mus:
            body_helio = planet_ecliptic_position(body_name, jd_now)
            r_body = body_helio - central_helio
            body_positions.append((r_body, mu_body))

        for i in range(n_spacecraft):
            idx = i * 6
            r = y[idx : idx + 3]
            v = y[idx + 3 : idx + 6]
            r_mag = np.linalg.norm(r)

            a_total = -(mu_central / r_mag**3) * r

            for r_body, mu_body in body_positions:
                r_rel = r_body - r
                d = np.linalg.norm(r_rel)
                d_body = np.linalg.norm(r_body)
                if d > 1e-3 and d_body > 1e-3:
                    a_total += mu_body * (r_rel / d**3 - r_body / d_body**3)

            for p_func in perts[i]:
                a_total += p_func(t, r, v)

            dy[idx : idx + 3] = v
            dy[idx + 3 : idx + 6] = a_total

        return dy

    return eom
