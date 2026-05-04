import numpy as np
from simulator.core.constants import MU_EARTH, R_EARTH, J2_EARTH
from simulator.physics.atmosphere import atmo_density


def j2_accel(r_vec: np.ndarray, mu: float = MU_EARTH,
             j2: float = J2_EARTH, Re: float = R_EARTH) -> np.ndarray:
    x, y, z = r_vec
    r = np.linalg.norm(r_vec)
    fac = (3.0 * mu * j2 * Re**2) / (2.0 * r**5)
    zr2 = (z / r)**2
    return np.array([
        fac * x * (5 * zr2 - 1),
        fac * y * (5 * zr2 - 1),
        fac * z * (5 * zr2 - 3),
    ])


def drag_accel(r_vec: np.ndarray, v_vec: np.ndarray,
               Cd: float = 2.2, B: float = 5.6e-9) -> np.ndarray:
    alt = np.linalg.norm(r_vec) - R_EARTH
    if alt >= 1000.0 or alt <= 0.0:
        return np.zeros(3)
    rho = atmo_density(alt)
    v_mag = np.linalg.norm(v_vec)
    if v_mag < 1e-12:
        return np.zeros(3)
    return -0.5 * Cd * B * rho * v_mag * v_vec


def third_body_accel(r_vec: np.ndarray, r_body: np.ndarray,
                     mu_body: float) -> np.ndarray:
    r_rel = r_body - r_vec
    d = np.linalg.norm(r_rel)
    d_body = np.linalg.norm(r_body)
    return mu_body * (r_rel / d**3 - r_body / d_body**3)


def j2_raan_rate(a: float, e: float, i_deg: float,
                 mu: float = MU_EARTH, j2: float = J2_EARTH,
                 Re: float = R_EARTH) -> float:
    i = np.radians(i_deg)
    n = np.sqrt(mu / a**3)
    p = a * (1.0 - e**2)
    rate_rad_s = -(3.0 / 2.0) * n * j2 * (Re / p)**2 * np.cos(i)
    return np.degrees(rate_rad_s) * 86400.0


def j2_argp_rate(a: float, e: float, i_deg: float,
                 mu: float = MU_EARTH, j2: float = J2_EARTH,
                 Re: float = R_EARTH) -> float:
    i = np.radians(i_deg)
    n = np.sqrt(mu / a**3)
    p = a * (1.0 - e**2)
    rate_rad_s = (3.0 / 4.0) * n * j2 * (Re / p)**2 * (5.0 * np.cos(i)**2 - 1.0)
    return np.degrees(rate_rad_s) * 86400.0
