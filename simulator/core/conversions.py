import numpy as np
from simulator.core.constants import MU_EARTH, R_EARTH
from simulator.core.state import StateVector, OrbitalElements


def coe_to_state(coe: OrbitalElements, mu: float = MU_EARTH) -> StateVector:
    i, O, w, th = [np.radians(x) for x in (coe.i, coe.raan, coe.omega, coe.theta)]
    a, e = coe.a, coe.e
    p = a * (1.0 - e**2)
    r_m = p / (1.0 + e * np.cos(th))
    r_pf = r_m * np.array([np.cos(th), np.sin(th), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(th), e + np.cos(th), 0.0])
    cO, sO = np.cos(O), np.sin(O)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(w), np.sin(w)
    Q = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci]
    ])
    return StateVector(r=Q @ r_pf, v=Q @ v_pf)


def state_to_coe(sv: StateVector, mu: float = MU_EARTH) -> OrbitalElements:
    r_vec, v_vec = sv.r, sv.v
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_v = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_v)
    e_v = (1.0 / mu) * ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    e = np.linalg.norm(e_v)
    i_d = np.degrees(np.arccos(np.clip(h_v[2] / h, -1.0, 1.0)))
    N_v = np.cross(np.array([0.0, 0.0, 1.0]), h_v)
    N = np.linalg.norm(N_v)
    Om = 0.0
    if N > 1e-10:
        Om = np.degrees(np.arccos(np.clip(N_v[0] / N, -1.0, 1.0)))
        if N_v[1] < 0:
            Om = 360.0 - Om
    om = 0.0
    if N > 1e-10 and e > 1e-10:
        om = np.degrees(np.arccos(np.clip(np.dot(N_v, e_v) / (N * e), -1.0, 1.0)))
        if e_v[2] < 0:
            om = 360.0 - om
    th = 0.0
    if e > 1e-10:
        th = np.degrees(np.arccos(np.clip(np.dot(e_v, r_vec) / (e * r), -1.0, 1.0)))
        if np.dot(r_vec, v_vec) < 0:
            th = 360.0 - th
    eps = 0.5 * v**2 - mu / r
    a = -mu / (2.0 * eps)
    return OrbitalElements(a=a, e=e, i=i_d, raan=Om, omega=om, theta=th)


def circular_orbit_state(alt_km: float, inc_deg: float = 0.0,
                         mu: float = MU_EARTH) -> StateVector:
    r = R_EARTH + alt_km
    v_c = np.sqrt(mu / r)
    i = np.radians(inc_deg)
    return StateVector(
        r=np.array([r, 0.0, 0.0]),
        v=np.array([0.0, v_c * np.cos(i), v_c * np.sin(i)])
    )


def orbital_period(a: float, mu: float = MU_EARTH) -> float:
    return 2.0 * np.pi * np.sqrt(a**3 / mu)


def vis_viva(r: float, a: float, mu: float = MU_EARTH) -> float:
    return np.sqrt(mu * (2.0 / r - 1.0 / a))


def specific_energy(r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_EARTH) -> float:
    return 0.5 * np.linalg.norm(v_vec)**2 - mu / np.linalg.norm(r_vec)


def spec_ang_mom(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    return np.cross(r_vec, v_vec)
