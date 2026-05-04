import numpy as np
from simulator.core.constants import MU_EARTH, G0
from simulator.core.conversions import vis_viva, orbital_period


def hohmann_dv(r1: float, r2: float, mu: float = MU_EARTH) -> dict:
    a_t = (r1 + r2) / 2.0
    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)
    v_tp = vis_viva(r1, a_t, mu)
    v_ta = vis_viva(r2, a_t, mu)
    dv1 = abs(v_tp - v_c1)
    dv2 = abs(v_c2 - v_ta)
    T_t = orbital_period(a_t, mu)
    return {
        "dv1": dv1, "dv2": dv2, "dv_total": dv1 + dv2,
        "a_t": a_t, "dt_transfer": T_t / 2.0,
        "r1": r1, "r2": r2,
    }


def bielliptic_dv(r1: float, r2: float, r_b: float, mu: float = MU_EARTH) -> dict:
    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)
    a1 = (r1 + r_b) / 2.0
    a2 = (r2 + r_b) / 2.0
    vA = vis_viva(r1, a1, mu)
    vB1 = vis_viva(r_b, a1, mu)
    vB2 = vis_viva(r_b, a2, mu)
    vC = vis_viva(r2, a2, mu)
    dv1 = abs(vA - v_c1)
    dv2 = abs(vB2 - vB1)
    dv3 = abs(v_c2 - vC)
    T = orbital_period(a1, mu) / 2 + orbital_period(a2, mu) / 2
    return {"dv1": dv1, "dv2": dv2, "dv3": dv3,
            "dv_total": dv1 + dv2 + dv3, "T_total": T, "r_b": r_b}


def plane_change_dv(v_orb: float, delta_i_deg: float) -> float:
    di = np.radians(delta_i_deg)
    return 2.0 * v_orb * np.sin(di / 2.0)


def combined_dv(v1: float, v2: float, delta_i_deg: float) -> float:
    di = np.radians(delta_i_deg)
    return np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(di))


def apply_burn(state: np.ndarray, dv_vec: np.ndarray) -> np.ndarray:
    new_state = state.copy()
    new_state[3:6] += dv_vec
    return new_state


def tangential_burn(state: np.ndarray, dv_magnitude: float) -> np.ndarray:
    v_vec = state[3:6]
    v_hat = v_vec / np.linalg.norm(v_vec)
    return apply_burn(state, dv_magnitude * v_hat)


def rocket_equation(dv_kms: float, isp_s: float) -> dict:
    ve = isp_s * G0
    mass_ratio = np.exp(dv_kms / ve)
    prop_fraction = 1.0 - np.exp(-dv_kms / ve)
    return {"ve": ve, "mass_ratio": mass_ratio, "prop_fraction": prop_fraction}
