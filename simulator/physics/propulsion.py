import numpy as np
from simulator.core.constants import R_UNIV, G0_SI


ENGINES = {
    "RP-1/LOX (Merlin-like)": {
        "T0": 3800, "M_mol": 21.5, "gamma": 1.25,
        "p0": 9.7e6, "eps": 16.0, "Isp_vac": 311.0, "F_kN": 845.0,
    },
    "LH2/LOX (RL-10-like)": {
        "T0": 3516, "M_mol": 11.8, "gamma": 1.24,
        "p0": 3.3e6, "eps": 61.0, "Isp_vac": 453.0, "F_kN": 110.0,
    },
    "LCH4/LOX (Raptor-like)": {
        "T0": 3600, "M_mol": 18.6, "gamma": 1.22,
        "p0": 30.0e6, "eps": 40.0, "Isp_vac": 380.0, "F_kN": 2200.0,
    },
    "Ion (Hall thruster)": {
        "T0": 0, "M_mol": 131.3, "gamma": 1.0,
        "p0": 0, "eps": 0, "Isp_vac": 1800.0, "F_kN": 0.001,
    },
    "Solid (SRB)": {
        "T0": 3500, "M_mol": 28.0, "gamma": 1.26,
        "p0": 7.0e6, "eps": 10.0, "Isp_vac": 275.0, "F_kN": 12000.0,
    },
}


def thrust_prograde(t: float, state7: np.ndarray) -> np.ndarray:
    v = state7[3:6]
    return v / np.linalg.norm(v)


def thrust_retrograde(t: float, state7: np.ndarray) -> np.ndarray:
    v = state7[3:6]
    return -v / np.linalg.norm(v)


def thrust_normal(t: float, state7: np.ndarray) -> np.ndarray:
    r, v = state7[:3], state7[3:6]
    h = np.cross(r, v)
    return h / np.linalg.norm(h)


def engine_mdot(F_kN: float, Isp_s: float) -> float:
    F_N = F_kN * 1000.0
    return F_N / (Isp_s * G0_SI)
