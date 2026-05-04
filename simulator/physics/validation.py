import numpy as np
from simulator.core.constants import MU_EARTH
from simulator.core.conversions import specific_energy, spec_ang_mom


def check_conservation(t: np.ndarray, r: np.ndarray, v: np.ndarray,
                       mu: float = MU_EARTH) -> dict:
    n = len(t)
    stride = max(1, n // 500)
    idx = np.arange(0, n, stride)

    eps_arr = np.array([specific_energy(r[i], v[i], mu) for i in idx])
    h_arr = np.array([np.linalg.norm(spec_ang_mom(r[i], v[i])) for i in idx])

    eps_drift = abs((eps_arr[-1] - eps_arr[0]) / eps_arr[0]) if eps_arr[0] != 0 else 0.0
    h_drift = abs((h_arr[-1] - h_arr[0]) / h_arr[0]) if h_arr[0] != 0 else 0.0

    return {
        "energy_drift": eps_drift,
        "angular_momentum_drift": h_drift,
        "energy_ok": eps_drift < 1e-6,
        "momentum_ok": h_drift < 1e-6,
        "t_sample": t[idx],
        "eps": eps_arr,
        "h_mag": h_arr,
    }
