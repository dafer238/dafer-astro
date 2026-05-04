from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp
from simulator.core.constants import R_EARTH


def propagate(eom: Callable, y0: np.ndarray, t_span: tuple[float, float],
              method: str = "DOP853", rtol: float = 1e-10, atol: float = 1e-12,
              max_step: float = np.inf, alt_min_km: float = 80.0,
              dense_output: bool = True) -> dict:
    def reentry_event(t, y):
        return np.linalg.norm(y[:3]) - (R_EARTH + alt_min_km)
    reentry_event.terminal = True
    reentry_event.direction = -1

    t_eval = np.linspace(t_span[0], t_span[1], max(2000, int((t_span[1] - t_span[0]) / 30.0)))

    sol = solve_ivp(
        eom, t_span, y0, method=method,
        rtol=rtol, atol=atol, max_step=max_step,
        events=[reentry_event], t_eval=t_eval,
        dense_output=dense_output,
    )

    reentered = len(sol.t_events[0]) > 0 if sol.t_events else False

    return {
        "t": sol.t,
        "r": sol.y[:3].T,
        "v": sol.y[3:6].T,
        "m": sol.y[6] if sol.y.shape[0] > 6 else None,
        "reentered": reentered,
        "sol": sol if dense_output else None,
        "n_steps": sol.nfev,
        "success": sol.success,
        "message": sol.message,
    }


def propagate_7dof(eom: Callable, y0: np.ndarray, t_span: tuple[float, float],
                   m_dry: float, method: str = "RK45",
                   rtol: float = 1e-8, atol: float = 1e-10) -> dict:
    def burnout_event(t, y):
        return y[6] - m_dry
    burnout_event.terminal = True
    burnout_event.direction = -1

    t_eval = np.linspace(t_span[0], t_span[1], max(500, int((t_span[1] - t_span[0]) / 1.0)))

    sol = solve_ivp(
        eom, t_span, y0, method=method,
        rtol=rtol, atol=atol,
        events=[burnout_event], t_eval=t_eval,
    )

    return {
        "t": sol.t,
        "r": sol.y[:3].T,
        "v": sol.y[3:6].T,
        "m": sol.y[6],
        "burnout": len(sol.t_events[0]) > 0 if sol.t_events else False,
        "success": sol.success,
    }
