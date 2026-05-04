from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

from simulator.core.conversions import coe_to_state, orbital_period, state_to_coe, StateVector
from simulator.physics.eom import EOMBuilder
from simulator.physics.propagator import propagate
from simulator.physics.validation import check_conservation
from simulator.sim.scenario import Scenario, ManeuverEvent
from simulator.sim.trajectory import TrajectoryData


class SimulationEngine:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._result: TrajectoryData | None = None
        self._validation: dict | None = None
        self._error: str | None = None
        self._burn_events: list[dict] = []

    @property
    def is_running(self) -> bool:
        return self._future is not None and not self._future.done()

    @property
    def is_complete(self) -> bool:
        return self._future is not None and self._future.done()

    @property
    def result(self) -> TrajectoryData | None:
        if self._result is not None:
            return self._result
        if self.is_complete and self._future is not None:
            try:
                self._result = self._future.result()
            except Exception as e:
                self._error = str(e)
        return self._result

    @property
    def validation(self) -> dict | None:
        return self._validation

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def burn_events(self) -> list[dict]:
        return self._burn_events

    def compute(self, scenario: Scenario) -> None:
        self._result = None
        self._validation = None
        self._error = None
        self._burn_events = []
        self._future = self._executor.submit(self._run, scenario)

    def _run(self, scenario: Scenario) -> TrajectoryData:
        sv = coe_to_state(scenario.initial_coe, scenario.central_body.mu)
        y0 = np.concatenate([sv.r, sv.v])

        T = orbital_period(scenario.initial_coe.a, scenario.central_body.mu)
        t_span = (0.0, T * scenario.n_orbits)

        builder = EOMBuilder(scenario.central_body)
        cfg = scenario.perturbations
        if cfg.j2_enabled:
            builder.add_j2(scenario.central_body.j2, scenario.central_body.radius)
        if cfg.drag_enabled:
            builder.add_drag(cfg.drag_cd, cfg.drag_ballistic_coeff)

        eom = builder.build()

        traj = self._propagate_with_maneuvers(
            eom=eom,
            y0=y0,
            t_span=t_span,
            maneuvers=scenario.maneuvers,
            method=scenario.integrator_method,
            rtol=scenario.rtol,
            atol=scenario.atol,
        )

        if not cfg.drag_enabled and len(scenario.maneuvers) == 0:
            self._validation = check_conservation(traj.t, traj.r, traj.v, scenario.central_body.mu)
        else:
            self._validation = None

        return traj

    def _propagate_with_maneuvers(
        self,
        eom,
        y0: np.ndarray,
        t_span: tuple[float, float],
        maneuvers: list[ManeuverEvent],
        method: str,
        rtol: float,
        atol: float,
    ) -> TrajectoryData:
        t0, tf = t_span
        planned = [m for m in maneuvers if t0 <= m.time <= tf and m.dv_magnitude > 0.0]
        planned.sort(key=lambda m: m.time)

        if len(planned) == 0:
            result = propagate(
                eom,
                y0,
                t_span,
                method=method,
                rtol=rtol,
                atol=atol,
            )
            return TrajectoryData(t=result["t"], r=result["r"], v=result["v"])

        t_parts: list[np.ndarray] = []
        r_parts: list[np.ndarray] = []
        v_parts: list[np.ndarray] = []

        y_curr = y0.copy()
        t_curr = t0

        for burn in planned:
            if burn.time <= t_curr + 1e-9:
                y_curr = self._apply_burn(y_curr, burn)
                t_curr = max(t_curr, burn.time)
                continue

            result = propagate(
                eom,
                y_curr,
                (t_curr, burn.time),
                method=method,
                rtol=rtol,
                atol=atol,
            )

            if len(result["t"]) > 0:
                self._append_segment(
                    t_parts, r_parts, v_parts, result["t"], result["r"], result["v"]
                )
                y_curr = np.concatenate([result["r"][-1], result["v"][-1]])

            y_curr = self._apply_burn(y_curr, burn)
            t_curr = burn.time

        if t_curr < tf:
            result = propagate(
                eom,
                y_curr,
                (t_curr, tf),
                method=method,
                rtol=rtol,
                atol=atol,
            )
            if len(result["t"]) > 0:
                self._append_segment(
                    t_parts, r_parts, v_parts, result["t"], result["r"], result["v"]
                )

        t_hist = np.concatenate(t_parts)
        r_hist = np.vstack(r_parts)
        v_hist = np.vstack(v_parts)
        return TrajectoryData(t=t_hist, r=r_hist, v=v_hist)

    def _append_segment(self, t_parts, r_parts, v_parts, t_seg, r_seg, v_seg):
        if len(t_parts) == 0:
            t_parts.append(t_seg)
            r_parts.append(r_seg)
            v_parts.append(v_seg)
            return

        # Avoid duplicate sample at segment boundaries.
        t_parts.append(t_seg[1:])
        r_parts.append(r_seg[1:])
        v_parts.append(v_seg[1:])

    def _apply_burn(self, state6: np.ndarray, burn: ManeuverEvent) -> np.ndarray:
        r = state6[:3]
        v = state6[3:6]

        v_norm = np.linalg.norm(v)
        r_norm = np.linalg.norm(r)
        if v_norm < 1e-10 or r_norm < 1e-10:
            return state6

        v_hat = v / v_norm
        r_hat = r / r_norm
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        h_hat = h / h_norm if h_norm > 1e-10 else np.array([0.0, 0.0, 1.0])

        direction = burn.direction.lower()
        direction_map = {
            "prograde": v_hat,
            "retrograde": -v_hat,
            "normal": h_hat,
            "antinormal": -h_hat,
            "radial_out": r_hat,
            "radial_in": -r_hat,
        }
        dv_vec = direction_map.get(direction, v_hat) * burn.dv_magnitude

        new_state = state6.copy()
        new_state[3:6] = v + dv_vec

        post_coe = state_to_coe(StateVector(r=new_state[:3], v=new_state[3:6]))
        self._burn_events.append(
            {
                "time": float(burn.time),
                "dv": float(burn.dv_magnitude),
                "direction": direction,
                "r": new_state[:3].copy(),
                "v": new_state[3:6].copy(),
                "a": float(post_coe.a),
                "e": float(post_coe.e),
                "i": float(post_coe.i),
            }
        )
        return new_state
