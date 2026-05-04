from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StateVector:
    r: np.ndarray
    v: np.ndarray
    m: float | None = None
    epoch: float = 0.0

    def to_array(self) -> np.ndarray:
        if self.m is not None:
            return np.concatenate([self.r, self.v, [self.m]])
        return np.concatenate([self.r, self.v])

    @classmethod
    def from_array(cls, arr: np.ndarray, epoch: float = 0.0) -> StateVector:
        if len(arr) == 7:
            return cls(r=arr[:3].copy(), v=arr[3:6].copy(), m=arr[6], epoch=epoch)
        return cls(r=arr[:3].copy(), v=arr[3:6].copy(), epoch=epoch)


@dataclass
class OrbitalElements:
    a: float
    e: float
    i: float
    raan: float
    omega: float
    theta: float

    def to_dict(self) -> dict:
        return {
            "a": self.a, "e": self.e, "i_deg": self.i,
            "raan_deg": self.raan, "omega_deg": self.omega, "theta_deg": self.theta
        }
