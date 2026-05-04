from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from simulator.core.bodies import CelestialBody
from simulator.core.state import OrbitalElements, StateVector


@dataclass
class PerturbationConfig:
    j2_enabled: bool = True
    drag_enabled: bool = False
    drag_cd: float = 2.2
    drag_ballistic_coeff: float = 5.6e-9
    third_body_moon: bool = False
    third_body_sun: bool = False


@dataclass
class ManeuverEvent:
    time: float
    dv_magnitude: float
    direction: str = "prograde"


@dataclass
class Scenario:
    initial_coe: OrbitalElements
    central_body: CelestialBody
    perturbations: PerturbationConfig = field(default_factory=PerturbationConfig)
    n_orbits: float = 3.0
    integrator_method: str = "DOP853"
    rtol: float = 1e-10
    atol: float = 1e-12
    maneuvers: list[ManeuverEvent] = field(default_factory=list)
    epoch: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
