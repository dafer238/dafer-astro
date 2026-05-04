from __future__ import annotations
from dataclasses import dataclass
from simulator.core.constants import (
    MU_EARTH, MU_SUN, MU_MOON, R_EARTH, R_SUN, R_MOON, J2_EARTH
)


@dataclass(frozen=True)
class CelestialBody:
    name: str
    mu: float
    radius: float
    j2: float = 0.0
    has_atmosphere: bool = False

    @classmethod
    def earth(cls) -> CelestialBody:
        return cls(name="Earth", mu=MU_EARTH, radius=R_EARTH,
                   j2=J2_EARTH, has_atmosphere=True)

    @classmethod
    def sun(cls) -> CelestialBody:
        return cls(name="Sun", mu=MU_SUN, radius=R_SUN)

    @classmethod
    def moon(cls) -> CelestialBody:
        return cls(name="Moon", mu=MU_MOON, radius=R_MOON)
