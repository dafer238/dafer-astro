import numpy as np
from simulator.core.constants import R_EARTH

ATMO_TABLE = [
    (0, 1.225e9, 8.44),
    (25, 3.899e7, 6.49),
    (40, 3.972e6, 7.07),
    (60, 3.206e5, 6.11),
    (80, 1.905e4, 5.53),
    (100, 5.604e2, 5.85),
    (110, 9.708e1, 7.69),
    (120, 2.222e1, 9.52),
    (130, 8.152e0, 12.30),
    (150, 2.076e0, 22.26),
    (180, 5.194e-1, 33.74),
    (200, 2.541e-1, 47.89),
    (250, 6.073e-2, 57.42),
    (300, 1.916e-2, 59.89),
    (350, 7.014e-3, 65.47),
    (400, 2.803e-3, 65.55),
    (450, 1.184e-3, 68.38),
    (500, 5.215e-4, 73.58),
    (600, 1.137e-4, 76.30),
    (700, 3.070e-5, 72.32),
    (800, 1.136e-5, 74.89),
    (1000, 3.561e-6, 124.64),
]

_ATMO_ALTS = np.array([row[0] for row in ATMO_TABLE])


def atmo_density(alt_km: float) -> float:
    if alt_km <= 0:
        return ATMO_TABLE[0][1]
    if alt_km >= 1000:
        return 0.0
    idx = np.searchsorted(_ATMO_ALTS, alt_km, side='right') - 1
    h0, rho0, H = ATMO_TABLE[idx]
    return rho0 * np.exp(-(alt_km - h0) / H)
