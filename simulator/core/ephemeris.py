"""
Solar system ephemeris using simplified analytical models (VSOP87-like truncated series
and lunar theory) for planetary positions at any given epoch.

All positions returned in km, in the J2000 ecliptic frame.
Epoch is specified as a Julian Date (JD) or datetime.

Accuracy: ~1 arcminute for planets, ~10 arcminutes for the Moon over centuries.
"""

from __future__ import annotations
import numpy as np
from datetime import datetime, timezone
from simulator.core.constants import AU


def datetime_to_jd(dt: datetime) -> float:
    """Convert a datetime to Julian Date."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    frac = (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
    return jdn + frac


def jd_to_centuries(jd: float) -> float:
    """Julian centuries since J2000.0."""
    return (jd - 2451545.0) / 36525.0


def _wrap_angle(angle_rad: float) -> float:
    return angle_rad % (2 * np.pi)


# ============================================================================
# Planetary mean orbital elements (J2000 epoch, ecliptic)
# From Standish (1992) / Meeus "Astronomical Algorithms"
# Format: [a_AU, e, i_deg, L_deg, lonperi_deg, Omega_deg] + rates per century
# ============================================================================

_PLANET_ELEMENTS = {
    "Mercury": {
        "a": (0.38709927, 0.00000037),
        "e": (0.20563593, 0.00001906),
        "i": (7.00497902, -0.00594749),
        "L": (252.25032350, 149472.67411175),
        "lonperi": (77.45779628, 0.16047689),
        "Omega": (48.33076593, -0.12534081),
    },
    "Venus": {
        "a": (0.72333566, 0.00000390),
        "e": (0.00677672, -0.00004107),
        "i": (3.39467605, -0.00078890),
        "L": (181.97909950, 58517.81538729),
        "lonperi": (131.60246718, 0.00268329),
        "Omega": (76.67984255, -0.27769418),
    },
    "Earth": {
        "a": (1.00000261, 0.00000562),
        "e": (0.01671123, -0.00004392),
        "i": (0.00001531, -0.01294668),
        "L": (100.46457166, 35999.37244981),
        "lonperi": (102.93768193, 0.32327364),
        "Omega": (0.0, 0.0),
    },
    "Mars": {
        "a": (1.52371034, 0.00001847),
        "e": (0.09339410, 0.00007882),
        "i": (1.84969142, -0.00813131),
        "L": (355.44656299, 19140.30268499),  # corrected: was -4.55343205
        "lonperi": (336.05637041, 0.44441088),  # corrected: was -23.94362959
        "Omega": (49.55953891, -0.29257343),
    },
    "Jupiter": {
        "a": (5.20288700, -0.00011607),
        "e": (0.04838624, -0.00013253),
        "i": (1.30439695, -0.00183714),
        "L": (34.39644051, 3034.74612775),
        "lonperi": (14.72847983, 0.21252668),
        "Omega": (100.47390909, 0.20469106),
    },
    "Saturn": {
        "a": (9.53667594, -0.00125060),
        "e": (0.05386179, -0.00050991),
        "i": (2.48599187, 0.00193609),
        "L": (49.95424423, 1222.49362201),
        "lonperi": (92.59887831, -0.41897216),
        "Omega": (113.66242448, -0.28867794),
    },
    "Uranus": {
        "a": (19.18916464, -0.00196176),
        "e": (0.04725744, -0.00004397),
        "i": (0.77263783, -0.00242939),
        "L": (313.23810451, 428.48202785),
        "lonperi": (170.95427630, 0.40805281),
        "Omega": (74.01692503, 0.04240589),
    },
    "Neptune": {
        "a": (30.06992276, 0.00026291),
        "e": (0.00859048, 0.00005105),
        "i": (1.77004347, 0.00035372),
        "L": (304.87997031, 218.45945325),  # corrected: was -55.12002969
        "lonperi": (44.96476227, -0.32241464),
        "Omega": (131.78422574, -0.00508664),
    },
}

# Gravitational parameters in km^3/s^2
_PLANET_MU = {
    "Mercury": 22032.09,
    "Venus": 324858.63,
    "Earth": 398600.4418,
    "Mars": 42828.375,
    "Jupiter": 126686534.9,
    "Saturn": 37931187.0,
    "Uranus": 5793939.0,
    "Neptune": 6836529.0,
    "Sun": 1.3271244e11,
    "Moon": 4902.800066,
}

_PLANET_RADIUS_KM = {
    "Mercury": 2439.7,
    "Venus": 6051.8,
    "Earth": 6371.0,
    "Mars": 3389.5,
    "Jupiter": 69911.0,
    "Saturn": 58232.0,
    "Uranus": 25362.0,
    "Neptune": 24622.0,
    "Sun": 695700.0,
    "Moon": 1737.4,
}

_PLANET_COLORS = {
    "Mercury": (169, 169, 169),
    "Venus": (255, 198, 73),
    "Earth": (70, 130, 180),
    "Mars": (205, 92, 92),
    "Jupiter": (210, 180, 140),
    "Saturn": (238, 232, 170),
    "Uranus": (173, 216, 230),
    "Neptune": (65, 105, 225),
    "Sun": (255, 223, 0),
    "Moon": (200, 200, 200),
}


def _kepler_solve(M_rad: float, e: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson."""
    E = M_rad + e * np.sin(M_rad) if e < 0.8 else np.pi
    for _ in range(50):
        dE = (E - e * np.sin(E) - M_rad) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


def planet_ecliptic_position(planet_name: str, jd: float) -> np.ndarray:
    """
    Compute heliocentric ecliptic position of a planet in km at given JD.
    Returns 3D position vector [x, y, z] in km, ecliptic J2000 frame.
    """
    if planet_name == "Sun":
        return np.zeros(3)

    if planet_name == "Moon":
        return _moon_position_ecliptic(jd)

    elems = _PLANET_ELEMENTS[planet_name]
    T = jd_to_centuries(jd)

    a_au = elems["a"][0] + elems["a"][1] * T
    e = elems["e"][0] + elems["e"][1] * T
    i_deg = elems["i"][0] + elems["i"][1] * T
    L_deg = elems["L"][0] + elems["L"][1] * T
    lonperi_deg = elems["lonperi"][0] + elems["lonperi"][1] * T
    Omega_deg = elems["Omega"][0] + elems["Omega"][1] * T

    omega_deg = lonperi_deg - Omega_deg
    M_deg = L_deg - lonperi_deg
    M_rad = np.radians(M_deg % 360.0)

    E = _kepler_solve(M_rad, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

    r_mag = a_au * (1.0 - e * np.cos(E))

    # Convert to heliocentric ecliptic Cartesian
    omega = np.radians(omega_deg)
    Omega = np.radians(Omega_deg)
    inc = np.radians(i_deg)

    cos_O, sin_O = np.cos(Omega), np.sin(Omega)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    cos_w, sin_w = np.cos(omega + nu), np.sin(omega + nu)

    x = r_mag * (cos_O * cos_w - sin_O * sin_w * cos_i)
    y = r_mag * (sin_O * cos_w + cos_O * sin_w * cos_i)
    z = r_mag * (sin_w * sin_i)

    return np.array([x, y, z]) * AU  # Convert AU to km


def planet_ecliptic_velocity(planet_name: str, jd: float) -> np.ndarray:
    """
    Compute heliocentric ecliptic velocity via finite differences (km/s).
    """
    if planet_name == "Sun":
        return np.zeros(3)
    dt = 60.0  # seconds
    dt_jd = dt / 86400.0
    r1 = planet_ecliptic_position(planet_name, jd - dt_jd)
    r2 = planet_ecliptic_position(planet_name, jd + dt_jd)
    return (r2 - r1) / (2.0 * dt)


def _moon_position_ecliptic(jd: float) -> np.ndarray:
    """
    Simplified lunar position relative to Earth-Moon barycenter (geocentric),
    returned as HELIOCENTRIC ecliptic position in km.
    Uses simplified Brown's lunar theory terms.
    """
    T = jd_to_centuries(jd)

    # Fundamental arguments (degrees)
    Lp = 218.3164477 + 481267.88123421 * T  # Mean longitude
    D = 297.8501921 + 445267.1114034 * T  # Mean elongation
    M = 357.5291092 + 35999.0502909 * T  # Sun mean anomaly
    Mp = 134.9633964 + 477198.8675055 * T  # Moon mean anomaly
    F = 93.2720950 + 483202.0175233 * T  # Argument of latitude

    Lp_r = np.radians(Lp)
    D_r = np.radians(D)
    M_r = np.radians(M)
    Mp_r = np.radians(Mp)
    F_r = np.radians(F)

    # Longitude (simplified, main terms)
    lon_deg = Lp + (
        6.289 * np.sin(Mp_r)
        - 1.274 * np.sin(2 * D_r - Mp_r)
        + 0.658 * np.sin(2 * D_r)
        - 0.214 * np.sin(2 * Mp_r)
        - 0.186 * np.sin(M_r)
        - 0.114 * np.sin(2 * F_r)
    )

    # Latitude
    lat_deg = (
        5.128 * np.sin(F_r)
        + 0.281 * np.sin(Mp_r + F_r)
        - 0.278 * np.sin(F_r - Mp_r)
        - 0.173 * np.sin(2 * D_r - F_r)
    )

    # Distance in km
    r_km = 385000.56 + (
        -20905.355 * np.cos(Mp_r) - 3699.111 * np.cos(2 * D_r - Mp_r) - 2955.968 * np.cos(2 * D_r)
    )

    # Geocentric ecliptic cartesian
    lon_r = np.radians(lon_deg)
    lat_r = np.radians(lat_deg)
    x_geo = r_km * np.cos(lat_r) * np.cos(lon_r)
    y_geo = r_km * np.cos(lat_r) * np.sin(lon_r)
    z_geo = r_km * np.sin(lat_r)

    # Add Earth's heliocentric position to get Moon heliocentric
    earth_pos = planet_ecliptic_position("Earth", jd)
    return earth_pos + np.array([x_geo, y_geo, z_geo])


def moon_geocentric_position(jd: float) -> np.ndarray:
    """Moon position relative to Earth center, in ecliptic km."""
    T = jd_to_centuries(jd)

    Lp = 218.3164477 + 481267.88123421 * T
    D = 297.8501921 + 445267.1114034 * T
    M = 357.5291092 + 35999.0502909 * T
    Mp = 134.9633964 + 477198.8675055 * T
    F = 93.2720950 + 483202.0175233 * T

    Lp_r = np.radians(Lp)
    D_r = np.radians(D)
    M_r = np.radians(M)
    Mp_r = np.radians(Mp)
    F_r = np.radians(F)

    lon_deg = Lp + (
        6.289 * np.sin(Mp_r)
        - 1.274 * np.sin(2 * D_r - Mp_r)
        + 0.658 * np.sin(2 * D_r)
        - 0.214 * np.sin(2 * Mp_r)
        - 0.186 * np.sin(M_r)
        - 0.114 * np.sin(2 * F_r)
    )

    lat_deg = (
        5.128 * np.sin(F_r)
        + 0.281 * np.sin(Mp_r + F_r)
        - 0.278 * np.sin(F_r - Mp_r)
        - 0.173 * np.sin(2 * D_r - F_r)
    )

    r_km = 385000.56 + (
        -20905.355 * np.cos(Mp_r) - 3699.111 * np.cos(2 * D_r - Mp_r) - 2955.968 * np.cos(2 * D_r)
    )

    lon_r = np.radians(lon_deg)
    lat_r = np.radians(lat_deg)
    x = r_km * np.cos(lat_r) * np.cos(lon_r)
    y = r_km * np.cos(lat_r) * np.sin(lon_r)
    z = r_km * np.sin(lat_r)
    return np.array([x, y, z])


def get_solar_system_state(jd: float) -> dict:
    """
    Get positions and velocities of all solar system bodies at a given JD.
    Returns dict: name -> {"position": np.ndarray, "velocity": np.ndarray, "mu": float, "radius": float, "color": tuple}
    All in heliocentric ecliptic km and km/s.
    """
    bodies = {}
    all_names = [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
        "Moon",
    ]

    for name in all_names:
        pos = planet_ecliptic_position(name, jd)
        vel = planet_ecliptic_velocity(name, jd)
        bodies[name] = {
            "position": pos,
            "velocity": vel,
            "mu": _PLANET_MU[name],
            "radius": _PLANET_RADIUS_KM[name],
            "color": _PLANET_COLORS[name],
        }

    return bodies


def ecliptic_to_equatorial(vec: np.ndarray) -> np.ndarray:
    """Rotate from ecliptic to J2000 equatorial frame."""
    obliquity = np.radians(23.439291)
    c, s = np.cos(obliquity), np.sin(obliquity)
    R = np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )
    return R @ vec
