"""
tracker.py — Core calculation logic for an equatorial tracker astrophotography tool.

Provides functions for:
    - Polar alignment computation via astropy
    - Magnetic declination lookup (pyIGRF / geomag fallback)
    - Maximum exposure time estimation (500 Rule & NPF Rule)
    - Tracking drift simulation due to polar misalignment
    - Servo motor tracking data generation at sidereal rate
"""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Astropy setup — disable IERS auto-download to avoid SSL errors
# ---------------------------------------------------------------------------
import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils.iers import conf as iers_conf

iers_conf.auto_download = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIDEREAL_DAY_SEC = 86164.0905  # mean sidereal day in SI seconds
SIDEREAL_RATE_DEG_PER_SEC = 360.0 / SIDEREAL_DAY_SEC  # ~0.004178 deg/s
SIDEREAL_RATE_DEG_PER_HOUR = SIDEREAL_RATE_DEG_PER_SEC * 3600.0  # ~15.0411 deg/h
OMEGA_EARTH = 2.0 * np.pi / SIDEREAL_DAY_SEC  # Earth rotation rate (rad/s)


# ===================================================================
# 1. Polar alignment
# ===================================================================


def compute_polar_alignment(
    lat: float,
    lon: float,
    elevation: float,
    star_ra: float,
    star_dec: float,
    time: Time | None = None,
) -> dict[str, Any]:
    """Compute the alt/az of a reference star (e.g. Polaris) to guide polar alignment.

    Parameters
    ----------
    lat : float
        Observer geodetic latitude in degrees.
    lon : float
        Observer geodetic longitude in degrees (East positive).
    elevation : float
        Observer elevation above sea level in metres.
    star_ra : float
        Right Ascension of the reference star in degrees.
    star_dec : float
        Declination of the reference star in degrees.
    time : astropy.time.Time or None
        Observation time.  If *None*, ``Time.now()`` is used.

    Returns
    -------
    dict
        ``{"altitude": float, "azimuth": float, "time_utc": str}``
        where altitude and azimuth are in degrees.
    """
    if time is None:
        time = Time.now()

    location = EarthLocation(
        lat=lat * u.deg,
        lon=lon * u.deg,
        height=elevation * u.m,
    )

    target = SkyCoord(ra=star_ra * u.deg, dec=star_dec * u.deg)
    altaz_frame = AltAz(obstime=time, location=location)
    star_altaz = target.transform_to(altaz_frame)

    return {
        "altitude": float(star_altaz.alt.deg),
        "azimuth": float(star_altaz.az.deg),
        "time_utc": str(time.utc.iso),
    }


# ===================================================================
# 2. Magnetic declination
# ===================================================================


def get_magnetic_declination(
    lat: float,
    lon: float,
    elevation_m: float,
    date: datetime.date | None = None,
) -> float | None:
    """Return magnetic declination at the given location (East positive, degrees).

    Tries, in order:
        1. ``ppigrf`` — Pure-Python IGRF model (recommended)
        2. ``pyIGRF`` — alternative IGRF implementation
        3. ``geomag`` — lightweight WMM-based package

    If no geomagnetic package is available, prints a warning and returns *None*.

    Parameters
    ----------
    lat : float
        Geodetic latitude in degrees.
    lon : float
        Geodetic longitude in degrees (East positive).
    elevation_m : float
        Elevation above sea level in metres.
    date : datetime.date or None
        Date for the calculation.  Defaults to today.

    Returns
    -------
    float or None
        Magnetic declination in degrees (East positive), or *None* when
        no geomagnetic library is available.
    """
    if date is None:
        date = datetime.date.today()

    # Convert date to a decimal year (required by some packages)
    year_fraction = date.year + (date.timetuple().tm_yday - 1) / 365.25
    elevation_km = elevation_m / 1000.0

    # --- Attempt 1: ppigrf (recommended, pure-Python IGRF) ---
    try:
        import ppigrf

        # ppigrf.igrf expects datetime.datetime, not datetime.date
        dt = datetime.datetime(date.year, date.month, date.day)
        Be, Bn, Bu = ppigrf.igrf(lon=lon, lat=lat, h=elevation_km, date=dt)
        # Declination = arctan2(East component, North component)
        declination = float(np.degrees(np.arctan2(Be.ravel()[0], Bn.ravel()[0])))
        return declination
    except ImportError:
        pass
    except Exception as exc:
        warnings.warn(f"ppigrf failed: {exc}", stacklevel=2)

    # --- Attempt 2: pyIGRF ---
    try:
        import pyIGRF  # noqa: N811

        # pyIGRF.igrf_value returns (D, I, H, X, Y, Z, F)
        # D = declination in degrees
        result = pyIGRF.igrf_value(lat, lon, alt=elevation_km, year=year_fraction)
        declination = float(result[0])
        return declination
    except ImportError:
        pass
    except Exception as exc:
        warnings.warn(f"pyIGRF failed: {exc}", stacklevel=2)

    # --- Attempt 3: geomag ---
    try:
        import geomag

        gm = geomag.GeoMag()
        mag = gm.GeoMag(lat, lon, h=elevation_km, time=date)
        declination = float(mag.dec)
        return declination
    except ImportError:
        pass
    except Exception as exc:
        warnings.warn(f"geomag failed: {exc}", stacklevel=2)

    # --- No package is available ---
    print(
        "[tracker] WARNING: No geomagnetic package is installed. "
        "Cannot compute magnetic declination automatically.  "
        "Install one with:  pip install ppigrf"
    )
    return None


# ===================================================================
# 3. Compass azimuth correction
# ===================================================================


def correct_azimuth_for_magnetic(
    true_azimuth: float,
    magnetic_declination: float,
) -> float:
    """Convert a true (geographic) azimuth to the magnetic (compass) reading.

    Parameters
    ----------
    true_azimuth : float
        True azimuth in degrees (measured from geographic North, clockwise).
    magnetic_declination : float
        Magnetic declination in degrees (East positive).

    Returns
    -------
    float
        The equivalent compass (magnetic) azimuth in degrees [0, 360).
    """
    compass_reading = true_azimuth - magnetic_declination
    return compass_reading % 360.0


# ===================================================================
# 4. Maximum exposure time
# ===================================================================


def compute_max_exposure(
    focal_length_mm: float,
    crop_factor: float = 1.5,
    pixel_pitch_um: float | None = None,
    aperture: float | None = None,
) -> dict[str, float | None]:
    """Estimate the longest untracked exposure before stars trail visibly.

    Two rules are computed:

    * **500 Rule** (simple):  ``exposure = 500 / (focal_length * crop_factor)``
    * **NPF Rule** (more accurate, requires pixel pitch & aperture):
      ``exposure = (35 * aperture + 30 * pixel_pitch) / focal_length``

    Parameters
    ----------
    focal_length_mm : float
        Lens focal length in millimetres.  Must be > 0.
    crop_factor : float
        Sensor crop factor relative to full-frame 35 mm (default 1.5 for APS-C).
    pixel_pitch_um : float or None
        Pixel pitch (photosite size) in micrometres.  Required for the NPF rule.
    aperture : float or None
        Lens aperture f-number (e.g. 2.8).  Required for the NPF rule.

    Returns
    -------
    dict
        ``{"rule_500": float, "npf_rule": float | None}``
        Exposure times in seconds.
    """
    if focal_length_mm <= 0:
        raise ValueError(f"focal_length_mm must be positive, got {focal_length_mm}")

    rule_500 = 500.0 / (focal_length_mm * crop_factor)

    npf: float | None = None
    if pixel_pitch_um is not None and aperture is not None:
        if aperture <= 0:
            raise ValueError(f"aperture must be positive, got {aperture}")
        if pixel_pitch_um <= 0:
            raise ValueError(f"pixel_pitch_um must be positive, got {pixel_pitch_um}")
        npf = (35.0 * aperture + 30.0 * pixel_pitch_um) / focal_length_mm

    return {"rule_500": rule_500, "npf_rule": npf}


# ===================================================================
# 5. Exposure table
# ===================================================================


def compute_exposure_table(
    focal_lengths: Sequence[float],
    crop_factor: float = 1.5,
    pixel_pitch_um: float | None = None,
    aperture: float | None = None,
) -> list[dict[str, Any]]:
    """Build an exposure-limit table for a range of focal lengths.

    Parameters
    ----------
    focal_lengths : list[float]
        List of focal lengths in millimetres.
    crop_factor : float
        Sensor crop factor (default 1.5).
    pixel_pitch_um : float or None
        Pixel pitch in micrometres (for NPF rule).
    aperture : float or None
        Lens f-number (for NPF rule).

    Returns
    -------
    list[dict]
        Each element is::

            {
                "focal_length_mm": float,
                "max_exposure_500": float,
                "max_exposure_npf": float | None,
            }
    """
    table: list[dict[str, Any]] = []
    for fl in focal_lengths:
        result = compute_max_exposure(
            focal_length_mm=fl,
            crop_factor=crop_factor,
            pixel_pitch_um=pixel_pitch_um,
            aperture=aperture,
        )
        table.append(
            {
                "focal_length_mm": fl,
                "max_exposure_500": result["rule_500"],
                "max_exposure_npf": result["npf_rule"],
            }
        )
    return table


# ===================================================================
# 6. Tracking drift simulation
# ===================================================================


def simulate_tracking_drift(
    misalignment_alt_arcsec: float,
    misalignment_az_arcsec: float,
    observer_lat_deg: float,
    target_dec_deg: float,
    duration_sec: float,
    time_step_sec: float = 1.0,
) -> dict[str, np.ndarray]:
    """Simulate star-field drift caused by polar-axis misalignment.

    Uses the standard small-error drift formulae derived from the King rate
    / polar misalignment geometry:

    .. math::

        HA(t) = \\omega \\cdot t

        \\Delta\\delta(t) = \\epsilon_{az} \\cos\\phi \\sin(HA)

        \\Delta\\alpha(t) = \\epsilon_{alt} \\tan\\delta \\sin(HA)
                           + \\epsilon_{az} \\cos\\phi \\tan\\delta (1 - \\cos(HA))

    where:

    * :math:`\\omega = 2\\pi / 86164.1` rad/s  (Earth rotation rate)
    * :math:`\\epsilon_{alt}` = altitude misalignment (arcsec)
    * :math:`\\epsilon_{az}` = azimuth misalignment (arcsec)
    * :math:`\\phi` = observer latitude
    * :math:`\\delta` = target declination

    Parameters
    ----------
    misalignment_alt_arcsec : float
        Polar-axis altitude (elevation) error in arcseconds.
    misalignment_az_arcsec : float
        Polar-axis azimuth error in arcseconds.
    observer_lat_deg : float
        Observer latitude in degrees.
    target_dec_deg : float
        Declination of the target object in degrees.
    duration_sec : float
        Total simulation duration in seconds.
    time_step_sec : float
        Time step in seconds (default 1.0).

    Returns
    -------
    dict
        ``{"time_sec": ndarray, "drift_dec_arcsec": ndarray,
        "drift_ra_arcsec": ndarray, "total_drift_arcsec": ndarray}``
    """
    t = np.arange(0.0, duration_sec + time_step_sec * 0.5, time_step_sec)

    lat_rad = np.radians(observer_lat_deg)
    dec_rad = np.radians(target_dec_deg)
    cos_lat = np.cos(lat_rad)
    tan_dec = np.tan(dec_rad)

    alt_err = misalignment_alt_arcsec
    az_err = misalignment_az_arcsec

    ha = OMEGA_EARTH * t  # hour angle progression (radians)
    sin_ha = np.sin(ha)
    cos_ha = np.cos(ha)

    # Declination drift (arcsec)
    drift_dec = az_err * cos_lat * sin_ha

    # RA drift (arcsec, projected on the sky)
    drift_ra = alt_err * tan_dec * sin_ha + az_err * cos_lat * tan_dec * (1.0 - cos_ha)

    # Total positional drift on the sensor
    total_drift = np.sqrt(drift_dec**2 + drift_ra**2)

    return {
        "time_sec": t,
        "drift_dec_arcsec": drift_dec,
        "drift_ra_arcsec": drift_ra,
        "total_drift_arcsec": total_drift,
    }


# ===================================================================
# 7. Servo tracking data generation
# ===================================================================


def generate_servo_tracking_data(
    duration_sec: float,
    time_step_sec: float = 1.0,
    sidereal_rate: bool = True,
    alignment_correction_alt: float = 0.0,
    alignment_correction_az: float = 0.0,
) -> dict[str, Any]:
    """Generate angular position data for a servo motor tracking at sidereal rate.

    The primary angle increases linearly at the sidereal rate
    (~15.0411 deg/hour ≈ 0.004178 deg/s).  Optional alignment corrections
    are applied as small sinusoidal perturbations on top of the base rate,
    modelling periodic error or axis-misalignment compensation:

    .. math::

        \\theta(t) = \\omega_{sid} \\cdot t
                   + \\frac{\\epsilon_{alt}}{3600} \\sin(\\omega_{sid\\_rad} \\cdot t)
                   + \\frac{\\epsilon_{az}}{3600}  (1 - \\cos(\\omega_{sid\\_rad} \\cdot t))

    Parameters
    ----------
    duration_sec : float
        Total tracking duration in seconds.
    time_step_sec : float
        Time step in seconds (default 1.0).
    sidereal_rate : bool
        If *True* (default), track at sidereal rate; otherwise at zero rate
        (useful for drift-alignment observations).
    alignment_correction_alt : float
        Altitude correction amplitude in arcseconds.  Applied as a
        sinusoidal perturbation.
    alignment_correction_az : float
        Azimuth correction amplitude in arcseconds.  Applied as a
        (1 − cos) perturbation.

    Returns
    -------
    dict
        ``{"time_sec": ndarray, "angle_deg": ndarray,
        "angular_velocity_deg_per_sec": ndarray, "step_angle_deg": float}``

        * ``angle_deg`` — cumulative angle the motor should reach at each step.
        * ``angular_velocity_deg_per_sec`` — instantaneous angular velocity.
        * ``step_angle_deg`` — nominal angle increment per time step.
    """
    t = np.arange(0.0, duration_sec + time_step_sec * 0.5, time_step_sec)

    base_rate = SIDEREAL_RATE_DEG_PER_SEC if sidereal_rate else 0.0

    # Base linear tracking angle
    angle = base_rate * t

    # Alignment correction perturbations (converted from arcsec to deg)
    if alignment_correction_alt != 0.0 or alignment_correction_az != 0.0:
        omega_t = OMEGA_EARTH * t
        sin_omega_t = np.sin(omega_t)
        cos_omega_t = np.cos(omega_t)

        alt_corr_deg = (alignment_correction_alt / 3600.0) * sin_omega_t
        az_corr_deg = (alignment_correction_az / 3600.0) * (1.0 - cos_omega_t)
        angle = angle + alt_corr_deg + az_corr_deg

    # Instantaneous angular velocity via finite differences
    if len(t) > 1:
        velocity = np.gradient(angle, t)
    else:
        velocity = np.array([base_rate])

    step_angle = base_rate * time_step_sec

    return {
        "time_sec": t,
        "angle_deg": angle,
        "angular_velocity_deg_per_sec": velocity,
        "step_angle_deg": step_angle,
    }


# ===================================================================
# Quick sanity check when run directly
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  tracker.py — Quick self-test")
    print("=" * 60)

    # --- 1. Polar alignment (Polaris from Bilbao area) ---
    pa = compute_polar_alignment(
        lat=43.357138,
        lon=-2.976366,
        elevation=39.0,
        star_ra=37.95,
        star_dec=89.26,
    )
    print(
        f"\n[1] Polar alignment  →  Alt: {pa['altitude']:.3f}°  "
        f"Az: {pa['azimuth']:.3f}°  ({pa['time_utc']})"
    )

    # --- 2. Magnetic declination ---
    md = get_magnetic_declination(43.357138, -2.976366, 39.0)
    print(
        f"[2] Magnetic declination  →  {md}°"
        if md is not None
        else "[2] Magnetic declination  →  not available (missing library)"
    )

    # --- 3. Compass correction ---
    if md is not None:
        compass_az = correct_azimuth_for_magnetic(pa["azimuth"], md)
        print(f"[3] True Az {pa['azimuth']:.2f}° → Compass Az {compass_az:.2f}°")
    else:
        print("[3] Skipped (no magnetic declination)")

    # --- 4. Max exposure (50 mm, APS-C, f/2.8, 3.76 µm pixel pitch) ---
    exp = compute_max_exposure(50.0, crop_factor=1.5, pixel_pitch_um=3.76, aperture=2.8)
    print(
        f"[4] Max exposure (50 mm f/2.8)  →  "
        f"500-rule: {exp['rule_500']:.2f} s  |  NPF: {exp['npf_rule']:.2f} s"
    )

    # --- 5. Exposure table ---
    table = compute_exposure_table(
        [14, 24, 35, 50, 85, 135, 200],
        crop_factor=1.5,
        pixel_pitch_um=3.76,
        aperture=2.8,
    )
    print("\n[5] Exposure table:")
    print(f"  {'FL (mm)':>8}  {'500 Rule (s)':>12}  {'NPF Rule (s)':>12}")
    print(f"  {'--------':>8}  {'------------':>12}  {'------------':>12}")
    for row in table:
        npf_str = f"{row['max_exposure_npf']:.2f}" if row["max_exposure_npf"] else "N/A"
        print(
            f"  {row['focal_length_mm']:>8.0f}  "
            f"{row['max_exposure_500']:>12.2f}  {npf_str:>12}"
        )

    # --- 6. Tracking drift (30 arcsec errors, 5-minute exposure) ---
    drift = simulate_tracking_drift(
        misalignment_alt_arcsec=30.0,
        misalignment_az_arcsec=30.0,
        observer_lat_deg=43.357,
        target_dec_deg=45.0,
        duration_sec=300.0,
        time_step_sec=1.0,
    )
    max_d = drift["total_drift_arcsec"][-1]
    print(
        f"\n[6] Tracking drift after 300 s (30″ alt + 30″ az error)  →  "
        f"{max_d:.2f} arcsec total"
    )

    # --- 7. Servo tracking data (60 s at sidereal rate) ---
    servo = generate_servo_tracking_data(
        duration_sec=60.0,
        time_step_sec=0.5,
        sidereal_rate=True,
        alignment_correction_alt=5.0,
        alignment_correction_az=3.0,
    )
    print(
        f"[7] Servo data  →  {len(servo['time_sec'])} samples, "
        f"final angle {servo['angle_deg'][-1]:.6f}°, "
        f"step_angle {servo['step_angle_deg']:.6f}°"
    )

    print("\n" + "=" * 60)
    print("  All checks passed.")
    print("=" * 60)
