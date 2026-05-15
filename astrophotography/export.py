"""
Export utilities for the equatorial tracker astrophotography tool.

Provides functions to serialize servo tracking profiles, alignment reports,
and exposure tables to JSON and CSV formats.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_python(value: Any) -> Any:
    """Convert numpy scalars and arrays to native Python types for JSON
    serialization.

    Args:
        value: Any value that might be a numpy type.

    Returns:
        The equivalent Python built-in type.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert every value in a dict so it is JSON-serializable."""
    sanitized: Dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_dict(value)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [_to_python(v) for v in value]
        else:
            sanitized[key] = _to_python(value)
    return sanitized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_servo_to_json(
    servo_data: Dict[str, Any],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export servo tracking data to a JSON file.

    The output JSON has the structure::

        {
            "metadata": { ... },
            "tracking_profile": [
                {"time_sec": t, "angle_deg": a, "velocity_deg_s": v},
                ...
            ]
        }

    Args:
        servo_data: Dictionary containing at least the keys
            ``"time_sec"``, ``"angle_deg"``, and
            ``"angular_velocity_deg_per_sec"`` (numpy arrays or plain
            Python sequences).  An optional ``"step_angle_deg"`` key is
            stored inside metadata when present.
        filepath: Destination file path (will be created / overwritten).
        metadata: Optional dictionary with observer location, alignment
            info, or any other auxiliary data to embed in the file.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.

    Raises:
        OSError: If the file cannot be created or written.
        KeyError: If required keys are missing from *servo_data*.
    """
    filepath = Path(filepath)

    time_arr = _to_python(servo_data["time_sec"])
    angle_arr = _to_python(servo_data["angle_deg"])
    velocity_arr = _to_python(servo_data["angular_velocity_deg_per_sec"])

    # Ensure we are working with lists
    if not isinstance(time_arr, list):
        time_arr = [time_arr]
    if not isinstance(angle_arr, list):
        angle_arr = [angle_arr]
    if not isinstance(velocity_arr, list):
        velocity_arr = [velocity_arr]

    tracking_profile: List[Dict[str, float]] = []
    for t, a, v in zip(time_arr, angle_arr, velocity_arr):
        tracking_profile.append(
            {
                "time_sec": float(t),
                "angle_deg": float(a),
                "velocity_deg_s": float(v),
            }
        )

    # Build metadata section
    meta: Dict[str, Any] = {}
    if metadata is not None:
        meta.update(_sanitize_dict(metadata))
    if "step_angle_deg" in servo_data:
        meta["step_angle_deg"] = _to_python(servo_data["step_angle_deg"])

    output = {
        "metadata": meta,
        "tracking_profile": tracking_profile,
    }

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(f"Failed to write servo JSON to '{filepath}': {exc}") from exc

    return filepath


def export_servo_to_csv(
    servo_data: Dict[str, Any],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export servo tracking data to a CSV file.

    If *metadata* is provided its key/value pairs are written as comment
    lines (prefixed with ``#``) at the top of the file.

    Columns: ``time_sec``, ``angle_deg``, ``angular_velocity_deg_per_sec``

    Args:
        servo_data: Dictionary containing at least the keys
            ``"time_sec"``, ``"angle_deg"``, and
            ``"angular_velocity_deg_per_sec"`` (numpy arrays or plain
            Python sequences).
        filepath: Destination file path (will be created / overwritten).
        metadata: Optional dictionary whose entries are written as
            comment lines at the top of the CSV.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.

    Raises:
        OSError: If the file cannot be created or written.
        KeyError: If required keys are missing from *servo_data*.
    """
    filepath = Path(filepath)

    time_arr = _to_python(servo_data["time_sec"])
    angle_arr = _to_python(servo_data["angle_deg"])
    velocity_arr = _to_python(servo_data["angular_velocity_deg_per_sec"])

    # Ensure we are working with lists
    if not isinstance(time_arr, list):
        time_arr = [time_arr]
    if not isinstance(angle_arr, list):
        angle_arr = [angle_arr]
    if not isinstance(velocity_arr, list):
        velocity_arr = [velocity_arr]

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="") as fh:
            # Write metadata as comment lines
            if metadata is not None:
                sanitized_meta = _sanitize_dict(metadata)
                for key, value in sanitized_meta.items():
                    fh.write(f"# {key}: {value}\n")

            writer = csv.writer(fh)
            writer.writerow(["time_sec", "angle_deg", "angular_velocity_deg_per_sec"])
            for t, a, v in zip(time_arr, angle_arr, velocity_arr):
                writer.writerow([float(t), float(a), float(v)])
    except OSError as exc:
        raise OSError(f"Failed to write servo CSV to '{filepath}': {exc}") from exc

    return filepath


def export_alignment_report(
    alignment_data: Dict[str, Any],
    filepath: Union[str, Path],
) -> Path:
    """Export a comprehensive polar-alignment report to a JSON file.

    The *alignment_data* dictionary may include (but is not limited to):

    * ``polar_alt`` – polar axis altitude (degrees)
    * ``polar_az`` – polar axis azimuth (degrees)
    * ``magnetic_declination`` – local magnetic declination (degrees)
    * ``compass_azimuth`` – compass reading towards the pole (degrees)
    * ``latitude`` / ``longitude`` – observer coordinates
    * ``time`` – observation time (ISO 8601 string or similar)
    * ``exposure_table`` – list of exposure entries

    All numpy types are automatically converted to native Python types.

    Args:
        alignment_data: Dictionary with alignment and observation data.
        filepath: Destination file path (will be created / overwritten).

    Returns:
        The resolved :class:`~pathlib.Path` that was written.

    Raises:
        OSError: If the file cannot be created or written.
    """
    filepath = Path(filepath)

    # Deep-sanitize the whole dict so it can be serialised
    report = _sanitize_dict(alignment_data)

    # Handle nested lists of dicts (e.g. exposure_table)
    for key, value in alignment_data.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            report[key] = [_sanitize_dict(entry) for entry in value]

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write alignment report to '{filepath}': {exc}"
        ) from exc

    return filepath


def export_exposure_table_csv(
    exposure_table: List[Dict[str, Any]],
    filepath: Union[str, Path],
) -> Path:
    """Export an exposure-time table to a CSV file.

    Each entry in *exposure_table* is expected to contain at least:

    * ``focal_length_mm`` – lens focal length in millimetres
    * ``max_exposure_500`` – max exposure via the 500-rule (seconds)
    * ``max_exposure_npf`` – max exposure via the NPF rule (seconds)

    Additional keys present in the dictionaries are also written.

    Args:
        exposure_table: List of dictionaries, one per focal length.
        filepath: Destination file path (will be created / overwritten).

    Returns:
        The resolved :class:`~pathlib.Path` that was written.

    Raises:
        OSError: If the file cannot be created or written.
        ValueError: If *exposure_table* is empty.
    """
    filepath = Path(filepath)

    if not exposure_table:
        raise ValueError("exposure_table is empty; nothing to export.")

    # Collect all unique column names preserving insertion order
    columns: list[str] = []
    for entry in exposure_table:
        for key in entry:
            if key not in columns:
                columns.append(key)

    # Guarantee the three main columns come first
    preferred_order = ["focal_length_mm", "max_exposure_500", "max_exposure_npf"]
    ordered_columns: list[str] = [c for c in preferred_order if c in columns]
    for c in columns:
        if c not in ordered_columns:
            ordered_columns.append(c)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=ordered_columns)
            writer.writeheader()
            for entry in exposure_table:
                sanitized_row = {k: _to_python(v) for k, v in entry.items()}
                writer.writerow(sanitized_row)
    except OSError as exc:
        raise OSError(
            f"Failed to write exposure table CSV to '{filepath}': {exc}"
        ) from exc

    return filepath
