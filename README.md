# Orbital Mechanics Simulator

A high-fidelity orbital mechanics simulator with real-time 3D visualization, multi-body physics, and KSP-inspired maneuver planning.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![DearPyGui](https://img.shields.io/badge/GUI-DearPyGui-green) ![Physics](https://img.shields.io/badge/Physics-Verified_10⁻¹²-brightgreen)

## Features

### Physics Engine
- **Two-body Keplerian propagation** with DOP853 integrator (8th-order Dormand-Prince)
- **N-body gravitational dynamics** with third-body perturbations (Moon, Sun, planets)
- **J2 oblateness perturbation** for realistic Earth-orbit modeling
- **Atmospheric drag** with exponential atmosphere model (configurable Cd and ballistic coefficient)
- **Energy conservation** verified to ~10⁻¹² relative drift
- **Integrator accuracy**: rtol=1e-10, atol=1e-12

### Multi-Spacecraft Support
- Create unlimited spacecraft with individual orbital elements
- Per-spacecraft color coding and naming
- Select any spacecraft to view/edit its parameters
- **Update SC** button to modify orbital elements after creation
- Simulation duration auto-adjusts to accommodate all spacecraft orbital periods

### Maneuver Planning (KSP-Style)
- **Impulsive burns** with 6 directions: prograde, retrograde, normal, antinormal, radial out, radial in
- **Burn timing** via percentage, periapsis, apoapsis, ascending/descending node
- **Live burn preview** — adjusting any parameter instantly shows the resulting orbit in the viewport
- **Multi-burn sequences** with automatic trajectory segmentation
- **Burn cursor** showing exact position and direction on the orbit
- **Hohmann transfer calculator** with automatic two-burn sequences
- **Inclination change** and **plane change** maneuver presets

### Real-Time 3D Visualization
- Interactive 3D viewport with mouse orbit, pan, and zoom
- **Central body rendering** with proper differentiation:
  - Earth: blue sphere with continents and latitude/longitude grid
  - Moon: gray sphere
  - Other planets: colored spheres at proportional scale
- **Orbit annotations**: periapsis, apoapsis, ascending/descending nodes
- **Multiple trajectory rendering** with per-spacecraft colors
- **Satellite markers** on all active spacecraft (animated during playback)
- **Closest approach line** with distance label between spacecraft
- **Preview orbit** (green) updates live as you adjust COE sliders
- **Transfer trajectory** visualization (yellow)
- **Burn markers** (orange dots) at maneuver locations
- View presets: Isometric, XY (ecliptic), XZ, body-specific zoom

### Two-Tab Interface

#### Tab 1: Orbital / Rendezvous
- Classical Orbital Elements input (a, e, i, Ω, ω, θ) with live orbit preview
- Multi-spacecraft builder with add/update/clear/select
- Full burn planning with live preview
- Perturbation toggles (J2, drag, Moon, Sun)
- Central body selector (Earth, Moon, Sun, Mars, Jupiter, etc.)
- Time warp: 1x → 50,000x
- Telemetry panel: altitude, velocity, orbital energy, period, elements
- Closest approach computation between all spacecraft pairs
- Comparison mode (burn vs. no-burn overlay)
- Orbit info panel with periapsis/apoapsis altitudes

#### Tab 2: Interplanetary / Multi-Body
- Full solar system ephemeris (Standish 1992 elements) for any date
- Multi-body propagation with N gravitational sources
- Spacecraft builder with custom COE per craft
- **Planet orbit visualization** — all planets Mercury→Saturn with full orbital paths
- **Proportional body rendering** via perspective limb-point projection
- View presets:
  - **Earth**: Low-Earth orbit scale with Moon orbit path
  - **Moon**: Camera centers on Moon's ephemeris position (close-up)
  - **Mars**: Inner solar system scale
  - **Jupiter**: Outer planet scale
  - **Solar System**: Full system view (Mercury to Saturn)
- Epoch selector (any date/time UTC)
- Perturbing body selection
- Telemetry and relative distance panels

### Solar System Ephemeris
- Accurate planetary positions for any Julian Date
- Based on Standish (1992) mean orbital elements with secular rates
- Covers: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
- Moon geocentric position (Brown's lunar theory simplified)
- Ecliptic-to-equatorial coordinate transformation
- Julian Date conversion from datetime

### Export & Analysis
- CSV trajectory export (time, position, velocity)
- Matplotlib orbit plots
- Energy/angular momentum conservation validation
- Post-burn orbital element computation

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.11+
- NumPy ≥ 1.24
- SciPy ≥ 1.11 (solve_ivp with DOP853)
- DearPyGui ≥ 1.11
- Matplotlib (for export plots)

## Usage

```bash
python -m simulator
```

## Units

All internal calculations use:
- **Distance**: km
- **Velocity**: km/s
- **Time**: seconds
- **Angles**: degrees (user-facing), radians (internal)

## Architecture

```
simulator/
├── core/           # Constants, state vectors, conversions, ephemeris, spacecraft
├── physics/        # EOM builder, perturbations, maneuvers, propagator, atmosphere
├── sim/            # Engine (threaded), scenario, trajectory, multi-body engine
├── render/         # Viewport3D, camera, projection
├── ui/             # Theme, interplanetary tab
├── export/         # CSV, matplotlib plots
├── app.py          # Main application (Tab 1 + orchestration)
└── __main__.py     # Entry point
```

## Theory Documentation

- [Phase 1: Two-Body Problem](phase01/theory_01_two_body.md)
- [Phase 2: Orbital Maneuvers](phase02/theory_02_maneuvers.md)
- [Phase 3: Perturbations](phase03/theory_03_perturbations.md)
- [Phase 4: Propulsion](phase04/theory_04_propulsion.md)

## Key Controls

| Action | Control |
|--------|---------|
| Rotate view | Left-click drag |
| Zoom | Scroll wheel |
| Pan | Right-click drag |
| Play/Pause | Play button or timeline |
| Time warp | Speed dropdown (1x–50,000x) |
| Scrub time | Timeline slider |

## Verification

The simulator validates its own physics:
- Energy conservation: |ΔE/E| < 10⁻¹² for unperturbed orbits
- Angular momentum conservation: |Δh/h| < 10⁻¹² for unperturbed orbits
- Orbital period accuracy verified against analytical solutions
- Perturbation effects compared against published data

## License

MIT
