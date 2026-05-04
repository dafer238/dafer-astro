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
- **Angular momentum conservation** verified to ~10⁻¹² relative drift
- **Integrator accuracy**: rtol=1e-10, atol=1e-12

### Multi-Spacecraft Support
- Create unlimited spacecraft with individual orbital elements
- Per-spacecraft color coding (Cyan, Green, Red, Yellow, Orange, Purple, White) and naming
- Select any spacecraft to view/edit its parameters
- **Update SC** button to modify orbital elements after creation
- **Delete SC** and **Clear SCs** buttons
- Click-to-select spacecraft orbit in 3D viewport
- Simulation duration auto-adjusts to accommodate all spacecraft orbital periods
- **Closest approach computation** between all spacecraft pairs with 3D line marker

### Orbital Elements Input
- Classical Orbital Elements (a, e, i, Ω, ω, θ) with fine-control drag inputs
- **Live orbit preview** — adjusting any COE instantly shows the resulting orbit in green
- **SC spawn marker** — yellow dot on preview orbit shows spacecraft position at current θ
- Ctrl+click on any drag field for direct typed input
- Speed 0.1°/px for angular values (i, RAAN, ω, θ)

### Orbit Presets
- ISS (6779 km, i=51.6°)
- GEO (42157 km, circular, equatorial)
- Molniya (600×39000 km, i=63.4°)
- SSO (6971 km, i=97.8°)
- HEO (300×60000 km)
- GPS/MEO (20200 km, i=55°)
- Tundra (24h period, e=0.268, i=63.4°)
- Lunar (TLI to 384400 km)
- GTO (185×35786 km)

### Maneuver Planning (KSP-Style)

#### Manual Burns
- **Impulsive burns** with 6 directions: prograde, retrograde, normal, antinormal, radial out, radial in
- **Burn timing** via: At current time, Periapsis, Apoapsis, Ascending Node, Descending Node
- **Live burn preview** — adjusting dv, direction, or anchor instantly shows the resulting trajectory
- **Multi-burn sequences** with automatic trajectory segmentation
- **Burn cursor** — 3D arrow at burn location showing direction
- Add/Remove/Clear burn management
- Burn plan display with numbered sequence

#### Maneuver Templates
- **Hohmann transfer** — automatic two-burn sequence with transfer arc visualization
- **Bi-elliptic transfer** — three-burn sequence with two transfer arcs
- **Plane change** — single normal burn for inclination adjustment
- **Combined transfer+plane** — Hohmann + plane change at circularization
- **Circularize** — prograde burn at apoapsis
- **De-orbit** — retrograde burn lowering periapsis to 80 km
- **Phasing** — two-burn period adjustment to drift 30°
- **Rendezvous** — co-planar Hohmann to target altitude

#### Template Controls
- Target altitude input (auto-updates maneuver preview)
- Plane change angle input
- Maneuver anchor selection (At current time, Periapsis, Apoapsis, Ascending/Descending Node)

### Perturbation Modeling
- **J2 oblateness** (with computed RAAN/argument of periapsis drift rates)
- **Atmospheric drag** (configurable Cd 1.5–3.5, ballistic coefficient)
- **Moon gravity** (third-body perturbation)
- **Sun gravity** (third-body perturbation)
- All independently toggleable per simulation

### Real-Time 3D Visualization
- Interactive 3D viewport with mouse orbit, pan, and zoom
- **Central body rendering** with proper visual differentiation (Earth blue, Moon gray, etc.)
- **Orbit annotations**: periapsis (P), apoapsis (A), ascending/descending nodes
- **Multiple trajectory rendering** with per-spacecraft colors
- **Satellite markers** on all active spacecraft (animated during playback)
- **Closest approach line** with distance label between spacecraft pairs
- **Preview orbit** (green) with SC spawn position marker (yellow)
- **Transfer trajectory** visualization (yellow arcs)
- **Burn markers** (orange dots) at maneuver locations
- **Burn direction cursor** (3D arrow at burn point)
- **Reference/compare trajectory** overlay

#### View Presets
- Isometric, XY (ecliptic top-down), XZ, YZ plane views
- Orbit-normal view, Velocity-track view, Nadir view
- Auto-fit to content

### Timeline & Playback
- Play/Pause with variable time warp (1x → 50,000x)
- Timeline scrub slider
- Time display with human-readable format (Xy Xm Xd HH:MM)
- Reset to t=0

### Telemetry Panel
- Real-time display at current time step:
  - Altitude, Velocity, Orbital Period, Specific Energy
  - Semi-major axis, Eccentricity, Inclination, RAAN, Argument of Periapsis, True Anomaly

### 2D Plots
- **Altitude vs Time** with vertical cursor line
- **Velocity vs Time** with vertical cursor line
- Cursor follows playback/scrub position

### Comparison Mode
- Toggle compare mode to overlay burn vs. no-burn trajectory
- Displays final differences: delta altitude, delta velocity, position miss

### Two-Tab Interface

#### Tab 1: Orbital / Rendezvous
Full orbital mechanics simulation with all features above.

#### Tab 2: Interplanetary / Multi-Body
- Full solar system ephemeris (Standish 1992 elements) for any date
- Multi-body propagation with N gravitational sources
- Spacecraft builder with custom COE per craft
- **Planet orbit visualization** — Mercury through Saturn with full orbital paths
- **Proportional body rendering** via perspective limb-point projection
- View presets: Earth (LEO scale), Moon (close-up), Mars (inner system), Jupiter (outer), Solar System (full)
- Epoch selector (any date/time UTC)
- Perturbing body selection
- Telemetry and relative distance panels

### Solar System Ephemeris
- Accurate planetary positions for any Julian Date
- Based on Standish (1992) mean orbital elements with secular rates
- Covers: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
- Moon geocentric position (Brown's lunar theory simplified)
- Ecliptic-to-equatorial coordinate transformation

### File Management
- **New** — start fresh (with unsaved-changes prompt)
- **Open** — load simulation from JSON file
- **Save / Save As** — serialize all spacecraft, orbits, burns, and settings to JSON
- Unsaved changes detection with save/discard/cancel prompt

### Export & Analysis
- **CSV trajectory export** (time, position, velocity)
- **PNG plot export** via Matplotlib
- Energy/angular momentum conservation validation logged to console
- Post-burn orbital element computation with cumulative ΔV tracking

### Console
- Built-in console log (last 50 messages)
- Reports: computation progress, conservation metrics, export paths, errors, burn events

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.11+
- NumPy ≥ 1.24
- SciPy ≥ 1.11 (solve_ivp with DOP853)
- DearPyGui ≥ 2.0
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
| Play/Pause | Play button |
| Time warp | Speed dropdown (1x–50,000x) |
| Scrub time | Timeline slider |
| Clear preview | Escape key |
| Direct value input | Ctrl+click on drag fields |

## Verification

The simulator validates its own physics:
- Energy conservation: |ΔE/E| < 10⁻¹² for unperturbed orbits
- Angular momentum conservation: |Δh/h| < 10⁻¹² for unperturbed orbits
- Orbital period accuracy verified against analytical solutions
- Perturbation effects compared against published data

## License

MIT
