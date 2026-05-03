# %% [markdown]
# # 🚀 Phase 2 — Orbital Maneuvers
# ## Delta-v, Hohmann Transfers, Plane Changes, Rocket Equation
#
# **Format:** `# %%` (py:percent) — Zed/VS Code Jupyter, Spyder, or
#   `marimo convert phase2_maneuvers.py -o phase2_marimo.py`
#
# **Depends on:** phase1_two_body.py (functions are re-defined here for self-containment)
# **Read alongside:** theory_02_maneuvers.md
#
# **References:**
# - **[BMT]** Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020
# - **[S&B]** Sutton & Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# %% [markdown]
# ---
# ## 2. Constants and Phase-1 Core (self-contained copy)
#
# Every module is self-contained so you can run Phase 2 independently.
# Functions identical to Phase 1 are included without duplication of comments.

# %% CoreFunctions
MU_EARTH = 398_600.4418   # km^3/s^2
R_EARTH  = 6_371.0        # km
G0       = 9.80665e-3     # km/s^2  standard gravity (for Isp -> ve conversion)

def two_body_eom(t, state, mu):
    r_vec=state[:3]; v_vec=state[3:]
    r_mag=np.linalg.norm(r_vec)
    if r_mag < 1e-10: raise ValueError("Singularity")
    return np.concatenate([v_vec, -(mu/r_mag**3)*r_vec])

def rk4_step(f, t, y, h, *args):
    k1=f(t,y,*args); k2=f(t+h/2,y+(h/2)*k1,*args)
    k3=f(t+h/2,y+(h/2)*k2,*args); k4=f(t+h,y+h*k3,*args)
    return y+(h/6.0)*(k1+2*k2+2*k3+k4)

def propagate(eom, state0, t_span, dt, mu):
    t0,tf=t_span; n=int((tf-t0)/dt)
    t_arr=np.zeros(n+1); states=np.zeros((n+1,6))
    t_arr[0]=t0; states[0]=state0.copy()
    for i in range(n):
        states[i+1]=rk4_step(eom,t_arr[i],states[i],dt,mu)
        t_arr[i+1]=t_arr[i]+dt
    return {"t":t_arr,"states":states,"r":states[:,:3],"v":states[:,3:]}

def specific_energy(r_vec, v_vec, mu):
    return 0.5*np.linalg.norm(v_vec)**2 - mu/np.linalg.norm(r_vec)

def orbital_period(a, mu):
    return 2.0*np.pi*np.sqrt(a**3/mu)

def vis_viva(r, a, mu):
    return np.sqrt(mu*(2.0/r - 1.0/a))

def circular_orbit_state(alt_km, inc_deg=0.0, mu=MU_EARTH):
    r=R_EARTH+alt_km; vc=np.sqrt(mu/r); i=np.radians(inc_deg)
    return np.array([r,0.,0.,0.,vc*np.cos(i),vc*np.sin(i)])

print("Core functions loaded.")

# %% [markdown]
# ---
# ## 3. Impulsive Burns — The Instantaneous Velocity Change
#
# > **[BMT 2nd ed.] Section 6.1** — Introduction
# > **[BMT 2nd ed.] Section 6.2, p. 248** — In-plane orbit changes
#
# An impulsive burn adds a delta-v vector to the current velocity instantaneously.
# Position does not change. The pre-burn and post-burn orbits share exactly one point:
# the burn location.
#
# **Rule:** A prograde burn at periapsis raises apoapsis.
#            A prograde burn at apoapsis raises periapsis.
#            This is true because energy additions raise the "opposite" side.

# %% ImpulsiveBurn
def apply_burn(state, dv_vec):
    """
    Apply an instantaneous delta-v to a state vector.
    [BMT 2nd ed.] Section 6.1 — impulsive maneuver model

    state  : (6,) ndarray [r, v]
    dv_vec : (3,) ndarray  delta-v in km/s in the same frame as state

    Returns new state with v^+ = v^- + dv
    """
    new_state = state.copy()
    new_state[3:] += dv_vec
    return new_state


def tangential_dv(state, dv_magnitude):
    """
    Apply a tangential (along-track) delta-v.
    Tangential is the most efficient direction for changing orbit energy.
    Positive magnitude = prograde (raises opposite side of orbit).
    Negative magnitude = retrograde (lowers opposite side).
    """
    v_vec = state[3:]
    v_hat = v_vec / np.linalg.norm(v_vec)
    return apply_burn(state, dv_magnitude * v_hat)


# Demo: apply a 0.5 km/s prograde burn from ISS orbit
s_iss = circular_orbit_state(408)
r_before = np.linalg.norm(s_iss[:3])
eps_before = specific_energy(s_iss[:3], s_iss[3:], MU_EARTH)

s_after = tangential_dv(s_iss, 0.5)
eps_after = specific_energy(s_after[:3], s_after[3:], MU_EARTH)
a_after  = -MU_EARTH/(2*eps_after)
ra_after = 2*a_after - r_before  # apoapsis rises, periapsis stays

print("Demo: +0.5 km/s prograde from LEO 408 km")
print(f"  eps_before = {eps_before:.4f} km^2/s^2")
print(f"  eps_after  = {eps_after:.4f} km^2/s^2")
print(f"  New a      = {a_after:.1f} km")
print(f"  New ra     = {ra_after - R_EARTH:.1f} km altitude  (periapsis still at 408 km)")
print(f"  Note: burn was AT periapsis -> only apoapsis changed (as expected)")

# %% [markdown]
# ---
# ## 4. Hohmann Transfer
#
# > **[BMT 2nd ed.] Section 6.3** — "The Hohmann Transfer"
#
# Minimum-energy two-impulse transfer between coplanar circular orbits.
# Transfer ellipse has a_t = (r1+r2)/2, tangent to both circles.
#
# Equations 6.3-1 through 6.3-3:
#   dv1 = sqrt(mu/r1) * (sqrt(2*r2/(r1+r2)) - 1)   at periapsis (departure)
#   dv2 = sqrt(mu/r2) * (1 - sqrt(2*r1/(r1+r2)))   at apoapsis  (arrival)
#   dt  = pi * sqrt(a_t^3 / mu)                     half period of transfer ellipse
#
# Optimal (minimum total dv) when r2/r1 <= 11.94.
# [BMT 2nd ed.] Section 6.3 discusses the optimality proof.

# %% Hohmann
def hohmann_dv(r1, r2, mu=MU_EARTH):
    """
    Delta-v for Hohmann transfer between circular orbits at r1 and r2.
    [BMT 2nd ed.] Section 6.3, Equations 6.3-1 to 6.3-3

    Parameters
    ----------
    r1, r2 : float  -- orbit radii in km (r2 can be < or > r1)
    mu     : float  -- gravitational parameter km^3/s^2

    Returns dict: dv1, dv2, dv_total, a_t, T_transfer, dt_transfer
    """
    a_t   = (r1 + r2) / 2.0
    v_c1  = np.sqrt(mu / r1)
    v_c2  = np.sqrt(mu / r2)
    v_tp  = vis_viva(r1, a_t, mu)   # speed at periapsis of transfer ellipse
    v_ta  = vis_viva(r2, a_t, mu)   # speed at apoapsis

    dv1 = abs(v_tp - v_c1)
    dv2 = abs(v_c2 - v_ta)

    T_t  = orbital_period(a_t, mu)
    dt_t = T_t / 2.0

    return {
        "dv1":         dv1,
        "dv2":         dv2,
        "dv_total":    dv1 + dv2,
        "a_t":         a_t,
        "T_transfer":  T_t,
        "dt_transfer": dt_t,
        "r1":          r1,
        "r2":          r2,
    }


# ── LEO -> GEO (the canonical mission design example) ─────────────────────────
r_leo = R_EARTH + 408.0
r_geo = R_EARTH + 35_786.0
h = hohmann_dv(r_leo, r_geo)

print("Hohmann Transfer: LEO (408 km) -> GEO (35,786 km)")
print(f"  r1 = {r_leo:.1f} km   r2 = {r_geo:.1f} km   r2/r1 = {r_geo/r_leo:.2f}")
print(f"  dv1        = {h['dv1']:.4f} km/s  (prograde burn at LEO periapsis)")
print(f"  dv2        = {h['dv2']:.4f} km/s  (prograde burn at GEO apoapsis)")
print(f"  dv_total   = {h['dv_total']:.4f} km/s")
print(f"  a_transfer = {h['a_t']:.1f} km")
print(f"  dt_transit = {h['dt_transfer']/3600:.3f} hours  ({h['dt_transfer']/60:.1f} min)")

# %% [markdown]
# ---
# ## 5. Simulate and Visualize the Full Hohmann Sequence
#
# We simulate three trajectory segments in sequence:
#   1. One orbit in LEO (reference)
#   2. Transfer ellipse (half period, from LEO to GEO altitude)
#   3. One orbit in GEO after circularization burn
#
# The burn locations are shown as colored markers.
# The 2D plot in the equatorial plane (XY) is clearest for in-plane maneuvers.

# %% HohmannSimulation
dt = 10.0   # integration step [s]

# ── Phase A: one orbit in LEO ─────────────────────────────────────────────────
s_leo   = circular_orbit_state(408, 0.0)
T_leo   = orbital_period(r_leo, MU_EARTH)
tr_leo  = propagate(two_body_eom, s_leo, (0., T_leo), dt, MU_EARTH)

# ── Burn 1 at LEO periapsis (satellite starts on +x axis, velocity in +y) ─────
v_leo_c  = np.linalg.norm(s_leo[3:])            # circular speed at LEO
v_tp     = vis_viva(r_leo, h["a_t"], MU_EARTH)  # periapsis speed of transfer ellipse
dv1_mag  = v_tp - v_leo_c                        # prograde (positive)
s_burn1  = tangential_dv(s_leo, dv1_mag)

# ── Phase B: transfer ellipse (half period) ───────────────────────────────────
dt_t    = h["dt_transfer"]
tr_xfer = propagate(two_body_eom, s_burn1, (0., dt_t), dt, MU_EARTH)

# Check apoapsis altitude
r_apo_actual = np.linalg.norm(tr_xfer["r"][-1])
print(f"Transfer ellipse validation:")
print(f"  Apoapsis reached : {r_apo_actual:.1f} km  (expect {r_geo:.1f} km)")
print(f"  Altitude error   : {abs(r_apo_actual - r_geo):.1f} km")

# ── Burn 2: circularize at GEO apoapsis ───────────────────────────────────────
# dv2 is tangential: add speed to circularize
v_at_apo = tr_xfer["v"][-1]
r_at_apo = tr_xfer["r"][-1]
v_apo_actual = np.linalg.norm(v_at_apo)
v_geo_circ   = np.sqrt(MU_EARTH / np.linalg.norm(r_at_apo))
dv2_mag  = v_geo_circ - v_apo_actual    # positive -> prograde
v_hat    = v_at_apo / v_apo_actual
s_burn2  = np.concatenate([r_at_apo, v_at_apo + dv2_mag * v_hat])

# ── Phase C: one orbit in GEO ─────────────────────────────────────────────────
T_geo  = orbital_period(r_geo, MU_EARTH)
tr_geo = propagate(two_body_eom, s_burn2, (0., T_geo), 60.0, MU_EARTH)

# Validate GEO is circular
r_geo_arr = np.linalg.norm(tr_geo["r"], axis=1)
print(f"\nGEO orbit validation:")
print(f"  r_mean = {r_geo_arr.mean():.1f} km  (expect {r_geo:.1f} km)")
print(f"  r_std  = {r_geo_arr.std():.2f} km  (expect ~0 for circular)")
print(f"\nActual dv1 = {dv1_mag:.4f} km/s")
print(f"Actual dv2 = {dv2_mag:.4f} km/s")
print(f"Total dv   = {dv1_mag+dv2_mag:.4f} km/s  (theory: {h['dv_total']:.4f} km/s)")

# %% [markdown]
# ---
# ## 6. Hohmann Transfer Plot

# %% HohmannPlot
fig_h, ax_h = plt.subplots(1, 1, figsize=(10, 10), facecolor="#0a0a1a")
ax_h.set_facecolor("#0a0a1a")
ax_h.set_aspect("equal")

# Earth
earth_circle = plt.Circle((0, 0), R_EARTH, color="deepskyblue", alpha=0.35, zorder=1)
ax_h.add_patch(earth_circle)
ax_h.plot(*np.array([[R_EARTH*np.cos(t), R_EARTH*np.sin(t)]
           for t in np.linspace(0,2*np.pi,200)]).T, color="royalblue", lw=0.5, alpha=0.5)

# Orbits
ax_h.plot(tr_leo["r"][:,0],  tr_leo["r"][:,1],  color="#00bfff", lw=1.2, label="LEO",  alpha=0.75, zorder=2)
ax_h.plot(tr_xfer["r"][:,0], tr_xfer["r"][:,1], color="#ffff00", lw=2.0, label="Transfer ellipse", alpha=0.95, zorder=3)
ax_h.plot(tr_geo["r"][:,0],  tr_geo["r"][:,1],  color="#ff6b35", lw=1.2, label="GEO",  alpha=0.75, zorder=2)

# Burn markers
ax_h.scatter(*tr_xfer["r"][0,  :2], color="lime",   s=120, zorder=6, label=f"Burn 1: +{dv1_mag:.3f} km/s")
ax_h.scatter(*tr_xfer["r"][-1, :2], color="red",    s=120, zorder=6, label=f"Burn 2: +{dv2_mag:.3f} km/s")
ax_h.scatter(0, 0,                  color="white",  s=80,  zorder=5, marker="*")

# Arrow showing burn direction at Burn 1
_b1 = tr_xfer["r"][0, :2]
_v1_hat = tr_xfer["v"][0, :2] / np.linalg.norm(tr_xfer["v"][0, :2])
ax_h.annotate("", xy=_b1 + 3000*_v1_hat, xytext=_b1,
               arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))

_b2 = tr_xfer["r"][-1, :2]
_v2_hat = tr_xfer["v"][-1, :2] / np.linalg.norm(tr_xfer["v"][-1, :2])
ax_h.annotate("", xy=_b2 + 3000*_v2_hat, xytext=_b2,
               arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

# Labels
ax_h.text(_b1[0]*0.82, _b1[1]*0.82, "Burn 1\n(prograde)", color="lime",   fontsize=9, ha="center")
ax_h.text(_b2[0]*1.08, _b2[1]*1.05, "Burn 2\n(prograde)", color="red",    fontsize=9, ha="center")
ax_h.text(0, -R_EARTH*0.5, "Earth", color="white", fontsize=11, ha="center")

_lim = 47_000
ax_h.set_xlim(-_lim, _lim); ax_h.set_ylim(-_lim, _lim)
ax_h.set_xlabel("X (km)", color="white"); ax_h.set_ylabel("Y (km)", color="white")
ax_h.tick_params(colors="gray")
ax_h.set_title(f"Hohmann Transfer: LEO 408 km -> GEO 35786 km\n"
               f"Total dv = {h['dv_total']:.4f} km/s  |  Transit time = {h['dt_transfer']/3600:.2f} h",
               color="white", fontsize=12)
ax_h.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10, loc="upper right")

plt.tight_layout()
plt.savefig("p2_hohmann.png", dpi=140, facecolor="#0a0a1a", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7. Bi-Elliptic Transfer
#
# > **[BMT 2nd ed.] Section 6.4** — "Bi-elliptic Transfers"
#
# Three burns, intermediate radius r_b > max(r1,r2).
# More efficient than Hohmann when r2/r1 > 11.94.
# Trade-off: always takes longer (up to 3x more time).
#
# Delta-v equations from [BMT 2nd ed.] Section 6.4:
#   dv1: LEO -> first transfer ellipse (to r_b)
#   dv2: at r_b, switch to second transfer ellipse (to r2)
#   dv3: circularize at r2

# %% BiElliptic
def bielliptic_dv(r1, r2, r_b, mu=MU_EARTH):
    """
    Delta-v for bi-elliptic transfer.
    [BMT 2nd ed.] Section 6.4

    r_b must be > max(r1, r2).
    Returns dict: dv1, dv2, dv3, dv_total, T_total
    """
    v_c1  = np.sqrt(mu/r1)
    v_c2  = np.sqrt(mu/r2)
    a1    = (r1 + r_b) / 2.0
    a2    = (r2 + r_b) / 2.0
    vA    = vis_viva(r1, a1, mu)
    vB1   = vis_viva(r_b, a1, mu)
    vB2   = vis_viva(r_b, a2, mu)
    vC    = vis_viva(r2, a2, mu)
    dv1   = abs(vA  - v_c1)
    dv2   = abs(vB2 - vB1)
    dv3   = abs(v_c2 - vC)
    T     = orbital_period(a1, mu)/2 + orbital_period(a2, mu)/2
    return {"dv1":dv1,"dv2":dv2,"dv3":dv3,"dv_total":dv1+dv2+dv3,"T_total":T,"r_b":r_b}


# Compare Hohmann vs bi-elliptic across a range of r2/r1 ratios
# The crossover at r2/r1 = 11.94 should be clearly visible
print("Hohmann vs Bi-elliptic comparison")
print(f"  r1 fixed = {R_EARTH+200:.1f} km  (LEO 200 km)\n")
print(f"  {'r2/r1':>8}  {'Hohmann dv':>12}  {'Bi-el dv':>12}  {'Saving':>10}  Winner")
print("  " + "-"*60)

r1 = R_EARTH + 200.0
for ratio in [3, 6, 8, 11, 11.94, 12, 15, 20, 50]:
    r2 = r1 * ratio
    r_b = r1 * ratio * 3.5  # intermediate radius = 3.5 * r2
    h_r  = hohmann_dv(r1, r2)
    be_r = bielliptic_dv(r1, r2, r_b)
    saving = h_r["dv_total"] - be_r["dv_total"]
    winner = "Bi-el" if saving > 0 else "Hohmann"
    print(f"  {ratio:>8.2f}  {h_r['dv_total']:>12.4f}  {be_r['dv_total']:>12.4f}  {saving:>+10.4f}  {winner}")

print("\n  The crossover at r2/r1 ~ 11.94 is exact for r_b -> infinity.")
print("  For finite r_b the crossover depends on the chosen intermediate radius.")

# %% [markdown]
# ---
# ## 8. Plane Changes
#
# > **[BMT 2nd ed.] Section 6.5, p. 263** — "Out-of-Plane Orbit Changes"
#
# Pure plane change: same altitude, different inclination.
#   dv = 2 * v_orb * sin(di/2)
#
# This is expensive. A 28.5° plane change at LEO costs ~3.77 km/s —
# MORE than a full Hohmann LEO->GEO transfer (3.85 km/s).
# Most launches choose low-inclination launch sites to avoid this cost.
#
# Combined maneuver (altitude change + plane change done simultaneously):
#   dv_combined = sqrt(v1^2 + v2^2 - 2*v1*v2*cos(di))
# This is always cheaper than sequential burns.

# %% PlaneChange
def plane_change_dv(v_orb, delta_i_deg):
    """
    Pure plane change delta-v.
    [BMT 2nd ed.] Section 6.5, Equation 6.5-1
    dv = 2 * v_orb * sin(di/2)
    """
    di = np.radians(delta_i_deg)
    return 2.0 * v_orb * np.sin(di / 2.0)


def combined_dv(v1, v2, delta_i_deg):
    """
    Combined altitude change + plane change (simultaneous).
    [BMT 2nd ed.] Section 6.5 — Combined maneuver
    Always cheaper than doing them sequentially.
    dv = sqrt(v1^2 + v2^2 - 2*v1*v2*cos(di))
    """
    di = np.radians(delta_i_deg)
    return np.sqrt(v1**2 + v2**2 - 2*v1*v2*np.cos(di))


# ── Plane change cost table ───────────────────────────────────────────────────
v_leo_circ = np.sqrt(MU_EARTH / (R_EARTH + 408))
print(f"Plane change cost at LEO (v_c = {v_leo_circ:.4f} km/s)\n")
print(f"  {'Delta-i (deg)':>14}  {'dv (km/s)':>12}  {'% of v_c':>10}")
print("  " + "-"*40)
for di in [5, 10, 15, 28.5, 45, 60, 90, 135, 180]:
    dv = plane_change_dv(v_leo_circ, di)
    print(f"  {di:>14.1f}  {dv:>12.4f}  {dv/v_leo_circ*100:>9.1f}%")

print(f"\nFor reference: Hohmann LEO->GEO total dv = {h['dv_total']:.4f} km/s")
print(f"A 28.5° plane change at LEO costs {plane_change_dv(v_leo_circ,28.5):.4f} km/s")
print(f"=> plane change MORE expensive than full LEO->GEO Hohmann!")

# ── Sequential vs combined maneuver comparison ────────────────────────────────
print("\nSequential vs Combined: LEO->GEO with 28.5° plane change")
di = 28.5
# Option A: Hohmann first, then plane change at GEO (cheapest point for plane change)
v_geo_circ = np.sqrt(MU_EARTH / r_geo)
dv_seq = h["dv_total"] + plane_change_dv(v_geo_circ, di)
# Option B: Combined plane change with burn 2 at GEO
v_tp_geo = vis_viva(r_geo, h["a_t"], MU_EARTH)  # apoapsis speed of transfer ellipse
dv_comb  = h["dv1"] + combined_dv(v_tp_geo, v_geo_circ, di)
print(f"  Sequential (Hohmann + plane change at GEO) : {dv_seq:.4f} km/s")
print(f"  Combined (fold plane change into burn 2)   : {dv_comb:.4f} km/s")
print(f"  Saving from combined maneuver              : {dv_seq - dv_comb:.4f} km/s")

# %% [markdown]
# ---
# ## 9. Plane Change Visualization — 3D

# %% PlaneChangePlot3D
# Simulate two coplanar orbits, then apply a plane change to show the geometry
s_inc0  = circular_orbit_state(408, inc_deg=0.0)
s_inc28 = circular_orbit_state(408, inc_deg=28.5)

T_leo2  = orbital_period(R_EARTH+408, MU_EARTH)
dt      = 10.0

tr_i0   = propagate(two_body_eom, s_inc0,  (0., T_leo2), dt, MU_EARTH)
tr_i28  = propagate(two_body_eom, s_inc28, (0., T_leo2), dt, MU_EARTH)

# Also 51.6 (ISS inclination)
s_inc51 = circular_orbit_state(408, inc_deg=51.6)
tr_i51  = propagate(two_body_eom, s_inc51, (0., T_leo2), dt, MU_EARTH)

def _draw_earth_3d(ax, n=25, alpha=0.2):
    u=np.linspace(0,2*np.pi,n); v=np.linspace(0,np.pi,n)
    ax.plot_surface(R_EARTH*np.outer(np.cos(u),np.sin(v)),
                    R_EARTH*np.outer(np.sin(u),np.sin(v)),
                    R_EARTH*np.outer(np.ones(n),np.cos(v)),
                    color='deepskyblue',alpha=alpha,linewidth=0)
    ax.plot_wireframe(R_EARTH*np.outer(np.cos(u),np.sin(v)),
                      R_EARTH*np.outer(np.sin(u),np.sin(v)),
                      R_EARTH*np.outer(np.ones(n),np.cos(v)),
                      color='royalblue',lw=0.2,alpha=0.3)

fig_pc = plt.figure(figsize=(11, 8), facecolor='#0a0a1a')
ax_pc  = fig_pc.add_subplot(111, projection='3d')
ax_pc.set_facecolor('#0a0a1a')
_draw_earth_3d(ax_pc)

ax_pc.plot(tr_i0["r"][:,0],  tr_i0["r"][:,1],  tr_i0["r"][:,2],  color='#00bfff', lw=1.4, label='i=0° (equatorial)')
ax_pc.plot(tr_i28["r"][:,0], tr_i28["r"][:,1], tr_i28["r"][:,2], color='#ff6b35', lw=1.4, label='i=28.5° (KSC latitude)')
ax_pc.plot(tr_i51["r"][:,0], tr_i51["r"][:,1], tr_i51["r"][:,2], color='#7fff00', lw=1.4, label='i=51.6° (ISS)')

# Show equatorial plane
_lim = 9500
_g   = np.linspace(-_lim, _lim, 4)
_GX, _GY = np.meshgrid(_g, _g)
ax_pc.plot_surface(_GX, _GY, np.zeros_like(_GX), alpha=0.07, color='white')

ax_pc.set_xlim(-_lim, _lim); ax_pc.set_ylim(-_lim, _lim); ax_pc.set_zlim(-_lim, _lim)
ax_pc.set_box_aspect([1,1,1])
ax_pc.set_xlabel("X (km)", color='white', fontsize=9)
ax_pc.set_ylabel("Y (km)", color='white', fontsize=9)
ax_pc.set_zlabel("Z (km)", color='white', fontsize=9)
ax_pc.set_title("LEO at Three Inclinations — ECI Frame\n"
                "Plane change is expensive: 28.5° costs ~3.77 km/s at LEO",
                color='white', fontsize=11, pad=10)
ax_pc.tick_params(colors='gray', labelsize=7)
ax_pc.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig("p2_plane_change.png", dpi=140, facecolor='#0a0a1a', bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 10. The Rocket Equation — Propellant Mass
#
# > **[S&B] Section 4.2, pp. 114-118** — "Rocket Flight Performance"
# > **[S&B] Section 2.1, p. 27** — Definition of specific impulse Isp
# > **[BMT 2nd ed.] Section 6.2, p. 249** — Delta-v budget and propellant
#
# Tsiolkovsky Rocket Equation:
#   dv = Isp * g0 * ln(m0/mf)
#   => m0/mf = exp(dv / (Isp * g0))
#   => propellant fraction = 1 - exp(-dv / (Isp * g0))
#
# Isp (specific impulse) is the thrust per unit weight flow rate of propellant [s].
# Higher Isp => more efficient engine => less propellant for same dv.
# [S&B] Section 2.1, p. 27: Isp = F / (m_dot * g0)

# %% RocketEquation
def rocket_equation(dv, isp, g0=G0):
    """
    Tsiolkovsky Rocket Equation.
    [S&B] Section 4.2, p. 114, Equation 4-6
    [BMT 2nd ed.] Section 6.2, p. 249

    dv  : required delta-v [km/s]
    isp : specific impulse [s]  -- [S&B] Section 2.1, p. 27
    g0  : standard gravity [km/s^2]  (9.80665e-3 km/s^2)

    Returns
    -------
    dict: ve (exhaust velocity), mass_ratio, prop_fraction
    """
    ve            = isp * g0                    # effective exhaust velocity [km/s]
    mass_ratio    = np.exp(dv / ve)             # m0/mf
    prop_fraction = 1.0 - np.exp(-dv / ve)     # mp/m0

    return {
        "ve":            ve,
        "mass_ratio":    mass_ratio,
        "prop_fraction": prop_fraction,
        "isp":           isp,
        "dv":            dv,
    }


def propellant_mass(dv, isp, m_payload, m_dry_struct_frac=0.1, g0=G0):
    """
    Required propellant mass for a given dv, payload, and structural fraction.

    m_payload       : payload mass after burn [kg]
    m_dry_struct_frac : fraction of initial mass that is structure (tanks, engine) [dimensionless]
    Returns: m_propellant, m_initial
    """
    ve = isp * g0
    # m0/mf = exp(dv/ve), mf = m_payload + m_struct
    # m_struct = m_dry_struct_frac * m0  =>  solvable for m0
    mr  = np.exp(dv / ve)
    # m0 = mr * (m_payload + m_struct) = mr * (m_payload + frac*m0)
    # m0 * (1 - mr*frac) = mr * m_payload
    m0  = mr * m_payload / (1.0 - mr * m_dry_struct_frac)
    mp  = m0 * (1.0 - 1.0/mr - m_dry_struct_frac)
    return {"m_propellant": mp, "m_initial": m0, "m_payload": m_payload}


# ── Mission budget table for LEO->GEO ────────────────────────────────────────
dv_mission = h["dv_total"]
print("Rocket Equation — LEO -> GEO mission")
print(f"  Required dv = {dv_mission:.4f} km/s\n")

# [S&B] Chapter 7 and Table 5-5 for typical Isp values
engines = [
    ("Cold gas (N2)",           65,  "[S&B] Typical cold gas thruster"),
    ("Solid (SRB)",            275,  "[S&B] Table 5-5, solid propellant"),
    ("Biprop RP-1/LOX",        311,  "[S&B] Table 5-5, Merlin-like"),
    ("Biprop LH2/LOX",         450,  "[S&B] Table 5-5, SSME/Vulcain-like"),
    ("Ion (Hall, Xenon)",     1800,  "[S&B] Chapter 19, Hall thruster"),
]

print(f"  {'Engine':<28}  {'Isp(s)':>7}  {'mr=m0/mf':>10}  {'Prop frac':>10}  Ref")
print("  " + "-"*78)
for name, isp, ref in engines:
    rk = rocket_equation(dv_mission, isp)
    print(f"  {name:<28}  {isp:>7}  {rk['mass_ratio']:>10.3f}  {rk['prop_fraction']:>9.1%}  {ref}")

# %% [markdown]
# ---
# ## 11. Delta-v Budget Visualization

# %% DvBudgetPlot
fig_dv, axes_dv = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0d0d1e')
fig_dv.suptitle("Delta-v Budget — LEO to GEO Mission", color='white', fontsize=13)

# Left: propellant fraction vs Isp
ax1 = axes_dv[0]
ax1.set_facecolor('#111122')
isps = np.linspace(50, 5000, 500)
for dv_label, dv_val, clr in [
        ("LEO->GEO total: 3.85 km/s", h["dv_total"], "#ff6b35"),
        ("Plane change 28.5° at LEO: 3.77 km/s", plane_change_dv(v_leo_circ, 28.5), "#00bfff"),
        ("Bi-elliptic r2/r1=20: 3.1 km/s", 3.1, "#7fff00"),
]:
    pf = 1 - np.exp(-dv_val / (isps * G0))
    ax1.plot(isps, pf * 100, color=clr, lw=2, label=f"{dv_label}")

# Mark representative engines (from [S&B])
for name, isp, clr in [("Solid\n[S&B]", 275, "gray"),
                        ("RP-1/LOX\n[S&B]", 311, "silver"),
                        ("LH2/LOX\n[S&B]", 450, "white"),
                        ("Ion\n[S&B]", 1800, "yellow")]:
    ax1.axvline(isp, color=clr, linewidth=0.8, alpha=0.5, linestyle=":")
    ax1.text(isp+20, 92, name, color=clr, fontsize=7, va="top")

ax1.set_xlabel("Specific Impulse Isp (s)  [S&B §2.1]", color="gray")
ax1.set_ylabel("Propellant fraction mp/m0 (%)", color="gray")
ax1.set_title("Propellant Fraction vs Isp\n(Tsiolkovsky equation  [S&B §4.2])", color="white", fontsize=10)
ax1.tick_params(colors="gray")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
ax1.set_xlim(50, 2000); ax1.set_ylim(0, 100)

# Right: Hohmann dv breakdown and comparison
ax2 = axes_dv[1]
ax2.set_facecolor('#111122')

maneuvers = {
    "Hohmann\ndv1": h["dv1"],
    "Hohmann\ndv2": h["dv2"],
    "Plane chg\n28.5° LEO": plane_change_dv(v_leo_circ, 28.5),
    "Plane chg\n28.5° GEO": plane_change_dv(v_geo_circ, 28.5),
}
bars = ax2.bar(list(maneuvers.keys()), list(maneuvers.values()),
               color=["lime", "lime", "#ff6b35", "#00bfff"], alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, maneuvers.values()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
             f"{val:.3f}", ha="center", color="white", fontsize=9)

ax2.set_ylabel("Delta-v (km/s)", color="gray")
ax2.set_title("Maneuver Comparison [BMT 2nd ed. §6.3-6.5]", color="white", fontsize=10)
ax2.tick_params(colors="gray")
ax2.set_ylim(0, max(maneuvers.values()) * 1.2)

# Hohmann total annotation
ax2.axhline(h["dv_total"], color="lime", lw=1.2, linestyle="--", alpha=0.6)
ax2.text(2.5, h["dv_total"]+0.05, f"Hohmann total = {h['dv_total']:.3f} km/s",
         color="lime", fontsize=9, ha="center")

plt.tight_layout()
plt.savefig("p2_dv_budget.png", dpi=140, facecolor='#0d0d1e', bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 12. Complete Mission Simulation — LEO to GEO with Inclination Change
#
# Realistic mission from KSC (28.5° latitude) to GEO.
# Strategy: fold the inclination change into burn 2 at GEO apoapsis
# (where the spacecraft is moving slowest => plane change is cheapest).

# %% FullMission
print("Full Mission: KSC launch (28.5 deg) -> GEO")
print("Strategy: Hohmann + fold plane change into GEO circularization burn\n")

# Initial orbit: LEO at 28.5 deg inclination
s_ksc = circular_orbit_state(408, inc_deg=28.5)

# Burn 1: same as before (tangential at LEO, does not change inclination efficiently)
# Best practice: do plane change at highest orbit (GEO apoapsis), not at LEO
v_ksc = np.linalg.norm(s_ksc[3:])
v_tp2 = vis_viva(r_leo, h["a_t"], MU_EARTH)
s_b1  = tangential_dv(s_ksc, v_tp2 - v_ksc)

# Propagate transfer ellipse
tr_xfer2 = propagate(two_body_eom, s_b1, (0., h["dt_transfer"]), dt, MU_EARTH)

# Burn 2 at apoapsis: combined circularization + 28.5° plane change
v_apo2    = tr_xfer2["v"][-1]
r_apo2    = tr_xfer2["r"][-1]
v_apo2_m  = np.linalg.norm(v_apo2)
v_geo_m   = np.sqrt(MU_EARTH / np.linalg.norm(r_apo2))
dv2_comb  = combined_dv(v_apo2_m, v_geo_m, 28.5)

# Direction of dv2: from current velocity direction to GEO velocity direction (equatorial)
# GEO velocity direction at this point: perpendicular to r, in equatorial plane
r_hat2    = r_apo2 / np.linalg.norm(r_apo2)
# Equatorial velocity direction (perpendicular to r in XY plane)
z_hat     = np.array([0., 0., 1.])
v_geo_dir = np.cross(z_hat, r_hat2)
v_geo_dir /= np.linalg.norm(v_geo_dir)
s_geo2    = np.concatenate([r_apo2, v_geo_m * v_geo_dir])

T_geo2    = orbital_period(np.linalg.norm(r_apo2), MU_EARTH)
tr_geo2   = propagate(two_body_eom, s_geo2, (0., T_geo2), 60., MU_EARTH)

# Check inclination of final GEO orbit
h_vec_geo2 = np.cross(tr_geo2["r"][0], tr_geo2["v"][0])
inc_geo2   = np.degrees(np.arccos(np.clip(h_vec_geo2[2]/np.linalg.norm(h_vec_geo2), -1, 1)))
r_geo2_arr = np.linalg.norm(tr_geo2["r"], axis=1)

print(f"  Burn 1 dv              = {v_tp2-v_ksc:.4f} km/s")
print(f"  Burn 2 dv (combined)   = {dv2_comb:.4f} km/s")
print(f"  Total mission dv       = {(v_tp2-v_ksc)+dv2_comb:.4f} km/s")
print(f"\n  Final orbit:")
print(f"    Inclination = {inc_geo2:.2f} deg  (expect ~0 for GEO)")
print(f"    r_mean      = {r_geo2_arr.mean():.1f} km  (expect {r_geo:.1f} km)")
print(f"    r_std       = {r_geo2_arr.std():.1f} km  (expect ~0)")

# %% [markdown]
# ---
# ## 13. Summary Reference Table
#
# | Function | Physics | Reference |
# |----------|---------|-----------|
# | apply_burn | v^+ = v^- + dv | [BMT 2nd ed.] Section 6.1 |
# | tangential_dv | dv along velocity direction | [BMT 2nd ed.] Section 6.2, p. 248 |
# | hohmann_dv | dv1, dv2, transfer geometry | [BMT 2nd ed.] Section 6.3 |
# | bielliptic_dv | 3-burn transfer | [BMT 2nd ed.] Section 6.4 |
# | plane_change_dv | dv = 2*v*sin(di/2) | [BMT 2nd ed.] Section 6.5, p. 263 |
# | combined_dv | sqrt(v1^2+v2^2-2*v1*v2*cos(di)) | [BMT 2nd ed.] Section 6.5 |
# | rocket_equation | Tsiolkovsky dv = Isp*g0*ln(m0/mf) | [S&B] Section 4.2, p. 114 |
#
# **Key mission numbers:**
#
# | Maneuver | Delta-v |
# |----------|---------|
# | Hohmann LEO (408 km) -> GEO | 3.8535 km/s |
# | Plane change 28.5° at LEO | 3.7750 km/s |
# | Plane change 28.5° at GEO | 1.1178 km/s |
# | LEO -> GEO + 28.5° combined | ~4.27 km/s |
#
# **Rule:** Do plane changes at the highest orbit (lowest velocity) — always cheaper.
#
# **Phase 3 -> phase3_perturbations.py**
# Topics: J2 oblateness, atmospheric drag, RAAN precession, orbital decay.
# References: [BMT 2nd ed.] Sections 9.3-9.7
