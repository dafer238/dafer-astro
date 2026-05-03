# %% [markdown]
# # 🚀 Phase 3 — Orbital Perturbations
# ## J₂ Oblateness, Atmospheric Drag, Sun-Synchronous Orbits
#
# **Format:** `# %%` (py:percent) — Zed/VS Code Jupyter, Spyder, or:
#   `marimo convert phase3_perturbations.py -o phase3_marimo.py`
#
# **Read alongside:** `theory_03_perturbations.md`
#
# **References:**
# - **[BMT]** Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020
# - **[S&B]** Sutton & Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# %% [markdown]
# ---
# ## 2. Constants and Core Functions
#
# Self-contained module — no dependency on Phase 1 or 2 files.

# %% CoreFunctions
# ── Physical constants ────────────────────────────────────────────────────────
MU_EARTH = 398_600.4418      # km^3/s^2
R_EARTH  = 6_371.0           # km
J2       = 1.082_627_0e-3    # dimensionless  [BMT 2nd ed.] Appendix / Table 9.3-1
G0       = 9.80665e-3        # km/s^2  standard gravity

# ── Integrator with re-entry guard ───────────────────────────────────────────
def rk4_step(f, t, y, h, *args):
    k1=f(t,y,*args); k2=f(t+h/2,y+(h/2)*k1,*args)
    k3=f(t+h/2,y+(h/2)*k2,*args); k4=f(t+h,y+h*k3,*args)
    return y+(h/6)*(k1+2*k2+2*k3+k4)

def propagate(eom, state0, t_span, dt, *args, alt_min_km=80.0):
    """
    Fixed-step RK4 propagator with re-entry guard.
    Stops if altitude drops below alt_min_km (default 80 km).
    Returns dict: t, r, v, reentered (bool).
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / dt)
    t_list = [t0]
    s_list = [state0.copy()]
    for i in range(n_steps):
        s_next = rk4_step(eom, t_list[-1], s_list[-1], dt, *args)
        t_list.append(t_list[-1] + dt)
        s_list.append(s_next)
        if np.linalg.norm(s_next[:3]) < R_EARTH + alt_min_km:
            break
    arr = np.array(s_list)
    reentered = np.linalg.norm(arr[-1, :3]) < R_EARTH + alt_min_km + 10
    return {"t": np.array(t_list), "r": arr[:, :3],
            "v": arr[:, 3:], "reentered": reentered}

def orbital_period(a, mu=MU_EARTH):
    return 2.0 * np.pi * np.sqrt(a**3 / mu)

def circ_state(alt_km, inc_deg=0.0, mu=MU_EARTH):
    r = R_EARTH + alt_km; vc = np.sqrt(mu/r); i = np.radians(inc_deg)
    return np.array([r, 0., 0., 0., vc*np.cos(i), vc*np.sin(i)])

def coe_to_state(a, e, i_d, O_d, w_d, th_d, mu=MU_EARTH):
    i,O,w,th=[np.radians(x) for x in (i_d,O_d,w_d,th_d)]
    p=a*(1-e**2); rm=p/(1+e*np.cos(th))
    rp=rm*np.array([np.cos(th),np.sin(th),0.])
    vp=np.sqrt(mu/p)*np.array([-np.sin(th),e+np.cos(th),0.])
    cO,sO=np.cos(O),np.sin(O); ci,si=np.cos(i),np.sin(i); cw,sw=np.cos(w),np.sin(w)
    Q=np.array([[cO*cw-sO*sw*ci,-cO*sw-sO*cw*ci,sO*si],
                [sO*cw+cO*sw*ci,-sO*sw+cO*cw*ci,-cO*si],[sw*si,cw*si,ci]])
    return np.concatenate([Q@rp, Q@vp])

def state_to_coe(r_vec, v_vec, mu=MU_EARTH):
    r=np.linalg.norm(r_vec); v=np.linalg.norm(v_vec)
    h_v=np.cross(r_vec,v_vec); h=np.linalg.norm(h_v)
    e_v=(1/mu)*((v**2-mu/r)*r_vec-np.dot(r_vec,v_vec)*v_vec); e=np.linalg.norm(e_v)
    i_d=np.degrees(np.arccos(np.clip(h_v[2]/h,-1,1)))
    N_v=np.cross([0.,0.,1.],h_v); N=np.linalg.norm(N_v)
    Om=0.
    if N>1e-10:
        Om=np.degrees(np.arccos(np.clip(N_v[0]/N,-1,1)))
        if N_v[1]<0: Om=360-Om
    om=0.
    if N>1e-10 and e>1e-10:
        om=np.degrees(np.arccos(np.clip(np.dot(N_v,e_v)/(N*e),-1,1)))
        if e_v[2]<0: om=360-om
    th=0.
    if e>1e-10:
        th=np.degrees(np.arccos(np.clip(np.dot(e_v,r_vec)/(e*r),-1,1)))
        if np.dot(r_vec,v_vec)<0: th=360-th
    a=-mu/(2*(0.5*v**2-mu/r))
    return {"a":a,"e":e,"i_deg":i_d,"raan_deg":Om,"omega_deg":om,"theta_deg":th}

print("Core functions loaded.")

# %% [markdown]
# ---
# ## 3. The J₂ Perturbation
#
# > **[BMT 2nd ed.] Section 9.3** — "Effects of Earth's Oblateness"
#
# Earth's equatorial bulge creates a gravitational asymmetry. The dominant correction
# to the gravitational potential uses the second zonal harmonic J₂ = 1.08263 × 10⁻³.
#
# The extra Cartesian acceleration added to the EOM ([BMT 2nd ed.] Section 9.3, Eq. 9.3-1):
#
#   a_x = (3μJ₂R_E²/2r⁵) · x · (5z²/r² − 1)
#   a_y = (3μJ₂R_E²/2r⁵) · y · (5z²/r² − 1)
#   a_z = (3μJ₂R_E²/2r⁵) · z · (5z²/r² − 3)
#
# Physical size: at LEO this is ~10⁻³g — tiny but cumulative.
# After 100 orbits, RAAN has precessed by ~1.5°. After one year: ~1800°.

# %% J2Perturbation
def j2_accel(r_vec, mu=MU_EARTH, j2=J2, Re=R_EARTH):
    """
    J2 oblateness perturbing acceleration.
    [BMT 2nd ed.] Section 9.3, Equation 9.3-1

    Returns acceleration vector in km/s^2.
    Magnitude at LEO: ~1e-5 km/s^2 = ~1e-2 m/s^2 ~ 1e-3 g.
    """
    x, y, z = r_vec
    r       = np.linalg.norm(r_vec)
    fac     = (3.0 * mu * j2 * Re**2) / (2.0 * r**5)
    zr2     = (z / r)**2
    return np.array([
        fac * x * (5*zr2 - 1),
        fac * y * (5*zr2 - 1),
        fac * z * (5*zr2 - 3),
    ])


def eom_two_body(t, state, mu):
    r_vec=state[:3]; v_vec=state[3:]; r_mag=np.linalg.norm(r_vec)
    return np.concatenate([v_vec, -(mu/r_mag**3)*r_vec])

def eom_j2(t, state, mu):
    """Two-body + J2 equations of motion."""
    r_vec=state[:3]; v_vec=state[3:]; r_mag=np.linalg.norm(r_vec)
    return np.concatenate([v_vec, -(mu/r_mag**3)*r_vec + j2_accel(r_vec, mu)])


# ── Verify J2 magnitude at LEO ────────────────────────────────────────────────
_r_iss = np.array([R_EARTH + 408, 0., 0.])
_a_grav = MU_EARTH / np.linalg.norm(_r_iss)**2
_a_j2   = np.linalg.norm(j2_accel(_r_iss))

print("J2 perturbation at ISS orbit (equatorial crossing):")
print(f"  a_gravity = {_a_grav:.6f} km/s^2")
print(f"  a_J2      = {_a_j2:.6e} km/s^2")
print(f"  Ratio     = a_J2/a_grav = {_a_j2/_a_grav:.2e}  (~J2 itself = {J2:.2e})")

# %% [markdown]
# ---
# ## 4. Secular Rates — Analytical Formulas
#
# > **[BMT 2nd ed.] Section 9.3, Equations 9.3-4 to 9.3-6**
#
# After orbit-averaging, J₂ produces the following secular (linearly accumulating) rates:
#
# **RAAN precession:**
#   dΩ/dt = -(3/2) · n · J₂ · (R_E/p)² · cos i
#
# **Apsidal drift:**
#   dω/dt = (3/4) · n · J₂ · (R_E/p)² · (5cos²i − 1)
#
# **Critical inclination** (ω frozen): 5cos²i − 1 = 0  =>  i_c = 63.435° or 116.565°
# This is the Molniya inclination. [BMT 2nd ed.] Section 9.7

# %% SecularRates
def j2_raan_rate(a, e, i_deg, mu=MU_EARTH, j2=J2, Re=R_EARTH):
    """
    Secular RAAN precession rate due to J2.
    [BMT 2nd ed.] Section 9.3, Equation 9.3-4
    Returns rate in deg/day.
    """
    i = np.radians(i_deg)
    n = np.sqrt(mu / a**3)
    p = a * (1.0 - e**2)
    rate_rad_s = -(3.0/2.0) * n * j2 * (Re/p)**2 * np.cos(i)
    return np.degrees(rate_rad_s) * 86400.0   # deg/day


def j2_argp_rate(a, e, i_deg, mu=MU_EARTH, j2=J2, Re=R_EARTH):
    """
    Secular argument-of-perigee drift due to J2.
    [BMT 2nd ed.] Section 9.3, Equation 9.3-5
    Returns rate in deg/day.
    """
    i = np.radians(i_deg)
    n = np.sqrt(mu / a**3)
    p = a * (1.0 - e**2)
    rate_rad_s = (3.0/4.0) * n * j2 * (Re/p)**2 * (5.0*np.cos(i)**2 - 1.0)
    return np.degrees(rate_rad_s) * 86400.0   # deg/day


def sso_inclination(alt_km, mu=MU_EARTH, j2=J2, Re=R_EARTH):
    """
    Inclination for sun-synchronous orbit at given altitude.
    [BMT 2nd ed.] Section 9.5
    SSO requires dOmega/dt = +0.9856 deg/day (matches Sun's apparent eastward motion).
    Resulting orbits are retrograde (i > 90°).
    """
    omega_sun = np.radians(0.9856) / 86400.0   # rad/s
    a = Re + alt_km
    n = np.sqrt(mu / a**3)
    cos_i = -omega_sun / ((3.0/2.0) * n * j2 * (Re/a)**2)
    if abs(cos_i) > 1.0:
        return None
    return np.degrees(np.arccos(cos_i))


# ── Print reference table ─────────────────────────────────────────────────────
print("J2 Secular Rates — Reference Table\n")
print(f"  {'Orbit':<28}  {'RAAN (deg/day)':>16}  {'Argp (deg/day)':>16}")
print("  " + "-"*64)

orbits_ref = [
    ("ISS  LEO 408 km, 51.6°",  R_EARTH+408,  0.,  51.6),
    ("Molniya 63.4°, 600×39000", (R_EARTH+600+R_EARTH+39000)/2,
     (R_EARTH+39000 - R_EARTH-600)/(R_EARTH+39000 + R_EARTH+600), 63.4),
    ("Sun-sync 500 km, 97.4°",  R_EARTH+500,  0.,  97.4),
    ("Equatorial 400 km, 0°",   R_EARTH+400,  0.,   0.0),
    ("Polar 500 km, 90°",       R_EARTH+500,  0.,  90.0),
    ("Critical incl 500 km, 63.435°", R_EARTH+500, 0., 63.435),
]

for name, a, e, i in orbits_ref:
    dr = j2_raan_rate(a, e, i)
    dw = j2_argp_rate(a, e, i)
    print(f"  {name:<28}  {dr:>+16.4f}  {dw:>+16.4f}")

print(f"\nCritical inclination: {np.degrees(np.arccos(1/np.sqrt(5))):.4f}°  (5cos²i−1=0)")
print("\nSSO inclinations by altitude:")
for alt in [400, 500, 600, 700, 800]:
    print(f"  {alt} km -> {sso_inclination(alt):.3f}°")

# %% [markdown]
# ---
# ## 5. Numerical J₂ Propagation — Comparing Perturbed vs Keplerian
#
# We propagate the same initial state with two EOMs:
# 1. Pure two-body (Keplerian — elements constant)
# 2. J₂-perturbed (elements drift at secular rates)
#
# Then extract orbital elements at each timestep and compare.
# The RAAN drift and apsidal drift emerge naturally from the numerical integration.

# %% J2Simulation
a_iss = R_EARTH + 408.0
T_iss = orbital_period(a_iss)
n_orbits = 120   # ~5 days

s0 = circ_state(408, inc_deg=51.6)
dt = 30.0

print(f"Propagating {n_orbits} ISS orbits (~{n_orbits*T_iss/86400:.1f} days) ...")
tr_kepler = propagate(eom_two_body, s0, (0., n_orbits*T_iss), dt, MU_EARTH)
tr_j2     = propagate(eom_j2,      s0, (0., n_orbits*T_iss), dt, MU_EARTH)
print(f"  Keplerian: {len(tr_kepler['t']):,} steps")
print(f"  J2 perturbed: {len(tr_j2['t']):,} steps")

# Extract COEs at sub-sampled points
stride = max(1, len(tr_j2["t"]) // 600)
idx    = np.arange(0, len(tr_j2["t"]), stride)

coe_k = np.array([state_to_coe(tr_kepler["r"][i], tr_kepler["v"][i]) for i in idx])
coe_j = np.array([state_to_coe(tr_j2["r"][i],     tr_j2["v"][i])     for i in idx])
t_days = tr_j2["t"][idx] / 86400.0

# Unwrap RAAN to remove 360-degree wrap-arounds
raan_k = np.degrees(np.unwrap(np.radians([c["raan_deg"] for c in coe_k])))
raan_j = np.degrees(np.unwrap(np.radians([c["raan_deg"] for c in coe_j])))

# Linear fit to J2 RAAN
raan_rate_num = np.polyfit(t_days, raan_j, 1)[0]
raan_rate_ana = j2_raan_rate(a_iss, 0., 51.6)
print(f"\nRAAN precession rate:")
print(f"  Analytical : {raan_rate_ana:.4f} deg/day")
print(f"  Numerical  : {raan_rate_num:.4f} deg/day")
print(f"  Error      : {abs(raan_rate_num-raan_rate_ana):.4f} deg/day")

# %% [markdown]
# ---
# ## 6. J₂ Effects Visualization

# %% J2Plots
fig_j2, axes = plt.subplots(3, 2, figsize=(15, 11), facecolor="#0d0d1e")
fig_j2.suptitle("J₂ Perturbation Effects — ISS Orbit (408 km, 51.6°)\n"
                 "[BMT 2nd ed.] Section 9.3", color="white", fontsize=13)

_keys  = ["raan_deg", "omega_deg", "i_deg", "a", "e"]
_lbls  = ["RAAN Ω (deg)", "Arg of Perigee ω (deg)", "Inclination i (deg)",
          "Semi-major axis a (km)", "Eccentricity e"]
_clrs  = ["#00bfff", "#ff6b35", "#7fff00", "#ff00ff", "#ffff00"]

# Use unwrapped RAAN for plotting
_coe_j_plot = {
    "raan_deg":  raan_j,
    "omega_deg": np.degrees(np.unwrap(np.radians([c["omega_deg"] for c in coe_j]))),
    "i_deg":     np.array([c["i_deg"]   for c in coe_j]),
    "a":         np.array([c["a"]       for c in coe_j]),
    "e":         np.array([c["e"]       for c in coe_j]),
}
_coe_k_plot = {
    "raan_deg":  raan_k,
    "omega_deg": [c["omega_deg"] for c in coe_k],
    "i_deg":     [c["i_deg"]   for c in coe_k],
    "a":         [c["a"]       for c in coe_k],
    "e":         [c["e"]       for c in coe_k],
}

for idx_p, (key, lbl, clr) in enumerate(zip(_keys, _lbls, _clrs)):
    ax = axes[idx_p // 2, idx_p % 2]
    ax.set_facecolor("#111122")
    ax.plot(t_days, _coe_k_plot[key], color="gray",  lw=0.8, alpha=0.6, label="Keplerian")
    ax.plot(t_days, _coe_j_plot[key], color=clr,     lw=1.1,            label="J₂ perturbed")
    ax.set_ylabel(lbl, color="gray", fontsize=9)
    ax.set_xlabel("Time (days)", color="gray", fontsize=8)
    ax.tick_params(colors="gray")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    if key == "raan_deg":
        # Overlay analytical rate line
        ana_line = raan_rate_ana * t_days + raan_j[0]
        ax.plot(t_days, ana_line, color="lime", lw=0.8, linestyle="--",
                alpha=0.8, label=f"Analytical: {raan_rate_ana:.3f}°/day")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# Last panel: 3D orbit showing RAAN drift
ax_3d = axes[2, 1]
ax_3d.remove()
ax_3d = fig_j2.add_subplot(3, 2, 6, projection="3d")
ax_3d.set_facecolor("#111122")

_r_k = tr_kepler["r"]
_r_j = tr_j2["r"]
ax_3d.plot(_r_k[:,0], _r_k[:,1], _r_k[:,2], color="gray",   lw=0.5, alpha=0.4, label="Keplerian")
ax_3d.plot(_r_j[:,0], _r_j[:,1], _r_j[:,2], color="#00bfff", lw=0.5, alpha=0.5, label="J₂ perturbed")

_lim = 8200
ax_3d.set_xlim(-_lim,_lim); ax_3d.set_ylim(-_lim,_lim); ax_3d.set_zlim(-_lim,_lim)
ax_3d.set_box_aspect([1,1,1])
ax_3d.set_xlabel("X",color="white",fontsize=7); ax_3d.set_ylabel("Y",color="white",fontsize=7)
ax_3d.set_zlabel("Z",color="white",fontsize=7); ax_3d.tick_params(colors="gray",labelsize=6)
ax_3d.set_title("3D trajectory (J₂ RAAN drift visible)", color="white", fontsize=9)
ax_3d.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)

plt.tight_layout()
plt.savefig("p3_j2_effects.png", dpi=140, facecolor="#0d0d1e", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7. Molniya — The Frozen Orbit Demonstration
#
# > **[BMT 2nd ed.] Section 9.7** — "Frozen Orbits"
#
# At the critical inclination i_c = 63.435°, the apsidal drift rate is exactly zero.
# The argument of perigee stays fixed. Compare this against a non-frozen HEO.
#
# This is not a coincidence or approximation — it is an exact result from the
# J₂ secular rate formula: 5cos²(63.435°) − 1 = 0 exactly.

# %% MolniyaFrozen
rp = R_EARTH + 600.0
ra = R_EARTH + 39_000.0
a_mol = (rp + ra) / 2.0
e_mol = (ra - rp) / (ra + rp)
T_mol = orbital_period(a_mol)

# Molniya: i=63.4° (frozen), i=45° (drifting) for comparison
n_mol_orbits = 30

s_frozen  = coe_to_state(a_mol, e_mol, 63.4,  0., 270., 0.)
s_drifting= coe_to_state(a_mol, e_mol, 45.0,  0., 270., 0.)

print(f"Propagating Molniya orbits ({n_mol_orbits} revolutions, ~{n_mol_orbits*T_mol/86400:.1f} days) ...")
tr_frz = propagate(eom_j2, s_frozen,   (0., n_mol_orbits*T_mol), 15., MU_EARTH)
tr_dft = propagate(eom_j2, s_drifting, (0., n_mol_orbits*T_mol), 15., MU_EARTH)

_stride_m = max(1, len(tr_frz["t"]) // 400)
_idx_m    = np.arange(0, len(tr_frz["t"]), _stride_m)
t_days_m  = tr_frz["t"][_idx_m] / 86400.0

om_frz = np.array([state_to_coe(tr_frz["r"][i], tr_frz["v"][i])["omega_deg"] for i in _idx_m])
om_dft = np.array([state_to_coe(tr_dft["r"][i], tr_dft["v"][i])["omega_deg"] for i in _idx_m])

# Analytical rates
dom_frozen   = j2_argp_rate(a_mol, e_mol, 63.4)
dom_drifting = j2_argp_rate(a_mol, e_mol, 45.0)

print(f"\nApsidal drift rates:")
print(f"  i=63.4° (frozen)  : analytical={dom_frozen:.5f}°/day  (expect ~0)")
print(f"  i=45.0° (drifting): analytical={dom_drifting:.4f}°/day")

fig_mol, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d1e")
fig_mol.suptitle("Molniya HEO — Frozen vs Drifting Orbit\n"
                  "[BMT 2nd ed.] Section 9.7: Critical Inclination = 63.435°",
                  color="white", fontsize=12)

for ax, om_arr, inc, clr, lbl in [
        (ax1, om_frz,  63.4, "#7fff00", "i=63.4° (frozen — ω constant)"),
        (ax1, om_dft,  45.0, "#ff6b35", "i=45.0° (drifting — ω changes)"),
]:
    ax.set_facecolor("#111122")
    ax.plot(t_days_m, om_arr, color=clr, lw=1.1, label=lbl)

ax1.set_xlabel("Time (days)", color="gray"); ax1.set_ylabel("Arg. of Perigee ω (deg)", color="gray")
ax1.set_title("Argument of Perigee Over Time", color="white")
ax1.tick_params(colors="gray")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

# 3D ground track of frozen Molniya (apoapsis always over Northern Hemisphere)
ax2.remove()
ax3d = fig_mol.add_subplot(1, 2, 2, projection="3d")
ax3d.set_facecolor("#111122")

_r = tr_frz["r"]
_c = plt.cm.plasma(np.linspace(0, 1, len(_r)))
for j in range(0, len(_r)-1, 20):
    ax3d.plot(_r[j:j+21,0], _r[j:j+21,1], _r[j:j+21,2],
              color=_c[j], lw=0.6, alpha=0.7)

# Earth sphere
u=np.linspace(0,2*np.pi,20); v=np.linspace(0,np.pi,20)
ax3d.plot_surface(R_EARTH*np.outer(np.cos(u),np.sin(v)),
                  R_EARTH*np.outer(np.sin(u),np.sin(v)),
                  R_EARTH*np.outer(np.ones(20),np.cos(v)),
                  color="deepskyblue", alpha=0.2, linewidth=0)

_lim3 = 45_000
ax3d.set_xlim(-_lim3,_lim3); ax3d.set_ylim(-_lim3,_lim3); ax3d.set_zlim(-_lim3,_lim3)
ax3d.set_box_aspect([1,1,1]); ax3d.tick_params(colors="gray",labelsize=6)
ax3d.set_title("Molniya frozen orbit\n(apoapsis fixed in Northern sky)", color="white", fontsize=9)
ax3d.set_xlabel("X",color="white",fontsize=7); ax3d.set_ylabel("Y",color="white",fontsize=7)
ax3d.set_zlabel("Z",color="white",fontsize=7)

plt.tight_layout()
plt.savefig("p3_molniya_frozen.png", dpi=140, facecolor="#0d0d1e", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 8. Atmospheric Drag — Exponential Atmosphere Model
#
# > **[BMT 2nd ed.] Section 9.4** — "Atmospheric Drag"
#
# The drag acceleration on a satellite:
#   a_D = −(1/2) · C_D · (A/m) · ρ · v² · v̂
#
# Units (km system): ρ [kg/km³], A/m [km²/kg], v [km/s] → a_D [km/s²]
#
# The ballistic coefficient B_c = m/(C_D·A) [kg/km²] governs drag sensitivity.
# High B_c (dense, small): resists drag. Low B_c (large, light): decays quickly.
#
# Atmosphere model: piecewise exponential, representative of US Standard Atmosphere.
# Solar activity changes ρ by 2–3 orders of magnitude at 400–600 km.
# [BMT 2nd ed.] Section 9.4, Table 9.4-1

# %% AtmosphereModel
# Piecewise exponential atmosphere (representative, solar moderate activity)
# Format: (base altitude km, base density kg/km^3, scale height km)
# Density values from US Standard Atmosphere / NRLMSISE representative
ATMO_TABLE = [
    (  0,   1.225e9,    8.44),
    ( 25,   3.899e7,    6.49),
    ( 40,   3.972e6,    7.07),
    ( 60,   3.206e5,    6.11),
    ( 80,   1.905e4,    5.53),
    (100,   5.604e2,    5.85),
    (110,   9.708e1,    7.69),
    (120,   2.222e1,    9.52),
    (130,   8.152e0,   12.30),
    (150,   2.076e0,   22.26),
    (180,   5.194e-1,  33.74),
    (200,   2.541e-1,  47.89),
    (250,   6.073e-2,  57.42),
    (300,   1.916e-2,  59.89),
    (350,   7.014e-3,  65.47),
    (400,   2.803e-3,  65.55),
    (450,   1.184e-3,  68.38),
    (500,   5.215e-4,  73.58),
    (600,   1.137e-4,  76.30),
    (700,   3.070e-5,  72.32),
    (800,   1.136e-5,  74.89),
    (1000,  3.561e-6, 124.64),
]


def atmo_density(alt_km):
    """
    Piecewise exponential atmospheric density.
    [BMT 2nd ed.] Section 9.4, Table 9.4-1
    Returns density in kg/km^3.  (1 kg/m^3 = 1e9 kg/km^3)
    """
    if alt_km <= 0:    return ATMO_TABLE[0][1]
    if alt_km >= 1000: return 0.0
    for idx in range(len(ATMO_TABLE)-1, -1, -1):
        if alt_km >= ATMO_TABLE[idx][0]:
            h0, rho0, H = ATMO_TABLE[idx]
            return rho0 * np.exp(-(alt_km - h0) / H)
    return ATMO_TABLE[0][1]


def drag_accel(r_vec, v_vec, Cd=2.2, B=5.6e-9):
    """
    Atmospheric drag acceleration.
    [BMT 2nd ed.] Section 9.4, Equation 9.4-1

    Parameters
    ----------
    r_vec  : (3,) position in km
    v_vec  : (3,) velocity in km/s
    Cd     : drag coefficient (dimensionless, typically 2.0-2.5)
    B      : Cd*A/m ballistic term in km^2/kg
             Typical values:
               ISS-like dense object: 5.6e-9 km^2/kg  (5.6e-3 m^2/kg in SI)
               Small cubesat:         ~1e-8  km^2/kg
               Large flat panel:      ~5e-8  km^2/kg
    Returns acceleration in km/s^2.
    """
    alt   = np.linalg.norm(r_vec) - R_EARTH
    rho   = atmo_density(alt)
    v_mag = np.linalg.norm(v_vec)
    if v_mag < 1e-12:
        return np.zeros(3)
    return -0.5 * Cd * B * rho * v_mag**2 * (v_vec / v_mag)


def eom_j2_drag(t, state, mu, Cd=2.2, B=5.6e-9):
    """Two-body + J2 + atmospheric drag EOM."""
    r_vec=state[:3]; v_vec=state[3:]; r_mag=np.linalg.norm(r_vec)
    return np.concatenate([v_vec,
        -(mu/r_mag**3)*r_vec + j2_accel(r_vec, mu) + drag_accel(r_vec, v_vec, Cd, B)])


# Plot the atmosphere density profile
alts = np.linspace(100, 900, 500)
rhos = np.array([atmo_density(a) for a in alts])

fig_atm, ax_atm = plt.subplots(figsize=(8, 7), facecolor="#0d0d1e")
ax_atm.set_facecolor("#111122")
ax_atm.semilogy(alts, rhos, color="#00bfff", lw=2)

# Mark layer boundaries
for h0, rho0, H in ATMO_TABLE[6:]:   # above 100 km
    ax_atm.axhline(rho0, color="gray", lw=0.4, alpha=0.3)

# Highlight key altitudes
for alt, label in [(200,"LEO min\n200 km"), (408,"ISS\n408 km"),
                    (600,"Typical\nEO sat\n600 km"), (800,"GPS\nshell\n~800 km")]:
    rho = atmo_density(alt)
    ax_atm.scatter(alt, rho, color="#ff6b35", s=60, zorder=5)
    ax_atm.text(alt+10, rho*1.5, label, color="#ff6b35", fontsize=8)

ax_atm.set_xlabel("Altitude (km)", color="gray")
ax_atm.set_ylabel("Density (kg/km³)", color="gray")
ax_atm.set_title("Piecewise Exponential Atmosphere Model\n"
                  "[BMT 2nd ed.] Section 9.4  —  representative solar moderate activity",
                  color="white", fontsize=11)
ax_atm.tick_params(colors="gray")
plt.tight_layout()
plt.savefig("p3_atmosphere.png", dpi=140, facecolor="#0d0d1e", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 9. Orbital Decay Under Drag
#
# We simulate the same orbit with two different ballistic coefficients:
# - **Dense object** B = 5.6×10⁻⁹ km²/kg (ISS-like: 5.6×10⁻³ m²/kg in SI)
# - **Light panel**  B = 2.0×10⁻⁸ km²/kg (larger A/m ratio, e.g. flat satellite)
#
# Both start at 300 km (highly sensitive altitude) and are propagated for 20 days.
# The difference in decay rate directly reflects the ballistic coefficient ratio.
#
# Note: J₂ is included — it affects the RAAN and precession but not the decay rate
# significantly. The altitude decay is driven almost entirely by drag.

# %% DragDecay
sim_days = 20
s300 = circ_state(300, inc_deg=28.5)

print(f"Propagating drag decay at 300 km, {sim_days} days ...\n")

# Dense object (ISS-like)
B_dense = 5.6e-9   # km^2/kg  ->  5.6e-3 m^2/kg in SI
tr_dense = propagate(eom_j2_drag, s300, (0., sim_days*86400), 10., MU_EARTH, 2.2, B_dense)
alt_dense = np.linalg.norm(tr_dense["r"], axis=1) - R_EARTH

# Light panel (higher A/m)
B_light = 2.0e-8   # km^2/kg  ->  2.0e-2 m^2/kg in SI
tr_light = propagate(eom_j2_drag, s300, (0., sim_days*86400), 10., MU_EARTH, 2.2, B_light)
alt_light = np.linalg.norm(tr_light["r"], axis=1) - R_EARTH

t_dense_days = tr_dense["t"] / 86400.0
t_light_days = tr_light["t"] / 86400.0

print(f"  Dense (B={B_dense:.1e} km²/kg = {B_dense*1e6:.3f} m²/kg):")
print(f"    300.0 -> {alt_dense[-1]:.1f} km  (decay {alt_dense[0]-alt_dense[-1]:.1f} km)")
if tr_dense["reentered"]: print("    STATUS: RE-ENTERED")

print(f"\n  Light (B={B_light:.1e} km²/kg = {B_light*1e6:.3f} m²/kg):")
print(f"    300.0 -> {alt_light[-1]:.1f} km  (decay {alt_light[0]-alt_light[-1]:.1f} km)")
if tr_light["reentered"]: print("    STATUS: RE-ENTERED")

fig_drag, (ax_drag1, ax_drag2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d0d1e")
fig_drag.suptitle("Orbital Decay Due to Atmospheric Drag — 300 km Initial Orbit\n"
                   "[BMT 2nd ed.] Section 9.4, a_D = −(1/2)·C_D·(A/m)·ρ·v²",
                   color="white", fontsize=12)

for ax_drag in [ax_drag1, ax_drag2]:
    ax_drag.set_facecolor("#111122"); ax_drag.tick_params(colors="gray")
    ax_drag.set_xlabel("Time (days)", color="gray")
    ax_drag.axhline(80, color="red", lw=0.8, linestyle="--", alpha=0.5)
    ax_drag.text(0.5, 82, "Approx. re-entry threshold (80 km)", color="red",
                 fontsize=8, alpha=0.7)

ax_drag1.plot(t_dense_days, alt_dense, color="#00bfff", lw=1.5,
              label=f"Dense  B={B_dense:.1e} km²/kg\n({B_dense*1e6:.3f} m²/kg SI)")
ax_drag1.plot(t_light_days, alt_light, color="#ff6b35", lw=1.5,
              label=f"Light  B={B_light:.1e} km²/kg\n({B_light*1e6:.3f} m²/kg SI)")
ax_drag1.set_ylabel("Altitude (km)", color="gray")
ax_drag1.set_title("Altitude Decay", color="white")
ax_drag1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

# Instantaneous drag magnitude over time
t_probe = tr_dense["t"][::100]
a_drag_mag = np.array([
    np.linalg.norm(drag_accel(tr_dense["r"][i], tr_dense["v"][i], 2.2, B_dense))
    for i in range(0, len(tr_dense["t"]), 100)
]) * 1e6   # km/s^2 -> micro-m/s^2 (micrometers/s²)

ax_drag2.semilogy(t_probe/86400, a_drag_mag, color="#00bfff", lw=1.2)
ax_drag2.set_ylabel("|a_drag| (μm/s²)", color="gray")
ax_drag2.set_title("Drag Acceleration Magnitude (Dense object)\n"
                    "grows as orbit decays into denser atmosphere", color="white", fontsize=9)

plt.tight_layout()
plt.savefig("p3_drag_decay.png", dpi=140, facecolor="#0d0d1e", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 10. Sun-Synchronous Orbit Demonstration
#
# > **[BMT 2nd ed.] Section 9.5** — "Sun-Synchronous Orbits"
#
# At SSO inclination, J₂ RAAN precession exactly matches the Sun's apparent eastward
# motion (+0.9856°/day). The orbit-to-sun geometry is frozen — the spacecraft crosses
# any latitude at the same local solar time on every pass.
#
# This is retrograde (i > 90°), which is why SSO launches go southward from mid-latitude
# sites (Vandenberg, Plesetsk, Kourou) — never eastward.

# %% SunSyncDemo
alt_sso  = 600.0          # km
inc_sso  = sso_inclination(alt_sso)
T_sso    = orbital_period(R_EARTH + alt_sso)
n_sso    = 30             # orbits to simulate

print(f"Sun-synchronous orbit at {alt_sso} km:")
print(f"  Required inclination : {inc_sso:.4f}°")
print(f"  Orbital period       : {T_sso/60:.2f} min")
print(f"  Simulating {n_sso} orbits (~{n_sso*T_sso/86400:.1f} days) ...")

s_sso    = circ_state(alt_sso, inc_deg=inc_sso)
tr_sso   = propagate(eom_j2, s_sso, (0., n_sso*T_sso), 30., MU_EARTH)

# Extract RAAN over time
_stride_s = max(1, len(tr_sso["t"]) // 300)
_idx_s    = np.arange(0, len(tr_sso["t"]), _stride_s)
t_sso_d   = tr_sso["t"][_idx_s] / 86400.0

raan_sso_raw = np.array([state_to_coe(tr_sso["r"][i], tr_sso["v"][i])["raan_deg"]
                          for i in _idx_s])
raan_sso_uw  = np.degrees(np.unwrap(np.radians(raan_sso_raw)))
raan_sso_rate= np.polyfit(t_sso_d, raan_sso_uw, 1)[0]

sun_rate     = 0.9856  # deg/day

print(f"\n  RAAN precession measured : {raan_sso_rate:.4f}°/day")
print(f"  Sun apparent motion      : +{sun_rate:.4f}°/day")
print(f"  Match quality            : {abs(raan_sso_rate-sun_rate):.5f}°/day error")

fig_sso, axes_sso = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d0d1e")
fig_sso.suptitle(f"Sun-Synchronous Orbit — {alt_sso} km, i={inc_sso:.2f}°\n"
                  "[BMT 2nd ed.] Section 9.5: RAAN tracks Sun's apparent motion",
                  color="white", fontsize=12)

ax_sso1, ax_sso2 = axes_sso
ax_sso1.set_facecolor("#111122"); ax_sso2.set_facecolor("#111122")

# RAAN vs time with Sun reference line
sun_line = sun_rate * t_sso_d + raan_sso_uw[0]
ax_sso1.plot(t_sso_d, raan_sso_uw, color="#ff6b35", lw=1.5, label=f"SSO RAAN ({raan_sso_rate:.4f}°/day)")
ax_sso1.plot(t_sso_d, sun_line,    color="#ffff00", lw=1.0, linestyle="--",
              label=f"Sun motion (+{sun_rate}°/day)")
ax_sso1.set_xlabel("Time (days)", color="gray"); ax_sso1.set_ylabel("RAAN Ω (deg)", color="gray")
ax_sso1.set_title("RAAN Precession Matches Sun", color="white")
ax_sso1.tick_params(colors="gray")
ax_sso1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

# 3D orbit
ax_sso2.remove()
ax3d_sso = fig_sso.add_subplot(1, 2, 2, projection="3d")
ax3d_sso.set_facecolor("#111122")

u=np.linspace(0,2*np.pi,22); v=np.linspace(0,np.pi,22)
ax3d_sso.plot_surface(R_EARTH*np.outer(np.cos(u),np.sin(v)),
                      R_EARTH*np.outer(np.sin(u),np.sin(v)),
                      R_EARTH*np.outer(np.ones(22),np.cos(v)),
                      color="deepskyblue", alpha=0.18, linewidth=0)

_r_sso = tr_sso["r"]
n_col  = len(_r_sso)
colors = plt.cm.plasma(np.linspace(0, 1, n_col))
for _j in range(0, n_col-1, 15):
    ax3d_sso.plot(_r_sso[_j:_j+16,0], _r_sso[_j:_j+16,1], _r_sso[_j:_j+16,2],
                  color=colors[_j], lw=0.7, alpha=0.8)

_lim_s = 9000
ax3d_sso.set_xlim(-_lim_s,_lim_s); ax3d_sso.set_ylim(-_lim_s,_lim_s); ax3d_sso.set_zlim(-_lim_s,_lim_s)
ax3d_sso.set_box_aspect([1,1,1]); ax3d_sso.tick_params(colors="gray",labelsize=6)
ax3d_sso.set_title(f"SSO 3D — {n_sso} orbits\nRAAO precesses like the Sun",
                   color="white", fontsize=9)
ax3d_sso.set_xlabel("X",color="white",fontsize=7); ax3d_sso.set_ylabel("Y",color="white",fontsize=7)
ax3d_sso.set_zlabel("Z",color="white",fontsize=7)

plt.tight_layout()
plt.savefig("p3_sun_sync.png", dpi=140, facecolor="#0d0d1e", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 11. Summary Reference Table
#
# | Function | Physics | Reference |
# |----------|---------|-----------|
# | j2_accel() | J₂ Cartesian acceleration | [BMT 2nd ed.] Section 9.3, Eq. 9.3-1 |
# | eom_j2() | Two-body + J₂ EOM | [BMT 2nd ed.] Section 9.3 |
# | j2_raan_rate() | dΩ/dt secular rate | [BMT 2nd ed.] Section 9.3, Eq. 9.3-4 |
# | j2_argp_rate() | dω/dt secular rate | [BMT 2nd ed.] Section 9.3, Eq. 9.3-5 |
# | sso_inclination() | i for SSO at altitude h | [BMT 2nd ed.] Section 9.5 |
# | atmo_density() | Piecewise exp. atmosphere | [BMT 2nd ed.] Section 9.4, Table 9.4-1 |
# | drag_accel() | a_D = −(1/2)C_D(A/m)ρv² | [BMT 2nd ed.] Section 9.4, Eq. 9.4-1 |
# | eom_j2_drag() | Full perturbed EOM | [BMT 2nd ed.] Chapter 9 |
#
# **Key numbers:**
#
# | Effect | Value | Condition |
# |--------|-------|-----------|
# | ISS RAAN precession | −4.99°/day | 408 km, 51.6° |
# | Critical inclination | 63.435° | ω frozen (Molniya) |
# | ISS drag makeup ΔV | ~60 m/s/year | 400 km, solar moderate |
# | SSO inclination 600 km | 97.8° | matches Sun's motion |
# | J₂ / central gravity ratio | ~10⁻³ | scales as (R_E/r)² |
#
# **Phase 4 → phase4_propulsion.py**
# Topics: thrust, specific impulse, de Laval nozzle thermodynamics,
# combustion chamber, finite-burn simulation, staging.
# References: [S&B] Chapters 2, 3, 4
