# %% [markdown]
# # 🚀 Phase 1 — The Two-Body Problem
# ## Orbital Mechanics & Astrodynamics — From First Principles to Simulation
#
# **Format:** `# %%` (py:percent) — run interactively in Zed/VS Code with the Jupyter
# extension, or in Spyder. Convert to Marimo with:
#   `marimo convert phase1_two_body.py -o phase1_marimo.py`
# Convert to Jupyter notebook with:
#   `jupytext --to notebook phase1_two_body.py`
#
# **Read alongside:** `theory_01_two_body.md` for complete derivations.
#
# **References:**
# - **[BMT]** Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020
# - **[S&B]** Sutton & Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  registers 3d projection
import csv
import json
import os
from datetime import datetime

# %% [markdown]
# ---
# ## 2. Physical Constants — km / s / kg System
#
# > **[BMT 2nd ed.] Section 1.2** — "Newtonian Mechanics"
#
# Standard gravitational parameters mu = G*M are known to far higher precision than
# G or M individually. For Earth: mu = 398600.4418 km^3/s^2 (10 sig. figs).
# Never compute G*M; always use mu directly.
#
# > Warning: Mixing km and m destroyed the Mars Climate Orbiter in 1999.

# %% Constants
MU_EARTH = 398_600.4418
MU_SUN   = 1.327_124_4e11
MU_MOON  = 4_902.800_066
MU_MARS  = 42_828.375
R_EARTH  = 6_371.0
AU       = 1.495_978_707e8

print("Constants (km / s / kg):")
print(f"  mu_Earth = {MU_EARTH:>16,.4f}  km^3/s^2")
print(f"  mu_Sun   = {MU_SUN:>16.6e}  km^3/s^2")
print(f"  R_Earth  = {R_EARTH:>16,.1f}  km")

# %% [markdown]
# ---
# ## 3. Equations of Motion
#
# > **[BMT 2nd ed.] Sections 1.3-1.4**
#
# Two-body EOM in state-space form (6-vector y = [r, v]):
#
#   dy/dt = [v,  -mu/|r|^3 * r]
#
# The system is autonomous (no explicit t), which is why two constants of motion exist:
# specific energy eps and specific angular momentum h. [BMT 2nd ed.] Section 1.4

# %% EOM
def two_body_eom(t, state, mu):
    """
    Two-body equations of motion.
    [BMT 2nd ed.] Section 1.3, Equation 1.3-3

    t     : float     -- time in s (unused; autonomous system)
    state : (6,) ndarray -- [x, y, z, vx, vy, vz] in km, km/s
    mu    : float     -- gravitational parameter in km^3/s^2
    Returns dstate/dt : (6,) ndarray -- [vx, vy, vz, ax, ay, az]
    """
    r_vec = state[:3]
    v_vec = state[3:]
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 1e-10:
        raise ValueError(f"Singularity: |r| = {r_mag:.2e}.")
    return np.concatenate([v_vec, -(mu / r_mag**3) * r_vec])


_g = np.linalg.norm(two_body_eom(0, np.array([R_EARTH,0,0,0,0,0.]), MU_EARTH)[3:]) * 1000
print(f"Surface gravity: {_g:.4f} m/s^2  (expect 9.8196)")

# %% [markdown]
# ---
# ## 4. RK4 Integrator
#
# 4th-order Runge-Kutta — global error O(h^4). Halving h reduces error by 16x.
#
#   y_{n+1} = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
#
# Accuracy check: track |d_eps/eps| — should stay below 1e-7 per orbit for LEO at dt=10s.

# %% RK4
def rk4_step(f, t, y, h, *args):
    """Single step of 4th-order Runge-Kutta. Global error O(h^4)."""
    k1 = f(t,       y,           *args)
    k2 = f(t+h/2,   y+(h/2)*k1, *args)
    k3 = f(t+h/2,   y+(h/2)*k2, *args)
    k4 = f(t+h,     y+h*k3,     *args)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def propagate(eom, state0, t_span, dt, mu):
    """
    Fixed-step RK4 trajectory propagator.
    Returns dict: t (s), states (N,6), r (N,3) km, v (N,3) km/s
    """
    t0, tf  = t_span
    n_steps = int((tf - t0) / dt)
    t_arr   = np.zeros(n_steps + 1)
    states  = np.zeros((n_steps + 1, 6))
    t_arr[0]  = t0
    states[0] = state0.copy()
    for i in range(n_steps):
        states[i+1] = rk4_step(eom, t_arr[i], states[i], dt, mu)
        t_arr[i+1]  = t_arr[i] + dt
    return {"t": t_arr, "states": states,
            "r": states[:, :3], "v": states[:, 3:]}

print("RK4 integrator ready.")

# %% [markdown]
# ---
# ## 5. Orbital Mechanics Helpers
#
# References:
# - Angular momentum  h = r x v          [BMT 2nd ed.] Section 1.4.1, Eq 1.4-1
# - Specific energy   eps = v^2/2 - mu/r [BMT 2nd ed.] Section 1.4.2, Eq 1.4-2
# - Eccentricity vec  e = (v x h)/mu - r_hat  [BMT 2nd ed.] Section 1.6, Eq 1.6-2
# - Vis-viva          v^2 = mu*(2/r - 1/a)    [BMT 2nd ed.] Section 1.7, Eq 1.7-1
# - Kepler third law  T = 2*pi*sqrt(a^3/mu)   [BMT 2nd ed.] Section 1.7, Eq 1.7-4
# - COE->state        perifocal + rotation Q  [BMT 2nd ed.] Section 2.4.3, Eq 2.4-3/4
# - State->COE        Algorithm 2.2           [BMT 2nd ed.] Section 2.4

# %% OrbitalHelpers
def specific_energy(r_vec, v_vec, mu):
    """eps = v^2/2 - mu/r  [km^2/s^2]. Negative for bound orbits."""
    return 0.5*np.linalg.norm(v_vec)**2 - mu/np.linalg.norm(r_vec)

def spec_ang_mom(r_vec, v_vec):
    """h = r x v  [km^2/s]. Conserved => orbit is planar."""
    return np.cross(r_vec, v_vec)

def ecc_vector(r_vec, v_vec, mu):
    """Eccentricity (LRL) vector. Points to periapsis, |e_vec| = eccentricity."""
    r, v = np.linalg.norm(r_vec), np.linalg.norm(v_vec)
    return (1.0/mu)*((v**2 - mu/r)*r_vec - np.dot(r_vec, v_vec)*v_vec)

def orbital_period(a, mu):
    """T = 2*pi*sqrt(a^3/mu)  [s]  Kepler Third Law."""
    return 2.0*np.pi*np.sqrt(a**3/mu)

def vis_viva(r, a, mu):
    """v = sqrt(mu*(2/r - 1/a))  [km/s]"""
    return np.sqrt(mu*(2.0/r - 1.0/a))

def circular_orbit_state(alt_km, inc_deg=0.0, mu=MU_EARTH):
    """State vector for circular orbit. Satellite starts on x-axis at ascending node."""
    r   = R_EARTH + alt_km
    v_c = np.sqrt(mu/r)
    i   = np.radians(inc_deg)
    return np.array([r, 0., 0., 0., v_c*np.cos(i), v_c*np.sin(i)])

def coe_to_state(a, e, i_deg, raan_deg, omega_deg, theta_deg, mu=MU_EARTH):
    """
    COE -> ECI state vector.
    [BMT 2nd ed.] Section 2.4.3, Algorithm 2.1
    Perifocal frame + rotation Q = R3(-Omega)*R1(-i)*R3(-omega)
    """
    i, O, w, th = [np.radians(x) for x in (i_deg, raan_deg, omega_deg, theta_deg)]
    p   = a*(1.0 - e**2)
    r_m = p/(1.0 + e*np.cos(th))
    r_pf = r_m*np.array([np.cos(th), np.sin(th), 0.])
    v_pf = np.sqrt(mu/p)*np.array([-np.sin(th), e+np.cos(th), 0.])
    cO,sO = np.cos(O),np.sin(O)
    ci,si = np.cos(i),np.sin(i)
    cw,sw = np.cos(w),np.sin(w)
    Q = np.array([
        [ cO*cw-sO*sw*ci,  -cO*sw-sO*cw*ci,  sO*si],
        [ sO*cw+cO*sw*ci,  -sO*sw+cO*cw*ci, -cO*si],
        [ sw*si,             cw*si,            ci   ]
    ])
    return np.concatenate([Q@r_pf, Q@v_pf])

def state_to_coe(r_vec, v_vec, mu=MU_EARTH):
    """
    ECI state -> COE dict {a, e, i_deg, raan_deg, omega_deg, theta_deg}.
    [BMT 2nd ed.] Section 2.4, Algorithm 2.2
    """
    r, v = np.linalg.norm(r_vec), np.linalg.norm(v_vec)
    h_v  = spec_ang_mom(r_vec, v_vec); h = np.linalg.norm(h_v)
    e_v  = ecc_vector(r_vec, v_vec, mu); e = np.linalg.norm(e_v)
    i_d  = np.degrees(np.arccos(np.clip(h_v[2]/h, -1., 1.)))
    N_v  = np.cross(np.array([0.,0.,1.]), h_v); N = np.linalg.norm(N_v)
    Om = 0.
    if N > 1e-10:
        Om = np.degrees(np.arccos(np.clip(N_v[0]/N, -1., 1.)))
        if N_v[1] < 0: Om = 360.-Om
    om = 0.
    if N > 1e-10 and e > 1e-10:
        om = np.degrees(np.arccos(np.clip(np.dot(N_v, e_v)/(N*e), -1., 1.)))
        if e_v[2] < 0: om = 360.-om
    th = 0.
    if e > 1e-10:
        th = np.degrees(np.arccos(np.clip(np.dot(e_v, r_vec)/(e*r), -1., 1.)))
        if np.dot(r_vec, v_vec) < 0: th = 360.-th
    a = -mu/(2.*specific_energy(r_vec, v_vec, mu))
    return {"a": a, "e": e, "i_deg": i_d, "raan_deg": Om, "omega_deg": om, "theta_deg": th}


_s = circular_orbit_state(408, 51.6)
_T = orbital_period(R_EARTH+408, MU_EARTH)
print(f"ISS orbit: |r|={np.linalg.norm(_s[:3]):.2f} km  |v|={np.linalg.norm(_s[3:]):.4f} km/s  T={_T/60:.2f} min")

# %% [markdown]
# ---
# ## 6. Simulate Three Reference Orbits
#
# LEO (ISS): 408 km, i=51.6 deg, ~92.7 min period.
# GEO: 35786 km, i=0 deg, T = 86164 s (1 sidereal day exactly).
# HEO Molniya: rp=600 km, ra=39000 km, i=63.4 deg.
#   63.4 deg freezes apsidal precession under J2. [BMT 2nd ed.] Section 9.7

# %% SimulateOrbits
_rp = R_EARTH + 600.0
_ra = R_EARTH + 39_000.0
A_HEO = (_rp + _ra) / 2.0
E_HEO = (_ra - _rp) / (_ra + _rp)

ORBIT_DEFS = {
    "LEO (ISS)":    {"state0": circular_orbit_state(408,   51.6), "a": R_EARTH+408,
                     "dt": 10., "n_per": 3, "color": "#00bfff"},
    "GEO":          {"state0": circular_orbit_state(35786, 0.0),  "a": R_EARTH+35786,
                     "dt": 60., "n_per": 1, "color": "#ff6b35"},
    "HEO (Molniya)":{"state0": coe_to_state(A_HEO, E_HEO, 63.4, 0., 270., 0.),
                     "a": A_HEO, "dt": 10., "n_per": 1, "color": "#7fff00"},
}

results = {}
print(f"  {'Orbit':<22}  {'Steps':>7}  {'de/e':>10}  {'dh/h':>10}  Status")
print("  " + "-"*58)
for name, cfg in ORBIT_DEFS.items():
    T    = orbital_period(cfg["a"], MU_EARTH)
    traj = propagate(two_body_eom, cfg["state0"], (0., T*cfg["n_per"]), cfg["dt"], MU_EARTH)
    eps  = np.array([specific_energy(traj["r"][i], traj["v"][i], MU_EARTH) for i in range(len(traj["t"]))])
    h    = np.array([np.linalg.norm(spec_ang_mom(traj["r"][i], traj["v"][i])) for i in range(len(traj["t"]))])
    de   = abs((eps[-1]-eps[0])/eps[0])
    dh   = abs((h[-1]-h[0])/h[0])
    results[name] = {**cfg, "traj": traj, "T": T, "eps": eps, "h": h}
    print(f"  {name:<22}  {len(traj['t'])-1:>7,}  {de:>10.2e}  {dh:>10.2e}  {'OK' if de<1e-5 else 'WARN'}")

# %% [markdown]
# ---
# ## 7. 3D Visualization — ECI Frame
#
# ECI: X -> vernal equinox, Z -> North Pole, non-rotating. [BMT 2nd ed.] Section 2.2
# Inclination is visually clear. GEO lies in the equatorial plane (Z=0).

# %% Plot3D
def _draw_earth(ax, n=28):
    u = np.linspace(0,2*np.pi,n); v = np.linspace(0,np.pi,n)
    x = R_EARTH*np.outer(np.cos(u),np.sin(v))
    y = R_EARTH*np.outer(np.sin(u),np.sin(v))
    z = R_EARTH*np.outer(np.ones(n),np.cos(v))
    ax.plot_surface(x,y,z,color='deepskyblue',alpha=0.20,linewidth=0)
    ax.plot_wireframe(x,y,z,color='royalblue',linewidth=0.22,alpha=0.35)

fig_3d = plt.figure(figsize=(12,9),facecolor='#0a0a1a')
ax_3d  = fig_3d.add_subplot(111,projection='3d')
ax_3d.set_facecolor('#0a0a1a')
_draw_earth(ax_3d)

for name, data in results.items():
    r = data["traj"]["r"]
    ax_3d.plot(r[:,0],r[:,1],r[:,2],color=data["color"],lw=1.4,label=name,alpha=0.9)
    ax_3d.scatter(*r[0],color=data["color"],s=55,zorder=5)

_lim = 44_500
_g   = np.linspace(-_lim,_lim,4)
_GX,_GY = np.meshgrid(_g,_g)
ax_3d.plot_surface(_GX,_GY,np.zeros_like(_GX),alpha=0.04,color='white')
ax_3d.plot([0,0],[0,0],[-_lim*.35,_lim*.35],color='white',lw=0.7,alpha=0.25,linestyle='--')

ax_3d.set_xlim(-_lim,_lim); ax_3d.set_ylim(-_lim,_lim); ax_3d.set_zlim(-_lim,_lim)
ax_3d.set_box_aspect([1,1,1])
ax_3d.set_xlabel("X (km) -> Vernal Equinox",color='white',fontsize=9)
ax_3d.set_ylabel("Y (km)",color='white',fontsize=9)
ax_3d.set_zlabel("Z (km) -> North Pole",color='white',fontsize=9)
ax_3d.set_title("LEO / GEO / HEO — ECI Frame",color='white',fontsize=13,pad=12)
ax_3d.tick_params(colors='gray',labelsize=7)
ax_3d.legend(facecolor='#1a1a2e',labelcolor='white',fontsize=10,loc='upper left')
plt.tight_layout()
plt.savefig("p1_orbits_3d.png",dpi=140,facecolor='#0a0a1a',bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 8. Conservation Diagnostics
#
# For pure two-body, eps and |h| are exactly constant.
# Any drift is numerical error only.
#
# Drift < 1e-8 : excellent    |   Drift > 1e-3 : reduce dt
# [BMT 2nd ed.] Section 1.4

# %% Conservation
fig_c, axes_c = plt.subplots(2,3,figsize=(16,7),facecolor='#0d0d1e')
fig_c.suptitle("Conservation Diagnostics — RK4 Accuracy",color='white',fontsize=13)

for col,(name,data) in enumerate(results.items()):
    t_hr = data["traj"]["t"]/3600.
    for row,(arr,lbl) in enumerate([
            ((data["eps"]-data["eps"][0])/abs(data["eps"][0]), "Energy eps (relative drift)"),
            ((data["h"]  -data["h"][0])  /abs(data["h"][0]),   "|h| (relative drift)")]):
        ax = axes_c[row,col]
        ax.set_facecolor('#111122')
        ax.plot(t_hr,arr,color=data["color"],lw=0.9)
        ax.axhline(0,color='white',lw=0.4,alpha=0.3,linestyle='--')
        ax.set_title(f"{name}\n{lbl}",fontsize=9,color='white')
        ax.tick_params(colors='gray')
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        if col==0: ax.set_ylabel("Relative drift",color='gray',fontsize=9)
        if row==1: ax.set_xlabel("Time (hours)",color='gray',fontsize=9)

plt.tight_layout()
plt.savefig("p1_conservation.png",dpi=140,facecolor='#0d0d1e',bbox_inches='tight')
plt.show()

print("Drift summary:")
for name,data in results.items():
    de=abs((data["eps"][-1]-data["eps"][0])/data["eps"][0])
    dh=abs((data["h"][-1] -data["h"][0]) /data["h"][0])
    print(f"  {name:<22}  de/e={de:.2e}   dh/h={dh:.2e}")

# %% [markdown]
# ---
# ## 9. Orbital Elements Over Time
#
# a, e, i, Omega, omega must be constant for pure two-body.
# theta varies (satellite moving along orbit).
# After Phase 3 perturbations: RAAN precesses, a decays under drag.
# [BMT 2nd ed.] Section 2.4, Section 9.3-9.7

# %% COEsOverTime
_name   = "LEO (ISS)"
_traj   = results[_name]["traj"]
_stride = max(1, len(_traj["t"])//500)
_idx    = np.arange(0, len(_traj["t"]), _stride)
_keys   = ("a","e","i_deg","theta_deg")
_coes   = np.array([[state_to_coe(_traj["r"][k],_traj["v"][k],MU_EARTH)[key]
                     for key in _keys] for k in _idx])
_t_sub  = _traj["t"][_idx]/3600.

fig_coe,axes_coe = plt.subplots(4,1,figsize=(12,10),sharex=True,facecolor='#0d0d1e')
fig_coe.suptitle(f"Orbital Elements — {_name}",color='white',fontsize=12)
_lbls  = ["a (km)","e","i (deg)","theta (deg)"]
_clrs  = [results[_name]["color"],"#ff6b35","#7fff00","#ff00ff"]
for ax,arr,lbl,clr in zip(axes_coe,_coes.T,_lbls,_clrs):
    ax.set_facecolor('#111122'); ax.plot(_t_sub,arr,color=clr,lw=1.)
    ax.set_ylabel(lbl,color='gray',fontsize=9); ax.tick_params(colors='gray')
axes_coe[-1].set_xlabel("Time (hours)",color='gray')
plt.tight_layout()
plt.savefig("p1_orbital_elements.png",dpi=140,facecolor='#0d0d1e',bbox_inches='tight')
plt.show()
print("Element ranges (expect ~0 for a, e, i):")
for lbl,arr in zip(["a","e","i"],_coes.T[:3]):
    print(f"  {lbl}: {arr.max()-arr.min():.2e}")

# %% [markdown]
# ---
# ## 10. Time Scaling — Physics vs Display
#
# The trajectory array is the ground truth, computed at full fidelity.
# Display speed uses frame striding — it never modifies the integration.
#
# Frame striding: skip every N points. Physics unchanged. Use this for animation.
# Larger dt: reduces accuracy. Only if conservation still validates.

# %% TimeScaling
_n  = len(results["LEO (ISS)"]["traj"]["t"])
_dt = ORBIT_DEFS["LEO (ISS)"]["dt"]
_T  = results["LEO (ISS)"]["T"]
print(f"LEO: {_n:,} points | dt={_dt}s | T={_T:.0f}s\n")
print(f"  {'Speed':<14}  {'Stride':>8}  {'Frames@30fps':>12}  {'Video(s)':>10}  {'Orbit-time(s)':>14}")
print("  "+"-"*60)
for _l,_s in [("1x",1),("10x",10),("100x",100),("1000x",1000),("10000x",10000)]:
    _nf=_n//_s; _vs=_nf/30; _ot=_n*_dt
    print(f"  {_l:<14}  {_s:>8,}  {_nf:>12,}  {_vs:>10.1f}  {_ot:>14.0f}")

# %% [markdown]
# ---
# ## 11. Export to CSV and JSON

# %% Export
os.makedirs("exports",exist_ok=True)

def export_csv(name,traj,filepath,stride=10):
    with open(filepath,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["t_s","x_km","y_km","z_km","vx_kms","vy_kms","vz_kms","r_km","v_kms"])
        for i in range(0,len(traj["t"]),stride):
            r,v=traj["r"][i],traj["v"][i]
            w.writerow([f"{traj['t'][i]:.3f}",
                        f"{r[0]:.6f}",f"{r[1]:.6f}",f"{r[2]:.6f}",
                        f"{v[0]:.8f}",f"{v[1]:.8f}",f"{v[2]:.8f}",
                        f"{np.linalg.norm(r):.6f}",f"{np.linalg.norm(v):.8f}"])

def export_json(name,traj,cfg,filepath,stride=10):
    idx=range(0,len(traj["t"]),stride)
    data={"metadata":{"orbit_name":name,"integrator":"RK4","dt_s":cfg["dt"],
                      "mu_km3_s2":MU_EARTH,"r0_km":traj["r"][0].tolist(),
                      "v0_kms":traj["v"][0].tolist(),"frame":"ECI",
                      "ref":"[BMT 2nd ed.] Bate et al., Dover 2020",
                      "exported_at":datetime.utcnow().isoformat()+"Z"},
          "t":[traj["t"][i] for i in idx],
          "r":[traj["r"][i].tolist() for i in idx],
          "v":[traj["v"][i].tolist() for i in idx]}
    with open(filepath,"w") as f: json.dump(data,f,indent=2)

for name,data in results.items():
    safe=name.replace(" ","_").replace("(","").replace(")","").replace("/","_")
    traj=data["traj"]
    export_csv( name,traj,f"exports/p1_{safe}.csv", stride=10)
    export_json(name,traj,ORBIT_DEFS[name],f"exports/p1_{safe}.json",stride=10)
    print(f"  Exported: p1_{safe}.{{csv,json}}")

# %% [markdown]
# ---
# ## 12. Summary Reference Table
#
# | Function | Equation | Reference |
# |----------|----------|-----------|
# | two_body_eom | r_ddot = -mu*r/|r|^3 | [BMT 2nd ed.] Section 1.3 |
# | specific_energy | eps = v^2/2 - mu/r | [BMT 2nd ed.] Section 1.4.2 |
# | spec_ang_mom | h = r x v | [BMT 2nd ed.] Section 1.4.1 |
# | ecc_vector | e = (v x h)/mu - r_hat | [BMT 2nd ed.] Section 1.6 |
# | orbital_period | T = 2*pi*sqrt(a^3/mu) | [BMT 2nd ed.] Section 1.7 |
# | vis_viva | v = sqrt(mu*(2/r-1/a)) | [BMT 2nd ed.] Section 1.7 |
# | coe_to_state | perifocal -> ECI via Q | [BMT 2nd ed.] Section 2.4.3 |
# | state_to_coe | ECI -> COE Algorithm 2.2 | [BMT 2nd ed.] Section 2.4 |
#
# Phase 2 -> phase2_maneuvers.py
