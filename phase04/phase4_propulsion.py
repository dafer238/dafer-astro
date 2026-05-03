# %% [markdown]
# # 🚀 Phase 4 — Rocket Propulsion Systems
# ## Thrust, Specific Impulse, Nozzle Thermodynamics, Finite Burns, Staging
#
# **Format:** `# %%` (py:percent) — Zed/VS Code Jupyter, Spyder, or:
#   `marimo convert phase4_propulsion.py -o phase4_marimo.py`
#
# **Read alongside:** `theory_04_propulsion.md`
#
# **References:**
# - **[S&B]** Sutton & Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017
# - **[BMT]** Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# %% [markdown]
# ---
# ## 2. Physical Constants

# %% Constants
R_UNIV   = 8314.46        # J/(kmol·K)  universal gas constant
G0       = 9.80665        # m/s^2       standard gravity (for Isp definition)
MU_EARTH = 398_600.4418   # km^3/s^2
R_EARTH  = 6_371.0        # km
J2       = 1.082_627_0e-3

print("Constants loaded.")
print(f"  R_universal = {R_UNIV} J/(kmol·K)")
print(f"  g0          = {G0} m/s^2  (defines Isp: Isp = F/(mdot*g0))")

# %% [markdown]
# ---
# ## 3. Isentropic Nozzle Flow
#
# > **[S&B] Section 3.1, Equations 3-12 to 3-22**
#
# The converging-diverging (de Laval) nozzle converts heat → kinetic energy.
# All flow is treated as isentropic (adiabatic + frictionless) with ideal gas.
#
# Key relations using stagnation (chamber) conditions and Mach number M:
#
#   T/T0 = [1 + (γ-1)/2 · M²]⁻¹                  [S&B] Eq 3-12
#   p/p0 = [1 + (γ-1)/2 · M²]^{-γ/(γ-1)}         [S&B] Eq 3-13
#   A/A* = (1/M)·[(2/(γ+1))·(1+(γ-1)/2·M²)]^{(γ+1)/(2(γ-1))}  [S&B] Eq 3-22
#
# Finding M from A/A* requires Newton-Raphson (no closed form).
# The equation has two solutions: subsonic (M<1) and supersonic (M>1).

# %% NozzleFlow
def area_ratio(M, gamma):
    """
    A/A* as a function of Mach number.
    [S&B] Equation 3-22.
    Equals 1.0 at M=1 (throat, by definition).
    """
    t = 1.0 + (gamma - 1.0)/2.0 * M**2
    e = (gamma + 1.0) / (2.0*(gamma - 1.0))
    return (1.0/M) * ((2.0/(gamma + 1.0)) * t)**e


def _d_area_ratio_dM(M, gamma):
    """Analytical derivative d(A/A*)/dM — used only inside mach_from_area_ratio."""
    AR = area_ratio(M, gamma)
    t  = 1.0 + (gamma - 1.0)/2.0 * M**2
    return AR * (-1.0/M + (gamma + 1.0)/(2.0*(gamma - 1.0)) * (gamma - 1.0)*M / t)


def mach_from_area_ratio(eps, gamma, supersonic=True, tol=1e-12, maxiter=100):
    """
    Invert A/A* = eps for Mach number using Newton-Raphson.
    [S&B] Section 3.3 — no closed-form inverse exists.

    supersonic=True  : returns the M>1 (diverging section) solution
    supersonic=False : returns the M<1 (converging section) solution
    """
    M = 3.0 if supersonic else 0.3
    for _ in range(maxiter):
        dM = (area_ratio(M, gamma) - eps) / _d_area_ratio_dM(M, gamma)
        M -= dM
        M = max(M, 1.0001) if supersonic else min(max(M, 1e-6), 0.9999)
        if abs(dM) < tol:
            break
    return M


def isentropic_T_ratio(M, gamma):
    """T/T0 at Mach M. [S&B] Eq 3-12."""
    return 1.0 / (1.0 + (gamma - 1.0)/2.0 * M**2)


def isentropic_p_ratio(M, gamma):
    """p/p0 at Mach M. [S&B] Eq 3-13."""
    return isentropic_T_ratio(M, gamma)**(gamma/(gamma - 1.0))


# ── Verify throat condition: area_ratio(M=1) must be exactly 1 ───────────────
for g in [1.2, 1.3, 1.4]:
    assert abs(area_ratio(1.0, g) - 1.0) < 1e-12, f"A/A*(M=1) != 1 for gamma={g}"

# Plot A/A* and p/p0 vs Mach number
fig_nozzle, axes_n = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d0d1e")
fig_nozzle.suptitle("Isentropic Nozzle Relations — [S&B] Section 3.1", color="white", fontsize=12)

M_arr = np.linspace(0.01, 5.0, 500)
gammas = [1.2, 1.3, 1.4]
clrs   = ["#00bfff", "#7fff00", "#ff6b35"]

for g, clr in zip(gammas, clrs):
    ax = axes_n[0]
    ax.set_facecolor("#111122")
    ax.semilogy([area_ratio(M, g) for M in M_arr], M_arr,
                color=clr, lw=1.5, label=f"γ={g}")
    ax.axhline(1.0, color="white", lw=0.5, alpha=0.3, linestyle="--")
    ax.set_xlabel("Area Ratio A/A*", color="gray")
    ax.set_ylabel("Mach Number M", color="gray")
    ax.set_title("Mach Number vs Area Ratio", color="white", fontsize=10)
    ax.tick_params(colors="gray")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.set_xlim(1, 30)

for g, clr in zip(gammas, clrs):
    ax = axes_n[1]
    ax.set_facecolor("#111122")
    ax.semilogy(M_arr, [isentropic_p_ratio(M, g) for M in M_arr],
                color=clr, lw=1.5, label=f"γ={g}")
    ax.axvline(1.0, color="white", lw=0.5, alpha=0.3, linestyle="--")
    ax.set_xlabel("Mach Number M", color="gray")
    ax.set_ylabel("Pressure Ratio p/p₀", color="gray")
    ax.set_title("Pressure Ratio vs Mach Number", color="white", fontsize=10)
    ax.tick_params(colors="gray")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

plt.tight_layout()
plt.savefig("p4_isentropic.png", dpi=140, facecolor="#0d0d1e", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 4. Characteristic Velocity c* and Thrust Coefficient C_F
#
# > **[S&B] Equation 3-32** — Characteristic velocity c*
# > **[S&B] Equation 3-30** — Thrust coefficient CF
#
# These two quantities decompose engine performance cleanly:
#
#   Isp = CF · c* / g0   [S&B] Section 3.3
#
# c*  depends only on the propellant combination (combustion efficiency).
# CF  depends only on the nozzle geometry and pressure ratio (expansion efficiency).
#
# The Vandenkerckhove function Γ:
#   Γ = sqrt(γ) · (2/(γ+1))^((γ+1)/(2(γ-1)))
#
# c* = sqrt(R_spec · T0) / Γ    where R_spec = R_universal / M_mol

# %% CstarCF
def cstar(T0_K, M_mol_kgkmol, gamma):
    """
    Characteristic velocity c* [m/s].
    [S&B] Equation 3-32: c* = sqrt(R_spec * T0) / Gamma
    Gamma = sqrt(gamma) * (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))

    T0_K          : chamber temperature [K]
    M_mol_kgkmol  : mean molecular weight of exhaust [kg/kmol]
    gamma         : ratio of specific heats (dimensionless)

    Note: c* depends ONLY on combustion quality, NOT on nozzle geometry.
    Higher T0 and lower M_mol → higher c* → higher Isp potential.
    """
    R_spec = R_UNIV / M_mol_kgkmol   # J/(kg·K)
    Gamma  = np.sqrt(gamma) * (2.0/(gamma + 1.0))**((gamma + 1.0)/(2.0*(gamma - 1.0)))
    return np.sqrt(R_spec * T0_K) / Gamma


def thrust_coefficient(gamma, eps, p0_Pa, p_atm_Pa=0.0):
    """
    Thrust coefficient CF = F / (p0 * At).
    [S&B] Equation 3-30

    CF depends ONLY on nozzle geometry (eps) and pressure ratio (p0, p_atm).
    Not on propellant or combustion temperature.

    Returns (CF, M_exit, pe_Pa/p0)
    """
    M_e     = mach_from_area_ratio(eps, gamma, supersonic=True)
    per     = isentropic_p_ratio(M_e, gamma)   # pe/p0
    # Momentum thrust term [S&B] Eq 3-30 first term
    t1      = 2.0*gamma**2/(gamma - 1.0) * (2.0/(gamma + 1.0))**((gamma + 1.0)/(gamma - 1.0))
    CF_mom  = np.sqrt(max(t1 * (1.0 - per**((gamma - 1.0)/gamma)), 0.0))
    # Pressure thrust term (positive in vacuum, can be negative if over-expanded)
    CF_prs  = (per - p_atm_Pa/p0_Pa) * eps
    return CF_mom + CF_prs, M_e, per


def engine_analysis(T0_K, M_mol, gamma, p0_Pa, eps, At_m2, p_atm_Pa=0.0):
    """
    Complete engine performance from chamber conditions and nozzle geometry.
    [S&B] Chapter 3 — combines c* and CF analyses.

    Parameters
    ----------
    T0_K     : chamber temperature [K]
    M_mol    : mean exhaust molecular weight [kg/kmol]
    gamma    : ratio of specific heats
    p0_Pa    : chamber pressure [Pa]
    eps      : area ratio Ae/At (expansion ratio)
    At_m2    : throat area [m^2]
    p_atm_Pa : ambient pressure [Pa] — 0 for vacuum, 101325 for sea level

    Returns dict with all engine performance parameters.
    """
    cs              = cstar(T0_K, M_mol, gamma)
    CF_vac, Me, per = thrust_coefficient(gamma, eps, p0_Pa, 0.0)
    CF_sl,  _,  _   = thrust_coefficient(gamma, eps, p0_Pa, p_atm_Pa)

    mdot     = p0_Pa * At_m2 / cs                # [S&B] Eq 3-24: mdot = p0*At/c*
    T_exit   = T0_K * isentropic_T_ratio(Me, gamma)
    ve_exit  = Me * np.sqrt(gamma * (R_UNIV/M_mol) * T_exit)  # [S&B] Eq 3-16

    return {
        "cstar_ms":   cs,
        "CF_vac":     CF_vac,
        "CF_sl":      CF_sl,
        "M_exit":     Me,
        "pe_Pa":      per * p0_Pa,
        "T_exit_K":   T_exit,
        "ve_exit_ms": ve_exit,
        "F_vac_N":    CF_vac * p0_Pa * At_m2,
        "F_sl_N":     CF_sl  * p0_Pa * At_m2,
        "mdot_kgs":   mdot,
        "Isp_vac_s":  CF_vac * cs / G0,
        "Isp_sl_s":   CF_sl  * cs / G0,
    }

# %% [markdown]
# ---
# ## 5. Engine Performance Comparison
#
# Model three representative liquid rocket engines using chamber conditions from
# literature. The ideal nozzle model gives values within ~5-10% of real performance.
# Real engines differ due to non-ideal gas, boundary layers, and nozzle divergence.
# [S&B] Section 3.4 covers efficiency corrections (η_c*, C_v, C_d, η_F).

# %% EngineModels
# Representative chamber conditions from published data
# Note: These are approximate thermodynamic averages for equilibrium composition
# at the given O/F ratio. Real c* values from [S&B] Table 5-5.

ENGINES = {
    "RP-1/LOX (Merlin 1D-like)": {
        "T0":     3800,       # K   combustion temperature
        "M_mol":  21.5,       # kg/kmol  mean exhaust molecular weight
        "gamma":  1.25,       # specific heat ratio
        "p0":     9.7e6,      # Pa  chamber pressure (97 bar)
        "eps_sl": 16.0,       # sea-level expansion ratio
        "eps_vac":117.0,      # vacuum expansion ratio (Merlin Vacuum)
        "At":     0.1,        # m^2  throat area (scaled to ~100 kN class)
        "color":  "#ff6b35",
        "ref":    "[S&B] Table 5-5, RP-1/LOX representative",
    },
    "LH2/LOX (RL-10-like)": {
        "T0":     3516,
        "M_mol":  11.8,
        "gamma":  1.24,
        "p0":     3.3e6,
        "eps_sl": None,       # vacuum only (Centaur upper stage)
        "eps_vac":61.0,
        "At":     0.05,
        "color":  "#00bfff",
        "ref":    "[S&B] Table 5-5, LH2/LOX representative",
    },
    "LCH4/LOX (Raptor-like)": {
        "T0":     3600,
        "M_mol":  18.6,
        "gamma":  1.22,
        "p0":     30.0e6,     # Pa  full-flow staged combustion, ~300 bar!
        "eps_sl": 40.0,
        "eps_vac":40.0,
        "At":     0.2,
        "color":  "#7fff00",
        "ref":    "[S&B] Table 5-5, methane/LOX representative",
    },
}

print(f"{'Engine':<30}  {'c* (m/s)':>10}  {'Isp_vac (s)':>12}  {'Isp_sl (s)':>11}  {'F_vac (kN)':>11}")
print("-" * 80)

eng_results = {}
for name, cfg in ENGINES.items():
    # Vacuum performance
    ev = engine_analysis(cfg["T0"], cfg["M_mol"], cfg["gamma"],
                         cfg["p0"], cfg["eps_vac"], cfg["At"], 0.0)
    # Sea-level performance (if applicable)
    if cfg["eps_sl"] is not None:
        es = engine_analysis(cfg["T0"], cfg["M_mol"], cfg["gamma"],
                             cfg["p0"], cfg["eps_sl"], cfg["At"], 101_325)
        isp_sl_str = f"{es['Isp_sl_s']:.1f}"
        F_sl_str   = f"{es['F_sl_N']/1e3:.1f}"
    else:
        es = None; isp_sl_str = "vac only"; F_sl_str = "—"

    eng_results[name] = {"vac": ev, "sl": es, **cfg}

    print(f"  {name:<28}  {ev['cstar_ms']:>10.0f}  {ev['Isp_vac_s']:>12.1f}  "
          f"{isp_sl_str:>11}  {ev['F_vac_N']/1e3:>11.1f}")

print()
print("Note: Ideal model differs from real engines by ~5-10% due to:")
print("  - Non-ideal gas (dissociation at T > 3000 K)")
print("  - Frozen (non-equilibrium) flow in nozzle")
print("  - Boundary layer losses, divergence angle correction")
print("  - [S&B] Section 3.4 for efficiency correction factors")

# %% [markdown]
# ---
# ## 6. Nozzle Performance vs Expansion Ratio
#
# Increasing the expansion ratio raises Isp in vacuum but reduces sea-level thrust
# (over-expanded nozzle: pe < pa → flow separation, thrust loss).
# The optimal expansion ratio in atmosphere is pe = pa — pressure-matched.
#
# This trade-off is why Merlin 1D has eps=16 (SL optimised) while
# Merlin Vacuum has eps=117 (vacuum optimised, big bell nozzle).

# %% ExpansionRatioSweep
fig_eps, axes_eps = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d0d1e")
fig_eps.suptitle("Nozzle Performance vs Expansion Ratio ε = Ae/At\n"
                  "[S&B] Section 3.3 — Optimal expansion: pe = pa",
                  color="white", fontsize=12)

eps_arr = np.linspace(2, 200, 300)

for ax_idx, (name, cfg) in enumerate(ENGINES.items()):
    ax  = axes_eps[ax_idx]
    ax.set_facecolor("#111122")
    clr = cfg["color"]

    Isp_vac = []; Isp_sl = []
    for eps in eps_arr:
        ev = engine_analysis(cfg["T0"], cfg["M_mol"], cfg["gamma"],
                             cfg["p0"], eps, cfg["At"], 0.0)
        Isp_vac.append(ev["Isp_vac_s"])
        es = engine_analysis(cfg["T0"], cfg["M_mol"], cfg["gamma"],
                             cfg["p0"], eps, cfg["At"], 101_325)
        Isp_sl.append(es["Isp_sl_s"])

    ax.plot(eps_arr, Isp_vac, color=clr,    lw=1.8, label="Vacuum (pa=0)")
    ax.plot(eps_arr, Isp_sl,  color="gray", lw=1.2, linestyle="--", label="Sea level")

    # Mark optimal SL expansion (pe = pa = 101325)
    for eps_check in eps_arr:
        Me  = mach_from_area_ratio(eps_check, cfg["gamma"])
        pe  = isentropic_p_ratio(Me, cfg["gamma"]) * cfg["p0"]
        if abs(pe - 101_325) < 500:
            ax.axvline(eps_check, color="yellow", lw=0.9, alpha=0.6, linestyle=":")
            ax.text(eps_check+2, min(Isp_sl)+5,
                    f"ε_opt={eps_check:.0f}\n(pe=pa)", color="yellow", fontsize=7)
            break

    # Mark real engine expansion ratios
    if cfg["eps_sl"] is not None:
        ax.axvline(cfg["eps_sl"],  color="lime", lw=0.8, alpha=0.5, linestyle=":")
        ax.text(cfg["eps_sl"]+1, min(Isp_vac)*1.002,
                f"SL: ε={cfg['eps_sl']:.0f}", color="lime", fontsize=7)
    ax.axvline(cfg["eps_vac"], color=clr, lw=0.8, alpha=0.5, linestyle=":")
    ax.text(cfg["eps_vac"]+1, min(Isp_vac)*1.002,
            f"Vac: ε={cfg['eps_vac']:.0f}", color=clr, fontsize=7)

    ax.set_xlabel("Expansion Ratio ε = Ae/At", color="gray")
    ax.set_ylabel("Specific Impulse Isp (s)", color="gray")
    ax.set_title(name, color="white", fontsize=9)
    ax.tick_params(colors="gray")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

plt.tight_layout()
plt.savefig("p4_expansion_ratio.png", dpi=140, facecolor="#0d0d1e", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7. The Tsiolkovsky Rocket Equation
#
# > **[S&B] Section 4.2, p. 114, Equation 4-6**
# > **[BMT 2nd ed.] Section 6.2, p. 249**
#
# dv = Isp * g0 * ln(m0/mf) = ve * ln(m0/mf)
#
# The propellant fraction grows exponentially with dv/ve.
# At the limit: to double the dv, you need the square of the mass ratio.
# This is why launch vehicles are 80-95% propellant by mass.

# %% RocketEquation
def tsiolkovsky_mass_ratio(dv_kms, isp_s):
    """m0/mf = exp(dv/ve). [S&B] Eq 4-6."""
    ve_km = isp_s * G0 * 1e-3   # km/s
    return np.exp(dv_kms / ve_km)

def tsiolkovsky_prop_fraction(dv_kms, isp_s):
    """mp/m0 = 1 - exp(-dv/ve)."""
    ve_km = isp_s * G0 * 1e-3
    return 1.0 - np.exp(-dv_kms / ve_km)


# Mission Δv budget (Earth surface to GEO)
print("Δv Budget: Earth Surface → GEO\n")
MISSIONS = [
    ("LEO insertion (gravity+drag losses+orbital)", 9.3),
    ("Hohmann LEO→GEO (burn 1)",                    2.40),
    ("Hohmann LEO→GEO (burn 2)",                    1.46),
    ("GEO station-keeping (10 years)",               0.50),
    ("Total (surface → GEO operational)",           13.66),
]
for label, dv in MISSIONS:
    print(f"  {label:<45} {dv:.2f} km/s")

print(f"\nPropellant fractions for LEO insertion (dv=9.3 km/s):")
print(f"  {'Engine':<30}  {'Isp (s)':>8}  {'mp/m0':>8}  {'m0/mf':>8}")
print("  " + "-"*55)
for isp, name in [(265,"Solid (SRB)"), (311,"RP-1/LOX (Merlin)"),
                   (380,"LCH4/LOX (Raptor)"), (453,"LH2/LOX (RS-25)")]:
    mr = tsiolkovsky_mass_ratio(9.3, isp)
    pf = tsiolkovsky_prop_fraction(9.3, isp)
    print(f"  {name:<30}  {isp:>8}  {pf:>7.1%}  {mr:>8.2f}")

# Visualize: propellant fraction vs Isp for multiple dv requirements
fig_tsk, ax_tsk = plt.subplots(figsize=(10, 6), facecolor="#0d0d1e")
ax_tsk.set_facecolor("#111122")

isp_range = np.linspace(200, 5000, 400)
dv_missions = [(9.3, "LEO insertion", "#ff6b35"),
               (3.85,"Hohmann LEO→GEO", "#00bfff"),
               (1.5, "Typical upper stage burn", "#7fff00"),
               (0.5, "GEO 10yr stationkeeping", "#ff00ff")]

for dv, label, clr in dv_missions:
    pf = 1 - np.exp(-dv / (isp_range * G0 * 1e-3))
    ax_tsk.plot(isp_range, pf*100, color=clr, lw=1.8, label=f"{label} ({dv} km/s)")

# Mark real engine Isp values
for isp_mark, name_mark, clr_mark in [(265,"Solid","gray"),
                                       (311,"RP-1/LOX","#ff6b35"),
                                       (380,"LCH4/LOX","#7fff00"),
                                       (453,"LH2/LOX","#00bfff"),
                                       (3000,"Ion","yellow")]:
    ax_tsk.axvline(isp_mark, color=clr_mark, lw=0.8, alpha=0.5, linestyle=":")
    ax_tsk.text(isp_mark+30, 92, name_mark, color=clr_mark, fontsize=7,
                rotation=90, va="top")

ax_tsk.set_xlabel("Specific Impulse Isp (s)  [S&B] Section 2.5", color="gray")
ax_tsk.set_ylabel("Propellant fraction mp/m0 (%)", color="gray")
ax_tsk.set_title("Tsiolkovsky Rocket Equation — Propellant Cost of Each Mission\n"
                  "dv = Isp · g0 · ln(m0/mf)  [S&B] Eq 4-6",
                  color="white", fontsize=11)
ax_tsk.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
ax_tsk.tick_params(colors="gray")
ax_tsk.set_xlim(200, 2000); ax_tsk.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("p4_tsiolkovsky.png", dpi=140, facecolor="#0d0d1e", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 8. Staging — Overcoming the Exponential Wall
#
# > **[S&B] Section 4.4, p. 125** — "Multistage Rockets"
#
# A single-stage-to-orbit vehicle needs m0/mf ~ 20 (for RP-1/LOX, dv ~ 9.3 km/s).
# No known structure can hold that mass ratio AND survive launch loads. Staging discards
# empty propellant tanks, letting each stage start with maximum possible mass ratio.
#
# Total dv = sum of each stage's Tsiolkovsky contribution.

# %% Staging
def stage_dv(m0, mf, isp):
    """Delta-v for one stage. m0, mf in any consistent mass units."""
    return isp * G0 * 1e-3 * np.log(m0 / mf)   # km/s

# Model a 3-stage launch vehicle (Falcon 9-ish)
stages = [
    # (name, m_wet kg, m_dry kg, Isp_vac s)
    ("Stage 1 (RP-1/LOX SL)",  549_054, 26_000, 311),   # includes payload+upper
    ("Stage 2 (RP-1/LOX vac)",  95_000,  4_000, 348),   # includes payload
    ("Payload to LEO",           22_800,      0, None),  # delivered to orbit
]

print("3-Stage Launch Vehicle Analysis (Falcon 9-like)\n")
m_current = stages[0][1]  # start with full vehicle mass
total_dv = 0.0
print(f"  {'Stage':<26}  {'m_wet':>8}  {'m_dry':>8}  {'Isp':>6}  {'dv (km/s)':>10}")
print("  " + "-"*64)

for name, m_wet, m_dry, isp in stages[:-1]:
    if isp is not None:
        dv = stage_dv(m_current, m_dry, isp)
        total_dv += dv
        print(f"  {name:<26}  {m_wet:>8,.0f}  {m_dry:>8,.0f}  {isp:>6}  {dv:>10.3f}")
        m_current = m_dry  # next stage starts with dry mass of this stage

print("  " + "-"*64)
print(f"  {'TOTAL':>26}  {'':>8}  {'':>8}  {'':>6}  {total_dv:>10.3f} km/s")
print(f"\n  Structural fraction stage 1: {26000/549054:.3f}  (4.7%)")
print(f"  Structural fraction stage 2: {4000/95000:.3f}  (4.2%)")
print(f"\n  Payload to LEO: {22800/549054*100:.2f}% of initial mass")

# Optimal staging plot: dv split for maximum payload
print("\nOptimal 2-stage split for LEO (dv_total=9.3 km/s):")
print("Equal Isp both stages (RP-1/LOX, Isp=320s), struct_frac=0.05")
Isp_both = 320; sf = 0.05; dv_tot = 9.3; ve = Isp_both*G0*1e-3
# Optimal split: equal mass ratios per stage for same Isp
# => each stage gets dv_tot/2
dv_per = dv_tot/2
mr1 = np.exp(dv_per/ve); mr2 = np.exp(dv_per/ve)
# Payload mass fraction given structural fraction sf and 2 stages
# Stage 2: m0_2/mf_2 = mr2, mf_2 = m_dry_2 + m_payload = sf*m0_2 + m_payload
# => m_payload = mf_2*(1-sf) ... iterative
print(f"  Each stage dv = {dv_per:.3f} km/s,  mass ratio = {mr1:.3f}")
print(f"  Stage 2 payload fraction = {(1/mr2-sf):.4f}")
print(f"  Stage 1 payload fraction = {(1/mr1-sf)*(1/mr2-sf):.4f}  ({(1/mr1-sf)*(1/mr2-sf)*100:.2f}%)")

# %% [markdown]
# ---
# ## 9. Finite Burn Simulation — 7-DOF EOM
#
# > **[S&B] Section 4.2** — Thrust and mass flow
# > **[BMT 2nd ed.] Section 6.1** — Impulsive vs finite burn
#
# Real burns have finite duration. The EOM expands to 7 DOF: [r, v, m].
# The thrust force F/m [km/s^2] where F [kN] = F [kg·km/s^2] by unit analysis.
#
# dm/dt = -mdot [kg/s]  (propellant consumed continuously)
#
# Key: 1 kN = 1000 N = 1000 kg·m/s^2 = 1 kg·km/s^2
# So F_kN [kN] / m_kg [kg] = F_kN [kg·km/s^2] / m_kg [kg] = F_kN/m_kg [km/s^2]

# %% FiniteBurnEOM
def j2_accel(r_vec, mu=MU_EARTH, j2=J2, Re=R_EARTH):
    """J2 acceleration [km/s^2]. [BMT 2nd ed.] Section 9.3."""
    x, y, z = r_vec; r = np.linalg.norm(r_vec)
    fac = (3*mu*j2*Re**2) / (2*r**5); zr2 = (z/r)**2
    return np.array([fac*x*(5*zr2-1), fac*y*(5*zr2-1), fac*z*(5*zr2-3)])


def eom_finite_burn(t, state7, F_kN, mdot_kgs, thrust_dir_fn, mu=MU_EARTH):
    """
    7-DOF equations of motion with variable mass.
    [S&B] Section 4.2, [BMT 2nd ed.] Section 6.1

    State: [x, y, z, vx, vy, vz, m]
    Units: km, km/s, kg

    Unit consistency for thrust:
      F_kN [kN] = F_kN [kg·km/s^2]
      F_kN / m_kg = [km/s^2]  ✓
    """
    r, v, m = state7[:3], state7[3:6], state7[6]
    rm = np.linalg.norm(r)
    a_thrust = (F_kN / m) * thrust_dir_fn(t, state7)
    a_grav   = -(mu/rm**3)*r + j2_accel(r, mu)
    return np.concatenate([v, a_grav + a_thrust, [-mdot_kgs]])


def _rk4_step_7(f, t, y, h, *args):
    k1=f(t,y,*args); k2=f(t+h/2,y+(h/2)*k1,*args)
    k3=f(t+h/2,y+(h/2)*k2,*args); k4=f(t+h,y+h*k3,*args)
    return y+(h/6)*(k1+2*k2+2*k3+k4)


def propagate_finite_burn(state0_7, t_total, dt, F_kN, mdot_kgs,
                          thrust_dir_fn, m_dry=1.0, mu=MU_EARTH):
    """
    Propagate finite burn with linear interpolation at propellant cutoff.
    Stops when mass reaches m_dry (dry mass after all propellant consumed).

    state0_7 : [x, y, z, vx, vy, vz, m_wet]
    Returns dict: t, r, v, m arrays + Tsiolkovsky dv delivered
    """
    n  = int(t_total / dt)
    ta = np.zeros(n + 1)
    st = np.zeros((n + 1, 7))
    ta[0] = 0.0; st[0] = state0_7.copy()

    for i in range(n):
        sn = _rk4_step_7(eom_finite_burn, ta[i], st[i], dt,
                         F_kN, mdot_kgs, thrust_dir_fn, mu)
        ta[i+1] = ta[i] + dt; st[i+1] = sn

        if sn[6] <= m_dry:
            # Linear interpolation: find exact moment mass = m_dry
            frac = (st[i][6] - m_dry) / (st[i][6] - sn[6])
            s_interp = st[i] + frac * (sn - st[i])
            ta[i+1] = ta[i] + frac * dt; st[i+1] = s_interp
            break

    last = i + 2
    m0   = state0_7[6]
    mf   = st[last-1][6]
    ve_km = (F_kN * 1e3 / mdot_kgs) / G0 * 1e-3   # ve = F/mdot / g0... no
    # Actually: ve = Isp * g0 = (F/(mdot*g0)) * g0 = F/mdot
    # F [kN] = F*1000 [N], mdot [kg/s] -> ve = F_N / mdot = F_kN*1000/mdot [m/s] -> /1000 [km/s]
    ve_km_correct = F_kN * 1000 / (mdot_kgs * 1000)  # simplified: F_kN/mdot [km/s? No...]
    # Correct: ve [m/s] = F [N] / mdot [kg/s] = F_kN*1000 / mdot
    # ve_km = ve_m / 1000 = F_kN / mdot
    ve_kms = F_kN / mdot_kgs    # km/s  (F_kN [kN] / mdot [kg/s] = [kN/kg] = [1000m/s / 1] ??? )
    # Wait: F_kN [kN = 1000 N = 1000 kg m/s^2]
    # ve = F/mdot [m/s] = F_kN*1000 / mdot [m/s]
    # ve_km = F_kN*1000 / mdot / 1000 = F_kN / mdot [km/s]
    # So ve_kms = F_kN / mdot_kgs [km/s] ✓
    dv_delivered = ve_kms * np.log(m0 / mf)

    return {
        "t": ta[:last], "r": st[:last, :3],
        "v": st[:last, 3:6], "m": st[:last, 6],
        "dv_tsiolkovsky_kms": dv_delivered,
    }


def thrust_prograde(t, state7):
    """Thrust always in the prograde (velocity) direction."""
    v = state7[3:6]; return v / np.linalg.norm(v)

def orbital_period(a, mu=MU_EARTH):
    return 2*np.pi*np.sqrt(a**3/mu)

print("Finite burn functions ready.")
print("  eom_finite_burn()           — 7-DOF EOM  [S&B §4.2, BMT §6.1]")
print("  propagate_finite_burn()     — RK4 + linear mass interpolation")
print("  thrust_prograde()           — prograde thrust direction")

# %% [markdown]
# ---
# ## 10. Finite Burn Simulation — Hohmann Burn 1 (LEO → GTO)
#
# We simulate the first Hohmann burn (LEO perigee → start of transfer ellipse)
# for a 5000 kg spacecraft with a 100 kN RP-1/LOX engine (Isp=311s).
#
# The burn lasts ~83 seconds — about 1.5% of the LEO orbital period.
# We compare the finite-burn trajectory to the ideal impulsive case.
#
# **Validation:** The Tsiolkovsky dv delivered = ve * ln(m0/mf)
# This is exact by construction (not affected by gravity losses).
# The gravity loss (difference between dv_Tsiol and the actual orbit energy change)
# is naturally small for such short burns (~0.1% of dv for chemical engines).

# %% FiniteBurnHohmann
# ── Setup ─────────────────────────────────────────────────────────────────────
a_leo  = R_EARTH + 408.0
a_geo  = R_EARTH + 35_786.0
a_t    = (a_leo + a_geo) / 2.0
v_c    = np.sqrt(MU_EARTH / a_leo)                   # LEO circular speed
v_tp   = np.sqrt(MU_EARTH * (2/a_leo - 1/a_t))       # transfer ellipse periapsis speed
dv1_impulsive = v_tp - v_c                            # impulsive Hohmann dv1

# Engine parameters (RP-1/LOX, Merlin 1D-like)
Isp_burn = 311.0                                      # s
F_burn   = 100.0                                      # kN
ve_burn  = F_burn / (F_burn * 1000 / (Isp_burn * G0)) / 1000  # km/s
# Simplified: ve_km = Isp * G0 / 1000
ve_burn  = Isp_burn * G0 * 1e-3                       # km/s
mdot     = F_burn * 1e3 / (Isp_burn * G0)             # kg/s

m0_sc    = 5000.0                                     # kg wet mass
mf_sc    = m0_sc * np.exp(-dv1_impulsive / ve_burn)  # target dry mass after burn
mp_sc    = m0_sc - mf_sc                             # propellant mass
t_burn   = mp_sc / mdot                              # burn time

print("Finite Burn Analysis — Hohmann Burn 1 (LEO → GTO Periapsis)")
print(f"  Spacecraft:  m0={m0_sc:.0f} kg,  mf={mf_sc:.1f} kg,  mp={mp_sc:.1f} kg")
print(f"  Engine:      F={F_burn:.0f} kN,  Isp={Isp_burn} s,  mdot={mdot:.2f} kg/s")
print(f"  Target dv:   {dv1_impulsive:.5f} km/s  (impulsive Hohmann)")
print(f"  Burn time:   {t_burn:.2f} s = {t_burn/60:.2f} min")
print(f"  Orbit fraction burned: {t_burn / orbital_period(a_leo) * 100:.2f}%")

def orbital_period(a, mu=MU_EARTH):
    return 2*np.pi*np.sqrt(a**3/mu)

# ── Simulate finite burn ───────────────────────────────────────────────────────
# Satellite starts at ascending node: r = [a_leo, 0, 0], v = [0, v_c, 0]
s0_fb = np.array([a_leo, 0.0, 0.0,   0.0, v_c, 0.0,   m0_sc])

tr_fb = propagate_finite_burn(
    s0_fb, t_burn * 1.1, 0.5,
    F_burn, mdot, thrust_prograde, m_dry=mf_sc
)

print(f"\n  dv Tsiolkovsky delivered: {tr_fb['dv_tsiolkovsky_kms']:.6f} km/s")
print(f"  Error vs impulsive:        {abs(tr_fb['dv_tsiolkovsky_kms']-dv1_impulsive)*1000:.4f} m/s")
print(f"  Final mass:                {tr_fb['m'][-1]:.3f} kg  (target {mf_sc:.3f} kg)")

# ── Compare orbital energy before/after ───────────────────────────────────────
def specific_energy(r_vec, v_vec, mu=MU_EARTH):
    return 0.5*np.linalg.norm(v_vec)**2 - mu/np.linalg.norm(r_vec)

eps_before = specific_energy(tr_fb["r"][0],  tr_fb["v"][0])
eps_after  = specific_energy(tr_fb["r"][-1], tr_fb["v"][-1])
a_actual   = -MU_EARTH / (2*eps_after)
ra_actual  = 2*a_actual - a_leo  # apoapsis of the ellipse achieved

print(f"\n  Orbital energy before burn: {eps_before:.4f} km^2/s^2")
print(f"  Orbital energy after burn:  {eps_after:.4f} km^2/s^2")
print(f"  Resulting semi-major axis:  {a_actual:.1f} km  (target {a_t:.1f} km)")
print(f"  Resulting apoapsis altitude: {ra_actual - R_EARTH:.1f} km  (target {R_EARTH+35786 - R_EARTH:.1f} km)")

# %% [markdown]
# ---
# ## 11. Finite vs Impulsive Trajectory Visualization

# %% FiniteBurnPlot
# Propagate both: full Hohmann (impulsive) and finite-burn + coast

def orbital_period(a, mu=MU_EARTH):   # already defined above — redefined here for cell independence
    return 2*np.pi*np.sqrt(a**3/mu)

def eom_twobody(t, s6, mu=MU_EARTH):
    r=s6[:3];v=s6[3:];rm=np.linalg.norm(r)
    return np.concatenate([v,-(mu/rm**3)*r])

def rk4_step_6(f,t,y,h,*a):
    k1=f(t,y,*a);k2=f(t+h/2,y+(h/2)*k1,*a)
    k3=f(t+h/2,y+(h/2)*k2,*a);k4=f(t+h,y+h*k3,*a)
    return y+(h/6)*(k1+2*k2+2*k3+k4)

def propagate_twobody(s0, t_span, dt, mu=MU_EARTH):
    t0,tf=t_span; n=int((tf-t0)/dt); ta=np.zeros(n+1); st=np.zeros((n+1,6))
    ta[0]=t0; st[0]=s0.copy()
    for i in range(n): st[i+1]=rk4_step_6(eom_twobody,ta[i],st[i],dt,mu); ta[i+1]=ta[i]+dt
    return {"t":ta,"r":st[:,:3],"v":st[:,3:]}

# ── Impulsive Hohmann ─────────────────────────────────────────────────────────
T_transfer = orbital_period(a_t)

# Start state
s_leo_6 = np.array([a_leo, 0., 0.,   0., v_c, 0.])

# Impulsive burn 1: add dv prograde
s_after1_imp = s_leo_6.copy(); s_after1_imp[4] += dv1_impulsive
tr_transfer_imp = propagate_twobody(s_after1_imp, (0, T_transfer/2), 10.0)

# ── Finite burn + coast ───────────────────────────────────────────────────────
# After finite burn, coast on the achieved ellipse to apoapsis
s_after_fb_6 = np.concatenate([tr_fb["r"][-1], tr_fb["v"][-1]])
# Estimate time to apoapsis (roughly T_transfer/2 - t_burn/2)
t_coast = T_transfer/2 - t_burn/2
tr_coast_fb = propagate_twobody(s_after_fb_6, (0, t_coast), 10.0)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig_fb, axes_fb = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0d0d1e")
fig_fb.suptitle("Finite Burn vs Impulsive — Hohmann Transfer Initiation\n"
                 "F=100 kN, Isp=311 s, m₀=5000 kg, burn ~83 s (1.5% of orbit)",
                 color="white", fontsize=12)

# 2D orbital view
ax1 = axes_fb[0]; ax1.set_facecolor("#111122"); ax1.set_aspect("equal")
ec = plt.Circle((0,0), R_EARTH, color="deepskyblue", alpha=0.3)
ax1.add_patch(ec)

# LEO arc (pre-burn)
th_leo = np.linspace(-np.pi/6, 0, 50)
ax1.plot(a_leo*np.cos(th_leo), a_leo*np.sin(th_leo),
         color="#444466", lw=1.2, alpha=0.6, label="LEO (pre-burn)")

# Impulsive transfer ellipse
ax1.plot(tr_transfer_imp["r"][:,0], tr_transfer_imp["r"][:,1],
         color="#ffff00", lw=2.0, linestyle="--", label="Impulsive Hohmann", alpha=0.8)

# Finite burn arc
ax1.plot(tr_fb["r"][:,0], tr_fb["r"][:,1],
         color="#ff6b35", lw=2.5, label=f"Finite burn ({t_burn:.0f} s)", zorder=5)

# Coast after finite burn
ax1.plot(tr_coast_fb["r"][:,0], tr_coast_fb["r"][:,1],
         color="#ff6b35", lw=1.5, linestyle=":", label="Coast to GEO", alpha=0.7)

# Markers
ax1.scatter(*s_after1_imp[:2], color="#ffff00", s=100, zorder=8, marker="*",
            label="Burn 1 (impulsive)")
ax1.scatter(*tr_fb["r"][0,:2],  color="#ff6b35", s=80, zorder=8)
ax1.scatter(*tr_transfer_imp["r"][-1,:2], color="lime",    s=100, zorder=8,
            marker="^", label=f"GEO apoapsis (imp.)")
ax1.scatter(*tr_coast_fb["r"][-1,:2],     color="#ffa040", s=100, zorder=8,
            marker="^", label=f"GEO apoapsis (finite)")

lim = 46_000
ax1.set_xlim(-lim*0.3, lim); ax1.set_ylim(-lim*0.5, lim*0.5)
ax1.set_xlabel("X (km)", color="gray"); ax1.set_ylabel("Y (km)", color="gray")
ax1.set_title("Orbital View (XY plane)", color="white")
ax1.tick_params(colors="gray")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, loc="upper left")

# Mass history during burn
ax2 = axes_fb[1]; ax2.set_facecolor("#111122")
t_s = tr_fb["t"]
ax2.plot(t_s, tr_fb["m"], color="#ff6b35", lw=2.0, label="Spacecraft mass")
ax2.axhline(mf_sc, color="lime",   lw=0.8, linestyle="--",
            label=f"Target dry mass {mf_sc:.0f} kg")
ax2.axhline(m0_sc, color="gray",   lw=0.8, linestyle="--", alpha=0.5)

ax2_twin = ax2.twinx()
F_instant = F_burn / tr_fb["m"] * tr_fb["m"] * 0 + F_burn  # constant F
ax2_twin.plot(t_s, F_burn * np.ones(len(t_s)), color="#00bfff",
              lw=1.2, linestyle=":", alpha=0.7)
ax2_twin.set_ylabel("Thrust F (kN)", color="#00bfff"); ax2_twin.tick_params(colors="#00bfff")

ax2.set_xlabel("Time (s)", color="gray")
ax2.set_ylabel("Mass (kg)", color="gray")
ax2.set_title(f"Mass Depletion During Finite Burn\n"
               f"mdot = {mdot:.2f} kg/s  →  {mp_sc:.0f} kg consumed in {t_burn:.0f} s",
               color="white", fontsize=9)
ax2.tick_params(colors="gray")
ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

plt.tight_layout()
plt.savefig("p4_finite_burn.png", dpi=140, facecolor="#0d0d1e", bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 12. Summary Reference Table
#
# | Function | Equation | Reference |
# |----------|----------|-----------|
# | area_ratio(M,γ) | A/A* from Mach number | [S&B] Eq 3-22 |
# | mach_from_area_ratio(ε,γ) | Newton-Raphson inversion | [S&B] Section 3.3 |
# | isentropic_T_ratio(M,γ) | T/T0 = (1+(γ-1)/2·M²)⁻¹ | [S&B] Eq 3-12 |
# | isentropic_p_ratio(M,γ) | p/p0 = (T/T0)^(γ/(γ-1)) | [S&B] Eq 3-13 |
# | cstar(T0,M_mol,γ) | c* = √(R·T0)/Γ | [S&B] Eq 3-32 |
# | thrust_coefficient(γ,ε,p0,pa) | CF = F/(p0·At) | [S&B] Eq 3-30 |
# | engine_analysis(...) | Full nozzle performance | [S&B] Chapter 3 |
# | tsiolkovsky_mass_ratio(dv,Isp) | m0/mf = exp(dv/ve) | [S&B] Eq 4-6 |
# | eom_finite_burn(...) | 7-DOF [r,v,m] EOM | [S&B] §4.2, [BMT] §6.1 |
# | propagate_finite_burn(...) | RK4 + mass interpolation | — |
#
# **Key engine parameters (ideal model):**
#
# | Engine | γ | M_mol | T0 (K) | c* (m/s) | Isp_vac (s) |
# |--------|---|-------|--------|----------|-------------|
# | RP-1/LOX | 1.25 | 21.5 | 3800 | 1842 | 330 |
# | LH2/LOX  | 1.24 | 11.8 | 3516 | 2399 | 458 |
# | LCH4/LOX | 1.22 | 18.6 | 3600 | 1944 | 370 |
#
# **Note:** Real engines differ by ~5-10% due to non-ideal gas, frozen flow,
# boundary layer losses. See [S&B] Section 3.4 for efficiency corrections.
#
# **Phase 5 → phase5_astropy.py**
# Real ephemerides (JPL Horizons), coordinate frames (ICRS, GCRS, ITRS),
# solar system bodies, astropy Time and SkyCoord.
# References: astropy docs + [BMT 2nd ed.] Chapter 2
