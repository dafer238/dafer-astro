# Theory — Module 4: Rocket Propulsion Systems

> **References:**
> - **[S&B]** Sutton & Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017
> - **[BMT]** Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020

---

## 1. Thrust — The Fundamental Equation

> **[S&B] Section 2.1, p. 27** — "Thrust"

A rocket engine expels mass at high velocity to generate thrust by Newton's third law.
The complete thrust equation accounts for both the momentum of the exhaust and the
pressure difference at the nozzle exit:

$$\boxed{F = \dot{m}\,v_e + (p_e - p_a)\,A_e}$$

Where:
- $\dot{m}$ = propellant mass flow rate [kg/s]
- $v_e$ = exhaust velocity at nozzle exit [m/s]
- $p_e$ = static pressure at nozzle exit [Pa]
- $p_a$ = ambient (atmospheric) pressure [Pa]
- $A_e$ = nozzle exit area [m²]

The **effective exhaust velocity** $c$ absorbs both terms:

$$c = v_e + \frac{(p_e - p_a)\,A_e}{\dot{m}}$$

In vacuum ($p_a = 0$): $c = v_e + p_e A_e/\dot{m}$ — always higher than at sea level,
which is why engines are more efficient in space.

---

## 2. Specific Impulse

> **[S&B] Section 2.5, p. 42** — "Specific Impulse"

The **specific impulse** $I_{sp}$ is the thrust per unit weight flow rate of propellant:

$$\boxed{I_{sp} = \frac{F}{\dot{m}\,g_0} = \frac{c}{g_0}} \quad \text{[seconds]}$$

Where $g_0 = 9.80665\ \text{m/s}^2$. This is the universal efficiency metric:

- Independent of engine scale (same Isp for a toy thruster and a huge booster if the propellants and design are the same)
- Higher Isp → less propellant burned for the same Δv
- The factor $g_0$ makes it dimensionless in the "seconds" sense: Isp is the time a 1 kg engine can produce 1 N of thrust on 1 kg of propellant

**Effective exhaust velocity:**
$$v_e = I_{sp} \cdot g_0 \quad \text{[m/s]}$$

This is the quantity that appears directly in the Tsiolkovsky equation.

---

## 3. The Tsiolkovsky Rocket Equation

> **[S&B] Section 4.2, p. 114, Equation 4-6**
> **[BMT 2nd ed.] Section 6.2, p. 249**

Integrating the thrust equation with decreasing mass:

$$\boxed{\Delta v = I_{sp}\,g_0\,\ln\!\frac{m_0}{m_f} = v_e\,\ln\!\frac{m_0}{m_f}}$$

The **mass ratio** $r = m_0/m_f = e^{\Delta v/v_e}$ and **propellant fraction**:

$$\frac{m_p}{m_0} = 1 - e^{-\Delta v/v_e}$$

The exponential is brutal: to double the Δv, you need $r^2$ mass ratio — not $2r$.

### Gravity Losses

> **[S&B] Section 4.3, p. 117** — "Flight Performance"

The Tsiolkovsky equation applies in free space (no external forces). For a real
orbital burn, gravity acts during the finite burn time. The **gravity loss** is:

$$\Delta v_\text{loss} = \int_0^{t_b} g \sin\alpha\, dt$$

where $\alpha$ is the thrust vector angle from horizontal. For a tangential (prograde)
burn from a circular orbit with burn fraction $t_b/T \ll 1$, gravity loss is:

$$\Delta v_\text{loss} \approx \frac{1}{2}\,g_\text{orbit}\,t_b \cdot \frac{\pi}{T} \cdot t_b$$

For short burns (ISS reboosting: $t_b \sim 10$ min, $T \sim 92$ min), gravity losses
are typically less than 1% — the impulsive approximation is excellent.

---

## 4. De Laval Nozzle Thermodynamics

> **[S&B] Chapter 3** — "Nozzle Theory and Thermodynamic Relations"

The nozzle converts thermal energy (hot, high-pressure combustion products) into
directed kinetic energy. The **de Laval (converging-diverging) nozzle** achieves
supersonic flow after a sonic throat.

### Isentropic Flow Relations

> **[S&B] Section 3.1, Equations 3-12 to 3-16**

For an ideal gas with ratio of specific heats $\gamma$ at Mach number $M$:

$$\frac{T}{T_0} = \left(1 + \frac{\gamma-1}{2}M^2\right)^{-1}$$

$$\frac{p}{p_0} = \left(1 + \frac{\gamma-1}{2}M^2\right)^{-\gamma/(\gamma-1)}$$

$$\frac{\rho}{\rho_0} = \left(1 + \frac{\gamma-1}{2}M^2\right)^{-1/(\gamma-1)}$$

The subscript $0$ denotes **stagnation** (total) conditions in the combustion chamber.

### Critical Conditions at the Throat

At the throat, $M = 1$ exactly (sonic condition). The critical temperature and pressure:

$$T^* = \frac{2}{\gamma+1}T_0, \quad p^* = p_0\left(\frac{2}{\gamma+1}\right)^{\gamma/(\gamma-1)}$$

For air ($\gamma = 1.4$): $p^*/p_0 = 0.528$ — that's why supersonic nozzles require a
chamber-to-ambient pressure ratio of at least ~1.9 to "start" (go supersonic).

### Area-Mach Relation

> **[S&B] Equation 3-22**

$$\frac{A}{A^*} = \frac{1}{M}\left[\frac{2}{\gamma+1}\left(1 + \frac{\gamma-1}{2}M^2\right)\right]^{(\gamma+1)/(2(\gamma-1))}$$

This equation has **two solutions** for each area ratio $A/A^* > 1$:
- $M < 1$: subsonic (converging section)
- $M > 1$: supersonic (diverging section)

For a given area ratio $\varepsilon = A_e/A^*$, the exit Mach number is found by
inverting this equation numerically (Newton-Raphson).

### Characteristic Velocity $c^*$

> **[S&B] Equation 3-32**

The characteristic velocity measures combustion efficiency:

$$\boxed{c^* = \frac{p_0\,A^*}{\dot{m}} = \frac{\sqrt{R_\text{spec}\,T_0}}{\Gamma}}$$

$$\Gamma = \sqrt{\gamma}\left(\frac{2}{\gamma+1}\right)^{(\gamma+1)/(2(\gamma-1))}$$

Where $R_\text{spec} = R_\text{universal}/M_\text{mol}$ is the specific gas constant.
$c^*$ depends only on the propellant combination and combustion temperature, not on the
nozzle geometry. Typical values: 1500–1900 m/s for liquid propellants.

### Thrust Coefficient $C_F$

> **[S&B] Equation 3-30**

$$C_F = \frac{F}{p_0\,A^*}$$

$$C_F = \sqrt{\frac{2\gamma^2}{\gamma-1}\left(\frac{2}{\gamma+1}\right)^{(\gamma+1)/(\gamma-1)}\!\left[1 - \left(\frac{p_e}{p_0}\right)^{(\gamma-1)/\gamma}\right]} + \frac{p_e - p_a}{p_0}\,\varepsilon$$

$C_F$ depends only on the nozzle geometry and operating pressure ratio, not on the propellant.
Typical values: 1.6–2.0.

### The Performance Identity

> **[S&B] Section 3.3**

$$\boxed{I_{sp} = \frac{C_F \cdot c^*}{g_0}}$$

This cleanly separates nozzle efficiency ($C_F$) from combustion efficiency ($c^*$).
To maximize $I_{sp}$: improve combustion temperature and molecular weight (increases $c^*$),
and increase expansion ratio (increases $C_F$ up to a limit).

---

## 5. Engine Staging

> **[S&B] Section 4.4, p. 125** — "Multistage Rockets"

For a single-stage rocket to reach orbit ($\Delta v \sim 9.3$ km/s), the mass ratio required is:

$$r = e^{\Delta v/v_e} = e^{9300/3050} \approx 20.7 \quad \text{(RP-1/LOX, } v_e = 3050\ \text{m/s)}$$

This means 19.7 kg of propellant per 1 kg of structure+payload — structurally impossible
with any known material.

**Staging** solves this by discarding empty hardware at each stage separation. The total
$\Delta v$ is the sum of each stage's $\Delta v$:

$$\Delta v_\text{total} = \sum_{k=1}^{N} I_{sp,k}\,g_0\,\ln\frac{m_{0,k}}{m_{f,k}}$$

The **optimal mass distribution** across stages (for maximum payload fraction with equal
$I_{sp}$ per stage and equal structural fraction) gives equal mass ratios per stage.
For unequal $I_{sp}$: heavier propellant (lower $I_{sp}$) goes in earlier stages —
the dense and cheap first stage uses kerosene/LOX, the vacuum upper stage uses LH₂/LOX.

---

## 6. Finite Burn Effects

> **[BMT 2nd ed.] Section 6.1** — "Impulsive Maneuver Approximation"

Real burns are finite, not instantaneous. Key differences from the impulsive model:

1. **Position changes** during the burn — the satellite moves along its orbit
2. **Thrust direction rotates** to stay prograde (or desired direction)
3. **Gravity acts** throughout the burn, causing gravity losses
4. **Mass decreases** continuously — the equation of motion has 7 DOF: $[\vec{r}, \vec{v}, m]$

The 7-DOF EOM:
$$\frac{d\vec{r}}{dt} = \vec{v}, \quad
\frac{d\vec{v}}{dt} = -\frac{\mu}{r^3}\vec{r} + \frac{F}{m}\hat{T}, \quad
\frac{dm}{dt} = -\dot{m}$$

where $\hat{T}$ is the thrust direction unit vector and $F/m$ [kN/kg = km/s²].

The **impulsive approximation** is valid when $t_b \ll T_\text{orbit}$, which holds for
all chemical engines on typical spacecraft. Electric propulsion burns for weeks and
requires continuous-thrust trajectory optimization.

---

## 7. Representative Engine Parameters

| Engine | Propellant | $I_{sp}$ vac (s) | $c^*$ (m/s) | $\varepsilon$ |
|--------|-----------|-----------------|------------|--------------|
| Merlin 1D | RP-1/LOX | 311 (SL) / 340 (vac) | ~1820 | 16 / 117 |
| Raptor 2 | LCH₄/LOX | 356 (SL) / 380 (vac) | ~1850 | 40 |
| RS-25 (SSME) | LH₂/LOX | 366 (SL) / 453 (vac) | ~2350 | 69 |
| RL-10C | LH₂/LOX | — / 465 (vac) | ~2390 | 84 |
| Rocketdyne RD-107 | Kerosene/LOX | 256 (SL) / 320 (vac) | ~1755 | 18.9 |

> **Note:** The ideal nozzle model presented here gives values within ~5–10% of real
> performance. Real engines differ due to non-ideal gas behavior at high temperatures
> (dissociation, real gas equations of state), non-equilibrium (frozen) flow in the nozzle,
> boundary layer losses (~1–2%), and nozzle divergence angle correction (~1–3%).
> See **[S&B] Section 3.4** for efficiency corrections ($\eta_c$, $C_v$, $C_d$, $\eta_F$).

---

## References

| Tag | Citation |
|-----|---------|
| **[S&B]** | Sutton, Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017. ISBN 978-1-118-75388-0 |
| **[BMT]** | Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020. ISBN 978-0-486-49704-4 |
