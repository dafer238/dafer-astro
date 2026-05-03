# Theory — Module 2: Orbital Maneuvers

> **Primary references:**
> - **[BMT]** Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020
> - **[S&B]** Sutton & Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017

---

## 1. The Concept of Delta-v

> **[BMT 2nd ed.] Section 6.1** — Introduction to Orbital Maneuvers

**Delta-v** ($\Delta v$) is the signed change in velocity required for a maneuver. It is the
universal currency of spaceflight: every trajectory change has a $\Delta v$ cost, and every
kilogram of propellant you carry can provide a finite $\Delta v$ budget.

The **impulsive burn approximation** treats any rocket burn as instantaneous:

$$\vec{v}^+ = \vec{v}^- + \Delta\vec{v}$$

This is valid when the burn time is much shorter than the orbital period. For a chemical
engine burning for seconds to minutes on a 90-minute LEO orbit, it is excellent. For
low-thrust electric propulsion (burn times comparable to orbital period), it is not — that
requires continuous-thrust trajectory optimization (Phase 6).

**Key insight:** only the velocity changes during an impulsive burn. Position is unchanged.
Therefore, the *current* orbit and the *target* orbit share exactly one point — the burn
location. This constraint defines where burns must occur.

---

## 2. Specific Energy Change

> **[BMT 2nd ed.] Section 6.2, p. 248**

After a burn, the new specific energy is:

$$\varepsilon^+ = \frac{|\vec{v}^- + \Delta\vec{v}|^2}{2} - \frac{\mu}{r}$$

For a tangential burn (burn direction parallel to velocity):

$$\varepsilon^+ = \frac{(v^- + \Delta v)^2}{2} - \frac{\mu}{r}$$

This is the most efficient direction for changing orbit size — every km/s of $\Delta v$ goes
entirely into changing energy rather than wasting any on changing direction.

A **prograde burn** (same direction as velocity) raises the opposite side of the orbit.
A **retrograde burn** lowers the opposite side.

> This is the single most important operational fact in astrodynamics: burns affect the
> *other* side of the orbit, not the burn point.

---

## 3. Hohmann Transfer

> **[BMT 2nd ed.] Section 6.3** — "The Hohmann Transfer"

The Hohmann transfer is the minimum-energy two-impulse transfer between two **coplanar
circular** orbits. It uses a single transfer ellipse tangent to both circles at periapsis
and apoapsis.

### Geometry

The transfer ellipse has:
$$a_t = \frac{r_1 + r_2}{2}$$

### Delta-v Calculation

> **[BMT 2nd ed.] Section 6.3, Equations 6.3-1 through 6.3-3**

At departure (burn 1, at periapsis of transfer ellipse):
$$\Delta v_1 = v_{t,p} - v_{c1} = \sqrt{\frac{\mu}{r_1}}\left(\sqrt{\frac{2r_2}{r_1+r_2}} - 1\right)$$

At arrival (burn 2, at apoapsis of transfer ellipse):
$$\Delta v_2 = v_{c2} - v_{t,a} = \sqrt{\frac{\mu}{r_2}}\left(1 - \sqrt{\frac{2r_1}{r_1+r_2}}\right)$$

Both burns are prograde for an outward transfer ($r_2 > r_1$).

### Transfer Time

Half the period of the transfer ellipse:
$$\Delta t = \frac{T_t}{2} = \pi\sqrt{\frac{a_t^3}{\mu}}$$

### Optimality Conditions

> **[BMT 2nd ed.] Section 6.3**

Hohmann is optimal (minimum $\Delta v$ for two-impulse circular-to-circular transfer) when
$r_2/r_1 \leq 11.94$. For ratios above this, a **bi-elliptic transfer** with three burns
can be more efficient.

**Physical intuition:** the Hohmann transfer exploits the fact that adding speed at one
point of an orbit raises the *opposite* point. You add the minimum energy to raise the
apoapsis to $r_2$, coast half an orbit, then add just enough to circularize.

---

## 4. Bi-Elliptic Transfer

> **[BMT 2nd ed.] Section 6.4** — "Bi-elliptic Transfers"

Uses three burns and an intermediate orbit that overshoots the target. More efficient than
Hohmann when $r_2/r_1 > 11.94$ (proven analytically), and sometimes when $r_2/r_1 > 6.73$
depending on the intermediate radius $r_b$.

### Delta-v Equations

$$\Delta v_1 = \sqrt{\frac{2\mu r_b}{r_1(r_1+r_b)}} - \sqrt{\frac{\mu}{r_1}}$$

$$\Delta v_2 = \sqrt{\frac{\mu}{r_b}}\left(\sqrt{\frac{2r_2}{r_b+r_2}} - \sqrt{\frac{2r_1}{r_b+r_1}}\right)$$

$$\Delta v_3 = \sqrt{\frac{\mu}{r_2}} - \sqrt{\frac{2\mu r_b}{r_2(r_2+r_b)}}$$

### Trade-off

Bi-elliptic saves $\Delta v$ but takes much longer (can be 2–3× the transfer time of Hohmann).
The practical choice depends on mission constraints.

---

## 5. Plane Changes

> **[BMT 2nd ed.] Section 6.5** — "Out-of-Plane Orbit Changes"

Changing the orbital plane requires rotating the velocity vector. For a pure plane change
(no altitude change), the required $\Delta v$ is:

$$\Delta v = 2 v_\text{orb} \sin\!\left(\frac{\Delta i}{2}\right)$$

This is derived from the law of cosines: the old and new velocity vectors have the same
magnitude $v_\text{orb}$, and the angle between them is $\Delta i$.

**Key numbers:**

| $\Delta i$ | $\Delta v / v_\text{orb}$ | At LEO ($v_c \approx 7.7$ km/s) |
|---|---|---|
| 10° | 0.174 | 1.34 km/s |
| 28.5° | 0.490 | 3.77 km/s |
| 45° | 0.765 | 5.89 km/s |
| 90° | $\sqrt{2}$ | 10.9 km/s |
| 180° | 2 | 15.4 km/s (entire velocity reversed!) |

> **[BMT 2nd ed.] Section 6.5, p. 263** — A 28.5° plane change at LEO costs MORE than
> a complete Hohmann LEO→GEO transfer (3.85 km/s). This is why most launches go to
> equatorial orbits from tropical launch sites when possible.

### Combined Maneuver (Efficiency)

The most efficient combined plane change + altitude change is to do both simultaneously
at the highest velocity point (lowest orbit):

$$\Delta v_\text{combined} = \sqrt{v_1^2 + v_2^2 - 2v_1 v_2 \cos(\Delta i)}$$

This is always cheaper than doing them sequentially.

---

## 6. The Rocket Equation

> **[S&B] Section 4.2, pp. 114-118** — "Rocket Flight Performance"
> **[BMT 2nd ed.] Section 6.2, p. 249** — Delta-v budget

The **Tsiolkovsky Rocket Equation** converts $\Delta v$ into propellant mass:

$$\boxed{\Delta v = v_e \ln\frac{m_0}{m_f} = I_{sp}\,g_0 \ln\frac{m_0}{m_f}}$$

where:
- $m_0$ = initial (wet) mass including propellant
- $m_f$ = final (dry) mass after burn
- $v_e = I_{sp} \cdot g_0$ = effective exhaust velocity [km/s]
- $I_{sp}$ = specific impulse [s] — the universal propulsion efficiency metric
- $g_0 = 9.80665 \times 10^{-3}$ km/s² (standard gravity)

The **mass ratio** $\xi = m_0/m_f = e^{\Delta v/v_e}$ and the **propellant fraction**:

$$\frac{m_p}{m_0} = 1 - e^{-\Delta v/v_e}$$

> **[S&B] Section 2.1, p. 27** — Specific impulse $I_{sp}$ is defined as thrust per unit
> weight flow rate of propellant. It is the specific impulse of a rocket engine regardless
> of its physical size — the efficiency metric independent of scale.

### Specific Impulse Values

> **[S&B] Table 5-5 and Chapter 7**

| Propellant | $I_{sp}$ (s, vacuum) | Engine example |
|------------|---------------------|----------------|
| Cold gas (N₂) | 50–70 | Attitude thrusters |
| Solid | 250–310 | Solid rocket boosters |
| RP-1/LOX (kerosene) | 311–340 | Merlin (Falcon 9) |
| LH₂/LOX | 420–460 | SSME (Space Shuttle), Vulcain |
| N₂O₄/UDMH (hypergolic) | 310–320 | Fregat, Ariane-4 upper |
| LCH₄/LOX (methane) | 350–380 | Raptor (Starship) |
| Ion (Xenon, Hall effect) | 1500–3000 | Dawn, Starlink |

### Staging

> **[S&B] Section 4.4, p. 125** — "Multistage Rockets"

For multi-stage vehicles, the Tsiolkovsky equation is applied **per stage**:

$$\Delta v_\text{total} = \sum_{k=1}^{N} I_{sp,k}\,g_0 \ln\frac{m_{0,k}}{m_{f,k}}$$

Staging is necessary because dead hardware (empty tanks, engines) is dead weight that
reduces the mass ratio of subsequent stages. Jettisoning it frees the next stage to start
with the maximum possible mass ratio.

---

## 7. Delta-v Budget for a Real Mission

> **[BMT 2nd ed.] Section 6.6** — Combining maneuvers
> **[S&B] Section 4.3, p. 117** — Mission performance

A complete mission $\Delta v$ budget sums all maneuvers:

| Maneuver | $\Delta v$ (km/s) | Notes |
|----------|------------------|-------|
| Gravity losses (vertical flight) | 0.1–1.5 | Depends on thrust/weight ratio |
| Drag losses (atmosphere) | 0.05–0.15 | Depends on trajectory and vehicle |
| LEO insertion | 7.7–7.9 | Circularization burn |
| Hohmann LEO→GEO (burn 1) | 2.40 | |
| Hohmann LEO→GEO (burn 2) | 1.46 | |
| Station-keeping (10 years) | 0.5 | East-West control |
| De-orbit | 0.1 | Depending on orbit |
| **Total LEO→GEO mission** | **~4.2** | Excluding gravity/drag losses |

---

## 8. Ground Track and Repeat Orbits (Preview)

The ground track of a satellite depends on the ratio between the orbital period and
Earth's rotation rate. For a **sun-synchronous orbit** (SSO), $\Omega$ precesses at
exactly +0.9856°/day eastward, matching the Sun's apparent motion — so the orbit geometry
relative to the Sun is fixed. SSO uses the J₂ perturbation deliberately.

> **[BMT 2nd ed.] Section 9.5** — Sun-synchronous orbits and J₂ precession

---

## References

| Tag | Full citation |
|-----|--------------|
| **[BMT]** | Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020. ISBN 978-0-486-49704-4. |
| **[S&B]** | Sutton, Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017. ISBN 978-1-118-75388-0. |
