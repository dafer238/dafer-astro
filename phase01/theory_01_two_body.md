# Theory — Module 1: The Two-Body Problem

> **References:**
> - **[BMT]** Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020
> - **[S&B]** Sutton & Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017

---

## 1. Newton's Law of Universal Gravitation

> **[BMT 2nd ed.] Section 1.2** — "Newtonian Mechanics"

$$\vec{F}_{12} = -\frac{G m_1 m_2}{r^2} \hat{r}_{12}$$

$G = 6.674 \times 10^{-11}\ \text{m}^3\text{kg}^{-1}\text{s}^{-2}$.
Both bodies orbit their common **barycenter** (Newton's third law).
For Earth + satellite the barycenter is inside Earth since $m_\text{sat} \ll M_\oplus$.

---

## 2. The One-Body Reduction

> **[BMT 2nd ed.] Section 1.3** — "The N-Body Problem"

Defining relative position $\vec{r} = \vec{r}_2 - \vec{r}_1$ and subtracting equations of motion:

$$\ddot{\vec{r}} = -\frac{G(m_1+m_2)}{r^3}\vec{r}$$

Define the **standard gravitational parameter**:

$$\boxed{\mu = G(m_1+m_2) \approx GM_\text{primary}}$$

> **Why use $\mu$ instead of $G \cdot M$?**
> $\mu_\oplus = 398600.4418\ \text{km}^3/\text{s}^2$ is known to 10 significant figures
> from tracking satellites and the lunar laser ranging experiment.
> $G$ is only known to 5 significant figures — the worst-known fundamental constant.
> Introducing $G$ would actually *reduce* accuracy. Always use $\mu$ directly.

The **two-body EOM** — the central equation of astrodynamics:

$$\boxed{\ddot{\vec{r}} = -\frac{\mu}{r^3}\vec{r}}$$

---

## 3. Constants of Motion

> **[BMT 2nd ed.] Section 1.4** — "Constants of the Motion"

### 3.1 Specific Angular Momentum

> **[BMT 2nd ed.] Section 1.4.1, Equation 1.4-1**

$$\vec{h} = \vec{r} \times \vec{v} \quad [\text{km}^2/\text{s}]$$

$\dot{\vec{h}} = 0$ (verified by taking derivative and applying EOM).
Consequences:
1. **Orbit is planar** — $\vec{h}$ defines the normal to the orbital plane
2. **Kepler's second law** — $dA/dt = h/2 = \text{const}$ (equal areas in equal times)

### 3.2 Specific Mechanical Energy

> **[BMT 2nd ed.] Section 1.4.2, Equation 1.4-2**

$$\varepsilon = \frac{v^2}{2} - \frac{\mu}{r} = -\frac{\mu}{2a} \quad [\text{km}^2/\text{s}^2]$$

| $\varepsilon$ | Orbit | Fate |
|---|---|---|
| $<0$ | Ellipse | Bound — returns to periapsis |
| $=0$ | Parabola | Escape with $v \to 0$ at infinity |
| $>0$ | Hyperbola | Escape with $v_\infty = \sqrt{2\varepsilon} > 0$ |

### 3.3 Eccentricity (Laplace-Runge-Lenz) Vector

> **[BMT 2nd ed.] Section 1.6, Equation 1.6-2**

$$\vec{e} = \frac{\vec{v}\times\vec{h}}{\mu} - \hat{r}$$

Conserved vector pointing toward **periapsis**. $|\vec{e}| = e$ (eccentricity).
Conservation means the orbit does not precess — periapsis is fixed in inertial space.
Real orbits do precess due to J₂ and other perturbations (Phase 3).

---

## 4. Orbital Geometry — Conic Sections

> **[BMT 2nd ed.] Section 1.5** — "The Orbit Equation"

$$\boxed{r = \frac{p}{1 + e\cos\theta}}$$

$p = h^2/\mu$ (semi-latus rectum), $\theta$ = true anomaly (angle from periapsis).
This is the **polar equation of a conic section** with the focus at the origin.

| $e$ | Conic | Orbit |
|-----|-------|-------|
| 0 | Circle | Special ellipse |
| $0<e<1$ | Ellipse | Closed, periodic |
| 1 | Parabola | Minimum escape |
| $e>1$ | Hyperbola | Flyby / escape |

For an ellipse: $r_p = a(1-e)$, $r_a = a(1+e)$, so $a = (r_p+r_a)/2$.

---

## 5. Vis-Viva Equation

> **[BMT 2nd ed.] Section 1.7, Equation 1.7-1**

$$\boxed{v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)}$$

The most-used formula in mission design. Special cases:
- Circular: $v_c = \sqrt{\mu/r}$
- Escape: $v_\text{esc} = \sqrt{2\mu/r} = \sqrt{2}\,v_c$

---

## 6. Kepler's Laws

> **[BMT 2nd ed.] Section 1.7** — All three derived from the two-body EOM

**First:** Orbits are conic sections. (from orbit equation, Section 1.5)

**Second:** Equal areas in equal times. (from $\vec{h}$ conservation, Section 1.4.1)

**Third:** $T = 2\pi\sqrt{a^3/\mu}$ (from integrating $dA/dt = h/2$, Equation 1.7-4)

---

## 7. Anomalies — Position as a Function of Time

> **[BMT 2nd ed.] Sections 4.1-4.4** — "Time of Flight"

**Mean anomaly:** $M = n(t-t_p)$, where $n = \sqrt{\mu/a^3}$ = mean motion [rad/s].

**Kepler's Equation:**

> **[BMT 2nd ed.] Section 4.2, Equation 4.2-4**

$$\boxed{M = E - e\sin E}$$

Transcendental — solve with Newton-Raphson:
$E_{n+1} = E_n - (E_n - e\sin E_n - M)/(1 - e\cos E_n)$

**True anomaly from E:**
$\tan(\theta/2) = \sqrt{(1+e)/(1-e)}\,\tan(E/2)$

Propagation chain: $t \to M \to E \to \theta \to r$

---

## 8. Classical Orbital Elements (COEs)

> **[BMT 2nd ed.] Section 2.4** — "Determination of Orbital Elements"

| Symbol | Name | Range |
|--------|------|-------|
| $a$ | Semi-major axis (km) | $>0$ |
| $e$ | Eccentricity | $[0,\infty)$ |
| $i$ | Inclination | $[0°,180°]$ |
| $\Omega$ | RAAN | $[0°,360°)$ |
| $\omega$ | Arg. of Perigee | $[0°,360°)$ |
| $\theta$ | True Anomaly | $[0°,360°)$ |

**Coordinate frames:**

> **[BMT 2nd ed.] Section 2.2** — ECI vs ECEF

- **ECI:** X to vernal equinox, Z to North Pole. Non-rotating. For orbital math.
- **ECEF:** Rotates with Earth. For ground tracks, launch sites.
- **Perifocal:** X toward periapsis, Y in motion direction. Natural for orbit calculations.

**COE to state vector:**

> **[BMT 2nd ed.] Section 2.4.3, Algorithm 2.1, Equations 2.4-3 and 2.4-4**

Compute $\vec{r}$, $\vec{v}$ in perifocal frame, then rotate to ECI with
$\mathbf{Q} = R_3(-\Omega)\cdot R_1(-i)\cdot R_3(-\omega)$.

**State vector to COE:**

> **[BMT 2nd ed.] Section 2.4, Algorithm 2.2**

Use $\vec{h}$, $\vec{e}$, node vector $\vec{N} = \hat{K}\times\vec{h}$ in sequence.

---

## 9. Numerical Integration — RK4

RK4 global error $O(h^4)$. For LEO at $h=10$ s: drift $\lesssim 10^{-10}$ per orbit.

State vector: $\vec{y} = [\vec{r},\vec{v}]^T$. Derivative: $\dot{\vec{y}} = [\vec{v},\,-\mu\vec{r}/r^3]^T$.

$$y_{n+1} = y_n + \frac{h}{6}(k_1+2k_2+2k_3+k_4)$$

**Validation:** conservation of $\varepsilon$ and $|\vec{h}|$ is your accuracy diagnostic.

---

## 10. Connection to Propulsion

> **[S&B] Section 4.2, pp. 114-118** — "Rocket Flight Performance"
> **[S&B] Section 2.1, p. 27** — Definition and measurement of $I_{sp}$

Every orbital maneuver has a $\Delta v$ cost. The rocket equation converts it to propellant:

$$\boxed{\Delta v = I_{sp}\,g_0\,\ln\frac{m_0}{m_f}}$$

$I_{sp}$ (specific impulse) = thrust per unit weight flow rate [s]. The universal efficiency
metric. $v_e = I_{sp} \cdot g_0$ = effective exhaust velocity [km/s]. See Module 2 and 4.

---

## References

| Tag | Citation |
|-----|---------|
| **[BMT]** | Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020. ISBN 978-0-486-49704-4. |
| **[S&B]** | Sutton, Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017. ISBN 978-1-118-75388-0. |
