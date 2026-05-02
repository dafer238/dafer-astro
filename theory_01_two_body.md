# Theory — Module 1: The Two-Body Problem

> **Prerequisites:** Newtonian mechanics, vectors, ODEs, basic linear algebra.
> This is the absolute foundation of astrodynamics. Master this and everything else follows.

---

## 1. Newton's Law of Universal Gravitation

Two point masses $m_1$ and $m_2$ separated by distance $r$ attract each other with force:

$$\vec{F}_{12} = -\frac{G m_1 m_2}{r^2} \hat{r}_{12}$$

Where:
- $G = 6.674 \times 10^{-11}\ \text{m}^3 \text{kg}^{-1} \text{s}^{-2}$ (gravitational constant)
- $\hat{r}_{12}$ is the unit vector from $m_1$ to $m_2$
- The minus sign means attraction (force points toward the other body)

**Key insight:** Newton's third law means $\vec{F}_{21} = -\vec{F}_{12}$. Both bodies orbit their common **center of mass (barycenter)**.

---

## 2. Reduction to the One-Body Problem

Let $\vec{r}_1$, $\vec{r}_2$ be inertial position vectors of the two bodies. Define:
- **Relative position:** $\vec{r} = \vec{r}_2 - \vec{r}_1$
- **Center of mass:** $\vec{R} = \frac{m_1 \vec{r}_1 + m_2 \vec{r}_2}{m_1 + m_2}$

The center of mass moves at constant velocity (no external forces), so we can work in the **CoM frame** where $\vec{R} = 0$.

Subtracting the equations of motion for each body:

$$\ddot{\vec{r}} = -\frac{G(m_1 + m_2)}{r^3} \vec{r}$$

Now define the **standard gravitational parameter**:

$$\boxed{\mu = G(m_1 + m_2) \approx GM_{\text{primary}}}$$

The approximation holds when $m_1 \gg m_2$ (e.g., Earth + satellite). This gives the **two-body equation of motion**:

$$\boxed{\ddot{\vec{r}} = -\frac{\mu}{r^3} \vec{r}}$$

This is a **second-order nonlinear ODE** in 3D (6 degrees of freedom). It is the central equation of orbital mechanics.

**Why is this powerful?**  
- The problem is exactly solvable analytically
- Solutions are **conic sections** (ellipses, parabolas, hyperbolas)
- Two conserved quantities exist: **specific energy** and **specific angular momentum**

---

## 3. Constants of Motion

### 3.1 Specific Angular Momentum

$$\vec{h} = \vec{r} \times \dot{\vec{r}} = \vec{r} \times \vec{v}$$

Taking the time derivative and using the EOM:

$$\dot{\vec{h}} = \dot{\vec{r}} \times \vec{v} + \vec{r} \times \ddot{\vec{r}} = \vec{v} \times \vec{v} + \vec{r} \times \left(-\frac{\mu}{r^3}\vec{r}\right) = 0$$

Both terms vanish (cross product of parallel vectors). Therefore **$\vec{h}$ is conserved**. This has two consequences:
1. The orbit is **planar** (motion always perpendicular to the fixed $\vec{h}$)
2. Kepler's second law: equal areas swept in equal times

$$\frac{dA}{dt} = \frac{h}{2} = \text{constant}$$

### 3.2 Specific Mechanical Energy (Vis-Viva)

The work-energy theorem gives the total specific energy (per unit mass):

$$\varepsilon = \frac{v^2}{2} - \frac{\mu}{r}$$

- Kinetic energy: $\frac{v^2}{2}$
- Gravitational potential energy: $-\frac{\mu}{r}$

Taking the time derivative and using the EOM shows $\dot{\varepsilon} = 0$, so **$\varepsilon$ is conserved**.

| $\varepsilon$ | Orbit type | Fate |
|---|---|---|
| $< 0$ | Ellipse (bound) | Returns to periapsis |
| $= 0$ | Parabola (marginally bound) | Escapes with $v \to 0$ at $\infty$ |
| $> 0$ | Hyperbola (unbound) | Escapes with $v > 0$ at $\infty$ |

Relating to the semi-major axis $a$:

$$\boxed{\varepsilon = -\frac{\mu}{2a}}$$

### 3.3 The Laplace-Runge-Lenz Vector (Bonus)

$$\vec{e} = \frac{\vec{v} \times \vec{h}}{\mu} - \hat{r}$$

This vector points from the focus toward periapsis (closest approach) and its magnitude equals the eccentricity. It's a third conserved vector, but since it's in the orbital plane, it provides only one independent constant (the direction of periapsis doesn't change).

---

## 4. Orbital Geometry — Conic Sections

### 4.1 The Orbit Equation

Using $\vec{h}$ and $\vec{e}$, the orbit can be written as:

$$\boxed{r = \frac{p}{1 + e\cos\theta}}$$

Where:
- $p = \frac{h^2}{\mu}$ is the **semi-latus rectum**
- $e = |\vec{e}|$ is the **eccentricity**
- $\theta$ is the **true anomaly** (angle from periapsis, measured in direction of motion)

This is the polar equation of a **conic section** with the focus at the origin. The type of conic depends entirely on $e$:

| $e$ | Shape | Orbit |
|-----|-------|-------|
| $0$ | Circle | Special ellipse |
| $0 < e < 1$ | Ellipse | Closed, periodic |
| $1$ | Parabola | Escape, marginally |
| $e > 1$ | Hyperbola | Flyby, escape |

### 4.2 Key Orbital Geometry

For an **ellipse** with semi-major axis $a$ and semi-minor axis $b$:

$$b = a\sqrt{1 - e^2}$$

$$p = a(1 - e^2) = \frac{b^2}{a}$$

**Periapsis** (closest point):
$$r_p = a(1-e)$$

**Apoapsis** (farthest point):
$$r_a = a(1+e)$$

Note: $a = \frac{r_p + r_a}{2}$ — the semi-major axis is just the average of the two extremes.

---

## 5. The Vis-Viva Equation

Combining energy conservation with the orbit equation:

$$\boxed{v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)}$$

This is arguably the most used equation in mission design. Given position and semi-major axis, you instantly get the speed. Special cases:

**Circular orbit** ($r = a$):
$$v_c = \sqrt{\frac{\mu}{r}}$$

**Escape velocity** (parabolic, $a \to \infty$):
$$v_{esc} = \sqrt{\frac{2\mu}{r}} = \sqrt{2} \cdot v_c$$

**At periapsis:**
$$v_p = \sqrt{\mu\left(\frac{2}{r_p} - \frac{1}{a}\right)} = \sqrt{\frac{\mu}{a} \cdot \frac{1+e}{1-e}}$$

---

## 6. Kepler's Three Laws (derived, not postulated)

**First Law:** Orbits are conic sections with the primary at one focus.
→ *Derived from the two-body EOM*

**Second Law:** Equal areas swept in equal times.
→ *Direct consequence of $\vec{h}$ conservation*

**Third Law:** $T^2 \propto a^3$

$$\boxed{T = 2\pi\sqrt{\frac{a^3}{\mu}}}$$

*Derived from integrating $\frac{dA}{dt} = \frac{h}{2}$ over one full orbit.*

---

## 7. Anomalies — Describing Position in Time

Given an orbit's shape, how do we find *where* the satellite is at time $t$?

### 7.1 True Anomaly $\theta$
The physical angle from periapsis. Non-uniform in time (satellite moves faster near periapsis — Kepler's 2nd law).

### 7.2 Mean Anomaly $M$
An artificial angle that grows uniformly with time:

$$M = n(t - t_p) \quad \text{where } n = \sqrt{\frac{\mu}{a^3}} = \frac{2\pi}{T}$$

$n$ is the **mean motion** (rad/s). At $t = t_p$ (time of periapsis), $M = 0$.

### 7.3 Eccentric Anomaly $E$
Intermediate variable. Defined geometrically on the auxiliary circle.

**Kepler's Equation** (connects $M$ and $E$):
$$M = E - e\sin E$$

This is **transcendental** — no closed-form solution for $E$ given $M$. Must be solved iteratively (Newton-Raphson).

**Converting $E$ to $\theta$:**
$$\tan\frac{\theta}{2} = \sqrt{\frac{1+e}{1-e}} \tan\frac{E}{2}$$

The sequence is:
$$t \xrightarrow{n} M \xrightarrow{\text{Kepler}} E \xrightarrow{\text{geometry}} \theta \xrightarrow{\text{orbit eq.}} r$$

---

## 8. Classical Orbital Elements (COEs)

A full description of an orbit + position requires **6 parameters**. The COEs give them physical meaning:

| Symbol | Name | Definition |
|--------|------|-----------|
| $a$ | Semi-major axis | Size of orbit (km) |
| $e$ | Eccentricity | Shape (dimensionless) |
| $i$ | Inclination | Tilt of orbital plane w.r.t. equatorial plane (0°–180°) |
| $\Omega$ | RAAN (Right Ascension of Ascending Node) | Where the orbit crosses the equator going north (0°–360°) |
| $\omega$ | Argument of Perigee | Angle from ascending node to periapsis (0°–360°) |
| $\theta$ (or $\nu$) | True Anomaly | Current position in orbit (0°–360°) |

### Coordinate frames:
- **ECI (Earth-Centered Inertial):** X points to vernal equinox, Z points to North Pole. Does NOT rotate with Earth. Used for orbits.
- **ECEF (Earth-Centered, Earth-Fixed):** Rotates with Earth. Used for ground tracks.
- **Perifocal frame:** X toward periapsis, Y in direction of motion, Z = $\hat{h}$. Natural for orbital calculations.

### Conversion: COE → State Vector
In the **perifocal frame**:
$$\vec{r}_{\text{pf}} = \frac{p}{1+e\cos\theta}\begin{pmatrix}\cos\theta \\ \sin\theta \\ 0\end{pmatrix}$$

$$\vec{v}_{\text{pf}} = \sqrt{\frac{\mu}{p}}\begin{pmatrix}-\sin\theta \\ e+\cos\theta \\ 0\end{pmatrix}$$

Then rotate to ECI using three successive rotations: $R_3(-\Omega) \cdot R_1(-i) \cdot R_3(-\omega)$.

---

## 9. Numerical Integration — Why and How

The analytical solution exists for two-body, but:
1. Real-world problems involve perturbations (drag, J₂, N-body) where no analytical solution exists
2. You need a numerical integrator for trajectory propagation

### Runge-Kutta 4th Order (RK4)

For an ODE $\dot{\vec{y}} = f(t, \vec{y})$, one step of size $h$:

$$k_1 = f(t_n,\ \vec{y}_n)$$
$$k_2 = f\!\left(t_n + \tfrac{h}{2},\ \vec{y}_n + \tfrac{h}{2}k_1\right)$$
$$k_3 = f\!\left(t_n + \tfrac{h}{2},\ \vec{y}_n + \tfrac{h}{2}k_2\right)$$
$$k_4 = f(t_n + h,\ \vec{y}_n + h\,k_3)$$

$$\vec{y}_{n+1} = \vec{y}_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

**Local truncation error:** $O(h^5)$ per step → **global error** $O(h^4)$.

The state vector for the two-body problem is:

$$\vec{y} = \begin{pmatrix}\vec{r} \\ \vec{v}\end{pmatrix} \in \mathbb{R}^6, \quad \dot{\vec{y}} = \begin{pmatrix}\vec{v} \\ -\frac{\mu}{r^3}\vec{r}\end{pmatrix}$$

### Choosing the time step $h$

A practical rule: the period $T$ should span at least ~100–1000 steps. For LEO (~90 min period), $h = 10$–$60$ s works well. Conservation of energy ($\varepsilon$) and angular momentum ($h$) is your accuracy diagnostic — errors should be $< 10^{-8}$ relative.

---

## 10. Physical Intuition Checklist

Before writing a single line of code, internalize these:

- [ ] A satellite in LEO (~400 km) moves at ~7.7 km/s and completes one orbit in ~92 min
- [ ] GEO is at 35,786 km — the magic altitude where $T = 24$ hours
- [ ] Escape velocity from Earth's surface is 11.2 km/s
- [ ] Adding speed at periapsis raises apoapsis (and vice versa) — never the periapsis itself
- [ ] Angular momentum prevents collapse: without $\vec{v}$, any orbit decays to a straight fall
- [ ] The two-body solution is *exact* — any drift in your simulation is numerical error, not physics

---

## References & Further Reading

1. **Bate, Mueller, White** — *Fundamentals of Astrodynamics* (Dover) → The classic textbook, rigorous yet accessible
2. **Vallado** — *Fundamentals of Astrodynamics and Applications* → Modern, with algorithms (matches what we implement)
3. **Curtis** — *Orbital Mechanics for Engineering Students* → Excellent for self-study, many worked examples
4. **Battin** — *An Introduction to the Mathematics and Methods of Astrodynamics* → Advanced, mathematical depth
5. **Sutton & Biblarz** — *Rocket Propulsion Elements* → Standard reference for propulsion

