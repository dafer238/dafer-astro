# Theory — Module 3: Orbital Perturbations

> **References:**
> - **[BMT]** Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020
> - **[S&B]** Sutton & Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017

---

## 1. Why the Two-Body Solution is Not Enough

> **[BMT 2nd ed.] Section 9.1** — "Reasons for Perturbations"

The pure two-body EOM $\ddot{\vec{r}} = -\mu\vec{r}/r^3$ is exact only for two perfect spheres
in otherwise empty space. Real satellite orbits experience additional forces — **perturbations**
— that cause the orbital elements to evolve over time.

We write the full EOM as:

$$\ddot{\vec{r}} = -\frac{\mu}{r^3}\vec{r} + \vec{a}_\text{pert}$$

where $\vec{a}_\text{pert}$ is the sum of all perturbing accelerations. Because these are
typically small compared to the central force, orbital elements change *slowly* — this is
what makes **perturbation theory** possible and useful.

| Perturbation | Magnitude (LEO) | Primary effect |
|---|---|---|
| Earth's $J_2$ oblateness | $\sim 10^{-3}\,g$ | RAAN precession, apsidal drift |
| Atmospheric drag | $\sim 10^{-7}$ to $10^{-5}\,g$ | Altitude decay |
| Lunar gravity | $\sim 10^{-6}\,g$ | Long-period oscillations |
| Solar radiation pressure | $\sim 10^{-8}\,g$ | Small eccentricity changes |
| Solar gravity | $\sim 10^{-7}\,g$ | Resonances in high orbits |

---

## 2. Earth's Oblateness — The $J_2$ Perturbation

> **[BMT 2nd ed.] Section 9.3** — "Effects of Earth's Oblateness"

Earth is not a perfect sphere. Its equatorial radius ($6378.137$ km) exceeds its polar radius
($6356.752$ km) by $21.4$ km — a flattening of $f = 1/298.257$. This oblateness concentrates
mass near the equator, creating an asymmetric gravitational field.

### Gravitational Potential with $J_2$

The gravitational potential including the dominant zonal harmonic:

$$U = \frac{\mu}{r}\left[1 - J_2\left(\frac{R_E}{r}\right)^2 P_2(\sin\phi)\right]$$

where $P_2(\sin\phi) = \frac{1}{2}(3\sin^2\phi - 1)$ is the Legendre polynomial,
$\phi$ is the geocentric latitude, and:

$$\boxed{J_2 = 1.082\,627 \times 10^{-3}}$$

This is the most precisely known harmonic and dominates all others by an order of magnitude.

### $J_2$ Acceleration (Cartesian)

> **[BMT 2nd ed.] Section 9.3, Equation 9.3-1**

$$a_x = \frac{3\mu J_2 R_E^2}{2r^5}\,x\!\left(\frac{5z^2}{r^2} - 1\right)$$
$$a_y = \frac{3\mu J_2 R_E^2}{2r^5}\,y\!\left(\frac{5z^2}{r^2} - 1\right)$$
$$a_z = \frac{3\mu J_2 R_E^2}{2r^5}\,z\!\left(\frac{5z^2}{r^2} - 3\right)$$

This is what you add directly to the EOM right-hand side.

### Secular Rates (orbit-averaged)

> **[BMT 2nd ed.] Section 9.3, Equations 9.3-4 through 9.3-6**

After averaging over one orbit, the secular (long-term accumulating) rates are:

**RAAN precession:**
$$\boxed{\dot{\Omega} = -\frac{3}{2}\,n\,J_2\!\left(\frac{R_E}{p}\right)^2\!\cos i}$$

**Apsidal drift (argument of perigee):**
$$\boxed{\dot{\omega} = \frac{3}{4}\,n\,J_2\!\left(\frac{R_E}{p}\right)^2\!(5\cos^2 i - 1)}$$

**Mean motion correction:**
$$\dot{M} = n\left[1 + \frac{3}{2}J_2\left(\frac{R_E}{p}\right)^2\sqrt{1-e^2}\left(1 - \frac{3}{2}\sin^2 i\right)\right]$$

where $p = a(1-e^2)$ and $n = \sqrt{\mu/a^3}$.

> **Physical interpretation:**
> - $\dot{\Omega} < 0$ for prograde orbits ($i < 90°$): the orbital plane rotates westward.
>   For the ISS at 51.6°: $\dot{\Omega} \approx -5.0°/\text{day}$.
> - $\dot{\omega}$ changes sign at the **critical inclination** $i_c = 63.435°$,
>   where $5\cos^2 i - 1 = 0$.

---

## 3. Critical Inclination and Frozen Orbits

> **[BMT 2nd ed.] Section 9.7** — "Frozen Orbits and Critical Inclination"

At $i_c = 63.435°$ (or its supplement $116.565°$), $\dot{\omega} = 0$ exactly.
The argument of perigee is **frozen** — it does not drift.

This is why Molniya orbits use $i = 63.4°$: the apoapsis stays permanently over the
Northern Hemisphere without active correction. It is a free gift from J₂.

Higher-order harmonics cause small residual drift even at $i_c$; for a "truly frozen orbit"
both $e$ and $\omega$ must be tuned to simultaneously satisfy J₂ through J₄ conditions
(operational Earth-observation satellites exploit this).

---

## 4. Sun-Synchronous Orbit

> **[BMT 2nd ed.] Section 9.5** — "Sun-Synchronous Orbits"

For Earth observation and remote sensing, it is highly desirable to cross any given
latitude at the same local solar time every orbit — guaranteeing consistent illumination
for imaging.

This requires $\dot{\Omega}$ to match the Sun's apparent eastward motion:

$$\dot{\Omega}_\text{SSO} = +\frac{360°}{365.25\ \text{days}} = +0.9856°/\text{day}$$

Setting $\dot{\Omega} = \dot{\Omega}_\text{SSO}$ in the secular rate equation:

$$\cos i_\text{SSO} = -\frac{\dot{\Omega}_\text{SSO}}{(3/2)\,n\,J_2\,(R_E/p)^2}$$

Since $\dot{\Omega}_\text{SSO} > 0$ but the right-hand side of the J₂ RAAN formula is
negative for prograde orbits, we need $\cos i < 0$ — i.e., **retrograde orbits** ($i > 90°$).

Typical SSO altitudes and inclinations:

| Altitude (km) | SSO inclination |
|---|---|
| 400 | 97.0° |
| 500 | 97.4° |
| 600 | 97.8° |
| 700 | 98.2° |
| 800 | 98.6° |

Sentinel-2, Landsat, Planet, and most Earth-observation satellites use SSO.

---

## 5. Atmospheric Drag

> **[BMT 2nd ed.] Section 9.4** — "Atmospheric Drag"

The drag force on a spacecraft moving through a rarefied atmosphere:

$$\vec{F}_D = -\frac{1}{2}C_D\,A\,\rho\,v_\text{rel}^2\,\hat{v}_\text{rel}$$

The **drag acceleration** (divide by mass $m$):

$$\boxed{\vec{a}_D = -\frac{1}{2}\,C_D\,\frac{A}{m}\,\rho\,v_\text{rel}^2\,\hat{v}_\text{rel}}$$

where:
- $C_D \approx 2.0$–$2.5$: aerodynamic drag coefficient (satellite-shape dependent)
- $A/m$: area-to-mass ratio [m²/kg] — the **ballistic coefficient** numerator
- $\rho$: atmospheric density at current altitude [kg/m³]
- $\vec{v}_\text{rel} = \vec{v} - \vec{\omega}_E \times \vec{r}$: velocity relative to the rotating atmosphere

The term $C_D A / m$ is sometimes written as $1/\beta$ where $\beta$ is the **ballistic coefficient**
in kg/m². High $\beta$ → dense/streamlined → resists drag. Low $\beta$ → decelerates quickly.

### Exponential Atmosphere Model

> **[BMT 2nd ed.] Section 9.4, Table 9.4-1**

$$\rho(h) = \rho_0\,\exp\!\left(-\frac{h - h_0}{H}\right)$$

where $h_0$ is the base altitude of a layer, $\rho_0$ is the base density, and $H$ is the
**scale height** (the altitude over which density drops by factor $e$).

Representative values:

| Altitude (km) | $\rho_0$ (kg/m³) | $H$ (km) |
|---|---|---|
| 200 | $2.54 \times 10^{-10}$ | 47.9 |
| 300 | $1.92 \times 10^{-11}$ | 59.9 |
| 400 | $2.80 \times 10^{-12}$ | 65.6 |
| 500 | $5.22 \times 10^{-13}$ | 73.6 |
| 600 | $1.14 \times 10^{-13}$ | 76.3 |

> **Important:** Atmospheric density at a given altitude varies by **2–3 orders of magnitude**
> between solar minimum and solar maximum due to UV heating of the upper atmosphere. The
> ISS required 7 km/year of altitude maintenance boosts at solar maximum vs ~1 km/year at
> solar minimum. For precise mission analysis, use NRLMSISE-00 (via `astropy` or standalone).

### Semi-Major Axis Decay Rate

For a circular orbit, the averaged decay rate is:

$$\dot{a} = -C_D\,\frac{A}{m}\,\rho\,\sqrt{\mu\,a}$$

This shows that lower orbits decay faster (higher $\rho$) and lighter/larger spacecraft
decay faster (higher $A/m$). Orbital lifetime scales roughly as:

$$\tau \approx \frac{H}{\dot{a}} \approx \frac{m}{C_D A}\,\frac{H}{\rho\sqrt{\mu a}}$$

---

## 6. Gauss's Variational Equations (GVE)

> **[BMT 2nd ed.] Section 9.6** — "Gauss's Variational Equations"

Instead of integrating the Cartesian EOM and then extracting COEs, we can directly
integrate the rates of change of the orbital elements under perturbations. In the
**RSW frame** (R = radial, S = along-track, W = cross-track):

$$\frac{da}{dt} = \frac{2a^2}{\sqrt{\mu p}}\left[e\sin\theta\cdot f_R + \frac{p}{r}\cdot f_S\right]$$

$$\frac{de}{dt} = \frac{\sqrt{p/\mu}}{1}\left[\sin\theta\cdot f_R + \frac{(e+2\cos\theta+e\cos^2\theta)}{1+e\cos\theta}\cdot f_S\right]$$

$$\frac{di}{dt} = \frac{r\cos(\omega+\theta)}{\sqrt{\mu p}}\cdot f_W$$

$$\frac{d\Omega}{dt} = \frac{r\sin(\omega+\theta)}{\sin i\sqrt{\mu p}}\cdot f_W$$

These equations are exact (not linearized). They show directly which perturbing force
components affect which elements: cross-track forces $f_W$ change inclination and RAAN;
in-plane forces $f_R$ and $f_S$ change $a$, $e$, and $\omega$.

---

## 7. Connecting Perturbations to Mission Design

| Effect | Formula | Application |
|--------|---------|-------------|
| RAAN precession | $\dot{\Omega} = -\frac{3}{2}nJ_2(R_E/p)^2\cos i$ | SSO design, station-keeping budget |
| Frozen orbit | $i = 63.435°$ gives $\dot{\omega}=0$ | Molniya, HEO communications |
| Drag decay | $\dot{a} = -C_D(A/m)\rho\sqrt{\mu a}$ | Deorbit lifetime, debris reentry |
| J₂ mean motion | Corrected $n$ | Precise repeat ground track |

### Station-Keeping $\Delta v$ Budget

> **[S&B] Section 4.3, p. 117** — Station-keeping propellant budget

GEO north-south station-keeping (combating inclination from lunar/solar gravity):
$\sim 50\ \text{m/s/year}$

GEO east-west station-keeping (combating eccentricity from tesseral harmonics):
$\sim 2\ \text{m/s/year}$

LEO drag make-up (ISS at 400 km, moderate solar activity):
$\sim 60\ \text{m/s/year}$

---

## References

| Tag | Citation |
|-----|---------|
| **[BMT]** | Bate, Mueller, White, Saylor — *Fundamentals of Astrodynamics*, 2nd ed., Dover, 2020. ISBN 978-0-486-49704-4 |
| **[S&B]** | Sutton, Biblarz — *Rocket Propulsion Elements*, 9th ed., Wiley, 2017. ISBN 978-1-118-75388-0 |
| **[Vallado]** | Vallado — *Fundamentals of Astrodynamics and Applications*, 4th ed., Microcosm, 2013 — NRLMSISE-00 atmosphere model and full GVE implementation |
