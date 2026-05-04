"""
Demo script for multi-body simulation with solar system ephemeris.
Run: python -m simulator.demo_multibody
"""

from datetime import datetime, timezone
import numpy as np

from simulator.core.spacecraft import Spacecraft
from simulator.core.state import OrbitalElements
from simulator.core.ephemeris import get_solar_system_state, datetime_to_jd
from simulator.sim.multibody import MultiBodyScenario, MultiBodyEngine


def demo_rendezvous():
    """Demo: Two spacecraft in different orbits for rendezvous planning."""
    print("=" * 60)
    print("MULTI-BODY SIMULATION DEMO")
    print("=" * 60)

    epoch = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    print(f"\nEpoch: {epoch}")

    # Print solar system state
    jd = datetime_to_jd(epoch)
    ss = get_solar_system_state(jd)
    print("\nSolar System State (heliocentric ecliptic):")
    for name, data in ss.items():
        dist_au = np.linalg.norm(data["position"]) / 1.496e8
        print(f"  {name:10s}: {dist_au:10.4f} AU from Sun")

    # Define two spacecraft
    chaser = Spacecraft(
        name="Chaser",
        color=(0, 255, 0),
        initial_coe=OrbitalElements(a=6778, e=0.0002, i=51.6, raan=30, omega=0, theta=0),
    )
    target = Spacecraft(
        name="Target",
        color=(255, 0, 0),
        initial_coe=OrbitalElements(a=6778, e=0.0001, i=51.6, raan=30, omega=0, theta=10),
    )

    scenario = MultiBodyScenario(
        spacecraft=[chaser, target],
        central_body="Earth",
        perturbing_bodies=["Moon", "Sun"],
        epoch=epoch,
        duration_seconds=5400.0,  # 1.5 hours
    )

    print("\nSimulating two spacecraft...")
    engine = MultiBodyEngine()
    result = engine.compute(scenario)

    if result is None:
        print(f"ERROR: {engine.error}")
        return

    print(f"  Time points: {len(result.t)}")
    print(f"  Duration: {result.t[-1]:.1f} s")

    # Compute relative distance over time
    r_chaser = result.positions["Chaser"]
    r_target = result.positions["Target"]
    rel_dist = np.linalg.norm(r_chaser - r_target, axis=1)
    print(f"\n  Relative distance:")
    print(f"    Initial: {rel_dist[0]:.3f} km")
    print(f"    Final:   {rel_dist[-1]:.3f} km")
    print(f"    Min:     {rel_dist.min():.3f} km")
    print(f"    Max:     {rel_dist.max():.3f} km")

    print("\nDone!")


if __name__ == "__main__":
    demo_rendezvous()
