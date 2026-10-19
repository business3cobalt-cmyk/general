"""Checks of the two solvers against closed-form general relativity.

Run with:  python3 -m tests.test_physics
"""

from decimal import Decimal as D

import numpy as np

from gargantua import kerr
from gargantua.render import SHADOW_B, trace

_failures = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}: got {got:.10g}, want {want:.10g}")
    if not ok:
        _failures.append(name)


def test_kerr_limits():
    print("Kerr metric limits")
    check("ISCO at a=0 is 6M", float(kerr.isco_radius(D(0))), 6.0, 1e-12)
    check("horizon at a=0 is 2M", float(kerr.horizon_radius(D(0))), 2.0, 1e-12)
    check(
        "ISCO at a->M is 1M",
        float(kerr.isco_radius_from_delta(D("1e-40"))),
        1.0,
        1e-12,
    )
    check(
        "retrograde ISCO at a->M is 9M",
        float(kerr.isco_radius_from_delta(D("1e-40"), prograde=False)),
        9.0,
        1e-12,
    )
    # Kerr with a = 0 must reproduce sqrt(1 - 3M/r).
    for r in (6, 10, 50):
        check(
            f"a=0 dilation at r={r}M matches Schwarzschild",
            float(kerr.time_dilation(D(r), D(0))),
            float(kerr.schwarzschild_time_dilation(D(r))),
            1e-15,
        )


def test_millers_planet():
    print("Miller's planet: 1 hour = 7 years")
    delta = kerr.solve_spin_for_dilation()
    r = kerr.isco_radius_from_delta(delta)
    achieved = D(1) / kerr.time_dilation(r, D(1) - delta)
    check("solved dilation returns 61320", float(achieved), 61320.0, 1e-3)
    # Kip Thorne quotes a spin deficit of roughly 1.3e-14.
    check("spin deficit near Thorne's 1.3e-14", float(delta) * 1e14, 1.3, 0.2)


def _deflection(b, r_start=1e6):
    """Total deflection of a ray with impact parameter b, by direct integration."""
    # Start far away moving inwards with the requested impact parameter.
    cam = np.array([r_start, 0.0, 0.0])
    sin_a = b / r_start
    d = np.array([-np.sqrt(1.0 - sin_a**2), sin_a, 0.0])
    outcome, _, b_tot, _, esc = trace(
        cam, d[None, :], disk_in=0.0, disk_out=0.0, dphi=0.002, max_turns=6.0,
        escape_r=r_start,
    )
    if outcome[0] != 1:
        return None, float(b_tot[0])
    # Incoming direction was d; outgoing is esc[0]. Angle between them is the
    # deflection.
    cos_defl = float(np.dot(d, esc[0]))
    return float(np.arccos(np.clip(cos_defl, -1.0, 1.0))), float(b_tot[0])


def test_light_deflection():
    """Compare against the post-Newtonian series for Schwarzschild bending:

        delta_phi = 4M/b + (15 pi/4)(M/b)^2 + (128/3)(M/b)^3 + O((M/b)^4)

    The leading term is Einstein's 1915 result -- 1.75 arcsec at the solar
    limb, the prediction Eddington went to Principe to measure.
    """
    print("Light bending: delta_phi = 4M/b + (15pi/4)(M/b)^2 + (128/3)(M/b)^3")
    for b in (200.0, 500.0, 1000.0):
        defl, b_measured = _deflection(b)
        series = 4.0 / b + (15.0 * np.pi / 4.0) / b**2 + (128.0 / 3.0) / b**3
        # Residual should be the next order, O((M/b)^4); allow 10x headroom.
        check(f"impact parameter b={b:.0f}M recovered", b_measured, b, b * 2e-3)
        check(f"deflection at b={b:.0f}M", defl, series, 10.0 * 400.0 / b**4)


def test_shadow_radius():
    """The capture threshold must land on b = 3*sqrt(3) M = sqrt(27) M."""
    print("Shadow: capture threshold at b = 3*sqrt(3) M")
    lo, hi = 4.0, 7.0  # lo captured, hi escapes
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        defl, _ = _deflection(mid)
        if defl is None:  # swallowed
            lo = mid
        else:
            hi = mid
    check("critical impact parameter", 0.5 * (lo + hi), SHADOW_B, 1e-3)


def main():
    test_kerr_limits()
    test_millers_planet()
    test_light_deflection()
    test_shadow_radius()
    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
