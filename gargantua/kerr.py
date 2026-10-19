"""Kerr spacetime: the metric Gargantua is built from.

The line element in Boyer-Lindquist coordinates (geometric units G = c = 1)::

    ds^2 = -(1 - 2Mr/S) dt^2
           - (4 M a r sin^2(th) / S) dt dphi
           + (S/D) dr^2
           + S dth^2
           + (r^2 + a^2 + 2 M a^2 r sin^2(th) / S) sin^2(th) dphi^2

    S = r^2 + a^2 cos^2(th)          ("Sigma")
    D = r^2 - 2Mr + a^2              ("Delta")

`a = J/M` is the spin parameter; `a = M` is a maximally spinning hole.

Everything below sets M = 1, so radii are in units of the gravitational
radius GM/c^2 and `a` is the dimensionless spin.

The near-extremal arithmetic (Miller's planet needs 1 - a/M ~ 1e-14) suffers
catastrophic cancellation in float64, so the analysis functions run on
`decimal.Decimal` at 60 significant digits.
"""

from decimal import Decimal, getcontext

getcontext().prec = 60

D0 = Decimal(0)
D1 = Decimal(1)
D2 = Decimal(2)
D3 = Decimal(3)
D4 = Decimal(4)

# Seconds per year, and the film's headline ratio: one hour on Miller's
# planet costs seven years for everyone else.
HOURS_PER_YEAR = Decimal(365) * Decimal(24)
INTERSTELLAR_DILATION = Decimal(7) * HOURS_PER_YEAR  # 61320


def _cbrt(x: Decimal) -> Decimal:
    """Cube root that stays exact-ish for the tiny arguments we feed it."""
    if x == 0:
        return D0
    if x < 0:
        return -((-x) ** (D1 / D3))
    return x ** (D1 / D3)


def horizon_radius(spin: Decimal) -> Decimal:
    """Outer event horizon r+ = M + sqrt(M^2 - a^2)."""
    return D1 + (D1 - spin * spin).sqrt()


def ergosphere_radius(spin: Decimal, theta_cos: Decimal = D0) -> Decimal:
    """Static limit r_E = M + sqrt(M^2 - a^2 cos^2(th)); equatorial by default."""
    return D1 + (D1 - spin * spin * theta_cos * theta_cos).sqrt()


def isco_radius(spin: Decimal, prograde: bool = True) -> Decimal:
    """Innermost stable circular orbit (Bardeen, Press & Teukolsky 1972).

        Z1 = 1 + (1-a^2)^(1/3) [ (1+a)^(1/3) + (1-a)^(1/3) ]
        Z2 = sqrt(3 a^2 + Z1^2)
        r  = 3 + Z2 -/+ sqrt((3 - Z1)(3 + Z1 + 2 Z2))

    Prograde takes the minus sign. As a -> M the prograde ISCO slides all the
    way down to r = M in these coordinates, which is exactly why a maximally
    spinning hole can host a planet with a survivable orbit and an absurd
    time dilation.
    """
    a = spin
    z1 = D1 + _cbrt(D1 - a * a) * (_cbrt(D1 + a) + _cbrt(D1 - a))
    z2 = (D3 * a * a + z1 * z1).sqrt()
    disc = ((D3 - z1) * (D3 + z1 + D2 * z2)).sqrt()
    return D3 + z2 - disc if prograde else D3 + z2 + disc


def isco_radius_from_delta(delta: Decimal, prograde: bool = True) -> Decimal:
    """ISCO for spin a = 1 - delta, written to avoid cancellation as delta -> 0.

    Substituting a = 1 - delta gives 1 - a^2 = delta(2 - delta), 1 - a = delta
    and 1 + a = 2 - delta, so every cube root is evaluated on a quantity that
    is computed exactly rather than by subtracting two near-equal numbers.
    """
    a = D1 - delta
    z1 = D1 + _cbrt(delta * (D2 - delta)) * (_cbrt(D2 - delta) + _cbrt(delta))
    z2 = (D3 * a * a + z1 * z1).sqrt()
    disc = ((D3 - z1) * (D3 + z1 + D2 * z2)).sqrt()
    return D3 + z2 - disc if prograde else D3 + z2 + disc


def orbital_angular_velocity(r: Decimal, spin: Decimal, prograde: bool = True) -> Decimal:
    """Coordinate angular velocity of a circular equatorial geodesic.

        Omega = +/- M^(1/2) / (r^(3/2) +/- a M^(1/2))
    """
    root_r = r.sqrt() * r  # r^(3/2)
    return D1 / (root_r + spin) if prograde else -D1 / (root_r - spin)


def time_dilation(r: Decimal, spin: Decimal, prograde: bool = True) -> Decimal:
    """dtau/dt for a circular equatorial orbit -- the Miller's planet factor.

        dtau/dt = sqrt(r^3 - 3 M r^2 +/- 2 a M^(1/2) r^(3/2)) / (r^(3/2) +/- a M^(1/2))

    One second on the orbit corresponds to 1/(dtau/dt) seconds far away.
    Setting a = 0 recovers the Schwarzschild result sqrt(1 - 3M/r).
    """
    root_r = r.sqrt() * r  # r^(3/2)
    sign = D1 if prograde else -D1
    radicand = r ** 3 - D3 * r * r + sign * D2 * spin * root_r
    if radicand <= 0:
        return D0
    return radicand.sqrt() / (root_r + sign * spin)


def schwarzschild_time_dilation(r: Decimal) -> Decimal:
    """dtau/dt = sqrt(1 - 3M/r) for a circular orbit around a static hole."""
    return (D1 - D3 / r).sqrt()


def solve_spin_for_dilation(
    target: Decimal = INTERSTELLAR_DILATION,
    lo_exp: int = -30,
    hi_exp: int = 0,
    iterations: int = 400,
) -> Decimal:
    """Find the spin deficit delta = 1 - a/M that puts a planet at the ISCO
    with the requested time dilation.

    `target` is the slow-down factor (61320 means one orbital hour per seven
    outside years). Bisects on delta because dtau/dt at the ISCO decreases
    monotonically as the hole approaches maximal spin.
    """
    want = D1 / target

    def dilation_at(delta: Decimal) -> Decimal:
        r = isco_radius_from_delta(delta)
        return time_dilation(r, D1 - delta)

    lo = Decimal(10) ** lo_exp   # nearly extremal -> dilation below target
    hi = Decimal(10) ** hi_exp   # slow spin       -> dilation above target
    for _ in range(iterations):
        mid = (lo * hi).sqrt()  # geometric bisection; delta spans many decades
        if dilation_at(mid) > want:
            hi = mid
        else:
            lo = mid
    return (lo * hi).sqrt()
