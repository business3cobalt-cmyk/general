"""Command line front end: print the Kerr numbers, render the image."""

import argparse
import time
from decimal import Decimal as D

from . import kerr
from .png import write_png
from .render import HORIZON, PHOTON_SPHERE, SHADOW_B, render
from .sky import build_sky

# Gargantua as specified in Kip Thorne's "The Science of Interstellar":
# 100 million solar masses, spinning within a hair of the maximum.
GARGANTUA_SOLAR_MASSES = D("1e8")
SUN_GM_OVER_C2_KM = D("1.4766250382")  # GM_sun/c^2 in km
KM_PER_AU = D("149597870.7")
C_KM_S = D("299792.458")
PI = D("3.14159265358979323846264338327950288419716939937510")


def fmt(x, places=6):
    return f"{float(x):,.{places}f}"


def report_metric():
    print("=" * 72)
    print("THE KERR METRIC  (Roy Kerr, 1963) -- the spacetime of Gargantua")
    print("=" * 72)
    print(
        """
  ds^2 = -(1 - 2Mr/S) dt^2  -  (4 M a r sin^2 th / S) dt dphi
         + (S/D) dr^2  +  S dth^2
         + (r^2 + a^2 + 2 M a^2 r sin^2 th / S) sin^2 th  dphi^2

    S = r^2 + a^2 cos^2 th          D = r^2 - 2Mr + a^2        a = J/M

  The dt-dphi cross term is frame dragging: it has no Schwarzschild
  counterpart, and it is what lets an orbit sit close enough to the
  horizon for one hour to cost seven years.
"""
    )


def report_millers_planet(target_hours_per_7yr=D(1)):
    print("=" * 72)
    print("MILLER'S PLANET -- 1 hour = 7 years")
    print("=" * 72)

    ratio = kerr.INTERSTELLAR_DILATION / target_hours_per_7yr
    delta = kerr.solve_spin_for_dilation(ratio)
    spin = D(1) - delta
    r = kerr.isco_radius_from_delta(delta)
    horizon = kerr.horizon_radius(spin)
    dtau_dt = kerr.time_dilation(r, spin)

    # Convert geometric units to physical ones for this mass.
    gm_c2_km = GARGANTUA_SOLAR_MASSES * SUN_GM_OVER_C2_KM

    print(f"\n  Required slow-down factor        {fmt(ratio, 1)}x")
    print(f"  Spin parameter a/M               {spin:.20f}...")
    print(f"  Spin deficit 1 - a/M             {float(delta):.6e}")
    print("                                   (Thorne quotes ~1.3e-14)")
    print(f"\n  Orbit radius (= prograde ISCO)   {float(r):.12f} M")
    print(f"  Event horizon r+                 {float(horizon):.12f} M")
    print(f"  Orbit sits above horizon by      {float(r - horizon):.3e} M")
    print(f"\n  dtau/dt on the orbit             {float(dtau_dt):.6e}")
    print(f"  1 hour there  =                  {float(ratio) / 24 / 365:.4f} years here")

    # Orbital period. Omega = M^(1/2)/(r^(3/2) + a) gives the coordinate
    # period 2*pi/Omega; multiplying by dtau/dt converts it to time measured
    # by the crew standing on the planet.
    m_seconds = gm_c2_km / C_KM_S  # GM/c^3
    period_far = 2 * PI * (r.sqrt() * r + spin) * m_seconds
    period_local = period_far * dtau_dt

    # Kerr horizon equatorial proper circumference: 2*pi (r+^2 + a^2)/r+.
    circumference = 2 * PI * (horizon * horizon + spin * spin) / horizon * gm_c2_km

    print(f"\n  For M = 1e8 solar masses (the film's figure):")
    print(f"    GM/c^2                         {float(gm_c2_km):,.0f} km")
    print(f"    horizon circumference          {float(circumference / KM_PER_AU):.2f} AU")
    print(f"    ... i.e. radius-equivalent     {float(circumference / KM_PER_AU / (2 * PI)):.2f} AU")
    print(f"\n    orbital period, far observer   {float(period_far) / 3600:.2f} hours")
    print(f"    orbital period, on the planet  {float(period_local):.4f} seconds")
    print()


def report_geometry():
    print("=" * 72)
    print("WHAT THE CAMERA SEES  --  d^2u/dphi^2 + u = 3 M u^2,   u = 1/r")
    print("=" * 72)
    print(
        f"""
  This is the exact null geodesic equation for a non-spinning hole. The
  3Mu^2 term is general relativity; without it the equation is a straight
  line. Double Negative integrated the spinning version of it, one ray per
  pixel, to make the images in the film.

    event horizon              r = {HORIZON:.0f} M
    photon sphere              r = {PHOTON_SPHERE:.0f} M     (light orbits here, unstably)
    shadow / critical impact   b = {SHADOW_B:.6f} M   = 3*sqrt(3) M

  Rays with b < {SHADOW_B:.3f} M spiral in and are swallowed: that black disc
  in the middle of the frame is bigger than the horizon itself, because
  the hole bends light from behind it into your eye.
"""
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gargantua",
        description="Run the equations behind Interstellar's black hole.",
    )
    parser.add_argument("-o", "--output", default="gargantua.png", help="PNG to write")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--supersample", type=int, default=2, help="anti-aliasing factor")
    parser.add_argument("--cam-r", type=float, default=150.0, help="camera radius in M")
    parser.add_argument(
        "--inclination", type=float, default=88.5,
        help="camera angle from the spin axis in degrees (90 = edge on)",
    )
    parser.add_argument("--fov", type=float, default=14.0, help="field of view in degrees")
    parser.add_argument("--disk-in", type=float, default=6.0, help="inner disc radius in M")
    parser.add_argument("--disk-out", type=float, default=40.0, help="outer disc radius in M")
    parser.add_argument("--disk-temp", type=float, default=4500.0, help="peak disc temperature in K")
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--disk-gain", type=float, default=4.0)
    parser.add_argument(
        "--doppler", action="store_true",
        help="enable Doppler shift and relativistic beaming (the film left these OFF)",
    )
    parser.add_argument("--no-image", action="store_true", help="print the physics only")
    parser.add_argument("--quiet", action="store_true", help="skip the physics write-up")
    args = parser.parse_args(argv)

    if not args.quiet:
        report_metric()
        report_millers_planet()
        report_geometry()

    if args.no_image:
        return 0

    print("=" * 72)
    print(f"RENDERING  {args.width}x{args.height} at {args.supersample}x supersampling")
    print("=" * 72)
    if args.doppler:
        print("  Doppler shift + beaming: ON (physically correct)")
    else:
        print("  Doppler shift + beaming: OFF (as in the film)")

    start = time.time()
    sky = build_sky()
    image = render(
        width=args.width,
        height=args.height,
        cam_r=args.cam_r,
        inclination_deg=args.inclination,
        fov_deg=args.fov,
        disk_in=args.disk_in,
        disk_out=args.disk_out,
        disk_temp=args.disk_temp,
        doppler=args.doppler,
        supersample=args.supersample,
        exposure=args.exposure,
        disk_gain=args.disk_gain,
        sky=sky,
        progress=lambda m: print(f"  {m}"),
    )
    write_png(args.output, image)
    print(f"  wrote {args.output} in {time.time() - start:.1f}s")
    return 0
