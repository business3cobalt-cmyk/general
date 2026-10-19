"""Backward ray tracing of null geodesics -- the DNGR idea, in miniature.

Double Negative's renderer integrated light rays through the Kerr metric to
make Gargantua. This module does the same job in the Schwarzschild limit
(a = 0), where the null geodesic collapses to one exact ordinary differential
equation for u = 1/r as a function of the angle phi swept around the hole:

    d^2u/dphi^2 + u = 3 M u^2

The `3 M u^2` term is the whole story. Delete it and you get a straight line
in polar coordinates; keep it and you get light bending, the photon sphere at
r = 3M, the shadow of radius sqrt(27) M, the Einstein ring, and the disc that
appears to arc over the top and under the bottom of the hole at the same time.

Each pixel is traced backwards from the camera. Because Schwarzschild is
spherically symmetric every ray stays in the plane spanned by the camera
position and the ray direction, so the 3-D problem reduces to this 1-D ODE
plus bookkeeping to rotate back into world space.
"""

import numpy as np

from .colorimetry import blackbody_rgb_lut, tonemap
from .sky import build_sky, sample_sky

HORIZON = 2.0          # r = 2M, in units of M
PHOTON_SPHERE = 3.0    # unstable circular photon orbit
SHADOW_B = 27.0**0.5   # critical impact parameter, 3*sqrt(3) M


def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _deriv(u, up):
    """RHS of the geodesic ODE written as a first-order system."""
    uc = np.minimum(u, 5.0)  # keep captured rays from overflowing mid-step
    return up, 3.0 * uc * uc - uc


def _rk4_step(u, up, h):
    k1u, k1p = _deriv(u, up)
    k2u, k2p = _deriv(u + 0.5 * h * k1u, up + 0.5 * h * k1p)
    k3u, k3p = _deriv(u + 0.5 * h * k2u, up + 0.5 * h * k2p)
    k4u, k4p = _deriv(u + h * k3u, up + h * k3p)
    u_next = u + (h / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    up_next = up + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
    return u_next, up_next


def disk_temperature(r, r_in, t_peak):
    """Shakura-Sunyaev / Novikov-Thorne thin disc profile.

        T(r) ~ [ (1 - sqrt(r_in/r)) / r^3 ]^(1/4)

    Normalised so the maximum equals `t_peak`. That maximum sits at
    r = (49/36) r_in, from d/dx[(1 - sqrt(x)) x^3] = 0 with x = r_in/r.
    """
    r = np.maximum(r, r_in)
    shape = (1.0 - np.sqrt(r_in / r)) / r**3
    r_peak = (49.0 / 36.0) * r_in
    peak = (1.0 - np.sqrt(r_in / r_peak)) / r_peak**3
    return t_peak * (np.maximum(shape, 0.0) / peak) ** 0.25


def build_camera_rays(width, height, cam_r, inclination_deg, fov_deg):
    """Pinhole camera at radius `cam_r`, aimed at the hole."""
    incl = np.radians(inclination_deg)
    cam_pos = cam_r * np.array([np.sin(incl), 0.0, np.cos(incl)])

    forward = _normalize(-cam_pos)
    right = _normalize(np.cross(forward, np.array([0.0, 0.0, 1.0])))
    up = np.cross(right, forward)

    aspect = width / height
    half = np.tan(np.radians(fov_deg) * 0.5)
    xs = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    ys = 1.0 - (np.arange(height) + 0.5) / height * 2.0
    gx, gy = np.meshgrid(xs * half * aspect, ys * half)

    dirs = forward[None, None, :] + gx[..., None] * right + gy[..., None] * up
    return cam_pos, _normalize(dirs).reshape(-1, 3)


def trace(
    cam_pos,
    dirs,
    disk_in,
    disk_out,
    dphi=0.02,
    max_turns=5.0,
    escape_r=600.0,
):
    """Integrate every ray until it is captured, hits the disc, or escapes.

    Returns (outcome, hit_radius, b_total, normal_z, escape_dir) where outcome
    is 0 = swallowed, 1 = escaped to the sky, 2 = struck the disc.
    """
    n = dirs.shape[0]
    cam_r = float(np.linalg.norm(cam_pos))

    # Per-ray orbital frame: e1 along the camera position, e2 along the
    # tangential part of the ray, and the plane normal n = e1 x e2.
    e1 = np.broadcast_to(cam_pos / cam_r, dirs.shape).copy()
    v_r = np.einsum("ij,ij->i", dirs, e1)
    tangential = dirs - v_r[:, None] * e1
    v_t = np.linalg.norm(tangential, axis=1)
    v_t = np.maximum(v_t, 1e-12)  # a perfectly radial ray never advances phi
    e2 = tangential / v_t[:, None]
    normal_z = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]  # (e1 x e2)_z

    u = np.full(n, 1.0 / cam_r)
    up = -v_r / (cam_r * v_t)
    phi = np.zeros(n)

    # Conserved impact parameter: 1/b^2 = u'^2 + u^2 (1 - 2Mu).
    b_total = 1.0 / np.sqrt(np.maximum(up**2 + u**2 * (1.0 - 2.0 * u), 1e-30))

    outcome = np.zeros(n, dtype=np.int8)
    hit_r = np.zeros(n)
    escape_dir = np.zeros((n, 3))

    active = np.arange(n)
    steps = int(max_turns * 2.0 * np.pi / dphi)

    for _ in range(steps):
        if active.size == 0:
            break

        s_old = e1[:, 2] * np.cos(phi) + e2[:, 2] * np.sin(phi)
        r_old = 1.0 / u

        u_new, up_new = _rk4_step(u, up, dphi)
        phi_new = phi + dphi
        r_new = 1.0 / np.maximum(u_new, 1e-12)
        s_new = e1[:, 2] * np.cos(phi_new) + e2[:, 2] * np.sin(phi_new)

        # --- equatorial plane crossing -> possible disc hit ---
        crossed = (s_old * s_new) < 0.0
        if crossed.any():
            frac = s_old / (s_old - s_new)
            r_cross = r_old + frac * (r_new - r_old)
            hit = crossed & (r_cross >= disk_in) & (r_cross <= disk_out)
            if hit.any():
                idx = active[hit]
                outcome[idx] = 2
                hit_r[idx] = r_cross[hit]

        # --- horizon capture ---
        captured = u_new >= (1.0 / HORIZON)

        # --- escape to infinity (moving outwards past the escape radius) ---
        escaped = (r_new > escape_r) & (up_new < 0.0)
        if escaped.any():
            cp, sp = np.cos(phi_new[escaped]), np.sin(phi_new[escaped])
            e_r = cp[:, None] * e1[escaped] + sp[:, None] * e2[escaped]
            e_phi = -sp[:, None] * e1[escaped] + cp[:, None] * e2[escaped]
            dr_dphi = -up_new[escaped] / u_new[escaped] ** 2
            vec = dr_dphi[:, None] * e_r + (1.0 / u_new[escaped])[:, None] * e_phi
            idx = active[escaped]
            outcome[idx] = 1
            escape_dir[idx] = _normalize(vec)

        done = crossed & (outcome[active] == 2)
        done |= captured | escaped
        if done.any():
            keep = ~done
            active = active[keep]
            u, up, phi = u_new[keep], up_new[keep], phi_new[keep]
            e1, e2 = e1[keep], e2[keep]
        else:
            u, up, phi = u_new, up_new, phi_new

    return outcome, hit_r, b_total, normal_z, escape_dir


def shade(
    outcome,
    hit_r,
    b_total,
    normal_z,
    escape_dir,
    cam_r,
    disk_in,
    disk_out,
    disk_temp,
    sky,
    doppler=True,
    disk_gain=1.0,
    sky_gain=1.0,
):
    """Turn ray outcomes into linear radiance."""
    rgb = np.zeros((outcome.size, 3))

    # --- background stars ---
    sky_mask = outcome == 1
    if sky_mask.any():
        rgb[sky_mask] = sample_sky(sky, escape_dir[sky_mask]) * sky_gain

    # --- accretion disc ---
    disk_mask = outcome == 2
    if disk_mask.any():
        r = hit_r[disk_mask]
        temp = disk_temperature(r, disk_in, disk_temp)
        # Feather the outer rim so the disc does not end on a hard geometric
        # edge. The inner rim needs no taper: the (1 - sqrt(r_in/r)) stress
        # term in the Novikov-Thorne profile already sends the flux to zero
        # at the ISCO, and fading it further would mute the fast-orbiting
        # inner ring where relativistic beaming is strongest.
        taper = np.clip((disk_out - r) / (0.28 * disk_out), 0.0, 1.0)

        if doppler:
            # Emitter: circular geodesic, prograde about +z.
            #   dt/dtau = 1/sqrt(1 - 3M/r),  Omega = M^(1/2)/r^(3/2)
            # The photon reaching us runs opposite to the traced ray, so its
            # conserved angular momentum about z is -b_total * n_z.
            u_t = 1.0 / np.sqrt(np.maximum(1.0 - 3.0 / r, 1e-9))
            omega = r ** (-1.5)
            b_z = -b_total[disk_mask] * normal_z[disk_mask]
            g_inf = 1.0 / np.maximum(u_t * (1.0 - omega * b_z), 1e-6)
            # Camera sits at finite radius, so it is slightly less redshifted
            # than an observer at infinity.
            g = g_inf / np.sqrt(1.0 - 2.0 / cam_r)
        else:
            g = np.ones_like(r)

        # Liouville: I_nu/nu^3 is invariant, so bolometric I scales as g^4.
        # A blackbody stays a blackbody at the shifted temperature g*T.
        emitted = (temp / disk_temp) ** 4 * taper
        rgb[disk_mask] = blackbody_rgb_lut(g * temp) * (emitted * g**4 * disk_gain)[:, None]

    return rgb


def render(
    width=960,
    height=540,
    cam_r=150.0,
    inclination_deg=88.5,
    fov_deg=14.0,
    disk_in=6.0,
    disk_out=40.0,
    disk_temp=4500.0,
    doppler=False,
    supersample=2,
    exposure=1.0,
    disk_gain=4.0,
    sky_gain=1.0,
    dphi=0.02,
    max_turns=5.0,
    sky=None,
    progress=None,
):
    """Render Gargantua. Returns an (H, W, 3) uint8 image.

    `doppler=False` reproduces the film's look: Double Negative deliberately
    switched off Doppler shift and relativistic beaming, which would otherwise
    make one side of the disc overwhelmingly brighter and blue. `doppler=True`
    shows what the disc would actually look like.
    """
    if sky is None:
        sky = build_sky()

    ss = max(1, int(supersample))
    w, h = width * ss, height * ss

    cam_pos, dirs = build_camera_rays(w, h, cam_r, inclination_deg, fov_deg)
    if progress:
        progress(f"tracing {dirs.shape[0]:,} rays through d2u/dphi2 + u = 3Mu^2")

    outcome, hit_r, b_total, normal_z, escape_dir = trace(
        cam_pos, dirs, disk_in, disk_out, dphi=dphi, max_turns=max_turns
    )

    if progress:
        swallowed = int((outcome == 0).sum())
        progress(
            f"  {int((outcome == 2).sum()):,} hit the disc, "
            f"{int((outcome == 1).sum()):,} reached the sky, "
            f"{swallowed:,} crossed the horizon"
        )

    rgb = shade(
        outcome,
        hit_r,
        b_total,
        normal_z,
        escape_dir,
        cam_r,
        disk_in,
        disk_out,
        disk_temp,
        sky,
        doppler=doppler,
        disk_gain=disk_gain,
        sky_gain=sky_gain,
    )

    rgb = rgb.reshape(h, w, 3)
    if ss > 1:
        rgb = rgb.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))

    return tonemap(rgb, exposure=exposure)
