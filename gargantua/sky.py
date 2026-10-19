"""A procedural star field for rays that escape to infinity.

Built once as an equirectangular texture so escaped rays are a texture lookup
rather than a search over every star. Without a background the lensing is
invisible -- the Einstein ring is made of *displaced stars*, so the sky is
part of the physics demo, not decoration.
"""

import numpy as np

from .colorimetry import blackbody_rgb


def build_sky(width=2048, height=1024, n_stars=24000, seed=20141107):
    """Return an (H, W, 3) float array of linear radiance."""
    rng = np.random.default_rng(seed)
    sky = np.zeros((height, width, 3), dtype=np.float32)

    # Faint dust lane so the empty regions are not pure black. The longitude
    # term must complete a whole number of cycles or the texture shows a seam
    # at the +/-pi wrap, which the lensing then smears across the frame.
    yy = np.linspace(-1.0, 1.0, height)[:, None]
    xx = (np.arange(width) / width)[None, :]
    band = np.exp(-((yy / 0.22) ** 2)) * (0.55 + 0.45 * np.sin(2 * np.pi * (2.0 * xx + 0.2)))
    haze = 0.010 * band + 0.0015
    sky += (haze[..., None] * np.array([0.42, 0.50, 1.00], dtype=np.float32)).astype(np.float32)

    # Stars uniform on the sphere: v uniform in cos(theta) avoids pole pile-up.
    u = rng.random(n_stars)
    v = rng.random(n_stars)
    px = (u * width).astype(np.int32) % width
    py = (np.arccos(1.0 - 2.0 * v) / np.pi * height).astype(np.int32).clip(0, height - 1)

    # Power-law brightnesses: many faint, a few bright.
    mag = rng.power(0.32, n_stars) * 3.4 + 0.02
    temps = np.clip(rng.lognormal(mean=np.log(5200.0), sigma=0.45, size=n_stars), 1800.0, 30000.0)
    colors = blackbody_rgb(temps) * mag[:, None]

    np.add.at(sky, (py, px), colors.astype(np.float32))

    # Bleed each star into its neighbours so they survive downsampling.
    # Longitude wraps, so roll rather than slice in x.
    for shift, weight in ((1, 0.28), (2, 0.06)):
        sky += weight * np.roll(sky, shift, axis=1)
        sky += weight * np.roll(sky, -shift, axis=1)
        sky[shift:, :] += weight * sky[:-shift, :]
        sky[:-shift, :] += weight * sky[shift:, :]

    return sky


def sample_sky(sky, directions):
    """Look up unit `directions` (N, 3) in an equirectangular texture."""
    height, width, _ = sky.shape
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    lon = np.arctan2(y, x)
    lat = np.arccos(np.clip(z, -1.0, 1.0))
    px = ((lon / (2 * np.pi) + 0.5) * width).astype(np.int32) % width
    py = np.clip((lat / np.pi * height).astype(np.int32), 0, height - 1)
    return sky[py, px]
