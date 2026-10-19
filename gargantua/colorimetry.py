"""Blackbody temperature -> sRGB, the honest way.

Planck's law is integrated against the CIE 1931 colour matching functions
using the analytic multi-lobe Gaussian fits from Wyman, Sloan & Shirley
(2013), then converted through sRGB primaries. The accretion disc is a
thermal emitter, so this is what turns a temperature map into pixels.
"""

import numpy as np

# Planck's law constants (SI).
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23

# Sample the visible band once and reuse.
_LAMBDA_NM = np.arange(380.0, 781.0, 2.0)
_LAMBDA_M = _LAMBDA_NM * 1e-9

# XYZ -> linear sRGB (IEC 61966-2-1, D65).
_XYZ_TO_RGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


def _piecewise_gauss(x, mu, sigma_lo, sigma_hi):
    sigma = np.where(x < mu, sigma_lo, sigma_hi)
    t = (x - mu) / sigma
    return np.exp(-0.5 * t * t)


def _cie_xyz_bar(lam_nm):
    """Analytic fits to xbar/ybar/zbar (Wyman, Sloan & Shirley 2013)."""
    x = (
        1.056 * _piecewise_gauss(lam_nm, 599.8, 37.9, 31.0)
        + 0.362 * _piecewise_gauss(lam_nm, 442.0, 16.0, 26.7)
        - 0.065 * _piecewise_gauss(lam_nm, 501.1, 20.4, 26.2)
    )
    y = 0.821 * _piecewise_gauss(lam_nm, 568.8, 46.9, 40.5) + 0.286 * _piecewise_gauss(
        lam_nm, 530.9, 16.3, 31.1
    )
    z = 1.217 * _piecewise_gauss(lam_nm, 437.0, 11.8, 36.0) + 0.681 * _piecewise_gauss(
        lam_nm, 459.0, 26.0, 13.8
    )
    return x, y, z


_XBAR, _YBAR, _ZBAR = _cie_xyz_bar(_LAMBDA_NM)


def _planck(lam_m, temperature):
    """Spectral radiance B_lambda(T); temperature may be an array."""
    t = np.asarray(temperature, dtype=np.float64)[..., None]
    lam = lam_m[None, :] if t.ndim > 1 else lam_m
    expo = _H * _C / (lam * _KB * np.maximum(t, 1.0))
    # Guard the exponential so cold pixels do not overflow.
    return (2.0 * _H * _C**2) / lam**5 / (np.expm1(np.minimum(expo, 700.0)))


def _blackbody_rgb_block(temperature):
    radiance = _planck(_LAMBDA_M, temperature)  # (N, n_lambda)

    xyz = np.stack(
        [
            radiance @ _XBAR,
            radiance @ _YBAR,
            radiance @ _ZBAR,
        ],
        axis=-1,
    )
    xyz /= np.maximum(xyz.sum(axis=-1, keepdims=True), 1e-30)

    rgb = xyz @ _XYZ_TO_RGB.T
    # Desaturate anything that fell outside the sRGB gamut instead of clipping
    # it to a hard edge, which would band across the disc.
    deficit = np.maximum(-rgb.min(axis=-1, keepdims=True), 0.0)
    rgb = rgb + deficit
    rgb /= np.maximum(rgb.max(axis=-1, keepdims=True), 1e-30)
    return np.clip(rgb, 0.0, 1.0)


def blackbody_rgb(temperature, chunk=8192):
    """Normalised linear-sRGB chromaticity of a blackbody at `temperature` (K).

    Returns an array shaped (N, 3) scaled so the brightest channel is 1;
    absolute brightness is applied separately by the renderer. Evaluated in
    blocks because the Planck integral is (N x n_lambda).
    """
    temperature = np.atleast_1d(np.asarray(temperature, dtype=np.float64)).ravel()
    out = np.empty((temperature.size, 3))
    for start in range(0, temperature.size, chunk):
        stop = min(start + chunk, temperature.size)
        out[start:stop] = _blackbody_rgb_block(temperature[start:stop])
    return out


_LUT_MIN, _LUT_MAX, _LUT_N = 200.0, 60000.0, 2048
_LUT_LOG = np.linspace(np.log(_LUT_MIN), np.log(_LUT_MAX), _LUT_N)
_LUT_RGB = None


def blackbody_rgb_lut(temperature):
    """Same mapping, interpolated from a log-temperature lookup table.

    The renderer shades millions of pixels whose colours vary smoothly with
    radius, so a 2048-entry table is visually identical to the exact integral
    and vastly cheaper.
    """
    global _LUT_RGB
    if _LUT_RGB is None:
        _LUT_RGB = blackbody_rgb(np.exp(_LUT_LOG))

    t = np.clip(np.asarray(temperature, dtype=np.float64), _LUT_MIN, _LUT_MAX)
    pos = np.log(t)
    return np.stack(
        [np.interp(pos, _LUT_LOG, _LUT_RGB[:, ch]) for ch in range(3)], axis=-1
    )


def tonemap(rgb, exposure=1.0, gamma=2.2):
    """Reinhard-style roll-off then gamma encode to display-referred bytes."""
    x = np.maximum(rgb * exposure, 0.0)
    x = x / (1.0 + x)
    x = np.clip(x, 0.0, 1.0) ** (1.0 / gamma)
    return (x * 255.0 + 0.5).astype(np.uint8)
