"""The equations behind Interstellar's black hole, Gargantua.

- `kerr`        the Kerr metric and Miller's planet time dilation
- `render`      null-geodesic ray tracing (d^2u/dphi^2 + u = 3Mu^2)
- `colorimetry` blackbody spectra through the CIE colour matching functions
"""

__all__ = ["kerr", "render", "colorimetry", "sky", "png"]
