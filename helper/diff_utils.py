# Diffusion model
from scipy.special import erfc
import numpy as np

def diff_profile(x, d, cs, t):
    """Analytical solution to Fick's second law of diffusion."""
    return cs * (erfc(x / (2.0 * np.sqrt(d * t))))