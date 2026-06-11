# Diffusion model
from scipy.special import erfc
import numpy as np

def diff_profile(x, d, cs, t):
    """Analytical solution to Fick's second law of diffusion in a semi-infinite medium."""
    return cs * (erfc(x / (2.0 * np.sqrt(d * t))))

def diff_profile_inf(x, d, cs, t):
    """Analytical solution to Fick's second law of diffusion with infinite source and infinite medium."""
    return 0.5 * cs * (erfc(x / 2.0 * np.sqrt(d * t)))