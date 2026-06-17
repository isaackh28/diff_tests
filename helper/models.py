################################################################################
# Analytical solutions to Fick's second law
################################################################################
# -------------------
# Import packages
# -------------------
from scipy.special import erfc
import numpy as np

# -------------------
# Models
# -------------------
def semi_infinite(x, x0, D, Cs, t):
    """
    Semi-infinite erfc model of diffusion with boundary at x0.
    """
    return Cs * erfc((x - x0) / (2 * np.sqrt(D * t)))

def infinite(x, x0, D, Cs, t):
    """
    Inifinite erfc model of diffusion with source at x0.
    """
    return 0.5 * Cs * erfc((x - x0) / (2 * np.sqrt(D * t)))