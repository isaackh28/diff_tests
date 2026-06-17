################################################################################
# Functions for fitting D, Cs and x0
################################################################################
# -------------------
# Import packages
# -------------------
import scipy
from scipy.optimize import curve_fit
import numpy as np
from helper import models

# -------------------
# fit_diffusion() function
# -------------------
def fit_diffusion(c_xt, x, time, 
                  model = "semi-infinite",  # semi-inifinite OR infinite
                  fit_x0 = False,           # allow boundary to shift or no
                  d_mode = "per-timepoint", # per-timepoint OR global
                  cs_mode = "per-timepoint",# per-timepoint OR global
                  fit_indices = None,
                  d_init = 1e-4,
                  min_points = 3):
    """
    
    """