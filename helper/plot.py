# Plotting profiles
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

def plot_prof(fig, ax,
              i, x, y, z,
              name, elem, cmap = plt.cm.YlOrRd,
              norm = None): # norm = norm only for plotting residuals
    """Creates a 3-D surface plot given x (depth), y (time), and z (concentration)."""
    
    if norm:
        surf = ax[i].plot_surface(x, y, z, cmap=cmap, norm=norm)
    else:
        surf = ax[i].plot_surface(x, y, z, cmap=cmap)
        
    ax[i].view_init(elev = 30, azim = -50, roll = 3)
    ax[i].set_xlabel("Depth (mm)")
    ax[i].set_ylabel("Time (s)")

    if norm: # for residual plots, centres at 0
        ax[i].set_zlabel(f"Residuals (mg {elem} mL$^{{-1}}$)")
    else:
        ax[i].set_zlabel(f"Concentration (mg {elem} mL$^{{-1}}$)")

    ax[i].set_yticks(np.arange(np.min(y), np.max(y), 300))
    ax[i].set_title(name)
    fig.colorbar(surf, aspect = 20, shrink = 0.5, pad = 0.12)