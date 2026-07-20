################################################################################
# Functions for fitting D, Cs and x0
################################################################################
# -------------------
# Import packages
# -------------------
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
from helper.models import semi_infinite, infinite
from dataclasses import dataclass
from typing import Optional, Sequence

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
VALID_D_MODES = {"global", "per-timepoint"}

VALID_CS_MODES = {"global", "per-timepoint", "fixed"}

VALID_X0_MODES = {"global", "per-timepoint"}

MODEL_REGISTRY = {
    "semi-infinite" : semi_infinite,
    "infinite"      : infinite
}

# ------------------------------------------------------------------------------
# DataClasses
# ------------------------------------------------------------------------------
@dataclass
class FitConfig:
    """Configuration for diffusion fitting"""
    model:          str = "semi-infinite"           # semi-infinite OR infinite
    fit_x0:         bool = False                    # allow boundary to shift or no
    x0_mode:        str = "global"                  # global OR per-timepoint (only used if fit_x0 = True)
    x0_bounds:      Optional[tuple] = None # px size = 0.166 mm, allow x0 to shift up or down 5 px
    d_mode:         str = "per-timepoint"           # per-timepoint OR global
    cs_mode:        str = "per-timepoint"           # per-timepoint OR global OR fixed
    cs_fixed:       Optional[float] = None          # Cs value to be fixed
    fit_indices:    Optional[Sequence[int]] = None  # List of range of timepoint indices to be fit
    d_init:         float = 1e-4                    # Initial estimate of D
    min_points:     int = 3                         # Minimum num of datapoints required for fit

    # Check to ensure cs_fixed is provided when cs_mode = 'fixed'
    def __post_init__(self):
        if self.model not in MODEL_REGISTRY:
            raise ValueError(f"Model must be one of {set(MODEL_REGISTRY.keys())}")

        if self.d_mode not in VALID_D_MODES:
            raise ValueError(f"d_mode must be one of {VALID_D_MODES}")
        
        if self.cs_mode not in VALID_CS_MODES:
            raise ValueError(f"cs_mode must be one of {VALID_CS_MODES}")
        
        if self.cs_mode == "fixed" and self.cs_fixed is None:
            raise ValueError("cs_fixed must be provided when cs_mode = 'fixed'")
        
        if self.fit_x0 and self.x0_mode not in VALID_X0_MODES:
            raise ValueError(f"x0_mode must be one of {VALID_X0_MODES}")

        if self.fit_x0 and self.x0_bounds is None:
            px = 0.166 # pixel size, maybe link back to DICOMs in the future
            self.x0_bounds = (-px * 3, px * 3)

        if self.x0_bounds is not None and self.x0_bounds[0] >= self.x0_bounds[1]:
            raise ValueError("x0_bounds must be (lower, upper) with lower < upper")

@dataclass
class FitResults:
    """Container for diffusion fitting results"""

    config:         FitConfig
    valid_indices:  Optional[np.ndarray] = None
    valid_times:    Optional[np.ndarray] = None
    n_obs_total:    Optional[int] = None
    n_params_total: Optional[int] = None

    # Core fitted values
    d_per_t:        Optional[np.ndarray] = None # per-timepoint D
    d_global:       Optional[float] = None
    se_d_per_t:     Optional[np.ndarray] = None
    se_d_global:    Optional[float] = None

    cs_per_t:       Optional[np.ndarray] = None
    cs_global:      Optional[float] = None
    se_cs_per_t:    Optional[np.ndarray] = None
    se_cs_global:   Optional[float] = None

    x0_per_t:       Optional[np.ndarray] = None
    se_x0_per_t:    Optional[np.ndarray] = None
    x0_global:      Optional[float]      = None
    se_x0_global:   Optional[float]      = None

    # Fit diagnostics
    r2_per_t:       Optional[np.ndarray] = None
    rmse_per_t:     Optional[np.ndarray] = None
    r2_global:      Optional[float] = None
    aic:            Optional[float] = None
    bic:            Optional[float] = None

    # Uncertainty
    lb_all:         Optional[np.ndarray] = None
    ub_all:         Optional[np.ndarray] = None
    corr:           Optional[np.ndarray] = None
    pcov:           Optional[np.ndarray] = None

# ------------------------------------------------------------------------------
# Routing function
# ------------------------------------------------------------------------------
def fit_diffusion(c_xt, x, t, config: FitConfig) -> FitResults:
    """
    Unified diffusion fitting function

    Parameters
    --------
    c_xt
    """
    # --- Fit model ---
    model_fn = MODEL_REGISTRY[config.model]

    # --- Collect valid profiles ---
    x_segments, c_segments, valid_times, valid_indices = _collect_segments(
        c_xt, x, t, config.fit_indices, config.min_points
    )

    # --- Build parameter structure ---
    param_specs = {
        "x0":   config.x0_mode if config.fit_x0 else "fixed",
        "D":    config.d_mode,
        "Cs":   config.cs_mode
    }

    p0, lower, upper, layout = _build_params(param_specs, x_segments, c_segments, config)

    # --- Model wrapper for curve_fit() ---
    def global_model(_x_all, *free_params):
        predicted = []
        
        for seg_idx, (x_seg, t_seg) in enumerate(zip(x_segments, valid_times)):
            x0_val = _get_param_value("x0", free_params, layout, seg_idx)
            d_val = _get_param_value("D", free_params, layout, seg_idx)
            cs_val = _get_param_value("Cs", free_params, layout, seg_idx)
            predicted.append(model_fn(x_seg, x0_val, d_val, cs_val, t_seg))

        return np.concatenate(predicted)
    
    x_all = np.concatenate(x_segments)
    c_all = np.concatenate(c_segments)

    # --- Initializing FitResults object to return
    results = FitResults(config = config,
                         valid_indices = np.array(valid_indices),
                         valid_times = np.array(valid_times))

    try:
        popt, pcov = curve_fit(
            global_model, 
            x_all, c_all,
            p0 = p0,
            bounds = (lower, upper)
        )
        
        _unpack_results(
            results, popt, pcov, layout,
            x_segments, c_segments, valid_times,
            valid_indices, model_fn
        )
    
    except Exception as e:
        print(f"Fit FAILED -> {type(e)}: {e}")
    
    return results

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def _collect_segments(c_xt, x, t, fit_indices, min_points):
    """
    Extracts valid (x, concentration) profiles for each timepoint to be fit

    Params
    --------
    c_xt        : 2D array, shape (T, X) - concentration profiles over time
    x           : 1D array - depth/position
    t           : 1D array - timepoints corresponding to rows of c_xt
    fit_indices : indices into c_xt to consider fitting (None = all except t=0)
    min_points  : minimum number of valid points required to include a timepoint

    Returns
    --------
    x_segments      : list of 1D arrays, one per valid timepoint
    c_segments      : list of 1D arrays, one per valid timepoint
    valid_times     : list of float, time value for each valid segment
    valid_indices   : list of int, original row index in c_xt for each valid segment
    """
    T, _ = c_xt.shape

    if fit_indices is None:
        fit_indices = range(1, T)
    
    # Collecting valid profiles
    x_segments = []
    c_segments = []
    valid_times = []
    valid_indices = []

    for i in fit_indices:
        t_i = t[i]
        if t_i <= 0:
            continue
        c_profile = c_xt[i, :]
        mask_valid = np.isfinite(c_profile)

        x_fit = x[mask_valid]
        c_fit = c_profile[mask_valid]

        if len(x_fit) < min_points:
            print(f"t = {t_i}: {len(x_fit)} points valid.")
            continue
            
        x_segments.append(x_fit)
        c_segments.append(c_fit)
        valid_times.append(t_i)
        valid_indices.append(i)
    
    return x_segments, c_segments, valid_times, valid_indices

def _build_params(param_specs, x_segments, c_segments, config):
    """Builds p0, lower, upper and a layout dict indicating which indices in the 
    flat parameter vector correspond to which (param_name, segment_idx)."""
    
    n_t = len(x_segments)
    
    p0 = []
    lower = []
    upper = []
    layout = {}

    for param_name, mode in param_specs.items():
        if mode == "fixed":
            # no free parameters added
            fixed_val = _get_fixed_value(param_name, config, x_segments)
            layout[param_name] = {"mode": "fixed", "value": fixed_val}
        
        elif mode == "global":
            # one shared free parameter
            idx = len(p0)
            p0.append(_initial_guess(param_name, config, x_segments, c_segments))
            lower.append(_lower_bound(param_name, config))
            upper.append(_upper_bound(param_name, config))
            layout[param_name] = {"mode": "global", "index": idx}
        
        elif mode == "per-timepoint":
            # one free parameter per segment
            indices = []
            for i in range(n_t):
                idx = len(p0)

                p0.append(_initial_guess(param_name, config, x_segments, c_segments, seg_idx = i))
                lower.append(_lower_bound(param_name, config))
                upper.append(_upper_bound(param_name, config))
                indices.append(idx)

            layout[param_name] = {"mode": "per-timepoint", "indices": indices}

    return p0, lower, upper, layout

def _get_param_value(param_name, params, layout, seg_idx):
    info = layout[param_name]

    if info["mode"] == "fixed":
        return info["value"]
    
    if info["mode"] == "global":
        return params[info["index"]]
    
    if info["mode"] == "per-timepoint":
        return params[info["indices"][seg_idx]]

    raise ValueError(f"Unknown mode {info['mode']}")

def _get_fixed_value(param_name, config, x_segments):
    if param_name == "x0":
        # assumes boundary is constant across all timepoints
        return x_segments[0][0] # boundary depth, i.e., first x-value
    if param_name == "Cs":
        return config.cs_fixed
    raise ValueError(f"No fixed value logic for {param_name}")

def _initial_guess(param_name, config, x_segments, c_segments, seg_idx = None):
    if param_name == "D":
        return config.d_init
    if param_name == "x0":
        seg = x_segments[seg_idx] if seg_idx is not None else x_segments[0]
        return seg[0]
    if param_name == "Cs":
        seg = c_segments[seg_idx] if seg_idx is not None else c_segments[0]
        return np.nanmax(seg)
    
    raise ValueError(f"No initial guess logic for {param_name}")

def _lower_bound(param_name, config):
    if param_name == "x0" and config.fit_x0:
        return config.x0_bounds[0]
    else:
        return 0 # given all params are non-negative

def _upper_bound(param_name, config):
    if param_name == "x0" and config.fit_x0:
        return config.x0_bounds[1]
    else:
        return np.inf

def _unpack_results(result, popt, pcov, layout,
                    x_segments, c_segments, valid_times,
                    valid_indices, model_fn, alpha = 0.05):
    """Top-level unpacking - calls three focused helpers"""
    stdevs = _check_and_extract_stdevs(pcov)
    _unpack_parameters(result, popt, stdevs, layout, valid_indices)

    n_obs = len(np.concatenate(c_segments))
    n_params = len(popt)

    _compute_uncertainty(result, popt, stdevs, pcov, n_obs, n_params, alpha)
    _compute_diagnostics(result, popt, layout, x_segments, c_segments, valid_times,
                        valid_indices, model_fn)

def _check_and_extract_stdevs(pcov):
    """Validates pcov and returns standard errors"""
    if np.any(np.diag(pcov) < 0):
        print("WARNING: Negative variance in pcov - fit may be ill-conditioned")
    
    if np.any(~np.isfinite(pcov)):
        print("WARNING: pcov contains inf/nan - parameters may be unidentifiable")
    
    return np.sqrt(np.diag(pcov))

def _unpack_parameters(result, popt, stdevs, layout, valid_indices):
    T = max(valid_indices) + 1

    # --- D ---
    d_spec = layout["D"]
    if d_spec["mode"] == "global":
        result.d_global     = popt[d_spec["index"]]
        result.se_d_global  = stdevs[d_spec["index"]]
    
    elif d_spec["mode"] == "per-timepoint":
        d_per_t             = np.full(T, np.nan)
        se_d_per_t          = np.full(T, np.nan)

        for seg_idx, idx in enumerate(valid_indices):
            param_idx       = d_spec["indices"][seg_idx]
            d_per_t[idx]    = popt[param_idx]
            se_d_per_t[idx] = stdevs[param_idx]
        
        result.d_per_t      = d_per_t
        result.se_d_per_t   = se_d_per_t

    # --- Cs ---
    cs_spec = layout["Cs"]
    if cs_spec["mode"] == "fixed":
        result.cs_global    = result.config.cs_fixed
    
    elif cs_spec["mode"] == "global":
        result.cs_global    = popt[cs_spec["index"]]
        result.se_cs_global = stdevs[cs_spec["index"]]
    
    elif cs_spec["mode"] == "per-timepoint":
        cs_per_t            = np.full(T, np.nan)
        se_cs_per_t         = np.full(T, np.nan)

        for seg_idx, idx in enumerate(valid_indices):
            param_idx       = cs_spec["indices"][seg_idx]
            cs_per_t[idx]   = popt[param_idx]
            se_cs_per_t[idx] = stdevs[param_idx]

        result.cs_per_t     = cs_per_t
        result.se_cs_per_t  = se_cs_per_t
    
    # --- x0 ---
    x0_spec = layout["x0"]
    if x0_spec["mode"] == "per-timepoint":
        x0_per_t            = np.full(T, np.nan)
        se_x0_per_t         = np.full(T, np.nan)

        for seg_idx, idx in enumerate(valid_indices):
            param_idx       = x0_spec["indices"][seg_idx]
            x0_per_t[idx]   = popt[param_idx]
            se_x0_per_t[idx] = stdevs[param_idx]
        
        result.x0_per_t     = x0_per_t
        result.se_x0_per_t  = se_x0_per_t
    
    elif x0_spec["mode"] == "global":
        result.x0_global    = popt[x0_spec["index"]]
        result.se_x0_global = stdevs[x0_spec["index"]]

def _compute_uncertainty(result, popt, stdevs, pcov, n_obs, n_params, alpha = 0.05):
    """Computes CI bounds and corrrelation matrix"""
    df = max(0, n_obs - n_params)
    tval = stats.t.ppf(1.0 - alpha / 2.0, df)
    ci = tval * stdevs

    result.lb_all   = popt - ci
    result.ub_all   = popt + ci
    result.pcov     = pcov

    with np.errstate(invalid = "ignore"):
        result.corr = pcov / np.outer(stdevs, stdevs)

def _compute_diagnostics(result, popt, layout, x_segments, c_segments,
                         valid_times, valid_indices, model_fn):
    """Computes per-segment R2 and RMSE"""
    T = max(valid_indices) + 1
    n_t = len(valid_indices)

    r2_per_t = np.full(T, np.nan)
    rmse_per_t = np.full(T, np.nan)

    n_free_global = sum(
        1 for name in ("x0", "D", "Cs")
        if layout[name]["mode"] == "global"
    )

    n_free_per_t = sum(
        1 for name in ("x0", "D", "Cs")
        if layout[name]["mode"] == "per-timepoint"
    )

    n_free_per_segment = n_free_per_t + n_free_global / max(1, n_t)

    ss_res_total = 0.0
    n_obs_total = 0

    for seg_idx, idx in enumerate(valid_indices):
        x0_val = _get_param_value("x0", popt, layout, seg_idx)
        d_val = _get_param_value("D", popt, layout, seg_idx)
        cs_val = _get_param_value("Cs", popt, layout, seg_idx)

        c_pred = model_fn(x_segments[seg_idx], x0_val, d_val,
                          cs_val, valid_times[seg_idx])
        
        c_seg = c_segments[seg_idx]

        ss_res = np.sum((c_seg - c_pred) ** 2)
        ss_tot = np.sum((c_seg - np.mean(c_seg)) ** 2)
        
        r2_per_t[idx] = 1 - (ss_res / ss_tot)
        rmse_per_t[idx] = np.sqrt(ss_res / max(1, len(c_seg) - n_free_per_segment))

        ss_res_total += ss_res
        n_obs_total += len(c_seg)
    
    result.r2_per_t     = r2_per_t
    result.rmse_per_t   = rmse_per_t
    result.r2_global    = float(np.nanmean(r2_per_t[valid_indices]))

    # --- AIC/BIC using total residuals across all segments ---
    k = len(popt)
    n = n_obs_total

    if n > 0 and ss_res_total > 0:
        result.aic = n * np.log(ss_res_total / n) + 2 * k
        result.bic = n * np.log(ss_res_total / n) + k * np.log(n)

    else:
        result.aic = np.nan
        result.bic = np.nan

    result.n_obs_total = n
    result.n_params_total = k