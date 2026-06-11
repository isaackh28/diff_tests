import scipy
from scipy.optimize import curve_fit
import numpy as np
from helper.diff_utils import diff_profile, diff_profile_inf

fit_model = diff_profile_inf

# --------
# Per-Timepoint D, Per-Timepoint Cs
# --------
def diff_free(c_xt, x, time,
              fit_indices=None,
              d_init=1e-4,
              min_points=3):
    T, _ = c_xt.shape
    d_per_t = np.full(T, np.nan)
    cs_per_t = np.full(T, np.nan)
    r2_per_t = np.full(T, np.nan)
    se_d_per_t = np.full(T, np.nan)
    se_cs_per_t = np.full(T, np.nan)
    lb_d_per_t = np.full(T, np.nan)
    ub_d_per_t = np.full(T, np.nan)
    lb_cs_per_t = np.full(T, np.nan)
    ub_cs_per_t = np.full(T, np.nan)
    corr_per_t = {}

    if fit_indices is None:
        fit_indices = range(1, T)

    alpha = 0.05

    print("--------")
    print("Reporting Results")
    print("--------")
    
    for i in fit_indices:
        t_i = time[i]
        if t_i <= 0:
            continue

        c_profile = c_xt[i, :]
        mask_valid = np.isfinite(c_profile)
        x_fit = x[mask_valid].flatten()
        c_fit = c_profile[mask_valid].flatten()

        if len(x_fit) < min_points:
            print(f"t = {t_i}: {len(x_fit)} points valid. Skipping.")
            continue

        p0 = [d_init, np.nanmax(c_fit)]
        lower = [0, 0]
        upper = [np.inf, np.inf]

        def model(x_seg, D, Cs):
            return fit_model(x_seg, D, Cs, t_i)
        
        try:
            popt, pcov = curve_fit(
                model,
                x_fit, c_fit,
                p0=p0,
                bounds=(lower, upper)
            )

            D_fit, Cs_fit = popt

            # pcov diagnostics
            if np.any(np.diag(pcov) < 0):
                print(f"t = {t_i}: WARNING: Negative variance in pcov.")
            if np.any(~np.isfinite(pcov)):
                print(f"t = {t_i}: WARNING: pcov contains inf/nan.")

            stdevs = np.sqrt(np.diag(pcov))
            n_obs = len(c_fit)
            df = max(0, n_obs - 2)
            tval = scipy.stats.t.ppf(1.0 - alpha / 2.0, df)
            ci = tval * stdevs

            # Store results
            d_per_t[i] = D_fit
            cs_per_t[i] = Cs_fit
            se_d_per_t[i] = stdevs[0]
            se_cs_per_t[i] = stdevs[1]
            lb_d_per_t[i] = popt[0] - ci[0]
            ub_d_per_t[i] = popt[0] + ci[0]
            lb_cs_per_t[i] = popt[1] - ci[1]
            ub_cs_per_t[i] = popt[1] + ci[1]

            # Correlation matrix (2x2 for D and Cs)
            with np.errstate(invalid='ignore'):
                corr_per_t[i] = pcov / np.outer(stdevs, stdevs)

            # R2
            c_pred = fit_model(x_fit, D_fit, Cs_fit, t_i)
            ss_res = np.sum((c_fit - c_pred) ** 2)
            ss_tot = np.sum((c_fit - np.mean(c_fit)) ** 2)
            r2_per_t[i] = 1 - (ss_res / ss_tot)

            print(f"t = {t_i} s: "
                  f"D = {D_fit:.3e}, SE = {stdevs[0]:.3e} "
                  f"(95% CI [{lb_d_per_t[i]:.3e}, {ub_d_per_t[i]:.3e}]), "
                  f"Cs = {Cs_fit:.3f}, SE = {stdevs[1]:.3f} "
                  f"(95% CI [{lb_cs_per_t[i]:.3f}, {ub_cs_per_t[i]:.3f}]), "
                  f"R2 = {r2_per_t[i]:.4f}, "
                  f"D-Cs corr = {corr_per_t[i][0, 1]:.3f}")

        except Exception as e:
            print(f"t = {t_i}: Fit FAILED -> {type(e)}: {e}")

    return {
        'd_per_t':     d_per_t,
        'se_d_per_t':  se_d_per_t,
        'cs_per_t':    cs_per_t,
        'se_cs_per_t': se_cs_per_t,
        'r2_per_t':    r2_per_t,
        'lb_d_per_t':  lb_d_per_t,
        'ub_d_per_t':  ub_d_per_t,
        'lb_cs_per_t': lb_cs_per_t,
        'ub_cs_per_t': ub_cs_per_t,
        'corr_per_t':  corr_per_t,   # dict of 2x2 corr matrices keyed by time index
    }

# --------
# Global D, Free Cs
# --------
def diff_dglobal(c_xt, x, time,
                   c_s = None,
                   fit_indices = None,
                   d_init = 1e-4,
                   min_points = 3):
    T, X = c_xt.shape
    d_global = np.nan
    cs_per_t = np.full(T, np.nan)
    r2_per_t = np.full(T, np.nan)
    se_d = np.nan
    se_cs_per_t = np.full(T, np.nan)
    pcov_out = None
    corr_out = None
    r2_global = np.nan
    lb_all = None
    ub_all = None

    if fit_indices is None:
        fit_indices = range(1, T)

    # Collect valid profiles
    x_segments = []
    c_segments = []
    valid_times = []
    valid_indices = []
    
    for i in fit_indices:
        t_i = time[i]
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

    x_all = np.concatenate(x_segments)
    c_all = np.concatenate(c_segments)
    n_t = len(valid_times)

    p0 = [d_init] + [np.nanmax(c) for c in c_segments]
    lower = [0] + [0] * n_t
    upper = [np.inf] + [np.inf] * n_t

    def global_model(x_all, D, *Cs_per_t):
        predicted = []
        for x_seg, t_pred, Cs in zip(x_segments, valid_times, Cs_per_t):
            predicted.append(fit_model(x_seg, D, Cs, t_pred))
        return np.concatenate(predicted)

    try:
        popt, pcov = curve_fit(
            global_model,
            x_all, c_all,
            p0 = p0,
            bounds = (lower, upper)
        )

        d_global = popt[0]
        cs_values = popt[1:]

        # --- pcov diagnostics ---
        if np.any(np.diag(pcov) < 0):
            print("WARNING: Negative variance in pcov - fit may be ill-conditioned.")
        if np.any(~np.isfinite(pcov)):
            print("WARNING: pcov contains inf/nan - parameters may be unidentifiable.")

        # SE directly from pcov diagonal
        stdevs = np.sqrt(np.diag(pcov))
        se_d = stdevs[0]
        for param_idx, idx in enumerate(valid_indices):
            se_cs_per_t[idx] = stdevs[param_idx + 1]

        # Correlation matrix
        with np.errstate(invalid = "ignore"):
            corr_out = pcov / np.outer(stdevs, stdevs)
        pcov_out = pcov

        # 95%CI for printing only
        alpha = 0.05
        n = len(c_all)
        p = len(popt)
        df = max(0, n - p)
        tval = scipy.stats.t.ppf(1.0 - alpha / 2.0, df)
        ci = tval * stdevs
        lb_all = popt - ci
        ub_all = popt + ci

        # Per-timepoint R2 and Cs
        for param_idx, (idx, x_seg, c_seg, cs, t_fit) in enumerate(
            zip(valid_indices, x_segments, c_segments, cs_values, valid_times)
        ):
            cs_per_t[idx] = cs
            c_pred = fit_model(x_seg, d_global, cs, t_fit)
            ss_res = np.sum((c_seg - c_pred) ** 2)
            ss_tot = np.sum((c_seg - np.mean(c_seg)) ** 2)
            r2_per_t[idx] = 1 - (ss_res / ss_tot)

        r2_global = np.nanmean(r2_per_t[list(valid_indices)])

        # --- Reporting ---
        print(f"\nGlobal fit complete.")
        print(f"D = {d_global:.3e} mm2/s, SE = {se_d:.3e} (95%CI [{lb_all[0]:.3e}, {ub_all[0]:.3e}]")
        print(f"Note: SE and CI are symmetric (curve_fit linear approximation)")
        for param_idx, (idx, cs, t_fit) in enumerate(
            zip(valid_indices, cs_values, valid_times)
        ):
            lb_cs = lb_all[param_idx + 1]
            ub_cs = ub_all[param_idx + 1]
            print(f"t = {t_fit} s: Cs = {cs:.3f}, SE = {se_cs_per_t[idx]:.3f} (95%CI [{lb_cs:.3f}, {ub_cs:.3f}])")

        print(f"Mean per-timepoint R2 = {r2_global:.4f}")

        # Correlation matrix
        param_labels = ["D"] + [f"Cs_t{i}" for i in range(n_t)]
        print("\nCorrelation matrix:")
        print(f"{'':>10}", end = "")
        for label in param_labels:
            print(f"{label:>10}", end = "")
        print()
        for i, label in enumerate(param_labels):
            print(f"{label:>10}", end = "")
            for j in range(len(param_labels)):
                print(f"{corr_out[i, j]:>10.3f}", end = "")
            print()

    except Exception as e:
        print(f"Global fit FAILED -> {type(e)}: {e}")

    return{
        "d_global": d_global,
        "se_d": se_d,
        "cs_per_t": cs_per_t,
        "se_cs_per_t": se_cs_per_t,
        "r2_per_t": r2_per_t,
        "r2_global": r2_global,
        "pcov": pcov_out,
        "corr": corr_out,
        "lb_all": lb_all,
        "ub_all": ub_all,
    }

# --------
# Per-Timepoint D, Global Cs
# --------
def diff_csglobal(c_xt, x, time,
                   c_s = None,
                   fit_indices = None,
                   d_init = 1e-4,
                   min_points = 3):
    T, _ = c_xt.shape
    cs_global = np.nan
    d_per_t = np.full(T, np.nan)
    r2_per_t = np.full(T, np.nan)
    se_cs = np.nan
    se_d_per_t = np.full(T, np.nan)
    pcov_out = None
    corr_out = None
    r2_global = np.nan
    lb_all = None
    ub_all = None

    if fit_indices is None:
        fit_indices = range(1, T)

    # Collect valid profiles
    x_segments = []
    c_segments = []
    valid_times = []
    valid_indices = []
    
    for i in fit_indices:
        t_i = time[i]
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

    x_all = np.concatenate(x_segments)
    c_all = np.concatenate(c_segments)
    n_t = len(valid_times)

    p0 = [np.nanmax(c_all)] + [d_init] * n_t
    lower = [0] + [0] * n_t
    upper = [np.inf] + [np.inf] * n_t

    def global_model(x_all, Cs, *D_per_t):
        predicted = []
        for x_seg, t_pred, D in zip(x_segments, valid_times, D_per_t):
            predicted.append(fit_model(x_seg, D, Cs, t_pred))
        return np.concatenate(predicted)

    try:
        popt, pcov = curve_fit(
            global_model,
            x_all, c_all,
            p0 = p0,
            bounds = (lower, upper)
        )

        cs_global = popt[0]
        d_values = popt[1:]

        # --- pcov diagnostics ---
        if np.any(np.diag(pcov) < 0):
            print("WARNING: Negative variance in pcov - fit may be ill-conditioned.")
        if np.any(~np.isfinite(pcov)):
            print("WARNING: pcov contains inf/nan - parameters may be unidentifiable.")

        # SE directly from pcov diagonal
        stdevs = np.sqrt(np.diag(pcov))
        se_cs = stdevs[0]
        for param_idx, idx in enumerate(valid_indices):
            se_d_per_t[idx] = stdevs[param_idx + 1]

        # Correlation matrix
        with np.errstate(invalid = "ignore"):
            corr_out = pcov / np.outer(stdevs, stdevs)
        pcov_out = pcov

        # 95%CI for printing only
        alpha = 0.05
        n = len(c_all)
        p = len(popt)
        df = max(0, n - p)
        tval = scipy.stats.t.ppf(1.0 - alpha / 2.0, df)
        ci = tval * stdevs
        lb_all = popt - ci
        ub_all = popt + ci

        # Per-timepoint R2 and Cs
        for param_idx, (idx, x_seg, c_seg, d, t_fit) in enumerate(
            zip(valid_indices, x_segments, c_segments, d_values, valid_times)
        ):
            d_per_t[idx] = d
            c_pred = fit_model(x_seg, d, cs_global, t_fit)
            ss_res = np.sum((c_seg - c_pred) ** 2)
            ss_tot = np.sum((c_seg - np.mean(c_seg)) ** 2)
            r2_per_t[idx] = 1 - (ss_res / ss_tot)

        r2_global = np.nanmean(r2_per_t[list(valid_indices)])

        # --- Reporting ---
        print(f"\nGlobal fit complete.")
        print(f"Cs = {cs_global:.3e} mg/mL, SE = {se_cs:.3e} (95%CI [{lb_all[0]:.3e}, {ub_all[0]:.3e}])")
        print(f"Note: SE and CI are symmetric (curve_fit linear approximation)")
        for param_idx, (idx, d, t_fit) in enumerate(
            zip(valid_indices, d_values, valid_times)
        ):
            lb_cs = lb_all[param_idx + 1]
            ub_cs = ub_all[param_idx + 1]
            print(f"t = {t_fit} s: D = {d:.3e}, SE = {se_d_per_t[idx]:.3e} (95%CI [{lb_cs:.3e}, {ub_cs:.3e}])")

        print(f"Mean per-timepoint R2 = {r2_global:.4f}")

        # Correlation matrix
        param_labels = ["Cs"] + [f"D_t{i}" for i in range(n_t)]
        print("\nCorrelation matrix:")
        print(f"{'':>10}", end = "")
        for label in param_labels:
            print(f"{label:>10}", end = "")
        print()
        for i, label in enumerate(param_labels):
            print(f"{label:>10}", end = "")
            for j in range(len(param_labels)):
                print(f"{corr_out[i, j]:>10.3e}", end = "")
            print()

    except Exception as e:
        print(f"Global fit FAILED -> {type(e)}: {e}")

    return{
        "cs_global": cs_global,
        "se_cs": se_cs,
        "d_per_t": d_per_t,
        "se_d_per_t": se_d_per_t,
        "r2_per_t": r2_per_t,
        "r2_global": r2_global,
        "pcov": pcov_out,
        "corr": corr_out,
        "lb_all": lb_all,
        "ub_all": ub_all,
    }

# --------
# Global D, Global Cs
# --------
def diff_global(c_xt, x, time,
                   c_s = None,
                   fit_indices = None,
                   d_init = 1e-4,
                   min_points = 3,
                ):
    T, _ = c_xt.shape
    cs_global = np.nan
    d_global = np.nan
    r2_per_t = np.full(T, np.nan)
    se_d = np.nan
    se_cs_per_t = np.full(T, np.nan)
    pcov_out = None
    corr_out = None
    r2_global = np.nan
    lb_all = None
    ub_all = None

    if fit_indices is None:
        fit_indices = range(1, T)

    # Collect valid profiles
    x_segments = []
    c_segments = []
    valid_times = []
    valid_indices = []
    
    for i in fit_indices:
        t_i = time[i]
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

    x_all = np.concatenate(x_segments)
    c_all = np.concatenate(c_segments)
    n_t = len(valid_times)

    p0 = [np.nanmax(c_fit), d_init]
    lower = [0, 0]
    upper = [np.inf, np.inf]

    def global_model(x_all, Cs, D):
        predicted = []
        for x_seg, t_pred in zip(x_segments, valid_times):
            predicted.append(fit_model(x_seg, D, Cs, t_pred))
        return np.concatenate(predicted)

    try:
        popt, pcov = curve_fit(
            global_model,
            x_all, c_all,
            p0 = p0,
            bounds = (lower, upper)
        )

        cs_global = popt[0]
        d_global = popt[1]

        # --- pcov diagnostics ---
        if np.any(np.diag(pcov) < 0):
            print("WARNING: Negative variance in pcov - fit may be ill-conditioned.")
        if np.any(~np.isfinite(pcov)):
            print("WARNING: pcov contains inf/nan - parameters may be unidentifiable.")

        # SE directly from pcov diagonal
        stdevs = np.sqrt(np.diag(pcov))
        se_cs = stdevs[0]
        se_d = stdevs[1]

        # Correlation matrix
        with np.errstate(invalid = "ignore"):
            corr_out = pcov / np.outer(stdevs, stdevs)
        pcov_out = pcov

        # 95%CI for printing only
        alpha = 0.05
        n = len(c_all)
        p = len(popt)
        df = max(0, n - p)
        tval = scipy.stats.t.ppf(1.0 - alpha / 2.0, df)
        ci = tval * stdevs
        lb_all = popt - ci
        ub_all = popt + ci

        # Per-timepoint R2
        for param_idx, (idx, x_seg, c_seg, t_fit) in enumerate(
            zip(valid_indices, x_segments, c_segments, valid_times)
        ):
            c_pred = fit_model(x_seg, d_global, cs_global, t_fit)
            ss_res = np.sum((c_seg - c_pred) ** 2)
            ss_tot = np.sum((c_seg - np.mean(c_seg)) ** 2)
            r2_per_t[idx] = 1 - (ss_res / ss_tot)

        r2_global = np.nanmean(r2_per_t[list(valid_indices)])

        # --- Reporting ---
        print(f"\nGlobal fit complete.")
        print(f"D = {d_global:.3e}, Cs = {cs_global:.3e} mg/mL, SE = {se_cs:.3e} (95%CI [{lb_all[0]:.3e}, {ub_all[0]:.3e}]")
        print(f"Note: SE and CI are symmetric (curve_fit linear approximation)")

        print(f"Mean per-timepoint R2 = {r2_global:.4f}")

        # Correlation matrix
        param_labels = ["Cs"] + ['D']
        print("\nCorrelation matrix:")
        print(f"{'':>10}", end = "")
        for label in param_labels:
            print(f"{label:>10}", end = "")
        print()
        for i, label in enumerate(param_labels):
            print(f"{label:>10}", end = "")
            for j in range(len(param_labels)):
                print(f"{corr_out[i, j]:>10.3e}", end = "")
            print()

    except Exception as e:
        print(f"Global fit FAILED -> {type(e)}: {e}")

    return{
        "cs_global": cs_global,
        "se_d": se_d,
        "d_global": d_global,
        "se_cs_per_t": se_cs_per_t,
        "r2_per_t": r2_per_t,
        "r2_global": r2_global,
        "pcov": pcov_out,
        "corr": corr_out,
        "lb_all": lb_all,
        "ub_all": ub_all,
    }

# --------
# Global D, Fixed Experimental Cs
# --------
def diff_csfixed(c_xt, x, time,
                   cs_fixed, # scalar (global) or array of length T (per-timepoint)
                   fit_indices = None,
                   d_init = 1e-4,
                   min_points = 3,
                   print_res = True): # print results
    T, _ = c_xt.shape
    d_global = np.nan
    d_per_t = np.full(T, np.nan)
    r2_per_t = np.full(T, np.nan)
    se_d_global = np.nan
    se_d_per_t = np.full(T, np.nan)
    lb_d_global = np.nan
    ub_d_global = np.nan
    lb_d_per_t = np.full(T, np.nan)
    ub_d_per_t = np.full(T, np.nan)
    r2_global = np.nan

    # Normalize cs_fixed to always be an array of length T
    if np.isscalar(cs_fixed):
        cs_array = np.full(T, cs_fixed)
    else:
        cs_array = np.array(cs_fixed)
        assert len(cs_array) == T # cs_fixed array must have length T

    if fit_indices is None:
        fit_indices = range(1, T)

    # Collect valid profiles
    x_segments = []
    c_segments = []
    valid_times = []
    valid_indices = []
    cs_segments = []
    
    for i in fit_indices:
        t_i = time[i]
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
        cs_segments.append(cs_array[i])

    x_all = np.concatenate(x_segments)
    c_all = np.concatenate(c_segments)

    alpha = 0.05
    
    n_t = len(valid_times)

    # ---- Global D Fit ----
    def global_model(x_all, D):
        predicted = []
        for x_seg, t_pred, cs_t in zip(x_segments, valid_times, cs_segments):
            predicted.append(fit_model(x_seg, D, cs_t, t_pred))
        return np.concatenate(predicted)

    try:
        popt, pcov = curve_fit(
            global_model,
            x_all, c_all,
            p0 = d_init,
            bounds = ([0], [np.inf])
        )

        d_global = popt[0]
        se_d_global = np.sqrt(pcov[0, 0])
        n_obs = len(c_all)
        df = max(0, n_obs - 1)
        tval = scipy.stats.t.ppf(1.0 - alpha / 2.0, df)
        ci_d = tval * se_d_global
        lb_d_global = d_global - ci_d
        ub_d_global = d_global + ci_d
        
        # Per-timepoint R2
        for idx, x_seg, c_seg, t_fit, cs_t, in zip(
            valid_indices, x_segments, c_segments, valid_times, cs_segments
        ):
            c_pred = fit_model(x_seg, d_global, cs_t, t_fit)
            ss_res = np.sum((c_seg - c_pred) ** 2)
            ss_tot = np.sum((c_seg - np.mean(c_seg)) ** 2)
            r2_per_t[idx] = 1 - (ss_res / ss_tot)

        r2_global = np.nanmean(r2_per_t[list(valid_indices)])

        if print_res:
            # --- Reporting ---
            print(f"\nGlobal D fit complete.")
            print(f"D = {d_global:.3e}, SE = {se_d_global:.3e} "
                f"(95%CI [{lb_d_global:.3e}, {ub_d_global:.3e}]")
            print(f"Mean per-timepoint R2 = {r2_global:.4f}")

    except Exception as e:
        print(f"Global D fit FAILED -> {type(e)}: {e}")

    # ---- Per-timepoint D fit ----
    if print_res:
        print(f"\nPer-timepoint D fit:")
    for i, x_seg, c_seg, t_i, cs_t in zip(
        valid_indices, x_segments, c_segments, valid_times, cs_segments
    ):
        def model(x_seg, D, cs_t = cs_t): # capture cs_t in default arg
            return fit_model(x_seg, D, cs_t, t_i)

        try:
            popt_t, pcov_t = curve_fit(
                model,
                x_seg, c_seg,
                p0 = [d_init],
                bounds = ([0], [np.inf])
            )

            D_fit = popt_t[0]
            se_d = np.sqrt(pcov_t[0, 0])
            df_t = max(0, len(c_seg) - 1)
            tval_t = scipy.stats.t.ppf(1.0 - alpha / 2.0, df_t)
            ci_t = tval_t * se_d

            d_per_t[i] = D_fit
            se_d_per_t[i] = se_d
            lb_d_per_t[i] = D_fit - ci_t
            ub_d_per_t[i] = D_fit + ci_t

            c_pred = fit_model(x_seg, D_fit, cs_t, t_i)
            ss_res = np.sum((c_seg - c_pred) ** 2)
            ss_tot = np.sum((c_seg - np.mean(c_seg)) ** 2)

            if print_res:
                print(f"t = {t_i} s: Cs = {cs_t:.3f}, D = {D_fit:.3e}, "
                  f"SE = {se_d:.3e} "
                f"(95% CI[{lb_d_per_t[i]:.3e}, {ub_d_per_t[i]:.3e}]), "
                f"R2 = {1 - ss_res/ss_tot:.4f}")

        except Exception as e:
            print(f"t = {t_i}: Fit FAILED -> {type(e)}: {e}")
    
    if print_res:
        print(f"\nNote: SE and CI are symmetric (curve_fit linear approximation)")

    return{
        "d_global": d_global,
        "se_d_global": se_d_global,
        "lb_d_global": lb_d_global,
        "ub_d_global": ub_d_global,
        "d_per_t": d_per_t,
        "se_d_per_t": se_d_per_t,
        "lb_d_per_t": lb_d_per_t,
        "ub_d_per_t": ub_d_per_t,
        "r2_per_t": r2_per_t,
        "r2_global": r2_global
    }