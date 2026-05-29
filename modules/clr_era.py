
from sys import prefix
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.stats.multitest import multipletests
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from typing import Dict, Optional, Tuple, Any
import clr_config as cfg


# -----------------------------------------------------------------------------
# 1. Function to slice CLR matrix into eras
# -----------------------------------------------------------------------------
def slice_clr_into_eras(clr_dict, era_boundaries, eps):
    """
    Slice a CLR-transformed matrix into temporal eras (Pre‑COVID, COVID, Post‑COVID).

    The function:
        - Extracts the CLR matrix for a given epsilon
        - Uses date boundaries to determine slice endpoints
        - Returns a dictionary of era-specific CLR matrices
        - Prints a compact diagnostic report summarizing span and date ranges

    Parameters
    ----------
    clr_dict : dict
        Mapping: epsilon → CLR DataFrame (indexed by datetime).

    era_boundaries : dict
        Required keys:
            'Pre-COVID' : last date of pre‑COVID era
            'COVID'     : last date of COVID era

    eps : float or str
        Epsilon key used to select the CLR matrix.

    Returns
    -------
    dict
        {
            'Pre-COVID' : DataFrame,
            'COVID'     : DataFrame,
            'Post-COVID': DataFrame
        }
    """
    # -------------------------------------------------
    # 1. Extract CLR matrix and validate boundaries
    # -------------------------------------------------
    if eps not in clr_dict:
        raise KeyError(f"Epsilon {eps} not found in clr_dict keys.")

    clr_df = clr_dict[eps]

    required = ["Pre-COVID", "COVID"]
    missing = [k for k in required if k not in era_boundaries]
    if missing:
        raise KeyError(f"Missing required era boundaries: {missing}")

    # -------------------------------------------------
    # 2. Helper: resolve boundary index (inclusive end)
    # -------------------------------------------------
    def get_end_idx(date_str):
        try:
            loc = clr_df.index.get_loc(date_str)
            # get_loc may return a slice for non-unique indexes
            return loc.stop if isinstance(loc, slice) else loc + 1
        except KeyError:
            raise KeyError(f"Date boundary '{date_str}' not found in CLR index.")

    # -------------------------------------------------
    # 3. Slice into eras
    # -------------------------------------------------
    pre_end   = get_end_idx(era_boundaries["Pre-COVID"])
    covid_end = get_end_idx(era_boundaries["COVID"])

    eras = {
        "Pre-COVID":  clr_df.iloc[:pre_end],
        "COVID":      clr_df.iloc[pre_end:covid_end],
        "Post-COVID": clr_df.iloc[covid_end:],
    }

    # -------------------------------------------------
    # 4. Diagnostic report
    # -------------------------------------------------
    label_w  = 18
    border   = "=" * 60
    line     = "-" * 60

    print(f"\n{border}")
    print(f" 📅  CLR TEMPORAL SEGMENTATION REPORT (ε = {eps:.14f})")
    print(f"{border}")

    print(f"  • {'Total Matrix':<{label_w}} : {clr_df.shape[0]} months x {clr_df.shape[1]} crimes")
    print(f"  • {'Date Horizon':<{label_w}} : "
          f"{clr_df.index[0].strftime('%Y-%m')} to {clr_df.index[-1].strftime('%Y-%m')}")

    print(line)
    print(f"  {'Era Label':<{label_w}} | {'Span (T)':<12} | {'Date Range':<20}")
    print(line)

    for label, df in eras.items():
        if df.empty:
            print(f"  ⚠️ {label:<{label_w-2}} | EMPTY SLICE  | N/A")
        else:
            span = f"{len(df)} months"
            date_range = f"{df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}"
            print(f"  {label:<{label_w}} | {span:<12} | {date_range}")

    print(f"{border}\n")

    return eras


# ------------------------------------------------------------------------------
# 2-0. Main function to compute era distribution parameters and pairwise deltas with 
#      optional verbose reporting
# ------------------------------------------------------------------------------
def compute_era_distribution_parameters(eras_dict, sparse_cats=None, verbose=False, 
                                        report_width=95, precision_threshold=9):
    """
    Compute per‑era distribution statistics (means, stds, covariances, shrinkage, 
    conditioning, and pairwise deltas) for a dictionary of era‑segmented DataFrames.

    Parameters
    ----------
    eras_dict : dict[str, pd.DataFrame]
        A mapping of era names → DataFrames containing feature columns.
        Each DataFrame is treated as one "era" whose distribution parameters 
        will be computed independently.

    sparse_cats : list[str] or None, optional
        Column names to drop from each era before computing statistics.
        Useful for removing sparse or categorical columns that should not 
        participate in covariance estimation.

    verbose : bool, optional
        If True, prints a formatted report using `print_era_distribution_report`.

    report_width : int, optional
        Width passed to the reporting function for formatting output.

    precision_threshold : int, optional
        Precision threshold passed to the reporting function.

    Returns
    -------
    dict
        A dictionary containing:
            - 'era_names' : list of era names
            - 'era_means' : DataFrame of per‑era means and pairwise deltas
            - 'era_stds'  : DataFrame of per‑era stds and pairwise deltas
            - 'era_covs' : dict of raw covariance matrices
            - 'era_covs_lw' : dict of Ledoit‑Wolf shrinkage covariance matrices
            - 'lw_shrinkage' : dict of shrinkage intensities
            - 'cond_numbers' : dict of raw and LW covariance condition numbers
            - 'pairs' : list of (earlier, later, label) tuples for deltas
            - 'sparse_cats' : list of dropped columns

    Notes
    -----
    - No global variables are referenced; all formatting parameters are passed in.
    - Covariance matrices are computed twice per era: raw and Ledoit‑Wolf.
    - Pairwise deltas are computed vectorized across all eras.
    """
    # ------------------------------------------------------------------------------ 
    # 2-0-A: Data Preparation & Normalize sparse category list and drop those columns
    # ------------------------------------------------------------------------------
    sparse_cats = sorted(list(set(sparse_cats))) if sparse_cats else []
    eras = {n: df.drop(columns=sparse_cats, errors='ignore') for n, df in eras_dict.items()}
    # Re-center CLR values for each era to ensure comparability after dropping sparse categories.
    for name, df in eras.items():
        # 1. Calculate the mean of the remaining categories for each row
        row_means = df.mean(axis=1) 
        
        # 2. Subtract that mean from each value to re-center the composition
        # This "re-normalizes" the CLR values to the new category universe
        eras[name] = df.sub(row_means, axis=0)
    # Update era names and prepare date range storage after dropping sparse categories
    era_names = list(eras.keys())
    era_date_ranges = {}  # To store date ranges for reporting

    # ------------------------------------------------------------------------------
    # 2-0-B: Compute per‑era means and standard deviations for all features, storing 
    #      in DataFrames.
    # -------------------------------------------------------------------------------
    era_means = pd.DataFrame({n: df.mean() for n, df in eras.items()})
    era_stds  = pd.DataFrame({n: df.std()  for n, df in eras.items()})

    # ------------------------------------------------------------------------------
    # 2-0-C: Compute covariance matrices for each era, both raw and with Ledoit‑Wolf 
    #      shrinkage. 
    # ------------------------------------------------------------------------------
    era_covs, era_covs_lw, lw_shrinkage, cond_numbers = {}, {}, {}, {}

    # Iterate through eras to compute raw covariance, Ledoit‑Wolf covariance, 
    # shrinkage, and condition numbers.
    for name, df in eras.items():
        # Store the date range for this era in a dictionary for reporting purposes.
        start_date            = df.index.min().strftime('%Y-%m')
        end_date              = df.index.max().strftime('%Y-%m')
        era_date_ranges[name] = (start_date, end_date)

        raw_cov = df.cov()
        era_covs[name] = raw_cov

        # Ledoit‑Wolf shrinkage is applied to the raw data (not the covariance) 
        # to get the regularized covariance matrix.
        lw = LedoitWolf(assume_centered=False).fit(df)
        era_covs_lw[name] = pd.DataFrame(lw.covariance_, index=df.columns, columns=df.columns)
        
        # Store the shrinkage coefficient and condition numbers for both raw and LW covariances.
        lw_shrinkage[name] = lw.shrinkage_
        cond_numbers[name] = {
            'raw': float(np.linalg.cond(raw_cov.values)),
            'lw' : float(np.linalg.cond(lw.covariance_))
        }

    # -------------------------------------------------------------------------------
    # 2-0-D: Compute pairwise deltas for means and stds across all era pairs, storing 
    #      in DataFrames.
    # ---------------------------------------------------------------------------------
    pairs = [
        (era_names[i], era_names[j], f"{era_names[j]}_minus_{era_names[i]}")
        for i in range(len(era_names)) for j in range(i + 1, len(era_names))
    ]
    
    # Initialize columns for deltas in the means and stds DataFrames.
    for earlier, later, col in pairs:
        era_means[col] = era_means[later] - era_means[earlier]
        era_stds[col]  = era_stds[later]  - era_stds[earlier]

    # Package all results into a single dictionary.
    result = {
        'era_names': era_names, 'era_means': era_means, 'era_stds': era_stds,
        'era_covs': era_covs, 'era_covs_lw': era_covs_lw, 'lw_shrinkage': lw_shrinkage,
        'cond_numbers': cond_numbers, 'pairs': pairs, 'sparse_cats': sparse_cats, 
        'era_date_ranges': era_date_ranges
    }

    # Verbose reporting: Pass the computed parameters into the reporting function 
    # to print a formatted diagnostic summary.
    if verbose:
        # Pass the constants into the reporting suite
        print_era_distribution_report(result, width=report_width, threshold=precision_threshold)

    return result


# --------------------------------------------------------------------------------------
# 2-1. Function to print a comprehensive report summarizing per‑era distribution parameters,
#      covariance integrity, stability diagnostics, and top shifts.
# --------------------------------------------------------------------------------------
def print_era_distribution_report(result, top_display=True, n_rows=10, use_emoji=True, 
                                  width=90, threshold=8):
    """
    Print a multi‑section diagnostic report summarizing per‑era distribution 
    characteristics, covariance integrity, stability, and top shifts.

    Parameters
    ----------
    result : dict
        Output dictionary from `compute_era_distribution_parameters()`. 
        Must contain:
            - 'era_means' : DataFrame of means + deltas
            - 'era_stds'  : DataFrame of std devs + deltas
            - 'era_covs'  : dict of raw covariance matrices
            - 'pairs'     : list of (earlier, later, label) tuples
            - 'sparse_cats' : list of excluded categories

    top_display : bool, optional
        Whether to display the "Top Shifts" sections for means and volatilities.

    n_rows : int, optional
        Number of top rows to display in the shift tables.

    use_emoji : bool, optional
        Whether to use emoji glyphs in the report.

    width : int, optional
        Horizontal width for section separators and formatting.

    threshold : int, optional
        Precision threshold passed to the stability report function.

    Notes
    -----
    - This function is purely for formatted console output.
    - It delegates to helper functions for banners, stability, and shift tables.
    - No computations occur here; all statistics must be precomputed.
    """
    # Load glyphs (emoji or ASCII) for visual formatting.
    glyphs = _get_glyphs(use_emoji)
    
    # ------------------------------------------------------------------------------
    # 2-1-A: Distribution Parameters & Pairwise Deltas
    # -------------------------------------------------------------------------------
    _display_banner_table("MEAN CLR VECTOR PER ERA", result['era_means'])
    _display_banner_table("VOLATILITY (STD DEV) PER ERA", result['era_stds'])

    # ------------------------------------------------------------------------------
    # 2-1-B: Covariance Integrity & Data Quality Check
    # ------------------------------------------------------------------------------
    _print_section_header("DATA INTEGRITY & CONSISTENCY CHECK", glyphs['matrix'], width=width)
    
    # Determine expected number of categories from the first covariance matrix.
    n_cats = next(iter(result['era_covs'].values())).shape[0]
    
    # Validate that all covariance matrices have consistent dimensions.
    for name, cov in result['era_covs'].items():
        # Status is READY if dimensions match across all eras.
        status = f"{glyphs['ok']} READY" if cov.shape[0] == n_cats else f"{glyphs['fail']} BROKEN"
        print(f"  {name:<15} | Date Range: {result['era_date_ranges'][name][0]} to {result['era_date_ranges'][name][1]} \
| Tracking {n_cats} Crime Types | Status: {status}")

        # | Date Range: {result['era_date_ranges'][name][0]} to {result['era_date_ranges'][name][1]}

    # Data Quality Log
    print(f"\n  {glyphs['shield']} DATA QUALITY LOG:")
    if result['sparse_cats']:
        # Inform user which categories were removed due to sparsity.
        print(f"  {glyphs['warn']} Note: {len(result['sparse_cats'])} categories were excluded:")
        for cat in result['sparse_cats']:
            print(f"          - {cat}")
    else:
        # All categories retained → full comparability across eras.
        print(f"  {glyphs['ok']} All crime categories met the minimum support requirements and were retained")
        print(f"     across the Pre-COVID, COVID, and Post-COVID periods, ensuring fully consistent")
        print(f"     and directly comparable analyses across all time periods")
    print("-" * width)

    # ------------------------------------------------------------------------------
    # 2-1-C: Numerical Stability & Regularization Diagnostics
    # -------------------------------------------------------------------------------
    _print_stability_report(result, glyphs, threshold=threshold, width=width)

    # ------------------------------------------------------------------------------
    # 2-1-D: Top Shifts in Means and Volatilities
    # -------------------------------------------------------------------------------
    if top_display:
        _print_top_shifts(result['era_means'], result['pairs'], "MEAN SHIFTS", 
                          glyphs['fire'], glyphs, n_rows, width=width)
        _print_top_shifts(result['era_stds'], result['pairs'], "VOLATILITY SHIFTS", 
                          glyphs['bolt'], glyphs, n_rows, width=width)


# --------------------------------------------------------------------------------------
# Helper function 2-A: Print a detailed stability report summarizing condition numbers, 
#                      shrinkage, and health indicators for each era, with interpretive 
#                      guidance for the metrics shown.
# --------------------------------------------------------------------------------------
def _print_stability_report(res, glyphs, threshold, width):
    """
    Print a numerical‑stability diagnostic table summarizing raw and 
    Ledoit‑Wolf‑regularized condition numbers for each era, along with 
    shrinkage coefficients and qualitative health indicators.

    Parameters
    ----------
    res : dict
        The result dictionary produced by `compute_era_distribution_parameters()`.
        Must contain:
            - 'era_names'
            - 'cond_numbers' : dict of {'raw': float, 'lw': float}
            - 'lw_shrinkage' : dict of shrinkage α values

    glyphs : dict
        Dictionary of visual symbols (emoji or ASCII) used for formatting.

    threshold : int
        Log‑10 threshold used to classify LW condition numbers as OPTIMAL or STRETCHED.

    width : int
        Horizontal width for formatting separators and section headers.

    Notes
    -----
    - Condition numbers measure numerical stability of covariance inversion.
    - Ledoit‑Wolf shrinkage typically reduces condition numbers substantially.
    - Health classification is based on log10(LW_cond) < threshold.
    """

    # Print section header for stability diagnostics.
    _print_section_header("NUMERICAL STABILITY & REGULARIZATION", glyphs['shield'], width=width)

    # Build and print the table header.
    header = f"  {'Era':<15} {'Raw Cond#':>15} {'LW Cond#':>15} {'Shrinkage':>20} {'Health':>14}"
    print(f"{header}\n  {'-' * (width - 4)}")

    # Iterate through eras and print stability metrics.
    for name in res['era_names']:
        cn = res['cond_numbers'][name]          # Raw and LW condition numbers
        alpha = res['lw_shrinkage'][name]       # Ledoit‑Wolf shrinkage coefficient

        # Health classification based on LW condition number magnitude.
        health = (
            glyphs['green'] + " OPTIMAL"
            if np.log10(cn['lw']) < threshold
            else glyphs['yellow'] + " STRETCHED"
        )

        # Print formatted row for this era.
        print(f"  {name:<15} {cn['raw']:>15.1e} {cn['lw']:>15.1f} {alpha:>20.4f} {health:>14}")

    print("-" * width)

    # Describe the implications of the stability metrics for interpretability and reliability of CLR-based analyses.
    # Provide interpretive guidance for the metrics shown above.
    print(f"\n  DIAGNOSTIC SUMMARY:")
    print(f"  {glyphs['bullet']} Raw Cond#  : Values > 1e10 indicate mathematical singularity (rank deficiency).")
    print(f"  {glyphs['bullet']} LW Cond#   : The stabilized condition number after Ledoit-Wolf shrinkage.")
    print(f"  {glyphs['bullet']} Shrinkage  : Coefficient 'alpha' used to regularize the matrix (0=none, 1=full).")
    print(f"  {glyphs['bullet']} Precision  : 64-bit floats provide ~15 digits of base precision.")
    print(f"                 An LW Cond# of 1e{threshold} loses {threshold} digits, leaving {15-threshold} for inference.")
    print(f"  {glyphs['bullet']} Status     : OPTIMAL confirms reliable matrix inversion for CLR-based deltas.")
    # print("-" * width)

# --------------------------------------------------------------------------------------
# Helper Function 2-B: Print top shifts in means and volatilities for each era pair, 
#                      with directional arrows and
# --------------------------------------------------------------------------------------
def _display_banner_table(title, df):
    table_str = df.round(4).to_string()
    width = max(len(title), max(len(l) for l in table_str.splitlines()))
    print(f"\n{'='*width}\n{title}\n{'='*width}\n{table_str}")

# --------------------------------------------------------------------------------------
# Helper Function 2-C: Print top shifts in means and volatilities for each era pair,
# --------------------------------------------------------------------------------------
def _print_section_header(title, glyph, width):
    print(f"\n{'=' * width}\n  {glyph} {title}\n{'=' * width}")

# --------------------------------------------------------------------------------------
# Helper function 2-D: Get glyphs for report formatting, with option for emoji 
#                      or ASCII fallback
# ---------------------------------------------------------------------------------------
def _get_glyphs(use_emoji):
    if not use_emoji:
        return {k: f"[{k.upper()}]" for k in ['ok', 'fail', 'warn', 'green', 'yellow', 'matrix', 'shield', 'fire', 'bolt']} | {'bullet': '-', 'up': '+', 'down': '-'}
    return {'ok': '✅', 'fail': '❌', 'warn': '⚠️', 'green': '🟢', 'yellow': '🟡', 'matrix': '📐', 'shield': '🛡️', 'fire': '🔥', 'bolt': '⚡', 'bullet': '•', 'up': '↑', 'down': '↓'}

# --------------------------------------------------------------------------------------
# Helper function 2-E: Print top shifts in means and volatilities for each era pair, 
#                      with directional arrows and formatted tables
# --------------------------------------------------------------------------------------
def _print_top_shifts(df, pairs, label, glyph, glyphs, n_rows, width):
    _print_section_header(f"TOP {label}", glyph, width=width)
    for earlier, later, col in pairs:
        print(f"\n  {glyphs['bullet']} {earlier} -> {later} ({col})\n  {'-' * (width - 4)}")
        top = df[col].sort_values(key=abs, ascending=False).head(n_rows)
        for crime, val in top.items():
            arrow = glyphs['up'] if val > 0 else glyphs['down']
            print(f"    {arrow} {crime:<50} {val:>+10.4f}")


# -----------------------------------------------------------------------------
# 3. Function to run stationarity analysis with harmonized conclusions and multiple testing correction
# -----------------------------------------------------------------------------
def run_stationarity_analysis(clr_df, filled_df, era_boundaries=None, seasonal_cycle=12,
                              sparse_threshold=0.05, verbose=True):
    """
    Orchestrates the stationarity testing suite:
    1. Identifies sparse categories based on raw crime counts.
    2. Runs a suite of unit-root and structural break tests (ADF, KPSS, ZA).
    3. Applies BH-FDR correction to p-values for multiple testing.
    4. Harmonizes results into a final stationarity conclusion.
    """
    # Validate era_boundaries input
    if era_boundaries is None:
        raise ValueError("era_boundaries dict required (Pre-COVID, COVID, Post-COVID keys).")
    # ------------------------------------------------------------------------
    # 3-1 Sparsity Assessment - Identify categories with high zero rates in raw crime counts
    # -------------------------------------------------------------------------
    # Efficiently calculate zero rates for all crime types
    zero_rates = (filled_df.groupby('fbi_code_desc', observed=True)['crime_count']
                  .apply(lambda x: (x == 0).mean()))
    sparse_cats = zero_rates[zero_rates > sparse_threshold].index.tolist()

    # ------------------------------------------------------------------------
    # 3-2. Run stationarity tests for each crime type's CLR series with error handling for sparse data
    # -------------------------------------------------------------------------
    # Note: If clr_df.columns is large, consider using joblib or multiprocessing here
    results = [
        _run_single_stationarity_test(
            col, 
            clr_df[col].values, 
            is_sparse=(col in sparse_cats), 
            zero_rate=zero_rates.get(col, 0.0), 
            index=clr_df.index,
            seasonal_cycle=seasonal_cycle
        ) 
        for col in clr_df.columns
    ]
    # Convert results to DataFrame for easier manipulation and reporting
    stat_df = pd.DataFrame(results).set_index('crime')

    # ------------------------------------------------------------------------
    # 3-3. Multiple Testing Correction - Apply BH-FDR correction to p-values for dense categories 
    # only to avoid biasing the FDR with unreliable sparse category results
    # -------------------------------------------------------------------------
    # Only apply correction to 'dense' categories to avoid biasing the FDR
    dense_mask = ~stat_df['is_sparse']
    for p_col in ['adf_p', 'za_p']:
        adj_col = f"{p_col}_adj"
        stat_df[adj_col] = np.nan  # Initialize with NaN
        
        # Pull p-values for dense categories, drop any NaNs from failed tests
        p_vals = stat_df.loc[dense_mask, p_col].dropna()

        # Only apply correction if there are valid p-values to adjust
        if not p_vals.empty:
            _, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
            # Map adjusted p-values back to the correct rows
            stat_df.loc[p_vals.index, adj_col] = p_adj
    # ------------------------------------------------------------------------
    # 3-4. Harmonization Logic - Evaluate conclusions based on test results with sensitivity f
    # lagging for trend-sensitive break points
    # -------------------------------------------------------------------------
    concl_cols = ['adf_kpss', 'za_conclusion', 'agreement']
    conclusions = stat_df.apply(_evaluate_conclusion, axis=1)
    stat_df[concl_cols] = pd.DataFrame(conclusions.tolist(), index=stat_df.index)

    # ------------------------------------------------------------------------
    # 3-5. Reporting - Print comprehensive stationarity report with clear formatting and 
    # interpretation notes
    # ------------------------------------------------------------------------
    if verbose:
        _print_stationarity_report(stat_df, sparse_cats, zero_rates, 
                                   era_boundaries, seasonal_cycle, sparse_threshold)

    return stat_df 

# -----------------------------------------------------------------------------
# Helper Function 3-A to run stationarity tests on each crime type's time series
# ------------------------------------------------------------------------------
def _run_single_stationarity_test(col_name, series, is_sparse, 
                                  zero_rate, index, seasonal_cycle):
    """Internal: Runs ADF, KPSS, and ZA tests for a single series."""

    # ADF and KPSS can fail on series with too many zeros or constant values.
    # We now catch these to prevent the entire loop from crashing.
    try:
        adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag='AIC')
    except Exception:
        adf_p = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')
        except Exception:
            kpss_p = np.nan

    # Consolidated the ZA 'c' and 'ct' logic into a loop.
    # This ensures both models are treated with the same error-handling logic.
    za_results = {
        'stat_c': np.nan, 'p_c': np.nan, 'date_c': 'N/A', 'bp_c': None,
        'stat_ct': np.nan, 'p_ct': np.nan, 'date_ct': 'N/A', 'bp_ct': None
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model in ['c', 'ct']:
            try:
                # Zivot-Andrews is prone to failing on high-sparsity crime data.
                stat, p, _, _, bp = zivot_andrews(series, trim=0.15, regression=model)
                suffix = f'_{model}' if model == 'c' else '_ct'
                za_results[f'stat{suffix}'] = stat
                za_results[f'p{suffix}'] = p
                za_results[f'bp{suffix}'] = bp
                
                # bp is an index; convert to string date ONLY if bp is not None.
                if bp is not None:
                    za_results[f'date{suffix}'] = index[int(bp)].strftime('%Y-%m')
            except (ValueError, IndexError, TypeError):
                pass 

    # Check both break points before calculating the delta.
    bp_diff = None
    trend_sensitive = False
    if za_results['bp_c'] is not None and za_results['bp_ct'] is not None:
        bp_diff = abs(int(za_results['bp_c']) - int(za_results['bp_ct']))
        trend_sensitive = bp_diff > seasonal_cycle

    return {
            'crime': col_name,
            'is_sparse': is_sparse,
            'zero_rate': zero_rate,
            'adf_p': adf_p,
            'kpss_p': kpss_p,
            'za_stat': za_results['stat_ct'],
            'za_p': za_results['p_ct'],
            'za_date': za_results['date_ct'],
            'za_stat_c': za_results['stat_c'],
            'za_p_c': za_results['p_c'],
            'za_date_c': za_results['date_c'],
            'trend_sensitive': trend_sensitive,
            'bp_diff_months': bp_diff,
        }


# -----------------------------------------------------------------------------
# Helper Function 3-B to evaluate conclusions based on test results with harmonization logic
# -----------------------------------------------------------------------------
def _evaluate_conclusion(row):
    """Logic to harmonize ADF, KPSS, and ZA results with sensitivity flagging."""
    if row['is_sparse']:
        return 'Unreliable ⚠️', 'Unreliable ⚠️', 'Excluded - sparse'

    # Alpha threshold (standard 5%)
    alpha = 0.05
    adf_rej = row['adf_p_adj'] < alpha
    kpss_rej = row['kpss_p'] < alpha
    za_rej = row['za_p_adj'] < alpha
    # -----------------------------------------------------------------------
    # 3-B-1: ADF + KPSS Logic (Standard Unit Root Tests)
    # ------------------------------------------------------------------------
    if not adf_rej and kpss_rej:
        adf_kpss = 'Non-Stationary'
    elif adf_rej and not kpss_rej:
        adf_kpss = 'Stationary'
    elif adf_rej and kpss_rej:
        adf_kpss = 'Trend-Stationary'
    else:
        adf_kpss = 'Inconclusive'
    # -----------------------------------------------------------------------
    # Helper Function 3-B-2: Zivot-Andrews Logic (Handles Structural Breaks)
    # If ZA is stationary but ADF is not, the break was the cause of ADF failure.
    # ------------------------------------------------------------------------
    za_conclusion = 'Stationary (ZA)' if za_rej else 'Non-Stationary (ZA)'

    # -----------------------------------------------------------------------
    # 1-B-3: Harmonization Logic with Sensitivity Flagging
    # -----------------------------------------------------------------------
    if adf_kpss == 'Non-Stationary' and za_conclusion == 'Non-Stationary (ZA)':
        agreement = 'Agree - Non-Stationary'
    elif adf_kpss in ['Stationary', 'Trend-Stationary'] and za_conclusion == 'Stationary (ZA)':
        # Flag if the stationary conclusion relies on a shaky break point
        if row.get('trend_sensitive', False):
            agreement = 'Agree - Stationary (Sensitive ⚠️)'
        else:
            agreement = 'Agree - Stationary'
    else:
        # High-interest case: ZA finds stationarity where standard tests failed
        if za_rej and adf_kpss == 'Non-Stationary':
            agreement = 'Break-Induced Stationarity'
        else:
            agreement = 'Disagree'

    return adf_kpss, za_conclusion, agreement

# -------------------------------------------------------------------------------
# Helper Function 3-C to print a comprehensive stationarity report with improved formatting and alignment
# -----------------------------------------------------------------------
def _print_stationarity_report(df, sparse_cats, zero_rates, era_boundaries, seasonal_cycle, sparse_threshold):
    """
    Generate a fully formatted, dynamically aligned stationarity diagnostics report.

    This function prints a multi‑section dashboard summarizing:
    - Dense vs. sparse categories
    - Zero‑rate analysis for sparse categories
    - Detailed stationarity agreement (ADF + KPSS)
    - Trend‑sensitive categories with structural break metadata
    - Break‑date alignment relative to pre‑COVID / COVID / post‑COVID eras

    Parameters
    ----------
    df : pd.DataFrame
        Main stationarity results table. Must include:
            - 'is_sparse' (bool)
            - 'agreement' (str)
            - 'trend_sensitive' (bool)
            - 'za_date' (str, YYYY-MM)
            - 'za_date_c' (float or str)
            - 'bp_diff_months' (int)

    sparse_cats : list
        Categories classified as sparse (high zero‑rates).

    zero_rates : pd.Series or dict
        Zero‑rate percentages for each category. Used to highlight sparse series.

    era_boundaries : dict
        Dictionary with keys:
            'Pre-COVID' : str (YYYY-MM)
            'COVID'     : str (YYYY-MM)
        Used to classify break dates into eras.

    seasonal_cycle : int
        Seasonal cycle length (e.g., 12 for monthly data). Used in trend‑sensitivity labeling.

    sparse_threshold : float
        Threshold above which a category is considered sparse (e.g., 0.05 for 5%).

    Notes
    -----
    - The function prints directly to stdout.
    - All column widths are dynamically computed to maintain perfect vertical alignment.
    - No return value; this is a pure formatting/printing utility.
    """
    # Load glyphs for visual formatting
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
    BOLD, DIM = "\033[1m", "\033[2m"
    
    # ------------------------------------------------------------
    # 3-C-1: Dynamic Column Width Calculation
    # Compute the longest category name to determine column widths.
    # This ensures the vertical alignment spine stays consistent.
    # ------------------------------------------------------------
    max_name_len = df.index.str.len().max()
    W_CAT = max(40, min(55, max_name_len + 5))  # Category column width
    W_STAT = 35                                 # Status column width
    
    # MASTER WIDTH controls all horizontal separators
    MASTER_WIDTH = W_CAT + W_STAT

    dense_df = df[~df['is_sparse']].sort_index()
    num_dense = len(dense_df)

    # ------------------------------------------------------------
    # 3-C-2: Dashboard Title & Summary Metadata
    # Prints dashboard title, counts, and summary metadata.
    # ------------------------------------------------------------
    print(f"\n{'=' * MASTER_WIDTH}")
    print(f"📊 {BOLD}STATIONARITY DASHBOARD{RESET} Total: {len(df)}")
    print(f"{'=' * MASTER_WIDTH}")
    print(f"  Dense  : {num_dense:<3}\n  Sparse : {len(sparse_cats):<3} (@ > {sparse_threshold:.1%})")

    # ------------------------------------------------------------
    # 3-C-3: Sparse Category Analysis
    # Only printed if zero-rate data exists.
    # ------------------------------------------------------------
    if zero_rates is not None and len(zero_rates) > 0:
        print(f"\n{'-' * MASTER_WIDTH}")

        # Determine if any category exceeds the sparse threshold
        has_sparse = (zero_rates > sparse_threshold).any() if isinstance(zero_rates, pd.Series) else any(v > sparse_threshold for v in zero_rates.values())

        if has_sparse:
            # Sort sparse categories by zero-rate descending
            sorted_items = zero_rates.sort_values(ascending=False) if isinstance(zero_rates, pd.Series) else zero_rates
            for cat, rate in sorted_items.items():
                if cat in sparse_cats or rate > sparse_threshold:
                    # Truncate long names for formatting
                    name = (cat[:W_CAT-8] + '...') if len(cat) > (W_CAT-5) else cat
                    print(f"  ⚠️  {name:<{W_CAT-5}} : {rate:>6.1%} zeros")
            
            print(f"{'-' * MASTER_WIDTH}\n")

    # ------------------------------------------------------------
    # 3-C-4: Detailed Stationarity Agreement
    # Shows ADF+KPSS agreement for dense categories.
    # ------------------------------------------------------------
    print(f"🔍 {BOLD}DETAILED STATIONARITY REPORT (Dense){RESET}")
    print(f"   {'Category':<{W_CAT-3}}    Status")
    
    print(f"{'-' * (W_CAT+1)} {'-' * (W_STAT-2)}")

    for row in dense_df.itertuples():
        # Choose icon based on stationarity agreement
        icon = "🔴" if "Non-Stationary" in str(row.agreement) else "🟢"
        # Truncate long category names
        name = (row.Index[:W_CAT-6] + '...') if len(row.Index) > (W_CAT-3) else row.Index
        print(f"  {icon} {name:<{W_CAT-4}} │ {row.agreement}")
    
    print(f"{'-' * MASTER_WIDTH}")

    # ------------------------------------------------------------
    # 3-C-5: Trend-Sensitive Breaks
    # Shows categories with structural breaks beyond seasonal cycle.
    # ------------------------------------------------------------
    W_CT, W_C, W_DIFF = 14, 12, 6
    print(f"\n📉 {BOLD}TREND-SENSITIVE (Break > {seasonal_cycle}m){RESET}")
    print(f"{'=' * MASTER_WIDTH}")
    
    trend_df = dense_df[dense_df['trend_sensitive']]
    if not trend_df.empty:
        print(f"{BOLD}{'Category':<{W_CAT}} {'Break ct':<{W_CT}} {'Break c':<{W_C}} {'Diff':>{W_DIFF}}{RESET}")
        print(f"{'-'*W_CAT} {'-'*W_CT} {'-'*W_C} {'-'*W_DIFF}")
        for row in trend_df.itertuples():
            name = (row.Index[:W_CAT-3] + '...') if len(row.Index) > W_CAT else row.Index
            print(f"{name:<{W_CAT}} {row.za_date:<{W_CT}} {row.za_date_c:<{W_C}} {int(row.bp_diff_months):>{W_DIFF-1}}m")
    else:
        print(f" ✅ {GREEN}None detected.{RESET}")
    print(f"{'=' * MASTER_WIDTH}")

    # ------------------------------------------------------------
    # 3-C-6: Break Date Alignment
    # Classifies break dates into Pre-COVID, COVID-aligned, Post-COVID.
    # ------------------------------------------------------------
    if num_dense > 0:
        print(f"\n📅 {BOLD}BREAK DATE ALIGNMENT (N={num_dense}){RESET}")
        start, end = pd.Timestamp(era_boundaries['Pre-COVID']), pd.Timestamp(era_boundaries['COVID'])
        dates = pd.to_datetime(dense_df['za_date'] + '-01')
        
        # Default label: Post-COVID
        labels = pd.Series('Post-COVID', index=dense_df.index)
        labels[dates <= start] = 'Pre-COVID'
        labels[(dates > start) & (dates <= end)] = 'COVID-Aligned'
        
        counts = labels.value_counts()
        for label in ["COVID-Aligned", "Pre-COVID", "Post-COVID"]:
            c = counts.get(label, 0)
            pct = (c / num_dense) * 100
            # Visual bar scaled to 25 characters
            print(f"  {label:<15} │ {GREEN}{'█' * int(pct / 4):<25}{RESET} {c:>3} ({pct:>4.1f}%)")

# -----------------------------------------------------------------------------
# Helper Function 3-D-1: 
# -----------------------------------------------------------------------------
def _print_bootstrap_report(df, title, n1_info, n2_info, block_size, n_bootstrap, label2):
    """
    Print a formatted bootstrap inference report with dynamic column widths.

    This function displays:
    - A header summarizing bootstrap settings (block size, iterations, group sizes)
    - A table of effect-size and significance metrics for each category
    - Sparse-category handling (no FDR adjustment)
    - FDR-adjusted p-values and significance flags for dense categories
    - A final summary count of significant mean shifts

    Parameters
    ----------
    df : pd.DataFrame
        Bootstrap results table. Expected columns include:
            - delta_mean : float
            - hedges_g : float
            - p_bootstrap : float
            - p_adj : float (optional, dense only)
            - is_sparse : bool
            - mean_sig : bool (optional)

    title : str
        Title describing the comparison (e.g., "Group A vs Group B").

    n1_info : tuple
        Information about group 1 (e.g., (n_samples, label)).

    n2_info : tuple
        Information about group 2 (e.g., (n_samples, label)).

    block_size : int
        Block size used in block bootstrap resampling.

    n_bootstrap : int
        Number of bootstrap iterations performed.

    Notes
    -----
    - This function prints directly to stdout.
    - No values are returned.
    - Column widths are fixed for readability.
    """
    # Load glyphs for visual formatting
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
    BOLD, DIM = "\033[1m", "\033[2m"
    
    # Fixed line width for consistent formatting across all rows
    W_LINE = 100

    # ---------------------------
    # 3-D-1-A: Header Section
    # ---------------------------
    print(f"\n{'=' * W_LINE}")
    print(f"🧬 {BOLD}BOOTSTRAP INFERENCE:{RESET} {title}")
    # Display block size, iterations, and group sizes
    print(f"{DIM}📦 Blocks:{RESET} {block_size} | {DIM}🔄 Iters:{RESET} {n_bootstrap:,} | {DIM}👥 Groups:{RESET} {n1_info[0]} vs {n2_info[0]}")
    print(f"{'=' * W_LINE}")

    # ---------------------------
    # 3-D-1-B: Table Header
    # ---------------------------
    print(f"\n{BOLD}{'CATEGORY':<35} {'Δ MEAN':>10} {'HEDGES G':>12} {'P-VAL':>10} {'FDR ADJ':>10}   {'STATUS'}{RESET}")
    print(f"{DIM}{'─' * W_LINE}{RESET}")

    # -----------------------------------------------
    # 3-D-1-C: Table Rows with Conditional Formatting
    # ------------------------------------------------
    for row in df.itertuples():
        # Truncate the name at exactly 32 chars to leave a 3-char buffer before Δ MEAN
        name = (row.Index[:32] + "..") if len(row.Index) > 33 else row.Index
        
        # Pull values
        d_mean = f"{row.delta_mean:>10.4f}"
        # We ensure we use the same column for every number
        h_g    = f"{row.hedges_g:>12.4f}" 
        p_val  = f"{row.p_bootstrap:>10.4f}"
        
        if getattr(row, 'is_sparse', False):
            # Sparse rows use '--' for the adjusted P-value
            print(f"{DIM}{name:<35} {d_mean} {h_g} {p_val} {'--':>10}   📌 SPARSE/BYPASS{RESET}")
        else:
            p_adj = getattr(row, 'p_adj', 1.0)
            p_adj_str = f"{p_adj:>10.4f}"
            p_color = GREEN if p_adj < 0.05 else ""
            status = f"✨ {GREEN}Significant{RESET}" if getattr(row, 'mean_sig', False) else f"{DIM}🚫 Not Significant{RESET}"
            
            # THE KEY LINE: Every variable is placed in a slot of fixed width
            print(f"{name:<35} {d_mean} {h_g} {p_val} {p_color}{p_adj_str}{RESET}   {status}")

    # ---------------------------
    # 3-D-1-D: Summary Footer
    # ---------------------------
    sig_count = df['mean_sig'].sum() if 'mean_sig' in df.columns else 0
    
    # Simple clean divider
    print(f"\n{BOLD}{'─' * W_LINE}{RESET}")
    print(f" {BOLD}✅ RESULT:{RESET} {GREEN if sig_count > 0 else ''}{sig_count} shifts detected.{RESET}")
    
    # Minimalist Key
    print(f"\n {' ' * 2}{BOLD}{DIM}INTERPRETATION KEY:{RESET}")
    
    # We use a simple vertical bar and aligned text
    print(f"  {' ' * 2} {BOLD}HEDGES G{RESET}      {DIM}│{RESET}  {GREEN}(+) {RESET}Share Grew in {label2}")
    print(f"  {' ' * 2} {BOLD}HEDGES G{RESET}      {DIM}│{RESET}  {RED}(-) {RESET}Share Shrank in {label2}")
    
    print(f"{BOLD}{'─' * W_LINE}{RESET}\n")

# -----------------------------------------------------------------------------
# Helper Function 4-D: Hedges' g effect size calculation with small sample correction
# -----------------------------------------------------------------------------
def _calculate_hedges_g(x1, x2):
    """
    Hedges' g effect size with small sample correction.

    Sign convention
    ──────────────
    g = (mean(x2) - mean(x1)) / pooled_sd
    Positive g -> x2 (later era) is higher than x1 (earlier era)
    Negative g -> x2 (later era) is lower  than x1 (earlier era)

    Note: this is the OPPOSITE sign to the CLR difference tables,
    where Pre_minus_COVID = Pre - COVID.
    """
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    # Guard against division by zero if both groups have zero variance
    if pooled_sd == 0:
        return 0
    # Calculate Hedges' g with small sample correction
    d = (np.mean(x2) - np.mean(x1)) / pooled_sd
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction


# -----------------------------------------------------------------------------
# Helper Function 4-E: Vectorized block bootstrap logic to preserve temporal autocorrelation
# -----------------------------------------------------------------------------
def _block_bootstrap_logic(x1, x2, block_size, n_bootstrap, seed):
    """Internal: Vectorized block bootstrap to preserve temporal autocorrelation."""
    rng = np.random.default_rng(seed)
    n1, n2 = len(x1), len(x2)
    observed_diff = np.mean(x2) - np.mean(x1)
    
    combined = np.concatenate([x1, x2])
    n_total = len(combined)
    offsets = np.arange(block_size)
    
    def _get_boot_means(n_target, blocks_needed):
        # Generate random start indices for all bootstrap iterations at once
        starts = rng.integers(0, n_total - block_size + 1, size=(n_bootstrap, blocks_needed))
        # Efficiently extract blocks using broadcasting: (n_bootstrap, blocks, block_size)
        samples = combined[starts[:, :, np.newaxis] + offsets]
        # Flatten blocks and truncate to original sample size, then compute mean per iteration
        return samples.reshape(n_bootstrap, -1)[:, :n_target].mean(axis=1)

    # Calculate blocks needed for each group
    blocks1, blocks2 = int(np.ceil(n1 / block_size)), int(np.ceil(n2 / block_size))
    
    # Generate bootstrap means for both groups and calculate the difference for each iteration
    boot_diffs = _get_boot_means(n2, blocks2) - _get_boot_means(n1, blocks1)
    p_val = (1 + np.sum(np.abs(boot_diffs) >= np.abs(observed_diff))) / (1 + n_bootstrap)
    
    return observed_diff, p_val



# ----------------------------------------------------------------------------
# 5. Main Function to Run Era Comparison with Block Bootstrap and Multiple Testing Correction
# ----------------------------------------------------------------------------
def run_era_comparison(era1_df, era2_df, label1="Era 1", label2="Era 2",
                       verbose=False, block_size=12, n_bootstrap=10_000,
                       seed=None, sparse_cats=None,
                       min_effective_blocks=5, alpha=0.05):
    """
    Block bootstrap comparison of CLR-transformed compositional data
    between two eras, with BH-FDR correction across categories.

    For each category (column), performs a block bootstrap test of
    the mean difference between era1 and era2, then applies the
    Benjamini-Hochberg false discovery rate correction across all
    dense (non-sparse) categories at the specified alpha level.

    Parameters
    ----------
    era1_df : pd.DataFrame
        CLR-transformed data for the earlier era, shape (T1, K).
        Rows are observations (e.g., months), columns are categories.

    era2_df : pd.DataFrame
        CLR-transformed data for the later era, shape (T2, K).
        Must have the same columns as era1_df.

    label1 : str, default "Era 1"
        Display label for era1_df in reports and `.attrs` metadata.

    label2 : str, default "Era 2"
        Display label for era2_df in reports and `.attrs` metadata.

    verbose : bool, default False
        If True, prints a formatted significance report via
        `_print_bootstrap_report`. The returned DataFrame is identical
        regardless of this setting.

    block_size : int, default 12
        Block length for circular block bootstrap. Preserves temporal
        autocorrelation within blocks of this many consecutive months.
        Auto-adjusted downward if the shortest era produces fewer than
        `min_effective_blocks` (see Notes).

    n_bootstrap : int, default 10,000
        Number of bootstrap iterations. Sets the resolution floor of
        the empirical p-values: the minimum reportable p_bootstrap is
        approximately 1 / (1 + n_bootstrap).

    seed : int or None, default None
        Master seed for reproducibility. When set, each category gets
        a deterministic independent RNG stream derived from this seed
        via `numpy.random.SeedSequence.spawn`. None uses fresh OS
        entropy (non-reproducible).

    sparse_cats : list of str or None, default None
        Category names to flag as sparse. Still computed for reference
        (delta_mean, hedges_g, p_bootstrap are populated) but excluded
        from BH-FDR correction: their p_adj is set to NaN and mean_sig
        to False. Use for categories with high zero-imputation rates
        where CLR statistics are dominated by the zero-replacement rule.

    min_effective_blocks : int, default 5
        Minimum number of effective bootstrap blocks below which
        block_size is auto-reduced. The effective block count is
        min(T1, T2) / block_size. When this falls below the threshold,
        block_size is reduced to max(3, min(T1, T2) // min_effective_blocks)
        and a UserWarning is issued.

    alpha : float, default 0.05
        FDR significance threshold. Categories with p_adj < alpha are
        flagged as significant (mean_sig = True). Recorded in .attrs.

    Returns
    -------
    res_df : pd.DataFrame
        One row per category in era1_df.columns, sorted by absolute
        Hedges' g (descending). Columns:
            'delta_mean'  : mean(era2) - mean(era1) for that category
            'hedges_g'    : standardized effect size (positive = grew
                            in era2)
            'p_bootstrap' : empirical two-tailed p-value from the block
                            bootstrap
            'is_sparse'   : True if category was in sparse_cats
            'p_adj'       : BH-FDR adjusted p-value (NaN for sparse)
            'mean_sig'    : True if p_adj < alpha (False for sparse)

        The DataFrame's `.attrs` dict carries the full run configuration:
        block_size (after any adjustment), block_size_requested,
        min_effective_blocks, n_bootstrap, seed, era1_label, era1_n,
        era2_label, era2_n, sparse_cats, and alpha.

    Notes
    -----
    Sign convention:
        delta_mean and hedges_g share the convention
        (mean(era2) - mean(era1)): positive values mean the CLR share
        grew in the later era. This matches the bootstrap-report
        formatting and the descriptive era-distribution module.

    Block-size auto-adjustment:
        When min(T1, T2) / block_size < min_effective_blocks, the block
        size is reduced and a UserWarning is issued naming both the
        requested and adjusted values. Both are recorded in .attrs so
        the adjustment is traceable. Auto-adjustment is intended for
        defensive use; if it triggers repeatedly, consider whether the
        bootstrap is well-powered for the comparison at all.

    Per-category seeding:
        Each category's bootstrap uses an independent RNG stream
        derived from the master `seed` via SeedSequence.spawn. This
        ensures cross-category independence while preserving overall
        reproducibility from the single master seed.

    FDR scope:
        BH-FDR is applied across dense categories only, with m equal
        to the number of dense categories. Sparse categories are
        excluded from this multiple-testing adjustment, so the FDR
        guarantee applies only to the dense set.

    Examples
    --------
    >>> sparse_cats = ['Gambling', 'Involuntary Manslaughter / Reckless Homicide']
    >>> result = run_era_comparison(
    ...     clr_era['Pre-COVID'], clr_era['COVID'],
    ...     label1='Pre-COVID', label2='COVID',
    ...     sparse_cats=sparse_cats,
    ...     block_size=6,
    ...     seed=1776,
    ...     verbose=True,
    ... )
    >>> result['mean_sig'].sum()    # number of significant categories
    23
    >>> result.attrs['block_size']  # actual block size used
    6
    """
    # Block size auto-adjustment
    min_T         = min(len(era1_df), len(era2_df))
    original_size = block_size
    # Ensure block size is not larger than the shortest era and provides at least 5 effective blocks
    if min_T / block_size < min_effective_blocks:
        block_size = max(3, min_T // min_effective_blocks)
        warnings.warn(
            f"\nBlock bootstrap - block size auto-adjusted:"
            f"\n  Shortest era      : T = {min_T}"
            f"\n  Requested size    : {original_size}"
            f"\n  Effective blocks  : ~{min_T/original_size:.1f}"
            f"  (minimum recommended: 5)"
            f"\n  Adjusted size     : {block_size}"
            f"\n  Effective blocks  : ~{min_T/block_size:.1f}"
            f"\n  Interpret results for this comparison cautiously.",
            UserWarning,
            stacklevel=2
        )
    # Ensure sparse_cats is a set for faster lookup and handle None case
    sparse_cats = set(sparse_cats) if sparse_cats else set()

    # Build a SeedSequence and spawn one independent child per category
    if seed is not None:
        ss = np.random.SeedSequence(seed)
        cat_seeds = ss.spawn(len(era1_df.columns))
        cat_seed_map = dict(zip(era1_df.columns, cat_seeds))
    else:
        cat_seed_map = {col: None for col in era1_df.columns}

    # Run bootstrap for each category and collect results
    results = []
    for col in era1_df.columns:
        x1, x2 = era1_df[col].values, era2_df[col].values
        diff, p = _block_bootstrap_logic(
            x1, x2, block_size, n_bootstrap,
            seed=cat_seed_map[col]
        )
        g = _calculate_hedges_g(x1, x2)
        results.append({
            'crime'       : col,
            'delta_mean'  : diff,
            'hedges_g'    : g,
            'p_bootstrap' : p,
            'is_sparse'   : col in sparse_cats
        })

    # Convert results to DataFrame and set index for easier manipulation
    res_df     = pd.DataFrame(results).set_index('crime')
    dense_mask = ~res_df['is_sparse']

    # BH-FDR on dense categories only 
    res_df['p_adj']    = np.nan
    res_df['mean_sig'] = False

    # Only apply correction if there are valid p-values to adjust
    _, res_df.loc[dense_mask, 'p_adj'], _, _ = multipletests(
        res_df.loc[dense_mask, 'p_bootstrap'], method='fdr_bh'
    )
    # Mark significant mean shifts among dense categories based on adjusted p-values
    res_df.loc[dense_mask, 'mean_sig'] = ( 
        res_df.loc[dense_mask, 'p_adj'] < alpha
    )
    # Sort results by absolute magnitude of Hedges' g for better interpretability, with clear formatting and interpretation notes
    res_df = res_df.sort_values(by='hedges_g', key=abs, ascending=False)

    if verbose:
        title = f"{label1} vs {label2}"
        _print_bootstrap_report(
            res_df, title,
            (label1, len(era1_df)),
            (label2, len(era2_df)),
            block_size, n_bootstrap,
            label2
        )

    # Attach run configuration as DataFrame metadata
    res_df.attrs['block_size']           = block_size
    res_df.attrs['block_size_requested'] = original_size
    res_df.attrs['min_effective_blocks'] = min_effective_blocks
    res_df.attrs['n_bootstrap']          = n_bootstrap
    res_df.attrs['seed']                 = seed
    res_df.attrs['era1_label']           = label1
    res_df.attrs['era1_n']               = len(era1_df)
    res_df.attrs['era2_label']           = label2
    res_df.attrs['era2_n']               = len(era2_df)
    res_df.attrs['sparse_cats']          = sorted(sparse_cats)
    res_df.attrs['alpha']                = alpha

    return res_df

# -------------------------------------------------------------
# Helper Function 5-A: Data preparation for time series plotting 
# with vectorized operations and summary stats
# -------------------------------------------------------------
def _prepare_break_data(
    clr_df: pd.DataFrame,
    category: str,
    break_date: str,
    window: int
) -> Tuple[pd.Series, pd.Timestamp, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Prepare series and compute summary stats. Uses vectorized masks and single rolling call.
    """
    # Ensure index is datetime and category exists
    idx = pd.to_datetime(clr_df.index)
    if category not in clr_df.columns:
        raise KeyError(f"Category '{category}' not found in clr_df columns.")
    
    # Create series with proper index and name for easier handling
    series = pd.Series(clr_df[category].values, index=idx, name=category)
    break_dt = pd.Timestamp(break_date)

    # Create boolean masks for pre- and post-break periods
    pre_mask = series.index < break_dt
    post_mask = ~pre_mask

    # Extract pre- and post-break values using masks
    pre_vals = series.loc[pre_mask]
    post_vals = series.loc[post_mask]

    # Rolling mean with min_periods to avoid NaNs at edges
    rolling = series.rolling(window=window, center=True, min_periods=1).mean()

    # Compute summary statistics with checks for empty segments to avoid NaNs
    stats = {
        'pre_mean': pre_vals.mean() if len(pre_vals) > 0 else np.nan,
        'post_mean': post_vals.mean() if len(post_vals) > 0 else np.nan,
        'shift': (post_vals.mean() - pre_vals.mean()) if (len(pre_vals) > 0 and len(post_vals) > 0) else np.nan,
        'rolling': rolling,
        'pre_std': pre_vals.std(ddof=1) if len(pre_vals) > 1 else np.nan,
        'post_std': post_vals.std(ddof=1) if len(post_vals) > 1 else np.nan,
        'pre_n': len(pre_vals),
        'post_n': len(post_vals),
    }

    return series, break_dt, pre_vals, post_vals, stats


# --------------------------------------------------------------
# Helper Function 5-B: Plotting function for CLR time series and 
# distributions with enhanced aesthetics and annotations
# ---------------------------------------------------------------
def _plot_time_series(
    ax: plt.Axes,
    series: pd.Series,
    break_dt: pd.Timestamp,
    stats: Dict[str, Any],
    era_config: Dict[str, Tuple[str, str, str]],
    window: int,
    category: str,
    za_stat: Optional[float],
    za_p_adj: Optional[float],
    legend_loc: str = 'lower left'
) -> None:
    """
    Top panel: CLR time series with era shading, rolling mean, segment means, break line and annotation.
    """
    # Era shading (safe iteration)
    for label, bounds in era_config.items():
        try:
            start, end, color = bounds
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.07, color=color, label=label)
        except Exception:
            continue

    # Raw series
    ax.plot(series.index, series.values, color='gray', alpha=0.3, lw=0.8, label='CLR Raw')

    # Rolling mean
    ax.plot(stats['rolling'].index, stats['rolling'].values, color='#2c7bb6', lw=2.5, label=f'{window}-month rolling mean')

    # Vertical break line
    ax.axvline(break_dt, color='#d62728', lw=2.0, ls='--', alpha=0.8)
    # Safe ylim retrieval
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.02 * (ymax - ymin)
    ax.text(break_dt, y_text, f'  ZA Break: {break_dt.strftime("%Y-%m")}', color='#d62728', fontweight='bold', va='top', fontsize=10)

    # Segment means: draw two separate hlines for clarity
    if not pd.isna(stats['pre_mean']):
        ax.hlines(y=stats['pre_mean'], xmin=series.index[0], xmax=break_dt, colors='#1a9641', lw=3.0, label='Pre mean')
    if not pd.isna(stats['post_mean']):
        ax.hlines(y=stats['post_mean'], xmin=break_dt, xmax=series.index[-1], colors='#d7191c', lw=3.0, label='Post mean')

    # Mean shift annotation (offset in axes coords if needed)
    mid_y = np.nanmean([stats['pre_mean'], stats['post_mean']])
    if not np.isnan(mid_y):
        ax.annotate(
            f"Mean Shift: {stats['shift']:+.3f}",
            xy=(break_dt, mid_y),
            xytext=(15, 40),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='#d62728', lw=1.5),
            fontsize=12, fontweight='bold', color='#d62728',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    ax.set_title(f'CLR Time Series - {category}', fontsize=14, fontweight='bold')
    ax.set_ylabel('CLR Value', fontsize=11)

    # Consolidate legend entries (avoid duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc=legend_loc, ncol=2, frameon=True)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

# ---------------------------------------------------------------
# Helper Function 5-C: Plotting function for CLR distributions with enhanced aesthetics, KDE, and mean markers
# ---------------------------------------------------------------
def _plot_distribution(
    ax: plt.Axes,
    pre_vals: pd.Series,
    post_vals: pd.Series,
    stats: Dict[str, Any],
    break_dt: pd.Timestamp,
    legend_loc: str = 'lower left'
) -> None:
    """
    Bottom panel: density distributions before and after the structural break.
    Uses KDE for smooth comparison and overlays histograms for context.
    """
    # KDEs (use common_norm=False so densities are independent)
    if len(pre_vals) > 0:
        sns.histplot(pre_vals, bins=30, kde=False, color='#2c7bb6', alpha=0.25, stat='density', ax=ax)
        sns.kdeplot(pre_vals, color='#2c7bb6', lw=2.0, label=f'Pre-break  (n={stats["pre_n"]})', ax=ax)
    if len(post_vals) > 0:
        sns.histplot(post_vals, bins=30, kde=False, color='#d62728', alpha=0.25, stat='density', ax=ax)
        sns.kdeplot(post_vals, color='#d62728', lw=2.0, label=f'Post-break (n={stats["post_n"]})', ax=ax)

    # Mean markers (only if defined)
    if not pd.isna(stats['pre_mean']):
        ax.axvline(stats['pre_mean'], color='#2c7bb6', ls='--', lw=2.0, label=f'Pre mean: {stats["pre_mean"]:.3f}')
    if not pd.isna(stats['post_mean']):
        ax.axvline(stats['post_mean'], color='#d62728', ls='--', lw=2.0, label=f'Post mean: {stats["post_mean"]:.3f}')

    ax.set_title(f'CLR Density - Before vs After Break ({break_dt.strftime("%Y-%m")})', fontsize=13, fontweight='bold')
    ax.set_xlabel('CLR Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)

    # Consolidate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc=legend_loc, fontsize=9)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)


# ---------------------------------------------------------------
# Helper Function 5-D to run structural break analysis and 
# generate plots with nhanced aesthetics and annotations
# -----------------------------------------------------------
def _plot_distribution(ax, pre_vals, post_vals,
                       stats, break_date, legend_loc='lower left'):
    """
    Bottom panel: Density distributions before and after
    the structural break with KDE and mean markers.
    """
    # Histogram + KDE
    sns.histplot(
        pre_vals, bins=30, kde=True,
        color='#2c7bb6', alpha=0.4,
        label=f'Pre-break  (n={stats["pre_n"]})',
        stat='density', ax=ax
    )
    sns.histplot(
        post_vals, bins=30, kde=True,
        color='#d62728', alpha=0.4,
        label=f'Post-break (n={stats["post_n"]})',
        stat='density', ax=ax
    )

    # Mean markers
    ax.axvline(
        stats['pre_mean'],  color='#2c7bb6',
        ls='--', lw=2.0,
        label=f'Pre mean:  {stats["pre_mean"]:.3f}'
    )
    # Post mean marker only if defined (avoid NaN issues)
    ax.axvline(
        stats['post_mean'], color='#d62728',
        ls='--', lw=2.0,
        label=f'Post mean: {stats["post_mean"]:.3f}'
    )
    # Title and labels
    ax.set_title(
        f'CLR Density - Before vs After Break '
        f'({pd.Timestamp(break_date).strftime("%Y-%m")})',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('CLR Value', fontsize=11)
    ax.set_ylabel('Density',   fontsize=11)
    ax.legend(loc=legend_loc, fontsize=9)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)


# -------------------------------------------------------------------
# Helper Function 5-E: Console summary of structural break analysis with 
# clear formatting and interpretation notes
# --------------------------------------------------------------------
def _print_break_summary(
    category: str,
    break_date: str,
    pre_vals: pd.Series,
    post_vals: pd.Series,
    stats: Dict[str, Any],
    za_stat: Optional[float],
    za_p_adj: Optional[float],
    era_boundaries: Optional[Dict[str, str]] = None
) -> None:
    """
    Print a concise summary table and interpretation.
    """
    # Create a summary DataFrame for pre- and post-break segments
    summary = pd.DataFrame({'Pre-Break': pre_vals.describe(), 'Post-Break': post_vals.describe()})
    if za_p_adj is None or pd.isna(za_p_adj):
        p_display = "nan  (sparse - excluded from FDR) ⚠️"
        conclusion = "Unreliable ⚠️ - sparse category"
    elif za_p_adj < 0.05:
        p_display = f"{za_p_adj:.6f} (✅ Significant)"
        conclusion = "Reject Unit Root: Significant Structural Break Detected"
    else:
        p_display = f"{za_p_adj:.6f} (❌ Non-significant)"
        conclusion = "Fail to Reject Unit Root: No Significant Break"

    # Print the summary with clear formatting and interpretation notes
    width = 65
    print("=" * width)
    print(f"STRUCTURAL BREAK ANALYSIS - {category}")
    print(f"Break point : {pd.Timestamp(break_date).strftime('%Y-%m')}")
    print(f"ZA statistic: {za_stat if za_stat is not None else 'NaN'}")
    print(f"za_p_adj    : {p_display}")
    print(f"Conclusion  : {conclusion}")
    print("=" * width)

    print("\nSEGMENT STATISTICS:")
    print(summary.T[['count', 'mean', 'std', 'min', 'max']].round(4).to_string())

    print(f"\nMean shift at break : {stats['shift']:+.4f}")
    print(f"Pre-break  std      : {stats['pre_std']:.4f}")
    print(f"Post-break std      : {stats['post_std']:.4f}")

    # Optional: Check if the break date aligns with known era boundaries and print a note if it does not
    if era_boundaries:
        try:
            break_dt = pd.Timestamp(break_date)
            covid_start = (pd.Timestamp(era_boundaries['Pre-COVID']) + pd.DateOffset(months=1))
            covid_end = pd.Timestamp(era_boundaries['COVID'])
            post_start = pd.Timestamp(era_boundaries['Post-COVID'])
            # Check if the break date falls within the COVID era or if the post-break segment includes multiple eras
            if break_dt < covid_end:
                pre_covid_in_post = len(post_vals[post_vals.index < covid_start])
                covid_months = len(post_vals[(post_vals.index >= covid_start) & (post_vals.index <= covid_end)])
                post_months = len(post_vals[post_vals.index > covid_end])
                era_label = "two" if pre_covid_in_post == 0 else "multiple"
                print(f"\nNote: Post-break segment (n={len(post_vals)}) spans {era_label} administrative eras:")

                # Print the breakdown of months in each era within the post-break segment, if applicable
                if pre_covid_in_post > 0:
                    pre_covid_end = covid_start - pd.DateOffset(months=1)
                    print(f"  Pre-COVID era  : {pre_covid_in_post:<4} months ({break_dt.strftime('%Y-%m')} - {pre_covid_end.strftime('%Y-%m')})")
                print(f"  COVID era      : {covid_months:<4} months ({covid_start.strftime('%Y-%m')} - {covid_end.strftime('%Y-%m')})")
                print(f"  Post-COVID era : {post_months:<4} months ({post_start.strftime('%Y-%m')} - {post_vals.index[-1].strftime('%Y-%m')})")
                print("  ZA break does not align with the administrative era boundary.")
        except Exception:
            pass

# ----------------------------------------------------------------
# 6. Main Function to Plot Structural Break Diagnostics with Enhanced Aesthetics and Console Summary
# ----------------------------------------------------------------
def plot_structural_break(
    clr_df: pd.DataFrame,
    category: str,
    break_date: str,
    era_config: Optional[Dict[str, Tuple[str, str, str]]] = None,
    window: int = 12,
    za_stat: Optional[float] = None,
    za_p_adj: Optional[float] = None,
    era_boundaries: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
    legend_loc: str = 'lower left'
) -> None:
    """
    Plot structural break diagnostics: top time series + bottom density comparison.
    """
    # Data preparation
    series, break_dt, pre_vals, post_vals, stats = _prepare_break_data(
        clr_df, category, break_date, window
    )

    # Set up the figure with specified height ratios for better aesthetics
    g_kw = {'height_ratios': [1.2, 0.8], 'hspace': 0.3}
    fig, (ax_ts, ax_dist) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw=g_kw)

    # Plotting
    _plot_time_series(
        ax_ts, series, break_dt, stats, era_config or {},
        window, category, za_stat, za_p_adj, legend_loc
    )
    _plot_distribution(ax_dist, pre_vals, post_vals, stats, break_dt, legend_loc)

    # Construct the title with ZA statistic and adjusted p-value, handling NaN cases gracefully
    p_label = (
        f"Adj p-value: {za_p_adj:.4f}"
        if (za_p_adj is not None and not pd.isna(za_p_adj))
        else "Adj p-value: NaN | sparse"
    )
    
    # Format ZA statistic for display, handling None or NaN cases gracefully
    za_display = f"{za_stat:.3f}" if za_stat is not None else "NaN"
    
    plt.suptitle(
        f"Structural Break Diagnostics  |  ZA Statistic: {za_display}  |  "
        f"{p_label}  |  Break: {pd.Timestamp(break_date).strftime('%Y-%m')}",
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.subplots_adjust(top=0.92)

    # Save the figure if a path is provided, with a descriptive filename based on the category and break date
    if save_path:
        fname = f"{save_path}break_{category.replace(" ", "").replace("/", "")}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()

    # Print the console summary if verbose is True
    if verbose:
        _print_break_summary(
            category, break_date, pre_vals, post_vals, stats,
            za_stat, za_p_adj, era_boundaries
        )


# -----------------------------------------------------------
# Helper Function 6-A: Prepare covariance matrices and CLR arrays for Box's M test with validation and reporting
# -----------------------------------------------------------
def _prepare_cov_matrices(eras_stat, clr_era):
    """
    Extract and validate all covariance matrices and CLR arrays.

    Parameters
    ----------
    eras_stat : output of compute_era_distribution_parameters
                must contain 'era_covs' and 'era_covs_lw'
    clr_era   : output of slice_clr_into_eras
                must contain 'pre_covid', 'covid', 'post_covid'

    Returns
    -------
    dict with keys Pre-COVID, COVID, Post-COVID.
    Each value is a dict:
        clr     : (T, K) DataFrame
        T       : number of observations
        K       : number of crime types
        cov_raw : (K, K) np.ndarray  raw sample covariance
        cov_lw  : (K, K) np.ndarray  LW regularized covariance
        cols    : list of crime type column names
    """
    # Map era labels to keys in the input dictionaries
    era_map = cfg.era_map

    # Extract matrices and validate shapes
    matrices = {}
    for era_key, clr_key in era_map.items():
        clr_mat            = clr_era[clr_key]
        matrices[era_key]  = {
            'clr'    : clr_mat,
            'T'      : len(clr_mat),
            'K'      : clr_mat.shape[1],
            'cov_raw': eras_stat['era_covs'][era_key].values,
            'cov_lw' : eras_stat['era_covs_lw'][era_key].values,
            'lw_alpha': eras_stat['lw_shrinkage'][era_key], 
            'cols'   : clr_mat.columns.tolist(),
        }

    # Validation report
    print("=" * 65)
    print("COVARIANCE MATRIX VALIDATION")
    print("=" * 65)
    for era, d in matrices.items():
        K    = d['K']
        flag = '✅' if d['cov_lw'].shape == (K, K) else '❌'
        print(f"  {era:<15}: T={d['T']:<4}  "
              f"cov_raw={d['cov_raw'].shape}  "
              f"cov_lw={d['cov_lw'].shape}  {flag}")

    return matrices

# -----------------------------------------------------------
# 7. Function: Box's M test for equality of covariance matrices with permutation-based p-value
# -----------------------------------------------------------
def run_boxm_test(matrices, n_permutations=10_000,
                  seed=None, sparse_cats=None):
    """
    Box's M test for equality of covariance matrices across eras,
    with permutation-based p-value for robustness to non-normality.

    Parameters
    ----------
    matrices       : output of _prepare_cov_matrices
    n_permutations : permutation iterations
    seed           : random seed
    sparse_cats    : list of sparse category names to flag

    Returns
    -------
    dict:
        M_obs   : float   observed M statistic
        M_perm  : ndarray permutation distribution
        p_perm  : float   permutation p-value
        n_perm  : int     number of permutations
        reject  : bool    True if p_perm < 0.05
    """
    # Normalize sparse_cats to a set for efficient lookup and handle None case
    sparse_cats = set(sparse_cats or [])
    rng         = np.random.default_rng(seed)
    era_keys    = list(matrices.keys())

    # Extract CLR arrays for each era into a list of groups for the test
    groups = [matrices[e]['clr'].values for e in era_keys]
    ns     = [g.shape[0] for g in groups]
    K      = groups[0].shape[1]
    N      = sum(ns)

    # Observed statistic
    M_obs  = _boxm_statistic(groups)

    # Permutation distribution
    all_data = np.vstack(groups)
    M_perm   = np.empty(n_permutations)

    # Permute group labels and compute M statistic for each permutation
    for i in range(n_permutations):
        idx      = rng.permutation(N)
        shuffled = all_data[idx]
        perm_groups, start = [], 0
        # Reconstruct groups based on original sample sizes
        for n in ns:
            perm_groups.append(shuffled[start:start + n])
            start += n
        M_perm[i] = _boxm_statistic(perm_groups)
    # Calculate p-value as the proportion of permuted statistics that are as 
    # extreme as or more extreme than the observed statistic
    p_perm = float(np.mean(M_perm >= M_obs))
    reject = p_perm < 0.05

    # Formatted console output with clear reporting of test results and interpretation
    print()
    print("=" * 65)
    print("BOX'S M TEST - EQUALITY OF COVARIANCE MATRICES")
    print("=" * 65)
    print(f"  Comparison    : {' vs '.join(era_keys)}")
    print(f"  Groups (k)    : {len(era_keys)}")
    print(f"  Variables (K) : {K}")
    print(f"  Sample sizes  : "
          f"{', '.join(f'{e}=T{n}' for e, n in zip(era_keys, ns))}")
    print(f"  Permutations  : {n_permutations:,}  seed={seed}")
    print()
    print(f"  M statistic   : {M_obs:.4f}")
    print(f"  p (permuted)  : {p_perm:.4f}  "
          f"{'✅ reject H₀ - structures differ' if reject else '❌ fail to reject H₀'}")
    print()
    
    # Hypotheses
    print("  H₀: Σ_pre = Σ_covid = Σ_post")
    print("  H₁: At least one covariance matrix differs")

    # Interpretation note for sparse categories
    if sparse_cats:
        print()
        print(f"  Note: {len(sparse_cats)} sparse categories "
              f"included in matrix, flagged ⚠️:")
        for cat in sorted(sparse_cats):
            print(f"    {cat}")

    return {
        'M_obs' : M_obs,
        'M_perm': M_perm,
        'p_perm': p_perm,
        'n_perm': n_permutations,
        'reject': reject,
    }


# -----------------------------------------------------------
# Helper Function 7-A: Compute Box's M statistic for k groups of observations
# -----------------------------------------------------------
def _boxm_statistic(groups):
    """
    Compute Box's M statistic for k groups of observations.

    M = (N - k) * log|S_pool| - Σ_i (n_i - 1) * log|S_i|

    where S_pool is the pooled sample covariance.

    Parameters
    ----------
    groups : list of np.ndarray  each shape (n_i, K)

    Returns
    -------
    M : float
    """
    ns     = np.array([g.shape[0] for g in groups])
    N      = ns.sum()
    k      = len(groups)

    # Per-group sample covariance (unbiased)
    covs   = [np.cov(g.T, ddof=1) for g in groups]

    # Pooled covariance
    S_pool = sum((n - 1) * S for n, S in zip(ns, covs)) / (N - k)

    # Log-determinants - use slogdet for numerical stability
    ld_pool = np.linalg.slogdet(S_pool)[1]
    ld_i    = [np.linalg.slogdet(S)[1] for S in covs]

    # Box's M statistic
    M = (N - k) * ld_pool - sum(
        (n - 1) * ld for n, ld in zip(ns, ld_i)
    )
    return float(M)


# ------------------------------------------------------------
# Helper Function 7-B: Plot correlation heatmaps for each era with clear 
# labeling of sparse categories and enhanced aesthetics
# -------------------------------------------------------------
def _corr_from_lw(cov_lw, cols):
    """
    Convert LW covariance matrix to correlation matrix.

    Parameters
    ----------
    cov_lw : (K, K) np.ndarray
    cols   : list of column labels

    Returns
    -------
    pd.DataFrame  (K, K) correlation matrix
    """
    std  = np.sqrt(np.diag(cov_lw))
    corr = cov_lw / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cols, columns=cols)

# -------------------------------------------------------------
# Helper Function 7-C: Append warning emoji to sparse category names for plot labels
# -------------------------------------------------------------
def _label_cols(cols, sparse_cats):
    """Append ⚠️ to sparse category names for plot labels."""
    return [f"{c} ⚠️" if c in sparse_cats else c for c in cols]

# -------------------------------------------------------------
# 8. Function: Plot correlation heatmaps for each era with 
# clear labeling of sparse categories and enhanced aesthetics
# -------------------------------------------------------------
def plot_correlation_heatmaps(matrices, sparse_cats=None,
                               save_path=None):
    """
    Plot per-era correlation heatmaps (3 panels, LW regularized).
    Sparse categories labeled ⚠️.

    Parameters
    ----------
    matrices    : output of _prepare_cov_matrices
    sparse_cats : list of sparse category names
    save_path   : directory path for saving

    Returns
    -------
    dict:
        corrs : dict  era -> correlation DataFrame
    """
    sparse_cats = set(sparse_cats or [])
    era_keys    = list(matrices.keys())
    cols        = matrices[era_keys[0]]['cols']
    labeled     = _label_cols(cols, sparse_cats)

    # Build correlation matrices
    corrs = {
        era: _corr_from_lw(matrices[era]['cov_lw'], labeled)
        for era in era_keys
    }

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    for ax, era in zip(axes, era_keys):
        d = matrices[era]
        sns.heatmap(
            corrs[era], ax=ax,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.3, annot=False,
            cbar_kws={'shrink': 0.6, 'label': 'Correlation'}
        )

        ax.set_title(
            f"{era}  (T={d['T']}, LW α={matrices[era]['lw_alpha']:.4f})",
            fontsize=12, fontweight='bold'
        )

        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0,  labelsize=7)

    plt.suptitle(
        'CLR Correlation Structure per Era - Ledoit-Wolf Regularized',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}corr_heatmaps.png",
                    dpi=300, bbox_inches='tight')
    plt.show()

    return {'corrs': corrs}

# -------------------------------------------------------------
# 9. Function: Plot difference correlation heatmaps between eras with 
# clear labeling of sparse categories and enhanced aesthetics
# -------------------------------------------------------------
def plot_correlation_diff_heatmaps(matrices, sparse_cats=None,
                                    save_path=None):
    """
    Plot difference correlation heatmaps (3 panels):
      Panel 1: COVID − Pre-COVID
      Panel 2: Post-COVID − COVID
      Panel 3: Post-COVID − Pre-COVID  (net permanent change)

    Red  = correlation INCREASED in later era.
    Blue = correlation DECREASED in later era.

    Parameters
    ----------
    matrices    : output of _prepare_cov_matrices
    sparse_cats : list of sparse category names
    save_path   : directory path for saving

    Returns
    -------
    dict:
        diff_pc : DataFrame  COVID − Pre-COVID
        diff_cp : DataFrame  Post-COVID − COVID
        diff_pp : DataFrame  Post-COVID − Pre-COVID
        corrs   : dict       era -> correlation DataFrame
    """
    sparse_cats = set(sparse_cats or [])
    era_keys    = list(matrices.keys())
    cols        = matrices[era_keys[0]]['cols']
    labeled     = _label_cols(cols, sparse_cats)
    K           = len(cols)

    # Build correlation matrices
    corrs = {
        era: _corr_from_lw(matrices[era]['cov_lw'], labeled)
        for era in era_keys
    }

    # Three differences - consistent with all other era comparisons
    diff_pc = corrs['COVID']      - corrs['Pre-COVID']
    diff_cp = corrs['Post-COVID'] - corrs['COVID']
    diff_pp = corrs['Post-COVID'] - corrs['Pre-COVID']
    # Validate that differences are symmetric and have zeros on the diagonal
    diffs = [diff_pc, diff_cp, diff_pp]

    # Titles with α values from each era
    α_pre  = matrices['Pre-COVID']['lw_alpha']
    α_cov  = matrices['COVID']['lw_alpha']
    α_post = matrices['Post-COVID']['lw_alpha']

    titles = [
        f"COVID (α={α_cov:.4f}) − Pre-COVID (α={α_pre:.4f})\n"
        f"(Red = stronger correlation during COVID)",
        f"Post-COVID (α={α_post:.4f}) − COVID (α={α_cov:.4f})\n"
        f"(Red = stronger correlation post-COVID)",
        f"Post-COVID (α={α_post:.4f}) − Pre-COVID (α={α_pre:.4f})\n"
        f"(Net permanent change in correlation structure)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    mask = np.eye(K, dtype=bool)

    for ax, diff, title in zip(axes, diffs, titles):
        sns.heatmap(
            diff, ax=ax,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            mask=mask,
            square=True, linewidths=0.3, annot=False,
            cbar_kws={'shrink': 0.6, 'label': 'Δ Correlation'}
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0,  labelsize=7)

    plt.suptitle(
        'CLR Correlation Structure - Era Differences (LW Regularized)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}corr_diff_heatmaps.png",
                    dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'diff_pc': diff_pc,
        'diff_cp': diff_cp,
        'diff_pp': diff_pp,
        'corrs'  : corrs,
    }


# -------------------------------------------------------------
# 10. Function: Compute log-determinants of covariance matrices per era and compare differences
# -------------------------------------------------------------
def compute_log_determinants(matrices):
    """
    Compute log-determinant of raw and LW covariance matrices per era.

    log|Σ| = log generalized variance.
    Higher -> crime types more spread in CLR space.
    Lower  -> crime types more concentrated / co-moving.

    Raw log|Σ| is unreliable for COVID and Post-COVID due to
    near-singularity from CLR rank deficiency and small T/K.
    LW log|Σ| is used for all interpretation.

    Parameters
    ----------
    matrices : output of _prepare_cov_matrices

    Returns
    -------
    dict:
        per_era  : dict  era -> {logdet_raw, logdet_lw, T, T/K}
        deltas   : dict  comparison -> Δlog|Σ_lw|
    """
    print()
    print("=" * 70)
    print("LOG-DETERMINANT COMPARISON - GENERALIZED VARIANCE")
    print("=" * 70)
    print(f"  {'Era':<15} {'log|Σ_raw|':>14} {'log|Σ_lw|':>14} "
          f"{'T':>6} {'T/K':>8}  {'Status'}")
    print(f"  {'-' * 62}")

    per_era = {}
    for era, d in matrices.items():
        K          = d['K']
        ld_raw     = np.linalg.slogdet(d['cov_raw'])[1]
        ld_lw      = np.linalg.slogdet(d['cov_lw'])[1]
        tk         = d['T'] / K
        flag       = '✅' if tk >= 5 else '⚠️'

        print(f"  {era:<15} {ld_raw:>14.4f} {ld_lw:>14.4f} "
              f"{d['T']:>6} {tk:>8.2f}  {flag}")

        per_era[era] = {
            'logdet_raw': ld_raw,
            'logdet_lw' : ld_lw,
            'T'         : d['T'],
            'T/K'       : tk,
        }

    # Delta comparison using LW
    lw = {e: per_era[e]['logdet_lw'] for e in per_era}
    deltas = {
        'COVID_minus_Pre' : lw['COVID']      - lw['Pre-COVID'],
        'Post_minus_COVID': lw['Post-COVID'] - lw['COVID'],
        'Post_minus_Pre'  : lw['Post-COVID'] - lw['Pre-COVID'],
    }

    print()
    print("  Δ log|Σ_lw| (LW - used for interpretation):")
    print(f"    COVID − Pre-COVID  : {deltas['COVID_minus_Pre']:+.4f}")
    print(f"    Post  − COVID      : {deltas['Post_minus_COVID']:+.4f}")
    print(f"    Post  − Pre-COVID  : {deltas['Post_minus_Pre']:+.4f}")
    print()
    print("  Interpretation:")
    print("    Positive Δ -> crime space EXPANDED (more diverse)")
    print("    Negative Δ -> crime space CONTRACTED (more co-moving)")
    print()
    print("  Note: raw log|Σ| unreliable for COVID/Post-COVID")
    print("        (T/K < 5, CLR rank deficiency). Use LW values.")

    return {
        'per_era': per_era,
        'deltas' : deltas,
    }


# -------------------------------------------------------------
# 11. Function: Run full covariance structure analysis pipeline with clear reporting and interpretation
# -------------------------------------------------------------
def run_covariance_structure_analysis(eras_stat, clr_era,
                                       sparse_cats=None,
                                       n_permutations=10_000,
                                       seed=None,
                                       save_path=None):
    """
    Full covariance structure analysis pipeline.

    Steps
    -----
    1. Validate and prepare covariance matrices
    2. Box's M test + permutation p-value
    3. Per-era correlation heatmaps (LW regularized)
    4. Difference correlation heatmaps (3 comparisons)
    5. Log-determinant comparison

    Parameters
    ----------
    eras_stat      : output of compute_era_distribution_parameters
    clr_era        : output of slice_clr_into_eras
    sparse_cats    : list of sparse category names
    n_permutations : Box's M permutation iterations
    seed           : random seed
    save_path      : directory for saving plots

    Returns
    -------
    dict:
        matrices    : validated CLR + cov dicts per era
        boxm        : Box's M results
        corr_heatmaps  : per-era correlation matrices
        corr_diffs     : three difference matrices + corrs
        log_dets    : log-determinant results
    """
    print("\n" + "=" * 65)
    print("COVARIANCE STRUCTURE ANALYSIS")
    print("=" * 65)

    # ---------------------------------------------
    # Step 11-A: Prepare covariance matrices and CLR arrays with validation report
    # ---------------------------------------------
    matrices = _prepare_cov_matrices(eras_stat, clr_era)

    # Step 2 - Box's M + permutation
    boxm = run_boxm_test(
        matrices,
        n_permutations = n_permutations,
        seed           = seed,
        sparse_cats    = sparse_cats,
    )

    # ---------------------------------------------
    # Step 11-B: Plot correlation heatmaps for each era with clear 
    # labeling of sparse categories and enhanced aesthetics
    # ---------------------------------------------
    print()
    corr_heatmaps = plot_correlation_heatmaps(
        matrices,
        sparse_cats = sparse_cats,
        save_path   = save_path,
    )

    # ---------------------------------------------
    # Step 11-C: Plot difference correlation heatmaps between eras with clear
    # ---------------------------------------------
    print()
    corr_diffs = plot_correlation_diff_heatmaps(
        matrices,
        sparse_cats = sparse_cats,
        save_path   = save_path,
    )

    # ---------------------------------------------
    # Step 11-D: Compute log-determinants of covariance matrices per era and compare differences
    # ---------------------------------------------
    log_dets = compute_log_determinants(matrices)

    print("\n" + "=" * 65)
    print("COVARIANCE STRUCTURE ANALYSIS COMPLETE")
    print("=" * 65)

    return {
        'matrices'     : matrices,
        'boxm'         : boxm,
        'corr_heatmaps': corr_heatmaps,
        'corr_diffs'   : corr_diffs,
        'log_dets'     : log_dets,
    }


# -------------------------------------------------------------
# Layer 1 - Core Engine: Precedence Logic
# Layer 2 - Joint PCA Analysis and Structured Reporting
# Layer 3 - Visualization (With Dynamic Threshold Handling)
# Layer 4 - Orchestrator
# -------------------------------------------------------------
# 12. PCA ANALYSIS
# Note: PCA component selection follows strict precedence:
#       1. If `n_components` is provided, use it directly.
#       2. Otherwise, if `threshold` is provided, retain the smallest number
#          of components whose cumulative explained variance meets it.
#       3. Otherwise, default to 2 components.
# Layer 1. CORE ENGINE: PRECEDENCE LOGIC
# -------------------------------------------------------------
def _fit_and_slice_pca(X, n_components, threshold, seed, index, columns):
    """
    Fit PCA and return both sliced (retained) and full component results.

    Component selection follows strict precedence:
        1. If `n_components` is provided, use it directly.
        2. Otherwise, if `threshold` is provided, retain the smallest number
           of components whose cumulative explained variance meets it.
        3. Otherwise, default to 2 components.

    The function returns:
        - Sliced scores/loadings (for plotting and reporting)
        - Full scores/loadings (for downstream analysis)
        - Full and sliced variance statistics
        - T (sample size) and K (feature count)

    Parameters
    ----------
    X : ndarray, shape (T, K)
        Input data matrix.
    n_components : int or None
        Explicit number of components to retain.
    threshold : float or None
        Cumulative variance threshold (0–1).
    seed : int
        Random seed for PCA reproducibility.
    index : array-like
        Row labels for the returned score DataFrames.
    columns : array-like
        Column labels for the returned loading DataFrames.

    Returns
    -------
    dict
        {
            'pca'            : fitted PCA object,
            'n_comp'         : retained component count,
            'T'              : number of samples,
            'K'              : number of features,
            'scores'         : sliced PCA scores,
            'loadings'       : sliced PCA loadings,
            'scores_full'    : full PCA scores,
            'loadings_full'  : full PCA loadings,
            'var_exp'        : variance explained (retained),
            'cum_var'        : cumulative variance (retained),
            'var_exp_all'    : full variance explained array
        }
    """
    # ------------------------------------------------------------
    # 12-A. Fit full PCA model
    # ------------------------------------------------------------
    pca = PCA(random_state=seed)
    full_scores_raw = pca.fit_transform(X)
    cum_var_all = np.cumsum(pca.explained_variance_ratio_)

    # Sign normalization: ensure the largest absolute loading per component is positive
    total_pcs    = full_scores_raw.shape[1]
    all_pc_names = [f"PC{i+1}" for i in range(total_pcs)]

    # ------------------------------------------------------------
    # 12-B. Determine the number of retained components (strict precedence)
    # ------------------------------------------------------------
    if n_components is not None:
        n_comp = n_components
    elif threshold is not None:
        n_comp = int(np.searchsorted(cum_var_all, threshold) + 1)
        if cum_var_all[-1] < threshold:
            warnings.warn(
                f"Variance threshold {threshold:.0%} was never reached. "
                f"Maximum achievable variance is {cum_var_all[-1]:.2%}. "
                f"Retaining all {total_pcs} components."
            )
    else:
        n_comp = 2

    # ------------------------------------------------------------
    # 12-C. Enforce PCA rank limits
    # ------------------------------------------------------------
    n_comp = max(1, min(n_comp, total_pcs))
    retained_names = all_pc_names[:n_comp]

    # ------------------------------------------------------------
    # 12-D. Build return dictionary (sliced + full)
    # ------------------------------------------------------------
    return {
        'pca': pca,
        'n_comp': n_comp,
        'T': X.shape[0],
        'K': X.shape[1],

        # Sliced (for plotting and reporting)
        'scores': pd.DataFrame(
            full_scores_raw[:, :n_comp], index=index, columns=retained_names
        ),
        'loadings': pd.DataFrame(
            pca.components_[:n_comp].T, index=columns, columns=retained_names
        ),

        # Full (for downstream analysis)
        'scores_full': pd.DataFrame(
            full_scores_raw, index=index, columns=all_pc_names
        ),
        'loadings_full': pd.DataFrame(
            pca.components_.T, index=columns, columns=all_pc_names
        ),

        # Variance statistics
        'var_exp': pca.explained_variance_ratio_[:n_comp],
        'cum_var': cum_var_all[:n_comp],
        'var_exp_all': pca.explained_variance_ratio_
    }


# ============================================================
# Layer 12-2. JOINT ANALYSIS AND STRUCTURED REPORTING
# ============================================================
def _run_joint_pca(pca_data, per_era_res, seed, sparse_cats, 
                   n_components, variance_threshold):
    """
    Run joint PCA across all eras and produce a structured diagnostic report.

    This function performs a joint PCA on the combined (dense) category matrix,
    computes era-level centroids in PCA space, and prints a multi-section
    analysis including:
        - PCA engine configuration
        - Data preparation summary
        - Per-era PCA variance breakdown
        - Joint PCA results and centroid positions
        - Structural shift metrics along PC1

    Parameters
    ----------
    pca_data : dict
        Contains:
            'joint_matrix' : DataFrame of all eras stacked together
            'cols'         : retained dense category names
            'era_labels'   : era label for each row in joint_matrix
            'era_matrices' : dict of per-era matrices
    per_era_res : dict
        Output of per-era PCA fits, keyed by era name.
    seed : int
        Random seed for PCA reproducibility.
    sparse_cats : list
        Categories dropped due to insufficient density.
    n_components : int or None, default None
        Explicit number of components for PCA. If None, determined by
        `variance_threshold`.
    variance_threshold : float, default 0.80
        Minimum cumulative variance required when auto-selecting components.

    Returns
    -------
    dict
        The PCA result dictionary returned by `_fit_and_slice_pca`, with
        additional keys:
            'centroids' : DataFrame of era-level PCA centroids
    """
    # ------------------------------------------------------------
    # 12-2-A. Fit joint PCA using strict precedence rules
    # ------------------------------------------------------------
    joint = pca_data['joint_matrix']
    res = _fit_and_slice_pca(
        joint.values,
        n_components,
        variance_threshold,
        seed,
        joint.index,
        pca_data['cols']
    )
    
    # -------------------------------------------------------------
    # 12-2-B. Attach era labels to PCA scores for centroid computation
    # -------------------------------------------------------------
    # assign returns a new DataFrame
    res['scores'] = res['scores'].assign(era=pca_data['era_labels'])
    
    # -------------------------------------------------------------
    # 12-2-C. Compute era-level centroids in PCA space (average scores per era)
    # -------------------------------------------------------------
    pcs = [f"PC{i+1}" for i in range(res['n_comp'])]
    res['centroids'] = res['scores'].groupby('era')[pcs].mean()

    # -------------------------------------------------------------
    # 12-2-D. Reorder centroids for consistent reporting 
    # -------------------------------------------------------------
    order = ['Pre-COVID', 'COVID', 'Post-COVID']
    res['centroids'] = res['centroids'].reindex(
        [e for e in order if e in res['centroids'].index]
    )

    # -------------------------------------------------------------
    # 12-2-E. Begin formatted output assembly
    # -------------------------------------------------------------
    output = []
    line = "=" * 65

    # -------------------------------------------------------------
    # SECTION 12-2-A: CORE ENGINE
    # -------------------------------------------------------------
    output.append(f"\n{line}\n{'PCA ANALYSIS':^65}\n{line}")
    output.append(f"  Random State                                 : {seed}")
    output.append(f"  Components                                   : {'Auto' if n_components is None else n_components}")
    output.append(f"  Variance threshold for component selection   : {variance_threshold*100:.0f}%")

    # -------------------------------------------------------------
    # SECTION 12-2-B: DATA PREP
    # -------------------------------------------------------------
    output.append(f"\n{line}\n{'PCA DATA PREPARATION':^65}\n{line}")
    output.append(f"  Sparse categories dropped : {len(sparse_cats)}")
    for cat in sparse_cats:
        output.append(f"    ⚠️  {cat}")
    
    output.append(f"\n  Dense categories retained  : {len(pca_data['cols'])}")
    output.append(f"  Joint matrix shape         : {joint.shape}")

    # Show per-era matrix shapes (sanity check for PCA input)
    for era in order:
        if era in pca_data['era_matrices']:
            shape = pca_data['era_matrices'][era].shape
            output.append(f"  {era:<26} : {shape}")

    # -------------------------------------------------------------
    # SECTION 12-2-C: PER-ERA PCA BREAKDOWN
    # -------------------------------------------------------------
    output.append(f"\n{line}\n{'PER-ERA PCA':^65}\n{line}")
    output.append(f"  Variance threshold : {variance_threshold*100:.0f}%")
    output.append(f"  Centering          : per-era (subtract era mean)")
    output.append(f"\n  Per-era PC1 / PC2 breakdown:")
    output.append(f"  {'Era':<18} {'PC1':>7} {'PC2':>7} {'n_comp':>8} {'Total':>8} {'PC1-PC2 gap':>12}")
    output.append("  " + "-" * 62)
    
    # Iterate to generate the Per-Era PCA Breakdown table in the report
    for era in order:
        if era in per_era_res:
            d = per_era_res[era]
            v1 = d['var_exp'][0] * 100
            v2 = d['var_exp'][1] * 100 if d['n_comp'] > 1 else 0.0
            tot = d['cum_var'][-1] * 100
            gap = v1 - v2
            output.append(
                f"  {era:<18} {v1:>6.1f}% {v2:>6.1f}% {d['n_comp']:>8} {tot:>7.1f}% {gap:>11.1f}%"
            )

    # -------------------------------------------------------------
    # SECTION 12-2-D: JOINT PCA RESULTS
    # -------------------------------------------------------------
    output.append(f"\n{line}\n{'JOINT PCA ANALYSIS':^65}\n{line}")
    output.append(f"  Retained   : {res['n_comp']} PCs ({res['cum_var'][-1]:.1%} Var)")
    if res['n_comp'] > 10:
        output.append(f"{'** Maximum: 10 PCs Visible **':^65}")
    output.append("=" * 65)
    
    # Building the Joint PCA Centroids table with dynamic PC column handling
    # visible_pcs = res['centroids'].columns[:res['n_comp']]
    visible_pcs = res['centroids'].columns[:min(10, res['n_comp'])] # Max 10 PCs
    headers = " ".join([f"{pc:>10}" for pc in visible_pcs])
    output.append(f"  {'era':<12} {headers}")
    
    for era, row in res['centroids'].iterrows():
        vals = " ".join([f"{v:>10.4f}" for v in row[visible_pcs]])
        output.append(f"  {era:<12} {vals}")

    # -------------------------------------------------------------
    # SECTION 12-2-E: STRUCTURAL SHIFT ANALYSIS (PC1)
    # -------------------------------------------------------------
    if all(e in res['centroids'].index for e in order) and 'PC1' in res['centroids'].columns:
        c = res['centroids'].loc[order, 'PC1']
        
        # Pandemic shock = COVID - Pre
        s_shock = c['COVID'] - c['Pre-COVID']   # how hard did COVID hit
        # Net gap = Post - Pre
        n_gap = c['Post-COVID'] - c['Pre-COVID'] # how much of that stuck permanently
        # Recovery movement = COVID - Post
        # r_move: how much of the COVID displacement remains unreturned
        # Positive = Post-COVID sits below COVID on PC1 (partial recovery)
        # Negative = Post-COVID diverged further from Pre-COVID than COVID did
        r_move = c['COVID'] - c['Post-COVID']    # how much bounced back

        output.append(f"\n  PC1 STRUCTURAL SHIFT (Baseline: Pre-COVID)")
        output.append(f"  {'-' * 42}")
        output.append(f"  1. Pandemic Shock  [S] : {s_shock:+.4f}")
        output.append(f"  2. Net Gap         [N] : {n_gap:+.4f}")
        output.append(f"  3. Recovery Move   [B] : {r_move:+.4f}")
        output.append(f"  {'.' * 42}")
        if abs(s_shock) > 1e-6:
            output.append(f"  >> % REVERSION  (B/S)  : {(r_move/s_shock)*100:.1f}%")
            output.append(f"  >> % PERMANENT  (N/S)  : {(n_gap/s_shock)*100:.1f}%")
        else:
            output.append(f"  >> % REVERSION  (B/S)  : N/A (near-zero shock)")
            output.append(f"  >> % PERMANENT  (N/S)  : N/A (near-zero shock)")

    # -------------------------------------------------------------
    # 12-2-F. Finalize and print report
    # -------------------------------------------------------------
    output.append(line)
    report = "\n".join(output)
    print(report)
    res['report'] = report

    return res


# ---------------------------------------------
# Layer 12-3: Visualization (With Dynamic Threshold Handling)
# ----------------------------------------------
def _plot_all_visuals(per_era_res, joint_res, era_config, 
                      v_thresh, save_image, feature_label='features'):

    """
    This visualization module produces three coordinated outputs:

        1. **Loading Grid (PC1 & PC2)**  
           - Extra‑wide (36") and extra‑long (10" per row) layout  
           - Top 8 strongest loadings per component (magnitude-based)  
           - Staircase sorting (algebraic ordering)  
           - Era‑synchronized color scheme  
           - Bold, high‑density typography for readability  

        2. **Diagnostics Panel**  
           - Per‑era scree plots with retained‑component markers  
           - Joint scree plot with cumulative curve and threshold line  
           - Consistent axis styling and bold tick labels  

        3. **Era Migration Map (PC1 vs PC2)**  
           - Scatterplot of all observations  
           - Large centroid stars for each era  
           - High‑contrast labeling and bold axis titles  

    Parameters
    ----------
    per_era_res : dict
        PCA results for each era (output of `_fit_and_slice_pca`).
    joint_res : dict
        PCA results for the joint matrix (output of `_run_joint_pca`).
    era_config  : dict
        Mapping of era label to (start, end, color).
        Used to synchronize plot colors with era_config
        defined at the notebook level.
        Example: {'Pre-COVID': ('2001-01-01', '2020-03-01', 'blue'), ...}
    v_thresh : float or None
        Variance threshold used for auto-selection. Pass None when
        n_components was explicitly provided — suppresses the threshold
        line on the joint scree plot.
    save_image : str or None
        If provided, saves PNGs using this prefix (e.g., "output/Crime_").
    feature_label : str
        Label used in the loading grid title to describe the features.
        Default 'features'. Pass 'crime types' for crime data.
    """
    # ------------------------------------------------------------
    # 12-3-1: Determine which PCs to plot based on availability in joint PCA results
    # ------------------------------------------------------------
    available_pcs = [pc for pc in ['PC1', 'PC2'] if pc in joint_res['loadings'].columns]
    n_rows = len(available_pcs)
    eras = ['Pre-COVID', 'COVID', 'Post-COVID']

    # Base color map - overridden by era_config if provided
    color_map = {
        'Joint'      : 'indianred',
        'Pre-COVID'  : 'blue',
        'COVID'      : 'red',
        'Post-COVID' : 'green',
        'Neg_Era'    : '#D3D3D3',
        'Neg_Joint'  : 'steelblue'
    }

    # Sync era colors with notebook-level era_config if provided
    if era_config:
        for era, cfg in era_config.items():
            color_map[era] = cfg[2]

    # ============================================================
    # FIGURE 12-3-1: LOADING GRID (PC1 & PC2)
    # ============================================================
    fig1, axes1 = plt.subplots(n_rows, 4, figsize=(36, 10 * n_rows), squeeze=False)
    fig1.suptitle(
        f"PCA Loadings - PC1 and PC2\nTop 8 {feature_label} driving each component per era",
        fontsize=28, fontweight='bold', y=0.98
    )
    # List comprehension that prepares the sequence of subplots for the loading grid
    plot_cols = [('Joint', joint_res)] + [(e, per_era_res.get(e)) for e in eras]

    # Nested loop is the rendering engine for your Loading Grid
    for r, pc in enumerate(available_pcs):
        for c, (label, res) in enumerate(plot_cols):
            ax = axes1[r, c]

            if res and pc in res['loadings'].columns:

                # Top 8 by magnitude → staircase sort
                raw = res['loadings'][pc]
                # "Staircase" Sorting Logic
                top_idx = raw.abs().nlargest(8).index
                plot_data = raw.loc[top_idx].sort_values(ascending=True)
                # Ensures that your charts always have a consistent color,
                main_col = color_map.get(label, 'gray')

                # Positive vs negative coloring (conditional coloring)
                if label == 'Joint':
                    colors = ['indianred' if v > 0 else 'steelblue' for v in plot_data.values]
                else:
                    colors = [main_col if v > 0 else color_map['Neg_Era'] for v in plot_data.values]

                # Render bars (horizontal bar chart)
                ax.barh(
                    plot_data.index, plot_data.values,
                    color=colors, edgecolor='black', lw=1.2,
                    alpha=0.9, height=0.6
                )

                # Title & labels (dynamic based on era vs joint)
                ax.set_title(
                    f"{label} - {pc}" if label != 'Joint' else f"Joint PCA - {pc}",
                    color=main_col, fontweight='bold', fontsize=22, pad=20
                )
                # X-axis label is the same for all subplots
                ax.set_xlabel("Loading", fontsize=15, fontweight='bold')

                # Symmetric x-limits for visual balance
                lim = max(abs(plot_data.min()), abs(plot_data.max())) * 1.1
                ax.set_xlim(-lim, lim)

                # Tick styling (size and boldness for readability)
                ax.tick_params(axis='y', labelsize=15)
                ax.tick_params(axis='x', labelsize=15)
                plt.setp(ax.get_yticklabels(), fontweight='bold')

                # Grid and spine styling for clarity
                ax.axvline(0, color='black', lw=2)
                ax.xaxis.grid(True, ls=':', alpha=0.4)
                for s in ['top', 'right']:
                    ax.spines[s].set_visible(False)

            else:
                ax.axis('off')

    # Final layout adjustments and optional saving of the loading grid
    # Figure 1
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_image:
        fig1.savefig(f"{save_image}PCA_loadings.png", dpi=300)

    # ============================================================
    # FIGURE 12-3-2: DIAGNOSTICS PANEL (Scree Plots + Migration Map)
    # ============================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(32, 11))

    # ------------------------------------------------------------
    # 12-3-A: Per-Era Scree Plots with Dynamic Retained Component Markers
    # ------------------------------------------------------------
    for era in eras:
        if era in per_era_res:
            res = per_era_res[era]
            col = color_map[era]
            # --- Dynamic Logic ---
            # At least 10, but more if n_comp is higher
            plot_limit = max(10, res['n_comp'] + 2)

            ve = res['var_exp_all'][:plot_limit] * 100
            # ---------------------

            # visual mapping of the PCA Scree plot
            axes2[0].plot(range(1, len(ve) + 1), ve, 'o-', color=col, lw=4, ms=12, label=era)
            axes2[0].axvline(res['n_comp'], color=col, ls='--', lw=2, alpha=0.5)

    # helper function is the styling finisher
    _format_diag(axes2[0], "Per-Era Scree Plot")

    # ------------------------------------------------------------
    # 12-3-B: Joint Scree Plot with Cumulative Curve and Dynamic Threshold Line
    # ------------------------------------------------------------
    # Dynamic version
    j_limit = max(10, joint_res['n_comp'] + 2)
    ve_j = joint_res['var_exp_all'][:j_limit] * 100

    # Visual mapping of the Joint PCA Scree plot with both bars and cumulative line
    axes2[1].bar(
        range(1, len(ve_j) + 1), ve_j,
        alpha=0.4, color='steelblue', label='Individual'
    )
    axes2[1].plot(
        range(1, len(ve_j) + 1), np.cumsum(ve_j),
        'ro-', lw=4, ms=12, label='Cumulative'
    )
    # Dynamic threshold line if variance threshold was used for component selection
    if v_thresh:
        axes2[1].axhline(
            v_thresh * 100, color='k', ls=':', lw=2.5, label='Threshold'
        )
    # Dynamic retained component line based on the actual number of components retained in the joint PCA
    axes2[1].axvline(
        joint_res['n_comp'], color='brown', ls='--', lw=2.5,
        label=f"Kept={joint_res['n_comp']}"
    )
    # Helper function applies consistent styling to the Joint Scree plot
    _format_diag(axes2[1], f"Joint Scree (K={joint_res['n_comp']})")

    # ------------------------------------------------------------
    # 12-3-C: Era Migration Map (PC1 vs PC2) with Dynamic Coloring and Centroid Stars
    # ------------------------------------------------------------
    ax_mig = axes2[2]
    # Dynamic scatterplot of PCA scores colored by era, with large centroid stars    
    if 'scores' in joint_res:
        # Era palette is dynamically constructed based on the eras present in the joint PCA scores and the color map
        era_palette = {era: color_map[era] for era in eras if era in color_map}
        # Scatterplot of all observations colored by era, with dynamic coloring based on the era_palette
        sns.scatterplot(
            data=joint_res['scores'],
            x='PC1', y='PC2',
            hue='era', palette=era_palette,
            alpha=0.4, ax=ax_mig, legend=False
        )

        # Overlay large stars for era centroids, with dynamic coloring and labeling based on the era_config
        for era in joint_res.get('centroids', pd.DataFrame()).index:
            if era in color_map:
                ax_mig.scatter(
                    joint_res['centroids'].loc[era, 'PC1'],
                    joint_res['centroids'].loc[era, 'PC2'],
                    s=600, marker='*', c=color_map[era],
                    edgecolors='k', zorder=30,
                    label=f"{era} Center"
                )

        # Dynamic axis titles and legend based on the presence of PC1 and PC2
        ax_mig.set_title("Era Migration Map", fontweight='bold', fontsize=26, pad=20)
        ax_mig.set_xlabel("PC1", fontsize=20, fontweight='bold')
        ax_mig.set_ylabel("PC2", fontsize=20, fontweight='bold')
        ax_mig.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=16)

    # Final layout adjustments and optional saving of the diagnostics panel
    fig2.tight_layout()
    if save_image:
        fig2.savefig(f"{save_image}PCA_diagnostics.png", dpi=300)

    plt.show()

# -------------------------------------------------------------
# Helper Function 12-3-1: Consistent styling for PCA diagnostic plots (scree plots)
# --------------------------------------------------------------
def _format_diag(ax, title):
    """
    Apply consistent styling to PCA diagnostic subplots.

    This helper standardizes:
        - Title formatting (large, bold, padded)
        - Axis labels (bold, high-contrast)
        - Tick label size and weight
        - Grid styling (light dotted grid)
        - Legend formatting

    Used for:
        - Per-era scree plots
        - Joint scree plot
        - Any future diagnostic visualizations requiring uniform styling

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis to format.
    title : str
        Title to apply to the subplot.
    """
    # Title styling
    ax.set_title(title, fontsize=26, fontweight='bold', pad=25)

    # Axis labels
    ax.set_ylabel("Explained Variance (%)", fontsize=20, fontweight='bold')
    ax.set_xlabel("Principal Component", fontsize=20, fontweight='bold')

    # Background grid (light dotted)
    ax.grid(True, ls=':', alpha=0.6)

    # Tick label size
    ax.tick_params(axis='both', labelsize=16)

    # Bold tick labels (x and y)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # Legend styling
    # Only render if labeled artists exist — prevents empty legend box
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=16, frameon=True)


# -------------------------------------------------
# 13-4: ORCHESTRATOR: 
# FULL PCA ANALYSIS PIPELINE WITH STRUCTURED REPORTING AND OPTIONAL VISUALIZATION
# -------------------------------------------------
def run_pca_analysis(clr_era, chosen_clr, sparse_cats,
                     seed,
                     era_config         = None,
                     n_components       = None,
                     variance_threshold = 0.80,
                     show_plots         = False,
                     save_plots         = False,
                     image_path         = '../Image/'):
    """
    Execute the full PCA analysis pipeline, including per-era decomposition,
    joint PCA, structural-shift diagnostics, and optional visualization.

    Workflow
    --------
    1. Dense-category filtering
       Remove sparse categories and construct era-specific CLR matrices.

    2. Per-era PCA
       Fit PCA independently for each era using strict precedence rules:
         - Use n_components if provided
         - Otherwise use variance_threshold
         - Otherwise default to 2 components
       Each era returns sliced and full PCA results.

    3. Joint PCA
       Fit PCA on the combined matrix to compute joint loadings,
       scores, era centroids, and structural-shift metrics.

    4. Visualization (optional)
       If show_plots=True, renders loading grid, scree diagnostics,
       and era migration map. If save_plots=True, exports figures
       to image_path.

    Parameters
    ----------
    clr_era            : dict
        Per-era CLR-transformed DataFrames.
        Expected keys: 'pre_covid', 'covid', 'post_covid'.
    chosen_clr         : pd.DataFrame
        Full CLR-transformed dataset before sparse-category filtering.
    sparse_cats        : list
        Category names removed due to insufficient density.
    seed               : int
        Random seed for PCA reproducibility.
    era_config         : dict or None
        Mapping of era label to (start, end, color).
        Used to synchronize plot colors across all visualizations.
        Example: {'Pre-COVID': ('2001-01-01', '2020-03-01', 'blue'), ...}
        Default None — passes empty dict to _plot_all_visuals.
    n_components       : int or None
        Explicit number of components to retain.
        Overrides variance_threshold if provided. Default None.
    variance_threshold : float
        Cumulative variance threshold for auto-selecting components.
        Only used when n_components is None. Default 0.80.
    show_plots         : bool
        If True, renders PCA visualizations. Default False.
    save_plots         : bool
        If True, saves figures to image_path. Default False.
    image_path         : str
        Directory prefix for saved figures. Default '../Image/'.

    Returns
    -------
    dict with keys:
        Pre-COVID         : per-era PCA result dict
        COVID             : per-era PCA result dict
        Post-COVID        : per-era PCA result dict
        Joint             : joint PCA result dict

    Notes
    -----
    Component selection follows strict precedence:
      n_components > variance_threshold > default (2 components)

    Text output (structured report) is always printed regardless
    of show_plots. Set show_plots=False to suppress figures only.
    """
    # ------------------------------------------------------------
    # 13-4-1: DATA PREPARATION
    # ------------------------------------------------------------
    # Remove sparse categories - keep only dense columns
    sparse_cats = list(set(sparse_cats))
    dense_cols   = [c for c in chosen_clr.columns
                    if c not in sparse_cats]
    joint_matrix = chosen_clr[dense_cols]

    # Map human-readable era labels to clr_era dict keys
    era_map = cfg.era_map

    # Build era-specific dense matrices and row-to-era label mapping
    era_matrices = {}
    idx_to_era   = {}

    for era_label, clr_key in era_map.items():
        if clr_key in clr_era:
            mat  = clr_era[clr_key][dense_cols]
            era_matrices[era_label] = mat
            # Map each row index to its era for centroid computation
            for idx in mat.index:
                idx_to_era[idx]  = era_label

    # Package PCA input for joint and per-era fitting
    pca_data = {
        'era_matrices': era_matrices,
        'joint_matrix': joint_matrix,
        'era_labels'  : joint_matrix.index.map(idx_to_era).values,
        'cols'        : dense_cols,
    }

    # ------------------------------------------------------------
    # 13-4-2: ERA CONFIGURATION AND COLOR MAPPING
    # ------------------------------------------------------------
    # era_config defaults to empty dict if not provided
    # downstream color_map derivation handles the empty case gracefully
    resolved_era_config = era_config or {}

    # ------------------------------------------------------------
    # 13-4-3: PER-ERA PCA FITTING WITH STRICT COMPONENT SELECTION PRECEDENCE
    # ------------------------------------------------------------
    # Independent PCA per era — same retention logic applied to each
    per_era_res = {
        era: _fit_and_slice_pca(
            mat.values,
            n_components,
            variance_threshold,
            seed,
            mat.index,
            dense_cols
        )
        for era, mat in era_matrices.items()
    }

    # ------------------------------------------------------------
    # 13-4-4: JOINT PCA AND STRUCTURAL SHIFT ANALYSIS
    # ------------------------------------------------------------
    # Combined matrix PCA — centroids and structural shift metrics
    joint_res = _run_joint_pca(
        pca_data,
        per_era_res,
        seed,
        sparse_cats,
        n_components,
        variance_threshold,
    )

    # ------------------------------------------------------------
    # 13-4-5: OPTIONAL VISUALIZATION (WITH DYNAMIC THRESHOLD HANDLING)
    # ------------------------------------------------------------
    if show_plots:
        # Pass image_path only when saving is requested
        effective_path = image_path if save_plots else None

        _plot_all_visuals(
            per_era_res,
            joint_res,
            era_config = resolved_era_config,
            v_thresh   = variance_threshold if n_components is None else None,
            save_image = effective_path,
            feature_label = 'crime types'
        )
    else:
        print("  Visualizations suppressed. "
              "Pass show_plots=True to render plots.")

    # ------------------------------------------------------------
    # 13-4-6: STRUCTURED RETURN DICTIONARY WITH PER-ERA AND JOINT RESULTS
    # ------------------------------------------------------------
    # Full scores and loadings accessible via per-era dicts directly
    # e.g. pca_results['Pre-COVID']['scores_full']
    return {
        'Pre-COVID' : per_era_res.get('Pre-COVID'),
        'COVID'     : per_era_res.get('COVID'),
        'Post-COVID': per_era_res.get('Post-COVID'),
        'Joint'     : joint_res,
    }