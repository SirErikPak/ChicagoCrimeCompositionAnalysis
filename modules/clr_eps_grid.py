import numpy as np
import pandas as pd
from typing import  Any, Tuple, Iterable
import clr_config as cfg
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# ----------------------------------------------------------------------------
# Configuration parameters for clr_eps_grid.py
# ----------------------------------------------------------------------------
_N_PER_DECADE               = cfg.config_grid["_N_PER_DECADE"]
_INCLUDE_FIXED              = cfg.config_grid["_INCLUDE_FIXED"]
_MIN_MULTIPLIER_CANDIDATES  = cfg.config_grid["_MIN_MULTIPLIER_CANDIDATES"]
_Q_LOW                      = cfg.config_grid["_Q_LOW"]
_FLOOR                      = cfg.config_grid["_FLOOR"]
_MIN_STEP                   = cfg.config_grid["_MIN_STEP"]
# Pivot table keys from clr_config.py
_DATE_KEY                   = cfg.config_agg["_DATE_KEY"]
_GROUP_KEY                  = cfg.config_agg["_GROUP_KEY"]
_COUNTER_KEY                = cfg.config_agg["_COUNTER_KEY"]
# Sweep and selection parameters from config_sweep
_LARGE_CLR_THRESHOLD        = cfg.config_sweep["large_clr_threshold"]
_KENDALL_THRESHOLD          = cfg.config_sweep["kendall_threshold"]
_SPEARMAN_THRESHOLD         = cfg.config_sweep["spearman_threshold"]
_NEAR_ZERO_THRESHOLD        = cfg.config_sweep["near_zero_threshold"]
_SLACK_ZERO                 = cfg.config_sweep["slack_zero"]
_SLACK_KENDALL              = cfg.config_sweep["slack_kendall"]
_SLACK_SPEAR                = cfg.config_sweep["slack_spear"]
_ELBOW_THRESHOLD            = cfg.config_sweep["elbow_threshold"]
_FALLBACK_WEIGHTS           = cfg.config_sweep["fallback_weights"]

# ---------------------------------------------------------------------------
# Helper function 1-A: Pivot Table for building the eps grid
# ---------------------------------------------------------------------------
def _pivot(data_df: pd.DataFrame,
           index: str = _DATE_KEY,
           column: str = _GROUP_KEY,
           values: str = _COUNTER_KEY,
           fill_value: float = 0.0) -> pd.DataFrame:
    """
    Pivot already-aggregated, gap-filled long data into a T×K wide matrix.

    Input is expected to come from fill_missing(): one row per (date, group),
    gaps already filled, no duplicate (index, column) pairs, complete month
    range. pivot() is therefore safe.

    Adds a NaN safety net (fillna) in case any cell is absent despite the
    upstream fill, since NaN would break the downstream CLR log.
    """
    work = data_df.copy()

    # Date normalization (pyarrow-safe) — only matters if fill_missing emits
    # a string/pyarrow date; harmless if already datetime.
    s = work[index]
    if not (pd.api.types.is_datetime64_any_dtype(s) or str(s.dtype).startswith('timestamp')):
        raw = s.astype('string[pyarrow]').to_numpy(dtype=object)
        cleaned = pd.Series(raw, index=work.index).str.replace('-', '', regex=False)
        work[index] = pd.to_datetime(cleaned, format='%Y%m')

    pivot_df = (
        work
        .pivot(index=index, columns=column, values=values)
        .sort_index()
    )
    pivot_df.columns.name = None
    pivot_df = pivot_df.fillna(fill_value)   # safety net; no-op if already complete

    return pivot_df


# ---------------------------------------------------------------------------
# Helper function 1-B: Extract positive values for anchor generation 
#                      (build_eps_grid)
# ---------------------------------------------------------------------------
def _extract_positive(pivot: pd.DataFrame) -> np.ndarray:
    """
    Extract strictly positive values from a pivoted TXK matrix.

    This helper isolates all values greater than zero from the wide
    pivot matrix. It is used to ensure log‑safety prior to CLR or
    log‑ratio transformations, which require strictly positive inputs.

    Parameters
    ----------
    pivot : pd.DataFrame
        Wide TXK matrix produced by the pivot step. Must contain only
        numeric values.

    Returns
    -------
    np.ndarray
        A 1‑D array containing all strictly positive entries from the
        pivot matrix, flattened.
    """
    # -------------------------------------------------
    # Extract strictly positive values
    # CLR/log transforms require values > 0.
    # This flattens the TXK matrix and filters out
    # zeros and negatives to guarantee log‑safety.
    # -------------------------------------------------
    vals = pivot.values
    return vals[vals > 0]


# ---------------------------------------------------------------------------
# 1. Main function: Build the adaptive epsilon grid
# ---------------------------------------------------------------------------
def build_eps_grid(
    data_df: pd.DataFrame,
    n_per_decade: int = _N_PER_DECADE,
    floor: float = _FLOOR,
    include_fixed: Iterable[float] = _INCLUDE_FIXED,
    q_low: float = _Q_LOW,
    min_step: float = _MIN_STEP,
    multipliers: Iterable[float] = _MIN_MULTIPLIER_CANDIDATES,
) -> dict:
    """
    Construct a multi-zone epsilon grid for CLR zero-replacement sweeps.

    This function builds a log-spaced epsilon grid tailored to the empirical
    distribution of positive values in the compositional dataset. The grid is
    divided into three conceptual zones:

        - Zone 1 (Sub-Min / Anchor Bridge):
          Very dense sampling between the global floor and the smallest
          observed positive value. Captures the sensitive region where
          epsilon interacts directly with structural zeros.

        - Zone 2 (Data Transition Zone):
          Dense sampling around the lower quantile of the positive data
          distribution. Tracks the transition from zero-dominated to
          data-dominated CLR behavior.

        - Zone 3 (Plateau Monitoring Zone):
          Coarser sampling at larger epsilons to detect flattening or
          over-smoothing in CLR geometry.

    The final grid merges:
        - zone-specific log-spaced sequences,
        - scaled data-driven multipliers,
        - applies adaptive thinning (smaller step in sensitive regions),
        - and then re-adds the fixed anchors unconditionally so they are
          guaranteed to appear in the final grid.

    Values are returned at full numerical precision. Rounding is the
    responsibility of presentation code (display, plotting, write-up).

    Parameters
    ----------
    data_df : pd.DataFrame
        Raw long-format dataset containing compositional counts.
    n_per_decade : int
        Density of log-spaced samples per decade of epsilon.
    floor : float
        Global minimum epsilon floor (e.g., 1e-12).
    include_fixed : Iterable[float]
        Fixed epsilon anchors guaranteed to appear in the final grid.
    q_low : float
        Lower quantile (0-1) used to define the data-driven scale region.
    min_step : float
        Minimum relative step used during thinning above d_scale.
        Below d_scale, the gate is tightened to min_step / 2.
    multipliers : Iterable[float]
        Multipliers applied to the data scale to generate additional anchors.

    Returns
    -------
    dict
        {
            'eps_values': np.ndarray of final epsilon grid (full precision),
            'pivot_data': pivoted compositional matrix,
            'meta': {
                'data_min':   smallest positive value, or None if no positives,
                'data_scale': lower-quantile scale, or None if no positives,
                'grid_size':  number of epsilons in final grid,
            }
        }
    """
    # Materialize iterables once so we can take min/max and iterate freely.
    fixed_anchors = sorted(set(float(v) for v in include_fixed))
    mult_list = [float(v) for v in multipliers]

    pivot_data = _pivot(data_df)
    pos = _extract_positive(pivot_data)

    # Degenerate case: no positive values to drive zone construction.
    if pos.size == 0:
        eps_values = np.array(fixed_anchors, dtype=float)
        return {
            'eps_values': eps_values,
            'pivot_data': pivot_data,
            'meta': {
                'data_min':   None,
                'data_scale': None,
                'grid_size':  len(eps_values),
            },
        }

    # -----------------------------------------------
    # 1-A: Statistical anchors: data-driven points
    # ------------------------------------------------
    d_min = float(np.nanmin(pos))
    d_scale = float(np.percentile(pos, q_low * 100))
    search_floor = min(d_min, fixed_anchors[0] if fixed_anchors else d_min, floor)

    # -------------------------------------------------
    # 1-B: Zone construction: build dense log grids
    # -------------------------------------------------
    decades_to_cover = np.log10(d_min) - np.log10(search_floor)
    num_z1 = max(int(decades_to_cover * n_per_decade), 20)
    # Zone 1 covers from the global floor up to the smallest positive value
    z1 = np.logspace(np.log10(search_floor), np.log10(d_min), num=num_z1)

    # Zone 2 focuses on the transition region around d_scale
    z2 = np.logspace(np.log10(d_min), np.log10(d_scale * 10), num=max(n_per_decade, 20))

    # Zone 3 monitors the plateau region up to the largest fixed anchor or 10.0, whichever is larger
    z3_upper = max(max(fixed_anchors), 10.0) if fixed_anchors else 10.0
    z3 = np.logspace(np.log10(d_scale * 10), np.log10(z3_upper), num=10)

    # -------------------------------------------------
    # 1-C: Merge non-anchor sources and thin adaptively
    # -------------------------------------------------
    multiplier_anchors = d_scale * np.array(mult_list)
    combined = np.unique(np.concatenate([z1, z2, z3, multiplier_anchors]))

    # -------------------------------------------------
    # 1-D: Adaptive thinning: use a smaller step threshold 
    #      below d_scale to preserve resolution in the 
    #      sensitive region
    # --------------------------------------------------
    thinned = [combined[0]]
    for val in combined[1:]:
        dynamic_step = (min_step / 2.0) if val < d_scale else min_step
        if (val - thinned[-1]) / thinned[-1] >= dynamic_step:
            thinned.append(val)

    # -------------------------------------------------
    # 1-E: Re-add fixed anchors unconditionally to ensure 
    #      they are included in the final grid, even if 
    #      they violate the thinning step
    # -------------------------------------------------
    final = np.unique(np.concatenate([np.array(thinned), np.array(fixed_anchors)]))

    return {
        'eps_values': final,
        'pivot_data': pivot_data,
        'meta': {
            'data_min':   d_min,
            'data_scale': d_scale,
            'grid_size':  len(final),
        },
    }


# ---------------------------------------------------------------------------
# 2. Main function: Sweep over epsilon grid and compute CLR diagnostics
# ---------------------------------------------------------------------------
def sweep_epsilon_grid(
    pivot: pd.DataFrame,
    eps_grid: Iterable[float],
    large_clr_threshold: float = _LARGE_CLR_THRESHOLD,
    plot: bool = True,
    auto_select: bool = False,
    verbose: bool = False,
    kendall_threshold: float = _KENDALL_THRESHOLD,
    spearman_threshold: float = _SPEARMAN_THRESHOLD,
    near_zero_threshold: float | None = _NEAR_ZERO_THRESHOLD,
    zero_strategy: str = 'add_all',
) -> dict[str, Any]:
    """
    Sweep over epsilon (pseudo-count) values and compute CLR diagnostics.

    Computes CLR transformation and rank stability metrics for each eps in the grid,
    optionally plots diagnostics, and performs automated eps selection if requested.

    Parameters
    ----------
    pivot : pd.DataFrame
        (T, K) count matrix where T is time intervals and K is feature types.
        Must have at least 2 rows for rank correlation computation.
    eps_grid : Iterable[float]
        Pseudo-count values to evaluate. Must be positive and non-duplicating.
    large_clr_threshold : float, default from config_sweep['large_clr_threshold']
        Upper bound for acceptable max |CLR| values; used to flag sensitivity.

    plot : bool, default=True
        If True, generate and display diagnostics plots.

    auto_select : bool, default=False
        If True, run staged epsilon selection and populate selection metadata.

    verbose : bool, default=False
        If True, print the diagnostics table to stdout.

    kendall_threshold : float, default from config_sweep['kendall_threshold']
        Minimum Kendall tau for rank stability to satisfy constraint C2.

    spearman_threshold : float, default from config_sweep['spearman_threshold']
        Fallback Spearman rho threshold if kendall_threshold is unachievable.

    near_zero_threshold : float or None, default from config_sweep['near_zero_threshold']
        If not None, compute per-eps "near-zero" diagnostics using this absolute threshold
        on the per-row probabilities (after adding eps and row-normalizing). Set to None to
        omit these diagnostics entirely.

    zero_strategy : {'add_all', 'zero_only', 'multiplicative'}, default 'add_all'
        How zeros (and counts) are handled before CLR:
        - 'add_all'        : add eps to EVERY cell, then row-normalize.
                             (Original behavior — Bayesian/Laplace smoothing.)
        - 'zero_only'      : replace only zero cells with eps; leave non-zeros
                             untouched, then row-normalize.
        - 'multiplicative' : replace zeros with eps, shrink non-zeros to preserve
                             row totals (Martin-Fernandez 2003), then normalize.

        IMPORTANT: the selected epsilon is convention-dependent. If you change
        zero_strategy, the chosen eps may differ. Use the SAME zero_strategy in
        your downstream CLR transform that you used here. The chosen strategy is
        recorded both per-row in diagnostics_df and once in the returned meta dict.

    Returns
    -------
    dict[str, Any]
        Keys:
        - diagnostics_df: pd.DataFrame indexed by eps with computed metrics
          (includes a 'zero_strategy' column so rows are self-labeling when stacked)
        - clr_dict: dict mapping eps -> CLR DataFrame (for external use)
        - meta: dict containing auto-selection results (always present; minimal when auto_select=False)
        - fig: matplotlib Figure object (None if plot=False)

    Notes
    -----
    - `diagnostics_df` is indexed by the evaluated `eps` values (float) and is sorted
      in ascending order.
    """
    # -------------------------------------------------
    # 2-0: Input validation and preprocessing
    # -------------------------------------------------
    if pivot.shape[0] < 2:
        raise ValueError(
            f"pivot must have at least 2 rows to compute rank diagnostics; got shape {pivot.shape}"
        )

    # -------------------------------------------------
    # 2-A: Validate zero_strategy parameter
    # -------------------------------------------------
    _VALID_STRATEGIES = ('add_all', 'zero_only', 'multiplicative')
    if zero_strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"zero_strategy must be one of {_VALID_STRATEGIES}, got {zero_strategy!r}"
        )

    # ------------------------------------------------
    # 2-B: Validate eps_grid values: must be positive and non-duplicating
    # -------------------------------------------------
    eps_list = [float(e) for e in eps_grid]
    if any(e <= 0.0 for e in eps_list):
        raise ValueError("eps_grid contains non-positive values; all eps must be > 0")

    # -------------------------------------------------
    # 2-C: Sort and deduplicate eps_grid to ensure consistent processing
    # -------------------------------------------------
    eps_arr = np.array(sorted(set(eps_list)))

    # --------------------------------------------------
    # 2-D: Initialize diagnostics storage and compute dimensions
    # ---------------------------------------------------
    T, K = pivot.shape
    n_zero_cells = int((pivot == 0).sum().sum())
    # baseline bool_mask of exact zeros before smoothing; used to measure contribution
    # of original zeros to CLR variance as eps grows.
    zero_bool_mask_pre = (pivot == 0).values

    # Precompute arrays needed by 'zero_only' and 'multiplicative' strategies (once).
    pivot_vals = pivot.to_numpy(dtype='float64')
    row_total  = pivot_vals.sum(axis=1, keepdims=True)            # (T, 1)
    n_zero_row = zero_bool_mask_pre.sum(axis=1, keepdims=True)    # (T, 1)

    # ------------------------------------------------
    # 2-E: Sweep over eps_grid and compute CLR + diagnostics
    # ------------------------------------------------
    diagnostics: list[dict[str, Any]] = []
    clr_dict: dict[float, pd.DataFrame] = {}
    prev_clr = None

    # ------------------------------------------------
    # 2-F: For each eps, compute CLR and diagnostics.
    #      Zero handling depends on zero_strategy.
    # ------------------------------------------------
    for eps in eps_arr:

        if zero_strategy == 'add_all':
            # Add eps to every cell, then row-normalize (original behavior).
            props = pivot + eps
            props = props.div(props.sum(axis=1), axis=0)

        elif zero_strategy == 'zero_only':
            # Replace only zero cells with eps; leave observed counts untouched.
            props_vals = np.where(zero_bool_mask_pre, eps, pivot_vals)
            props_vals = props_vals / props_vals.sum(axis=1, keepdims=True)
            props = pd.DataFrame(props_vals, index=pivot.index, columns=pivot.columns)

        else:  # 'multiplicative'
            # Replace zeros with eps; shrink non-zeros to preserve each row total.
            with np.errstate(divide='ignore', invalid='ignore'):
                shrink = (row_total - n_zero_row * eps) / row_total
            shrink = np.where(row_total > 0, shrink, 1.0)
            props_vals = np.where(zero_bool_mask_pre, eps, pivot_vals * shrink)

            # Guard: eps too large would push a non-zero count to <= 0.
            if (props_vals <= 0).any():
                raise ValueError(
                    f"zero_strategy='multiplicative' produced non-positive values at "
                    f"eps={eps:g}: eps is too large relative to some row totals. "
                    f"Reduce the eps grid upper bound or use a different strategy."
                )
            props_vals = props_vals / props_vals.sum(axis=1, keepdims=True)
            props = pd.DataFrame(props_vals, index=pivot.index, columns=pivot.columns)

        # ---- CLR (identical for all strategies from here) ----
        logp = np.log(props)
        clr = logp.sub(logp.mean(axis=1), axis=0)
        abs_clr = np.abs(clr.values)

        # CLR variance contribution from original zeros vs total variance
        clr_var_total = float(np.var(abs_clr))
        clr_var_zeros = float(np.var(abs_clr[zero_bool_mask_pre])) if zero_bool_mask_pre.any() else 0.0

        # optional per-eps near-zero diagnostics
        if near_zero_threshold is not None:
            thresh = float(near_zero_threshold)
            near_bool_mask = (props.values < thresh)
            pct_cells_near_zero = float(near_bool_mask.mean()) * 100.0
            clr_var_near_zero = float(np.var(abs_clr[near_bool_mask])) if near_bool_mask.any() else 0.0
        else:
            pct_cells_near_zero = None
            clr_var_near_zero = None

        # Compute rank diagnostics compared to previous eps (None for the first iteration)
        rank_diag = _compute_rank_diagnostics(clr, prev_clr)
        prev_clr = clr

        diagnostics.append(
            {
                "eps": eps,
                "zero_strategy": zero_strategy,
                "max_abs_clr": float(np.nanmax(abs_clr)),
                "mean_max_abs_clr": float(np.nanmean(np.nanmax(abs_clr, axis=1))),
                "pct_rows_large_clr": float((np.nanmax(abs_clr, axis=1) > large_clr_threshold).mean()) * 100.0,
                "rank_stability_spearman": rank_diag["rank_stability_spearman"],
                "rank_stability_kendall": rank_diag["rank_stability_kendall"],
                "rank_unique_ratio": rank_diag["rank_unique_ratio"],
                "rank_entropy": rank_diag["rank_entropy"],
                "zero_contribution_ratio": clr_var_zeros / clr_var_total if clr_var_total > 1e-12 else 0.0,
                "n_zero_cells_pre_smooth": n_zero_cells,
                "T": T,
                "K": K,
            }
        )
        # Store optional near-zero diagnostics
        if near_zero_threshold is not None:
            diagnostics[-1]["pct_cells_near_zero"] = pct_cells_near_zero
            diagnostics[-1]["clr_var_near_zero"] = clr_var_near_zero

        # Store CLR in the dictionary for potential external use (e.g., plotting, further analysis)
        clr_dict[eps] = clr

    # Convert diagnostics list to DataFrame indexed by eps for easier analysis and plotting
    diagnostics_df = pd.DataFrame(diagnostics).set_index("eps")

    # -------------------------------------------------
    # 2-G: Optional automated staged eps selection based on diagnostics
    # -------------------------------------------------
    if auto_select:
        chosen_eps, chosen_reason, chosen_status = select_eps(
            diagnostics_df, kendall_threshold, spearman_threshold
        )
        # -------------------------------------------------
        # Helper Function 2-H: Stage-specific criteria function to evaluate
        # --------------------------------------------------
        def _passes_criteria(row):
            # All stages require pct_rows_large_clr == 0
            if row.get("pct_rows_large_clr", 0) > 0:
                return False

            k = row.get("rank_stability_kendall")
            s = row.get("rank_stability_spearman")
            if pd.isna(k) or pd.isna(s):
                return False

            if chosen_status == "optimal":
                # Stage 1 strict criteria: artifact-free plus stable ranks.
                if k < kendall_threshold or s < spearman_threshold:
                    return False
            elif chosen_status == "near_optimal":
                # Stage 2 soft plateau criteria.
                # IMPORTANT: must match select_eps Stage 2 — compute from artifact-free subset.
                artifact_free = diagnostics_df[diagnostics_df['pct_rows_large_clr'] == 0]
                min_zero = artifact_free['pct_cells_near_zero'].min()
                max_k    = artifact_free['rank_stability_kendall'].max()
                max_s    = artifact_free['rank_stability_spearman'].max()
                if row.get("pct_cells_near_zero", 0) > min_zero + _SLACK_ZERO:
                    return False
                if k < max_k - _SLACK_KENDALL or s < max_s - _SLACK_SPEAR:
                    return False
            else:
                # Stage 3/4: use the artifact-free + stability baseline.
                if k < kendall_threshold or s < spearman_threshold:
                    return False

            return True

        # -------------------------------------------------
        # 2-H: Evaluate neighboring epsilons to determine
        #      if the chosen epsilon is at an edge or isolated
        # --------------------------------------------------
        pass_mask = diagnostics_df.apply(_passes_criteria, axis=1)
        sorted_eps = list(diagnostics_df.index)
        idx = sorted_eps.index(chosen_eps)
        # Identify neighbors and their pass/fail status
        neighbors = {
            "low":  sorted_eps[idx - 1] if idx > 0 else None,
            "high": sorted_eps[idx + 1] if idx < len(sorted_eps) - 1 else None
        }
        # Determine pass/fail/not_in_grid status for neighbors
        stats = {k: ("pass" if (v and pass_mask.loc[v]) else "fail" if v else "not_in_grid")
                 for k, v in neighbors.items()}

        # ------------------------------------------------
        # 2-I: Classify the chosen epsilon's position in
        #      the grid based on neighbor statuses
        # ------------------------------------------------
        if stats["low"] in ("fail", "not_in_grid") and stats["high"] == "pass":
            pos = "lower_edge"
        elif stats["high"] in ("fail", "not_in_grid") and stats["low"] == "pass":
            pos = "upper_edge"
        elif stats["low"] == "pass" and stats["high"] == "pass":
            pos = "interior"
        else:
            pos = "isolated"

        # -------------------------------------------------
        # 2-J: Generate a caveat message if the chosen
        #      epsilon is at an edge
        # --------------------------------------------------
        caveat = None
        if pos == "lower_edge":
            caveat = f"Selection ε={chosen_eps:g} is at the LOWER EDGE of stability. Neighbors below failed."
        elif pos == "isolated":
            caveat = f"Selection ε={chosen_eps:g} is ISOLATED. Nearby grid points are unstable."

        # -------------------------------------------------
        # 2-K: Compile meta information about the selection
        #      for reporting and plotting
        # -------------------------------------------------
        status_map = {
            'optimal': '🟢 Optimal stability', 'near_optimal': '🟡 Near-optimal',
            'elbow': '🔵 Elbow-based', 'fallback': '🟠 Fallback (Caution)'
        }

        # Meta Data Structure:
        meta = {
            "auto_select": True,
            "chosen_eps": chosen_eps,
            "chosen_reason": chosen_reason,
            "chosen_tag": chosen_status,
            "chosen_status": status_map.get(chosen_status, chosen_status),
            "grid_position": pos,
            "boundary_caveat": caveat,
            "grid_spacing_below_log10": float(np.log10(chosen_eps) - np.log10(neighbors["low"])) if neighbors["low"] else None,
            "chosen_row": diagnostics_df.loc[chosen_eps].to_dict(),
            "pass_mask": pass_mask.to_dict(),
            "zero_strategy": zero_strategy,
        }
    else:
        meta = {
            "auto_select": False,
            "chosen_eps": None,
            "boundary_caveat": None,
            "zero_strategy": zero_strategy,
        }

    # -------------------------------------------------
    # 2-L: Finalization (Plotting & Verbose)
    # -------------------------------------------------
    fig = _plot_diagnostics(
        diagnostics_df,
        large_clr_threshold,
        kendall_threshold,
        meta["chosen_eps"],
        zero_strategy=zero_strategy,
    ) if plot else None
    if plot: plt.show()

    if verbose:
        # Summary header showing the strategy and selection result
        print("\n" + "=" * 60)
        print(f"  ε-SWEEP SUMMARY")
        print("-" * 60)
        print(f"  Zero strategy : {zero_strategy}")
        if auto_select and meta.get("chosen_eps") is not None:
            print(f"  Chosen ε      : {meta['chosen_eps']:.10f}")
            print(f"  Status        : {meta.get('chosen_status', 'n/a')}")
            if meta.get("boundary_caveat"):
                print(f"  ⚠ Caveat      : {meta['boundary_caveat']}")
        print("=" * 60 + "\n")

        # Diagnostics table.
        # zero_strategy is dropped from the printed table (constant across rows, shown
        # in the header above) but REMAINS in diagnostics_df for programmatic use.
        cols_to_drop = ["T", "K", "rank_unique_ratio", "n_zero_cells_pre_smooth", "zero_strategy"]
        formatters = {
            'eps': '{:.16f}'.format,
            'max_abs_clr': '{:.6f}'.format,
            'mean_max_abs_clr': '{:.6f}'.format,
        }
        print(
            diagnostics_df.drop(columns=cols_to_drop, errors="ignore")
            .reset_index()
            .to_string(index=False, formatters=formatters)
        )

    return {"diagnostics_df": diagnostics_df, "clr_dict": clr_dict, "meta": meta, "fig": fig}

# ---------------------------------------------------------------------------
# Helper Plot Function 2-A: Automated epsilon selection based on diagnostics
# ---------------------------------------------------------------------------
def _plot_diagnostics(
    data_df: pd.DataFrame,
    large_clr_threshold: float,
    kendall_threshold: float,
    chosen_eps: float | None = None,
    zero_strategy: str = None,
) -> plt.Figure:
    """
    Plot ε-sweep diagnostics across six panels on a log ε-axis.

    Each panel visualizes one diagnostic metric. If a chosen ε is provided,
    it is marked with a dashed vertical line. Legends are placed below each
    panel to avoid overlap.
    """
    # -------------------------------------------------
    # 2-A-1: Theme: Set a clean, minimalist style with a muted color palette
    # -------------------------------------------------
    plt.style.use("default")
    plt.rcParams.update({
        "axes.edgecolor":    "#444444",
        "axes.labelcolor":   "#222222",
        "axes.titlesize":    13,
        "axes.titleweight":  "600",
        "grid.color":        "#E5E5E5",
        "grid.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
    })
    # -------------------------------------------------
    # 2-A-2: Header values: extract T, K, and n_zero_cells 
    #        from the first row of the diagnostics DataFrame
    # -------------------------------------------------
    T  = int(data_df["T"].iloc[0])
    K  = int(data_df["K"].iloc[0])
    nz = int(data_df["n_zero_cells_pre_smooth"].iloc[0])

    # -------------------------------------------------
    # 2-A-3: Panel config: Define colors, labels, and 
    #        titles for each diagnostic panel
    # -------------------------------------------------
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#1B9E77"]

    # Create a 2x3 grid of subplots for the six diagnostics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Define the panels with their corresponding DataFrame column, color, y-label, and title
    panels = [
        (axes[0, 0], "max_abs_clr",            colors[0], "Max |CLR|",      "Sensitivity: Max |CLR|"),
        (axes[0, 1], "pct_rows_large_clr",     colors[1], "% Observations", f"Sparsity Impact: % Rows > {large_clr_threshold}"),
        (axes[0, 2], "rank_stability_kendall", colors[2], "Kendall τ",      "Rank Stability (Kendall)"),
        (axes[1, 0], "rank_unique_ratio",      colors[3], "Unique Ratio",   "Rank Collapse Detection"),
        (axes[1, 1], "pct_cells_near_zero",    colors[4], "% Cells < thr",  "Near-zero Cells"),
        (axes[1, 2], "clr_var_near_zero",      colors[5], "CLR Var",        "CLR Variance from Near-zero Cells"),
    ]
    # ------------------------------------------------
    # 2-A-4: Prepare the label for the chosen ε line if applicable
    # ------------------------------------------------
    chosen_label = fr"chosen $\epsilon$ = {chosen_eps:.6f}" if chosen_eps is not None else None

    # -------------------------------------------------
    # 2-A-5: Draw each panel: plot the metric if present, 
    #        set log x-axis, titles, labels, and mark chosen ε
    # -------------------------------------------------
    for ax, col, color, ylabel, title in panels:

        # Plot metric if present
        if col in data_df.columns:
            ax.plot(
                data_df.index, data_df[col],
                marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.6,
                color=color, linewidth=1.8, label=col,
            )

        # Shared formatting
        ax.set_xscale("log")
        ax.set_title(title, pad=10)
        ax.set_xlabel("ε (Pseudocount)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="both", labelsize=9, colors="#555555")

        # Mark chosen ε with a vertical dashed line if provided
        if chosen_eps is not None:
            ax.axvline(chosen_eps, color="#2E7D32", lw=1.6, ls="--", label=chosen_label)

    # Kendall threshold reference line
    axes[0, 2].axhline(
        kendall_threshold, color="#F58518", lw=1.2, ls=":",
        label=fr"$\tau = {kendall_threshold:.3f}$",
    )

    # ------------------------------------------------
    # 2-A-6: Legends below each panel: collect handles 
    #        and labels, then place a single legend 
    #        below each subplot
    # ------------------------------------------------
    for ax in axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                frameon=False,
                fontsize=9,
                ncol=min(len(handles), 3),
                handlelength=1.8,
                columnspacing=1.4,
                borderaxespad=0,
            )

    # -------------------------------------------------
    # 2-A-7: Title and spacing: Add a main suptitle plus
    #        a smaller subheading line for strategy/eps
    # -------------------------------------------------
    fig.suptitle(
        fr"$\boldsymbol{{\epsilon}}$ - Sweep Diagnostics  |  Data: {T} × {K}  |  "
        fr"Pre-smooth Zeros: {nz:,}",
        fontsize=16, fontweight="600", y=0.995,
    )

    # Subheading: zero strategy (and chosen eps if marked)
    sub_parts = []
    if zero_strategy is not None:
        sub_parts.append(f"Zero strategy: {zero_strategy}")
    if chosen_eps is not None:
        sub_parts.append(fr"chosen $\epsilon$ = {chosen_eps:.6f}")
    if sub_parts:
        fig.text(
            0.5, 0.945,                          # centered, just below the suptitle
            "  |  ".join(sub_parts),
            ha="center", va="top",
            fontsize=12, fontweight="500", color="#555555",
        )

    # subplots_adjust parameters are tuned to balance space for the suptitle, x/y labels, 
    # and legends below each panel while maximizing the plot area for the data.
    fig.subplots_adjust(
        top=0.89,
        bottom=0.14,
        left=0.06,
        right=0.98,
        hspace=0.5,
        wspace=0.15,
    )

    return fig


# ---------------------------------------------------------------------------
# Helper function 2-B: Compute rank‑order stability diagnostics
#                      (sweep_epsilon_grid)
# ---------------------------------------------------------------------------
def _compute_rank_diagnostics(clr: pd.DataFrame, prev_clr: pd.DataFrame | None) -> dict[str, float]:
    """
    Compute rank‑order stability diagnostics for CLR‑transformed data.

    Compares the current CLR matrix to the previous epsilon iteration
    (if provided) and summarizes how stable the rank structure is across
    iterations. Metrics include:

      - Spearman rank correlation between flattened CLR matrices
      - Kendall rank correlation between column‑wise rank orders
      - Rank‑uniqueness ratio (fraction of distinct ranks per column)
      - Rank‑entropy (Shannon entropy of rank distributions)

    These diagnostics help determine whether the epsilon grid search has
    reached a region where rank structure stabilizes.

    Parameters
    ----------
    clr : pd.DataFrame
        Current CLR‑transformed TXK matrix.

    prev_clr : pd.DataFrame or None
        CLR matrix from the previous epsilon value. If None, stability
        metrics are returned as NaN.

    Returns
    -------
    dict[str, float]
        {
            "rank_stability_spearman": Spearman correlation of flattened CLR values,
            "rank_stability_kendall": Kendall correlation of rank orders,
            "rank_unique_ratio": mean fraction of unique ranks per column,
            "rank_entropy": mean entropy of rank distributions
        }
    """
    # -------------------------------------------------
    # A — Compute integer rank matrix for current CLR
    #     Ranks are computed column‑wise using "min" to
    #     ensure deterministic handling of ties.
    # -------------------------------------------------
    ranks = clr.rank(axis=0, method="min").values.astype(int)
    n_rows, n_cols = ranks.shape

    # -------------------------------------------------
    # B — If a previous CLR exists, compute rank‑stability
    #     metrics (Spearman on raw values, Kendall on ranks).
    # -------------------------------------------------
    if prev_clr is not None:
        prev_ranks = prev_clr.rank(axis=0, method="min").values.astype(int)
        rho, _ = spearmanr(clr.values.ravel(), prev_clr.values.ravel())
        tau, _ = kendalltau(ranks.ravel(), prev_ranks.ravel())
        spearman, kendall = float(rho), float(tau)
    else:
        spearman = kendall = float("nan")

    # -------------------------------------------------
    # C — Compute per‑column rank‑uniqueness ratios and
    #     rank‑entropy values to summarize distributional
    #     structure of ranks within each column.
    # -------------------------------------------------
    unique_ratios, entropies = [], []
    for col in range(n_cols):
        counts = np.bincount(ranks[:, col])
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        unique_ratios.append(len(counts) / n_rows)
        entropies.append(float(-np.sum(probs * np.log2(probs + 1e-12))))

    # -------------------------------------------------
    # D — Aggregate diagnostics into a structured dict
    # -------------------------------------------------
    return {
        "rank_stability_spearman": spearman,
        "rank_stability_kendall": kendall,
        "rank_unique_ratio": float(np.mean(unique_ratios)),
        "rank_entropy": float(np.mean(entropies)),
    }


# ---------------------------------------------------------------------------
# 3: Automated epsilon selection based on diagnostics (sweep_epsilon_grid)
# ---------------------------------------------------------------------------
def select_eps(
    df: pd.DataFrame,
    kendall_threshold: float,
    spearman_threshold: float,
    slack_zero: float = _SLACK_ZERO,
    slack_kendall: float = _SLACK_KENDALL,
    slack_spear: float = _SLACK_SPEAR,
    elbow_threshold: float = _ELBOW_THRESHOLD,
    fallback_weights: tuple = _FALLBACK_WEIGHTS,
) -> Tuple[float, str, str]:
    """
    Select the CLR zero-handling epsilon via cascading-fallback criteria.

    Evaluates an epsilon sweep against three diagnostic metrics and returns
    the smallest epsilon that satisfies progressively weaker stage criteria.
    The cascade order is hard plateau -> soft plateau -> elbow -> composite
    fallback; each stage runs only if the previous stage finds no qualifying
    rows.

    Parameters
    ----------
    df : pd.DataFrame
        Sweep diagnostics indexed by candidate epsilon values (sorted
        ascending internally). Required columns:
            'pct_cells_near_zero'      : fraction of cells imputed as ~zero
                                         (e.g., from sweep_epsilon_grid)
            'rank_stability_kendall'   : Kendall tau vs reference ranking
            'rank_stability_spearman'  : Spearman rho vs reference ranking
            'mean_max_abs_clr'         : distortion magnitude (Stage 3 elbow)

    kendall_threshold : float
        Strict Kendall threshold for Stage 1 hard plateau (e.g., 0.98).

    spearman_threshold : float
        Strict Spearman threshold for Stage 1 hard plateau (e.g., 0.999).

    slack_zero : float, default 0.005
        Stage 2 additive tolerance for pct_cells_near_zero. A row qualifies
        if its zero rate is within `slack_zero` of the sweep's minimum.
        The default of 0.5 percentage point reflects the small sampling
        noise of a metric computed on T*K cells.

    slack_kendall : float, default 0.010
        Stage 2 additive tolerance for rank_stability_kendall. A row
        qualifies if its Kendall is within `slack_kendall` of the sweep's
        maximum. The default of 1 percentage point matches the typical
        sampling SE of Kendall tau on T~300 samples (~0.01-0.02), so it
        avoids rejecting rows that are statistically indistinguishable
        from the best.

    slack_spear : float, default 0.005
        Stage 2 additive tolerance for rank_stability_spearman. Spearman
        has tighter sampling SE than Kendall, so a smaller slack is used.

    elbow_threshold : float, default 0.25
        Stage 3 trigger: a row's normalized curvature must exceed this
        fraction of the sweep's max curvature to count as an elbow.
        Lower values detect gentler bends; higher values demand sharper
        knees.

    fallback_weights : tuple of 3 floats, default (1.0, 1.0, 1.0)
        Stage 4 weights for (zero_score, kendall, spearman) in the
        composite score. Increase the first weight to penalize zero
        artifacts more aggressively in the fallback.

    Returns
    -------
    eps : float
        The selected epsilon value (drawn from df.index).

    reason : str
        Human-readable description of which stage triggered.

    status : str
        Plain status tag for downstream display logic:
            'optimal'      - Stage 1 hard plateau
            'near_optimal' - Stage 2 soft plateau
            'elbow'        - Stage 3 elbow detection
            'fallback'     - Stage 4 composite fallback

    Notes
    -----
    Cascading philosophy:
        Stage 1 is strict by design and may not fire on noisy real data.
        That is expected. Stage 2 then catches "good enough" cases using
        metric-specific additive slack calibrated to each metric's
        sampling noise. Stage 3 falls back to elbow detection if metrics
        never plateau. Stage 4 is a least-bad-option choice that flags
        a caution status; recurring Stage 4 hits suggest the sweep itself
        is underpowered or poorly designed.

    Stage 2 slack calibration:
        The three slack parameters are intentionally different to reflect
        the underlying sampling noise of each metric:
            - Zero rate is computed on T*K cells (thousands of values)
              and has tiny SE; small slack (0.005) is conservative.
            - Kendall has larger SE on rank correlations from T~300
              samples; slack of 0.010 lets rows within ~1 SE qualify.
            - Spearman has SE about 60-70% of Kendall's at the same N;
              slack of 0.005 reflects this tighter distribution.
        The prior uniform slack design implicitly demanded ~10x stricter
        agreement on Kendall than the metric's sampling noise warranted.

    Stage 3 curvature:
        Computed via two applications of `np.gradient` on the log10(eps)
        axis. This produces a 5-point stencil approximation of d^2/dx^2,
        handles non-uniform spacing correctly, and gives values at the
        endpoints (unlike a centered finite difference). The log
        transform is essential: a finite difference on the raw eps
        axis would produce spurious large "curvature" values driven by
        the non-uniform spacing of a log-spaced sweep, not by the
        underlying function shape.

    Stage 4 composite:
        A normalized sum across the three metrics rather than a
        lexicographic sort. Avoids the failure mode where a row with
        marginally lower zero rate but much worse rank stability beats
        a row with slightly higher zero rate and excellent stability,
        purely on tiebreaker order.

    Examples
    --------
    >>> from sweep import sweep_epsilon_grid
    >>> sweep = sweep_epsilon_grid(pivot, np.logspace(-6, -1, 30))
    >>> eps, reason, status = select_eps(
    ...     sweep['diagnostics_df'],
    ...     kendall_threshold=0.98,
    ...     spearman_threshold=0.999,
    ... )
    >>> print(f"Selected eps={eps:.2e} via {status}: {reason}")
    Selected eps=3.16e-04 via near_optimal: Soft plateau (near-asymptotic)
    """
    df = df.copy().sort_index()

    # ------------------------------------------------------------
    # Stage 1 - Hard Plateau (strict)
    # ------------------------------------------------------------
    hard_mask = (
        (df['pct_rows_large_clr']       == 0) &   
        (df['rank_stability_kendall']   >= kendall_threshold) &
        (df['rank_stability_spearman']  >= spearman_threshold)
    )

    hard_plateau = df[hard_mask]
    if not hard_plateau.empty:
        eps = float(hard_plateau.index.min())
        return eps, "Hard plateau (artifact-free + rank-stable)", "optimal"

    # ------------------------------------------------------------
    # Stage 2 - Soft Plateau (additive tolerance)
    # ------------------------------------------------------------
    # First, filter to rows where pct_rows_large_clr == 0 (no large CLR artifacts)
    # Then compute plateau thresholds from THOSE rows only. This ensures we're finding
    # the "softest" epsilon among candidates that have already eliminated distortion.
    artifact_free = df[df['pct_rows_large_clr'] == 0]
    
    if not artifact_free.empty:
        min_zero = artifact_free['pct_cells_near_zero'].min()
        max_k    = artifact_free['rank_stability_kendall'].max()
        max_s    = artifact_free['rank_stability_spearman'].max()

        soft_mask = (
            (df['pct_cells_near_zero']      <= min_zero + slack_zero) &
            (df['pct_rows_large_clr']       == 0) &
            (df['rank_stability_kendall']   >= max_k    - slack_kendall) &
            (df['rank_stability_spearman']  >= max_s    - slack_spear)
        )
        # Soft plateau relaxes the rank stability constraints to allow epsilons 
        # that are close to the best observed values (within artifact-free subset),
        # while still requiring no large CLR artifacts. This captures the 
        # "near-asymptotic" region where metrics have essentially plateaued but may 
        # not meet the strict criteria of Stage 1 due to minor sampling noise.
        soft_plateau = df[soft_mask]
        if not soft_plateau.empty:
            eps = float(soft_plateau.index.min())
            return eps, "Soft plateau (near-asymptotic)", "near_optimal"    

    # ------------------------------------------------------------
    # Stage 3 - Elbow Detection (curvature on log10(eps) axis)
    # ------------------------------------------------------------
    if len(df) >= 3:
        y = df['mean_max_abs_clr'].values
        x = np.log10(df.index.values.astype(float))

        # Second derivative via np.gradient (handles non-uniform spacing,
        # gives values at endpoints, reads as standard calculus)
        dy        = np.gradient(y, x)
        curvature = np.gradient(dy, x)

        max_abs_curv = np.max(np.abs(curvature))
        if max_abs_curv > 1e-12:
            curv_norm = curvature / max_abs_curv
            elbow_candidates = np.where(curv_norm > elbow_threshold)[0]
            if len(elbow_candidates) > 0:
                best_idx = elbow_candidates[np.argmax(curv_norm[elbow_candidates])]
                eps = float(df.index.values[best_idx])
                return eps, "Elbow detected in distortion (log-axis curvature)", "elbow"

    # ------------------------------------------------------------
    # Stage 4 - Fallback (composite normalized score)
    # ------------------------------------------------------------
    # Min-max normalize zero-rate to [0, 1] where 1 is best (lowest zeros).
    # Min-max (rather than divide-by-max) prevents amplifying noise when
    # max_zero is small and keeps zero_norm on the same [0, 1] scale as
    # Kendall/Spearman so all three contribute commensurate signal.
    zero_min  = df['pct_cells_near_zero'].min()
    zero_max  = df['pct_cells_near_zero'].max()
    zero_norm = 1.0 - (df['pct_cells_near_zero'] - zero_min) / (zero_max - zero_min + 1e-12)

    # Kendall and Spearman are already in [-1, 1]; treat negative values as 0
    k_norm = df['rank_stability_kendall'].clip(lower=0)
    s_norm = df['rank_stability_spearman'].clip(lower=0)

    w_zero, w_k, w_s = fallback_weights
    composite = w_zero * zero_norm + w_k * k_norm + w_s * s_norm

    eps = float(composite.idxmax())
    return eps, "Fallback (composite score across metrics)", "fallback"


# ---------------------------------------------------------------------------
# 4: CLR transform with flexible zero handling (external utility function)
# ---------------------------------------------------------------------------
def clr_transform(counts: pd.DataFrame,
                  epsilon: float,
                  zero_strategy: str = 'add_all',
                  exclude_features: list = None,
                  validate: bool = True,
                  verbose: bool = False,
                  index: str = _DATE_KEY,
                  column: str = _GROUP_KEY,
                  values: str = _COUNTER_KEY) -> pd.DataFrame:
    """
    Centered Log-Ratio (CLR) transform on an aggregated long panel.

    Expects already-aggregated, gap-filled long data (e.g. fill_results['filled_df']):
    one row per (date, category), no duplicate (date, category) pairs. The function
    optionally excludes categories, reshapes to wide via _pivot(), then applies CLR.

    Each output row sums to ~0. CLR is denominator-dependent: excluding categories
    here (before the pivot) removes them from the geometric-mean denominator,
    producing a true sub-composition.

    Parameters
    ----------
    counts : pd.DataFrame
        Aggregated long panel with columns [index, column, values].
        Typically fill_results['filled_df']. pyarrow-backed dtypes are fine.
    epsilon : float
        Pseudocount for zero handling. Must be > 0.
    zero_strategy : {'add_all', 'zero_only', 'multiplicative'}, default 'add_all'
        'add_all'        : add epsilon to EVERY cell, then row-normalize.
                           Matches sweep_epsilon_grid (`pivot + eps`).
        'zero_only'      : replace only zero cells with epsilon.
        'multiplicative' : replace zeros with epsilon, shrink non-zeros to
                           preserve each row total (Martin-Fernandez 2003).
    exclude_features : list, optional
        Category names (in `column`) to drop BEFORE pivoting — controls the
        sub-composition.
    validate : bool, default True
        Check for NaN and negatives on the wide matrix before CLR.
    verbose : bool, default False
        Print exclusion and CLR diagnostics.
    index, column, values : str
        Long-panel keys; default to _DATE_KEY / _GROUP_KEY / _COUNTER_KEY.

    Returns
    -------
    pd.DataFrame
        CLR coordinates, wide format (months × categories).
    """
    # ----------------------------------------------------------------------
    # 4-1: Validation of scalar args
    # ----------------------------------------------------------------------
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    valid = ('add_all', 'zero_only', 'multiplicative')
    if zero_strategy not in valid:
        raise ValueError(f"zero_strategy must be one of {valid}, got {zero_strategy!r}")

    # Copy to avoid modifying the original DataFrame (especially if it's a view or has pyarrow-backed dtypes)
    work = counts.copy()

    # ----------------------------------------------------------------------
    # 4-2: Exclude categories BEFORE pivot (true sub-composition)
    # ----------------------------------------------------------------------
    if exclude_features:
        existing = set(work[column].unique())
        present  = [c for c in exclude_features if c in existing]
        missing  = [c for c in exclude_features if c not in existing]
        before   = work[column].nunique()
        work     = work[~work[column].isin(exclude_features)]
        after    = work[column].nunique()
        if verbose:
            noun = 'category' if len(present) == 1 else 'categories'
            print(f"Excluded {len(present)} {noun} before pivot: {present}")
            if missing:
                print(f"  (requested but not found in data: {missing})")
            print(f"  categories: {before} -> {after}")

    # ----------------------------------------------------------------------
    # 4-3: Reshape long -> wide via _pivot (safe: no duplicate (date,group) pairs)
    # ----------------------------------------------------------------------
    wide = _pivot(work, index=index, column=column, values=values)

    # ----------------------------------------------------------------------
    # 4-4: To numpy + validation
    # ----------------------------------------------------------------------
    X = wide.to_numpy(dtype='float64', na_value=np.nan)
    if validate:
        if np.isnan(X).any():
            raise ValueError("Wide matrix contains NaN (check _pivot fill).")
        if (X < 0).any():
            raise ValueError("Wide matrix contains negative values.")

    n_zeros = int((X == 0).sum())
    zero_mask = (X == 0)

    # ----------------------------------------------------------------------
    # 4-5: Zero handling
    # ----------------------------------------------------------------------
    if zero_strategy == 'add_all':
        props = X + epsilon
        props = props / props.sum(axis=1, keepdims=True)
        log_X = np.log(props)

    elif zero_strategy == 'zero_only':
        X_pos = np.where(zero_mask, epsilon, X)
        log_X = np.log(X_pos)

    else:  # multiplicative
        row_total  = X.sum(axis=1, keepdims=True)
        n_zero_row = zero_mask.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            shrink = (row_total - n_zero_row * epsilon) / row_total
        shrink = np.where(row_total > 0, shrink, 1.0)
        X_pos = np.where(zero_mask, epsilon, X * shrink)
        if (X_pos <= 0).any():
            raise ValueError(
                "Multiplicative replacement produced non-positive values — "
                "epsilon too large relative to some row totals."
            )
        log_X = np.log(X_pos)

    # ----------------------------------------------------------------------
    # 4-6: CLR core
    # ----------------------------------------------------------------------
    clr_vals = log_X - log_X.mean(axis=1, keepdims=True)

    # ----------------------------------------------------------------------
    # 4-7: Diagnostics
    # ----------------------------------------------------------------------
    if verbose:
        max_abs = np.abs(clr_vals).max()
        row_sum_err = np.abs(clr_vals.sum(axis=1)).max()
        print(f"CLR transform [{zero_strategy}]: {X.shape[0]} × {X.shape[1]}")
        print(f"  epsilon          : {epsilon}")
        print(f"  zeros in data    : {n_zeros} ({n_zeros / X.size:.2%} of cells)")
        print(f"  max |CLR|        : {max_abs:.4f}")
        print(f"  max row-sum error: {row_sum_err:.2e}")

    return pd.DataFrame(clr_vals, index=wide.index, columns=wide.columns)