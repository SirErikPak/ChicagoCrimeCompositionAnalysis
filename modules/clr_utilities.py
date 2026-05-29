import pandas as pd
from typing import Dict
import clr_config as config

# ---------------------------------------------------------------------------------
# 0. Global constants for CLR transformation and pseudocount handling
# ---------------------------------------------------------------------------------
_DATE_KEY    = config.config_agg["_DATE_KEY"]
_COUNTER_KEY = config.config_agg["_COUNTER_KEY"]
_GROUP_KEY   = config.config_agg["_GROUP_KEY"]

# ---------------------------------------------------------------------------------
# AGG_DICT_RESULT is a module‑level cache storing the output of _aggregate_counts().
# 
# Why this exists:
#   • Both the integrity report and the fill‑missing routines depend on the same
#     aggregated count dictionary.
#   • Computing these counts repeatedly is expensive for large datasets.
#   • By caching the result, we avoid redundant work and significantly speed up
#     sequential operations.
#
# How it works:
#   • The first call to _aggregate_counts() populates AGG_DICT_RESULT.
#   • Subsequent calls reuse the cached dictionary unless force_refresh=True is passed.
#   • force_refresh=True recomputes the aggregation when the underlying data changes.
#
# This pattern ensures correctness while providing efficient repeated access.
# ---------------------------------------------------------------------------------
AGG_DICT_RESULT = None

# ---------------------------------------------------------------------------------
# Helper function to aggregate counts safely (removing duplicates)
# ---------------------------------------------------------------------------------
def _aggregate_counts(
    data_df: pd.DataFrame,
    force_refresh: bool = None,
    group_col: str = _GROUP_KEY,
    date_col: str = _DATE_KEY,
    counter_col: str = _COUNTER_KEY,
) -> Dict:
    """
    Aggregate raw records into monthly group-level counts, with caching.

    This function groups the input DataFrame by `(group_col, date_col)` and
    computes the number of occurrences for each group–month combination.
    Dates are parsed as YYYYMM monthly timestamps, invalid dates are removed,
    and the date column is normalized to a proper datetime dtype. The group
    column is coerced to a categorical type for consistency and efficiency.

    A module‑level cache is used to avo_aggregate_countsid recomputing the aggregation on
    repeated calls. The cached result is returned unless `force_refresh=True`
    or the cache is empty.

    Returns
    -------
    Dict
        A dictionary containing:
        • "data"       : aggregated DataFrame with counts per group–month
        • "start_date" : earliest valid month in 'YYYY‑MM' format
        • "end_date"   : latest valid month in 'YYYY‑MM' format
    """   
    # MUST be declared: ensures assignments update the shared module-level 
    # cache instead of shadowing it locally
    global AGG_DICT_RESULT

    # -------------------------------------------------
    # A. Fast path: reuse previously computed aggregation
    #    Only recompute when force_refresh=True or cache is empty
    # -------------------------------------------------
    if AGG_DICT_RESULT is not None and not force_refresh:
        return AGG_DICT_RESULT

    # -------------------------------------------------
    # B. Aggregate counts:
    #    Group by (group_col, date_col) and compute the number of rows in each group.
    # -------------------------------------------------
    data = (
        data_df.groupby([group_col, date_col], sort=True, observed=False)
        .size()                     # count rows per group/date combination
        .rename(counter_col)        # rename the count column
        .reset_index()              # convert MultiIndex → flat DataFrame
    )

    # -------------------------------------------------
    # C. Convert date column to datetime (safe now that grouping is done)
    #    Using '%Y%m' ensures YYYYMM strings become proper monthly timestamps.
    # -------------------------------------------------
    data[date_col] = pd.to_datetime(data[date_col], format='%Y%m', errors='coerce')

    # -------------------------------------------------
    # D. Remove any rows with invalid dates
    #    These arise when pd.to_datetime() returned NaT
    # -------------------------------------------------
    data = data.dropna(subset=[date_col])

    # -------------------------------------------------
    # E. Guarantee datetime dtype.
    #    Even after cleaning, dtype may still be object; enforce datetime
    #    so sorting, merging, and resampling behave correctly.
    # -------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])

    # -------------------------------------------------
    # F. Compute the valid date range.
    #    After dropping invalid rows and enforcing datetime dtype,
    #    extract the earliest and latest timestamps and format them
    #    as YYYY‑MM for consistent monthly summaries.
    # -------------------------------------------------
    start_date = data[date_col].min().strftime('%Y-%m')
    end_date   = data[date_col].max().strftime('%Y-%m')

    # -------------------------------------------------
    # G. Normalize dtypes:
    #    Convert group column to a canonical type: string → category.
    #    (Ensures consistent grouping behavior and reduces memory usage.)
    # -------------------------------------------------
    data[group_col] = data[group_col].astype("string").astype("category")

    # -------------------------------------------------
    # H. Store results in the global cache
    #    (makes aggregated data + date range available for reuse)
    # -------------------------------------------------
    AGG_DICT_RESULT = {
        "data": data,
        "start_date": start_date,
        "end_date": end_date
    }

    return AGG_DICT_RESULT

# -------------------------------------------------------------------
# 1. Main function for integrity report (no filling, just analysis)
# -------------------------------------------------------------------
def run_integrity_report(
    data_df       : pd.DataFrame,
    force_refresh : bool = True,
    group_col     : str  = _GROUP_KEY,
    date_col      : str  = _DATE_KEY,
    freq          : str  = "MS"
) -> Dict:
    """
    Generate a full panel‑integrity and sparsity diagnostic for crime count data.

    This function evaluates the completeness and structural soundness of a
    monthly crime‑count panel. It aggregates raw records, constructs the full
    expected (group X month) grid, identifies true missing panel entries,
    computes per‑group coverage ratios, summarizes completeness statistics,
    and flags groups with incomplete temporal coverage.

    The report distinguishes between:
      • true panel gaps - months where a (group, month) row is entirely absent
      • structural zeros - months present in the data but with zero incidents
    Only true gaps are treated as missingness relevant for CLR preprocessing.

    Parameters
    ----------
    data_df : pd.DataFrame
        Raw crime incident records containing at least `group_col` and `date_col`.
    force_refresh : bool, default True
        If True, recompute the monthly aggregation even when a cached result exists.
    group_col : str
        Column identifying the crime category.
    date_col : str
        Column identifying the observation month (YYYYMM or datetime‑like).
    freq : str, default "MS"
        Pandas frequency string defining the monthly grid (e.g., "MS" = month start).

    Returns
    -------
    dict
        A structured dictionary containing:
          - date_range       : (start, end) bounds of the panel in YYYY‑MM format
          - missing          : DataFrame of true missing (group, month) rows
          - missing_by_group : Series of missing‑month counts per group
          - coverage_ratio   : Series of observed / expected months per group
          - duplicates       : Number of duplicate (group, month) rows
          - expected_rows    : Total rows in the complete panel grid
          - actual_rows      : Total rows present in the aggregated data
          - missing_rows     : Number of missing panel entries
          - completeness     : Overall panel completeness ratio in [0, 1]

    Notes
    -----
    This function performs diagnostics only; it does not fill, impute, or
    modify the underlying data. It is intended to be run prior to CLR
    transformation to assess sparsity, detect structural gaps, and evaluate
    pseudocount sensitivity.

    To force regeneration of the aggregated monthly counts, call with:
        run_integrity_report(data_df, force_refresh=True)
    """

    # -------------------------------------------------
    # Step 1-A: Aggregate raw records into monthly counts
    #    _aggregate_counts returns:
    #       - 'data'       : cleaned + aggregated DataFrame
    #       - 'start_date' : earliest YYYY‑MM in the dataset
    #       - 'end_date'   : latest YYYY‑MM in the dataset
    #    (Uses cached result unless force_refresh=True)
    # -------------------------------------------------
    data_dict = _aggregate_counts(data_df, force_refresh=force_refresh)
    data      = data_dict["data"]
    start     = data_dict["start_date"]
    end       = data_dict["end_date"]

    # -------------------------------------------------
    # Step 1-B: Construct the complete panel structure.
    #    full_months: continuous monthly range
    #    groups: all valid group identifiers
    #    full_index: full grid (group X month) ensuring no missing combinations
    # -------------------------------------------------
    full_months = pd.date_range(start=start, end=end, freq=freq)
    groups      = data[group_col].dropna().unique()

    full_index  = pd.MultiIndex.from_product(
        [groups, full_months],
        names=[group_col, date_col]
    )

    # -------------------------------------------------
    # Step 1-C: Detect gaps in the panel.
    #    existing_index: observed (group, date) combinations
    #    missing_index: all expected combinations not present in the data
    #    missing_df: tidy DataFrame of missing rows for downstream filling
    # -------------------------------------------------
    existing_index = pd.MultiIndex.from_frame(data[[group_col, date_col]])
    missing_index  = full_index.difference(existing_index)
    missing_df     = missing_index.to_frame(index=False)

    # -------------------------------------------------
    # Step 1-D: Summarize missingness by group.
    #    Produces a Series: group → count of missing (group, month) entries.
    #    Sorted so the worst offenders appear first.
    # -------------------------------------------------
    missing_by_group = (
        missing_df
        .groupby(group_col, observed=False)
        .size()
        .sort_values(ascending=False)
    )
    missing_by_group = missing_by_group[missing_by_group > 0]

    # -------------------------------------------------
    # Step 1-E: Per‑group coverage ratio
    #    Fraction of expected months actually observed for each group.
    #    coverage < 1.0 indicates at least one missing month.
    # -------------------------------------------------
    coverage_ratio = (
        data
        .groupby(group_col, observed=False)[date_col]
        .nunique()              # number of months actually present per group
        / len(full_months)      # divide by total expected months
    ).sort_values()

    # -------------------------------------------------
    # Step 1-F: Summary statistics
    #    expected     :total (group X month) combinations in the full panel
    #    actual       : number of observed rows in the dataset
    #    missing      : number of missing (group, month) entries
    #    duplicates   : count of duplicate group–month rows in the observed data
    #    completeness : overall panel completeness ratio (1 = fully complete)
    # -------------------------------------------------
    expected     = len(full_index)
    actual       = len(data)
    missing      = len(missing_df)
    duplicates   = int(data.duplicated([group_col, date_col]).sum())
    completeness = 1 - (missing / expected)

    # -------------------------------------------------
    # Step 1-G: Print integrity summary (Modernized)
    # -------------------------------------------------
    header_w = 50
    border = "-" * header_w
    sub_border = "-" * header_w

    print(f"\n{border}")
    print(f" 📑  PANEL INTEGRITY & SPARSITY DIAGNOSTIC")
    print(f"{border}")
    
    # Core KPIs
    print(f"  • Date Horizon   : {start} to {end}")
    print(f"  • Total Months   : {len(full_months)}")
    print(f"  • Feature Space  : {len(groups):,} crime categories")
    print(f"  • Total Panel    : {expected:,} (expected combinations)")
    print(f"  • Actual Density : {actual:,} (observed rows)")
    
    # Performance/Health Score
    health_color = "✅" if completeness > 0.99 else "⚠️" if completeness > 0.90 else "🚨"
    print(f"{sub_border}")
    print(f"  {health_color} Completeness  : {completeness:.2%}")
    print(f"  🔍 Missing Gaps  : {missing:,} rows")
    print(f"  👯 Duplicates    : {duplicates:,} rows")
    print(f"{sub_border}")

    # -------------------------------------------------
    # Step 1-H: True Panel Gaps (Lined Up)
    # -------------------------------------------------
    if missing > 0:
        print(f"\n 🔥  CRITICAL: TRUE MISSING GAPS")
        print("    (These months are entirely absent from the raw data)")
        for group, count in missing_by_group.items():
            # <46: Left-align text in a 46-character block
            # >4:  Right-align numbers in a 4-character block
            print(f"    ↳ {group:<46} | {count:>4} months missing")

    # -------------------------------------------------
    # Step 1-I: Sparse / Risky Groups (Lined Up)
    # -------------------------------------------------
    print(f"\n 🧪  METHODOLOGICAL RISK: SPARSE GROUPS")
    print("    (Coverage < 100% | High sensitivity to pseudocounts)")

    sparse = coverage_ratio[coverage_ratio < 1.0]

    if not sparse.empty:
        for group, ratio in sparse.items():
            # >7.1%: Right-align percentage to keep decimals aligned
            print(f"    ↳ {group:<46} | {ratio:>7.1%} coverage")
    else:
        print("    ✨ None - Feature set is temporally dense")
    return {
        "date_range"      : (start, end),
        "missing"         : missing_df,
        "missing_by_group": missing_by_group,
        "coverage_ratio"  : coverage_ratio,
        "duplicates"      : duplicates,
        "expected_rows"   : expected,
        "actual_rows"     : actual,
        "missing_rows"    : missing,
        "completeness"    : completeness,
    }

# -------------------------------------------------------------------
# 2. Main function to fill missing (group, month) entries in the panel
# -------------------------------------------------------------------
def fill_missing(
    data_df: pd.DataFrame,
    force_refresh: bool = True,
    group_col: str = _GROUP_KEY,
    date_col: str = _DATE_KEY,
    value_col: str = _COUNTER_KEY,
    freq: str = "MS",
    fill_value: float = 0,
    verbose: bool = True
) -> dict:
    """
    Construct a complete group–date panel and fill missing observations.

    This function ensures that every group has an entry for every period
    between the observed start and end dates. Missing (group, date) pairs
    are introduced via reindexing and filled with `fill_value`. Two audit
    flags are added:

        - was_missing        — True if the (group, date) pair did not exist
        - is_zero_after_fill — True if the filled value equals zero

    These flags allow downstream models to distinguish structural zeros
    from imputed gaps.

    Parameters
    ----------
    data_df : pd.DataFrame
        Input dataset containing group, date, and count columns.

    group_col : str
        Column identifying groups (e.g., precinct, district).

    date_col : str
        Column containing datetime-like values.

    value_col : str
        Column containing the numeric count to be filled.

    freq : str, default "MS"
        Frequency for constructing the complete date range.

    fill_value : float, default 0
        Value used to fill missing (group, date) entries.

    verbose : bool, default True
        If True, prints a diagnostic summary.

    Returns
    -------
    dict
        {
            "date_range": (start_date, end_date),
            "filled_df":  panel_with_flags,
            "summary": {
                "total_groups": int,
                "total_periods": int,
                "total_rows": int,
                "filled_missing": int,
                "fill_value": float
            }
        }
    """
    # -------------------------------------------------
    # 2-1: Aggregate and extract global date bounds
    # -------------------------------------------------
    data_dict = _aggregate_counts(data_df, force_refresh=force_refresh)
    data      = data_dict["data"]
    start     = data_dict["start_date"]
    end       = data_dict["end_date"]

    # -------------------------------------------------
    # 2-2: Build complete group × date index
    # -------------------------------------------------
    unique_groups = data[group_col].unique()
    full_range    = pd.date_range(start=start, end=end, freq=freq, name=date_col)

    full_index = pd.MultiIndex.from_product(
        [unique_groups, full_range],
        names=[group_col, date_col]
    )

    # -------------------------------------------------
    # 2-3: Reindex and fill missing entries
    # -------------------------------------------------
    panel = (
        data.set_index([group_col, date_col])
            .reindex(full_index)
            .sort_index()
    )

    # Audit flags
    panel["was_missing"] = panel[value_col].isna()
    panel[value_col]     = panel[value_col].fillna(fill_value)
    panel["is_zero_after_fill"] = panel[value_col].eq(0)

    # -------------------------------------------------
    # 2-4: Optional diagnostics
    # -------------------------------------------------
    if verbose:
        label_w = 30
        border  = "=" * 66
        line    = "-" * 66

        print(f"\n{border}")
        print(f" 🧩  CRIME DATA FILLING & PANEL ALIGNMENT")
        print(f"{border}")

        print(f"  • {'Date Range':<{label_w}} : {start} to {end}")
        print(f"  • {'Total Groups':<{label_w}} : {len(unique_groups):,}")
        print(f"  • {'Total Periods':<{label_w}} : {len(full_range):,}")

        print(line)
        print(f"  📦 {'Final Panel Size':<{label_w}} : {len(panel):,} rows")
        print(f"  🩹 {'Filled Gaps':<{label_w}} : {int(panel['was_missing'].sum()):,} rows")
        print(f"  🔢 {'Fill Value Used':<{label_w}} : {fill_value}")

        print(line)
        print(f" ✅ Audit flags: 'was_missing', 'is_zero_after_fill'")
        print(f"{border}\n")

    # -------------------------------------------------
    # 2-5: Return structured output
    # -------------------------------------------------
    return {
        "date_range": (start, end),
        "filled_df": panel.reset_index(),
        "summary": {
            "total_groups": len(unique_groups),
            "total_periods": len(full_range),
            "total_rows": len(panel),
            "filled_missing": int(panel["was_missing"].sum()),
            "fill_value": fill_value,
        }
    }


# -------------------------------------------------------------------
# 3. Validation function to check crime data integrity after filling
# -------------------------------------------------------------------
def validate_crime_data(
    data: pd.DataFrame,
    zero_rate_threshold: float = 0.70,
    group_col: str = _GROUP_KEY,
    date_col: str = _DATE_KEY,
    value_col: str = _COUNTER_KEY
) -> None:
    """
    Perform structural integrity checks on a crime panel dataset.

    This validator inspects:
        • Date dtype and global sorting
        • Temporal continuity within each group
        • Synchronization of period counts across groups
        • Value integrity (no negatives, no NaNs)
        • Duplicate (group, date) records
        • Presence of audit columns from panel construction
        • Overall and group-level sparsity

    Output is formatted as a vertically aligned diagnostic report
    with PASS / FAIL / WARN indicators.

    Parameters
    ----------
    data : pd.DataFrame
        Crime panel dataset.

    zero_rate_threshold : float, default 0.70
        Groups exceeding this zero-rate threshold are flagged.

    group_col : str
        Group identifier column.

    date_col : str
        Date column (must be datetime-like).

    value_col : str
        Numeric count column.

    Returns
    -------
    None
        Prints a formatted validation report.
    """
    # -------------------------------------------------
    # Formatting configuration
    # -------------------------------------------------
    header_w = 80
    label_w  = 26
    list_w   = 50

    border = "=" * header_w
    line   = "-" * header_w

    print(f"\n{border}")
    print(f"{'🛡️  CRIME DATA STRUCTURAL INTEGRITY CHECK':^{header_w}}")
    print(f"{border}")

    # Helper for aligned logging
    def log(icon, label, status, detail):
        print(f"  {icon} {label:<{label_w}} : {status:<4} | {detail}")

    # -------------------------------------------------
    # 3-A: Date dtype
    # -------------------------------------------------
    is_dt = pd.api.types.is_datetime64_any_dtype(data[date_col])
    log("✅" if is_dt else "❌", "Date Dtype",
        "PASS" if is_dt else "FAIL",
        "datetime64 required")

    # -------------------------------------------------
    # 3-B: Global sorting
    # -------------------------------------------------
    expected = data.sort_values([group_col, date_col]).reset_index(drop=True)
    is_sorted = expected.equals(data.reset_index(drop=True))
    log("✅" if is_sorted else "❌", "Global Sort",
        "PASS" if is_sorted else "FAIL",
        "Group-Date ordering")

    # -------------------------------------------------
    # 3-C: Temporal continuity within groups
    # -------------------------------------------------
    bad_groups = [
        g for g, sub in data.groupby(group_col, observed=False)
        if not sub[date_col].is_monotonic_increasing
    ]
    log("✅" if not bad_groups else "❌", "Temporal Continuity",
        "PASS" if not bad_groups else "FAIL",
        "Linear timeline" if not bad_groups else f"{len(bad_groups)} unordered groups")

    # -------------------------------------------------
    # 3-D: Period synchronization
    # -------------------------------------------------
    periods = data.groupby(group_col, observed=False)[date_col].nunique()
    sync = periods.nunique() == 1
    log("✅" if sync else "❌", "Period Sync",
        "PASS" if sync else "FAIL",
        f"All groups have {periods.iloc[0]} months" if sync else "Inconsistent months")

    # -------------------------------------------------
    # 3-E: Value integrity
    # -------------------------------------------------
    n_neg = int((data[value_col] < 0).sum())
    n_nan = int(data[value_col].isna().sum())

    log("✅" if n_neg == 0 else "❌", "Value Range",
        "PASS" if n_neg == 0 else "FAIL",
        f"{n_neg} negative values")

    log("✅" if n_nan == 0 else "❌", "NaN Cleanup",
        "PASS" if n_nan == 0 else "FAIL",
        f"{n_nan} missing entries")

    # -------------------------------------------------
    # 3-F: Duplicate records
    # -------------------------------------------------
    n_dupes = int(data.duplicated([group_col, date_col]).sum())
    log("✅" if n_dupes == 0 else "❌", "Record Uniqueness",
        "PASS" if n_dupes == 0 else "FAIL",
        f"{n_dupes} duplicates")

    # -------------------------------------------------
    # 3-G: Audit columns
    # -------------------------------------------------
    for col in ["was_missing", "is_zero_after_fill"]:
        present = col in data.columns
        log("✅" if present else "⚠️", f"Audit: {col}",
            "PASS" if present else "WARN",
            "Traceability")

    # -------------------------------------------------
    # 3-H: Overall sparsity
    # -------------------------------------------------
    print(line)
    zero_rate = (data[value_col] == 0).mean()
    log("🔍", "Overall Zero Rate", "INFO", f"{zero_rate:.2%}")

    # -------------------------------------------------
    # 3-I: Group-level sparsity
    # -------------------------------------------------
    print(f"\n 🧪  SPARSITY EXPOSURE BY CATEGORY")
    print(f"    (Threshold: > {zero_rate_threshold:.0%} zeros)")

    group_rates = (
        data.groupby(group_col, observed=False)[value_col]
            .apply(lambda x: (x == 0).mean())
            .sort_values(ascending=False)
    )

    flagged = group_rates[group_rates > zero_rate_threshold]

    if not flagged.empty:
        for group, rate in flagged.items():
            print(f"    ↳ {group:<{list_w}} | {rate:>7.1%} zero rate")
    else:
        print("    ✨ No groups exceed the sparsity threshold.")

    print(f"{border}\n")


# ------------------------------------------------------------------------------------
# 4. Utility function to print a local transition window around the chosen epsilon
# ------------------------------------------------------------------------------------
def get_epsilon_transition(sweep_result, window_prior=5, window_past=10):
    """
    Extracts the transition window and returns a Styled DataFrame
    filtered for specific metrics.
    """
    # Extract data sources
    df_diag = sweep_result['diagnostics_df']
    meta = sweep_result['meta']
    mask = meta['pass_mask']
    chosen_eps = meta['chosen_eps']

    # Identify the slice indices
    all_eps = df_diag.index.tolist()
    winner_idx = all_eps.index(chosen_eps)
    # Prior epsilon (if exists) is the one immediately before the chosen epsilon in the list
    prior_eps = all_eps[winner_idx - 1] if winner_idx > 0 else None

    
    start_idx = max(0, winner_idx - window_prior)
    end_idx = min(len(all_eps), winner_idx + window_past + 1)
    transition_eps = all_eps[start_idx:end_idx]

    # Build the DataFrame and remove unwanted columns
    df = df_diag.loc[transition_eps].copy()
    
    cols_to_drop = ['T', 'K', 'pct_cells_near_zero', 'clr_var_near_zero',  'n_zero_cells_pre_smooth']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df = df.reset_index()

    # Insert metadata columns
    df.insert(0, 'index', [all_eps.index(e) for e in transition_eps])
    df.insert(1, 'status', [("✅ TRUE" if mask[e] else "❌ FALSE") for e in transition_eps])
    df.insert(2, 'is_chosen', [("⭐ [CHOSEN]" if e == chosen_eps else "") for e in transition_eps])
    
    # Define formatters
    float_cols = df.select_dtypes(include=['float64']).columns
    format_map = {col: "{:.6f}" for col in float_cols}
    format_map['eps'] = "{:.10f}" 

    # Return styled result & chosen eps
    return df.style.format(format_map).hide(axis="index"), chosen_eps, prior_eps


# ------------------------------------------------------------------------------------
# 5: Main function to audit CLR‑era matrices and reconstructed wide‑format dataset
# ------------------------------------------------------------------------------------
def audit_clr_and_wide(clr_era: dict, fill_results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a combined structural audit of CLR‑era matrices and the reconstructed
    wide‑format dataset.

    This function:
      • Rebuilds the full CLR dataframe by concatenating the three era‑specific
        CLR matrices (Pre‑COVID, COVID, Post‑COVID).
      • Reconstructs the raw wide‑format dataset from the filled long‑format data.
      • Prints a structured audit report summarizing:
            - Matrix dimensionality
            - Timeline boundaries
            - Feature schema alignment
            - Column‑set discrepancies (if any)
      • Returns both reconstructed dataframes for downstream analysis.

    Parameters
    ----------
    clr_era : dict
        Dictionary containing three CLR‑transformed dataframes keyed by:
        'Pre-COVID', 'COVID', and 'Post-COVID'.
    fill_results : dict
        Must contain 'filled_df', the long‑format filled dataset used to rebuild
        the wide‑format matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        clr_df  : Full reconstructed CLR dataframe (concatenated across eras)
        raw_wide: Reconstructed wide‑format dataframe (pivoted from filled_df)
    """

    # ---------------------------------------------------------
    # 5-1: Concatenate the three CLR era segments into a single timeline
    # ---------------------------------------------------------
    clr_df = pd.concat([
        clr_era['Pre-COVID'],
        clr_era['COVID'],
        clr_era['Post-COVID']
    ])

    # Extract the long-format filled dataset
    filled_long = fill_results['filled_df']

    # Reconstruct wide-format dataset using a safe pivot fallback
    raw_wide = (
        filled_long.pivot(index='date', columns='category', values='value')
        if 'date' in filled_long.columns
        else clr_df.copy()
    )

    # ---------------------------------------------------------
    # 5-2: Set Up Formatting Parameters
    # ---------------------------------------------------------
    width = 80
    row_fmt = "  {:<38} | {:>26}"
    divider = "─" * width

    # ---------------------------------------------------------
    # 5-3: Header block for the audit report
    # ---------------------------------------------------------
    print("═" * width)
    print(f"{'📊 SYSTEM INTEGRITY AUDIT REPORT':^{width}}")
    print(f"{'Structural Dimensionality & Alignment Verification':^{width}}")
    print("═" * width)
    print(row_fmt.format("Inspection Target Node", "Properties / Shapes"))
    print(divider)

    # ---------------------------------------------------------
    # 5-1-A: Matrix dimensionality section
    # ---------------------------------------------------------
    print(f" [ DATA MATRIX DIMENSIONALITY ]")
    print(row_fmt.format(
        "  ├── Reconstructed CLR Space",
        f"{clr_df.shape[0]:,} Rows × {clr_df.shape[1]:,} Cols"
    ))
    print(row_fmt.format(
        "  └── Reference Filled Long Space",
        f"{filled_long.shape[0]:,} Rows × {filled_long.shape[1]:,} Cols"
    ))
    print(divider)

    # ---------------------------------------------------------
    # 5-1-B: Timeline boundary section
    # ---------------------------------------------------------
    print(f" [ TEMPORAL TIMELINE BOUNDARIES ]")

    # Extract first and last three timestamps for inspection
    formatted_start = clr_df.index[:3].astype(str).str[:7].tolist()
    formatted_end = clr_df.index[-3:].astype(str).str[:7].tolist()

    print(row_fmt.format(
        "  ├── Timeline Head (Start Window)",
        f"{formatted_start} …"
    ))
    print(row_fmt.format(
        "  └── Timeline Tail (End Window)",
        f"… {formatted_end}"
    ))
    print(divider)

    # ---------------------------------------------------------
    # 5-1-C: Schema comparison section
    # ---------------------------------------------------------
    print(f" [ FEATURE SCHEMA COMPARISON ]")

    print(row_fmt.format(
        "  ├── CLR Feature Dimensions",
        f"{len(clr_df.columns)} Columns"
    ))
    print(row_fmt.format(
        "  ├── RAW‑WIDE Feature Dimensions",
        f"{len(raw_wide.columns)} Columns"
    ))

    # Check if the two column sets match exactly
    sets_identical = set(clr_df.columns) == set(raw_wide.columns)

    print(row_fmt.format(
        "  └── Column Feature Sets Identical",
        str(sets_identical).upper()
    ))
    print(divider)

    # ---------------------------------------------------------
    # 5-1-D: Column discrepancy section
    # ---------------------------------------------------------
    print(f" [ INTERSECTION DISCREPANCY ANALYSIS ]")

    # Identify columns exclusive to each dataset
    only_clr = set(clr_df.columns) - set(raw_wide.columns)
    only_raw = set(raw_wide.columns) - set(clr_df.columns)

    # Report alignment or mismatches
    if not only_clr and not only_raw:
        print(row_fmt.format("  └── Schema Alignment State", "✅ PERFECT ALIGNMENT"))
    else:
        if only_clr:
            print(f"  ├── ❗ Exclusive to CLR: {sorted(only_clr)}")
        if only_raw:
            print(f"  └── ❗ Exclusive to RAW-WIDE: {sorted(only_raw)}")

    print("═" * width)

    # ---------------------------------------------------------
    # Return both reconstructed dataframes
    # ---------------------------------------------------------
    return clr_df, raw_wide
