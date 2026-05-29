
# -------------------------------------------------------------------------------------
# 1: Era Configuration Mapping
# -------------------------------------------------------------------------------------
# Era Configuration settings for defining eras and parameters for 
# crime data analysis and visualization.
# Map era labels to keys in the input dictionaries
era_map = {
    'pre_covid' : 'Pre-COVID',
    'covid'     : 'COVID',
    'post_covid': 'Post-COVID'
}

# -------------------------------------------------------------------------------------
# 2: Era Boundaries
# -------------------------------------------------------------------------------------
# Era boundaries for defining the three eras based on time periods, which are 
# used to categorize crime data into distinct phases for analysis and visualization.
era_boundaries = {
    # end time
    'Pre-COVID'  : '2020-02',
    'COVID'      : '2022-12',
    'Post-COVID' : '2023-01' 
}

# -------------------------------------------------------------------------------------
# 3: Era Date Ranges and Colors
# -------------------------------------------------------------------------------------
# Era configuration for plotting, including start and end dates for each era 
# and associated colors for visualization.
era_config = {
    'Pre-COVID' : ('2001-01-01', '2020-03-01', 'blue'),
    'COVID'     : ('2020-03-01', '2023-01-01', 'red'),
    'Post-COVID': ('2023-01-01', '2025-12-01', 'green'),
}
# -------------------------------------------------------------------------------------
# 4: Aggregation & Pivot Keys (Dataset Specific)
# -------------------------------------------------------------------------------------
"""
_DATE_KEY: The key for the date column in the dataset, used for grouping and analysis.
_COUNTER_KEY: The key for the count of crimes, used for aggregating crime data.
_GROUP_KEY: The key for the crime type or category, used for grouping crime data.
_EPS_GRID: A list of small values representing the Jeffreys prior grid for smoothing in 
    statistical analysis, which helps to prevent overfitting and provides a more robust 
    estimation of probabilities in the presence of sparse data.
"""
# Configuration for the crime data aggregation and analysis, including keys for date, 
# crime count, and crime type, as well as a predefined grid of epsilon values for smoothing 
# in the CLR method. (clr_utilities.py) 
config_agg = {
    "_DATE_KEY"     : "year_month",
    "_COUNTER_KEY"  : "crime_count",
    "_GROUP_KEY"    : "fbi_code_desc",
}

"""
Configuration parameters for CLR epsilon grid search and selection.
Adjust these settings to tune the sensitivity and stability of zero-handling.
"""

# ----------------------------------------------------------------------------
# 5:. Generic Grid Generation Parameters
# ----------------------------------------------------------------------------
config_grid = {
    # 1. Increase the Density (The "Resolution" knob)
    # 24 points per decade means a point roughly every 10% change.
    "_N_PER_DECADE": 24, 
    
    # 2. Keep the Anchors clean and standard
    "_INCLUDE_FIXED": (1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0),
    
    # 3. Tighten the step (The "Precision" knob)
    # Allows the grid to keep points that are 5% apart.
    "_MIN_STEP": 0.05, 

    # Keep these the same for generic scaling
    "_Q_LOW": 0.05,
    "_MIN_MULTIPLIER_CANDIDATES": (0.1, 0.25, 0.5, 1.0, 2.0),
    "_FLOOR": 1e-12,
}


# ----------------------------------------------------------------------------
# 6: Stability Sweep & Selection Thresholds
# ----------------------------------------------------------------------------
config_sweep = {
    # C1: Magnitude Constraint (Max |CLR| allowed before flagging distortion)
    "large_clr_threshold": 10.0,
    
    # C2: Stability Constraints (Minimum rank correlation required)
    "kendall_threshold": 0.99,
    "spearman_threshold": 0.999,
    
    # C3: Artifact Detection (Probability threshold for "near-zero" cells)
    "near_zero_threshold": 1e-6,
    
    # Stage 2 "Soft Plateau" slacks (Additive tolerance for finding plateaus)
    "slack_zero": 0.005,
    "slack_kendall": 0.010,
    "slack_spear": 0.005,
    
    # Stage 3 Elbow Detection sensitivity
    "elbow_threshold": 0.25,
    
    # Stage 4 Fallback weights (Zero artifacts, Kendall stability, Spearman stability)
    "fallback_weights": (1.0, 1.0, 1.0),
}