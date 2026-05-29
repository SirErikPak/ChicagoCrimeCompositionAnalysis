import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.decomposition import PCA

# Define valid methods for sign normalization
VALID_SIGN_METHODS = ('anchor', 'max_abs', 'mean', 'sum')

# ---------------------------------------------------------------------------------------------
# 1. Function to apply stable sign normalization to PCA loadings and scores
# ---------------------------------------------------------------------------------------------
def stable_sign_normalize(loadings: np.ndarray, coords: np.ndarray, feature_names: list, 
                          method: str='anchor', anchor_observation: int=0, top_n: int=3) -> Dict:
    """
    Apply a stable sign convention to PCA loadings and scores to ensure
    comparability across runs, eras, or feature-set changes.

    For each principal component, the sign of both the loading vector and
    the corresponding score column is flipped (if necessary) according to
    the chosen method.

    Parameters
    ----------
    loadings : array-like
        Shape (K_components, K_features). PCA loadings, e.g., pca.components_.
    coords : array-like
        Shape (T_observations, K_components). PCA scores.
    feature_names : sequence of str
        Column names of the original feature matrix, length K_features.
        Used only to build the interpretation labels in `pc_positive_top`.
    method : {'anchor', 'max_abs', 'mean', 'sum'}, default 'anchor'
        Sign-convention strategy. See Notes for stability trade-offs.
    anchor_observation : int, default 0
        Row index used when method='anchor'. Ignored otherwise.
    top_n : int, default 3
        Number of dominant positive-loading features to report per PC.

    Returns
    -------
    loadings_norm : np.ndarray, shape (K_components, K_features)
    coords_norm   : np.ndarray, shape (T_observations, K_components)
    pc_positive_top : dict[str, list[str]]
        Per-PC mapping like {'PC1': ['Homicide', 'Weapons'], ...},
        listing the top-N features by positive loading after normalization.

    Notes
    -----
    Stability trade-offs:

        'anchor'  : Flips so the score of `anchor_observation` is >= 0
                    on every PC. STABLE under feature inclusion/exclusion
                    because the decision depends on a time-stable
                    observation, not on any individual feature. Recommended
                    default for compositional/CLR PCA where the feature
                    set may change between runs.

        'max_abs' : Flips so the feature with the largest absolute loading
                    has a positive value on each PC. INTERPRETABLE but
                    NOT stable under feature exclusion -- the 'largest
                    absolute loading' feature can change between runs, so
                    the convention itself silently changes. Subject to
                    floating-point tie-breaking when two features have
                    similar absolute loadings.

        'mean'    : Flips so the mean score across observations is positive.
                    Only meaningful when scores were computed without centering;
                    sklearn's PCA produces zero-mean scores by construction, so
                    this method is degenerate for sklearn output. Use only with
                    manually-projected, non-centered scores.

        'sum'     : Flips so the sum of loadings on each PC is positive.
                    DEGENERATE for CLR-transformed data because CLR
                    loadings sum to ~0 by construction, making the sign
                    decision dominated by floating-point noise. Raises
                    ValueError if the sum is below 1e-8. Appropriate for
                    raw or correlation-based PCA only.
    """
    # Validate inputs and convert to numpy arrays
    if method not in VALID_SIGN_METHODS:
        raise ValueError(
            f"method must be one of {VALID_SIGN_METHODS}, got {method!r}"
        )

    # ----------------------------
    # Step 1-A: Input validation
    # ----------------------------
    loadings = np.asarray(loadings)
    coords   = np.asarray(coords)

    if loadings.ndim != 2:
        raise ValueError("loadings must be 2D (components X features).")

    if coords.ndim != 2:
        raise ValueError("coords must be 2D (observations X components).")

    n_components, n_features = loadings.shape

    if coords.shape[1] != n_components:
        raise ValueError("coords columns must match number of components.")

    if len(feature_names) != n_features:
        raise ValueError("feature_names length must match number of features.")

    if not (0 <= anchor_observation < coords.shape[0]):
        raise IndexError("anchor_observation out of bounds.")

    # --------------------------------------------------------------------
    # Step 1-B: Create copies of loadings and coords to apply sign 
    # normalization without modifying originals
    # --------------------------------------------------------------------
    loadings_norm = loadings.copy()
    coords_norm   = coords.copy()

    # --------------------------------------------------------------------
    # Step 1-C: Apply sign normalization per component according to the 
    # chosen method
    # --------------------------------------------------------------------
    for i in range(n_components):
        if method == 'anchor':
            anchor_value = coords_norm[anchor_observation, i] 
            # Avoid unstable sign flips when anchor score is numerically ~0
            if np.isclose(anchor_value, 0.0, atol=1e-10):
                should_flip = False
            else:
                should_flip = anchor_value < 0

        elif method == 'max_abs':
            # NOTE: not stable under feature changes; argmax breaks ties
            # by returning the first index, which is column-order-dependent
            max_feat_idx = np.argmax(np.abs(loadings_norm[i]))
            should_flip  = loadings_norm[i, max_feat_idx] < 0

        elif method == 'mean':
            mean_score = np.mean(coords_norm[:, i])
            if abs(mean_score) < 1e-8:
                raise ValueError(
                    f"method='mean' is degenerate on PC{i+1}: scores have mean "
                    f"{mean_score:.2e}, suggesting PCA was fit on already-centered data. "
                    f"Use method='anchor' for CLR/centered inputs."
                )
            should_flip = mean_score < 0
        
        else:  # 'sum'
            loading_sum = np.sum(loadings_norm[i])
            if abs(loading_sum) < 1e-8:
                raise ValueError(
                    f"method='sum' is degenerate on PC{i+1}: loadings sum to "
                    f"{loading_sum:.2e}, indicating CLR-transformed input. "
                    f"Use method='anchor' for CLR data."
                )
            should_flip = loading_sum < 0

        # Apply the determined sign flip to both loadings and scores for this component        
        if should_flip:
            loadings_norm[i]  *= -1
            coords_norm[:, i] *= -1

    # --------------------------------------------------------------------
    # Step 1-D: Record top-N positive-loading features per PC for 
    # substantive labeling
    # --------------------------------------------------------------------
    pc_positive_top = {}
    for i in range(n_components):
        ser = pd.Series(loadings_norm[i], index=feature_names)
        pc_positive_top[f'PC{i+1}'] = ser.nlargest(top_n).index.tolist()

    # --------------------------------------------------------------------
    # Step 1-E: Convert loadings to a DataFrame with feature names
    # --------------------------------------------------------------------
    loadings_df = pd.DataFrame(
        loadings_norm,
        index=[f"PC{i+1}" for i in range(n_components)],
        columns=feature_names
    )

    # Convert coordinates to a DataFrame with PC labels
    coords_df = pd.DataFrame(
        coords_norm,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Keep both DataFrame and ndarray forms in the result for downstream flexibility
    return {
        'feature_names'     : list(feature_names),   # Original feature names for reference
        'loadings_norm'     : loadings_df.values,    # ndarray (back-compat)
        'loadings_norm_df'  : loadings_df,           # DataFrame (named-index)
        'coords_norm'       : coords_df.values,      # ndarray (back-compat)
        'coords_norm_df'    : coords_df,             # DataFrame (named-columns)
        'pc_positive_top'   : pc_positive_top        # Dict of top-N positive-loading features per PC
    }

# ---------------------------------------------------------------------------------------------
# 2. Main function to compute PCA with optional feature exclusion and sign normalization
# ---------------------------------------------------------------------------------------------
def compute_pca(data: pd.DataFrame,  exclude_features: list=None, epsilon: float=None, 
                verbose: bool=True, method: str='anchor', anchor_observation: int=0,
                top_n: int=3):
    """
    Compute PCA loadings, scores, and sign‑normalized components with optional
    feature exclusion and user‑selectable sign‑normalization strategy.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset where rows are observations and columns are features.
        PCA is performed directly on this matrix after optional exclusions.

    exclude_features : list, optional
        List of feature names to remove before PCA. Blank strings are ignored.
        Features not found in the dataset are reported separately.

    epsilon : float, optional
        Placeholder epsilon value for workflows involving CLR or log transforms.
        Not used internally but returned for bookkeeping.

    verbose : bool, default True
        If True, prints a PCA manifest summarizing:
            - dataset shape
            - excluded features found
            - excluded features missing
            - epsilon value

    method : {'anchor', 'max_abs', 'mean', 'sum'}, default 'anchor'
        Sign‑normalization rule passed to `stable_sign_normalize`:
            - 'anchor' : flip based on a specific observation's score
            - 'max_abs': flip so the dominant loading is positive
            - 'mean'   : flip so mean score is positive
            - 'sum'    : flip so sum of loadings is positive

    anchor_observation : int, default 0
        Row index used when method='anchor'. Ignored for other methods.

    top_n : int, default 3
        Number of top positive‑loading features to extract per PC after
        sign normalization.

    Returns
    -------
    dict
        A structured dictionary containing:
            - working_data      : DataFrame after exclusions
            - excluded_found    : list of excluded features present
            - excluded_missing  : list of excluded features not present
            - epsilon           : passthrough epsilon value
            - pca               : fitted sklearn PCA object
            - loadings_raw      : raw PCA loadings
            - loadings_norm     : sign‑normalized loadings (ndarray)
            - loadings_norm_df  : sign‑normalized loadings (DataFrame)
            - coords_raw        : raw PCA scores
            - coords_norm       : sign‑normalized scores (ndarray)
            - coords_norm_df    : sign‑normalized scores (DataFrame)
            - variance_ratio    : explained variance ratio per PC
            - singular_values   : PCA singular values
            - observation_index : index of observations used
            - feature_index     : feature names after exclusion
            - pc_positive_top   : dict of top‑N positive‑loading features per PC

    Notes
    -----
    - PCA uses sklearn defaults (centered data, no scaling unless pre‑applied).
    - Sign normalization is delegated to `stable_sign_normalize`, allowing
      consistent orientation across runs, eras, or feature‑set changes.
    """
    # --------------------------------------------------------------------
    # Step 2-A: Feature exclusion
    # --------------------------------------------------------------------
    # Deduplicate and remove blank strings from the exclusion list
    exclude_features = sorted(
        list(dict.fromkeys([x for x in exclude_features if x != ''])) 
        if exclude_features else [])

    # Features that exist in the dataset
    excluded_found   = [col for col in exclude_features if col in data.columns]

    # Features requested for exclusion but not present
    excluded_missing = [col for col in exclude_features if col not in data.columns]

    # Drop only the features that actually exist
    working_data    = data.drop(columns=excluded_found).copy()
    working_columns = working_data.columns

    if verbose:
        # Print a summary of PCA inputs and exclusions
        _print_pca_manifest(working_data, epsilon, excluded_found, excluded_missing)

    # --------------------------------------------------------------------
    # Step 2-B: Fit PCA on the working dataset and extract raw loadings, 
    # scores, and variance ratios
    # --------------------------------------------------------------------
    pca           = PCA()
    coords_raw    = pca.fit_transform(working_data)  # PCA scores
    loadings_raw  = pca.components_                  # PCA loadings
    ratios        = pca.explained_variance_ratio_    # Variance explained

    # Apply stable sign normalization using the user-selected method
    results = stable_sign_normalize(loadings=loadings_raw,
                                    coords=coords_raw,
                                    method=method,
                                    anchor_observation=anchor_observation,
                                    feature_names=working_columns,
                                    top_n=top_n)

    # Return a structured dictionary of PCA results
    return {
        'working_data'      : working_data,
        'excluded_found'    : excluded_found,
        'excluded_missing'  : excluded_missing,
        'epsilon'           : epsilon,
        'pca'               : pca,
        'loadings_raw'      : loadings_raw,
        'loadings_norm'     : results['loadings_norm'],
        'loadings_norm_df'  : results['loadings_norm_df'],
        'coords_raw'        : coords_raw,
        'coords_norm'       : results['coords_norm'],
        'coords_norm_df'    : results['coords_norm_df'],
        'variance_ratio'    : ratios,
        'singular_values'   : pca.singular_values_,
        'observation_index' : working_data.index.values,
        'feature_index'     : working_data.columns,
        'pc_positive_top'   : results['pc_positive_top']
    }


# ---------------------------------------------------------------------------------------------
# 2-1-A: Helper function to print a formatted PCA manifest summarizing the dataset and exclusions
# ---------------------------------------------------------------------------------------------
def _print_pca_manifest(working_data: pd.DataFrame, epsilon: float = None, 
                        exclude_features: list=None, excluded_missing: list=None, se_emoji: bool=True):
    """
    Print a formatted PCA data manifest summarizing the dataset used for PCA,
    excluded features, and epsilon settings. This helper centralizes all
    formatting and display logic so that PCA‑related functions produce a
    consistent, readable summary.

    Parameters
    ----------
    working_data : pd.DataFrame
        The dataset after applying feature exclusions. Row count and column
        count are displayed in the manifest.

    epsilon : float or None
        Optional epsilon value used in CLR or log‑ratio transformations.
        Displayed in the header if provided.

    exclude_features : list or None
        List of features that were successfully excluded from the dataset.
        Printed under the "Excluded" section.

    excluded_missing : list or None
        List of features requested for exclusion but not found in the dataset.
        Printed under the "Excluded Not Found" section.

    se_emoji : bool or None, default True
        Controls whether emoji icons are used in the manifest:
            True  → use emoji symbols
            False → use ASCII symbols
            None  → fallback to ASCII

    Notes
    -----
    - This function is purely for display and has no computational role.
    - All formatting decisions (width, labels, icons) are centralized here
      so that PCA output remains consistent across the codebase.
    """
    # Width of the printed manifest box
    width = 60
    # Label and value column widths for aligned printing
    lbl_w, val_w = 28, 24

    # Choose emoji or ASCII symbols depending on user preference
    if se_emoji:
        chart, warn, ok, no = '📊', '[⚠️]', '[✅]', '[🚫]'
    elif se_emoji is False:
        chart, warn, ok, no = '[*]', '[⚠️]', '[✅]', '[🚫]'

    # Header includes epsilon value if provided
    header = (
        f"{chart} PCA DATA MANIFEST (ε = {epsilon:.5f})"
        if epsilon is not None else f"{chart} PCA DATA MANIFEST"
    )

    # Begin printing the manifest
    print("\n" + "=" * width)
    print(header.center(width))
    print("-" * width)

    # Basic dataset dimensions
    print(f"{ok} {'Observations':<{lbl_w}}: {working_data.shape[0]:>{val_w}}")
    print(f"{ok} {'Included Features':<{lbl_w}}: {working_data.shape[1]:>{val_w}}")
    
    # Excluded features (present in data)
    n_excluded = len(exclude_features) if exclude_features else 0
    print(f"{no} {'Excluded':<{lbl_w}}: {n_excluded:>{val_w}}")
    for item in (exclude_features or []):
        print(f"      - {item}")

    # Excluded features not found in the dataset
    m_excluded = len(excluded_missing) if excluded_missing else 0
    print(f"{warn} {'Excluded Not Found':<{lbl_w}}: {m_excluded:>{val_w}}")
    for item in (excluded_missing or []):
        print(f"      - {item}")    

    # End of manifest
    print("=" * width + "\n")


# ---------------------------------------------------------------------------------------------
# 3. Function to plot the first three principal component loadings with a clean, modern style
# ---------------------------------------------------------------------------------------------
def plot_three_loadings(
    pca_dict,
    order=None,          
    epsilon=None,
    figsize=(20, 10),
    label_fmt="{:.2f}"
):
    """
    Create a clean, modern horizontal‑bar visualization of the first three
    principal component loadings. Each PC is displayed in its own subplot,
    with positive loadings highlighted in color and negative loadings shown
    in a muted tone.

    Parameters
    ----------
    pca_dict : dict
        Dictionary returned by `compute_pca` or `compute_pc_loadings`.
        Must contain:
            - 'loadings_norm'   : sign‑normalized loadings (ndarray)
            - 'variance_ratio'  : explained variance ratio per PC
            - 'feature_index'   : feature names
            - optionally 'epsilon'

    order : list or None, default None
        Optional custom ordering of feature names for the y‑axis.
        If None, features are sorted by PC1 loadings (ascending).

    epsilon : float or None, default None
        Optional epsilon value to display in the plot title.
        If None, attempts to read from `pca_dict['epsilon']`.

    figsize : tuple, default (20, 10)
        Figure size passed to matplotlib.

    label_fmt : str, default "{:.2f}"
        Format string for bar‑end numeric labels.

    Returns
    -------
    dict
        {
            'figure': matplotlib Figure,
            'axis'  : array of Axes objects
        }

    Notes
    -----
    - Uses seaborn styling for a clean, modern look.
    - Negative loadings are shown in a neutral color; positive loadings use
      distinct colors per PC.
    - Displays cumulative variance for PC1–PC3 in the title.
    """

    # --------------------------------------------------------------------
    # Step 3-A: Extract necessary data from the PCA dictionary
    # --------------------------------------------------------------------
    # Use sign-normalized loadings for consistent orientation
    loadings_norm = pca_dict["loadings_norm"]
    ratios        = pca_dict["variance_ratio"]
    feature_names = list(pca_dict["feature_index"])

    # --------------------------------------------------------------------
    # Step 3-B: Smart Ordering Logic
    # --------------------------------------------------------------------
    # If no custom order is provided, sort features by PC1 loadings
    if order is None:
        order = pd.Series(loadings_norm[0], index=feature_names).sort_values().index.tolist()
    else:
        order = list(order)

    # --------------------------------------------------------------------
    # Step 3-C: Aesthetic Setup and Bar Plotting
    # --------------------------------------------------------------------
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    # Colors for positive loadings per PC
    pc_colors = ["#E67E22", "#2980B9", "#27AE60"]  # Orange, Blue, Green
    neg_color = "#BDC3C7"  # Silver for negative loadings

    # Loop through the first three PCs and create horizontal bar plots
    for i, ax in enumerate(axes):
        # Align loadings with the chosen order
        ser = pd.Series(loadings_norm[i], index=feature_names).reindex(order)
        y_pos = np.arange(len(ser))

        # Positive loadings get PC-specific color; negatives get neutral color
        colors = [pc_colors[i] if v >= 0 else neg_color for v in ser.values]

        # Plot horizontal bars
        bars = ax.barh(y_pos, ser.values, color=colors, edgecolor="white", lw=0.8)

        # Add numeric labels to bar ends
        ax.bar_label(
            bars, fmt=label_fmt, padding=8,
            fontsize=10, fontweight="bold", color="#34495e"
        )

        # Add vertical zero line and grid
        ax.axvline(0, color="#2c3e50", lw=1.2, zorder=3)
        ax.xaxis.grid(True, ls="--", alpha=0.3)

        # Title with variance explained
        ax.set_title(
            f"PC{i+1}\n{ratios[i]:.1%} Variance",
            fontweight="bold", size=15, pad=35, color="#2c3e50"
        )
        # Remove spines and ticks for a cleaner look
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_xticks([])

        # Only the first subplot shows feature names
        if i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ser.index, fontweight="bold", fontsize=11, color="#2c3e50")
            ax.tick_params(axis="y", which="major", pad=40)
        else:
            ax.tick_params(axis="y", length=0)
    # --------------------------------------------------------------------
    # Step 3-D: Super-Title with Dynamic Epsilon Logic
    # --------------------------------------------------------------------
    display_eps = epsilon if epsilon is not None else pca_dict.get("epsilon")
    eps_str = fr"($\boldsymbol{{\epsilon}}$ = {display_eps:.3g})" if display_eps is not None else ""
    # show cumulative variance for PC1–PC3 in the title
    main_title = (
        f"Principal Component Structural {eps_str}\n"
        f"Cumulative Variance (PC1-PC3): {sum(ratios[:3]):.1%}"
    )
    # Use a larger font size and bold weight for the main title, with a subtle color
    plt.suptitle(
        main_title,
        fontweight="bold",
        size=20,
        y=1.08,
        color="#2c3e50"
    )
    # Adjust title position to prevent overlap with subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)
    # --------------------------------------------------------------------
    return {'figure': fig, 'axis': axes}