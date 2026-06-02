import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from typing import Mapping, Dict, Any, Tuple, Optional
import seaborn as sns
from numpy.linalg import norm
from scipy.linalg import subspace_angles
from scipy import stats


# ---------------------------------------------------------------------------
# 1: PCA stability comparison across epsilon values
# ---------------------------------------------------------------------------
def compare_pca_stability(
    results,
    eps1,
    eps2,
    n_components=3,
    drift_threshold=0.95,
    show_plot=True,
    image_path: str = None,
    image_save: str = None
):
    """
    Compare PCA stability across two epsilon values.

    Computes variance-structure drift, loading-vector alignment,
    subspace rotation, and sample-embedding stability between the PCA
    decompositions at eps1 and eps2. Produces:

      - Singular-value drift (global variance structure)
      - Cosine similarity of loading vectors (directional stability)
      - Principal angles between PC subspaces (rotation-invariant drift)
      - Correlation of sample coordinates (embedding stability)
      - Composite 0–1 stability index
      - Optional 3-panel diagnostic dashboard

    Parameters
    ----------
    results : dict
        Mapping eps -> PCA result bundles containing singular values,
        loadings, and sample coordinates.

    eps1, eps2 : float
        Epsilon values to compare.

    n_components : int, default=3
        Number of principal components to evaluate.

    drift_threshold : float, default=0.95
        Minimum acceptable cosine similarity for PC robustness.

    show_plot : bool, default=True
        Whether to render the diagnostic dashboard.

    image_path : str, optional
        Path to save the figure. If provided, saves as PNG at 300 DPI.

    image_save : str, optional
        Image file name to save the figure. If provided, saves as PNG at 300 DPI.

    Returns
    -------
    dict
        {
            "composite_score": float,
            "angles": np.ndarray of principal angles in degrees,
            "cosines": np.ndarray of per-PC cosine similarities,
            "coord_corrs": np.ndarray of per-PC sample-coordinate correlations,
            "sv_diff_norm": float, L2 norm of singular-value differences
        }
    """
    # -------------------------------------------------
    # 1-A: Extract PCA artifacts for both eps values with safe access
    # -------------------------------------------------
    r1, r2 = results[eps1], results[eps2]

    # Helper for safe extraction with fallback keys
    def _get(bundle, keys):
        for k in keys:
            val = bundle.get(k)
            if val is not None:
                return val
        raise KeyError(f"Data missing: {keys}")

    # Extract singular values, loadings, and coordinates
    sv1 = _get(r1, ['singular_values'])[:n_components]
    sv2 = _get(r2, ['singular_values'])[:n_components]
    L1  = _get(r1, ['loadings_norm'])[:n_components]
    L2  = _get(r2, ['loadings_norm'])[:n_components]
    C1  = _get(r1, ['coords_norm'])[:, :n_components]
    C2  = _get(r2, ['coords_norm'])[:, :n_components]

    # -------------------------------------------------
    # 1-B: Compute stability metrics for variance structure, 
    # loading alignment, subspace rotation, and embedding consistency
    # -------------------------------------------------
    # Global variance-structure drift (L2 difference of singular values)
    sv_diff_norm = norm(sv1 - sv2)

    # Directional stability: cosine similarity of loading vectors
    cosines = np.einsum('ij,ij->i', L1, L2) / (norm(L1, axis=1) * norm(L2, axis=1))

    # Rotation-invariant drift: principal angles between PC subspaces
    angles_deg = np.degrees(subspace_angles(L1.T, L2.T))

    # Embedding stability: correlation of sample coordinates
    coord_corrs = np.array([
        np.corrcoef(C1[:, i], C2[:, i])[0, 1]
        for i in range(n_components)
    ])

    # Identify the PC with the largest directional drift
    worst_idx = np.argmin(cosines)
    worst_val = cosines[worst_idx]

    # -------------------------------------------------
    # 1-C: Compute a composite stability score
    # -------------------------------------------------
    # 1-C-1: SV Score: Normalize the drift by the magnitude of the original values
    # This prevents large raw SV values from artificially tanking the score.
    sv_score = np.exp(-norm(sv1 - sv2) / norm(sv1))

    # 1-C-2: Cosine Score: Use the MINIMUM instead of the MEAN. 
    # This ensures that if even ONE PC flips or drifts, the health score reflects it.
    cos_score = np.min(cosines) 

    # 1-C-3: Coordinate Score: Keep the mean correlation for embedding stability.
    coord_score = np.mean(coord_corrs)

    # 1-C-4: Angle Score: Normalize by 90 degrees (the max possible tilt).
    # This makes the "tilt" penalty linear and more intuitive.
    angle_score = 1 - (angles_deg.max() / 90.0)

    # 1-C-5: Final Unified Index: Using Geometric Mean makes the score "stricter"
    # (Formula: nth root of the product of n scores)
    composite_score = (sv_score * cos_score * coord_score * angle_score) ** (1/4)

    # -------------------------------------------------
    # 1-D: Print a formatted summary of stability metrics and optionally render a diagnostic dashboard
    # -------------------------------------------------
    width, lbl_w, val_w = 65, 37, 12
    print("\n" + "=" * width)
    print(f"{'🔍 PCA STABILITY CROSS-CHECK':^{width}}")
    print(f"{f'(ε={eps1:.6f} vs ε={eps2:.6f})':^{width}}")
    print(f"{f'Number of Principal Components: {n_components}':^{width}}")
    print("-" * width)
    print(f"⚖️  {'SV Magnitude Shift (L2)':<{lbl_w}} : {sv_diff_norm:>{val_w}.4f}")
    print(f"📐  {'Max Subspace Angle':<{lbl_w}} : {f'{angles_deg.max():.2f}°':>{val_w}}")
    print(f"💎  {'Composite Health Score':<{lbl_w}} : {composite_score:>{val_w}.4f}")
    print("-" * width)
    for i in range(n_components):
        status = "✅" if cosines[i] >= drift_threshold else "⚠️"
        print(f"PC{i+1} Robustness (Cosine Similarity) {status:<5} : {cosines[i]:>{val_w}.4f}")
    print("=" * width + "\n")

    # -------------------------------------------------
    # 1-E: If show_plot is True, render a 3-panel diagnostic dashboard visualizing 
    # loading stability, subspace rotation, and contribution patterns
    # -------------------------------------------------
    if show_plot:
        sns.set_theme(style="white", context="talk")
        plt.rcParams['font.family'] = 'sans-serif'

        fig = plt.figure(figsize=(24, 8), facecolor="#F8F9FA", constrained_layout=True)
        gs = fig.add_gridspec(1, 3)

        # Color palette for the dashboard elements
        accent, drift, safe, neutral = "#1A5276", "#CB4335", "#28B463", "#D5DBDB"

        # Panel A: Loading stability with cosine similarity and coordinate correlation
        ax1 = fig.add_subplot(gs[0, 0])
        x, bw = np.arange(n_components), 0.35
        pcs = [f"PC{i+1}" for i in range(n_components)]
        # Plot loading cosine bars with conditional coloring based on drift threshold, 
        # and a distinct color for the worst drift
        ax1.bar(
            x - bw/2,
            cosines,
            bw,
            color=[drift if i == worst_idx else accent for i in range(n_components)],
            label='Loading Cosine',
            edgecolor='white',
            lw=2,
            alpha=0.9,
            zorder=3
        )
        # Overlay coordinate correlation bars with a different color and slight transparency
        ax1.bar(
            x + bw/2,
            coord_corrs,
            bw,
            color=neutral,
            label='Coord Corr',
            edgecolor='white',
            lw=2,
            alpha=0.7,
            zorder=3
        )
        # Reference line for drift threshold
        ax1.axhline(
            drift_threshold,
            color=drift,
            ls='--',
            lw=1.5,
            alpha=0.4,
            label='Stability Threshold',
            zorder=2
        )
        # Annotate the worst loading drift with a callout box and arrow
        ax1.annotate(
            f'LARGEST DRIFT: {pcs[worst_idx]}',
            xy=(worst_idx - bw/2, worst_val),
            xytext=(worst_idx - bw/2, worst_val + 0.12),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='900',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=drift, lw=2),
            arrowprops=dict(arrowstyle="->", color=drift, lw=1.5),
            zorder=5
        )

        ax1.set_title("COMPONENT ROBUSTNESS", fontweight="900", size=14, pad=25, loc='left')
        ax1.set_ylim(min(0.5, worst_val - 0.15), 1.3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(pcs, fontweight="bold")
        ax1.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, -0.28), ncol=3, fontsize=10)

        # Panel B: Subspace rotation
        ax2 = fig.add_subplot(gs[0, 1])
        angle_labels = [f"θ{i+1}" for i in range(len(angles_deg))]
        sns.barplot(
            x=angle_labels,
            y=angles_deg,
            ax=ax2,
            hue=angle_labels,
            palette="GnBu_d",
            legend=False,
            alpha=0.8
        )
        # Annotate the maximum angle with a label and an arrow
        max_v = angles_deg.max()
        ax2.annotate(
            f'MAX TILT: {max_v:.2f}°',
            xy=(0, max_v),
            xytext=(0, max_v + (max_v * 0.3)),
            ha='center',
            fontweight='bold',
            size=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=safe, lw=2),
            arrowprops=dict(arrowstyle="->", color=safe)
        )

        ax2.set_title("SUBSPACE ROTATION", fontweight="900", size=14, pad=25, loc='left')
        ax2.set_ylabel("Degrees (°)")
        ax2.set_ylim(0, max_v * 1.8)

        # Panel C - Contribution matrix
        ax3 = fig.add_subplot(gs[0, 2])
        U, _, _ = np.linalg.svd(L1 @ L2.T)
        contrib_norm = np.abs(U.T) / np.abs(U.T).sum(axis=0, keepdims=True)
        # Note: The contribution matrix is visualized as a heatmap where each cell represents 
        # the normalized contribution of a PC from the first decomposition to the tilt observed 
        # in the second decomposition. The heatmap uses a blue color palette to indicate the 
        # strength of contributions, with annotations showing the exact values for clarity. 
        # This panel provides insight into which components are most responsible for the 
        # observed drift, complementing the loading stability and subspace rotation analyses
        # in the first two panels. 
        sns.heatmap(
            contrib_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            xticklabels=angle_labels,
            yticklabels=pcs,
            ax=ax3,
            linewidths=2,
            linecolor='white',
            annot_kws={"size": 12, "fontweight": "bold"}
        )
        ax3.set_title("PC CONTRIBUTION TO TILT", fontweight="900", size=14, pad=25, loc='left')

        sns.despine(left=True, bottom=True)
        for ax in [ax1, ax2]:
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)

        plt.suptitle(
            f"PCA STABILITY DIAGNOSTIC: COMPOSITE HEALTH {composite_score:.2f}",
            fontsize=24,
            fontweight='900',
            color='#1B2631',
            y=1.08
        )
        
        # Save the figure if image_save and image_path are provided
        if image_save and image_path:
            fig.savefig(image_path+image_save, dpi=300, bbox_inches='tight')
        
        plt.show()

    return {
        "composite_score": composite_score,
        "angles": angles_deg,
        "cosines": cosines,
        "coord_corrs": coord_corrs,
        "sv_diff_norm": sv_diff_norm,
    }


# ---------------------------------------------------------------------------
# 2: PCA sensitivity visualization across multiple epsilon values
# ---------------------------------------------------------------------------
def pca_sensitivity_plot(
    results: Dict,
    title: str = "PCA Noise Sensitivity Analysis",
    threshold_var: float = 0.80,
    k_components: int = None,
    image_path: str = None,
    image_save: str = None
):
    """
    Visualize PCA sensitivity across exactly two epsilon values.

    This function produces a two‑panel diagnostic figure comparing:
      - Singular value spectra across epsilon values
      - Cumulative variance explained across epsilon values

    Component selection:
      - If `k_components` is provided, it is used directly.
      - Otherwise, the smallest k satisfying cumulative variance ≥ `threshold_var`
        (default 0.80) is selected.

    The chosen k is highlighted on both panels with vertical and horizontal
    reference lines, and the corresponding tick labels are emphasized.

    Parameters
    ----------
    results : dict
        Mapping: epsilon -> {"singular_values": array, "variance_ratio": array}

    title : str, default "PCA Noise Sensitivity Analysis"
        Main title for the figure.

    threshold_var : float, default 0.80
        Minimum cumulative variance required to determine k when
        `k_components` is not provided.

    k_components : int, optional
        Explicit number of components to highlight.

    image_path : str, optional
        Directory where the figure should be saved.

    image_save : str, optional
        Filename for saving the figure.

    results : dict
        Mapping: epsilon -> {"singular_values": array, "variance_ratio": array}
        Must contain exactly two epsilon values for comparison.
    """
    # -------------------------------------------------
    # Step 2-1: Set up the figure
    # -------------------------------------------------
    sns.set_theme(style="white")
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))

    epsilons = sorted(results.keys())
    if len(epsilons) != 2:
        raise ValueError(
            f"pca_sensitivity_plot requires exactly 2 epsilon values, got {len(epsilons)}"
    )
    palette = ["#2d3436", "#d85a30"]

    # Baseline epsilon for determining k
    first_eps = epsilons[0]
    s_vals_first = results[first_eps]["singular_values"]
    v_ratios_first = results[first_eps]["variance_ratio"]
    cum_var_first = np.cumsum(v_ratios_first)

    # -------------------------------------------------
    # Step 2-2: Determine target k and capture values
    # -------------------------------------------------
    if k_components is not None:
        target_k = int(k_components)
    else:
        target_k = int(np.argmax(cum_var_first >= threshold_var) + 1)

    # DEFINITIONS NEEDED FOR THE PLOT LINES:
    var_at_k = cum_var_first[target_k - 1]
    k_singular_val = s_vals_first[target_k - 1]

    # -------------------------------------------------
    # Step 2-3: Ultra-Tight Subtitle (Optimized)
    # -------------------------------------------------
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    # 1. Grab raw values
    v0, v1 = [np.cumsum(results[e]["variance_ratio"])[target_k-1] for e in epsilons]
    s0, s1 = [results[e]["singular_values"][target_k-1] for e in epsilons]

    # 2. Anchors and Data mapping
    x_anchor = 0.52 
    # Create a list of tuples: (Label, Value_Eps0, Value_Eps1, Y_Position, Format)
    rows = [
        (f"Cumulative Variance at k={target_k}: ", v0, v1, 0.915, ".1%"),
        (f"Singular Value at k={target_k}: ", s0, s1, 0.885, ".2f")
    ]

    for label, val0, val1, y, fmt in rows:
        # Prefix (Right-aligned)
        fig.text(x_anchor, y, label, ha='right', fontsize=10, color="#636e72")
        
        # Values and "vs" (Left/Center-aligned)
        fig.text(x_anchor + 0.01, y, f"{val0:{fmt}}", ha='left', fontsize=10, fontweight='bold', color=palette[0])
        fig.text(x_anchor + 0.06, y, "vs", ha='center', fontsize=9, fontweight='bold', color="#2d3436")
        fig.text(x_anchor + 0.08, y, f"{val1:{fmt}}", ha='left', fontsize=10, fontweight='bold', color=palette[1])

    # -------------------------------------------------
    # Step 2-4: Plot singular values + cumulative variance
    # -------------------------------------------------
    for i, eps in enumerate(epsilons):
        s_vals = results[eps]["singular_values"]
        ratios = results[eps]["variance_ratio"]

        # Singular values
        sns.lineplot(
            x=range(1, len(s_vals) + 1),
            y=s_vals,
            marker="o",
            color=palette[i],
            label=fr" = {eps:.6f}",
            ax=axes[0]
        )

        # Cumulative variance
        sns.lineplot(
            x=range(1, len(ratios) + 1),
            y=np.cumsum(ratios),
            marker="o",
            color=palette[i],
            label=f"ε = {eps:.6f}",
            ax=axes[1]
        )

    # -------------------------------------------------
    # Step 2-5: Highlighting + formatting
    # -------------------------------------------------
    two_dec_formatter = FuncFormatter(lambda x, pos: f"{x:.2f}")
    
    # Iterates over both subplots to add vertical reference lines 
    for i, ax in enumerate(axes):
        ax.set_xlabel("Component Index" if i == 0 else "Number of Components")
        ax.set_ylabel("Singular Value" if i == 0 else "Cumulative Variance")

        # Vertical line at k
        ax.axvline(target_k, color="#636e72", ls="--", lw=1.2, alpha=0.6)

        # Horizontal reference line
        if i == 0:
            ax.axhline(k_singular_val, color="#b2bec3", ls=":", lw=1)
            target_val = k_singular_val
        else:
            ax.axhline(var_at_k, color="#b2bec3", ls=":", lw=1.5)
            ax.yaxis.set_major_formatter(two_dec_formatter)
            target_val = var_at_k

        # Ensure k is in x‑ticks
        xticks = sorted(set(ax.get_xticks()).union({target_k}))
        ax.set_xticks([t for t in xticks if t >= 0])

        # Highlight tick labels
        plt.draw()
        for tick in ax.get_xticklabels():
            if tick.get_text() == str(target_k):
                tick.set_fontweight("bold")
                tick.set_color("#d85a30")

        y_min, y_max = ax.get_ylim()
        tol = (y_max - y_min) * 0.02  # 2% of axis range

        for tick in ax.get_yticklabels():
            try:
                if abs(float(tick.get_text().replace('−', '-')) - target_val) < tol:
                    tick.set_fontweight("bold")
                    tick.set_color("#d85a30")
            except ValueError:
                pass

    # -------------------------------------------------
    # Step 2-6: Final styling
    # -------------------------------------------------
    titles = ["Singular Value Spectrum", "Cumulative Variance Explained"]
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontweight="semibold", pad=20)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

    sns.despine(trim=True)
    
    # Adjust layout to prevent overlap and ensure the subtitle and annotations are clear
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) 

    # Save figure
    if image_save and image_path:
        fig.savefig(os.path.join(image_path, image_save), dpi=300, bbox_inches='tight')

    plt.show()


# ---------------------------------------------------------------------------
# 3: Aitchison variance structure visualization across epsilon values
# ---------------------------------------------------------------------------
def plot_aitchison(
    clr_data: Mapping[float, pd.DataFrame],
    chosen_eps: float | None = None,
    compare_eps: float | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] = (12, 5.5)
) -> dict[str, Any]:
    """
    Plot Aitchison variance diagnostics across epsilon thresholds.

    This visualization shows:
    - Total Aitchison variance (left y-axis)
    - PC1 variance ratio (right y-axis)
    - Optional vertical markers for a chosen ε and a comparison ε
    - Optional annotated callout boxes for both markers

    The plot uses a dual-axis layout:
    - Left axis: total variance (blue)
    - Right axis: PC1 ratio (coral)
    - Vertical lines: chosen ε (green), alternative ε (gray)

    Parameters
    ----------
    clr_data : Mapping[float, pd.DataFrame]
        Dictionary mapping epsilon -> CLR-transformed DataFrame.
    chosen_eps : float or None
        Primary epsilon to highlight with a vertical line and annotation.
    compare_eps : float or None
        Secondary epsilon to highlight for comparison.
    annotate : bool
        Whether to draw callout boxes for chosen_eps and compare_eps.
    figsize : tuple[float, float]
        Figure size passed to matplotlib.

    Returns
    -------
    dict
        A dictionary containing:
        - "figure": the matplotlib Figure
    """
    # Color palette for the plot elements
    c_blue, c_coral, c_green, c_gray = "#185FA5", "#D85A30", "#3B6D11", "#666666"

    # Compute the Aitchison profile DataFrame from the provided CLR data
    profile = compute_aitchison_profile(clr_data)

    # Set up the dual-axis plot with a clean, modern aesthetic
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # ------------------------------------------------------------------
    # 3-1. Base series: Total variance (left) and PC1 ratio (right)
    # ------------------------------------------------------------------
    # Note: The total variance is plotted on the primary y-axis with a distinct color a
    # nd marker style to ensure it stands out as the main focus of the plot. The use of 
    # circular markers and a solid line style emphasizes the continuity of the variance 
    # trend across epsilon values. 
    l1, = ax1.plot(
        profile.index, profile["total_variance"],
        color=c_blue, marker="o", ms=5, lw=1.8,
        label="Total Variance"
    )
    # Note: The PC1 ratio is plotted on the secondary y-axis with a distinct 
    # color and marker style to differentiate it from the total variance series. 
    # The dashed line style emphasizes that this is a ratio metric, while the square 
    # markers provide visual contrast to the circular markers used for total variance.
    l2, = ax2.plot(
        profile.index, profile["pc1_ratio"],
        color=c_coral, marker="s", ms=5, lw=1.8, ls="--",
        label="PC1 Ratio"
    )

    # Legend handles will be built in a specific order
    handles = [l1, l2]

    # ------------------------------------------------------------------
    # 3-2. Primary chosen epsilon marker
    # ------------------------------------------------------------------
    # Note: The chosen epsilon is highlighted with a distinct color and a vertical line.
    if chosen_eps is not None:
        v_line = ax1.axvline(
            chosen_eps, color=c_green, ls=":", lw=1.8,
            label=rf"Chosen $\epsilon$ ({chosen_eps:.6f})"
        )
        handles.append(v_line)
        # Note: The annotation for the chosen epsilon is designed to be prominent and informative,
        # with a callout box that includes key metrics. The offset is tuned to avoid overlap
        # with the data points while maintaining a clear visual connection to the marker.
        if annotate:
            # Offset tuned for readability: left-shifted, higher placement
            _add_callout(
                ax1, profile, chosen_eps,
                offset=(-60, 100),
                prefix=r"$\epsilon_{chosen}$",
                ha="right"
            )

    # ------------------------------------------------------------------
    # 3-3. Secondary comparison epsilon marker
    # ------------------------------------------------------------------
    if compare_eps is not None:
        c_line = ax1.axvline(
            compare_eps, color=c_gray, ls="--", lw=1.2, alpha=0.6,
            label=rf"Alt $\epsilon$ ({compare_eps:.6f})"
        )
        handles.append(c_line)
        # Note: The annotation for the alternative epsilon is intentionally less prominent 
        # and placed to avoid overlap with the primary marker, ensuring both are readable 
        # without cluttering the plot.
        if annotate:
            # Offset tuned for readability: right-shifted, slightly lower
            _add_callout(
                ax1, profile, compare_eps,
                offset=(60, 40),
                prefix=r"$\epsilon_{alt}$",
                ha="left"
            )

    # Apply consistent styling to axes, grid, and ticks
    _style_axes(ax1, ax2, c_blue, c_coral)
    _finalize_layout(fig, ax1, handles)

    return {"figure": fig}

# ---------------------------------------------------------------------------
# Helper function 3-A: to add annotated callout boxes for epsilon markers
# ---------------------------------------------------------------------------
def _add_callout(ax, profile, eps, offset, prefix, ha):
    """
    Add a labeled callout box pointing to the nearest epsilon value.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis on which to draw the annotation.
    profile : DataFrame
        Aitchison profile indexed by epsilon.
    eps : float
        Target epsilon (will snap to nearest available index).
    offset : tuple[int, int]
        Pixel offset for the annotation text box.
    prefix : str
        LaTeX prefix for the bold label (e.g., $\\epsilon_{chosen}$).
    ha : str
        Horizontal alignment of the text ("left" or "right").
    """
    # Snap to the nearest available epsilon in the profile index
    idx = profile.index.get_indexer([eps], method="nearest")[0]
    snapped_eps = profile.index[idx]
    row = profile.iloc[idx]

    # Construct the label text with LaTeX formatting and key metrics
    label_text = (
        f"{prefix}: {snapped_eps:.6f}" + "\n"
        f"Total Var: {row.total_variance:.3f}\n"
        f"PC1 Ratio: {row.pc1_ratio:.1%}"
    )

    # Draw the annotation with a styled box and an arrow pointing to the data point
    ax.annotate(
        label_text,
        xy=(snapped_eps, row.total_variance),
        xytext=offset,
        textcoords="offset points",
        fontsize=10,
        ha=ha, va="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="white", ec="#CCCCCC",
            alpha=1.0, lw=1, zorder=10
        ),
        # Arrow properties for the callout, with a subtle curve to enhance readability
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=.1",
            color="#333", lw=1.5
        ),
        zorder=11
    )

# ---------------------------------------------------------------------------
# Helper function 3-B: to apply consistent styling to the Aitchison plot axes
# ---------------------------------------------------------------------------
def _style_axes(ax1, ax2, c1, c2):
    """
    Apply consistent styling to the dual-axis Aitchison plot.
    """
    # Logarithmic x-axis for better spacing of epsilon values
    ax1.set_xscale("log")

    # Axis labels with color coding and bold styling
    lbl_sz, tick_sz = 12, 10
    ax1.set_xlabel(r"$\epsilon$ (pseudocount)", size=lbl_sz, fontweight="medium")
    ax1.set_ylabel("Total Aitchison Variance", color=c1, size=lbl_sz, fontweight="bold")
    ax2.set_ylabel("PC1 Variance Ratio", color=c2, size=lbl_sz, fontweight="bold")

    # Style spines, grid, and ticks for both axes
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.3, which="both")
        ax.tick_params(labelsize=tick_sz)

    # Hide the right spine of ax1 and left spine of ax2 for a cleaner look
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Light grid on the left axis for better readability of variance values
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Color the tick labels to match their respective axes
    ax1.tick_params(axis="y", labelcolor=c1)
    ax2.tick_params(axis="y", labelcolor=c2)

# ---------------------------------------------------------------------------
# Helper function 3-C: to finalize the layout, title, and legend for the Aitchison plot
# ---------------------------------------------------------------------------
def _finalize_layout(fig, ax, handles):
    """
    Finalize title, legend, and layout spacing for the Aitchison plot.
    """
    # Main title with LaTeX formatting
    ax.set_title(
        r"CLR Variance Structure vs $\varepsilon$",
        loc="left", pad=25,
        fontweight="bold", size=13
    )

    # Legend order: Total, PC1, Chosen, Alt
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.4
    )
    # Adjust the bottom margin to accommodate the legend without overlap
    plt.subplots_adjust(bottom=0.28)


# ---------------------------------------------------------------------------
# 3-4. Math Kernels (Aitchison Geometry)
# ---------------------------------------------------------------------------
"""
Math Kernels for Aitchison Geometry
-----------------------------------
This module provides helper functions for analyzing compositional data
expressed in CLR (centered log-ratio) coordinates. It includes:

- Total Aitchison variance across all CLR dimensions
- Variance ratio explained by the first principal component (PC1)
- A convenience routine to compute these metrics across multiple
  epsilon-thresholded CLR datasets

All functions assume inputs are NumPy arrays or pandas DataFrames
containing valid finite CLR-transformed values.
"""
# ------------------------------------------------------------------
# 3-4-A: Total Aitchison variance computation
# ------------------------------------------------------------------
def get_total_aitchison_variance(X: np.ndarray) -> float:
    """
    Compute the total Aitchison variance of a CLR matrix.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of CLR-transformed compositional data.

    Returns
    -------
    float
        The sum of variances across all CLR coordinates.
        Returns NaN if the array is empty or contains non-finite values.
    """
    # Validate input: must be non-empty and finite
    if X.size == 0 or not np.isfinite(X).all():
        return np.nan

    # Variance is computed per column; sum gives total Aitchison variance
    return float(np.var(X, axis=0, ddof=0).sum())

# ------------------------------------------------------------------
# 3-4-B: PC1 variance ratio computation using SVD
# ------------------------------------------------------------------
def get_pc1_variance_ratio(X: np.ndarray) -> float:
    """
    Compute the proportion of total variance explained by the first
    principal component (PC1) using SVD.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of CLR-transformed compositional data.

    Returns
    -------
    float
        The variance ratio of PC1 (largest singular value squared divided
        by total variance). Returns NaN on invalid input or SVD failure.
    """
    # Validate input
    if X.size == 0 or not np.isfinite(X).all():
        return np.nan

    # Center the data before SVD
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD can fail for degenerate matrices, so we catch exceptions
    try:
        # SVD: singular values s correspond to sqrt of eigenvalues
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)

        # PC1 variance ratio = s1^2 / sum(s_i^2)
        return float(s[0]**2 / np.sum(s**2))

    except Exception:
        # SVD can fail for degenerate matrices
        return np.nan

# ------------------------------------------------------------------
# 3-4-C: Convenience routine to compute Aitchison profile across multiple epsilons
# ------------------------------------------------------------------
def compute_aitchison_profile(clr_data):
    """
    Compute the total Aitchison variance and the PC1 variance ratio 
    for a dictionary of CLR datasets indexed by epsilon thresholds.

    Parameters
    ----------
    clr_data : dict
        A mapping {eps: DataFrame} where each value is a pandas DataFrame
        containing CLR-transformed data for that epsilon threshold.
        Same output files from sweep_epsilon_grid() `clr_dict` can be used here.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by epsilon with columns:
        - 'total_variance'
        - 'pc1_ratio'
    """
    records = []

    # Iterate in sorted epsilon order for reproducibility
    for eps in sorted(clr_data):
        # Convert DataFrame to float NumPy array
        m = clr_data[eps].values.astype(float)

        # Compute metrics
        records.append({
            "eps": eps,
            "total_variance": get_total_aitchison_variance(m),
            "pc1_ratio": get_pc1_variance_ratio(m)
        })

    # Return tidy DataFrame indexed by epsilon
    return pd.DataFrame(records).set_index("eps")


# ---------------------------------------------------------------------------
# 4: PC1 loading stability visualization across epsilon values
# ---------------------------------------------------------------------------
def plot_pc1_loading_stability(
    clr_data: Mapping[float, pd.DataFrame],
    chosen_eps: float | None = None,
    top_k: int = 10,
    figsize: tuple[float, float] = (12, 8),
) -> dict:
    """
    Plot PC1 loading stability across epsilon values.

    This visualization shows how the top-K high-variance features behave
    in the first principal component as epsilon varies. It highlights:

    - Feature trajectories (PC1 loadings vs epsilon)
    - Optional vertical marker for a chosen epsilon
    - A legend entry indicating the chosen epsilon
    - Log-scaled epsilon axis for clarity

    Parameters
  ----
    clr_data : Mapping[float, pd.DataFrame]
        Dictionary mapping epsilon → CLR DataFrame.
    chosen_eps : float or None
        Epsilon to highlight with a vertical line.
    top_k : int
        Number of high-variance features to track.
    figsize : tuple
        Figure size.

    Returns
    -------
    dict
        {
            "figure": matplotlib Figure,
            "data": DataFrame of aligned PC1 loadings
        }

    Notes
    -----
    The figure includes a caption positioned below the axes bounding box.
    When saving, use `bbox_inches='tight'` to ensure the caption is included:

        result = plot_pc1_loading_stability(...)
        result["figure"].savefig("path.png", bbox_inches="tight")
    """
    # ------------------------------------------------------------------
    # 4-1. Compute PC1 loadings for top-K features across all epsilons
    # ------------------------------------------------------------------
    df_loadings = _compute_pc1_stability(clr_data, chosen_eps, top_k=top_k)
    eps_values  = sorted(clr_data)

    # ------------------------------------------------------------------
    # 4-2. Setup figure and color palette
    # ------------------------------------------------------------------
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Distinct colors for each feature trajectory
    colors = sns.color_palette("husl", n_colors=len(df_loadings.columns))
    # ------------------------------------------------------------------
    # 4-3. Plot PC1 loadings for each feature across epsilons
    # ------------------------------------------------------------------
    for i, feature in enumerate(df_loadings.columns):
        ax.plot(
            df_loadings.index,
            df_loadings[feature],
            marker="o",
            markersize=4,
            linewidth=1.5,
            alpha=0.8,
            color=colors[i],
            label=feature,
            markeredgecolor="white",
            markeredgewidth=0.5
        )

    # ------------------------------------------------------------------
    # 4-4. Highlight chosen epsilon with a vertical line and add to legend
    # ------------------------------------------------------------------
    if chosen_eps is not None:
        ax.axvline(
            chosen_eps, linestyle="--", color="#2d3436",
            linewidth=1.2, alpha=0.6
        )
        ax.axvspan(
            chosen_eps * 0.9, chosen_eps * 1.1,
            color="gray", alpha=0.05
        )
        eps_proxy = Line2D(
            [0], [0],
            color="#2d3436",
            linestyle="--",
            linewidth=1.2,
            label=f"Target ε: {chosen_eps:.6f}"
        )
        handles, _ = ax.get_legend_handles_labels()
        handles.append(eps_proxy)
    else:
        handles, _ = ax.get_legend_handles_labels()

    # ------------------------------------------------------------------
    # 4-5. Formatting and styling
    # ------------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (ε) Pseudocount", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("PC1 Loading Value", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Stability of Top {len(df_loadings.columns)} High-Variance Features",
        fontsize=16, loc="left", pad=20, fontweight="bold"
    )
    # Grid and spines
    sns.despine(trim=True, offset=10)
    ax.grid(True, axis="y", color="#f0f0f0", linestyle="-", zorder=0)
    ax.axhline(0, color="#d1d1d1", linewidth=1.0, zorder=1)

    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=12, colors="#2d3436")
    ax.tick_params(axis="x", which="minor", bottom=False)

    # ------------------------------------------------------------------
    # 4-6. Legend formatting
    # ------------------------------------------------------------------
    leg = ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=10,
        title="Features & Settings"
    )
    plt.setp(leg.get_title(), fontsize=12, fontweight="bold")

    # ------------------------------------------------------------------
    # 4-7. Informative caption about the feature selection method
    # ------------------------------------------------------------------
    fig.text(
        0.05, -0.05,
        f"Features selected by variance at ε={chosen_eps if chosen_eps is not None else eps_values[0]:.6f}.",
        fontsize=9, color="#636e72", style="italic"
    )
    # Final layout adjustments
    fig.tight_layout()

    return {"figure": fig, "data": df_loadings}

# ---------------------------------------------------------------------------
# Helper functions 4-A: for PC1 stability computation
# ---------------------------------------------------------------------------
def _filter_by_variance(clr_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Select the top-K highest-variance CLR features.

    This step ensures that PCA focuses on the features that actually move
    across samples, rather than low-variance or static components.

    Parameters
  ----
    clr_df : pd.DataFrame
        CLR-transformed feature table.
    top_k : int
        Number of highest-variance features to retain.

    Returns
  -
    pd.DataFrame
        Filtered DataFrame containing only the top-K movers.
    """
    # Compute per-feature variance and sort descending
    variances = clr_df.var().sort_values(ascending=False)

    # Select the top-K feature names
    keep_cols = variances.head(top_k).index

    # Return filtered CLR matrix
    return clr_df[keep_cols]

# ---------------------------------------------------------------------------
# Helper functions 5-B: for computing and aligning PC1 loadings across epsilons
# ---------------------------------------------------------------------------
def _compute_pc1_stability(
    clr_data: Mapping[float, pd.DataFrame],
    chosen_eps: float | None = None,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Compute aligned PC1 loadings across epsilon values.

    Steps:
    1. Identify the top-K high-variance features at a reference epsilon.
    2. Compute PC1 loadings for these features at every epsilon.
    3. Align the sign of PC1 across epsilons using a stable anchor feature.

    Parameters
  ----
    clr_data : Mapping[float, pd.DataFrame]
        Dictionary mapping epsilon → CLR DataFrame.
    chosen_eps : float or None
        Reference epsilon for selecting high-variance features.
        If None, the smallest epsilon is used.
    top_k : int
        Number of high-variance features to track.

    Returns
  -
    pd.DataFrame
        Rows = epsilon values, columns = selected features,
        values = aligned PC1 loadings.
    """
    eps_values = sorted(clr_data)

    # Reference epsilon for selecting movers
    ref_eps = chosen_eps if chosen_eps is not None else eps_values[0]

    # ------------------------------------------------------------------
    # 4-B-1. Identify movers at the reference epsilon
    # ------------------------------------------------------------------
    ref_df_filtered = _filter_by_variance(clr_data[ref_eps], top_k=top_k)
    target_features = ref_df_filtered.columns

    # ------------------------------------------------------------------
    # 4-B-2. Compute reference PC1 to establish orientation anchor
    # ------------------------------------------------------------------
    ref_X        = ref_df_filtered.values.astype(float)
    ref_Xc       = ref_X - ref_X.mean(axis=0)
    _, _, Vt_ref = np.linalg.svd(ref_Xc, full_matrices=False)

    # Anchor = feature with the largest absolute loading in PC1
    anchor_idx = np.argmax(np.abs(Vt_ref[0]))
    # Note: The anchor feature is selected based on the largest absolute loading in the reference PC1. 
    # This ensures that the sign alignment is based on the most influential feature, providing a stable 
    # reference point for comparing PC1 loadings across different epsilon values. By aligning the sign 
    # of the PC1 loadings to this anchor, we can meaningfully interpret the trajectories of the selected 
    # features across epsilon thresholds, as they will be oriented in a consistent manner relative to the 
    # most significant contributor to the variance in the reference dataset.
    pc1_rows = []

    # ------------------------------------------------------------------
    # 4-B-3. Compute aligned PC1 loadings for each epsilon
    # ------------------------------------------------------------------
    for eps in eps_values:
        # Use the same feature set across all epsilons
        X = clr_data[eps][target_features].values.astype(float)
        # Center the data before SVD
        Xc = X - X.mean(axis=0)

        # SVD -> PC1 loadings: Vt[0] is the first right singular vector (PC1 loadings)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Align the sign of PC1 loadings to the reference anchor feature
        aligned_pc1 = Vt[0] * np.sign(Vt[0][anchor_idx])
        pc1_rows.append(aligned_pc1)

    return pd.DataFrame(pc1_rows, index=eps_values, columns=target_features)


# ---------------------------------------------------------------------------
# 5: Crime count visualization with a modern data-journalism style
# ---------------------------------------------------------------------------
def plot_crime_counts(
    esp_results: dict,
    column: str,
    figsize: Tuple[int, int] = (12, 10),
    bins: int = 35,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Plot monthly crime counts using a modern “data-journalism” visual style.

    This function produces a two-panel figure:
    1. **Trend Plot (Top)** — A line chart showing the monthly trajectory of a
       selected crime category, with a status-bar annotation summarizing:
       - number of zero-count months
       - total time span

    2. **Distribution Plot (Bottom)** — A histogram (with KDE overlay) of all
       non-zero monthly counts, including a compact statistical summary:
       - mean
       - median
       - maximum observed count

    The design emphasizes:
    - consistent typography across panels
    - aligned “status bar” annotations
    - clean, minimalistic styling suitable for reports or journalism-style graphics

    Parameters
    ----------
    esp_results : dict
        Dictionary containing processed ESP output. Must include a key
        `"pivot_data"` mapping to a DataFrame indexed by dates.
    column : str
        Name of the crime category (column in `pivot_data`) to visualize.
    figsize : tuple of int
        Size of the overall figure (width, height).
    bins : int
        Maximum number of histogram bins for the distribution plot.
    show : bool
        Whether to display the figure immediately.

    Returns
    -------
        dict
            {
                "series": pd.Series of monthly counts indexed 0..N-1.
                        The original date index is used only for x-axis
                        tick labels and is not preserved in the returned series.,
                "fig": matplotlib Figure object
            }
    """
    # ------------------------------------------------------------
    # Robust Data Extraction
    # ------------------------------------------------------------
    pivot = esp_results.get("pivot_data")
    if not isinstance(pivot, pd.DataFrame):
        raise TypeError("`esp_results['pivot_data']` must be a pandas DataFrame.")
    if column not in pivot.columns:
        raise KeyError(f"Column '{column}' not found in pivot_data.")

    # Extract the time series and preserve original date index
    series_raw = pd.Series(pivot[column])
    date_index = series_raw.index
    series = series_raw.reset_index(drop=True)

    # Non-zero values for distribution analysis
    nonzero = series[series > 0]

    # ------------------------------------------------------------
    # Theme Configuration (consistent across both subplots)
    # ------------------------------------------------------------
    sns.set_theme(style="white", context="paper")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={'height_ratios': [1.2, 1]}
    )

    PRIMARY    = "#1A237E"  # Deep navy for main line
    ACCENT     = "#00BFA5"  # Emerald for histogram
    TITLE_SIZE = 16
    LABEL_SIZE = 9

    # ------------------------------------------------------------
    # 5-A. TOP PLOT — Monthly Trend Line
    # ------------------------------------------------------------
    sns.lineplot(
        x=series.index, y=series.values,
        ax=ax1, color=PRIMARY, linewidth=2
    )

    # Soft fill under the line for visual depth
    ax1.fill_between(series.index, series.values, color=PRIMARY, alpha=0.04)

    # Title and axis labels
    ax1.set_title(
        column.upper(), loc='left',
        fontsize=TITLE_SIZE, fontweight='800',
        pad=30, color="#111111"
    )
    ax1.set_ylabel("INCIDENTS", fontsize=8, fontweight='700',
                   labelpad=12, color="#9E9E9E")
    ax1.set_xlabel("")
    ax1.set_xlim(0, len(series) - 1)

    # Year ticks (approx. 6 evenly spaced)
    tick_pos = np.linspace(0, len(series) - 1, 6, dtype=int)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(
        [
            date_index[i].strftime('%Y')
            if hasattr(date_index[i], 'strftime')
            else str(date_index[i])[:4]
            for i in tick_pos
        ],
        fontsize=9, color="#757575"
    )

    # Status-bar annotation (aligned right)
    top_info = (
        f"ZERO MONTHS: {len(series) - len(nonzero)}   •   "
        f"TOTAL SPAN: {len(series)}"
    )
    ax1.text(
        1.0, 1.05, top_info,
        transform=ax1.transAxes,
        ha='right', va='bottom',
        fontsize=LABEL_SIZE, fontweight='bold',
        color="#757575", fontfamily='monospace'
    )

    # ------------------------------------------------------------
    # 5-B: BOTTOM PLOT — Distribution of Non-Zero Counts
    # ------------------------------------------------------------
    if not nonzero.empty:
        # Smart binning for small integer ranges
        actual_bins = (
            min(bins, int(nonzero.max() - nonzero.min()) + 1)
            if nonzero.max() < 10 else bins
        )

        sns.histplot(
            nonzero, bins=actual_bins, kde=True,
            ax=ax2, color=ACCENT, edgecolor="white", alpha=0.5
        )

        # Style the KDE line if present
        if ax2.lines:
            ax2.lines[0].set_color(PRIMARY)
            ax2.lines[0].set_linewidth(1.5)

        ax2.set_title(
            "OBSERVED INTENSITY", loc='left',
            fontsize=TITLE_SIZE - 2, fontweight='800',
            pad=30, color="#111111"
        )
        ax2.set_xlabel(
            "COUNT PER MONTH",
            fontsize=8, fontweight='700',
            labelpad=10, color="#9E9E9E"
        )

        # Status-bar annotation (mirrors top plot)
        stats_text = (
            f"AVG {nonzero.mean():.1f}   •   "
            f"MED {nonzero.median():.1f}   •   "
            f"MAX {nonzero.max():.0f}"
        )
        ax2.text(
            1.0, 1.05, stats_text,
            transform=ax2.transAxes,
            ha='right', va='bottom',
            fontsize=LABEL_SIZE, fontfamily='monospace',
            color="#757575", fontweight='bold'
        )
    else:
        # If all values are zero
        ax2.text(
            0.5, 0.5, "NO DATA RECORDED",
            ha='center', va='center',
            color="#999999", fontweight='bold'
        )

    # ------------------------------------------------------------
    # 5-C: Final Aesthetic Polish
    # ------------------------------------------------------------
    sns.despine(offset=20, trim=True)
    ax1.grid(axis='y', color="#EEEEEE", linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', color="#EEEEEE", linestyle='-', linewidth=0.5)

    plt.tight_layout(pad=5.0)

    if show:
        plt.show()

    return {"series": series, "fig": fig}


# ---------------------------------------------------------------------------
# 6: Structural break visualization with deep-contrast era shading and delta annotation
# ---------------------------------------------------------------------------
def plot_structural_break(
    clr_df: pd.DataFrame,
    category: str,
    break_date: str,
    window: int = 12,
    za_stat: Optional[float] = None,
    za_p_adj: Optional[float] = None,
    era_boundaries: Optional[Dict[str, str]] = None,
    image_path: Optional[str] = None,
    image_name: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Visualize a structural break in a CLR‑transformed time series using a
    high‑contrast, publication‑grade layout.

    This function produces a two‑panel diagnostic figure:
      - A time‑series panel showing the raw CLR signal, rolling mean,
        pre‑ and post‑break means, the structural break point, and
        optional era shading (Pre‑COVID, COVID, Post‑COVID).
      - A density comparison panel contrasting pre‑break and post‑break
        CLR distributions.

    The visualization is optimized for clarity, interpretability, and
    presentation quality. It highlights regime shifts, annotates the
    magnitude of the mean shift, and optionally embeds Zivot–Andrews
    test statistics and adjusted p‑values.

    Parameters
    ----------
    clr_df : pd.DataFrame
        CLR‑transformed dataset indexed by datetime, with columns
        representing crime categories.
    category : str
        The category/column to visualize.
    break_date : str
        The structural break date (YYYY‑MM or full timestamp). Must fall
        within the series timeline.
    window : int, default 12
        Rolling‑mean window size for smoothing the CLR signal.
    za_stat : float, optional
        Zivot–Andrews test statistic for the break point.
    za_p_adj : float, optional
        Adjusted p‑value for the Zivot–Andrews test.
    era_boundaries : dict, optional
        Dictionary with keys {'Pre-COVID', 'COVID'} mapping to boundary
        dates. Used to shade eras in the time‑series panel.
    image_path : str, optional
        Directory path for saving the figure.
    image_name : str, optional
        File name (without extension) for saving the figure.
    verbose : bool, default True
        If True, prints confirmation when the figure is saved.

    Returns
    -------
    dict
        A dictionary containing:
            'fig'           : The matplotlib Figure object.
            'ax_time_series': The main time‑series axis.
            'ax_density'    : The density comparison axis.

    Notes
    -----
    - The function automatically converts the index to a DatetimeIndex
      if needed.
    - Era shading is optional and only applied when boundaries are
      provided.
    - The function is designed for high‑resolution export and
      presentation‑quality output.
    """
    # ------------------------------------------------------------
    # 6-1: Slicing & Metric Prep
    # ------------------------------------------------------------
    series = clr_df[category].dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    # Validate break_date is within the series timeline   
    idx = series.index
    idx_min, idx_max = idx.min(), idx.max()
    
    # Convert break_date to Timestamp for comparison
    t_break = pd.to_datetime(break_date)
    if t_break < idx_min or t_break > idx_max:
        raise ValueError(f"Break date {break_date} falls outside the dataset timeline.")

    # Split the series into pre- and post-break segments
    pre_break = series[idx < t_break]
    post_break = series[idx >= t_break]
    
    # Calculate means and rolling mean for the main panel
    mean_pre, mean_post = pre_break.mean(), post_break.mean()
    rolling_smoothed = series.rolling(window=window, center=True).mean()
    delta_shift = mean_post - mean_pre

    # --------------------------------------------
    # 6-2: Set up the figure and axes with a clean, journalistic style
    # --------------------------------------------
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, 
        figsize=(19, 6.5), 
        gridspec_kw={'width_ratios': [2.4, 1]},
        facecolor="white"
    )

    # Core palette mappings
    C_RAW, C_TREND = '#707B7C', '#0F2C59'
    C_PRE, C_POST, C_BREAK_LINE = '#0B4619', '#7B1113', '#801515'

    # --------------------------------------------
    # 6-3: Panel 1: Time Series with Rolling Mean, Break Point, and Era Shading
    # --------------------------------------------
    ax1.plot(idx, series.values, color=C_RAW, alpha=0.45, lw=0.8, label="Raw CLR Signal", zorder=2)
    ax1.plot(idx, rolling_smoothed.values, color=C_TREND, lw=2.7, label=f"{window}m Rolling Mean", zorder=4)
    
    # High-contrast horizontal lines for pre/post means and vertical line for break point
    ax1.hlines(mean_pre, idx_min, t_break, colors=C_PRE, linestyles='-', lw=2.5, label="Pre-Break Mean", zorder=3)
    ax1.hlines(mean_post, t_break, idx_max, colors=C_POST, linestyles='-', lw=2.5, label="Post-Break Mean", zorder=3)
    ax1.axvline(t_break, color=C_BREAK_LINE, linestyle='-', lw=1.8, label="ZA Break Point", zorder=4)

    # --------------------------------------------
    # 6-4: Era shading with high-contrast labels 
    #       centered within each era span
    # --------------------------------------------
    if era_boundaries:
        t_pre_end = max(idx_min, min(pd.to_datetime(era_boundaries.get('Pre-COVID', '2020-02')), idx_max))
        t_covid_end = max(t_pre_end, min(pd.to_datetime(era_boundaries.get('COVID', '2022-12')), idx_max))
        
        # Define eras with their respective colors for shading
        eras_to_plot = [
            ("Pre-COVID", idx_min, t_pre_end, "#BCD2EE"),
            ("COVID", t_pre_end, t_covid_end, "#F8CECC"),
            ("Post-COVID", t_covid_end, idx_max, "#D5E8D4")
        ]
        
        # Use the x-axis transform to position era labels in data coordinates but aligned with the x-axis
        xaxis_transform = ax1.get_xaxis_transform()
        for name, start, end, color in eras_to_plot:
            if start < end:
                ax1.axvspan(start, end, color=color, alpha=0.95, zorder=1)
                mid_point = start + (end - start) / 2
                ax1.text(mid_point, 0.92, name.upper(), fontsize=10.5, 
                         color='#1C2833', fontweight='bold', ha='center', va='center',
                         transform=xaxis_transform, zorder=5)

    # Add the break date label above the vertical line with high contrast and bold styling
    ax1.text(t_break, 1.01, f"{break_date}", color=C_BREAK_LINE, fontsize=9.5, 
             fontweight='bold', ha='center', transform=ax1.get_xaxis_transform(), zorder=5)

    # Annotate the delta shift with an arrow pointing to the post-break mean, 
    # using a try-except block to handle potential issues with missing data
    try:
        target_y = rolling_smoothed.dropna().loc[rolling_smoothed.dropna().index >= t_break].values[0]
        ax1.annotate(
            f"Δ Shift: {delta_shift:.3f}",
            xy=(t_break, target_y),
            xytext=(-75, 0),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="-|>", color=C_BREAK_LINE, lw=1.2),
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=C_BREAK_LINE, alpha=0.95, lw=0.8),
            color=C_BREAK_LINE, fontsize=10, fontweight='bold', va='center', zorder=6
        )
    except Exception:
        pass

    # ---------------------------------------------
    # 6-5: Styling for axes, grid, ticks, and legend
    # ---------------------------------------------
    fig.suptitle(category.upper(), fontsize=16, fontweight='bold', x=0.06, ha='left', y=0.96)
    ax1.set_ylabel("CLR INTENSITY", fontsize=10, fontweight='bold', color='#333333')
    ax1.set_xlabel("YEAR-MONTH", fontsize=10, fontweight='bold', color='#333333')
    
    # Custom grid lines with a subtle, dotted style and a light gray color for a clean look
    ax1.grid(True, linestyle=':', alpha=0.6, color='#95A5A6', zorder=1.5)
    
    # Remove top and right spines for a cleaner look, and set custom ticks with formatted labels
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Set custom x-ticks to show every 3 months with formatted labels, ensuring 
    # they are spaced appropriately for readability
    tick_indices = np.linspace(0, len(idx) - 1, 9, dtype=int)
    ax1.set_xticks(idx[tick_indices])
    ax1.set_xticklabels([d.strftime('%Y-%m') for d in idx[tick_indices]], fontsize=8.5, color='#333333')

    # ----------------------------------------------------
    # 6-6: Panel 2: Density plots for pre- and post-break 
    #       with high-contrast styling
    # -----------------------------------------------------
    sns.kdeplot(pre_break, ax=ax2, color=C_PRE, fill=True, alpha=0.15, lw=2, label="Pre-Break")
    sns.kdeplot(post_break, ax=ax2, color=C_POST, fill=True, alpha=0.15, lw=2, label="Post-Break")
    
    # Set high-contrast titles and labels for the density panel, with a clean grid for reference
    ax2.set_title("DENSITY PROFILE", fontweight='bold', fontsize=11, loc='left', pad=12)
    ax2.set_xlabel("CLR Value Space", fontsize=9, fontweight='bold')
    ax2.set_ylabel("")
    ax2.grid(True, linestyle=':', alpha=0.4, color='#BDC3C7')

    # ------------------------------------------------------
    # 6-7: Legend handling for both panels with a clean, 
    #      non-overlapping layout
    # ------------------------------------------------------
    by_label_ax1 = dict(zip(*reversed(ax1.get_legend_handles_labels())))
    ax1.legend(
        by_label_ax1.values(), by_label_ax1.keys(),
        bbox_to_anchor=(1.02, 1.0), loc='upper left', 
        frameon=True, facecolor='white', edgecolor='#E0E0E0', fontsize=9.5
    )
    
    # For the density plot, we ensure the legend is placed in a way that 
    # does not overlap with the main time series panel, using a similar 
    # approach to extract unique labels and handles.
    by_label_ax2 = dict(zip(*reversed(ax2.get_legend_handles_labels())))
    ax2.legend(
        by_label_ax2.values(), by_label_ax2.keys(),
        bbox_to_anchor=(1.1, 1.10), loc='upper left', 
        frameon=False, fontsize=9.5
    )

    # ------------------------------------------------------
    # 6-8: Annotate the structural break insights in a 
    #      high-contrast text box
    # ------------------------------------------------------
    status_text = "SIGNIFICANT SHIFT" if (za_p_adj is not None and za_p_adj <= 0.05) else "STABLE / INSIGNIFICANT"
    stat_box_str = (
        f": STRUCTURAL BREAK INSIGHTS :\n"
        f"─────────────────────────────\n"
        f"• Break Identified  : {break_date}\n"
        f"• ZA Test Statistic : {f'{za_stat:.3f}' if za_stat is not None else 'N/A'}\n"
        f"• Adjusted p-value  : {f'{za_p_adj:.4f}' if za_p_adj is not None else 'N/A'}\n"
        f"• Regime Status     : {status_text}"
        f"\n\n{'::: SEGMENT STATISTICS :::':^48}\n"
        f"{'──────────────────────────':^49}\n" 
        f"              {'count':>6} {'mean':>6} {'std':>5} {'min':>6} {'max':>6}\n"
        f"PRE-BREAK   {len(pre_break):>8.1f} {mean_pre:>6.2f} {pre_break.std():>6.2f} {pre_break.min():>6.2f} {pre_break.max():>6.2f}\n"
        f"POST-BREAK  {len(post_break):>8.1f} {mean_post:>6.2f} {post_break.std():>6.2f} {post_break.min():>6.2f} {post_break.max():>6.2f}\n\n"
        f"  ▶ MEAN SHIFT (Δ)  : {delta_shift:.4f}"
    )
    
    # Place the annotation box in the lower right corner of the figure, ensuring 
    # it does not overlap with the main panels
    fig.text(0.725, 0.38, stat_box_str, fontsize=9, fontfamily='monospace', 
             color='#1A1A1A', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#E0E0E0', alpha=0.95, lw=0.8))

    # Final layout adjustments to ensure a polished presentation and optimal spacing
    plt.subplots_adjust(left=0.06, right=0.71, top=0.86, bottom=0.12, wspace=0.45)

    # ------------------------------------------------------
    # 6-9: Optional saving of the figure with 
    #      performance-conscious settings
    # ------------------------------------------------------
    if image_path and image_name:
        os.makedirs(image_path, exist_ok=True)
        export_target = os.path.join(image_path, f"{image_name}.png")
        plt.savefig(export_target, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✔ Chart saved successfully to target: {export_target}")

    plt.show()

    return {
        'fig': fig,
        'ax_time_series': ax1,
        'ax_density': ax2
    }


# ---------------------------------------------------------------------------
# 7: Era-level CLR heatmap visualization with regime-shift deltas
# ---------------------------------------------------------------------------
def plot_era_heatmaps(data_dict, stat_type='mean', image_path=None, image_name=None, verbose=False):
    """
        Plot era-level CLR heatmaps and regime-shift deltas in a four-panel layout.

        The figure includes:
        1. **Per-Era Panel** — CLR statistics across the three eras
        (Pre-COVID, COVID, Post-COVID)
        2. **Pre → COVID Delta Panel** — Shift between the first two eras
        3. **COVID → Post Delta Panel** — Shift between the last two eras
        4. **Pre → Post Delta Panel** — Net shift across the full period

        Crime categories are sorted by the magnitude of the initial COVID
        disruption (|COVID − Pre-COVID|) for visual prominence.

        Parameters
        ----------
        data_dict : dict
            Must contain:
            - 'era_means' or 'era_stds' (selected by `stat_type`): DataFrame indexed
            by crime category with era-level statistics in columns. Columns must
            include 'Pre-COVID', 'COVID', 'Post-COVID' and the three pairwise
            delta columns ('COVID_minus_Pre-COVID', etc.).
            - 'era_date_ranges' (optional): dict mapping era names to (start, end)
            date pairs. Used for the figure title.
        stat_type : {'mean', 'std'}, default 'mean'
            Which statistic to plot. Selects `era_means` or `era_stds` from
            `data_dict` and labels the figure accordingly.
        image_path : str, optional
            Directory path prefix for the saved figure. Concatenated directly with
            `image_name`, so should end with the appropriate path separator.
        image_name : str, optional
            Filename stem for the saved figure. The final filename is
            f"{image_name}_{stat_type}.png". Both `image_path` and `image_name`
            must be set for saving to occur.
        verbose : bool, default False
            Whether to print the sorted DataFrame to stdout after plotting.

        Returns
        -------
        None
            Displays the figure and optionally saves it.
    """
    # ------------------------------------------------------------
    # 7-1: Data validation and preparation
    # ------------------------------------------------------------
    stat_type_lower = stat_type.lower()
    if stat_type_lower not in ('mean', 'std'):
        raise ValueError(
            f"stat_type must be 'mean' or 'std', got '{stat_type}'"
        )

    is_mean = stat_type_lower == 'mean'
    df_key = 'era_means' if is_mean else 'era_stds'
    
    # Robust check for required DataFrame in data_dict
    if df_key not in data_dict:
        raise KeyError(f"'{df_key}' not found in data_dict.")

    # Deep copy to avoid mutating original data
    df = data_dict[df_key].copy()
    
    # Extract date range for title annotation
    ranges = data_dict.get("era_date_ranges", {})
    all_dates = [d for pair in ranges.values() for d in pair]
    date_range_str = f"{min(all_dates)} to {max(all_dates)}" if all_dates else "N/A"

    # Define metric names and labels based on the statistic type
    metric_name = "Mean CLR" if is_mean else "CLR Volatility"
    delta_label = r"$\mathbf{\Delta}$ CLR"
    
    # ------------------------------------------------------------
    # 7-2: Sort by the magnitude of the initial COVID disruption 
    #      if available 
    # ------------------------------------------------------------
    sort_col = 'COVID_minus_Pre-COVID'
    if sort_col in df.columns:
        df = df.reindex(df[sort_col].abs().sort_values(ascending=False).index)

    # ------------------------------------------------------------
    # 7-3: Set up the figure and panel layout
    # ------------------------------------------------------------
    sns.set_theme(style="white")
    fig, axes = plt.subplots(
        1, 4, figsize=(26, max(5, len(df) * 0.40)),
        gridspec_kw={'width_ratios': [3, 1, 1, 1], 'wspace': 0.15},
        constrained_layout=True
    )

    # Define the 4 panels: (Columns, Title, Cmap, IsDelta)
    panels = [
        (['Pre-COVID', 'COVID', 'Post-COVID'], f"{metric_name} per Era", 'RdBu_r' if is_mean else 'YlOrRd', False),
        (['COVID_minus_Pre-COVID'], "Shift:\nPre → COVID", 'RdBu_r', True),
        (['Post-COVID_minus_COVID'], "Shift:\nCOVID → Post", 'RdBu_r', True),
        (['Post-COVID_minus_Pre-COVID'], "Shift:\nPre → Post", 'RdBu_r', True)
    ]

    # -------------------------------------------------------------
    # 7-4: Plot each panel with consistent styling and annotations
    # -------------------------------------------------------------
    for i, (cols, title, cmap, is_delta) in enumerate(panels):
        ax = axes[i]
        label = delta_label if is_delta else metric_name
        
        sns.heatmap(
            df[cols], ax=ax, cmap=cmap, annot=True, fmt=".2f",
            center=0 if (is_delta or is_mean) else None,
            linewidths=0.5, linecolor="white",
            cbar_kws={'label': label, 'shrink': 0.7},
            annot_kws={'size': 12}
        )

        # Style titles and axes
        ax.set_title(title, fontsize=15, fontweight='bold', pad=18)
        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis='x', labelsize=13)
        
        # Style the colorbar label
        ax.collections[0].colorbar.set_label(label, fontsize=13, fontweight='bold')

        # Hide y-axis labels for all but the first panel
        if i == 0:
            ax.tick_params(axis='y', labelsize=13)
        else:
            ax.set_yticks([])

    # ------------------------------------------------------------
    # 7-5: Final layout adjustments and optional saving
    # ------------------------------------------------------------
    plt.suptitle(
        f"{metric_name} Analysis - Chicago Crime ({date_range_str})\n"
        f"Sorted by Magnitude of Initial COVID Disruption",
        fontsize=22, fontweight='bold', y=1.08
    )

    # Save the figure if path and name are provided
    if image_path and image_name:
        filename = f"{image_name}_{stat_type}.png"
        plt.savefig(os.path.join(image_path, filename), dpi=300, bbox_inches='tight')

    plt.show()

    # ------------------------------------------------------------
    # 7-6: Verbose terminal output of the DataFrame for display purposes
    # ------------------------------------------------------------
    if verbose:
        line = "=" * 100
        print(f"\n{line}\n📊 {metric_name.upper()} DATA ANALYSIS ({date_range_str})\n{line}")
        print(df.round(3).to_string())


# ---------------------------------------------------------------------------
# 8: Function to plot PC scores over time with era shading
# ---------------------------------------------------------------------------
def plot_pc_scores_over_time(
    pca_dict: dict,
    data_index: pd.DatetimeIndex,
    era_config: dict = None,
    n_components: int = 3,
    figsize: tuple = (14, 12),
    show_rolling: bool = True,
    rolling_window: int = 6,
    title_suffix: str = "",
    **kwargs
) -> dict:
    """
    Plot PCA component scores over time with shaded eras and rolling trends.
    """
    # ------------------------------------------------------------
    # 8-1: Unified configuration fallback
    # ------------------------------------------------------------
    era_config = era_config or kwargs.get('era_boundaries') or {
        'Pre-COVID' : ('2001-01-01', '2020-03-01', 'blue'),
        'COVID'     : ('2020-03-01', '2023-01-01', 'red'),
        'Post-COVID': ('2023-01-01', '2025-12-01', 'green'),
    }

    # ------------------------------------------------------------
    # 8-2: Extract PCA data and validate dimensions
    # ------------------------------------------------------------
    coords, ratios = pca_dict['coords_norm'], pca_dict['variance_ratio']
    n_components = min(n_components, coords.shape[1])
    
    # Validate that data_index length matches the number of observations in coords
    if len(data_index) != coords.shape[0]:
        raise ValueError(f"data_index length ({len(data_index)}) != observations ({coords.shape[0]})")

    # ------------------------------------------------------------
    # 8-3: Prepare the scores DataFrame for plotting
    # ------------------------------------------------------------
    scores = pd.DataFrame(
        coords[:, :n_components],
        index=data_index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    ).sort_index()

    # ------------------------------------------------------------
    # 8-4: Pre-process era configurations for efficient plotting
    # ------------------------------------------------------------
    processed_eras = []
    # Ensure eras are sorted by their start date for correct layering
    sorted_keys = sorted(era_config.keys(), key=lambda k: pd.Timestamp(era_config[k][0]))
    # Determine the overall time span of the data for proper handling of open-ended eras
    t_min, t_max = scores.index.min(), scores.index.max()

    # Process each era configuration into a standardized format for plotting
    for idx, key in enumerate(sorted_keys):
        start, end, color_name = era_config[key]
        t_start = t_min if idx == 0 else pd.Timestamp(start)
        t_end = t_max if idx == len(sorted_keys) - 1 else pd.Timestamp(end)
        processed_eras.append((key.upper(), t_start, t_end, color_name))

    # ------------------------------------------------------------
    # 8-5: Set up the figure and axes with a clean, journalistic style
    # ------------------------------------------------------------
    sns.set_theme(style='white', font='sans-serif')
    fig, axes = plt.subplots(n_components, 1, figsize=figsize, sharex=True, gridspec_kw={'hspace': 0.85})
    axes = [axes] if n_components == 1 else list(axes)
    
    # Define a cohesive color palette for the PCs and a neutral background color for the eras
    pc_colors, bg_neutral = ['#D35400', '#2C3E50', '#27AE60', '#7F8C8D'], '#2C3E50'

    # ------------------------------------------------------------
    # 8-6: Plotting loop for each principal component with era 
    # shading and rolling trends
    # ------------------------------------------------------------
    for i, ax in enumerate(axes):
        pc = f'PC{i+1}'
        series = scores[pc]
        color = pc_colors[i % len(pc_colors)]

        # Background anchor baseline and data lines
        ax.axhline(0, color=bg_neutral, lw=0.6, alpha=0.2, zorder=1)
        ax.plot(series.index, series.values, color=color, lw=1.1, alpha=0.42, label='Monthly Noise', zorder=2)
        
        # Soft fill under the line for visual depth
        if show_rolling and len(series) >= rolling_window:
            roll = series.rolling(rolling_window, center=True).mean()
            ax.plot(roll.index, roll.values, color=color, lw=2.8, alpha=0.95, solid_capstyle='round', label=f'{rolling_window}-Month Trend', zorder=3)

        # Plot cleanly pre-calculated background eras and centered labels
        y_min, y_max = ax.get_ylim()
        text_y = y_max + ((y_max - y_min) * 0.02)
        # Iterate through the pre-processed eras to draw shaded spans and vertical lines with centered labels
        for label, t_start, t_end, color_name in processed_eras:
            ax.axvspan(t_start, t_end, color=color_name, alpha=0.15, zorder=0)
            ax.axvline(t_start, color=color_name, ls='--', lw=1.3, alpha=0.5, zorder=1)
            ax.text(t_start, text_y, label, fontsize=9, color=color_name, fontweight='bold', va='bottom', ha='center')
        
        # Terminal far-right border line cap
        ax.axvline(t_max, color=processed_eras[-1][3], ls='--', lw=1.3, alpha=0.5, zorder=1)

        # Clean Left-Aligned Subplot Titles with Deep Padding
        top_feats = pca_dict.get('pc_positive_top', {}).get(pc, [])
        title_text = f"{pc} ({ratios[i]:.1%} Explained Variance)" + (f"\n  Key Indicators: {', '.join(top_feats)}" if top_feats else "")
        ax.set_title(title_text, fontweight='bold', fontsize=11, color=bg_neutral, loc='left', pad=32, linespacing=1.4)

        # Labels, Minimalist Grid lines, and Legend placement UNDERNEATH the axis floor
        ax.set_ylabel('Component Score', fontsize=10, color=bg_neutral, fontweight='medium')
        ax.tick_params(colors='#7F8C8D', labelsize=9.5)
        ax.yaxis.grid(True, linestyle=':', alpha=0.4, color='#BDC3C7')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8.5, handletextpad=0.5)
        sns.despine(ax=ax, left=True, bottom=True)

    # Global x-axis label centered under the last subplot with deep padding
    axes[-1].set_xlabel('Timeline', fontweight='bold', fontsize=11, color=bg_neutral, labelpad=38)
    
    # ------------------------------------------------------------
    # 8-7: Final suptitle and layout adjustments for a polished presentation
    # ------------------------------------------------------------
    plt.suptitle(
        f'Principal Component Analytics Over Time{title_suffix.upper()}\n'
        f'Total System Variance Captured (PC1–PC{n_components}): {sum(ratios[:n_components]):.1%}',
        fontweight='bold', size=14, y=0.98, color=bg_neutral, ha='center'
    )
    plt.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.95, hspace=0.85)

    return {'figure': fig, 'axes': axes, 'scores_df': scores}


# ---------------------------------------------------------------------------
# 9: Optimized per-category distribution audit for CLR-transformed data with severity ranking
# ---------------------------------------------------------------------------
def check_clr_distributions(
    clr_df: pd.DataFrame,
    raw_counts: pd.DataFrame | None = None,
    epsilon: float | None = None,
    n_cols: int = 5,
    flag_skew: float = 1.5,
    flag_kurtosis: float = 5.0
) -> pd.DataFrame:
    """
    Perform an optimized per‑category distribution audit on a CLR‑transformed dataset.

    This function evaluates the distributional behavior of each crime category
    after CLR transformation, identifying categories whose distributions appear
    distorted (e.g., heavy tails, strong skewness, or abnormal shape). These
    distortions often arise from sparse raw counts interacting with the chosen
    epsilon. The function:

    - Computes skewness, excess kurtosis, Shapiro–Wilk p‑values, and zero‑rates.
    - Flags categories exceeding user‑defined skew/kurtosis thresholds.
    - Plots histograms with normal overlays, ordered worst‑first by severity.
    - Highlights flagged categories visually.
    - Returns a diagnostics DataFrame summarizing all metrics.

    Parameters
    -------
    clr_df : pd.DataFrame
        CLR‑transformed matrix (rows = months, columns = categories).
    raw_counts : pd.DataFrame or None
        Source for zero‑rates. Two accepted forms:
          - LONG filled_df with an 'is_zero_after_fill' flag and a
            'fbi_code_desc' column → zero‑rate read from the flag (authoritative).
          - WIDE count matrix aligned with clr_df → zeros detected via (== 0).
        If None, zero‑rate is NaN.
    epsilon : float or None
        Epsilon used in CLR transformation (display only).
    n_cols : int
        Number of subplot columns in the histogram grid.
    flag_skew : float
        Absolute skewness threshold for flagging.
    flag_kurtosis : float
        Excess kurtosis threshold for flagging.

    Returns
    ----
    pd.DataFrame
        Diagnostics table for each category, excluding internal cache fields.
    """

    # Extract category names and compute number of rows in the subplot grid
    cats = clr_df.columns.tolist()
    n_rows = int(np.ceil(len(cats) / n_cols))

    # ------------------------------------------------------------
    # 9-1: Compute Per‑Category Diagnostics (Single‑Pass Loop)
    # ------------------------------------------------------------
    rows = []
    # We loop through each category once, computing all diagnostics and
    # caching the raw data for plotting. This minimizes redundant passes
    # over the data and ensures efficient computation even for larger datasets.
    for c in cats:
        # Pull raw CLR values as a NumPy array for speed
        x = clr_df[c].to_numpy(dtype=float)
        x = x[~np.isnan(x)]  # remove missing months

        if len(x) == 0:
            continue  # skip empty categories

        # Basic distribution statistics
        mu, sd = float(np.mean(x)), float(np.std(x))
        skew = float(stats.skew(x))
        exkurt = float(stats.kurtosis(x))  # excess kurtosis

        # Shapiro–Wilk normality test (safe for ~300 points)
        try:
            _, sh_p = stats.shapiro(x)
        except Exception:
            sh_p = np.nan

        # Compute zero‑rate. If a LONG filled_df with the is_zero_after_fill
        # flag is provided, use it (authoritative). Otherwise fall back to
        # detecting zeros in a WIDE count matrix.
        if raw_counts is None:
            zero_rate = np.nan
        elif 'is_zero_after_fill' in raw_counts.columns:
            cat_rows = raw_counts[raw_counts['fbi_code_desc'] == c]
            n_months = len(cat_rows)
            zero_rate = (float(cat_rows['is_zero_after_fill'].sum()) / n_months * 100.0
                         if n_months > 0 else np.nan)
        else:
            zero_rate = float((raw_counts[c] == 0).values.mean()) * 100.0

        # Flag categories with problematic distribution shape
        flagged = (abs(skew) > flag_skew) or (exkurt > flag_kurtosis)

        # Store metrics + cached arrays for plotting
        rows.append({
            'category': c,
            'mean_clr': mu,
            'std_clr': sd,
            'skew': skew,
            'excess_kurtosis': exkurt,
            'shapiro_p': float(sh_p),
            'zero_rate_%': zero_rate,
            'flagged': flagged,
            '_severity': abs(skew) + max(exkurt, 0),  # ranking metric
            '_x_data': x,      # cached data for plotting
            '_xmin': x.min(),  # cached min
            '_xmax': x.max()   # cached max
        })

    # Sort categories worst‑first by severity score
    processed_records = sorted(rows, key=lambda k: k['_severity'], reverse=True)

    # --------------------------------------------
    # 9-2: Render Grid Infrastructure
    # --------------------------------------------
    plt.rcParams['font.family'] = 'sans-serif'

    # Set up the grid of subplots with a clean, journalistic style
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 2.6),
        facecolor="#FDFDFD"
    )
    axes = np.atleast_1d(axes).ravel()

    # Color palette for consistent styling
    C_HIST = '#2E4057'
    C_NORM = '#D6604F'
    C_FLAG_BG = '#FDF2F2'
    C_TEXT = '#1A1A1A'
    C_TEXT_FLAG = '#922B21'

    # Plot each category in severity order
    for idx, meta in enumerate(processed_records):
        ax = axes[idx]
        x = meta['_x_data']
        is_flagged = meta['flagged']

        # Highlight flagged categories with background color
        ax.set_facecolor(C_FLAG_BG if is_flagged else 'white')

        # Histogram
        ax.hist(
            x, bins=22, density=True,
            color=C_HIST, alpha=0.75,
            edgecolor='white', linewidth=0.5, zorder=3
        )

        # Normal overlay using cached limits
        xs = np.linspace(meta['_xmin'], meta['_xmax'], 200)
        ax.plot(xs, stats.norm.pdf(xs, meta['mean_clr'], meta['std_clr']),
                color=C_NORM, lw=1.8, zorder=4)

        # Clean axis styling
        ax.tick_params(labelsize=8, colors='#555555', length=0)
        ax.grid(axis='y', linestyle=':', color='#E0E0E0', alpha=0.6, zorder=1)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Title with skew/kurtosis summary
        title_color = C_TEXT_FLAG if is_flagged else C_TEXT
        font_w = 'bold' if is_flagged else 'normal'
        ax.set_title(
            f"{meta['category'][:20]}\nskew: {meta['skew']:.2f} | kurt: {meta['excess_kurtosis']:.1f}",
            fontsize=9, color=title_color, fontweight=font_w, pad=6
        )

    # Turn off unused subplot cells
    for ax in axes[len(processed_records):]:
        ax.axis('off')

    # --------------------------------------------
    # 9-3: Global Figure Title & Layout
    # --------------------------------------------
    eps_str = f" (ε = {epsilon:.6g})" if epsilon is not None else ""

    fig.suptitle(
        f"Per-Category CLR Structural Distributions{eps_str}\n"
        f"Solid Overlay: Normal Fit  |  Highlighted Panels: Flagged (|Skew| > {flag_skew} or Kurtosis > {flag_kurtosis})",
        fontsize=14, fontweight='bold', color='#222222', y=0.98
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.91, hspace=0.35, wspace=0.25)

    plt.show()
    plt.close(fig)

    # Remove cached plotting fields before returning diagnostics
    diag = pd.DataFrame(processed_records).drop(
        columns=['_severity', '_x_data', '_xmin', '_xmax']
    )

    return diag


# ---------------------------------------------------------------------------
# 10: Plotting PC1 coordinate trajectory with algorithmic breakpoints and era shading
# ---------------------------------------------------------------------------
def plot_pc1_regime_segmentation(
    dates,
    pc1_values,
    covid_start,
    covid_end,
    breaks_A,
    breaks_C,
    title="REGIME SEGMENTATION ANALYSIS: PC1 COORDINATE TRAJECTORY",
    figsize=(16, 5.5)
):
    """
    Plot a PC1 coordinate trajectory with macro-era shading and dynamic-programming
    structural break annotations.

    Parameters
    ----------
    dates : array-like of datetime64
        Time index corresponding to the PCA coordinates.
    
    pc1_values : array-like of float
        PC1 coordinate trajectory (e.g., results['coordinates_normalized'][:, 0]).
    
    covid_start : datetime-like
        Start date of the assumed COVID baseline shading.
    
    covid_end : datetime-like
        End date of the assumed COVID baseline shading.
    
    breaks_A : list of datetime-like
        Structural break dates for Specification A (primary model).
    
    breaks_C : list of datetime-like
        Structural break dates for Specification C (sensitivity model).
    
    title : str, optional
        Main plot title. Defaults to a descriptive regime segmentation heading.
    
    figsize : tuple, optional
        Figure size. Defaults to (16, 5.5).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    
    ax : matplotlib.axes.Axes
        The axis containing the plotted elements.
    """

    # ----------------------------------------------------------
    # 10-1: Create figure and axis with a wide layout for annotations
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # ----------------------------------------------------------
    # 10-2: Plot the PC1 trajectory line
    # ----------------------------------------------------------
    ax.plot(
        dates,
        pc1_values,
        lw=1.75,
        color='#2c3e50',
        alpha=0.85,
        label='PC1 Coordinate'
    )

    # ----------------------------------------------------------
    # 10-3: COVID baseline shading (macro-era assumption)
    # ----------------------------------------------------------
    ax.axvspan(
        covid_start,
        covid_end,
        color='#e1e8ed',
        alpha=0.4,
        label='Assumed COVID Baseline'
    )

    # Add subtle vertical boundary markers at COVID start/end
    for d in [covid_start, covid_end]:
        ax.axvline(d, color='#7f8c8d', ls=':', lw=1.2, alpha=0.7)

    # ----------------------------------------------------------
    # 10-4: Structural Breaks: Specification A (primary model)
    # ----------------------------------------------------------
    for d in breaks_A:
        ax.axvline(d, color='#e74c3c', ls='-', lw=1.5, alpha=0.9)

        # Annotate break with rotated label
        ax.text(
            d + pd.Timedelta(days=45),
            ax.get_ylim()[1] * 0.82,
            f'Dynp Spec A ({d.strftime("%Y-%m")})',
            color='#e74c3c',
            rotation=90,
            verticalalignment='center',
            fontsize=9,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )

    # ----------------------------------------------------------
    # 10-5: Structural Breaks: Specification C (sensitivity model)
    # ----------------------------------------------------------
    for d in breaks_C:
        ax.axvline(d, color='#f39c12', ls='--', lw=1.5, alpha=0.9)

        # Annotate break with rotated label
        ax.text(
            d + pd.Timedelta(days=45),
            ax.get_ylim()[1] * 0.64,
            f'Dynp Spec C ({d.strftime("%Y-%m")})',
            color='#f39c12',
            rotation=90,
            verticalalignment='center',
            fontsize=9,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )

    # ----------------------------------------------------------
    # 10-6Title and subtitle with extra padding
    # ----------------------------------------------------------
    ax.set_title(
        title + "\nComparison of Algorithmic Kernel Shifts Against Assumed Macro-Milestones",
        fontsize=12,
        fontweight='bold',
        pad=45,
        loc='left',
        color='#2c3e50'
    )

    # ----------------------------------------------------------
    # 10-7: Grid and background styling
    # ----------------------------------------------------------
    ax.grid(True, which='major', color='#f0f3f4', linestyle='-', linewidth=1)
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')

    # ----------------------------------------------------------
    # 10-8: Format the x-axis for multi-year time series
    # ----------------------------------------------------------
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#34495e')

    # ----------------------------------------------------------
    # 10-9: Clean up plot spines for a modern look
    # ----------------------------------------------------------
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#bdc3c7')

    # ----------------------------------------------------------
    # 10-10: Legend placed outside the plot area
    # ----------------------------------------------------------
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        facecolor='#ffffff',
        edgecolor='#e2e8f0',
        fontsize=9
    )

    # ----------------------------------------------------------
    # 10-11: Metadata panel placed under the legend
    # ----------------------------------------------------------
    metadata_text = (
        "PELT Exploratory Sweep:\n"
        "• Model Type: RBF Kernel\n"
        "• Constraint Penalty: BIC\n"
        "• Status: [0 Breaks Detected]\n"
        "  (Signals local mean adjustments\n"
        "   do not cross global variance threshold)"
    )

    # styled bounding box for metadata panel
    props = dict(boxstyle='round,pad=0.5', facecolor='#fafafa', edgecolor='#e2e8f0', alpha=0.9)

    fig.text(
        0.865,
        0.58,
        metadata_text,
        fontsize=8.5,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=props,
        color='#4a5568'
    )

    # ----------------------------------------------------------
    # 10-12: Adjust layout to make room for legend + metadata panel
    # ----------------------------------------------------------
    plt.subplots_adjust(right=0.85, top=0.82)
    plt.show()

    return {'fig': fig, 'axis': ax}


# ---------------------------------------------------------------------------
# 11: Plot function for regime segmentation plotting with flexible inputs
# ---------------------------------------------------------------------------
def plot_regime_segmentation(dates, pca_coords, breaks_A, breaks_C,
                             pelt_break=None,
                             target_idx_1=229, target_idx_2=263,
                             pelt_status="[0 Breaks Detected]"):
    """
    Plot the PC1 coordinate trajectory with three layers of changepoint evidence:
    Dynp Spec A breaks (red, primary 25-cat CLR), Dynp Spec C breaks
    (orange dashed, 23-cat vice-excluded CLR), and an optional Pelt break
    detected only under permissive (0.25x BIC) penalty (green dotted).
    The assumed COVID era is shaded gray for reference.

    Uses explicit Line2D and Patch handles so legend entries match the
    visual elements exactly, independent of matplotlib's automatic
    handle detection.

    Parameters
    ----------
    dates : array-like of datetime
        Monthly timestamps, length matches pca_coords.
    pca_coords : array-like
        PC1 score series.
    breaks_A : list of str or Timestamp
        Dynp breakpoints from the 25-category specification.
    breaks_C : list of str or Timestamp
        Dynp breakpoints from the 23-category (vice-excluded) specification.
    pelt_break : str or Timestamp, optional
        Single Pelt break detected at 0.25x BIC (e.g., '2015-07').
        Omitted from the figure if None.
    target_idx_1, target_idx_2 : int
        Indices into `dates` that define the assumed COVID-era bounds.
    pelt_status : str
        Status text shown in the metadata panel.
    """
    # Defensive input handling
    pca_coords = np.asarray(pca_coords).ravel()
    if len(pca_coords) != len(dates):
        raise ValueError(
            f"Length mismatch: pca_coords has {len(pca_coords)}, "
            f"dates has {len(dates)}"
        )

    # -------------------------------------------
    # 11-1: Anchor label y-positions to the data
    # -------------------------------------------
    y_low, y_high = pca_coords.min(), pca_coords.max()
    y_range = y_high - y_low
    label_y_A = y_high - 0.18 * y_range
    label_y_C = y_high - 0.36 * y_range
    label_y_P = y_high - 0.54 * y_range

    # -------------------------------------------
    # 11-2: Create figure and axis with a wide 
    #       layout for annotations
    # -------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 5.5))

    # -------------------------------------------
    # 11-3: Assumed COVID-era band
    # -------------------------------------------
    ax.axvspan(dates[target_idx_1], dates[target_idx_2],
               color='#e1e8ed', alpha=0.8)
    # Add subtle vertical boundary markers at COVID start/end
    for idx in [target_idx_1, target_idx_2]:
        ax.axvline(dates[idx], color='#7f8c8d', ls=':', lw=1.2, alpha=0.7)

    # --------------------------------------------
    # 11-4: Primary series
    # --------------------------------------------
    ax.plot(dates, pca_coords, lw=1.75, color='#2c3e50', alpha=0.9)

    # --------------------------------------------
    # 11-5:Dynp Spec A breaks (primary, 25 categories)
    # --------------------------------------------
    for d in breaks_A:
        ts = pd.Timestamp(d)
        ax.axvline(ts, color='#e74c3c', ls='-', lw=1.75, alpha=0.9)
        ax.text(ts + pd.Timedelta(days=45), label_y_A,
                f'Dynp Spec A ({ts.strftime("%Y-%m")})',
                color='#e74c3c', rotation=90, verticalalignment='center',
                fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8,
                          edgecolor='none', pad=2))

    # ---------------------------------------------
    # 11-6: Dynp Spec C breaks (sensitivity, 23 categories)
    # ---------------------------------------------
    for d in breaks_C:
        ts = pd.Timestamp(d)
        ax.axvline(ts, color='#f39c12', ls='--', lw=1.5, alpha=0.9)
        ax.text(ts + pd.Timedelta(days=45), label_y_C,
                f'Dynp Spec C ({ts.strftime("%Y-%m")})',
                color='#f39c12', rotation=90, verticalalignment='center',
                fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8,
                          edgecolor='none', pad=2))

    # ---------------------------------------------
    # 11-7: Pelt permissive-penalty break (only at 0.25x BIC)
    # ---------------------------------------------
    if pelt_break is not None:
        ts = pd.Timestamp(pelt_break)
        ax.axvline(ts, color='#27ae60', ls=':', lw=1.5, alpha=0.8)
        ax.text(ts + pd.Timedelta(days=45), label_y_P,
                f'Pelt @ 0.25×BIC ({ts.strftime("%Y-%m")})',
                color='#27ae60', rotation=90, verticalalignment='center',
                fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8,
                          edgecolor='none', pad=2))

    # ---------------------------------------------
    # 11-8: Titles, grid, axes
    # ---------------------------------------------
    ax.set_title(
        'REGIME SEGMENTATION ANALYSIS: PC1 COORDINATE TRAJECTORY\n'
        'Comparison of Algorithmic Kernel Shifts Against Assumed Macro-Milestones',
        fontsize=12, fontweight='bold', pad=20, loc='left', color='#2c3e50'
    )
    ax.grid(True, which='major', color='#f0f3f4', linestyle='-', linewidth=1)
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')

    # ---------------------------------------------
    # 11-9: Format the x-axis for multi-year time series
    # ---------------------------------------------
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#34495e')

    # ---------------------------------------------
    # 11-10: Clean up plot spines for a modern look
    # ---------------------------------------------
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#bdc3c7')

    # ---------------------------------------------
    # 11-11:Lock axes position before legend placement
    # ---------------------------------------------
    plt.subplots_adjust(left=0.06, right=0.78, top=0.88, bottom=0.10)

    # ---------------------------------------------
    # 11-12: Legend with explicit proxy handles
    # ---------------------------------------------
    line_pc1 = plt.Line2D([0], [0], color='#2c3e50', lw=1.75, alpha=0.9,
                          label='PC1 Coordinate')
    covid_patch = mpatches.Patch(facecolor='#ced6dc', edgecolor='none',
                                 label='Assumed COVID Era')
    mock_spec_A = plt.Line2D([0], [0], color='#e74c3c', ls='-', lw=1.75,
                             label='Dynp Spec A (25 cats)')
    mock_spec_C = plt.Line2D([0], [0], color='#f39c12', ls='--', lw=1.5,
                             label='Dynp Spec C (23 cats)')
    handles = [line_pc1, covid_patch, mock_spec_A, mock_spec_C]
    if pelt_break is not None:
        mock_pelt = plt.Line2D([0], [0], color='#27ae60', ls=':', lw=1.5,
                               label='Pelt @ 0.25×BIC')
        handles.append(mock_pelt)

    # ---------------------------------------------
    # 11-12: Create the legend outside the plot area
    # ---------------------------------------------
    leg = ax.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=True, facecolor='#ffffff',
        edgecolor='#e2e8f0', fontsize=9.5,
        borderpad=0.8, labelspacing=0.7,
    )

    # ---------------------------------------------
    # 11-12: Metadata panel placed under the legend 
    #        in axes coordinates
    # ---------------------------------------------
    pelt_metadata_text = (
        f"PELT Exploratory Sweep:\n"
        f"• Model Type: RBF Kernel\n"
        f"• Constraint Penalty: BIC\n"
        f"• Status: {pelt_status}\n"
        f"  (Cost reductions at candidate\n"
        f"   breakpoints do not justify\n"
        f"   the BIC penalty)"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='#fafafa',
                 edgecolor='#e2e8f0', alpha=0.95)

    # ---------------------------------------------
    # 11-12: Force a draw to ensure the legend's 
    #        bounding box is updated before we position 
    #        the metadata panel
    # ---------------------------------------------
    fig.canvas.draw()
    leg_bbox = leg.get_window_extent().transformed(ax.transAxes.inverted())

    # ----------------------------------------------
    # 11-13: Position the metadata panel just below the legend
    #        using the legend's bounding box for reference
    # ----------------------------------------------
    ax.text(
        leg_bbox.x0,
        leg_bbox.y0 - 0.03,
        pelt_metadata_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=props,
        color='#4a5568',
    )

    return {'figure': fig, 'axis': ax}