import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------------------------------------------------------------------
# 1. Bar Plot of Crime Distribution
# ---------------------------------------------------------------------
def bar_plot(
    data: pd.DataFrame,
    count: str,
    index: str,
    crime_code: str,
    column: str,
    col_wrap: int = 3,
    save_path: str = "../figures/"
) -> None:
    """
    Creates a faceted horizontal bar plot showing crime distributions across categories.

    Args:
        data (pd.DataFrame): Input dataframe containing crime statistics.
        count (str): Column representing incident counts (x-axis).
        index (str): 'I' for indexed data, anything else for non-indexed.
        crime_code (str): Column representing crime categories (y-axis and hue).
        column (str): Column used to facet the grid (e.g., district).
        col_wrap (int): Number of columns before wrapping facets.
        save_path (str): Directory path for saving the figure.

    Returns:
        None
    """

    # Determine title prefix
    head = "Indexed" if index == "I" else "Non-Indexed"

    # Build dynamic filename using actual column names + index
    if save_path:
        safe_count = count.replace(" ", "_").lower()
        safe_code = crime_code.replace(" ", "_").lower()
        safe_col = column.replace(" ", "_").lower()
        safe_index = str(index).replace(" ", "_").lower()
        filename = f"{safe_count}_{safe_code}_{safe_col}_{safe_index}_bar_plot.png"
        full_path = save_path + filename

    # Apply seaborn styling and create faceted bar plot
    with sns.axes_style("whitegrid"):
        g = sns.catplot(
            data=data,
            kind="bar",
            x=count,
            y=crime_code,
            hue=crime_code,
            col=column,
            col_wrap=col_wrap,
            height=4,
            aspect=1.5,
            palette="viridis",
            sharex=False,
            legend=False
        )

    # Add global title with reduced padding
    g.fig.suptitle(
        f"FBI {head} Crime Distribution by District (2001-2025)",
        fontsize=22,
        fontweight="bold",
        y=0.92
    )

    # Set facet titles
    g.set_titles("{col_name} District", size=16, fontweight="bold", pad=10)

    # Format each subplot
    for ax in g.axes.flatten():

        # Axis labels
        ax.set(
            xlabel="Number of Incidents",
            ylabel="FBI Crime Code"
        )

        # Rotate x‑tick labels
        ax.tick_params(axis="x", labelrotation=45)

        # Add comma‑formatted labels to bars
        for container in ax.containers:
            ax.bar_label(container, fmt="{:,.0f}", padding=4, fontsize=9)

        # Format x‑axis ticks with commas
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # Adjust layout spacing
    g.fig.subplots_adjust(hspace=0.6, top=0.88)

    # Save figure if requested
    if save_path:
        g.fig.savefig(full_path, dpi=600)

    # Display plot
    plt.show()



# ---------------------------------------------------------------------
# 2. Line Plot of Crime Trends with Peak Indicators
# ---------------------------------------------------------------------
def line_plot(
    data: pd.DataFrame,
    column_name: str,
    category_name: str,
    numeric_name: str,
    image_path: str = "../figures/",
    col_wrap: int = 4,
    rotation: int = 45,
    ha: str = "right"
) -> None:
    """
    Creates a faceted line plot for time-series crime data with peak indicators.

    Args:
        data (pd.DataFrame): Long-form dataframe containing the data.
        column_name (str): Column used to facet the grid (e.g., crime type).
        category_name (str): X-axis variable (e.g., month or quarter).
        numeric_name (str): Y-axis variable (e.g., count).
        image_path (str): Directory to save the output image.
        col_wrap (int): Number of columns before wrapping facets.
        rotation (int): Rotation angle for x-tick labels.
        ha (str): Horizontal alignment for rotated x-tick labels.

    Returns:
        None
    """

    # Create FacetGrid for line plots
    g = sns.FacetGrid(
        data,
        col=column_name,
        col_wrap=col_wrap,
        hue=column_name,
        sharey=False,
        height=4,
        aspect=1.2
    )

    # Build dynamic filename using actual column names
    if image_path is not None:
        safe_col = column_name.replace(" ", "_").lower()
        safe_cat = category_name.replace(" ", "_").lower()
        safe_num = numeric_name.replace(" ", "_").lower()
        image_name = f"{image_path}{safe_col}_{safe_cat}_{safe_num}_line_plot.png"

    # Map lineplot to each facet
    g.map(sns.lineplot, category_name, numeric_name, marker="o")

    # Group data once for efficiency
    grouped = data.groupby(column_name, sort=False)

    # Add peak indicators and formatting
    for ax, (name, facet_data) in zip(g.axes.flatten(), grouped):

        if not facet_data.empty:
            # Identify peak value
            max_idx = facet_data[numeric_name].idxmax()
            max_cat = facet_data.loc[max_idx, category_name]
            peak_val = facet_data.loc[max_idx, numeric_name]

            # Draw vertical line at peak
            ax.axvline(x=max_cat, color="black", linestyle="--", alpha=0.5, zorder=0)

            # Annotate peak value
            ax.text(
                x=max_cat,
                y=peak_val,
                s=f" {peak_val:,.0f}",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
            )

        # Ensure x-labels appear on all facets
        ax.tick_params(labelbottom=True, labelsize=9)

        # Axis labels
        ax.set_xlabel(category_name.capitalize())
        ax.set_ylabel(numeric_name.capitalize())

        # Rotate x-tick labels
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha=ha)

    # Facet titles
    g.set_titles("{col_name}")

    # Adjust spacing for readability
    plt.subplots_adjust(hspace=0.7)
    g.tight_layout()

    # Save figure if requested
    if image_path:
        plt.savefig(image_name, dpi=600)

    # Display plot
    plt.show()