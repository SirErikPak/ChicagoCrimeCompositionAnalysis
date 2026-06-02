import pandas as pd
import pyarrow as pa         # arrow array construction + typed scalars
import pyarrow.compute as pc # C++ compute kernels

# Typed Arrow string dtype used by composite-key helpers below.
arrow_string = pd.ArrowDtype(pa.string())

# --------------------------------------------------------------------------------------------------
# 1: Identify and display columns with null values, optimized for large datasets (e.g., PyArrow-backed).
# --------------------------------------------------------------------------------------------------
def any_nans(data: pd.DataFrame) -> None:
    """
    Identifies and displays columns containing null values with their counts and percentages.
    
    Optimized for large datasets (e.g., PyArrow-backed) by using vectorized operations.
    
    Args:
        data: The pandas DataFrame to inspect.
    """
    total_rows = len(data)
    
    # 1. Get counts of nulls for all columns
    null_counts = data.isnull().sum()
    
    # 2. Filter for only columns that have at least one null
    null_counts = null_counts[null_counts > 0]
    
    if not null_counts.empty:
        print(f"--- Missing Values Found (Total Rows: {total_rows:,}) ---")
        
        # 3. Calculate percentage and build summary table
        percent = (null_counts / total_rows) * 100
        
        summary = pd.DataFrame({
            'Count': null_counts,
            'Percentage': percent.map("{:.4f}%".format)
        }).sort_values(by='Count', ascending=False)
        
        print(summary)
    else:
        print(f"Clean Dataset: No NaNs found across {total_rows:,} rows.")


# --------------------------------------------------------------------------------------------------
# 2: Convert a wide pivot table (crime types per year) into a tidy long-format DataFrame suitable 
# for plotting or statistical analysis.
# --------------------------------------------------------------------------------------------------
def melt_pivot_for_plotting(pivot_data: pd.DataFrame, id_vars: str, var_name: str):
    """
    Convert a wide pivot table (crime types per year) into a tidy long-format
    DataFrame suitable for plotting or statistical analysis.
    """
    data = (
        pivot_data
        .reset_index()               # normal column so it can be used as an identifier during melting
        .melt(                       # converts wide/long format
            id_vars=id_vars,         # Columns to keep fixed - do NOT unpivot these. multiple identifiers: List
            var_name=var_name,       # Name of the new column that will contain the former column headers
            value_name='count'       # Name of the new column that will contain the cell values.”
        )
        .fillna(0)
        .convert_dtypes(dtype_backend='numpy_nullable')  # Convert each column to the NumPy‑nullable dtype
    )
    
    return data


# --------------------------------------------------------------------------------------------------
# 3: Create a crosstab of crime counts by year and crime type, optimized for large datasets (e.g., PyArrow-backed).
# --------------------------------------------------------------------------------------------------
def make_crime_year_crosstab(data: pd.DataFrame, colA: str, rowB: str):
    """
    Create a crosstab of crime counts.
    
    Parameters
    ----------
    data : pandas.DataFrame
    
    Returns
    -------
    pandas.DataFrame
        Pivot table
    """
    # Aggregate counts
    crime = (
        data.groupby([colA, rowB], observed=False)
          .size()
          .reset_index(name='count')  # GroupBy results return a Series with a MultiIndex 
    )                                 # converts the index levels back into normal columns
    
    # Cast Arrow categoricals to str for pivot compatibility
    crime[colA] = crime[colA].astype(str)
    crime[rowB] = crime[rowB].astype(str)

    # Pivot to wide format
    # index becomes the rows of the pivot table
    # columns becomes the columns  
    # values fill the cells of the matrix
    # index X columns crosstab
    pivot = (
        crime
            .pivot(index=rowB, columns=colA, values='count')
            .sort_index()
            .fillna(0)    # any crime type NaN columns, the pivot table will have NaN in that cell
    )                     # Replace missing count values with 0 (counts-missing means zero incidents)

    return pivot