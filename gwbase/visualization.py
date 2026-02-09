"""
Visualization functions for GWBASE.

This module provides plotting functions for:
- Well time series plots
- Delta metrics scatter plots
- Regression analysis visualizations
- MI comparison plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import linregress
import os
from typing import Tuple


def format_p_value(p_value: float) -> str:
    """
    Format p-value to 2 decimal places without scientific notation.
    
    Parameters
    ----------
    p_value : float
        P-value to format
        
    Returns
    -------
    str
        Formatted p-value string
    """
    if np.isnan(p_value):
        return "N/A"
    
    if p_value < 0.01:
        # For very small values, show as < 0.01
        return "< 0.01"
    else:
        # Format to 2 decimal places
        return f"{p_value:.2f}"


def _compute_and_plot_regression(x, y, linewidth=2.5, label=None, zorder=10):
    """
    Compute regression and plot regression line.
    
    Returns
    -------
    tuple: (slope, intercept, r_squared, p_value, std_err)
    """
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2
        
        # Plot regression line
        x_range = np.array([x.min(), x.max()])
        y_range = intercept + slope * x_range
        if label is None:
            label = f'Regression (R²={r_squared:.3f})'
        plt.plot(x_range, y_range, 'r-', linewidth=linewidth, label=label, zorder=zorder)
        return slope, intercept, r_squared, p_value, std_err, r_value
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def _add_stats_text(stats_text, x_pos=0.02, y_pos=0.98, fontsize=11, alpha=0.9):
    """Add statistics text box to plot."""
    plt.text(x_pos, y_pos, stats_text,
            transform=plt.gca().transAxes,
            fontsize=fontsize, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=alpha))


def plot_well_timeseries(
    data: pd.DataFrame,
    output_dir: str = 'figures/well_timeseries',
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    q_col: str = 'q',
    bfd_col: str = 'bfd',
    max_wells_per_gage: int = None,
    figsize: Tuple[int, int] = (15, 8),
    start_date: str = None,
    end_date: str = None
) -> None:
    """
    Create dual time series plots for wells (WTE and streamflow).

    Parameters
    ----------
    data : pd.DataFrame
        Paired well-streamflow data
    output_dir : str
        Directory to save plots
    well_id_col : str, default 'well_id'
        Column name for well ID
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    date_col : str, default 'date'
        Column name for date
    wte_col : str, default 'wte'
        Column name for WTE
    q_col : str, default 'q'
        Column name for streamflow
    bfd_col : str, default 'bfd'
        Column name for BFD indicator
    max_wells_per_gage : int, optional
        Maximum wells to plot per gage
    figsize : tuple, default (15, 8)
        Figure size
    start_date : str, optional
        Start date for filtering (YYYY-MM-DD)
    end_date : str, optional
        End date for filtering (YYYY-MM-DD)

    Example
    -------
    >>> plot_well_timeseries(paired_data, 'figures/timeseries')
    """
    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Apply date filters
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    # Group by gage
    for gage_id in df[gage_id_col].unique():
        gage_data = df[df[gage_id_col] == gage_id]
        wells = gage_data[well_id_col].unique()

        if max_wells_per_gage:
            wells = wells[:max_wells_per_gage]

        print(f"Processing Gage {gage_id}: {len(wells)} wells")

        for well_id in wells:
            well_data = gage_data[gage_data[well_id_col] == well_id].copy()
            well_data = well_data.sort_values(date_col)

            if len(well_data) < 10:
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

            # Top panel: WTE
            ax1.scatter(well_data[date_col], well_data[wte_col],
                       alpha=0.6, s=3, color='blue')
            ax1.set_ylabel('WTE (feet)', fontsize=12)
            ax1.set_title(f'Well {well_id} - Gage {gage_id}\n'
                         f'({len(well_data):,} observations)', fontsize=14)
            ax1.grid(True, alpha=0.3)

            # Bottom panel: Streamflow
            ax2.scatter(well_data[date_col], well_data[q_col],
                       alpha=0.6, s=3, color='red')
            ax2.set_ylabel('Streamflow (cfs)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Highlight BFD periods
            if bfd_col in well_data.columns:
                bfd_data = well_data[well_data[bfd_col] == 1]
                if len(bfd_data) > 0:
                    ax1.scatter(bfd_data[date_col], bfd_data[wte_col],
                               color='orange', s=12, alpha=0.8, zorder=5,
                               label='BFD periods')
                    ax2.scatter(bfd_data[date_col], bfd_data[q_col],
                               color='orange', s=12, alpha=0.8, zorder=5)
                    ax1.legend(loc='upper right')

            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax2.xaxis.set_major_locator(mdates.YearLocator(5))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            filename = f'well_{well_id}_gage_{gage_id}_timeseries.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()

    print(f"Plots saved to: {output_dir}")


def plot_filtered_pairs_scatter(
    data: pd.DataFrame,
    well_stats: pd.DataFrame,
    output_dir: str = 'figures/filtered_pairs_scatter',
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    figsize: Tuple[int, int] = (10, 8)
) -> pd.DataFrame:
    """
    Create scatter plots for filtered well-gage pairs (R² > threshold) with regression lines.

    Parameters
    ----------
    data : pd.DataFrame
        Filtered paired data with delta metrics
    well_stats : pd.DataFrame
        Regression statistics per well-gage pair
    output_dir : str
        Directory to save plots
    well_id_col : str, default 'well_id'
        Column name for well ID
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    figsize : tuple, default (10, 8)
        Figure size

    Returns
    -------
    pd.DataFrame
        Summary statistics for plotted pairs

    Example
    -------
    >>> stats = plot_filtered_pairs_scatter(filtered_data, well_stats, 'figures/filtered')
    """
    os.makedirs(output_dir, exist_ok=True)

    stats_data = []

    # Group by well-gage pair
    for (well_id, gage_id), group in data.groupby([well_id_col, gage_id_col]):
        group = group.dropna(subset=[delta_wte_col, delta_q_col])

        if len(group) < 2:
            continue

        # Get regression stats for this pair
        pair_stats = well_stats[
            (well_stats[well_id_col] == well_id) & 
            (well_stats[gage_id_col] == gage_id)
        ]

        if len(pair_stats) == 0:
            continue

        stats_row = pair_stats.iloc[0]
        slope = stats_row['slope']
        intercept = stats_row['intercept']
        r_squared = stats_row['r_squared']
        p_value = stats_row['p_value']
        n_obs = stats_row['n_observations']

        # Create plot
        plt.figure(figsize=figsize)

        # Scatter plot
        plt.scatter(
            group[delta_wte_col],
            group[delta_q_col],
            alpha=0.6,
            s=30,
            color='steelblue',
            edgecolors='black',
            linewidth=0.5
        )

        # Regression line (use stats from well_stats, not recompute)
        x_range = np.array([
            group[delta_wte_col].min(),
            group[delta_wte_col].max()
        ])
        y_range = intercept + slope * x_range
        plt.plot(x_range, y_range, 'r-', linewidth=2, label='Regression line')

        # Add statistics text
        legend_text = (
            f"Well ID: {well_id}\n"
            f"Gage ID: {gage_id}\n"
            f"N: {n_obs}\n"
            f"Slope: {slope:.2f}\n"
            f"R²: {r_squared:.2f}\n"
            f"p-value: {format_p_value(p_value)}"
        )
        _add_stats_text(legend_text, alpha=0.8)

        plt.xlabel('ΔWTE (ft)', fontsize=12)
        plt.ylabel('ΔQ (cfs)', fontsize=12)
        plt.title(f'Well {well_id} - Gage {gage_id}\nΔQ vs ΔWTE (R² = {r_squared:.3f})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')

        # Save plot
        filename = f'well_{well_id}_gage_{gage_id}_scatter.png'
        plt.savefig(os.path.join(output_dir, filename),
                   bbox_inches='tight', dpi=150)
        plt.close()

        stats_data.append({
            'well_id': well_id,
            'gage_id': gage_id,
            'n_observations': n_obs,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'r_value': np.sqrt(r_squared) if r_squared >= 0 else -np.sqrt(-r_squared),
            'p_value': p_value,
            'std_err': stats_row.get('std_err', np.nan)
        })

    stats_df = pd.DataFrame(stats_data)
    if len(stats_df) > 0:
        stats_df.to_csv(os.path.join(output_dir, 'filtered_pairs_statistics.csv'), index=False)
        print(f"\nGenerated {len(stats_data)} scatter plots")
        print(f"Summary statistics:")
        print(f"  Mean slope: {stats_df['slope'].mean():.4f} cfs/ft")
        print(f"  Median slope: {stats_df['slope'].median():.4f} cfs/ft")
        print(f"  Mean R²: {stats_df['r_squared'].mean():.4f}")
        print(f"  Median R²: {stats_df['r_squared'].median():.4f}")
    else:
        print("No pairs to plot")

    return stats_df


def plot_pairs_by_r2_category(
    data: pd.DataFrame,
    well_stats: pd.DataFrame,
    output_dir: str = 'figures/pairs_by_r2_category',
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    figsize: Tuple[int, int] = (12, 8),
    r2_categories: list = None
) -> pd.DataFrame:
    """
    Create scatter plots grouped by R² categories and by gage.
    Each gage gets multiple plots, one for each R² category.

    Parameters
    ----------
    data : pd.DataFrame
        Full paired data with delta metrics
    well_stats : pd.DataFrame
        Regression statistics per well-gage pair
    output_dir : str
        Directory to save plots
    well_id_col : str, default 'well_id'
        Column name for well ID
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    figsize : tuple, default (12, 8)
        Figure size
    r2_categories : list, optional
        List of R² category boundaries. Default: [(0.1, 0.2), (0.2, 0.3), (0.3, None)]
        Each tuple is (min_r2, max_r2), where None means no upper limit

    Returns
    -------
    pd.DataFrame
        Summary statistics per gage and R² category

    Example
    -------
    >>> stats = plot_pairs_by_r2_category(data, well_stats, 'figures/by_category')
    """
    os.makedirs(output_dir, exist_ok=True)

    if r2_categories is None:
        r2_categories = [(0.1, 0.2), (0.2, 0.3), (0.3, None)]

    stats_data = []

    # Merge well_stats with data to get R² for each record
    data_with_r2 = data.merge(
        well_stats[[well_id_col, gage_id_col, 'r_squared']],
        on=[well_id_col, gage_id_col],
        how='inner'
    )

    # Group by gage
    for gage_id, gage_data in data_with_r2.groupby(gage_id_col):
        gage_data = gage_data.dropna(subset=[delta_wte_col, delta_q_col])

        if len(gage_data) < 2:
            continue

        # Create plots for each R² category
        for min_r2, max_r2 in r2_categories:
            # Filter by R² category
            if max_r2 is None:
                category_data = gage_data[
                    (gage_data['r_squared'] >= min_r2)
                ]
                category_name = f'r2_{min_r2}_plus'
                category_label = f'R² ≥ {min_r2}'
            else:
                category_data = gage_data[
                    (gage_data['r_squared'] >= min_r2) & 
                    (gage_data['r_squared'] < max_r2)
                ]
                category_name = f'r2_{min_r2}_{max_r2}'
                category_label = f'{min_r2} ≤ R² < {max_r2}'

            if len(category_data) < 2:
                continue

            # Get all wells for this category
            wells = category_data[well_id_col].unique()

            # Create plot
            plt.figure(figsize=figsize)

            # Use a colormap to distinguish different wells
            colors = plt.cm.tab20(np.linspace(0, 1, len(wells)))

            # Plot each well with different color
            for i, well_id in enumerate(wells):
                well_data = category_data[category_data[well_id_col] == well_id]

                plt.scatter(
                    well_data[delta_wte_col],
                    well_data[delta_q_col],
                    alpha=0.6,
                    s=30,
                    color=colors[i],
                    edgecolors='black',
                    linewidth=0.3,
                    label=f'Well {well_id}'
                )

            # Compute overall regression for this category
            x = category_data[delta_wte_col].values
            y = category_data[delta_q_col].values
            slope, intercept, r_squared, p_value, std_err, r_value = _compute_and_plot_regression(
                x, y, label=f'Overall regression (R²={r_squared:.3f})' if not np.isnan(r_squared) else None
            )

            # Add statistics text
            stats_text = (
                f"Gage ID: {gage_id}\n"
                f"R² Category: {category_label}\n"
                f"Number of wells: {len(wells)}\n"
                f"Total observations: {len(category_data)}\n"
                f"Overall slope: {slope:.2f}\n"
                f"Overall R²: {r_squared:.2f}\n"
                f"p-value: {format_p_value(p_value)}"
            )
            _add_stats_text(stats_text)

            plt.xlabel('ΔWTE (ft)', fontsize=12)
            plt.ylabel('ΔQ (cfs)', fontsize=12)
            plt.title(f'Gage {gage_id} - {category_label}\nΔQ vs ΔWTE', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # Add legend (limit to first 20 wells to avoid clutter)
            if len(wells) <= 20:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                         fontsize=6, ncol=2)

            plt.tight_layout()

            # Save plot
            filename = f'gage_{gage_id}_{category_name}_scatter.png'
            plt.savefig(os.path.join(output_dir, filename),
                       bbox_inches='tight', dpi=150)
            plt.close()

            stats_data.append({
                'gage_id': gage_id,
                'r2_category': category_label,
                'min_r2': min_r2,
                'max_r2': max_r2 if max_r2 is not None else np.inf,
                'n_wells': len(wells),
                'n_observations': len(category_data),
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'r_value': r_value if not np.isnan(r_squared) else np.nan,
                'p_value': p_value,
                'std_err': std_err
            })

    stats_df = pd.DataFrame(stats_data)
    if len(stats_df) > 0:
        stats_df.to_csv(os.path.join(output_dir, 'category_statistics.csv'), index=False)
        print(f"\nGenerated scatter plots by R² category")
        print(f"Total plots: {len(stats_data)}")
        print(f"\nSummary by category:")
        for category in stats_df['r2_category'].unique():
            cat_data = stats_df[stats_df['r2_category'] == category]
            print(f"  {category}:")
            print(f"    Gages: {cat_data['gage_id'].nunique()}")
            print(f"    Mean slope: {cat_data['slope'].mean():.4f} cfs/ft")
            print(f"    Mean R²: {cat_data['r_squared'].mean():.4f}")
    else:
        print("No data to plot")

    return stats_df


def plot_delta_scatter(
    data: pd.DataFrame,
    output_dir: str = 'figures/scatter_plots',
    gage_id_col: str = 'gage_id',
    well_id_col: str = 'well_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    class_col: str = None,
    figsize: Tuple[int, int] = (12, 6)
) -> pd.DataFrame:
    """
    Create scatter plots of ΔQ vs ΔWTE by gage with regression lines.

    Parameters
    ----------
    data : pd.DataFrame
        Data with delta metrics
    output_dir : str
        Directory to save plots
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    well_id_col : str, default 'well_id'
        Column name for well ID
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    class_col : str, optional
        Column for gage classification
    figsize : tuple, default (12, 6)
        Figure size

    Returns
    -------
    pd.DataFrame
        Regression statistics per gage

    Example
    -------
    >>> stats = plot_delta_scatter(data_with_deltas, 'figures/scatter')
    """
    os.makedirs(output_dir, exist_ok=True)

    stats_data = []

    for gage_id, group in data.groupby(gage_id_col):
        group = group.dropna(subset=[delta_wte_col, delta_q_col])

        if len(group) < 2:
            continue

        plt.figure(figsize=figsize)

        sns.scatterplot(
            data=group,
            x=delta_wte_col,
            y=delta_q_col,
            hue=well_id_col,
            palette='viridis',
            legend=False,
            alpha=0.6
        )

        # Regression analysis
        if len(group[delta_wte_col].unique()) > 1:
            x = group[delta_wte_col].values
            y = group[delta_q_col].values
            slope, intercept, r_squared, p_value, std_err, r_value = _compute_and_plot_regression(
                x, y, linewidth=2, label=None, zorder=5
            )

            class_value = None
            if class_col and class_col in group.columns:
                class_value = group[class_col].iloc[0]

            stats_data.append({
                'gage_id': gage_id,
                'n_wells': group[well_id_col].nunique(),
                'n_observations': len(group),
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'p_value': p_value,
                'class': class_value
            })

            legend_text = (
                f"Wells: {group[well_id_col].nunique()}\n"
                f"N: {len(group)}\n"
                f"Slope: {slope:.2f}\n"
                f"R²: {r_value ** 2:.2f}\n"
                f"p: {p_value:.4f}"
            )

            plt.text(0.98, 0.95, legend_text,
                    transform=plt.gca().transAxes,
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        title = f'Gage {gage_id}'
        if class_col and class_col in group.columns:
            class_val = group[class_col].iloc[0]
            title += f' - {class_val}'
        title += '\nΔQ vs ΔWTE'

        plt.title(title, fontsize=14)
        plt.xlabel('ΔWTE (ft)')
        plt.ylabel('ΔQ (cfs)')
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(output_dir, f'gage_{gage_id}.png'),
                   bbox_inches='tight', dpi=150)
        plt.close()

    stats_df = pd.DataFrame(stats_data)
    if len(stats_df) > 0:
        stats_df.to_csv(os.path.join(output_dir, 'regression_statistics.csv'), index=False)

    print(f"Generated {len(stats_data)} scatter plots")
    return stats_df


def plot_filtered_pairs_by_gage(
    data: pd.DataFrame,
    well_stats: pd.DataFrame,
    output_dir: str = 'figures/filtered_pairs_by_gage',
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    figsize: Tuple[int, int] = (12, 8),
    r_squared_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Create scatter plots for filtered pairs grouped by gage.
    Each gage gets one plot showing all wells' measurements with different colors.

    Parameters
    ----------
    data : pd.DataFrame
        Filtered paired data with delta metrics
    well_stats : pd.DataFrame
        Regression statistics per well-gage pair
    output_dir : str
        Directory to save plots
    well_id_col : str, default 'well_id'
        Column name for well ID
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    figsize : tuple, default (12, 8)
        Figure size

    Returns
    -------
    pd.DataFrame
        Summary statistics per gage

    Example
    -------
    >>> stats = plot_filtered_pairs_by_gage(filtered_data, well_stats, 'figures/by_gage')
    """
    os.makedirs(output_dir, exist_ok=True)

    stats_data = []

    # Group by gage
    for gage_id, gage_data in data.groupby(gage_id_col):
        gage_data = gage_data.dropna(subset=[delta_wte_col, delta_q_col])

        if len(gage_data) < 2:
            continue

        # Get all wells for this gage
        wells = gage_data[well_id_col].unique()
        
        # Get well stats for this gage
        gage_well_stats = well_stats[well_stats[gage_id_col] == gage_id]

        # Create plot
        plt.figure(figsize=figsize)

        # Use a colormap to distinguish different wells
        colors = plt.cm.tab20(np.linspace(0, 1, len(wells)))
        
        # Plot each well with different color (no label to avoid legend clutter)
        for i, well_id in enumerate(wells):
            well_data = gage_data[gage_data[well_id_col] == well_id]
            
            plt.scatter(
                well_data[delta_wte_col],
                well_data[delta_q_col],
                alpha=0.6,
                s=30,
                color=colors[i],
                edgecolors='black',
                linewidth=0.3
            )

        # Compute overall regression for the gage (all wells combined)
        x = gage_data[delta_wte_col].values
        y = gage_data[delta_q_col].values
        slope, intercept, r_squared, p_value, std_err, r_value = _compute_and_plot_regression(
            x, y, label=f'Overall regression (R²={r_squared:.3f})' if not np.isnan(r_squared) else None
        )

        # Add statistics text
        stats_text = (
            f"Gage ID: {gage_id}\n"
            f"Number of wells: {len(wells)}\n"
            f"Total observations: {len(gage_data)}\n"
            f"Overall slope: {slope:.2f}\n"
            f"Overall R²: {r_squared:.2f}\n"
            f"p-value: {format_p_value(p_value)}"
        )
        _add_stats_text(stats_text)

        plt.xlabel('ΔWTE (ft)', fontsize=12)
        plt.ylabel('ΔQ (cfs)', fontsize=12)
        plt.title(f'Gage {gage_id}\nΔQ vs ΔWTE (Filtered pairs, R² > {r_squared_threshold})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Only show regression line legend, not individual well legends
        if not np.isnan(r_squared):
            plt.legend(loc='best', fontsize=10)

        plt.tight_layout()

        # Save plot
        filename = f'gage_{gage_id}_scatter.png'
        plt.savefig(os.path.join(output_dir, filename),
                   bbox_inches='tight', dpi=150)
        plt.close()

        stats_data.append({
            'gage_id': gage_id,
            'n_wells': len(wells),
            'n_observations': len(gage_data),
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'r_value': r_value if not np.isnan(r_squared) else np.nan,
            'p_value': p_value,
            'std_err': std_err
        })

    stats_df = pd.DataFrame(stats_data)
    if len(stats_df) > 0:
        stats_df.to_csv(os.path.join(output_dir, 'gage_statistics.csv'), index=False)
        print(f"\nGenerated {len(stats_data)} scatter plots (one per gage)")
        print(f"Summary statistics:")
        print(f"  Mean slope: {stats_df['slope'].mean():.4f} cfs/ft")
        print(f"  Median slope: {stats_df['slope'].median():.4f} cfs/ft")
        print(f"  Mean R²: {stats_df['r_squared'].mean():.4f}")
        print(f"  Median R²: {stats_df['r_squared'].median():.4f}")
    else:
        print("No gages to plot")

    return stats_df


def plot_mi_comparison(
    merged_mi: pd.DataFrame,
    output_dir: str = 'figures/mi_compare',
    mi_no_lag_col: str = 'mi_no_lag',
    mi_lag_col: str = 'mi_lag',
    figsize: Tuple[int, int] = (6, 6)
) -> None:
    """
    Create comparison plots for lagged vs non-lagged MI analysis.

    Parameters
    ----------
    merged_mi : pd.DataFrame
        Merged MI results from compare_lag_vs_no_lag
    output_dir : str
        Directory to save plots
    mi_no_lag_col : str, default 'mi_no_lag'
        Column name for non-lagged MI
    mi_lag_col : str, default 'mi_lag'
        Column name for lagged MI
    figsize : tuple, default (6, 6)
        Figure size

    Example
    -------
    >>> plot_mi_comparison(merged_mi, 'figures/mi_compare')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Scatter plot: MI no-lag vs MI lag
    plt.figure(figsize=figsize)
    sns.scatterplot(data=merged_mi, x=mi_no_lag_col, y=mi_lag_col, alpha=0.6, s=30)
    lims = [0, max(merged_mi[[mi_no_lag_col, mi_lag_col]].max().max(), 0.001)]
    plt.plot(lims, lims, 'r--', label='y=x')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('MI (No-lag: ΔQ vs ΔWTE)')
    plt.ylabel('MI (Lag: ΔQ vs ΔWTE_lag)')
    plt.title('MI Comparison: Lag vs No-Lag')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mi_scatter_lag_vs_no_lag.png'), dpi=200)
    plt.close()

    # Histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(merged_mi[mi_no_lag_col].dropna(), bins=30, ax=axes[0],
                color='skyblue', edgecolor='black')
    axes[0].set_title('Distribution of MI (No-lag)')
    axes[0].grid(True, alpha=0.3)

    sns.histplot(merged_mi[mi_lag_col].dropna(), bins=30, ax=axes[1],
                color='lightgreen', edgecolor='black')
    axes[1].set_title('Distribution of MI (Lag)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mi_distributions.png'), dpi=200)
    plt.close()

    # Delta MI distribution
    if 'delta_mi' in merged_mi.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(merged_mi['delta_mi'].dropna(), bins=40, color='salmon', edgecolor='black')
        plt.axvline(0, color='k', linestyle='--', label='0')
        plt.title('Distribution of ΔMI = MI_lag - MI_no_lag')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delta_mi_distribution.png'), dpi=200)
        plt.close()

    print(f"MI comparison plots saved to: {output_dir}")


def plot_regression_summary(
    gage_stats: pd.DataFrame,
    output_dir: str = 'figures/regression',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Create summary plots for regression analysis results.

    Parameters
    ----------
    gage_stats : pd.DataFrame
        Regression statistics by gage
    output_dir : str
        Directory to save plots
    figsize : tuple, default (10, 6)
        Figure size

    Example
    -------
    >>> plot_regression_summary(gage_stats, 'figures/regression')
    """
    os.makedirs(output_dir, exist_ok=True)

    # R² distribution
    plt.figure(figsize=figsize)
    sns.histplot(gage_stats['r_squared'], bins=30, edgecolor='black')
    plt.axvline(gage_stats['r_squared'].median(), color='r', linestyle='--',
               label=f'Median: {gage_stats["r_squared"].median():.3f}')
    plt.xlabel('R²')
    plt.ylabel('Count')
    plt.title('Distribution of R² by Gage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r_squared_distribution.png'), dpi=150)
    plt.close()

    # Slope distribution
    plt.figure(figsize=figsize)
    sns.histplot(gage_stats['slope'], bins=30, edgecolor='black')
    plt.axvline(0, color='k', linestyle='-', alpha=0.5)
    plt.axvline(gage_stats['slope'].median(), color='r', linestyle='--',
               label=f'Median: {gage_stats["slope"].median():.3f}')
    plt.xlabel('Slope (cfs/ft)')
    plt.ylabel('Count')
    plt.title('Distribution of Slope by Gage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slope_distribution.png'), dpi=150)
    plt.close()

    # R² vs number of observations
    plt.figure(figsize=figsize)
    plt.scatter(gage_stats['n_observations'], gage_stats['r_squared'], alpha=0.6)
    plt.xlabel('Number of Observations')
    plt.ylabel('R²')
    plt.title('R² vs Number of Observations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r_squared_vs_n_obs.png'), dpi=150)
    plt.close()

    print(f"Regression summary plots saved to: {output_dir}")


def plot_elevation_filter_sensitivity(
    sensitivity_results: pd.DataFrame,
    output_dir: str = 'figures',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot sensitivity analysis for elevation buffer values.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        Results from analyze_elevation_sensitivity
    output_dir : str
        Directory to save plot
    figsize : tuple, default (10, 6)
        Figure size

    Example
    -------
    >>> plot_elevation_filter_sensitivity(sensitivity_df, 'figures')
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(sensitivity_results['buffer_m'], sensitivity_results['n_records'],
            'b-o', label='Records')
    ax1.set_xlabel('Elevation Buffer (m)')
    ax1.set_ylabel('Number of Records', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(sensitivity_results['buffer_m'], sensitivity_results['retention_pct'],
            'r-s', label='Retention %')
    ax2.set_ylabel('Retention (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Elevation Buffer Sensitivity Analysis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'elevation_buffer_sensitivity.png'), dpi=150)
    plt.close()

    print(f"Sensitivity plot saved to: {output_dir}")
