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

import matplotlib.patches as mpatches
import warnings

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from tqdm import tqdm
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False


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
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
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
            plt.savefig(os.path.join(output_dir, filename), dpi=600, bbox_inches='tight')
            plt.close()

    print(f"Plots saved to: {output_dir}")


def plot_well_timeseries_with_interpolation(
    paired: pd.DataFrame,
    well_ts_raw: pd.DataFrame,
    well_ts_monthly: pd.DataFrame,
    output_dir: str = 'figures/well_timeseries',
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    q_col: str = 'q',
    max_wells_per_gage: int = None,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """
    Create well time series plots with original observations, daily interpolation,
    and monthly interpolation in one figure.

    Top panel: WTE - raw obs (scatter), daily interp (line), monthly interp (line)
    Bottom panel: Streamflow
    """
    from .interpolation import interpolate_daily

    os.makedirs(output_dir, exist_ok=True)

    paired = paired.copy()
    paired[date_col] = pd.to_datetime(paired[date_col])
    well_ts_raw = well_ts_raw.copy()
    well_ts_raw[date_col] = pd.to_datetime(well_ts_raw[date_col])
    well_ts_monthly = well_ts_monthly.copy()
    well_ts_monthly[date_col] = pd.to_datetime(well_ts_monthly[date_col])

    for gage_id in paired[gage_id_col].unique():
        gage_data = paired[paired[gage_id_col] == gage_id]
        wells = gage_data[well_id_col].unique()
        if max_wells_per_gage:
            wells = wells[:max_wells_per_gage]

        for well_id in wells:
            well_paired = gage_data[gage_data[well_id_col] == well_id].copy()
            well_paired = well_paired.sort_values(date_col)
            if len(well_paired) < 10:
                continue

            # Get raw observations for this well
            raw_well = well_ts_raw[well_ts_raw[well_id_col] == well_id].dropna(subset=[wte_col])
            if len(raw_well) < 2:
                continue

            # Get monthly interpolated for this well
            monthly_well = well_ts_monthly[well_ts_monthly[well_id_col] == well_id].copy()

            # Compute daily interpolation for this well
            try:
                daily_df = interpolate_daily(
                    raw_well[[well_id_col, date_col, wte_col]],
                    well_id_col=well_id_col, date_col=date_col, value_col=wte_col
                )
            except Exception:
                daily_df = pd.DataFrame()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

            # Top panel: WTE - raw, daily, monthly
            ax1.scatter(raw_well[date_col], raw_well[wte_col],
                       alpha=0.8, s=25, color='black', zorder=5, label='Original observations')
            if len(daily_df) > 0:
                daily_sorted = daily_df.sort_values(date_col)
                ax1.plot(daily_sorted[date_col], daily_sorted[wte_col],
                        '-', color='blue', alpha=0.7, linewidth=1, label='Daily interpolation')
            if len(monthly_well) > 0:
                monthly_sorted = monthly_well.sort_values(date_col)
                ax1.plot(monthly_sorted[date_col], monthly_sorted[wte_col],
                        'o-', color='red', alpha=0.8, markersize=4, linewidth=0.8,
                        label='Monthly interpolation')
            ax1.set_ylabel('WTE (feet)', fontsize=12)
            ax1.set_title(f'Well {well_id} - Gage {gage_id}\n'
                         f'Raw: {len(raw_well)}, Daily: {len(daily_df)}, Monthly: {len(monthly_well)}', fontsize=14)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Bottom panel: Streamflow
            ax2.scatter(well_paired[date_col], well_paired[q_col],
                       alpha=0.6, s=5, color='steelblue')
            ax2.set_ylabel('Streamflow (cfs)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax2.xaxis.set_major_locator(mdates.YearLocator(5))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            filename = f'well_{well_id}_gage_{gage_id}_timeseries.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=600, bbox_inches='tight')
            plt.close()

    print(f"Plots with interpolation saved to: {output_dir}")


def plot_high_r2_gages(
    data_with_deltas: pd.DataFrame,
    gage_stats: pd.DataFrame,
    output_dir: str,
    r2_threshold: float = 0.1,
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    gage_id_col: str = 'gage_id',
    figsize: Tuple[int, int] = (12, 6)
) -> pd.DataFrame:
    """
    Filter to gages with R² > threshold and create scatter plots for their well data.
    """
    os.makedirs(output_dir, exist_ok=True)
    if len(gage_stats) == 0 or 'r_squared' not in gage_stats.columns:
        print(f"  No gages with R² > {r2_threshold}; skipping high-R² plots")
        return pd.DataFrame()
    high_r2_gages = gage_stats[gage_stats['r_squared'] > r2_threshold][gage_id_col].unique()
    if len(high_r2_gages) == 0:
        print(f"  No gages with R² > {r2_threshold}; skipping high-R² plots")
        return pd.DataFrame()

    filtered = data_with_deltas[data_with_deltas[gage_id_col].isin(high_r2_gages)]
    print(f"  High-R² gages (R²>{r2_threshold}): {list(high_r2_gages)}")
    print(f"  Filtered to {len(filtered):,} records, {filtered['well_id'].nunique()} wells")

    plot_delta_scatter(
        filtered, output_dir,
        delta_wte_col=delta_wte_col, delta_q_col=delta_q_col,
        gage_id_col=gage_id_col, figsize=figsize
    )
    return filtered


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
                   bbox_inches='tight', dpi=600)
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
                       bbox_inches='tight', dpi=600)
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
    figsize: Tuple[int, int] = (12, 6),
    gage_name_map: dict = None
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

        gage_name = gage_name_map.get(str(gage_id), '') if gage_name_map else ''
        title = f'Gage {gage_id}'
        if gage_name:
            title += f'\n{gage_name}'
        if class_col and class_col in group.columns:
            class_val = group[class_col].iloc[0]
            title += f' - {class_val}'
        title += '\nΔQ vs ΔWTE'

        plt.title(title, fontsize=14)
        plt.xlabel('ΔWTE (ft)')
        plt.ylabel('ΔQ (cfs)')
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(output_dir, f'gage_{gage_id}.png'),
                   bbox_inches='tight', dpi=600)
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
                   bbox_inches='tight', dpi=600)
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

    def _add_trend(ax, x, y, color='steelblue'):
        mask = ~(np.isnan(x) | np.isnan(y))
        xc, yc = x[mask], y[mask]
        if len(xc) < 3:
            return
        slope, intercept, r_val, p_val, _ = linregress(xc, yc)
        xs = np.array([xc.min(), xc.max()])
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.8, label='OLS trend')
        p_str = f'{p_val:.2e}' if p_val < 0.001 else f'{p_val:.3f}'
        ax.text(0.05, 0.95, f'r = {r_val:.3f}\np = {p_str}\nn = {len(xc)}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Scatter: MI no-lag vs MI lag
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(merged_mi[mi_no_lag_col], merged_mi[mi_lag_col],
               alpha=0.6, s=30, color='steelblue', zorder=2)
    lims = [0, max(merged_mi[[mi_no_lag_col, mi_lag_col]].max().max(), 0.001)]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='y = x (1:1)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    _add_trend(ax, merged_mi[mi_no_lag_col].values, merged_mi[mi_lag_col].values)
    ax.set_xlabel('MI (No-lag: ΔQ vs ΔWTE)')
    ax.set_ylabel('MI (Lag: ΔQ vs ΔWTE_lag)')
    ax.set_title('MI Comparison: Lag vs No-Lag')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mi_scatter_lag_vs_no_lag.png'), dpi=600)
    plt.close()

    # Histograms with mean/median/std annotations
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, color, label in [
        (axes[0], mi_no_lag_col, 'skyblue',    'No-lag'),
        (axes[1], mi_lag_col,    'lightgreen', 'Lag'),
    ]:
        vals = merged_mi[col].dropna()
        sns.histplot(vals, bins=30, ax=ax, color=color, edgecolor='black')
        mn, med, sd = vals.mean(), vals.median(), vals.std()
        ax.axvline(mn,  color='red',    linestyle='--', linewidth=1.5, label=f'Mean: {mn:.3f}')
        ax.axvline(med, color='orange', linestyle='-',  linewidth=1.5, label=f'Median: {med:.3f}')
        ax.text(0.97, 0.95,
                f'Mean:   {mn:.3f}\nMedian: {med:.3f}\nStd:    {sd:.3f}\nn = {len(vals)}',
                transform=ax.transAxes, va='top', ha='right', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_title(f'Distribution of MI ({label})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mi_distributions.png'), dpi=600)
    plt.close()

    # Delta MI distribution with stats
    if 'delta_mi' in merged_mi.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        dmi = merged_mi['delta_mi'].dropna()
        sns.histplot(dmi, bins=40, ax=ax, color='salmon', edgecolor='black')
        ax.axvline(0,            color='k',      linestyle='--', linewidth=1.5, label='Zero')
        ax.axvline(dmi.mean(),   color='red',    linestyle='--', linewidth=1.5,
                   label=f'Mean: {dmi.mean():.3f}')
        ax.axvline(dmi.median(), color='orange', linestyle='-',  linewidth=1.5,
                   label=f'Median: {dmi.median():.3f}')
        pct_pos = (dmi > 0).mean() * 100
        ax.text(0.97, 0.95,
                f'Mean:   {dmi.mean():.3f}\nMedian: {dmi.median():.3f}\n'
                f'Std:    {dmi.std():.3f}\n% lag > no-lag: {pct_pos:.1f}%\nn = {len(dmi)}',
                transform=ax.transAxes, va='top', ha='right', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_title('Distribution of ΔMI = MI_lag − MI_no_lag')
        ax.set_xlabel('ΔMI')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delta_mi_distribution.png'), dpi=600)
        plt.close()

    # Nonlinearity gain vs |Pearson r|
    if 'nl_gain_lag' in merged_mi.columns and 'pearson_lag' in merged_mi.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        x_nl = merged_mi['pearson_lag'].abs().values
        y_nl = merged_mi['nl_gain_lag'].values
        ax.scatter(x_nl, y_nl, alpha=0.6, s=30, color='purple', zorder=2)
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        _add_trend(ax, x_nl, y_nl, color='darkorchid')
        ax.set_xlabel('|Pearson r| (Lag)')
        ax.set_ylabel('Nonlinearity gain (MI_lag − |r_lag|)')
        ax.set_title('Nonlinearity gain vs |Pearson r| (Lag)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'nonlinearity_vs_pearson_lag.png'), dpi=600)
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

    if gage_stats.empty or 'r_squared' not in gage_stats.columns:
        print(f"  No regression results to plot; skipping regression summary")
        return

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
    plt.savefig(os.path.join(output_dir, 'r_squared_distribution.png'), dpi=600)
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
    plt.savefig(os.path.join(output_dir, 'slope_distribution.png'), dpi=600)
    plt.close()

    # R² vs number of observations
    plt.figure(figsize=figsize)
    plt.scatter(gage_stats['n_observations'], gage_stats['r_squared'], alpha=0.6)
    plt.xlabel('Number of Observations')
    plt.ylabel('R²')
    plt.title('R² vs Number of Observations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r_squared_vs_n_obs.png'), dpi=600)
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
    plt.savefig(os.path.join(output_dir, 'elevation_buffer_sensitivity.png'), dpi=600)
    plt.close()

    print(f"Sensitivity plot saved to: {output_dir}")


def plot_mi_results(
    mi_results: pd.DataFrame,
    output_dir: str = 'figures/mi',
    gage_id_col: str = 'gage_id',
    figsize: Tuple[int, int] = (10, 6),
    basin_names: dict = None
) -> None:
    """
    Create MI distribution and by-gage summary plots.

    Parameters
    ----------
    mi_results : pd.DataFrame
        MI analysis results from compute_mi_analysis
    output_dir : str
        Directory to save plots
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    figsize : tuple, default (10, 6)
        Figure size
    basin_names : dict, optional
        Mapping from gage ID (str) to basin name for axis labels.
        When provided, replaces numeric gage IDs with basin names on the x-axis.

    Example
    -------
    >>> plot_mi_results(mi_results, 'figures/mi')
    """
    os.makedirs(output_dir, exist_ok=True)

    # MI distribution with mean/median/std
    fig, ax = plt.subplots(figsize=figsize)
    vals = mi_results['mi'].dropna()
    sns.histplot(vals, bins=40, ax=ax, edgecolor='black', color='steelblue')
    mn, med, sd = vals.mean(), vals.median(), vals.std()
    ax.axvline(mn,  color='red',    linestyle='--', linewidth=1.5, label=f'Mean: {mn:.3f}')
    ax.axvline(med, color='orange', linestyle='-',  linewidth=1.5, label=f'Median: {med:.3f}')
    ax.text(0.97, 0.95,
            f'Mean:   {mn:.3f}\nMedian: {med:.3f}\nStd:    {sd:.3f}\nn = {len(vals)}',
            transform=ax.transAxes, va='top', ha='right', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of MI by Well-Gage Pair')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mi_distribution.png'), dpi=600)
    plt.close()

    # MI by gage (box plot)
    if gage_id_col in mi_results.columns and mi_results[gage_id_col].nunique() > 1:
        plot_df = mi_results.copy()
        if basin_names:
            plot_df['_gage_label'] = plot_df[gage_id_col].astype(str).map(
                lambda g: basin_names.get(g, g)
            )
            label_col = '_gage_label'
            xlabel = 'Basin'
        else:
            label_col = gage_id_col
            xlabel = 'Gage ID'
        plt.figure(figsize=(max(8, plot_df[gage_id_col].nunique() * 1.2), 5))
        sns.boxplot(data=plot_df, x=label_col, y='mi')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(xlabel)
        plt.ylabel('Mutual Information')
        plt.title('MI Distribution by Gage')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mi_by_gage.png'), dpi=600)
        plt.close()

    # MI vs |Pearson r| with OLS trend
    if 'pearson_r' in mi_results.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        x = mi_results['pearson_r'].abs().values
        y = mi_results['mi'].values
        ax.scatter(x, y, alpha=0.5, s=20, color='steelblue', zorder=2)
        mask = ~(np.isnan(x) | np.isnan(y))
        xc, yc = x[mask], y[mask]
        if len(xc) >= 3:
            slope, intercept, r_val, p_val, _ = linregress(xc, yc)
            xs = np.array([xc.min(), xc.max()])
            ax.plot(xs, slope * xs + intercept, color='red', linewidth=1.8, label='OLS trend')
            p_str = f'{p_val:.2e}' if p_val < 0.001 else f'{p_val:.3f}'
            ax.text(0.05, 0.95, f'r = {r_val:.3f}\np = {p_str}\nn = {len(xc)}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_xlabel('|Pearson r|')
        ax.set_ylabel('Mutual Information')
        ax.set_title('MI vs |Pearson Correlation|')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mi_vs_pearson.png'), dpi=600)
        plt.close()

    print(f"MI results plots saved to: {output_dir}")


def plot_seasonal_monthly_analysis(
    seasonal_stats: pd.DataFrame,
    monthly_stats: pd.DataFrame,
    output_dir: str = 'figures/seasonal_monthly',
    gage_id_col: str = 'gage_id',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Create seasonal and monthly analysis visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    season_order = ['Winter', 'Spring', 'Summer', 'Fall']

    if len(seasonal_stats) > 0:
        # R² by season
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=seasonal_stats, x='season', y='r_squared',
                    order=season_order, ax=ax)
        ax.set_title('R² of ΔQ vs ΔWTE by Season')
        ax.set_ylabel('R²')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'r_squared_by_season.png'), dpi=600)
        plt.close()

        # Slope by season
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=seasonal_stats, x='season', y='slope',
                    order=season_order, ax=ax)
        ax.axhline(0, color='k', linestyle='-', alpha=0.5)
        ax.set_title('Slope (ΔQ vs ΔWTE) by Season')
        ax.set_ylabel('Slope (cfs/ft)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'slope_by_season.png'), dpi=600)
        plt.close()

        # Mean R² and slope by season (summary bar)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        season_agg = seasonal_stats.groupby('season').agg(
            mean_r2=('r_squared', 'mean'),
            mean_slope=('slope', 'mean')
        )
        # Use season_order for display, keep only existing
        season_agg = season_agg.reindex([s for s in season_order if s in season_agg.index])
        if len(season_agg) > 0:
            season_agg.plot(kind='bar', y='mean_r2', ax=axes[0], legend=False)
            axes[0].set_title('Mean R² by Season')
            axes[0].set_xticklabels(season_agg.index, rotation=0)
            axes[0].set_ylabel('Mean R²')
            season_agg.plot(kind='bar', y='mean_slope', ax=axes[1], legend=False)
            axes[1].axhline(0, color='k', linestyle='-', alpha=0.5)
            axes[1].set_title('Mean Slope by Season')
            axes[1].set_xticklabels(season_agg.index, rotation=0)
            axes[1].set_ylabel('Mean Slope (cfs/ft)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'seasonal_summary.png'), dpi=600)
        plt.close()

    if len(monthly_stats) > 0:
        # R² by month
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_sorted = monthly_stats.sort_values('month')
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=monthly_sorted, x='month_name', y='r_squared',
                    order=[m for m in month_order if m in monthly_sorted['month_name'].values],
                    ax=ax)
        ax.set_title('R² of ΔQ vs ΔWTE by Month')
        ax.set_ylabel('R²')
        ax.set_xlabel('Month')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'r_squared_by_month.png'), dpi=600)
        plt.close()

        # Slope by month
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=monthly_sorted, x='month_name', y='slope',
                    order=[m for m in month_order if m in monthly_sorted['month_name'].values],
                    ax=ax)
        ax.axhline(0, color='k', linestyle='-', alpha=0.5)
        ax.set_title('Slope (ΔQ vs ΔWTE) by Month')
        ax.set_ylabel('Slope (cfs/ft)')
        ax.set_xlabel('Month')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'slope_by_month.png'), dpi=600)
        plt.close()

    print(f"Seasonal/monthly plots saved to: {output_dir}")


def plot_seasonal_monthly_scatter(
    data: pd.DataFrame,
    output_dir: str = 'figures/seasonal_monthly',
    date_col: str = 'date',
    gage_id_col: str = 'gage_id',
    well_id_col: str = 'well_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    min_observations: int = 5,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Create ΔQ vs ΔWTE scatter plots: one figure per gage with 4 seasonal subplots,
    one figure per gage with 12 monthly subplots.
    """
    os.makedirs(output_dir, exist_ok=True)
    scatter_season_dir = os.path.join(output_dir, 'scatter_seasonal')
    scatter_month_dir = os.path.join(output_dir, 'scatter_monthly')
    os.makedirs(scatter_season_dir, exist_ok=True)
    os.makedirs(scatter_month_dir, exist_ok=True)

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[delta_wte_col, delta_q_col, date_col, gage_id_col])

    if len(df) == 0:
        print(f"  No data for seasonal/monthly scatter plots; skipping")
        return

    def _get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['month'] = df[date_col].dt.month
    df['month_name'] = df['month'].map(lambda m: month_names[m])
    df['season'] = df['month'].apply(_get_season)

    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def _plot_subplot(ax, group, period_name):
        """Plot one subplot: scatter + regression + reference lines + stats."""
        if len(group) < min_observations:
            ax.text(0.5, 0.5, f'N < {min_observations}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(period_name)
            return
        sns.scatterplot(
            data=group, x=delta_wte_col, y=delta_q_col,
            hue=well_id_col, palette='viridis', legend=False, alpha=0.6, ax=ax
        )
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        x = group[delta_wte_col].values
        y = group[delta_q_col].values
        r2 = slope = p_value = np.nan
        if len(x) >= 2 and len(np.unique(x)) > 1 and np.std(y) > 0:
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                r2 = r_value ** 2
                x_range = np.array([x.min(), x.max()])
                ax.plot(x_range, intercept + slope * x_range, 'r-', linewidth=1.5, zorder=5)
            except Exception:
                pass
        ax.set_title(f'{period_name}\nN={len(group)}', fontsize=10)
        ax.set_xlabel('ΔWTE (ft)')
        ax.set_ylabel('ΔQ (cfs)')
        ax.grid(True, alpha=0.3)
        if not np.isnan(r2):
            p_str = '< 0.01' if p_value < 0.01 else f'{p_value:.2f}'
            stats = f"Slope={slope:.2f}\nR²={r2:.3f}\np={p_str}"
            ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                    fontsize=8, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Seasonal: one figure per gage, 2x2 subplots
    for gage_id in df[gage_id_col].unique():
        gage_df = df[df[gage_id_col] == gage_id]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        for idx, season in enumerate(season_order):
            group = gage_df[gage_df['season'] == season]
            _plot_subplot(axes[idx], group, season)
        fig.suptitle(f'Gage {gage_id} - ΔQ vs ΔWTE by Season', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(scatter_season_dir, f'gage_{gage_id}.png'), dpi=600, bbox_inches='tight')
        plt.close()

    # Monthly: one figure per gage, 4x3 subplots
    for gage_id in df[gage_id_col].unique():
        gage_df = df[df[gage_id_col] == gage_id]
        fig, axes = plt.subplots(4, 3, figsize=(14, 14), sharex=True, sharey=True)
        axes = axes.flatten()
        for idx, month_name in enumerate(month_order):
            group = gage_df[gage_df['month_name'] == month_name]
            _plot_subplot(axes[idx], group, month_name)
        fig.suptitle(f'Gage {gage_id} - ΔQ vs ΔWTE by Month', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(scatter_month_dir, f'gage_{gage_id}.png'), dpi=600, bbox_inches='tight')
        plt.close()

    print(f"  Seasonal scatter: {df[gage_id_col].nunique()} gages -> {scatter_season_dir}")
    print(f"  Monthly scatter: {df[gage_id_col].nunique()} gages -> {scatter_month_dir}")


# ============================================================
# Additional visualization functions from notebooks 02, 05, 07
# ============================================================



def get_gage_watersheds_styled(gage_id, subbasin_gdf, gage_df, terminal_relationships=None):
    """Get terminal and upstream basins for a gage - styled version (shared)."""
    gage_info = gage_df[gage_df['id'] == gage_id]
    if gage_info.empty:
        return gpd.GeoDataFrame(geometry=[], crs=subbasin_gdf.crs), gpd.GeoDataFrame(geometry=[], crs=subbasin_gdf.crs)

    gage_lat = gage_info.iloc[0]['latitude']
    gage_lon = gage_info.iloc[0]['longitude']

    gage_point = Point(gage_lon, gage_lat)
    gage_gdf = gpd.GeoDataFrame([1], geometry=[gage_point], crs="EPSG:4326")
    gage_gdf = gage_gdf.to_crs(subbasin_gdf.crs)
    gage_point = gage_gdf.geometry.iloc[0]

    containing = subbasin_gdf[subbasin_gdf.geometry.contains(gage_point)]
    if len(containing) > 0:
        terminal_basin = containing.iloc[[0]]
    else:
        distances = subbasin_gdf.geometry.distance(gage_point)
        terminal_basin = subbasin_gdf.iloc[[distances.idxmin()]]

    upstream_basins = gpd.GeoDataFrame(geometry=[], crs=subbasin_gdf.crs)

    if terminal_relationships is not None:
        if 'Gage_ID' in terminal_relationships.columns:
            gage_col, upstream_col = 'Gage_ID', 'Upstream_Catchment_ID'
        else:
            gage_col, upstream_col = 'gage_id', 'upstream_catchment_id'

        linkno_col = None
        for col in ['linkno', 'LINKNO', 'LinkNo', 'LINK_NO']:
            if col in subbasin_gdf.columns:
                linkno_col = col
                break

        if linkno_col and gage_col in terminal_relationships.columns:
            upstream_catchments = terminal_relationships.loc[
                terminal_relationships[gage_col] == gage_id, upstream_col
            ].dropna().astype(int).tolist()

            if upstream_catchments:
                upstream_basins = subbasin_gdf[
                    subbasin_gdf[linkno_col].astype(int).isin(upstream_catchments)
                ].copy()

                if not terminal_basin.empty:
                    terminal_linkno = int(terminal_basin.iloc[0][linkno_col])
                    upstream_basins = upstream_basins[
                        upstream_basins[linkno_col].astype(int) != terminal_linkno
                    ]

    return terminal_basin, upstream_basins


def _ensure_point_in_subbasin_crs(lon, lat, subbasin_gdf):
    """
    Build a GeoSeries point at (lon, lat) and convert to the subbasin CRS if needed.
    """
    from shapely.geometry import Point
    gpt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    try:
        if subbasin_gdf.crs is not None and subbasin_gdf.crs != "EPSG:4326":
            gpt = gpt.to_crs(subbasin_gdf.crs)
    except Exception as e:
        print(f"CRS transform failed; proceeding in EPSG:4326. Error: {e}")
    return gpt


def get_gage_terminal_basin(gage_id, subbasin_gdf, gage_df):
    """
    Get the terminal basin (catchment polygon) that contains the gage.
    If containment fails (e.g., due to topology or slight offsets), use nearest.
    Handles CRS properly.
    """
    gage_info = gage_df[gage_df['id'] == gage_id]
    if gage_info.empty:
        print(f"Warning: No gage info found for gage {gage_id}")
        return gpd.GeoDataFrame(geometry=[], crs=subbasin_gdf.crs)

    gage_lat = gage_info.iloc[0]['latitude']
    gage_lon = gage_info.iloc[0]['longitude']
    print(f"Debug: Looking for terminal basin for gage {gage_id} at ({gage_lon}, {gage_lat})")

    gpt = _ensure_point_in_subbasin_crs(gage_lon, gage_lat, subbasin_gdf)

    # Containment test
    try:
        containing = subbasin_gdf[subbasin_gdf.geometry.contains(gpt.iloc[0])]
        if len(containing) > 0:
            idx = containing.index[0]
            print(f"Debug: Found containing basin {idx} for gage {gage_id}")
            return subbasin_gdf.loc[[idx]]
    except Exception as e:
        print(f"Debug: Containment check failed for gage {gage_id}: {e}")

    # Fallback: nearest polygon (distance computed in subbasin CRS)
    print(f"Debug: No containing basin found, finding nearest for gage {gage_id}")
    try:
        distances = subbasin_gdf.geometry.distance(gpt.iloc[0])
        nearest_idx = distances.idxmin()
        nearest_dist = distances.min()
        print(f"Debug: Nearest basin {nearest_idx} at distance {nearest_dist} for gage {gage_id}")
        return subbasin_gdf.loc[[nearest_idx]]
    except Exception as e:
        print(f"Warning: Nearest basin selection failed for gage {gage_id}: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=subbasin_gdf.crs)


def get_gage_watersheds(gage_id, subbasin_gdf, gage_df, terminal_relationships=None):
    """
    Get the terminal basin and all upstream basins for a gage.
    - If relationships are provided, select upstream catchments by linkno/LINKNO.
    - The terminal basin is excluded from the upstream set.
    """
    terminal_basin = get_gage_terminal_basin(gage_id, subbasin_gdf, gage_df)
    print(f"Debug: Gage {gage_id} - Terminal basin found: {not terminal_basin.empty}")

    upstream_basins = gpd.GeoDataFrame(geometry=[], crs=subbasin_gdf.crs)

    if terminal_relationships is not None and 'gage_id' in terminal_relationships.columns:
        # Identify the linkno column
        linkno_col = None
        for cand in ['linkno', 'LINKNO', 'LinkNo', 'LINK_NO']:
            if cand in subbasin_gdf.columns:
                linkno_col = cand
                break

        if linkno_col is None:
            print("Warning: No linkno-like column found in subbasin_gdf; upstream basins cannot be resolved.")
            return terminal_basin, upstream_basins

        # Get upstream catchments list for this gage
        upstream_catchments = terminal_relationships.loc[
            terminal_relationships['gage_id'] == gage_id, 'upstream_catchment_id'
        ].dropna().astype(int).tolist()

        print(f"Debug: Gage {gage_id} - Upstream catchments: {len(upstream_catchments)}")

        if upstream_catchments:
            upstream_basins = subbasin_gdf[subbasin_gdf[linkno_col].astype(int).isin(upstream_catchments)].copy()
            # Remove terminal basin if included
            if not terminal_basin.empty:
                terminal_linkno = int(terminal_basin.iloc[0][linkno_col])
                upstream_basins = upstream_basins[upstream_basins[linkno_col].astype(int) != terminal_linkno]

    return terminal_basin, upstream_basins


def load_watershed_data():
    """Load watershed relationship data (terminal gage → upstream catchments)."""
    try:
        df = pd.read_csv('../data/processed/terminal_gage_upstream_catchments.csv')

        # Standardize column names
        rename_map = {}
        if 'Gage_ID' in df.columns:
            rename_map['Gage_ID'] = 'gage_id'
        if 'Upstream_Catchment_ID' in df.columns:
            rename_map['Upstream_Catchment_ID'] = 'upstream_catchment_id'
        if rename_map:
            df = df.rename(columns=rename_map)

        # Enforce types where possible
        if 'gage_id' in df.columns:
            df['gage_id'] = pd.to_numeric(df['gage_id'], errors='coerce').astype('Int64')
        if 'upstream_catchment_id' in df.columns:
            df['upstream_catchment_id'] = pd.to_numeric(df['upstream_catchment_id'], errors='coerce').astype('Int64')

        return df
    except Exception as e:
        print(f"Warning: Could not load watershed relationships: {e}")
        return None


def plot_overview_inset(ax, subbasin_gdf, stream_gdf, lake_gdf,
                        terminal_basin, upstream_basins, gage_lon, gage_lat, gage_id):
    """
    Plot the basin-wide overview inset with highlighted terminal/upstream basins.
    """
    # All catchments
    subbasin_gdf.plot(ax=ax, color='#FAFAFA', edgecolor='#B0B0B0',
                      linewidth=0.6, alpha=0.9)

    # Lakes for context
    lake_gdf.plot(ax=ax, color='#D6EAF8', alpha=0.9,
                  edgecolor='#3498DB', linewidth=0.6)

    # Highlight watersheds
    if not terminal_basin.empty:
        terminal_basin.plot(ax=ax, color='#CD853F', alpha=0.8,
                            edgecolor='#8B7355', linewidth=1.5)
    if not upstream_basins.empty:
        upstream_basins.plot(ax=ax, color='#F5E6D3', alpha=0.7,
                             edgecolor='#D2B48C', linewidth=1.2)

    # Gage star (small)
    ax.scatter(gage_lon, gage_lat, color='#FFD700', marker='*', s=150,
               edgecolor='black', linewidth=1.5, zorder=5)

    # Full basin extent
    basin_bounds = subbasin_gdf.total_bounds
    ax.set_xlim(basin_bounds[0], basin_bounds[2])
    ax.set_ylim(basin_bounds[1], basin_bounds[3])

    # Style
    ax.set_title('Basin Overview', fontsize=10, fontweight='bold', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    ax.set_aspect('equal')


def plot_overview_inset_styled(ax, subbasin_gdf, stream_gdf, lake_gdf,
                               terminal_basin, upstream_basins, gage_lon, gage_lat, gage_id):
    """Overview inset (shared)."""
    subbasin_gdf.plot(ax=ax, color='#FAFAFA', edgecolor='#B0B0B0', linewidth=0.6, alpha=0.9)
    lake_gdf.plot(ax=ax, color='#D6EAF8', alpha=0.9, edgecolor='#3498DB', linewidth=0.6)

    if not terminal_basin.empty:
        terminal_basin.plot(ax=ax, color='#CD853F', alpha=0.8, edgecolor='#8B7355', linewidth=1.5)
    if not upstream_basins.empty:
        upstream_basins.plot(ax=ax, color='#F5E6D3', alpha=0.7, edgecolor='#D2B48C', linewidth=1.2)

    ax.scatter(gage_lon, gage_lat, color='#FFD700', marker='*', s=150,
               edgecolor='black', linewidth=1.5, zorder=5)

    basin_bounds = subbasin_gdf.total_bounds
    ax.set_xlim(basin_bounds[0], basin_bounds[2])
    ax.set_ylim(basin_bounds[1], basin_bounds[3])

    ax.set_title('Basin Overview', fontsize=10, fontweight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(2); spine.set_edgecolor('black')
    ax.set_aspect('equal')


def add_labels_with_leader_lines(ax, wells_data, gage_lon, gage_lat, max_distance=0.05):
    """
    Add R² labels with leader lines to avoid overlaps.
    If wells_data is empty, this function does nothing.
    """
    if wells_data is None or wells_data.empty:
        return

    positions = []
    for i, (_, well) in enumerate(wells_data.iterrows()):
        number = i + 1
        well_x, well_y = well['well_lon'], well['well_lat']

        # Try different angles around the well to find non-overlapping position
        angles = [45, 135, 315, 225, 90, 270, 0, 180]
        best_pos = None
        min_conflict = float('inf')

        for angle in angles:
            angle_rad = np.radians(angle)
            label_x = well_x + max_distance * np.cos(angle_rad)
            label_y = well_y + max_distance * np.sin(angle_rad)

            conflict_count = 0
            gage_dist = np.sqrt((label_x - gage_lon)**2 + (label_y - gage_lat)**2)
            if gage_dist < max_distance * 0.6:
                conflict_count += 2

            for other_pos in positions:
                other_dist = np.sqrt((label_x - other_pos[0])**2 + (label_y - other_pos[1])**2)
                if other_dist < max_distance * 0.7:
                    conflict_count += 1

            if conflict_count < min_conflict:
                min_conflict = conflict_count
                best_pos = (label_x, label_y, angle)

        if best_pos:
            label_x, label_y, angle = best_pos
            positions.append((label_x, label_y))

            ax.plot([well_x, label_x], [well_y, label_y],
                    color='black', linewidth=1.2, alpha=0.7, zorder=11)

            ha = 'left' if angle < 180 else 'right'
            va = 'bottom' if 45 <= angle <= 135 else 'top'
            ax.text(label_x, label_y,
                    f'{number}: R² = {well["r_squared"]:.3f}',
                    ha=ha, va=va,
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.95, edgecolor='black', linewidth=1),
                    zorder=12)
        else:
            # Fallback to simple offset
            offset_x = 0.025 if i % 2 == 0 else -0.025
            offset_y = 0.020 if i < 3 else -0.010
            ax.text(well_x + offset_x, well_y + offset_y,
                    f'{number}: R² = {well["r_squared"]:.3f}',
                    ha='left' if offset_x > 0 else 'right',
                    va='bottom' if offset_y > 0 else 'top',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.95, edgecolor='black', linewidth=1),
                    zorder=12)


def add_statistics_and_legend(fig, ax_main, gage_wells, gage_id, terminal_basin, upstream_basins):
    """
    Add statistics box and legend to the figure.
    Works whether or not wells exist for the gage.
    """
    # Stats text
    watershed_info = ""
    if not terminal_basin.empty:
        watershed_info += f"Terminal Basin: 1\n"
    if not upstream_basins.empty:
        watershed_info += f"Upstream Basins: {len(upstream_basins)}\n"

    if not gage_wells.empty:
        max_r2 = gage_wells['r_squared'].max()
        mean_r2 = gage_wells['r_squared'].mean()
        wells_n = len(gage_wells)
    else:
        max_r2 = float('nan')
        mean_r2 = float('nan')
        wells_n = 0

    stats_text = (f"{watershed_info}"
                  f"Wells: {wells_n}\n"
                  f"Max R²: {max_r2:.3f}" if wells_n > 0 else f"{watershed_info}Wells: 0")

    if wells_n > 0:
        stats_text += f"\nMean R²: {mean_r2:.3f}"

    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='white',
                           alpha=0.95, edgecolor='black', linewidth=1),
                 verticalalignment='top', fontsize=10,
                 fontweight='bold', zorder=15)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='#FFD700', markersize=18,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Gage {gage_id}', linestyle='None'),
    ]

    if wells_n > 0:
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='red', markersize=10,
                       markeredgecolor='white', markeredgewidth=1,
                       label='Top 10 Wells (numbered)', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='lightgray', markersize=8,
                       markeredgecolor='gray', markeredgewidth=0.5,
                       label='Other Wells', linestyle='None')
        ])

    if not terminal_basin.empty:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor='#B8860B', edgecolor='#8B7355',
                          alpha=0.8, label='Terminal Basin (Gage Location)')
        )
    if not upstream_basins.empty:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor='#F5E6D3', edgecolor='#D2B48C',
                          alpha=0.6, label='Upstream Basins')
        )

    ax_main.legend(handles=legend_elements, loc='lower left',
                   bbox_to_anchor=(0.02, 0.02), fontsize=9,
                   frameon=True, fancybox=False, shadow=False,
                   edgecolor='black', facecolor='white', framealpha=0.95)


def add_distance_statistics_and_legend_styled(fig, ax_main, gage_wells, gage_id,
                                            terminal_basin, upstream_basins):
    """Add statistics box and legend - SAME STYLE AS CORRELATION MAPS."""
    # Stats text
    watershed_info = ""
    if not terminal_basin.empty:
        watershed_info += f"Terminal Basin: 1\n"
    if not upstream_basins.empty:
        watershed_info += f"Upstream Basins: {len(upstream_basins)}\n"

    if not gage_wells.empty:
        min_dist = gage_wells['Distance_to_Reach'].min()
        max_dist = gage_wells['Distance_to_Reach'].max()
        mean_dist = gage_wells['Distance_to_Reach'].mean()
        wells_n = len(gage_wells)
    else:
        min_dist = float('nan')
        max_dist = float('nan')
        mean_dist = float('nan')
        wells_n = 0

    stats_text = (f"{watershed_info}"
                  f"Wells: {wells_n}\n"
                  f"Min Distance: {min_dist:.1f}m\n"
                  f"Max Distance: {max_dist:.1f}m" if wells_n > 0 else f"{watershed_info}Wells: 0")

    if wells_n > 0:
        stats_text += f"\nMean Distance: {mean_dist:.1f}m"

    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='white',
                           alpha=0.95, edgecolor='black', linewidth=1),
                 verticalalignment='top', fontsize=10,
                 fontweight='bold', zorder=15)

    # Legend - SAME AS CORRELATION MAPS
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='#FFD700', markersize=18,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Gage {gage_id}', linestyle='None'),
    ]

    if wells_n > 0:
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='lightgray', markersize=8,
                       markeredgecolor='gray', markeredgewidth=0.5,
                       label='Wells (colored by distance)', linestyle='None')
        ])

    if not terminal_basin.empty:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor='#B8860B', edgecolor='#8B7355',
                          alpha=0.8, label='Terminal Basin (Gage Location)')
        )
    if not upstream_basins.empty:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor='#F5E6D3', edgecolor='#D2B48C',
                          alpha=0.6, label='Upstream Basins')
        )

    ax_main.legend(handles=legend_elements, loc='lower left',
                   bbox_to_anchor=(0.02, 0.02), fontsize=9,
                   frameon=True, fancybox=False, shadow=False,
                   edgecolor='black', facecolor='white', framealpha=0.95)


def add_vertical_distance_statistics_and_legend_styled(fig, ax_main, gage_wells, gage_id,
                                                      terminal_basin, upstream_basins):
    """
    Add statistics box and legend (MI version).
    Shows counts and Mutual Information (mi_delta_wte_delta_q) stats instead of vertical distance.
    """
    # Watershed info lines
    watershed_info = ""
    if not terminal_basin.empty:
        watershed_info += "Terminal Basin: 1\n"
    if not upstream_basins.empty:
        watershed_info += f"Upstream Basins: {len(upstream_basins)}\n"

    # MI stats for this gage's wells
    wells_n = len(gage_wells) if gage_wells is not None else 0
    if wells_n > 0 and 'mi_delta_wte_delta_q' in gage_wells.columns:
        mi_series = gage_wells['mi_delta_wte_delta_q'].astype(float)
        mi_series = mi_series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(mi_series) > 0:
            min_mi  = np.nanmin(mi_series)
            max_mi  = np.nanmax(mi_series)
            mean_mi = np.nanmean(mi_series)
            p90_mi  = np.nanpercentile(mi_series, 90)

            stats_text = (
                f"{watershed_info}"
                f"Wells: {wells_n}\n"
                f"Min MI:  {min_mi:.3f}\n"
                f"Mean MI: {mean_mi:.3f}\n"
                f"P90 MI:  {p90_mi:.3f}\n"
                f"Max MI:  {max_mi:.3f}"
            )
        else:
            stats_text = f"{watershed_info}Wells: {wells_n}\nNo valid MI values"
    else:
        stats_text = f"{watershed_info}Wells: {wells_n}\nNo MI column"

    # Draw stats box
    ax_main.text(
        0.02, 0.98, stats_text, transform=ax_main.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor='white',
                  alpha=0.95, edgecolor='black', linewidth=1),
        verticalalignment='top', fontsize=10, fontweight='bold', zorder=15
    )

    # Legend (unchanged except wording focuses on basins + gage)
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='#FFD700', markersize=18,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Gage {int(gage_id)}', linestyle='None'),
    ]
    if not terminal_basin.empty:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor='#B8860B', edgecolor='#8B7355',
                          alpha=0.8, label='Terminal Basin (Gage Location)')
        )
    if not upstream_basins.empty:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor='#F5E6D3', edgecolor='#D2B48C',
                          alpha=0.6, label='Upstream Basins')
        )

    ax_main.legend(
        handles=legend_elements, loc='lower left', bbox_to_anchor=(0.02, 0.02),
        fontsize=9, frameon=True, fancybox=False, shadow=False,
        edgecolor='black', facecolor='white', framealpha=0.95
    )


def add_mi_statistics_and_legend(fig, ax_main, gage_wells, gage_id,
                                 terminal_basin, upstream_basins):
    """Stats box + legend (MI version for NO LAG)."""
    watershed_info = ""
    if not terminal_basin.empty:
        watershed_info += "Terminal Basin: 1\n"
    if not upstream_basins.empty:
        watershed_info += f"Upstream Basins: {len(upstream_basins)}\n"

    wells_n = len(gage_wells) if gage_wells is not None else 0
    if wells_n > 0 and 'mi_delta_wte_delta_q' in gage_wells.columns:
        mi_series = gage_wells['mi_delta_wte_delta_q'].astype(float)
        mi_series = mi_series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(mi_series) > 0:
            min_mi  = np.nanmin(mi_series)
            mean_mi = np.nanmean(mi_series)
            p90_mi  = np.nanpercentile(mi_series, 90)
            max_mi  = np.nanmax(mi_series)
            stats_text = (
                f"{watershed_info}"
                f"Wells: {wells_n}\n"
                f"Min MI:  {min_mi:.3f}\n"
                f"Mean MI: {mean_mi:.3f}\n"
                f"P90 MI:  {p90_mi:.3f}\n"
                f"Max MI:  {max_mi:.3f}"
            )
        else:
            stats_text = f"{watershed_info}Wells: {wells_n}\nNo valid MI values"
    else:
        stats_text = f"{watershed_info}Wells: {wells_n}\nNo MI column"

    ax_main.text(
        0.02, 0.98, stats_text, transform=ax_main.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor='white',
                  alpha=0.95, edgecolor='black', linewidth=1),
        verticalalignment='top', fontsize=10, fontweight='bold', zorder=15
    )

    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='#FFD700', markersize=18,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Gage {int(gage_id)}', linestyle='None'),
    ]
    if not terminal_basin.empty:
        legend_elements.append(
            plt.Rectangle((0,0), 1,1, facecolor='#B8860B', edgecolor='#8B7355',
                          alpha=0.8, label='Terminal Basin (Gage Location)')
        )
    if not upstream_basins.empty:
        legend_elements.append(
            plt.Rectangle((0,0), 1,1, facecolor='#F5E6D3', edgecolor='#D2B48C',
                          alpha=0.6, label='Upstream Basins')
        )

    ax_main.legend(handles=legend_elements, loc='lower left',
                   bbox_to_anchor=(0.02, 0.02), fontsize=9,
                   frameon=True, fancybox=False, shadow=False,
                   edgecolor='black', facecolor='white', framealpha=0.95)


def add_delta_mi_statistics_and_legend(fig, ax_main, gage_wells, gage_id,
                                       terminal_basin, upstream_basins):
    """Stats box + legend (ΔMI version)."""
    watershed_info = ""
    if not terminal_basin.empty:
        watershed_info += "Terminal Basin: 1\n"
    if not upstream_basins.empty:
        watershed_info += f"Upstream Basins: {len(upstream_basins)}\n"

    wells_n = len(gage_wells) if gage_wells is not None else 0
    if wells_n > 0 and 'delta_mi' in gage_wells.columns:
        dmi = gage_wells['delta_mi'].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(dmi) > 0:
            min_d  = float(np.nanmin(dmi))
            mean_d = float(np.nanmean(dmi))
            p90_d  = float(np.nanpercentile(dmi, 90))
            max_d  = float(np.nanmax(dmi))
            pos_pct = 100.0 * float(np.mean(dmi > 0))

            stats_text = (
                f"{watershed_info}"
                f"Wells: {wells_n}\n"
                f"Min ΔMI:  {min_d:.3f}\n"
                f"Mean ΔMI: {mean_d:.3f}\n"
                f"P90 ΔMI:  {p90_d:.3f}\n"
                f"Max ΔMI:  {max_d:.3f}\n"
                f"% ΔMI>0:  {pos_pct:.1f}%"
            )
        else:
            stats_text = f"{watershed_info}Wells: {wells_n}\nNo valid ΔMI values"
    else:
        stats_text = f"{watershed_info}Wells: {wells_n}\nNo ΔMI column"

    ax_main.text(
        0.02, 0.98, stats_text, transform=ax_main.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor='white',
                  alpha=0.95, edgecolor='black', linewidth=1),
        verticalalignment='top', fontsize=10, fontweight='bold', zorder=15
    )

    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='#FFD700', markersize=18,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Gage {int(gage_id)}', linestyle='None'),
    ]
    if not terminal_basin.empty:
        legend_elements.append(
            plt.Rectangle((0,0), 1,1, facecolor='#B8860B', edgecolor='#8B7355',
                          alpha=0.8, label='Terminal Basin (Gage Location)')
        )
    if not upstream_basins.empty:
        legend_elements.append(
            plt.Rectangle((0,0), 1,1, facecolor='#F5E6D3', edgecolor='#D2B48C',
                          alpha=0.6, label='Upstream Basins')
        )

    ax_main.legend(handles=legend_elements, loc='lower left',
                   bbox_to_anchor=(0.02, 0.02), fontsize=9,
                   frameon=True, fancybox=False, shadow=False,
                   edgecolor='black', facecolor='white', framealpha=0.95)


def calculate_well_gage_correlations(no_lag_data, min_points=10):
    """
    Calculate R-squared values between each well and its associated gage
    using a no-lag relationship: delta_q(t) vs delta_wte(t).

    This function:
    - Drops NaNs in both variables
    - Skips pairs with constant X or Y (to avoid linregress errors)
    - Returns a DataFrame with one row per well–gage pair
    """
    print("=== Calculating Well-Gage Correlations (ΔQ vs ΔWTE_no_lag) ===")

    # Group by well and gage to calculate R-squared
    correlation_results = []
    grouped = no_lag_data.groupby(['well_id', 'gage_id'])
    print(f"Processing {len(grouped)} well-gage pairs...")

    skipped_low_n = 0
    skipped_constant_x = 0
    skipped_constant_y = 0

    for (well_id, gage_id), group in tqdm(grouped, desc="Calculating correlations"):
        # Remove NaN values in the variables used
        clean_data = group.dropna(subset=['delta_wte', 'delta_q'])

        # Enforce minimum observations
        if len(clean_data) < min_points:
            skipped_low_n += 1
            continue

        # Require variability in X and Y
        if clean_data['delta_wte'].nunique() <= 1:
            skipped_constant_x += 1
            continue
        if clean_data['delta_q'].nunique() <= 1:
            skipped_constant_y += 1
            continue

        try:
            # Linear regression of ΔQ(t) on ΔWTE(t) - NO LAG
            slope, intercept, r_value, p_value, std_err = linregress(
                clean_data['delta_wte'],
                clean_data['delta_q']
            )
            r_squared = r_value ** 2

            # Extract coordinates (assumed constant within the group)
            well_lat = clean_data['well_lat'].iloc[0]
            well_lon = clean_data['well_lon'].iloc[0]
            gage_lat = clean_data['gage_lat'].iloc[0]
            gage_lon = clean_data['gage_lon'].iloc[0]

            correlation_results.append({
                'well_id': well_id,
                'gage_id': gage_id,
                'r_squared': r_squared,
                'r_value': r_value,
                'p_value': p_value,
                'n_observations': len(clean_data),
                'well_lat': well_lat,
                'well_lon': well_lon,
                'gage_lat': gage_lat,
                'gage_lon': gage_lon
            })
        except Exception as e:
            # Defensive: this should be very rare after checks above
            print(f"Error calculating correlation for well {well_id}, gage {gage_id}: {e}")

    correlation_df = pd.DataFrame(correlation_results)
    print(f"Successfully calculated correlations for {len(correlation_df)} well-gage pairs")
    print(f"Skipped (n<{min_points}): {skipped_low_n} | constant X: {skipped_constant_x} | constant Y: {skipped_constant_y}")

    return correlation_df


def create_single_watershed_map(
    gage_id,
    correlation_df,
    subbasin_gdf,
    stream_gdf,
    lake_gdf,
    gage_df,
    terminal_relationships,
    save_dir
):
    """
    Create a single clean correlation map with watershed boundaries and inset map.

    Defensive behaviors:
    - If gage has 0 valid wells, still plot the terminal basin + gage star, and a stats box (Wells: 0).
    - If R² is constant (or a single value), avoid normalizer/colorbar errors.
    """
    print(f"\n=== Processing Gage {gage_id} ===")

    # Well correlations available for this gage
    gage_wells = correlation_df[correlation_df['gage_id'] == gage_id].copy()

    # Gage metadata
    gage_info = gage_df[gage_df['id'] == gage_id]
    if gage_info.empty:
        print(f"No gage info found for gage {gage_id}")
        return False

    gage_name = gage_info.iloc[0].get('name', f'Gage {gage_id}')
    gage_lat = gage_info.iloc[0]['latitude']
    gage_lon = gage_info.iloc[0]['longitude']

    # Watersheds: terminal + upstream
    terminal_basin, upstream_basins = get_gage_watersheds(
        gage_id, subbasin_gdf, gage_df, terminal_relationships
    )

    # Sort wells by R-squared and get top 10
    if not gage_wells.empty:
        gage_wells = gage_wells.sort_values('r_squared', ascending=False)
        top_10_wells = gage_wells.head(10)
    else:
        top_10_wells = gage_wells  # empty

    # Figure layout
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=7, rowspan=10)
    ax_main.set_facecolor('white')

    # Plot watershed polygons
    if not terminal_basin.empty:
        terminal_basin.plot(ax=ax_main, color='#B8860B', alpha=0.8,
                            edgecolor='#8B7355', linewidth=2, zorder=1)
        print(f"✅ Plotted terminal basin for gage {gage_id}")
    else:
        print(f"⚠️ No terminal basin found for gage {gage_id}")

    if not upstream_basins.empty:
        upstream_basins.plot(ax=ax_main, color='#F5E6D3', alpha=0.6,
                             edgecolor='#D2B48C', linewidth=1.5, zorder=1)
        print(f"✅ Plotted {len(upstream_basins)} upstream basins for gage {gage_id}")

    # Clip streams/lakes by watershed extent if possible
    all_basins = pd.concat([terminal_basin, upstream_basins], ignore_index=True)
    if not all_basins.empty:
        watershed_union = all_basins.unary_union
        local_streams = stream_gdf[stream_gdf.geometry.intersects(watershed_union)]
        local_lakes = lake_gdf[lake_gdf.geometry.intersects(watershed_union)]
    else:
        # Fallback: extent around the gage and any wells (if exist)
        all_lons = [gage_lon] + (gage_wells['well_lon'].tolist() if not gage_wells.empty else [])
        all_lats = [gage_lat] + (gage_wells['well_lat'].tolist() if not gage_wells.empty else [])
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        buffer = max(max_lon - min_lon, max_lat - min_lat) * 0.3 if (max_lon > min_lon and max_lat > min_lat) else 0.1

        from shapely.geometry import box
        extent_box = box(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
        local_streams = stream_gdf[stream_gdf.geometry.intersects(extent_box)]
        local_lakes = lake_gdf[lake_gdf.geometry.intersects(extent_box)]
        print(f"Using extent-based clipping for gage {gage_id}")

    # Base layers
    if not local_streams.empty:
        local_streams.plot(ax=ax_main, color='#4A90E2', linewidth=1.5, alpha=0.8, zorder=2)
    if not local_lakes.empty:
        local_lakes.plot(ax=ax_main, color='#E6F3FF', alpha=0.8, edgecolor='#4A90E2', linewidth=0.8, zorder=2)

    # Plot ALL wells as small gray circles first
    ax_main.scatter(
        gage_wells['well_lon'],
        gage_wells['well_lat'],
        c='lightgray',
        s=50,
        alpha=0.6,
        edgecolor='gray',
        linewidth=0.5,
        zorder=3
    )

    # Plot wells colored by R-squared using viridis colormap (yellow-green-blue)
    norm = plt.Normalize(vmin=gage_wells['r_squared'].min(), vmax=gage_wells['r_squared'].max())
    scatter = ax_main.scatter(
        gage_wells['well_lon'],
        gage_wells['well_lat'],
        c=gage_wells['r_squared'],
        cmap='viridis',  # Yellow-green-blue colormap
        s=90,
        alpha=0.9,
        edgecolor='black',
        linewidth=0.8,
        zorder=4,
        norm=norm
    )

    # Plot gage as large bright yellow star - NO LABELS
    ax_main.scatter(gage_lon, gage_lat,
              color='#FFD700',  # Bright yellow/gold
              marker='*',       # Star
              s=500,           # Much larger size
              edgecolor='black',  # Black edge
              linewidth=2,
              zorder=10)

    # Add numbers for top 10 wells
    for i, (_, well) in enumerate(top_10_wells.iterrows()):
        number = i + 1

        # Larger red marker for top 10 wells
        ax_main.scatter(well['well_lon'], well['well_lat'],
                  color='red',
                  s=180,
                  marker='o',
                  edgecolor='white',
                  linewidth=2,
                  zorder=8,
                  alpha=0.95)

        # Add number inside the marker
        ax_main.text(well['well_lon'], well['well_lat'],
               str(number),
               ha='center', va='center',
               fontsize=11, fontweight='bold',
               color='white',
               zorder=9)

    # Add R-squared labels with leader lines
    add_labels_with_leader_lines(ax_main, top_10_wells, gage_lon, gage_lat)

    # Set main map extent
    if not all_basins.empty:
        bounds = all_basins.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax_main.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
        ax_main.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
    else:
        # Fallback to well and gage extent
        all_lons = list(gage_wells['well_lon']) + [gage_lon]
        all_lats = list(gage_wells['well_lat']) + [gage_lat]
        lon_buffer = (max(all_lons) - min(all_lons)) * 0.2
        lat_buffer = (max(all_lats) - min(all_lats)) * 0.2
        ax_main.set_xlim(min(all_lons) - lon_buffer, max(all_lons) + lon_buffer)
        ax_main.set_ylim(min(all_lats) - lat_buffer, max(all_lats) + lat_buffer)

    # Style main map
    ax_main.set_title(f'Well–Gage Correlations (No Lag): {gage_name}\nGage ID: {gage_id}',
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('Longitude', fontsize=11)
    ax_main.set_ylabel('Latitude', fontsize=11)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    ax_main.tick_params(labelsize=9)

    # Inset
    ax_inset = plt.subplot2grid((10, 10), (0, 7), colspan=3, rowspan=3)
    ax_inset.set_facecolor('white')
    plot_overview_inset(ax_inset, subbasin_gdf, stream_gdf, lake_gdf,
                        terminal_basin, upstream_basins, gage_lon, gage_lat, gage_id)

    # Colorbar only if we actually plotted colored wells
    if scatter is not None:
        cbar_ax = plt.subplot2grid((10, 10), (4, 7), colspan=3, rowspan=1)
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('R² (Correlation Strength)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

    # Stats + legend
    add_statistics_and_legend(fig, ax_main, gage_wells, gage_id, terminal_basin, upstream_basins)

    # Save
    safe_name = gage_name.replace('/', '_').replace('\\', '_')[:40]
    filename = f"gage_{gage_id}_watershed_no_lag_{safe_name}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()

    print(f"✅ Saved map for gage {gage_id}: {filename}")
    return True


def create_single_styled_map(gage_id, gage_wells, subbasin_gdf, stream_gdf, lake_gdf,
                           gage_df, terminal_relationships, save_dir):
    """Create a single map using the correlation map style."""

    # Get gage info
    gage_info = gage_df[gage_df['id'] == gage_id]
    if gage_info.empty:
        print(f"No gage info for {gage_id}")
        return False

    gage_name = gage_info.iloc[0].get('name', f'Gage {gage_id}')
    gage_lat = gage_info.iloc[0]['latitude']
    gage_lon = gage_info.iloc[0]['longitude']

    # Get watersheds: terminal + upstream
    terminal_basin, upstream_basins = get_gage_watersheds_styled(
        gage_id, subbasin_gdf, gage_df, terminal_relationships
    )

    if terminal_basin.empty:
        print(f"No terminal basin found for gage {gage_id}")
        return False

    print(f"Found terminal basin and {len(upstream_basins)} upstream basins for gage {gage_id}")

    # Sort wells by distance (closest first)
    gage_wells = gage_wells.sort_values('Distance_to_Reach')

    # Figure layout - SAME AS CORRELATION MAPS
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=7, rowspan=10)
    ax_main.set_facecolor('white')

    # Plot watershed polygons - SAME COLORS AS CORRELATION MAPS
    if not terminal_basin.empty:
        terminal_basin.plot(ax=ax_main, color='#B8860B', alpha=0.8,
                            edgecolor='#8B7355', linewidth=2, zorder=1)
        print(f"✅ Plotted terminal basin for gage {gage_id}")

    if not upstream_basins.empty:
        upstream_basins.plot(ax=ax_main, color='#F5E6D3', alpha=0.6,
                             edgecolor='#D2B48C', linewidth=1.5, zorder=1)
        print(f"✅ Plotted {len(upstream_basins)} upstream basins for gage {gage_id}")

    # Clip streams/lakes by watershed extent
    all_basins = pd.concat([terminal_basin, upstream_basins], ignore_index=True)
    if not all_basins.empty:
        watershed_union = all_basins.unary_union
        local_streams = stream_gdf[stream_gdf.geometry.intersects(watershed_union)]
        local_lakes = lake_gdf[lake_gdf.geometry.intersects(watershed_union)]
    else:
        # Fallback: extent around the gage and wells
        all_lons = [gage_lon] + gage_wells['well_lon'].tolist()
        all_lats = [gage_lat] + gage_wells['well_lat'].tolist()
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        buffer = max(max_lon - min_lon, max_lat - min_lat) * 0.3 if (max_lon > min_lon and max_lat > min_lat) else 0.1

        from shapely.geometry import box
        extent_box = box(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
        local_streams = stream_gdf[stream_gdf.geometry.intersects(extent_box)]
        local_lakes = lake_gdf[lake_gdf.geometry.intersects(extent_box)]

    # Base layers - SAME AS CORRELATION MAPS
    if not local_streams.empty:
        local_streams.plot(ax=ax_main, color='#4A90E2', linewidth=1.5, alpha=0.8, zorder=2)
    if not local_lakes.empty:
        local_lakes.plot(ax=ax_main, color='#E6F3FF', alpha=0.8, edgecolor='#4A90E2', linewidth=0.8, zorder=2)

    # Plot wells colored by distance - NO TOP 5 HIGHLIGHTING
    if len(gage_wells) > 0:
        norm = plt.Normalize(
            vmin=gage_wells['Distance_to_Reach'].min(),
            vmax=gage_wells['Distance_to_Reach'].max()
        )

        scatter = ax_main.scatter(
            gage_wells['well_lon'],
            gage_wells['well_lat'],
            c=gage_wells['Distance_to_Reach'],
            cmap='viridis',  # Same as correlation maps
            s=90,
            alpha=0.9,
            edgecolor='black',
            linewidth=0.8,
            zorder=4,
            norm=norm
        )

    # Plot gage as large bright yellow star - SAME AS CORRELATION MAPS
    ax_main.scatter(gage_lon, gage_lat,
              color='#FFD700',  # Bright yellow/gold
              marker='*',       # Star
              s=500,           # Much larger size
              edgecolor='black',  # Black edge
              linewidth=2,
              zorder=10)

    # Set main map extent - SAME AS CORRELATION MAPS
    if not all_basins.empty:
        bounds = all_basins.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax_main.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
        ax_main.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
    else:
        # Fallback to well and gage extent
        all_lons = list(gage_wells['well_lon']) + [gage_lon]
        all_lats = list(gage_wells['well_lat']) + [gage_lat]
        lon_buffer = (max(all_lons) - min(all_lons)) * 0.2
        lat_buffer = (max(all_lats) - min(all_lats)) * 0.2
        ax_main.set_xlim(min(all_lons) - lon_buffer, max(all_lons) + lon_buffer)
        ax_main.set_ylim(min(all_lats) - lat_buffer, max(all_lats) + lat_buffer)

    # Style main map - SAME AS CORRELATION MAPS - FIXED TITLE LINEBREAK
    ax_main.set_title(f'Wells by Distance to Reach:\n {gage_name}\nGage ID: {int(gage_id)}',
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('Longitude', fontsize=11)
    ax_main.set_ylabel('Latitude', fontsize=11)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    ax_main.tick_params(labelsize=9)

    # Inset - SAME AS CORRELATION MAPS
    ax_inset = plt.subplot2grid((10, 10), (0, 7), colspan=3, rowspan=3)
    ax_inset.set_facecolor('white')
    plot_overview_inset_styled(ax_inset, subbasin_gdf, stream_gdf, lake_gdf,
                               terminal_basin, upstream_basins, gage_lon, gage_lat, gage_id)

    # Colorbar - SAME LAYOUT AS CORRELATION MAPS
    if len(gage_wells) > 0:
        cbar_ax = plt.subplot2grid((10, 10), (4, 7), colspan=3, rowspan=1)
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Distance to Reach (meters)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

    # Add statistics and legend - SAME STYLE AS CORRELATION MAPS
    add_distance_statistics_and_legend_styled(fig, ax_main, gage_wells, gage_id,
                                            terminal_basin, upstream_basins)

    # Save
    safe_name = gage_name.replace('/', '_').replace('\\', '_')[:40]
    filename = f"gage_{gage_id}_distance_{safe_name}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()

    print(f"✅ Saved map for gage {gage_id}: {filename}")
    return True


def create_single_styled_map_vertical(gage_id, gage_wells, subbasin_gdf, stream_gdf, lake_gdf,
                           gage_df, terminal_relationships, save_dir):
    """Create a single map using the correlation map style, colored by mutual information."""

    # Get gage info
    gage_info = gage_df[gage_df['id'] == gage_id]
    if gage_info.empty:
        print(f"No gage info for {gage_id}")
        return False

    gage_name = gage_info.iloc[0].get('name', f'Gage {gage_id}')
    gage_lat = gage_info.iloc[0]['latitude']
    gage_lon = gage_info.iloc[0]['longitude']

    # Get watersheds: terminal + upstream
    terminal_basin, upstream_basins = get_gage_watersheds_styled(
        gage_id, subbasin_gdf, gage_df, terminal_relationships
    )

    if terminal_basin.empty:
        print(f"No terminal basin found for gage {gage_id}")
        return False

    print(f"Found terminal basin and {len(upstream_basins)} upstream basins for gage {gage_id}")

    # Sort wells by mutual information (highest first)
    gage_wells = gage_wells.sort_values('mi_delta_wte_delta_q', ascending=False)

    # Figure layout - SAME AS CORRELATION MAPS
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=7, rowspan=10)
    ax_main.set_facecolor('white')

    # Plot watershed polygons - SAME COLORS AS CORRELATION MAPS
    if not terminal_basin.empty:
        terminal_basin.plot(ax=ax_main, color='#B8860B', alpha=0.8,
                            edgecolor='#8B7355', linewidth=2, zorder=1)
        print(f"✅ Plotted terminal basin for gage {gage_id}")

    if not upstream_basins.empty:
        upstream_basins.plot(ax=ax_main, color='#F5E6D3', alpha=0.6,
                             edgecolor='#D2B48C', linewidth=1.5, zorder=1)
        print(f"✅ Plotted {len(upstream_basins)} upstream basins for gage {gage_id}")

    # Clip streams/lakes by watershed extent
    all_basins = pd.concat([terminal_basin, upstream_basins], ignore_index=True)
    if not all_basins.empty:
        watershed_union = all_basins.unary_union
        local_streams = stream_gdf[stream_gdf.geometry.intersects(watershed_union)]
        local_lakes = lake_gdf[lake_gdf.geometry.intersects(watershed_union)]
    else:
        # Fallback: extent around the gage and wells
        all_lons = [gage_lon] + gage_wells['well_lon'].tolist()
        all_lats = [gage_lat] + gage_wells['well_lat'].tolist()
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        buffer = max(max_lon - min_lon, max_lat - min_lat) * 0.3 if (max_lon > min_lon and max_lat > min_lat) else 0.1

        from shapely.geometry import box
        extent_box = box(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
        local_streams = stream_gdf[stream_gdf.geometry.intersects(extent_box)]
        local_lakes = lake_gdf[lake_gdf.geometry.intersects(extent_box)]

    # Base layers - SAME AS CORRELATION MAPS
    if not local_streams.empty:
        local_streams.plot(ax=ax_main, color='#4A90E2', linewidth=1.5, alpha=0.8, zorder=2)
    if not local_lakes.empty:
        local_lakes.plot(ax=ax_main, color='#E6F3FF', alpha=0.8, edgecolor='#4A90E2', linewidth=0.8, zorder=2)

    # Plot wells colored by mutual information
    if len(gage_wells) > 0:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        scatter = ax_main.scatter(
            gage_wells['well_lon'],
            gage_wells['well_lat'],
            c=gage_wells['mi_delta_wte_delta_q'],
            cmap='viridis',  # Yellow-green-blue colormap
            s=90,
            alpha=0.9,
            edgecolor='black',
            linewidth=0.8,
            zorder=4,
            norm=norm
        )

    # Plot gage as large bright yellow star - SAME AS CORRELATION MAPS
    ax_main.scatter(gage_lon, gage_lat,
              color='#FFD700',  # Bright yellow/gold
              marker='*',       # Star
              s=500,           # Much larger size
              edgecolor='black',  # Black edge
              linewidth=2,
              zorder=10)

    # Set main map extent - SAME AS CORRELATION MAPS
    if not all_basins.empty:
        bounds = all_basins.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax_main.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
        ax_main.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
    else:
        # Fallback to well and gage extent
        all_lons = list(gage_wells['well_lon']) + [gage_lon]
        all_lats = list(gage_wells['well_lat']) + [gage_lat]
        lon_buffer = (max(all_lons) - min(all_lons)) * 0.2
        lat_buffer = (max(all_lats) - min(all_lats)) * 0.2
        ax_main.set_xlim(min(all_lons) - lon_buffer, max(all_lons) + lon_buffer)
        ax_main.set_ylim(min(all_lats) - lat_buffer, max(all_lats) + lat_buffer)

    # Style main map - SAME AS CORRELATION MAPS - FIXED TITLE LINEBREAK
    ax_main.set_title(f'Wells by Mutual Information (1-year lag):\n {gage_name}\nGage ID: {gage_id}',
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('Longitude', fontsize=11)
    ax_main.set_ylabel('Latitude', fontsize=11)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    ax_main.tick_params(labelsize=9)

    # Inset - SAME AS CORRELATION MAPS
    ax_inset = plt.subplot2grid((10, 10), (0, 7), colspan=3, rowspan=3)
    ax_inset.set_facecolor('white')
    plot_overview_inset_styled(ax_inset, subbasin_gdf, stream_gdf, lake_gdf,
                               terminal_basin, upstream_basins, gage_lon, gage_lat, gage_id)

    # Colorbar - SAME LAYOUT AS CORRELATION MAPS
    if len(gage_wells) > 0:
        cbar_ax = plt.subplot2grid((10, 10), (4, 7), colspan=3, rowspan=1)
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Mutual Information (MI)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

    # Add statistics and legend - SAME STYLE AS CORRELATION MAPS
    add_vertical_distance_statistics_and_legend_styled(fig, ax_main, gage_wells, gage_id,
                                            terminal_basin, upstream_basins)

    # Save
    safe_name = gage_name.replace('/', '_').replace('\\', '_')[:40]
    filename = f"gage_{int(gage_id)}_mi_{safe_name}_lag.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()

    print(f"✅ Saved map for gage {gage_id}: {filename}")
    return True


def create_single_styled_map_mi_no_lag(gage_id, gage_wells, subbasin_gdf, stream_gdf, lake_gdf,
                                       gage_df, terminal_relationships, save_dir):
    """Create a single NO-LAG MI map, wells colored by mi_delta_wte_delta_q."""
    gage_info = gage_df[gage_df['id'] == gage_id]
    if gage_info.empty:
        print(f"No gage info for {gage_id}")
        return False

    gage_name = gage_info.iloc[0].get('name', f'Gage {gage_id}')
    gage_lat  = gage_info.iloc[0]['latitude']
    gage_lon  = gage_info.iloc[0]['longitude']

    terminal_basin, upstream_basins = get_gage_watersheds_styled(
        gage_id, subbasin_gdf, gage_df, terminal_relationships
    )
    if terminal_basin.empty:
        print(f"No terminal basin found for gage {gage_id}")
        return False

    # Sort wells by MI
    gage_wells = gage_wells.sort_values('mi_delta_wte_delta_q', ascending=False)

    # --- Figure layout (same as correlation maps)
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=7, rowspan=10)
    ax_main.set_facecolor('white')

    if not terminal_basin.empty:
        terminal_basin.plot(ax=ax_main, color='#B8860B', alpha=0.8,
                            edgecolor='#8B7355', linewidth=2, zorder=1)
    if not upstream_basins.empty:
        upstream_basins.plot(ax=ax_main, color='#F5E6D3', alpha=0.6,
                             edgecolor='#D2B48C', linewidth=1.5, zorder=1)

    # Clip base layers by watershed extent
    all_basins = pd.concat([terminal_basin, upstream_basins], ignore_index=True)
    if not all_basins.empty:
        watershed_union = all_basins.unary_union
        local_streams = stream_gdf[stream_gdf.geometry.intersects(watershed_union)]
        local_lakes   = lake_gdf[lake_gdf.geometry.intersects(watershed_union)]
    else:
        from shapely.geometry import box
        all_lons = [gage_lon] + gage_wells['well_lon'].tolist()
        all_lats = [gage_lat] + gage_wells['well_lat'].tolist()
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        buffer = max(max_lon - min_lon, max_lat - min_lat) * 0.3 if (max_lon > min_lon and max_lat > min_lat) else 0.1
        extent_box = box(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
        local_streams = stream_gdf[stream_gdf.geometry.intersects(extent_box)]
        local_lakes   = lake_gdf[lake_gdf.geometry.intersects(extent_box)]

    if not local_streams.empty:
        local_streams.plot(ax=ax_main, color='#4A90E2', linewidth=1.5, alpha=0.8, zorder=2)
    if not local_lakes.empty:
        local_lakes.plot(ax=ax_main, color='#E6F3FF', alpha=0.8, edgecolor='#4A90E2', linewidth=0.8, zorder=2)

    # Scatter wells colored by MI (robust color range)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        scatter = ax_main.scatter(
            gage_wells['well_lon'], gage_wells['well_lat'],
            c=gage_wells['mi_delta_wte_delta_q'],
            cmap='viridis', s=90, alpha=0.9,
            edgecolor='black', linewidth=0.8, zorder=4, norm=norm
        )

    # Gage star
    ax_main.scatter(gage_lon, gage_lat, color='#FFD700', marker='*', s=500,
                    edgecolor='black', linewidth=2, zorder=10)

    # Extent
    if not all_basins.empty:
        bounds = all_basins.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax_main.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
        ax_main.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
    else:
        all_lons = list(gage_wells['well_lon']) + [gage_lon]
        all_lats = list(gage_wells['well_lat']) + [gage_lat]
        lon_buffer = (max(all_lons) - min(all_lons)) * 0.2
        lat_buffer = (max(all_lats) - min(all_lats)) * 0.2
        ax_main.set_xlim(min(all_lons) - lon_buffer, max(all_lons) + lon_buffer)
        ax_main.set_ylim(min(all_lats) - lat_buffer, max(all_lats) + lat_buffer)

    # Titles / axes
    ax_main.set_title(f'Wells by Mutual Information (NO LAG):\n {gage_name}\nGage ID: {gage_id}',
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('Longitude', fontsize=11)
    ax_main.set_ylabel('Latitude', fontsize=11)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    ax_main.tick_params(labelsize=9)

    # Inset
    ax_inset = plt.subplot2grid((10, 10), (0, 7), colspan=3, rowspan=3)
    ax_inset.set_facecolor('white')
    plot_overview_inset_styled(ax_inset, subbasin_gdf, stream_gdf, lake_gdf,
                               terminal_basin, upstream_basins, gage_lon, gage_lat, gage_id)

    # Colorbar
    if len(gage_wells) > 0:
        cbar_ax = plt.subplot2grid((10, 10), (4, 7), colspan=3, rowspan=1)
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Mutual Information (MI)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

    # Stats + legend
    add_mi_statistics_and_legend(fig, ax_main, gage_wells, gage_id, terminal_basin, upstream_basins)

    # Save
    safe_name = gage_name.replace('/', '_').replace('\\', '_')[:40]
    filename = f"gage_{int(gage_id)}_mi_no_lag_{safe_name}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    print(f"✅ Saved map for gage {gage_id}: {filename}")
    return True


def create_single_styled_map_mi_delta(gage_id, gage_wells, subbasin_gdf, stream_gdf, lake_gdf,
                                      gage_df, terminal_relationships, save_dir,
                                      dmi_abs_max_global=None):
    """
    Create a single ΔMI map, wells colored by delta_mi, and annotate the top-10 wells
    by absolute change |delta_mi| with callouts (arrow + label box).

    Parameters
    ----------
    gage_id : int
    gage_wells : DataFrame
        Must contain columns: ['well_lon','well_lat','delta_mi', 'well_id']
    dmi_abs_max_global : float or None
        If provided, use this as symmetric color scale ±dmi_abs_max_global; otherwise
        compute from current gage_wells (robust ±95%).
    """
    # ---- gage info
    gage_info = gage_df[gage_df['id'] == gage_id]
    if gage_info.empty:
        print(f"No gage info for {gage_id}")
        return False

    gage_name = gage_info.iloc[0].get('name', f'Gage {gage_id}')
    gage_lat  = gage_info.iloc[0]['latitude']
    gage_lon  = gage_info.iloc[0]['longitude']

    # ---- watersheds
    terminal_basin, upstream_basins = get_gage_watersheds_styled(
        gage_id, subbasin_gdf, gage_df, terminal_relationships
    )
    if terminal_basin.empty:
        print(f"No terminal basin found for gage {gage_id}")
        return False

    # ---- sort for plotting (largest positive first is fine, we will select by abs below)
    gage_wells = gage_wells.copy()
    gage_wells = gage_wells.sort_values('delta_mi', ascending=False)

    # ---- figure layout
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=7, rowspan=10)
    ax_main.set_facecolor('white')

    # polygons
    if not terminal_basin.empty:
        terminal_basin.plot(ax=ax_main, color='#B8860B', alpha=0.8,
                            edgecolor='#8B7355', linewidth=2, zorder=1)
    if not upstream_basins.empty:
        upstream_basins.plot(ax=ax_main, color='#F5E6D3', alpha=0.6,
                             edgecolor='#D2B48C', linewidth=1.5, zorder=1)

    # clip base layers
    all_basins = pd.concat([terminal_basin, upstream_basins], ignore_index=True)
    if not all_basins.empty:
        watershed_union = all_basins.unary_union
        local_streams = stream_gdf[stream_gdf.geometry.intersects(watershed_union)]
        local_lakes   = lake_gdf[lake_gdf.geometry.intersects(watershed_union)]
    else:
        from shapely.geometry import box
        all_lons = [gage_lon] + gage_wells['well_lon'].tolist()
        all_lats = [gage_lat] + gage_wells['well_lat'].tolist()
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        buffer = max(max_lon - min_lon, max_lat - min_lat) * 0.3 if (max_lon > min_lon and max_lat > min_lat) else 0.1
        extent_box = box(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
        local_streams = stream_gdf[stream_gdf.geometry.intersects(extent_box)]
        local_lakes   = lake_gdf[lake_gdf.geometry.intersects(extent_box)]

    if not local_streams.empty:
        local_streams.plot(ax=ax_main, color='#4A90E2', linewidth=1.5, alpha=0.8, zorder=2)
    if not local_lakes.empty:
        local_lakes.plot(ax=ax_main, color='#E6F3FF', alpha=0.8, edgecolor='#4A90E2', linewidth=0.8, zorder=2)

    # ---- color normalization (global symmetric optional)
    dmi_vals = gage_wells['delta_mi'].astype(float).replace([np.inf, -np.inf], np.nan)
    if dmi_abs_max_global is not None and np.isfinite(dmi_abs_max_global) and dmi_abs_max_global > 0:
        vlim = float(dmi_abs_max_global)
    else:
        # robust symmetric scaling by local 95% percentile of |ΔMI|
        p = 95
        vmax = np.nanpercentile(np.abs(dmi_vals.dropna()), p) if dmi_vals.notna().any() else 1e-6
        vlim = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1e-6
    norm = plt.Normalize(vmin=-vlim, vmax=vlim)

    # ---- scatter of wells
    scatter = None
    if len(gage_wells) > 0:
        scatter = ax_main.scatter(
            gage_wells['well_lon'], gage_wells['well_lat'],
            c=gage_wells['delta_mi'], cmap='coolwarm', norm=norm,
            s=90, alpha=0.9, edgecolor='black', linewidth=0.8, zorder=4
        )

    # ---- gage star
    ax_main.scatter(gage_lon, gage_lat, color='#FFD700', marker='*', s=500,
                    edgecolor='black', linewidth=2, zorder=10)

    # ---- extent
    if not all_basins.empty:
        bounds = all_basins.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax_main.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
        ax_main.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
    else:
        all_lons = list(gage_wells['well_lon']) + [gage_lon]
        all_lats = list(gage_wells['well_lat']) + [gage_lat]
        lon_buffer = (max(all_lons) - min(all_lons)) * 0.2
        lat_buffer = (max(all_lats) - min(all_lats)) * 0.2
        ax_main.set_xlim(min(all_lons) - lon_buffer, max(all_lons) + lon_buffer)
        ax_main.set_ylim(min(all_lats) - lat_buffer, max(all_lats) + lat_buffer)

    # ---- titles / axes
    ax_main.set_title(f'Wells by ΔMI (Lag − No-lag):\n {gage_name}\nGage ID: {gage_id}',
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('Longitude', fontsize=11)
    ax_main.set_ylabel('Latitude', fontsize=11)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    ax_main.tick_params(labelsize=9)

    # ---- inset
    ax_inset = plt.subplot2grid((10, 10), (0, 7), colspan=3, rowspan=3)
    ax_inset.set_facecolor('white')
    plot_overview_inset_styled(ax_inset, subbasin_gdf, stream_gdf, lake_gdf,
                               terminal_basin, upstream_basins, gage_lon, gage_lat, gage_id)

    # ---- colorbar
    if scatter is not None:
        cbar_ax = plt.subplot2grid((10, 10), (4, 7), colspan=3, rowspan=1)
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('ΔMI = MI(lag) − MI(no-lag)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

    # ---- annotate top-10 by |ΔMI|
    if len(gage_wells) > 0:
        # Clean NaN; requires well_lon/lat, delta_mi, well_id
        annot_df = gage_wells[['well_id_numeric', 'well_lon', 'well_lat', 'delta_mi']].dropna(subset=['well_lon','well_lat','delta_mi'])
        if len(annot_df) > 0:
            topN = min(10, len(annot_df))
            top_wells = annot_df.reindex(
                annot_df['delta_mi'].abs().sort_values(ascending=False).index
            ).head(topN)

            # Calculate offset, adaptive to map extent
            x0, x1 = ax_main.get_xlim()
            y0, y1 = ax_main.get_ylim()
            dx = (x1 - x0) * 0.05
            dy = (y1 - y0) * 0.05

            # A set of scattered offsets to minimize overlap
            offsets = [
                ( dx,  dy), (-dx,  dy), ( dx, -dy), (-dx, -dy),
                (2*dx,  0), (-2*dx, 0), (0,  2*dy), (0, -2*dy),
                (1.5*dx, -1.5*dy), (-1.5*dx, 1.5*dy)
            ]
            # Add bold outline circles to top wells
            ax_main.scatter(
                top_wells['well_lon'], top_wells['well_lat'],
                facecolors='none', edgecolors='black', linewidths=2.0,
                s=160, zorder=12
            )

            for i, (_, row) in enumerate(top_wells.iterrows()):
                ox, oy = offsets[i % len(offsets)]
                label = f"Well {int(row['well_id_numeric'])}\nΔMI={row['delta_mi']:.3f}"
                ax_main.annotate(
                    label,
                    xy=(row['well_lon'], row['well_lat']),             # anchor point
                    xytext=(row['well_lon'] + ox, row['well_lat'] + oy),  # label box
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
                    fontsize=8, zorder=20
                )

    # ---- stats + legend
    add_delta_mi_statistics_and_legend(fig, ax_main, gage_wells, gage_id, terminal_basin, upstream_basins)

    # ---- save
    safe_name = gage_name.replace('/', '_').replace('\\', '_')[:40]
    filename = f"gage_{int(gage_id)}_delta_mi_{safe_name}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    print(f"✅ Saved map for gage {gage_id}: {filename}")
    return True


def create_single_styled_map_mi_delta_fixed(gage_id, gage_wells, subbasin_gdf, stream_gdf, lake_gdf,
                                           gage_df, terminal_relationships, save_dir,
                                           dmi_abs_max_global=None):
    """
    Create a single ΔMI map with FIXED top-10 annotation and RED highlights.
    Wells colored by delta_mi, and annotate the top-10 wells with red circles and labels.
    """
    # ---- gage info
    gage_info = gage_df[gage_df['id'] == gage_id]
    if gage_info.empty:
        print(f"No gage info for {gage_id}")
        return False

    gage_name = gage_info.iloc[0].get('name', f'Gage {gage_id}')
    gage_lat  = gage_info.iloc[0]['latitude']
    gage_lon  = gage_info.iloc[0]['longitude']

    # ---- watersheds
    terminal_basin, upstream_basins = get_gage_watersheds_styled(
        gage_id, subbasin_gdf, gage_df, terminal_relationships
    )
    if terminal_basin.empty:
        print(f"No terminal basin found for gage {gage_id}")
        return False

    # ---- sort for plotting 
    gage_wells = gage_wells.copy()
    gage_wells = gage_wells.sort_values('delta_mi', ascending=False)

    # ---- figure layout
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=7, rowspan=10)
    ax_main.set_facecolor('white')

    # polygons
    if not terminal_basin.empty:
        terminal_basin.plot(ax=ax_main, color='#B8860B', alpha=0.8,
                            edgecolor='#8B7355', linewidth=2, zorder=1)
    if not upstream_basins.empty:
        upstream_basins.plot(ax=ax_main, color='#F5E6D3', alpha=0.6,
                             edgecolor='#D2B48C', linewidth=1.5, zorder=1)

    # clip base layers
    all_basins = pd.concat([terminal_basin, upstream_basins], ignore_index=True)
    if not all_basins.empty:
        watershed_union = all_basins.unary_union
        local_streams = stream_gdf[stream_gdf.geometry.intersects(watershed_union)]
        local_lakes   = lake_gdf[lake_gdf.geometry.intersects(watershed_union)]
    else:
        from shapely.geometry import box
        all_lons = [gage_lon] + gage_wells['well_lon'].tolist()
        all_lats = [gage_lat] + gage_wells['well_lat'].tolist()
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        buffer = max(max_lon - min_lon, max_lat - min_lat) * 0.3 if (max_lon > min_lon and max_lat > min_lat) else 0.1
        extent_box = box(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
        local_streams = stream_gdf[stream_gdf.geometry.intersects(extent_box)]
        local_lakes   = lake_gdf[lake_gdf.geometry.intersects(extent_box)]

    if not local_streams.empty:
        local_streams.plot(ax=ax_main, color='#4A90E2', linewidth=1.5, alpha=0.8, zorder=2)
    if not local_lakes.empty:
        local_lakes.plot(ax=ax_main, color='#E6F3FF', alpha=0.8, edgecolor='#4A90E2', linewidth=0.8, zorder=2)

    # ---- color normalization
    dmi_vals = gage_wells['delta_mi'].astype(float).replace([np.inf, -np.inf], np.nan)
    if dmi_abs_max_global is not None and np.isfinite(dmi_abs_max_global) and dmi_abs_max_global > 0:
        vlim = float(dmi_abs_max_global)
    else:
        # robust symmetric scaling by local 95% percentile of |ΔMI|
        p = 95
        vmax = np.nanpercentile(np.abs(dmi_vals.dropna()), p) if dmi_vals.notna().any() else 1e-6
        vlim = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1e-6
    norm = plt.Normalize(vmin=-vlim, vmax=vlim)

    # ---- scatter of wells
    scatter = None
    if len(gage_wells) > 0:
        scatter = ax_main.scatter(
            gage_wells['well_lon'], gage_wells['well_lat'],
            c=gage_wells['delta_mi'], cmap='coolwarm', norm=norm,
            s=90, alpha=0.9, edgecolor='black', linewidth=0.8, zorder=4
        )

    # ---- gage star
    ax_main.scatter(gage_lon, gage_lat, color='#FFD700', marker='*', s=500,
                    edgecolor='black', linewidth=2, zorder=10)

    # ---- extent
    if not all_basins.empty:
        bounds = all_basins.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax_main.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
        ax_main.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
    else:
        all_lons = list(gage_wells['well_lon']) + [gage_lon]
        all_lats = list(gage_wells['well_lat']) + [gage_lat]
        lon_buffer = (max(all_lons) - min(all_lons)) * 0.2
        lat_buffer = (max(all_lats) - min(all_lats)) * 0.2
        ax_main.set_xlim(min(all_lons) - lon_buffer, max(all_lons) + lon_buffer)
        ax_main.set_ylim(min(all_lats) - lat_buffer, max(all_lats) + lat_buffer)

    # ---- titles / axes
    ax_main.set_title(f'Wells by ΔMI (Lag − No-lag) - TOP 10 HIGHLIGHTED:\n {gage_name}\nGage ID: {gage_id}',
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('Longitude', fontsize=11)
    ax_main.set_ylabel('Latitude', fontsize=11)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    ax_main.tick_params(labelsize=9)

    # ---- inset
    ax_inset = plt.subplot2grid((10, 10), (0, 7), colspan=3, rowspan=3)
    ax_inset.set_facecolor('white')
    plot_overview_inset_styled(ax_inset, subbasin_gdf, stream_gdf, lake_gdf,
                               terminal_basin, upstream_basins, gage_lon, gage_lat, gage_id)

    # ---- colorbar
    if scatter is not None:
        cbar_ax = plt.subplot2grid((10, 10), (4, 7), colspan=3, rowspan=1)
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('ΔMI = MI(lag) − MI(no-lag)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

    # ---- FIXED: annotate top-10 by |ΔMI| with RED highlights and smart positioning
    if len(gage_wells) > 0:
        # Fix column name issue: use well_id_numeric instead of well_id
        required_cols = ['well_id_numeric', 'well_lon', 'well_lat', 'delta_mi']
        if all(col in gage_wells.columns for col in required_cols):
            annot_df = gage_wells[required_cols].dropna(subset=['well_lon','well_lat','delta_mi'])
            if len(annot_df) > 0:
                topN = min(10, len(annot_df))
                top_wells = annot_df.reindex(
                    annot_df['delta_mi'].abs().sort_values(ascending=False).index
                ).head(topN)

                # Get plot boundaries for adaptive positioning
                x0, x1 = ax_main.get_xlim()
                y0, y1 = ax_main.get_ylim()
                dx = (x1 - x0) * 0.08  # Increased offset for better separation
                dy = (y1 - y0) * 0.08

                # RED outline circles for top 10 wells
                ax_main.scatter(
                    top_wells['well_lon'], top_wells['well_lat'],
                    facecolors='none', edgecolors='red', linewidths=2.5,  # Changed to red
                    s=180, zorder=12, alpha=0.9
                )

                # Smart positioning algorithm to avoid overlaps
                placed_positions = [(gage_lon, gage_lat)]  # Avoid gage position
                
                for i, (_, row) in enumerate(top_wells.iterrows()):
                    well_x, well_y = row['well_lon'], row['well_lat']
                    
                    # Try different positions around the well
                    candidate_positions = [
                        (well_x + dx, well_y + dy),    # NE
                        (well_x - dx, well_y + dy),    # NW  
                        (well_x + dx, well_y - dy),    # SE
                        (well_x - dx, well_y - dy),    # SW
                        (well_x + 1.5*dx, well_y),     # E
                        (well_x - 1.5*dx, well_y),     # W
                        (well_x, well_y + 1.5*dy),     # N
                        (well_x, well_y - 1.5*dy),     # S
                        (well_x + 2*dx, well_y + 0.5*dy),  # Far NE
                        (well_x - 2*dx, well_y - 0.5*dy),  # Far SW
                    ]
                    
                    # Find the position with minimum conflicts
                    best_pos = candidate_positions[0]
                    min_conflicts = float('inf')
                    
                    for candidate_x, candidate_y in candidate_positions:
                        # Count conflicts with existing positions
                        conflict_count = 0
                        min_distance = float('inf')
                        
                        for placed_x, placed_y in placed_positions:
                            distance = np.sqrt((candidate_x - placed_x)**2 + (candidate_y - placed_y)**2)
                            min_distance = min(min_distance, distance)
                            if distance < dx * 1.2:  # Too close
                                conflict_count += 1
                        
                        # Check if position is within plot bounds
                        if not (x0 <= candidate_x <= x1 and y0 <= candidate_y <= y1):
                            conflict_count += 10  # Heavy penalty for out of bounds
                            
                        if conflict_count < min_conflicts or (conflict_count == min_conflicts and min_distance > dx * 0.8):
                            min_conflicts = conflict_count
                            best_pos = (candidate_x, candidate_y)
                    
                    placed_positions.append(best_pos)
                    label_x, label_y = best_pos
                    
                    # Create label
                    label = f"#{i+1}: Well {int(row['well_id_numeric'])}\nΔMI={row['delta_mi']:.3f}"
                    
                    # Draw arrow from well to label
                    ax_main.annotate(
                        label,
                        xy=(well_x, well_y),           # anchor point (well location)
                        xytext=(label_x, label_y),     # label position
                        textcoords='data',
                        arrowprops=dict(arrowstyle="->", color='red', lw=1.3),  # Red arrows
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                edgecolor="red", linewidth=1.2, alpha=0.95),    # Red borders
                        fontsize=8, fontweight='bold', zorder=20,
                        ha='center', va='center'
                    )

    # ---- stats + legend
    add_delta_mi_statistics_and_legend(fig, ax_main, gage_wells, gage_id, terminal_basin, upstream_basins)

    # ---- save
    safe_name = gage_name.replace('/', '_').replace('\\', '_')[:40]
    filename = f"gage_{int(gage_id)}_delta_mi_{safe_name}_FIXED.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    print(f"✅ Saved FIXED map for gage {gage_id}: {filename}")
    return True


def create_summary_statistics(correlation_df):
    """
    Create summary statistics and save CSVs.
    """
    print("=== Creating Summary Statistics ===")
    correlation_df.to_csv('../data/processed/well_gage_correlations_no_lag.csv', index=False)

    if not correlation_df.empty:
        gage_summary = correlation_df.groupby('gage_id').agg({
            'r_squared': ['count', 'mean', 'std', 'max', 'min'],
            'n_observations': ['mean', 'sum']
        }).round(4)
        gage_summary.columns = ['_'.join(col) for col in gage_summary.columns]
        gage_summary.to_csv('../data/processed/gage_correlation_summary_no_lag.csv')

        print("✅ Summary statistics saved")
        print(f"Overall correlation statistics:")
        print(f"  Mean R²: {correlation_df['r_squared'].mean():.4f}")
        print(f"  Median R²: {correlation_df['r_squared'].median():.4f}")
        print(f"  Max R²: {correlation_df['r_squared'].max():.4f}")
        print(f"  Wells with R² > 0.1: {(correlation_df['r_squared'] > 0.1).sum()}")
        print(f"  Wells with R² > 0.5: {(correlation_df['r_squared'] > 0.5).sum()}")
    else:
        print("⚠️ correlation_df is empty; no summary CSVs created.")


def create_clean_correlation_maps_with_watersheds(
    correlation_df,
    no_lag_data,
    save_dir='../reports/figures/gage_well_correlations_watershed_no_lag'
):
    """
    Create clean correlation maps with watershed boundaries and overview inset.

    Key changes:
    - Iterate over gages found in the *input no_lag_data*, not only those that survived correlation,
      so gages with 0 valid wells still get a QA map (terminal basin + star).
    - Defensive plotting when no wells exist.
    """
    print("=== Creating Clean Correlation Maps with Watersheds (NO LAG) ===")
    os.makedirs(save_dir, exist_ok=True)

    # Load geographic data
    try:
        subbasin_gdf = gpd.read_file('../data/raw/hydrography/gsl_catchment.shp')
        stream_gdf = gpd.read_file('../data/raw/hydrography/gslb_stream.shp')
        lake_gdf = gpd.read_file('../data/raw/hydrography/lake.shp')
        gage_df = pd.read_csv('../data/raw/hydrography/gsl_nwm_gage.csv')

        # Load watershed relationships
        terminal_relationships = load_watershed_data()

        print("✅ Geographic data loaded successfully")
        print(f"Subbasin columns: {list(subbasin_gdf.columns)}")
        print(f"Subbasin CRS: {subbasin_gdf.crs}")
        print(f"Total catchments: {len(subbasin_gdf)}")
    except Exception as e:
        print(f"❌ Error loading geographic data: {e}")
        return

    # Use all gages from the input no_lag dataset to ensure coverage
    unique_gages = pd.Series(no_lag_data['gage_id'].dropna().unique()).astype(int).tolist()
    print(f"Creating watershed maps for {len(unique_gages)} gages...")

    for gage_id in tqdm(unique_gages, desc="Creating watershed correlation maps"):
        try:
            result = create_single_watershed_map(
                gage_id,
                correlation_df,
                subbasin_gdf,
                stream_gdf,
                lake_gdf,
                gage_df,
                terminal_relationships,
                save_dir
            )
            if result is False:
                print(f"⚠️ Skipped map creation for Gage {gage_id}")
        except Exception as e:
            print(f"❌ Failed to create map for Gage {gage_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ All watershed correlation maps saved to: {save_dir}")


def create_watershed_distance_maps_styled():
    """
    Create watershed maps for each gage using the same style as correlation maps.
    Uses well_gage data for well-gage relationships.
    """
    print("=== Creating Watershed Distance Maps (Styled Version) ===")
    save_dir = "../reports/figures/watershed_distance_maps_styled"
    os.makedirs(save_dir, exist_ok=True)

    # Load all datasets
    print("Loading datasets...")

    # Geographic data
    subbasin_gdf = gpd.read_file('../data/raw/hydrography/gsl_catchment.shp')
    stream_gdf = gpd.read_file('../data/raw/hydrography/gslb_stream.shp')
    lake_gdf = gpd.read_file('../data/raw/hydrography/lake.shp')
    gage_df = pd.read_csv('../data/raw/hydrography/gsl_nwm_gage.csv')

    # Well data
    reach_distances = pd.read_csv('../data/processed/well_reach_relationships_final.csv')
    well_locations = pd.read_csv('../data/processed/well_reach.csv')[['well_id', 'well_lat', 'well_lon']].drop_duplicates()

    # CHANGED: Use well_gage data for well-gage relationships
    well_gage_data = pd.read_csv('../data/processed/wells_with_catchment_info.csv')
    well_gage_pairs = well_gage_data[['well_id', 'gage_id']].dropna().drop_duplicates()
    print(f"Using well_gage data for well-gage relationships: {len(well_gage_pairs)} unique pairs")

    # Load watershed relationships
    try:
        terminal_relationships = pd.read_csv('../data/processed/terminal_gage_upstream_catchments.csv')
        print("✅ Loaded terminal relationships")
    except FileNotFoundError:
        print("⚠️ Terminal relationships file not found, using simplified approach")
        terminal_relationships = None

    print("✅ All datasets loaded")

    # Merge data using numeric IDs
    print("Merging data using numeric approach...")

    # Convert to numeric
    reach_distances['well_id_numeric'] = pd.to_numeric(reach_distances['Well_ID'], errors='coerce')
    well_locations['well_id_numeric'] = pd.to_numeric(well_locations['well_id'], errors='coerce')
    well_gage_pairs['well_id_numeric'] = pd.to_numeric(well_gage_pairs['well_id'], errors='coerce')

    # Merge step by step
    merge1 = pd.merge(reach_distances, well_locations, on='well_id_numeric', how='inner')
    print(f"After adding coordinates: {len(merge1)} wells")

    final_data = pd.merge(merge1, well_gage_pairs, on='well_id_numeric', how='inner')
    print(f"Final merged data: {len(final_data)} wells with gage relationships")

    if len(final_data) == 0:
        print("❌ No data after merging - cannot create maps")
        return

    # Ensure CRS consistency
    subbasin_gdf = subbasin_gdf.to_crs("EPSG:4326")
    stream_gdf = stream_gdf.to_crs("EPSG:4326")
    lake_gdf = lake_gdf.to_crs("EPSG:4326")

    # Get unique gages (process all gages)
    unique_gages = final_data['gage_id'].dropna().unique()
    print(f"Creating maps for {len(unique_gages)} gages: {unique_gages}")

    # Create maps
    for gage_id in tqdm(unique_gages, desc="Creating styled maps"):
        try:
            gage_wells = final_data[final_data['gage_id'] == gage_id].copy()
            print(f"\nProcessing Gage {gage_id} with {len(gage_wells)} wells")

            if len(gage_wells) == 0:
                continue

            # Create the map
            create_single_styled_map(gage_id, gage_wells, subbasin_gdf, stream_gdf,
                                   lake_gdf, gage_df, terminal_relationships, save_dir)

        except Exception as e:
            print(f"❌ Error for gage {gage_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ Maps saved to: {save_dir}")


def create_watershed_vertical_distance_maps_styled():
    """
    Create watershed maps for each gage using the same style as correlation maps.
    Uses well_gage data for well-gage relationships.
    Colors wells by mutual information (mi_delta_wte_delta_q).
    """
    print("=== Creating Watershed Vertical Distance Maps (Styled Version) ===")
    save_dir = "../reports/figures/mi_lag"
    os.makedirs(save_dir, exist_ok=True)

    # Load all datasets
    print("Loading datasets...")

    # Geographic data
    subbasin_gdf = gpd.read_file('../data/raw/hydrography/gsl_catchment.shp')
    stream_gdf = gpd.read_file('../data/raw/hydrography/gslb_stream.shp')
    lake_gdf = gpd.read_file('../data/raw/hydrography/lake.shp')
    gage_df = pd.read_csv('../data/raw/hydrography/gsl_nwm_gage.csv')

    # Well data
    reach_distances = pd.read_csv('../data/processed/well_reach_relationships_final.csv')
    well_locations = pd.read_csv('../data/processed/well_reach.csv')[['well_id', 'well_lat', 'well_lon']].drop_duplicates()

    # CHANGED: Use lag_mi data for well-gage relationships
    lag_mi_data = pd.read_csv('../data/features/well_gage_mi_lag.csv')
    well_gage_pairs = lag_mi_data[['well_id', 'gage_id', 'mi_delta_wte_delta_q']].dropna().drop_duplicates()

    # Convert gage_id to integer (remove decimal points)
    well_gage_pairs['gage_id'] = well_gage_pairs['gage_id'].astype(int)

    print(f"Using lag_mi data for well-gage relationships: {len(well_gage_pairs)} unique pairs")

    # Load watershed relationships
    try:
        terminal_relationships = pd.read_csv('../data/processed/terminal_gage_upstream_catchments.csv')
        print("✅ Loaded terminal relationships")
    except FileNotFoundError:
        print("⚠️ Terminal relationships file not found, using simplified approach")
        terminal_relationships = None

    print("✅ All datasets loaded")

    # Merge data using numeric IDs
    print("Merging data using numeric approach...")

    # Convert to numeric
    reach_distances['well_id_numeric'] = pd.to_numeric(reach_distances['Well_ID'], errors='coerce')
    well_locations['well_id_numeric'] = pd.to_numeric(well_locations['well_id'], errors='coerce')
    well_gage_pairs['well_id_numeric'] = pd.to_numeric(well_gage_pairs['well_id'], errors='coerce')

    # Merge step by step
    merge1 = pd.merge(reach_distances, well_locations, on='well_id_numeric', how='inner')
    print(f"After adding coordinates: {len(merge1)} wells")

    final_data = pd.merge(merge1, well_gage_pairs, on='well_id_numeric', how='inner')
    print(f"Final merged data: {len(final_data)} wells with gage relationships")

    if len(final_data) == 0:
        print("❌ No data after merging - cannot create maps")
        return

    # Ensure CRS consistency
    subbasin_gdf = subbasin_gdf.to_crs("EPSG:4326")
    stream_gdf = stream_gdf.to_crs("EPSG:4326")
    lake_gdf = lake_gdf.to_crs("EPSG:4326")

    # Get unique gages (process all gages)
    unique_gages = final_data['gage_id'].dropna().unique()
    print(f"Creating maps for {len(unique_gages)} gages: {unique_gages}")

    # Create maps
    for gage_id in tqdm(unique_gages, desc="Creating styled maps"):
        try:
            gage_wells = final_data[final_data['gage_id'] == gage_id].copy()
            print(f"\nProcessing Gage {gage_id} with {len(gage_wells)} wells")

            if len(gage_wells) == 0:
                continue

            # Create the map
            create_single_styled_map_vertical(gage_id, gage_wells, subbasin_gdf, stream_gdf,
                                   lake_gdf, gage_df, terminal_relationships, save_dir)

        except Exception as e:
            print(f"❌ Error for gage {gage_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ Maps saved to: {save_dir}")


def create_watershed_mi_maps_no_lag():
    """
    Create watershed maps for each gage (no-lag version, same styling as correlation maps).
    Colors wells by mutual information (mi_delta_wte_delta_q).
    Saves to ../reports/figures/mi_no_lag
    """
    print("=== Creating Watershed MI Maps (NO LAG, Styled Version) ===")
    save_dir = "../reports/figures/mi_no_lag"
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Load geographic datasets
    # -----------------------------
    print("Loading datasets...")
    subbasin_gdf = gpd.read_file('../data/raw/hydrography/gsl_catchment.shp')
    stream_gdf   = gpd.read_file('../data/raw/hydrography/gslb_stream.shp')
    lake_gdf     = gpd.read_file('../data/raw/hydrography/lake.shp')
    gage_df      = pd.read_csv('../data/raw/hydrography/gsl_nwm_gage.csv')

    # Support tables for joining well geometry
    reach_distances = pd.read_csv('../data/processed/well_reach_relationships_final.csv')
    well_locations  = pd.read_csv('../data/processed/well_reach.csv')[['well_id', 'well_lat', 'well_lon']].drop_duplicates()

    # -----------------------------
    # NO-LAG well–gage MI data (edit the path if your filename differs)
    # -----------------------------
    no_lag_mi_path = '../data/features/well_gage_mi_no_lag.csv'
    no_lag_mi = pd.read_csv(no_lag_mi_path)
    # Expecting columns:
    # well_id, gage_id, mi_delta_wte_delta_q, pearson_r, spearman_r, n_records
    no_lag_mi = no_lag_mi[['well_id', 'gage_id', 'mi_delta_wte_delta_q']].dropna().drop_duplicates()

    # gage_id integer
    no_lag_mi['gage_id'] = no_lag_mi['gage_id'].astype(int)
    print(f"Using NO-LAG MI data for well-gage relationships: {len(no_lag_mi)} unique pairs")

    # -----------------------------
    # Upstream relationships (optional)
    # -----------------------------
    try:
        terminal_relationships = pd.read_csv('../data/processed/terminal_gage_upstream_catchments.csv')
        print("✅ Loaded terminal relationships")
    except FileNotFoundError:
        print("⚠️ Terminal relationships file not found, using simplified extent")
        terminal_relationships = None

    print("✅ All datasets loaded")

    # -----------------------------
    # Merge data using numeric IDs
    # -----------------------------
    print("Merging data...")
    reach_distances['well_id_numeric'] = pd.to_numeric(reach_distances['Well_ID'], errors='coerce')
    well_locations['well_id_numeric']  = pd.to_numeric(well_locations['well_id'], errors='coerce')
    no_lag_mi['well_id_numeric']       = pd.to_numeric(no_lag_mi['well_id'], errors='coerce')

    merge1 = pd.merge(reach_distances, well_locations, on='well_id_numeric', how='inner')
    print(f"After adding coordinates: {len(merge1)} wells")

    final_data = pd.merge(merge1, no_lag_mi, on='well_id_numeric', how='inner')
    print(f"Final merged data: {len(final_data)} wells with gage relationships")

    if len(final_data) == 0:
        print("❌ No data after merging - cannot create maps")
        return

    # Ensure CRS
    subbasin_gdf = subbasin_gdf.to_crs("EPSG:4326")
    stream_gdf   = stream_gdf.to_crs("EPSG:4326")
    lake_gdf     = lake_gdf.to_crs("EPSG:4326")

    # Unique gages to map
    unique_gages = final_data['gage_id'].dropna().unique()
    print(f"Creating maps for {len(unique_gages)} gages: {unique_gages}")

    for gage_id in tqdm(unique_gages, desc="Creating NO-LAG MI maps"):
        try:
            gage_wells = final_data[final_data['gage_id'] == gage_id].copy()
            if len(gage_wells) == 0:
                continue
            create_single_styled_map_mi_no_lag(
                gage_id, gage_wells, subbasin_gdf, stream_gdf,
                lake_gdf, gage_df, terminal_relationships, save_dir
            )
        except Exception as e:
            print(f"❌ Error for gage {gage_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ Maps saved to: {save_dir}")


def create_watershed_mi_delta_maps():
    """
    Create watershed maps for each gage (ΔMI version).
    Colors wells by delta_mi (mi_lag - mi_no_lag).
    Saves to ../reports/figures/mi_delta
    """
    print("=== Creating Watershed ΔMI Maps (Styled Version) ===")
    save_dir = "../reports/figures/mi_delta_top10"
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Load geographic datasets
    # -----------------------------
    print("Loading datasets...")
    subbasin_gdf = gpd.read_file('../data/raw/hydrography/gsl_catchment.shp')
    stream_gdf   = gpd.read_file('../data/raw/hydrography/gslb_stream.shp')
    lake_gdf     = gpd.read_file('../data/raw/hydrography/lake.shp')
    gage_df      = pd.read_csv('../data/raw/hydrography/gsl_nwm_gage.csv')

    # Support tables for joining well geometry
    reach_distances = pd.read_csv('../data/processed/well_reach_relationships_final.csv')
    well_locations  = pd.read_csv('../data/processed/well_reach.csv')[['well_id', 'well_lat', 'well_lon']].drop_duplicates()

    # -----------------------------
    # ΔMI well–gage data
    # -----------------------------
    delta_mi_df = pd.read_csv('../data/processed/mi_compare_lag_vs_no_lag_by_pair.csv')

    # Expecting columns like:
    # well_id, gage_id, mi_no_lag, mi_lag, delta_mi, ratio_mi, ...
    delta_mi_df = delta_mi_df[['well_id', 'gage_id', 'delta_mi']].dropna().drop_duplicates()
    delta_mi_df['gage_id'] = delta_mi_df['gage_id'].astype(int)

    print(f"Using ΔMI data for well-gage relationships: {len(delta_mi_df)} unique pairs")

    # -----------------------------
    # Upstream relationships (optional)
    # -----------------------------
    try:
        terminal_relationships = pd.read_csv('../data/processed/terminal_gage_upstream_catchments.csv')
        print("✅ Loaded terminal relationships")
    except FileNotFoundError:
        print("⚠️ Terminal relationships file not found, using simplified extent")
        terminal_relationships = None

    print("✅ All datasets loaded")

    # -----------------------------
    # Merge data using numeric IDs
    # -----------------------------
    print("Merging data...")
    reach_distances['well_id_numeric'] = pd.to_numeric(reach_distances['Well_ID'], errors='coerce')
    well_locations['well_id_numeric']  = pd.to_numeric(well_locations['well_id'], errors='coerce')
    delta_mi_df['well_id_numeric']     = pd.to_numeric(delta_mi_df['well_id'], errors='coerce')

    merge1 = pd.merge(reach_distances, well_locations, on='well_id_numeric', how='inner')
    print(f"After adding coordinates: {len(merge1)} wells")

    final_data = pd.merge(merge1, delta_mi_df, on='well_id_numeric', how='inner')
    print(f"Final merged data: {len(final_data)} wells with gage relationships")

    if len(final_data) == 0:
        print("❌ No data after merging - cannot create maps")
        return

    # Ensure CRS
    subbasin_gdf = subbasin_gdf.to_crs("EPSG:4326")
    stream_gdf   = stream_gdf.to_crs("EPSG:4326")
    lake_gdf     = lake_gdf.to_crs("EPSG:4326")

    # Unique gages to map
    unique_gages = final_data['gage_id'].dropna().unique()
    print(f"Creating maps for {len(unique_gages)} gages: {unique_gages}")

    for gage_id in tqdm(unique_gages, desc="Creating ΔMI maps"):
        try:
            gage_wells = final_data[final_data['gage_id'] == gage_id].copy()
            if len(gage_wells) == 0:
                continue
            create_single_styled_map_mi_delta(
                gage_id, gage_wells, subbasin_gdf, stream_gdf,
                lake_gdf, gage_df, terminal_relationships, save_dir
            )
        except Exception as e:
            print(f"❌ Error for gage {gage_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ Maps saved to: {save_dir}")


def create_watershed_mi_delta_maps_fixed():
    """
    Create watershed maps for each gage (ΔMI version) with fixed top-10 annotations.
    Colors wells by delta_mi (mi_lag - mi_no_lag).
    Saves to ../reports/figures/mi_delta_top10_fixed
    """
    print("=== Creating Watershed ΔMI Maps (FIXED VERSION with Red Highlights) ===")
    save_dir = "../reports/figures/mi_delta_top10_fixed"
    os.makedirs(save_dir, exist_ok=True)

    # Load geographic datasets
    print("Loading datasets...")
    subbasin_gdf = gpd.read_file('../data/raw/hydrography/gsl_catchment.shp')
    stream_gdf   = gpd.read_file('../data/raw/hydrography/gslb_stream.shp')
    lake_gdf     = gpd.read_file('../data/raw/hydrography/lake.shp')
    gage_df      = pd.read_csv('../data/raw/hydrography/gsl_nwm_gage.csv')

    # Support tables for joining well geometry
    reach_distances = pd.read_csv('../data/processed/well_reach_relationships_final.csv')
    well_locations  = pd.read_csv('../data/processed/well_reach.csv')[['well_id', 'well_lat', 'well_lon']].drop_duplicates()

    # ΔMI well–gage data
    delta_mi_df = pd.read_csv('../data/processed/mi_compare_lag_vs_no_lag_by_pair.csv')
    delta_mi_df = delta_mi_df[['well_id', 'gage_id', 'delta_mi']].dropna().drop_duplicates()
    delta_mi_df['gage_id'] = delta_mi_df['gage_id'].astype(int)

    print(f"Using ΔMI data for well-gage relationships: {len(delta_mi_df)} unique pairs")

    # Upstream relationships (optional)
    try:
        terminal_relationships = pd.read_csv('../data/processed/terminal_gage_upstream_catchments.csv')
        print("✅ Loaded terminal relationships")
    except FileNotFoundError:
        print("⚠️ Terminal relationships file not found, using simplified extent")
        terminal_relationships = None

    print("✅ All datasets loaded")

    # Merge data using numeric IDs
    print("Merging data...")
    reach_distances['well_id_numeric'] = pd.to_numeric(reach_distances['Well_ID'], errors='coerce')
    well_locations['well_id_numeric']  = pd.to_numeric(well_locations['well_id'], errors='coerce')
    delta_mi_df['well_id_numeric']     = pd.to_numeric(delta_mi_df['well_id'], errors='coerce')

    merge1 = pd.merge(reach_distances, well_locations, on='well_id_numeric', how='inner')
    print(f"After adding coordinates: {len(merge1)} wells")

    final_data = pd.merge(merge1, delta_mi_df, on='well_id_numeric', how='inner')
    print(f"Final merged data: {len(final_data)} wells with gage relationships")

    if len(final_data) == 0:
        print("❌ No data after merging - cannot create maps")
        return

    # Ensure CRS
    subbasin_gdf = subbasin_gdf.to_crs("EPSG:4326")
    stream_gdf   = stream_gdf.to_crs("EPSG:4326")
    lake_gdf     = lake_gdf.to_crs("EPSG:4326")

    # Unique gages to map
    unique_gages = final_data['gage_id'].dropna().unique()
    print(f"Creating maps for {len(unique_gages)} gages: {unique_gages}")

    for gage_id in tqdm(unique_gages, desc="Creating FIXED ΔMI maps"):
        try:
            gage_wells = final_data[final_data['gage_id'] == gage_id].copy()
            if len(gage_wells) == 0:
                continue
            create_single_styled_map_mi_delta_fixed(
                gage_id, gage_wells, subbasin_gdf, stream_gdf,
                lake_gdf, gage_df, terminal_relationships, save_dir
            )
        except Exception as e:
            print(f"❌ Error for gage {gage_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ Maps saved to: {save_dir}")


def run_watershed_correlation_analysis():
    """
    Run the complete watershed correlation analysis using TOP 10% correlation data.

    Steps:
    1) Load top 10% correlation subset (pre-computed)
    2) Load the original no-lag data for geographic context
    3) Render per-gage maps for ALL gages in the subset
    4) Save summary statistics
    """
    print("🚀 Starting Watershed Correlation Analysis (TOP 10% CORRELATION DATA)")

    # Load the top 10% correlation subset created in 05_delta_metrics.ipynb
    try:
        top10_data = pd.read_csv('../data/features/no_lag_top10_correlation_by_gage.csv')
        print(f"✅ Loaded top 10% correlation data with {len(top10_data):,} records")
    except Exception as e:
        print(f"❌ Error loading top 10% correlation data: {e}")
        return

    # Calculate correlations using the top 10% subset
    correlation_df = calculate_well_gage_correlations(top10_data, min_points=10)
    if correlation_df is None:
        print("❌ No correlations calculated (None). Check your data.")
        return

    # Create watershed maps using the top 10% data
    create_clean_correlation_maps_with_watersheds(
        correlation_df, 
        top10_data,
        save_dir='../reports/figures/gage_well_correlations_watershed_top10_no_lag'
    )

    # Create summary stats
    create_summary_statistics(correlation_df)

    print("🎉 Watershed Correlation Analysis (TOP 10%) Complete!")


# ============================================================
# Seasonal reporting functions (from notebook 06)
# ============================================================


def get_season_from_month(month):
    """
    Map month number to season name.

    Parameters
    ----------
    month : int
        Month number (1-12)

    Returns
    -------
    str
        Season name (Winter, Spring, Summer, or Fall)
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    return 'Unknown'


def generate_simple_seasonal_report(stats_df):
    """
    Generate a simple seasonal report showing percentage of positive slopes and R squared values.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with columns: season, slope_delta_q, r_squared_delta_q,
        slope_lag_wte, r_squared_lag_wte

    Returns
    -------
    pd.DataFrame
        Seasonal summary with positive slope percentages and mean R squared values
    """
    print("SEASONAL STATISTICAL REPORT - Delta Q vs Lagged Delta WTE")
    print("=" * 60)

    def _get_season_from_name(month_name):
        if month_name in ['December', 'January', 'February']:
            return 'Winter'
        elif month_name in ['March', 'April', 'May']:
            return 'Spring'
        elif month_name in ['June', 'July', 'August']:
            return 'Summer'
        elif month_name in ['September', 'October', 'November']:
            return 'Fall'
        return 'Unknown'

    if 'season' not in stats_df.columns:
        if 'month_name' in stats_df.columns:
            stats_df['season'] = stats_df['month_name'].apply(_get_season_from_name)

    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    seasonal_results = []

    for season in season_order:
        if season in stats_df['season'].values:
            season_data = stats_df[stats_df['season'] == season]
            total_records = len(season_data)
            pos_dq = (season_data['slope_delta_q'] > 0).sum()
            pct_dq = (pos_dq / total_records) * 100
            r2_dq = season_data['r_squared_delta_q'].mean()
            pos_wte = (season_data['slope_lag_wte'] > 0).sum()
            pct_wte = (pos_wte / total_records) * 100
            r2_wte = season_data['r_squared_lag_wte'].mean()

            seasonal_results.append({
                'Season': season,
                'Total_Records': total_records,
                'Positive_Slopes_Delta_Q_Percentage': pct_dq,
                'Mean_R_Squared_Delta_Q': r2_dq,
                'Positive_Slopes_Lag_WTE_Percentage': pct_wte,
                'Mean_R_Squared_Lag_WTE': r2_wte
            })

            print(f"{season:12} | Records: {total_records:3d} | Pos Slopes DQ: {pct_dq:5.1f}% | Mean R2 DQ: {r2_dq:.3f}")
            print(f"{'':12} | {'':13} | Pos Slopes Lag WTE: {pct_wte:5.1f}% | Mean R2 WTE: {r2_wte:.3f}")
            print("-" * 60)

    total_all = len(stats_df)
    pct_all_dq = ((stats_df['slope_delta_q'] > 0).sum() / total_all) * 100
    r2_all_dq = stats_df['r_squared_delta_q'].mean()
    pct_all_wte = ((stats_df['slope_lag_wte'] > 0).sum() / total_all) * 100
    r2_all_wte = stats_df['r_squared_lag_wte'].mean()

    print(f"{'OVERALL':12} | Records: {total_all:3d} | Pos Slopes DQ: {pct_all_dq:5.1f}% | Mean R2 DQ: {r2_all_dq:.3f}")
    print(f"{'':12} | {'':13} | Pos Slopes Lag WTE: {pct_all_wte:5.1f}% | Mean R2 WTE: {r2_all_wte:.3f}")

    return pd.DataFrame(seasonal_results)



# ============================================================
# CCF visualization functions (from notebook 05)
# ============================================================



def plot_correlation_lag_curves(ccf_results_extended, overall_stats, watershed_summary):
    """
    Create comprehensive plots showing correlation vs lag relationships
    """
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall correlation vs lag curve (averaged across all wells)
    ax1 = plt.subplot(3, 3, 1)
    
    # Calculate average CCF across all wells
    max_lag_days = 10 * 365
    lag_range = np.arange(-max_lag_days, max_lag_days + 1, 30)  # Every 30 days
    
    # Interpolate all CCF curves to common lag grid
    all_ccf_interpolated = []
    
    for gage_id, wells in ccf_results_extended.items():
        for well_id, result in wells.items():
            # Interpolate to common grid
            ccf_interp = np.interp(lag_range, result['lags'], result['correlation'])
            all_ccf_interpolated.append(ccf_interp)
    
    if all_ccf_interpolated:
        mean_ccf = np.mean(all_ccf_interpolated, axis=0)
        std_ccf = np.std(all_ccf_interpolated, axis=0)
        
        ax1.plot(lag_range / 365.25, mean_ccf, 'b-', linewidth=2, label='Mean CCF')
        ax1.fill_between(lag_range / 365.25, mean_ccf - std_ccf, mean_ccf + std_ccf, 
                        alpha=0.3, color='blue', label='±1 STD')
        
        # Mark overall optimal lag
        optimal_lag_years = overall_stats['median_optimal_lag'] / 365.25
        ax1.axvline(optimal_lag_years, color='red', linestyle='--', linewidth=2,
                   label=f'Median optimal lag: {optimal_lag_years:.2f} years')
        ax1.axvline(-optimal_lag_years, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Lag (years)')
    ax1.set_ylabel('Cross-correlation')
    ax1.set_title('Overall CCF: All Watersheds Combined')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10, 10)
    
    # 2. Individual watershed CCF curves
    ax2 = plt.subplot(3, 3, 2)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(watershed_summary)))
    
    for i, (gage_id, summary) in enumerate(watershed_summary.items()):
        wells = ccf_results_extended[gage_id]
        
        # Calculate mean CCF for this watershed
        watershed_ccf = []
        for well_id, result in wells.items():
            ccf_interp = np.interp(lag_range, result['lags'], result['correlation'])
            watershed_ccf.append(ccf_interp)
        
        if watershed_ccf:
            mean_watershed_ccf = np.mean(watershed_ccf, axis=0)
            ax2.plot(lag_range / 365.25, mean_watershed_ccf, 
                    color=colors[i], linewidth=2, label=f'Gage {gage_id}')
            
            # Mark watershed optimal lag
            optimal_lag_years = summary['median_optimal_lag'] / 365.25
            ax2.axvline(optimal_lag_years, color=colors[i], linestyle=':', alpha=0.7)
    
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Lag (years)')
    ax2.set_ylabel('Cross-correlation')
    ax2.set_title('CCF by Watershed')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-10, 10)
    
    # 3. Distribution of optimal lags
    ax3 = plt.subplot(3, 3, 3)
    
    all_lags_years = np.array(overall_stats['optimal_lags']) / 365.25
    ax3.hist(all_lags_years, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.median(all_lags_years), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(all_lags_years):.2f} years')
    ax3.axvline(1, color='green', linestyle='--', label='1 year', alpha=0.7)
    ax3.axvline(2, color='orange', linestyle='--', label='2 years', alpha=0.7)
    ax3.axvline(5, color='purple', linestyle='--', label='5 years', alpha=0.7)
    
    ax3.set_xlabel('Optimal Lag (years)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Optimal Lags')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-9. Individual watershed details (top 6 watersheds by number of wells)
    sorted_watersheds = sorted(watershed_summary.items(), 
                              key=lambda x: x[1]['n_wells'], reverse=True)[:6]
    
    for idx, (gage_id, summary) in enumerate(sorted_watersheds):
        ax = plt.subplot(3, 3, 4 + idx)
        
        wells = ccf_results_extended[gage_id]
        
        # Plot individual well CCF curves (sample up to 10 wells)
        well_items = list(wells.items())
        if len(well_items) > 10:
            well_items = well_items[::len(well_items)//10]  # Sample evenly
        
        for well_id, result in well_items:
            ax.plot(result['lags'] / 365.25, result['correlation'], 
                   alpha=0.3, color='gray', linewidth=0.5)
        
        # Plot mean CCF for this watershed
        watershed_ccf = []
        for well_id, result in wells.items():
            ccf_interp = np.interp(lag_range, result['lags'], result['correlation'])
            watershed_ccf.append(ccf_interp)
        
        if watershed_ccf:
            mean_watershed_ccf = np.mean(watershed_ccf, axis=0)
            ax.plot(lag_range / 365.25, mean_watershed_ccf, 
                   color='red', linewidth=3, label='Mean CCF')
            
            # Mark optimal lag
            optimal_lag_years = summary['median_optimal_lag'] / 365.25
            ax.axvline(optimal_lag_years, color='blue', linestyle='--', linewidth=2,
                      label=f'Optimal: {optimal_lag_years:.2f} yr')
            ax.axvline(-optimal_lag_years, color='blue', linestyle='--', 
                      linewidth=2, alpha=0.5)
        
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Lag (years)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title(f'Gage {gage_id} ({summary["n_wells"]} wells)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 10)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_ccf_summary(
    ccf_results_extended,
    overall_stats,
    watershed_summary,
    max_lag_years=10,
    save_dir="./figures",
    dpi=600
):
    """
    Plot Figure 1 (Summary):
    - (A) Overall mean CCF with ±1 STD
    - (B) Watershed-level mean CCF overlay
    - (C) Histogram of optimal lags (mean & median lines)

    Saves both PNG and SVG.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Common lag grid (in days) for interpolation and plotting
    max_lag_days = int(max_lag_years * 365)
    lag_range_days = np.arange(-max_lag_days, max_lag_days + 1, 30)  # 30-day step
    lag_range_years = lag_range_days / 365.25

    # ---- Collect interpolated CCFs for overall mean/std ----
    all_ccf_interpolated = []
    for gage_id, wells in ccf_results_extended.items():
        for well_id, result in wells.items():
            # Interpolate each well CCF to the common lag grid
            ccf_interp = np.interp(lag_range_days, result['lags'], result['correlation'])
            all_ccf_interpolated.append(ccf_interp)

    # Prepare Figure 1 (1x3)
    fig = plt.figure(figsize=(20, 6))  # wide aspect for three panels

    # (A) Overall mean CCF
    ax1 = plt.subplot(1, 3, 1)
    if all_ccf_interpolated:
        mean_ccf = np.mean(all_ccf_interpolated, axis=0)
        std_ccf = np.std(all_ccf_interpolated, axis=0)

        ax1.plot(lag_range_years, mean_ccf, linewidth=2, label="Mean CCF")
        ax1.fill_between(lag_range_years, mean_ccf - std_ccf, mean_ccf + std_ccf,
                         alpha=0.3, label="±1 STD")

        # Mark overall median optimal lag
        med_opt_lag_years = overall_stats['median_optimal_lag'] / 365.25
        ax1.axvline(med_opt_lag_years, linestyle="--", linewidth=1.8, label=f"Median optimal lag: {med_opt_lag_years:.2f} yr")

    ax1.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax1.axvline(0, color='black', linewidth=1, alpha=0.3)
    ax1.set_xlim(-max_lag_years, max_lag_years)
    ax1.set_xlabel("Lag (years)")
    ax1.set_ylabel("Cross-correlation")
    ax1.set_title("(A) Overall Mean CCF (All Wells)")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    # (B) Watershed-level mean CCF overlay
    ax2 = plt.subplot(1, 3, 2)
    if len(watershed_summary) > 0:
        # Build a stable color cycle
        cmap = plt.cm.get_cmap("tab10", len(watershed_summary))
        for i, (gage_id, summary) in enumerate(watershed_summary.items()):
            wells = ccf_results_extended.get(gage_id, {})
            if len(wells) == 0:
                continue

            # Average CCF for this watershed
            ws_ccf_list = []
            for w_id, res in wells.items():
                ccf_interp = np.interp(lag_range_days, res['lags'], res['correlation'])
                ws_ccf_list.append(ccf_interp)
            if len(ws_ccf_list) == 0:
                continue

            ws_mean = np.mean(ws_ccf_list, axis=0)
            ax2.plot(lag_range_years, ws_mean, linewidth=2, label=f"{gage_id}", color=cmap(i))

            # Mark watershed median optimal lag
            med_ws = summary['median_optimal_lag'] / 365.25
            ax2.axvline(med_ws, linestyle=":", linewidth=1.5, color=cmap(i), alpha=0.8)

    ax2.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax2.axvline(0, color='black', linewidth=1, alpha=0.3)
    ax2.set_xlim(-max_lag_years, max_lag_years)
    ax2.set_xlabel("Lag (years)")
    ax2.set_ylabel("Cross-correlation")
    ax2.set_title("(B) Mean CCF by Watershed")
    ax2.legend(title="Gage ID", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax2.grid(alpha=0.3)

    # (C) Histogram of optimal lags (years) with mean/median lines
    ax3 = plt.subplot(1, 3, 3)
    all_lags_years = np.array(overall_stats['optimal_lags']) / 365.25
    if len(all_lags_years) > 0:
        ax3.hist(all_lags_years, bins=50, alpha=0.8, edgecolor='black')
        med = np.median(all_lags_years)
        mean = np.mean(all_lags_years)
        ax3.axvline(med, color='red', linestyle='--', linewidth=2, label=f"Median: {med:.2f} yr")
        ax3.axvline(mean, color='green', linestyle='--', linewidth=2, label=f"Mean: {mean:.2f} yr")

    ax3.set_xlabel("Optimal Lag (years)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("(C) Distribution of Optimal Lags")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    f_png = os.path.join(save_dir, f"CCF_Summary_maxLag{max_lag_years}yr.png")
    f_svg = os.path.join(save_dir, f"CCF_Summary_maxLag{max_lag_years}yr.svg")
    fig.savefig(f_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(f_svg, dpi=dpi, bbox_inches="tight")
    print(f"[Saved] {f_png}\n[Saved] {f_svg}")
    return fig

def plot_ccf_watershed_details(
    ccf_results_extended,
    watershed_summary,
    max_lag_years=10,
    save_dir="./figures",
    dpi=600,
    max_wells_per_panel=10
):
    """
    Plot Figure 2 (Details): 2x3 panels, one per watershed (up to 6).
    Each panel shows:
    - thin gray lines for sampled individual well CCFs,
    - thick red line for the watershed mean CCF,
    - vertical blue dashed line for the watershed median optimal lag.

    Saves both PNG and SVG.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Choose the 6 watersheds with most wells
    top6 = sorted(
        watershed_summary.items(),
        key=lambda x: x[1]['n_wells'],
        reverse=True
    )[:6]

    # Prepare common lag grid
    max_lag_days = int(max_lag_years * 365)
    lag_range_days = np.arange(-max_lag_days, max_lag_days + 1, 30)
    lag_range_years = lag_range_days / 365.25

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, (gage_id, summary) in enumerate(top6):
        ax = axes[idx]
        wells = ccf_results_extended.get(gage_id, {})
        if len(wells) == 0:
            ax.set_title(f"{gage_id} (no wells)")
            continue

        # Sample up to N wells (evenly) for gray curves
        items = list(wells.items())
        if len(items) > max_wells_per_panel:
            step = max(1, len(items) // max_wells_per_panel)
            items = items[::step]

        ws_ccf_list = []
        # Plot individual wells (light gray)
        for w_id, res in items:
            y = np.interp(lag_range_days, res['lags'], res['correlation'])
            ws_ccf_list.append(y)
            ax.plot(lag_range_years, y, color='gray', alpha=0.35, linewidth=0.6)

        # Mean CCF for the watershed (thick red)
        if len(ws_ccf_list) > 0:
            ws_mean = np.mean(ws_ccf_list, axis=0)
            ax.plot(lag_range_years, ws_mean, color='red', linewidth=2.5, label="Mean CCF")

        # Median optimal lag (blue dashed)
        med_ws = summary['median_optimal_lag'] / 365.25
        ax.axvline(med_ws, color='blue', linestyle='--', linewidth=1.8, label=f"Median lag: {med_ws:.2f} yr")

        ax.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax.axvline(0, color='black', linewidth=1, alpha=0.3)
        ax.set_xlim(-max_lag_years, max_lag_years)
        ax.set_title(f"Gage {gage_id}  (n={summary['n_wells']})")
        ax.grid(alpha=0.3)
        if idx % 3 == 0:
            ax.set_ylabel("Cross-correlation")
        if idx >= 3:
            ax.set_xlabel("Lag (years)")
        ax.legend(loc="best", fontsize=9)

    # If fewer than 6 watersheds, hide extra panels
    for j in range(len(top6), 6):
        axes[j].axis("off")

    plt.tight_layout()

    # Save
    f_png = os.path.join(save_dir, f"CCF_WatershedDetails_maxLag{max_lag_years}yr.png")
    f_svg = os.path.join(save_dir, f"CCF_WatershedDetails_maxLag{max_lag_years}yr.svg")
    fig.savefig(f_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(f_svg, dpi=dpi, bbox_inches="tight")
    print(f"[Saved] {f_png}\n[Saved] {f_svg}")
    return fig


# ============================================================
# Terminal gage visualization functions (from notebook 02)
# ============================================================



def create_enhanced_watershed_visualization():
    """
    Enhanced watershed visualization:
    - Terminal gages only (stars), no other gages
    - Stream network filtered to order >= 4 (major rivers)
    - Wells plotted explicitly
    """
    print("=== Creating Enhanced Watershed Visualization ===")

    # Load data
    terminal_mapping = pd.read_csv('../data/processed/terminal_gage_upstream_catchments.csv')
    subbasin_gdf = gpd.read_file('../data/raw/hydrography/gsl_catchment.shp')
    gage_df = pd.read_csv('../data/raw/hydrography/gsl_nwm_gage.csv')
    well_gdf = gpd.read_file('../data/raw/hydrography/well_shp.shp')
    stream_gdf = gpd.read_file('../data/raw/hydrography/gslb_stream.shp')
    lake_gdf = gpd.read_file('../data/raw/hydrography/lake.shp')

    # Filter to major rivers only (Strahler order >= 4)
    major_streams = stream_gdf[stream_gdf['strmOrder'] >= 4].copy()
    print(f"  Streams total: {len(stream_gdf)}, order >= 4: {len(major_streams)}")

    # Find linkno column
    linkno_col = 'linkno' if 'linkno' in subbasin_gdf.columns else 'LINKNO'

    # Data preprocessing
    if 'Gage_ID' in terminal_mapping.columns:
        terminal_mapping = terminal_mapping.rename(columns={
            'Gage_ID': 'gage_id',
            'Upstream_Catchment_ID': 'upstream_catchment_id'
        })

    terminal_mapping = terminal_mapping.dropna(subset=['upstream_catchment_id'])
    terminal_mapping['gage_id'] = terminal_mapping['gage_id'].astype(int)
    terminal_mapping['upstream_catchment_id'] = terminal_mapping['upstream_catchment_id'].astype(int)

    gage_df['id'] = gage_df['id'].astype(int)
    subbasin_gdf = subbasin_gdf.dropna(subset=[linkno_col])
    subbasin_gdf[linkno_col] = subbasin_gdf[linkno_col].astype(int)

    # Get terminal gages and create catchments dictionary
    terminal_gage_ids = terminal_mapping['gage_id'].unique().tolist()
    terminal_gages = gage_df[gage_df['id'].isin(terminal_gage_ids)].copy()

    terminal_gage_catchments = {}
    available_catchments = set(subbasin_gdf[linkno_col].unique())

    for gage_id in terminal_gage_ids:
        upstream_catchments = terminal_mapping[
            terminal_mapping['gage_id'] == gage_id
        ]['upstream_catchment_id'].tolist()

        valid_catchments = [c for c in upstream_catchments if c in available_catchments]
        if valid_catchments:
            terminal_gage_catchments[gage_id] = set(valid_catchments)

    # Clip wells to subbasin boundary (spatial join — keep only wells inside any subbasin polygon)
    subbasin_union = subbasin_gdf.dissolve()                        # single polygon covering entire basin
    well_gdf_proj = well_gdf.to_crs(subbasin_gdf.crs)              # match CRS before join
    well_in_basin = gpd.sjoin(
        well_gdf_proj, subbasin_union[['geometry']],
        how='inner', predicate='within'
    ).drop(columns=['index_right'])
    print(f"  Wells total: {len(well_gdf)}, within subbasin: {len(well_in_basin)}")

    # Convert to Web Mercator
    subbasin_web = subbasin_gdf.to_crs('EPSG:3857')
    major_streams_web = major_streams.to_crs('EPSG:3857')
    lake_web = lake_gdf.to_crs('EPSG:3857')
    well_web = well_in_basin.to_crs('EPSG:3857')

    terminal_gages_web = gpd.GeoDataFrame(
        terminal_gages,
        geometry=gpd.points_from_xy(terminal_gages['longitude'], terminal_gages['latitude']),
        crs='EPSG:4326'
    ).to_crs('EPSG:3857')

    # Bright, playful palette: high luminosity, vibrant colors, wide hue distribution
    bright_vivid_colors = [
        '#EF4444',  # bright coral red
        '#10B981',  # fresh green
        '#3B82F6',  # bright blue
        '#FBBF24',  # sunny yellow
        '#8B5CF6',  # vivid purple
        '#06B6D4',  # electric cyan
        '#F59E0B',  # amber orange
        '#EC4899',  # hot pink
        '#14B8A6',  # teal
        '#A855F7',  # bright violet
        '#6366F1',  # indigo
        '#84CC16',  # lime green
    ]
    terminal_gage_colors = dict(zip(terminal_gage_ids, bright_vivid_colors[:len(terminal_gage_ids)]))

    # ------------------------------------------------------------------
    # White-background, HIGH-IMPACT figure for white PPT slides
    # ------------------------------------------------------------------
    BG_COLOR = 'white'
    BASIN_FILL = '#FAFBFC'          # nearly white (was #F8FAFC, now even lighter)
    BASIN_EDGE = '#E8ECF0'          # very light gray border (was #CBD5E1, now much lighter)
    STREAM_COLOR = '#0369A1'        # deep blue rivers
    LAKE_COLOR = '#38BDF8'          # brighter sky blue (was #7DD3FC, now more vivid)
    WELL_COLOR = '#BE185D'          # deep magenta-pink
    OUTLINE_COLOR = '#475569'

    fig, ax = plt.subplots(1, 1, figsize=(22, 16))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # 1. Subbasin background (very light, subtle grid)
    subbasin_web.plot(
        ax=ax,
        color=BASIN_FILL,
        edgecolor=BASIN_EDGE,
        linewidth=0.2,
        alpha=1.0,
        zorder=1
    )

    # 2. Colored watersheds — saturated colors with solid alpha for clarity
    for gage_id in terminal_gage_ids:
        if gage_id not in terminal_gage_catchments:
            continue
        upstream_catchments = list(terminal_gage_catchments[gage_id])
        upstream_basins = subbasin_web[subbasin_web[linkno_col].isin(upstream_catchments)]
        if not upstream_basins.empty:
            upstream_basins.plot(
                ax=ax,
                color=terminal_gage_colors[gage_id],
                alpha=0.75,         # higher saturation for eye-catching PPT display
                edgecolor='none',
                zorder=2
            )

    # Basin outer boundary — dark outline for definition
    subbasin_union_web = subbasin_web.dissolve()
    subbasin_union_web.boundary.plot(
        ax=ax,
        color=OUTLINE_COLOR,
        linewidth=2.2,
        alpha=0.85,
        zorder=2.5
    )

    # 3. Lakes — bright sky blue
    lake_web.plot(
        ax=ax,
        color=LAKE_COLOR,
        edgecolor='#38BDF8',
        linewidth=0.5,
        alpha=0.6,
        zorder=3
    )

    # 4. Streams — deep blue, width hierarchy by order
    major_streams_web_copy = major_streams_web.copy()
    major_streams_web_copy['strmOrder'] = pd.to_numeric(
        major_streams_web_copy['strmOrder'], errors='coerce'
    )
    major_streams_web_copy['__lw__'] = 1.5 + 0.6 * (
        major_streams_web_copy['strmOrder'] - 4
    ).clip(lower=0)

    for lw_val, grp in major_streams_web_copy.groupby('__lw__'):
        grp.plot(
            ax=ax,
            color=STREAM_COLOR,
            linewidth=float(lw_val),
            alpha=0.9,
            zorder=4
        )

    # 5. Wells — deep magenta solid dots, LARGE for PPT projection visibility
    #    Increased size and opacity ensure visibility from back of room
    well_web.plot(
        ax=ax,
        marker='o',
        markersize=8,           # larger for projection
        color=WELL_COLOR,
        edgecolor='none',
        alpha=0.80,             # more opaque for clarity
        zorder=5,
        rasterized=True
    )

    # 6. Terminal gages — Stars with DARK shadow glow for maximum contrast
    #    Black shadow creates "embossed" effect on bright backgrounds
    from matplotlib.patheffects import withStroke, Normal
    
    terminal_gage_info = []
    for _, row in terminal_gages_web.iterrows():
        gage_id = int(row['id'])
        if gage_id not in terminal_gage_catchments:
            continue

        x, y = row.geometry.x, row.geometry.y
        c = terminal_gage_colors[gage_id]

        # Create star with dramatic dark shadow effect
        star = ax.scatter(
            [x], [y],
            c=c,
            marker='*',
            s=1900,
            edgecolors='none',
            linewidths=0,
            alpha=1.0,
            zorder=10
        )
        
        # Apply dark shadow effect (creates "raised" appearance)
        # Layer 1: Large dark shadow (creates depth)
        # Layer 2: Medium dark shadow (transition)
        # Layer 3: White outline (creates crisp separation)
        # Layer 4: Colored star on top
        star.set_path_effects([
            withStroke(linewidth=16, foreground='#000000', alpha=0.35),  # outer dark shadow
            withStroke(linewidth=10, foreground='#000000', alpha=0.55),  # inner dark shadow
            withStroke(linewidth=4.0, foreground='white', alpha=1.0),    # crisp white edge
            Normal()  # colored star
        ])

        gage_name = row.get('name', f'Gage {gage_id}')
        terminal_gage_info.append({
            'id': gage_id,
            'name': gage_name,
            'color': c
        })

    # ------------------------------------------------------------------
    # Legend — lower RIGHT corner, light-styled box
    # ------------------------------------------------------------------
    from matplotlib.patches import Patch

    legend_elements = [
        plt.Line2D([0], [0], color=STREAM_COLOR, linewidth=3.0,
                   label='Major Streams'),
        Patch(facecolor=LAKE_COLOR, edgecolor='#38BDF8', alpha=0.6,
              label='Lakes'),
        Patch(facecolor='#94A3B8', edgecolor='none', alpha=0.65,
              label='Upstream Watersheds'),
        plt.Line2D([0], [0], marker='o', color='none',
                   markerfacecolor=WELL_COLOR, markeredgecolor='none',
                   markersize=9, label=f'Groundwater Wells (n={len(well_in_basin)})'),
        plt.Line2D([0], [0], marker='*', color='none',
                   markerfacecolor='#94A3B8', markeredgecolor='#1E293B',
                   markeredgewidth=1.5, markersize=15,
                   label=f'Terminal Gages (n={len(terminal_gage_info)})'),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc='lower right',          # moved to lower right
        fontsize=11,
        title='Map Elements',
        title_fontsize=12,
        frameon=True,
        framealpha=0.95,
        facecolor='white',
        edgecolor='#475569',
        labelcolor='#1E293B',
    )
    legend.get_title().set_color('#1E293B')
    legend.get_title().set_fontweight('bold')

    # No title for cleaner PPT integration
    ax.set_aspect('equal')
    ax.axis('off')
    ax.margins(0.01)
    plt.tight_layout()

    out_path = '../reports/figures/enhanced_terminal_gages_watersheds.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.show()

    print(f"Saved: {out_path}")
    print(f"Wells plotted: {len(well_in_basin)} of {len(well_gdf)} (within subbasin)")
    print(f"Streams plotted: {len(major_streams)} of {len(stream_gdf)} (order >= 4)")

def create_single_map(terminal_gage_id, terminal_relationships, subbasin_gdf,
                      stream_gdf, lake_gdf, well_gdf, gage_gdf, gage_df,
                      linkno_col, output_dir):
    """Create map for a single gage"""

    # Harmonious color scheme
    colors = {
        'terminal_gage': '#FFFF00',  # Bright yellow star
        'terminal_catchment': '#CD5C5C',  # Indian red
        'upstream_catchments': '#D2B48C',  # Tan
        'other_gages': '#FF8C42',  # Orange
        'wells': '#654321',  # Dark brown
        'streams': '#4682B4',  # Steel blue
        'lakes': '#87CEEB',  # Sky blue
        'inset_all': '#F5DEB3',  # Wheat
        'inset_highlight': '#CD853F',  # Peru
    }

    # Get gage info
    terminal_gage_info = gage_df[gage_df['id'] == terminal_gage_id]
    if terminal_gage_info.empty:
        return

    gage_name = terminal_gage_info.iloc[0].get('name', f'Gage {terminal_gage_id}')

    # Get upstream catchments
    upstream_catchments = terminal_relationships[
        terminal_relationships['gage_id'] == terminal_gage_id
        ]['upstream_catchment_id'].tolist()

    if not upstream_catchments:
        return

    upstream_basins = subbasin_gdf[subbasin_gdf[linkno_col].isin(upstream_catchments)]
    if upstream_basins.empty:
        return

    terminal_gage_point = gage_gdf[gage_gdf['id'] == terminal_gage_id]
    if terminal_gage_point.empty:
        return

    # Find terminal catchment
    terminal_catchment_id = None
    terminal_point = terminal_gage_point.geometry.iloc[0]

    for _, basin in upstream_basins.iterrows():
        if basin.geometry.contains(terminal_point):
            terminal_catchment_id = basin[linkno_col]
            break

    # Separate basins
    if terminal_catchment_id is not None:
        terminal_basin = upstream_basins[upstream_basins[linkno_col] == terminal_catchment_id]
        other_upstream_basins = upstream_basins[upstream_basins[linkno_col] != terminal_catchment_id]
    else:
        terminal_basin = gpd.GeoDataFrame()
        other_upstream_basins = upstream_basins

    # Filter features within watershed only
    upstream_union = upstream_basins.unary_union

    # Use within instead of intersects to ensure features are fully contained
    local_streams = stream_gdf[
        stream_gdf.geometry.apply(lambda x: upstream_union.contains(x) or upstream_union.intersects(x))]
    local_lakes = lake_gdf[lake_gdf.geometry.apply(lambda x: upstream_union.contains(x))]
    local_wells = well_gdf[well_gdf.geometry.apply(lambda x: upstream_union.contains(x))]
    local_gages = gage_gdf[gage_gdf.geometry.apply(lambda x: upstream_union.contains(x))]
    local_gages = local_gages[local_gages['id'] != terminal_gage_id]

    # Create figure
    fig, ax_main = plt.subplots(1, 1, figsize=(14, 11))
    ax_main.set_facecolor('#FAFAFA')

    # Draw main map
    plot_main_map(ax_main, colors, other_upstream_basins, terminal_basin,
                  local_lakes, local_streams, local_wells, local_gages,
                  terminal_gage_point, upstream_basins)

    # Create inset map in top right corner
    ax_inset = fig.add_axes([0.72, 0.72, 0.26, 0.26])
    plot_inset_map(ax_inset, colors, subbasin_gdf, stream_gdf, lake_gdf,
                   upstream_basins, terminal_gage_point)

    # Add legend and title
    add_legend_and_title(fig, ax_main, colors, terminal_gage_id, gage_name,
                         len(local_gages), len(local_wells), len(local_lakes))

    # Save figure
    safe_name = gage_name.replace('/', '_').replace('\\', '_')[:50]
    filename = f"gage_{terminal_gage_id}_{safe_name}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_visualization(result_df):
    """Create visualization of upstream gage relationships"""

    print("🎨 Creating upstream gage relationship visualization...")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Number of upstream gages per terminal gage
        terminal_counts = result_df.groupby('terminal_gage_id').size().sort_values(ascending=False)
        ax1.bar(range(len(terminal_counts)), terminal_counts.values,
                color='steelblue', alpha=0.7)
        ax1.set_xlabel('Terminal Gage Index')
        ax1.set_ylabel('Number of Upstream Gages')
        ax1.set_title('Upstream Gages per Terminal Gage')
        ax1.grid(True, alpha=0.3)

        # 2. Distance distribution histogram
        if result_df['distance_km'].notna().sum() > 0:
            distances = result_df['distance_km'].dropna()
            ax2.hist(distances, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Distance (km)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Distances')
            ax2.grid(True, alpha=0.3)

        # 3. Average distance for each terminal gage
        avg_distances = result_df.groupby('terminal_gage_id')['distance_km'].mean().sort_values()
        ax3.bar(range(len(avg_distances)), avg_distances.values,
                color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Terminal Gage Index')
        ax3.set_ylabel('Average Distance (km)')
        ax3.set_title('Average Distance to Upstream Gages')
        ax3.grid(True, alpha=0.3)

        # 4. Distance vs count scatter plot
        terminal_stats = result_df.groupby('terminal_gage_id').agg({
            'non_terminal_gage_id': 'count',
            'distance_km': 'mean'
        }).reset_index()

        ax4.scatter(terminal_stats['distance_km'], terminal_stats['non_terminal_gage_id'],
                    alpha=0.7, s=100, color='purple')
        ax4.set_xlabel('Average Distance (km)')
        ax4.set_ylabel('Number of Upstream Gages')
        ax4.set_title('Distance vs Count Relationship')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("../reports/figures/upstream_gages_analysis.png", dpi=600, bbox_inches='tight')
        plt.show()

        print("✅ Visualization saved to: ../reports/figures/upstream_gages_analysis.png")

    except Exception as e:
        print(f"⚠️ Failed to create visualization: {e}")

def create_enhanced_gage_maps():
    """Create enhanced maps with insets for each terminal gage"""

    print("=== Creating Enhanced Terminal Gage Maps ===")

    # Load data
    try:
        subbasin_gdf = gpd.read_file('../data/raw/hydrography/gsl_catchment.shp')
        stream_gdf = gpd.read_file('../data/raw/hydrography/gslb_stream.shp')
        lake_gdf = gpd.read_file('../data/raw/hydrography/lake.shp')
        well_gdf = gpd.read_file('../data/raw/hydrography/well_shp.shp')
        gage_df = pd.read_csv('../data/raw/hydrography/gsl_nwm_gage.csv')

        try:
            terminal_relationships = pd.read_csv('../data/processed/terminal_gage_upstream_catchments.csv')
        except:
            terminal_relationships = pd.read_csv('../data/processed/terminal_gage_upstream_catchments.csv')

        print("✅ Data loaded successfully")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Data preprocessing
    linkno_col = 'linkno' if 'linkno' in subbasin_gdf.columns else 'LINKNO'

    if 'Gage_ID' in terminal_relationships.columns:
        terminal_relationships = terminal_relationships.rename(columns={
            'Gage_ID': 'gage_id',
            'Upstream_Catchment_ID': 'upstream_catchment_id'
        })

    gage_gdf = gpd.GeoDataFrame(
        gage_df,
        geometry=gpd.points_from_xy(gage_df['longitude'], gage_df['latitude']),
        crs='EPSG:4326'
    )

    terminal_gage_ids = terminal_relationships['gage_id'].unique()
    print(f"🎯 Will create maps for {len(terminal_gage_ids)} terminal gages")

    output_dir = '../reports/figures/enhanced_gage_maps'
    os.makedirs(output_dir, exist_ok=True)

    # Create map for each terminal gage
    for terminal_gage_id in tqdm(terminal_gage_ids, desc="Creating maps"):
        try:
            create_single_map(
                terminal_gage_id, terminal_relationships, subbasin_gdf,
                stream_gdf, lake_gdf, well_gdf, gage_gdf, gage_df,
                linkno_col, output_dir
            )
        except Exception as e:
            print(f"❌ Failed to create map for Gage {terminal_gage_id}: {e}")

    print(f"✅ All maps saved to: {output_dir}")

def create_upstream_catchment_schematic():
    """
    Create a schematic diagram showing upstream catchment concept using watershed shapes
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Colors
    colors = {
        'terminal': '#FF6B6B',     # Red for terminal catchment
        'upstream': '#4ECDC4',     # Teal for upstream catchments
        'stream': '#45B7D1',       # Blue for streams
        'gage': '#FFA07A',         # Orange for gages
        'well': '#8B4513',         # Brown for wells
        'boundary': '#2C3E50'      # Dark for boundaries
    }
    
    # Draw watershed boundaries as irregular polygons
    # Terminal catchment (bottom center)
    terminal_catchment = patches.Polygon(
        [(7, 1), (10, 1.5), (11, 3), (10, 4), (8, 4.5), (6, 3.5), (5.5, 2)], 
        closed=True, facecolor=colors['terminal'], 
        edgecolor=colors['boundary'], linewidth=3, alpha=0.8)
    ax.add_patch(terminal_catchment)
    
    # Upstream catchments
    upstream_catchments = [
        # Left upstream catchment
        patches.Polygon([(1, 4), (3.5, 3.5), (4.5, 5), (3, 6.5), (1, 6), (0, 5)], 
                       closed=True, facecolor=colors['upstream'], 
                       edgecolor=colors['boundary'], linewidth=2, alpha=0.7),
        
        # Top left upstream catchment
        patches.Polygon([(2, 7), (4, 6.5), (5, 8), (4, 9.5), (2, 9), (1, 8)], 
                       closed=True, facecolor=colors['upstream'], 
                       edgecolor=colors['boundary'], linewidth=2, alpha=0.7),
        
        # Top middle upstream catchment  
        patches.Polygon([(5, 8.5), (7, 8), (8.5, 9.5), (7, 10.5), (5, 10), (4, 9)], 
                       closed=True, facecolor=colors['upstream'], 
                       edgecolor=colors['boundary'], linewidth=2, alpha=0.7),
        
        # Right upstream catchment
        patches.Polygon([(8.5, 5), (11, 4.5), (12, 6), (11, 7.5), (9, 7), (8, 6)], 
                       closed=True, facecolor=colors['upstream'], 
                       edgecolor=colors['boundary'], linewidth=2, alpha=0.7),
        
        # Top right upstream catchment
        patches.Polygon([(10, 8), (12.5, 7.5), (13, 9), (12, 10), (10, 9.5), (9, 8.5)], 
                       closed=True, facecolor=colors['upstream'], 
                       edgecolor=colors['boundary'], linewidth=2, alpha=0.7),
    ]
    
    for catchment in upstream_catchments:
        ax.add_patch(catchment)
    
    # Draw stream network connecting all catchments
    stream_segments = [
        # Main stem
        [(2.5, 5.5), (4, 4.5), (6, 3.5), (8.5, 2.8)],
        # Left tributary
        [(2.5, 8), (3, 7), (3.5, 6)],
        # Top tributary  
        [(6, 9), (6.5, 8.5), (5.5, 7), (5, 5.5)],
        # Right tributary
        [(10, 6.5), (9.5, 5.5), (8.5, 4.5)],
        # Far right tributary
        [(11.5, 8.5), (10.5, 7.5), (10, 6.5)],
    ]
    
    for segment in stream_segments:
        x_coords = [point[0] for point in segment]
        y_coords = [point[1] for point in segment]
        ax.plot(x_coords, y_coords, color=colors['stream'], linewidth=4, alpha=0.9, zorder=3)
        
        # Add flow direction arrows
        for i in range(len(segment)-1):
            start = segment[i]
            end = segment[i+1]
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                       xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                       arrowprops=dict(arrowstyle='->', color=colors['stream'], lw=2))
    
    # Add gages at key locations
    gage_locations = [
        (8.5, 2.8, 'Terminal Gage'),   # Terminal gage
        (4, 4.5, 'Gage A'),           # Upstream gage
        (6, 9, 'Gage B'),             # Upstream gage  
        (10, 6.5, 'Gage C'),          # Upstream gage
        (2.5, 8, 'Gage D'),           # Upstream gage
    ]
    
    for x, y, label in gage_locations:
        if 'Terminal' in label:
            ax.scatter(x, y, s=400, c=colors['gage'], marker='*', 
                      edgecolor='black', linewidth=2, zorder=5)
            ax.text(x+0.5, y, label, fontsize=11, fontweight='bold', 
                   ha='left', va='center')
        else:
            ax.scatter(x, y, s=200, c=colors['gage'], marker='o', 
                      edgecolor='black', linewidth=1.5, zorder=5)
            ax.text(x+0.3, y+0.3, label, fontsize=9, ha='left', va='bottom')
    
    # Add wells scattered throughout upstream catchments
    well_positions = [
        (2, 5), (3, 4.5), (1.5, 6), (3, 8), (2.5, 9), (6, 9.5), (7, 8.5),
        (9.5, 6), (11, 6.5), (10.5, 8.5), (11.5, 9), (9, 5.5), (10.5, 5.5)
    ]
    
    for x, y in well_positions:
        ax.scatter(x, y, s=80, c=colors['well'], marker='s', 
                  edgecolor='black', linewidth=1, alpha=0.8, zorder=4)
    
    # Add catchment labels
    catchment_labels = [
        (8.5, 2.5, 'Terminal\nCatchment', colors['terminal']),
        (2.5, 5, 'Upstream\nCatchment 1', colors['upstream']),
        (3, 8, 'Upstream\nCatchment 2', colors['upstream']),
        (6.5, 9, 'Upstream\nCatchment 3', colors['upstream']),
        (10, 6, 'Upstream\nCatchment 4', colors['upstream']),
        (11.5, 8.5, 'Upstream\nCatchment 5', colors['upstream']),
    ]
    
    for x, y, label, color in catchment_labels:
        ax.text(x, y, label, fontsize=10, ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor=color, linewidth=2, alpha=0.9))
    
    # Add concept explanation
    concept_text = """
    UPSTREAM CATCHMENT CONCEPT:
    
    • Terminal Gage: Point where we measure flow
    • Terminal Catchment: Area directly draining to terminal gage
    • Upstream Catchments: All areas that contribute flow 
      to the terminal gage through the stream network
    • Wells: Groundwater monitoring points within catchments
    • Stream Flow Direction: Water flows from upstream 
      catchments → terminal catchment → terminal gage
    """
    
    ax.text(0.5, 1, concept_text, fontsize=11, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', 
                    edgecolor='navy', alpha=0.9),
           verticalalignment='bottom', transform=ax.transAxes)
    
    # Add title
    ax.set_title('Upstream Catchment Identification\nWatershed Connectivity Concept', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [
        patches.Patch(color=colors['terminal'], label='Terminal Catchment'),
        patches.Patch(color=colors['upstream'], label='Upstream Catchments'),
        plt.Line2D([0], [0], color=colors['stream'], linewidth=3, label='Stream Network'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=colors['gage'], 
                  markersize=15, label='Terminal Gage', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['gage'], 
                  markersize=10, label='Upstream Gages', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['well'], 
                  markersize=8, label='Wells', markeredgecolor='black')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', 
             fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Set limits and styling
    ax.set_xlim(-1, 14)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

    print("✅ Upstream catchment schematic displayed")


def plot_delta_scatter_by_gage(
    data: pd.DataFrame,
    gage_class_df: pd.DataFrame,
    output_dir: str,
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    title_suffix: str = '',
    stats_prefix: str = 'Wells'
) -> pd.DataFrame:
    """
    Create per-gage ΔQ vs ΔWTE scatter plots with linear regression.

    Parameters
    ----------
    data : pd.DataFrame
        Paired well-streamflow data with delta columns
    gage_class_df : pd.DataFrame
        Gage classification table with STAID and CLASS columns
    output_dir : str
        Directory to save PNG plots and statistics CSV
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE values
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ values
    title_suffix : str, default ''
        Appended to each plot title
    stats_prefix : str, default 'Wells'
        Label prefix for the stats box (e.g. 'Wells' or 'Top 10% Wells')

    Returns
    -------
    pd.DataFrame
        Statistics table with one row per gage
    """
    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['gage_id'] = df['gage_id'].astype(str)

    if 'class' not in df.columns:
        gage_class_df = gage_class_df.copy()
        gage_class_df['STAID'] = gage_class_df['STAID'].astype(str)
        df = df.merge(gage_class_df[['STAID', 'CLASS']], left_on='gage_id',
                      right_on='STAID', how='left')
        df.drop('STAID', axis=1, inplace=True)
        df.rename(columns={'CLASS': 'class'}, inplace=True)

    stats_data = []
    for gage_id, group in df.groupby('gage_id'):
        group = group.dropna(subset=[delta_wte_col, delta_q_col])
        if len(group) < 2:
            continue

        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=group, x=delta_wte_col, y=delta_q_col,
                        hue='well_id', palette='viridis', legend=False, alpha=0.6)

        class_val = group['class'].iloc[0] if 'class' in group.columns else None
        class_display = {'Non-ref': 'Unregulated', 'Ref': 'Regulated'}.get(
            str(class_val) if pd.notna(class_val) else None, 'Unknown')

        if len(group[delta_wte_col].unique()) > 1:
            slope, intercept, r_value, p_value, _ = linregress(
                group[delta_wte_col], group[delta_q_col])
            x_range = [group[delta_wte_col].min(), group[delta_wte_col].max()]
            plt.plot(x_range, [intercept + slope * x for x in x_range], 'r-', linewidth=2)
            stats_data.append({'gage_id': gage_id, 'num_wells': group['well_id'].nunique(),
                                'num_measurements': len(group), 'slope': slope,
                                'intercept': intercept, 'r_squared': r_value ** 2,
                                'p_value': p_value, 'class': class_val})
            legend_text = (f"{stats_prefix}: {group['well_id'].nunique()}\n"
                           f"Measurements: {len(group)}\nSlope: {slope:.2f}\n"
                           f"R²: {r_value ** 2:.2f}\np-value: {p_value:.4f}")
        else:
            legend_text = (f"{stats_prefix}: {group['well_id'].nunique()}\n"
                           f"Measurements: {len(group)}\nNo variation in {delta_wte_col}")

        plt.title(f'Gage ID: {gage_id} - Class: {class_display}\n'
                  f'ΔQ vs. ΔWTE{title_suffix}', fontsize=16)
        plt.xlabel('ΔWTE (ft)')
        plt.ylabel('ΔQ (cfs)')
        plt.grid(True, alpha=0.3)
        plt.text(0.98, 0.95, legend_text, transform=plt.gca().transAxes,
                 fontsize=10, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.savefig(os.path.join(output_dir, f'gage_{gage_id}.png'),
                    bbox_inches='tight', dpi=600)
        plt.close()

    stats_df = pd.DataFrame(stats_data)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(output_dir, 'scatter_statistics.csv'), index=False)
        print(f"Generated {len(stats_df)} plots and saved statistics")
    else:
        print("Not enough data to generate charts")
    return stats_df


def plot_monthly_delta_scatter_by_gage(
    data: pd.DataFrame,
    gage_class_df: pd.DataFrame,
    output_dir: str,
    delta_wte_col: str,
    delta_q_col: str = 'delta_q',
    lag_label: str = ''
) -> pd.DataFrame:
    """
    Create per-gage monthly ΔQ vs lagged ΔWTE scatter plots (subplot grid per month).

    Parameters
    ----------
    data : pd.DataFrame
        Paired data including delta columns and date
    gage_class_df : pd.DataFrame
        Gage classification table with STAID and CLASS columns
    output_dir : str
        Directory to save PNG plots and statistics CSV
    delta_wte_col : str
        Column name for ΔWTE (or lagged ΔWTE)
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    lag_label : str, default ''
        Label for the lag in axis/title text (e.g. '2-year')

    Returns
    -------
    pd.DataFrame
        Statistics table with one row per gage-month combination
    """
    os.makedirs(output_dir, exist_ok=True)

    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                   5: 'May', 6: 'June', 7: 'July', 8: 'August',
                   9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    df = data.copy()
    df['gage_id'] = df['gage_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    gage_class_df = gage_class_df.copy()
    gage_class_df['STAID'] = gage_class_df['STAID'].astype(str)
    df = df.merge(gage_class_df[['STAID', 'CLASS']], left_on='gage_id',
                  right_on='STAID', how='left')
    df.drop('STAID', axis=1, inplace=True)
    df.rename(columns={'CLASS': 'class'}, inplace=True)

    stats_data = []
    xlabel = f'Lagged ΔWTE (ft{", " + lag_label + " lag" if lag_label else ""})'

    for gage_id, group in df.groupby('gage_id'):
        group = group.dropna(subset=[delta_wte_col, delta_q_col])
        if len(group) < 2:
            continue

        class_val = group['class'].iloc[0] if 'class' in group.columns else None
        class_display = {'Non-ref': 'Unregulated', 'Ref': 'Regulated'}.get(
            str(class_val) if pd.notna(class_val) else None, 'Unknown')

        available_months = sorted(group['month'].unique())
        n = len(available_months)
        if n <= 4:
            rows, cols = 2, 2
        elif n <= 6:
            rows, cols = 2, 3
        elif n <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 3, 4

        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.flatten()

        for idx, month in enumerate(available_months):
            if idx >= len(axes):
                break
            ax = axes[idx]
            mdata = group[group['month'] == month]
            if len(mdata) == 0:
                ax.set_visible(False)
                continue

            sns.scatterplot(data=mdata, x=delta_wte_col, y=delta_q_col,
                            hue='well_id', palette='viridis', edgecolor='none',
                            legend=False, ax=ax)

            if len(mdata[delta_wte_col].unique()) > 1:
                slope, intercept, r_value, p_value, _ = linregress(
                    mdata[delta_wte_col], mdata[delta_q_col])
                x_vals = mdata[delta_wte_col].sort_values()
                ax.plot(x_vals, intercept + slope * x_vals, 'r', linewidth=2)
                stats_data.append({'gage_id': gage_id, 'month': month,
                                   'month_name': month_names[month],
                                   'num_wells': mdata['well_id'].nunique(),
                                   'num_measurements': len(mdata), 'slope': slope,
                                   'intercept': intercept, 'r_squared': r_value ** 2,
                                   'p_value': p_value, 'class': class_display})
                legend_text = (f"Wells: {mdata['well_id'].nunique()}\n"
                               f"Measurements: {len(mdata)}\nSlope: {slope:.2f}\n"
                               f"R²: {r_value ** 2:.2f}\np-value: {p_value:.4f}")
            else:
                legend_text = (f"Wells: {mdata['well_id'].nunique()}\n"
                               f"Measurements: {len(mdata)}\nNo variation in ΔWTE")

            ax.set_title(month_names[month], fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel('ΔQ (cfs)', fontsize=10)
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.xaxis.grid(False)
            ax.set_facecolor('white')
            ax.text(0.98, 0.95, legend_text, transform=ax.transAxes, fontsize=8,
                    ha='right', va='top', linespacing=1.2,
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.75,
                              boxstyle='square,pad=0.3'))

        for idx in range(len(available_months), len(axes)):
            axes[idx].set_visible(False)

        lag_title = f' ({lag_label})' if lag_label else ''
        fig.suptitle(f'Gage ID: {gage_id} - Class: {class_display}\n'
                     f'ΔQ vs Lagged ΔWTE{lag_title} by Month',
                     fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(os.path.join(output_dir, f'gage_{gage_id}_monthly.png'),
                    bbox_inches='tight', dpi=600)
        plt.close()

    stats_df = pd.DataFrame(stats_data)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(output_dir, 'monthly_scatter_statistics.csv'), index=False)
        print(f"Generated monthly scatter plots for {df['gage_id'].nunique()} gages")
    else:
        print("No valid data found for analysis")
    return stats_df


def plot_seasonal_delta_scatter_by_gage(
    data: pd.DataFrame,
    gage_class_df: pd.DataFrame,
    output_dir: str,
    delta_wte_col: str,
    delta_q_col: str = 'delta_q'
) -> pd.DataFrame:
    """
    Create per-gage seasonal ΔQ vs lagged ΔWTE scatter plots.

    Parameters
    ----------
    data : pd.DataFrame
        Paired data including delta columns and date
    gage_class_df : pd.DataFrame
        Gage classification table with STAID and CLASS columns
    output_dir : str
        Directory to save PNG plots and statistics CSV
    delta_wte_col : str
        Column name for lagged ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ

    Returns
    -------
    pd.DataFrame
        Statistics table with one row per gage-season combination
    """
    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()
    df['gage_id'] = df['gage_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].apply(get_season_from_month)
    gage_class_df = gage_class_df.copy()
    gage_class_df['STAID'] = gage_class_df['STAID'].astype(str)
    df = df.merge(gage_class_df[['STAID', 'CLASS']], left_on='gage_id',
                  right_on='STAID', how='left')
    df.drop('STAID', axis=1, inplace=True)
    df.rename(columns={'CLASS': 'class'}, inplace=True)

    stats_data = []
    for gage_id, gage_group in df.groupby('gage_id'):
        for season in sorted(gage_group['season'].unique()):
            sdata = gage_group[gage_group['season'] == season].dropna(
                subset=[delta_wte_col, delta_q_col])
            if len(sdata) < 2:
                continue

            class_val = sdata['class'].iloc[0] if 'class' in sdata.columns else None
            class_display = {'Non-ref': 'Unregulated', 'Ref': 'Regulated'}.get(
                str(class_val) if pd.notna(class_val) else None, 'Unknown')

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=delta_wte_col, y=delta_q_col, data=sdata,
                            hue='well_id', palette='viridis', alpha=0.6, s=50, legend=False)

            if len(sdata[delta_wte_col].unique()) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(
                    sdata[delta_wte_col], sdata[delta_q_col])
                sns.regplot(x=delta_wte_col, y=delta_q_col, data=sdata,
                            scatter=False, color='red', line_kws={'linewidth': 2})
                stats_data.append({'gage_id': gage_id, 'season': season,
                                   'num_wells': sdata['well_id'].nunique(),
                                   'num_measurements': len(sdata), 'slope': slope,
                                   'intercept': intercept, 'r_squared': r_value ** 2,
                                   'p_value': p_value, 'std_err': std_err,
                                   'class': class_val})
                stats_text = (f"Wells: {sdata['well_id'].nunique()}\n"
                              f"Measurements: {len(sdata)}\nSlope: {slope:.2f} cfs/ft\n"
                              f"R²: {r_value**2:.3f}\nP-value: {p_value:.4f}")
            else:
                stats_text = (f"Wells: {sdata['well_id'].nunique()}\n"
                              f"Measurements: {len(sdata)}\n"
                              "All lagged ΔWTE values identical\nNo regression line")

            plt.title(f'Gage {gage_id} - {season} - Class: {class_display}\n'
                      f'ΔQ vs Lagged ΔWTE', fontsize=14)
            plt.xlabel(f'Lagged ΔWTE (ft)', fontsize=12)
            plt.ylabel('ΔQ (cfs)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
            fname = f'gage_{gage_id}_{season}_delta_q_vs_lag_delta_wte.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=600)
            plt.close()

    stats_df = pd.DataFrame(stats_data)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(output_dir, 'seasonal_scatter_statistics.csv'), index=False)
        print("Seasonal Summary Statistics:")
        for season in sorted(stats_df['season'].unique()):
            sd = stats_df[stats_df['season'] == season]
            print(f"  {season}: {len(sd)} gages, avg R²={sd['r_squared'].mean():.3f}, "
                  f"{(sd['slope'] > 0).mean()*100:.1f}% positive slope")
    else:
        print("No valid data for regression analysis")
    return stats_df


def plot_monthly_timeseries_by_gage(
    data: pd.DataFrame,
    output_dir: str,
    delta_wte_col: str = 'delta_wte_lag1_year',
    delta_q_col: str = 'delta_q'
) -> pd.DataFrame:
    """
    Create per-gage monthly dual-panel time series (ΔQ on top, lagged ΔWTE on bottom).

    Parameters
    ----------
    data : pd.DataFrame
        Paired data with delta columns and date
    output_dir : str
        Directory to save PNG plots and statistics CSV
    delta_wte_col : str, default 'delta_wte_lag1_year'
        Column name for lagged ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ

    Returns
    -------
    pd.DataFrame
        Statistics table with one row per gage-month combination
    """
    os.makedirs(output_dir, exist_ok=True)
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')

    stats_data = []
    for gage_id, group in df.groupby('gage_id'):
        group = group.dropna(subset=[delta_wte_col, 'date', delta_q_col]).sort_values('date')
        for month in sorted(group['month'].unique()):
            mdata = group[group['month'] == month].copy()
            month_name = mdata['month_name'].iloc[0]
            if len(mdata) < 2:
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            ax1.scatter(mdata['date'], mdata[delta_q_col], color='orange', alpha=0.6)

            if len(mdata['date'].unique()) > 1:
                mdata['date_num'] = mdata['date'].map(pd.Timestamp.toordinal)
                sq, iq, rq, pq, _ = linregress(mdata['date_num'], mdata[delta_q_col])
                ax1.plot(mdata['date'], iq + sq * mdata['date_num'], 'r')
                txt_q = (f"Points: {len(mdata)}\nSlope: {sq:.6f}\n"
                         f"R²: {rq**2:.2f}\nP: {pq:.4f}")
            else:
                sq = rq = pq = None
                txt_q = f"Points: {len(mdata)}\nInsufficient data"

            ax1.set_ylabel('ΔQ (cfs)')
            ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax1.text(0.05, 0.95, txt_q, transform=ax1.transAxes, fontsize=10,
                     va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            ax2.scatter(mdata['date'], mdata[delta_wte_col], color='blue', alpha=0.6)
            if sq is not None:
                sw, iw, rw, pw, _ = linregress(mdata['date_num'], mdata[delta_wte_col])
                ax2.plot(mdata['date'], iw + sw * mdata['date_num'], 'r')
                stats_data.append({'gage_id': gage_id, 'month': month, 'month_name': month_name,
                                   'num_wells': mdata['well_id'].nunique(),
                                   'num_measurements': len(mdata),
                                   'slope_delta_q': sq, 'r_squared_delta_q': rq**2,
                                   'p_value_delta_q': pq,
                                   'slope_lag_wte': sw, 'r_squared_lag_wte': rw**2,
                                   'p_value_lag_wte': pw})
                txt_w = (f"Wells: {mdata['well_id'].nunique()}\nMeasurements: {len(mdata)}\n"
                         f"Slope: {sw:.6f}\nR²: {rw**2:.2f}\nP: {pw:.4f}")
            else:
                txt_w = f"Wells: {mdata['well_id'].nunique()}\nInsufficient data"

            ax2.set_ylabel('Lagged ΔWTE (ft)')
            ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax2.text(0.05, 0.95, txt_w, transform=ax2.transAxes, fontsize=10,
                     va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            fig.suptitle(f'Gage {gage_id} - {month_name} - ΔQ and Lagged ΔWTE vs. Time',
                         fontsize=16, y=0.92)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(output_dir, f'gage_{gage_id}_{month:02d}_{month_name}.png'),
                        dpi=600, bbox_inches='tight')
            plt.close()

    stats_df = pd.DataFrame(stats_data)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(output_dir, 'monthly_timeseries_statistics.csv'), index=False)
        print(f"Monthly timeseries: {df['gage_id'].nunique()} gages")
    else:
        print("No statistical data collected")
    return stats_df


def plot_seasonal_timeseries_by_gage(
    data: pd.DataFrame,
    output_dir: str,
    delta_wte_col: str = 'delta_wte_lag1_year',
    delta_q_col: str = 'delta_q'
) -> pd.DataFrame:
    """
    Create per-gage seasonal dual-panel time series (ΔQ on top, lagged ΔWTE on bottom).

    Parameters
    ----------
    data : pd.DataFrame
        Paired data with delta columns and date
    output_dir : str
        Directory to save PNG plots and statistics CSV
    delta_wte_col : str, default 'delta_wte_lag1_year'
        Column name for lagged ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ

    Returns
    -------
    pd.DataFrame
        Statistics table with one row per gage-season combination
    """
    os.makedirs(output_dir, exist_ok=True)
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].apply(get_season_from_month)

    stats_data = []
    for gage_id, group in df.groupby('gage_id'):
        group = group.dropna(subset=[delta_wte_col, 'date', delta_q_col]).sort_values('date')
        for season in sorted(group['season'].unique()):
            sdata = group[group['season'] == season].copy()
            if len(sdata) < 2:
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            ax1.scatter(sdata['date'], sdata[delta_q_col], color='orange', alpha=0.6)

            if len(sdata['date'].unique()) > 1:
                sdata['date_num'] = sdata['date'].map(pd.Timestamp.toordinal)
                sq, iq, rq, pq, _ = linregress(sdata['date_num'], sdata[delta_q_col])
                ax1.plot(sdata['date'], iq + sq * sdata['date_num'], 'r')
                txt_q = (f"Points: {len(sdata)}\nSlope: {sq:.6f}\n"
                         f"R²: {rq**2:.2f}\nP: {pq:.4f}")
            else:
                sq = rq = pq = None
                txt_q = f"Points: {len(sdata)}\nInsufficient data"

            ax1.set_ylabel('ΔQ (cfs)')
            ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax1.text(0.05, 0.95, txt_q, transform=ax1.transAxes, fontsize=10,
                     va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            ax2.scatter(sdata['date'], sdata[delta_wte_col], color='blue', alpha=0.6)
            if sq is not None:
                sw, iw, rw, pw, _ = linregress(sdata['date_num'], sdata[delta_wte_col])
                ax2.plot(sdata['date'], iw + sw * sdata['date_num'], 'r')
                stats_data.append({'gage_id': gage_id, 'season': season,
                                   'num_wells': sdata['well_id'].nunique(),
                                   'num_measurements': len(sdata),
                                   'slope_delta_q': sq, 'r_squared_delta_q': rq**2,
                                   'p_value_delta_q': pq,
                                   'slope_lag_wte': sw, 'r_squared_lag_wte': rw**2,
                                   'p_value_lag_wte': pw})
                txt_w = (f"Wells: {sdata['well_id'].nunique()}\nMeasurements: {len(sdata)}\n"
                         f"Slope: {sw:.6f}\nR²: {rw**2:.2f}\nP: {pw:.4f}")
            else:
                txt_w = f"Wells: {sdata['well_id'].nunique()}\nInsufficient data"

            ax2.set_ylabel('Lagged ΔWTE (ft)')
            ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax2.text(0.05, 0.95, txt_w, transform=ax2.transAxes, fontsize=10,
                     va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            fig.suptitle(f'Gage {gage_id} - {season} - ΔQ and Lagged ΔWTE vs. Time',
                         fontsize=16, y=0.92)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(output_dir, f'gage_{gage_id}_{season}.png'),
                        dpi=600, bbox_inches='tight')
            plt.close()

    stats_df = pd.DataFrame(stats_data)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(output_dir, 'seasonal_timeseries_statistics.csv'), index=False)
        print(f"Seasonal timeseries: {df['gage_id'].nunique()} gages")
    else:
        print("No statistical data collected")
    return stats_df


def plot_slope_lag_analysis(data: pd.DataFrame) -> None:
    """
    Create 4-panel slope analysis across lag periods (line, heatmap, boxplot, faceted).

    Parameters
    ----------
    data : pd.DataFrame
        Slope table with columns: gage_id, no lag, 3 month, 6 month, 1 yr, 2 yr, 3 yr
    """
    df = data.copy()
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['gage_id'] = df['gage_id'].astype(str).str.replace('.0', '', regex=False)

    lag_periods = ['no lag', '3 month', '6 month', '1 yr', '2 yr', '3 yr']

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        ax.plot(lag_periods, row[lag_periods].values, marker='o',
                label=f"Gage {row['gage_id']}", linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag Period', fontsize=12)
    ax.set_ylabel('Slope (ft/year)', fontsize=12)
    ax.set_title('Slope Changes Across Different Lag Periods', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('slope_by_lag_lines.png', dpi=600, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.set_index('gage_id')[lag_periods], annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, cbar_kws={'label': 'Slope (ft/year)'}, ax=ax)
    ax.set_title('Slope Heatmap Across Different Lag Periods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('slope_heatmap.png', dpi=600, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    df_melt = df.melt(id_vars=['gage_id'], value_vars=lag_periods,
                      var_name='Lag Period', value_name='Slope')
    sns.boxplot(data=df_melt, x='Lag Period', y='Slope', ax=ax)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Distribution of Slopes Across Different Lag Periods', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('slope_boxplot.png', dpi=600, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes.flatten()[idx]
        vals = row[lag_periods].values.astype(float)
        colors = ['steelblue'] * len(vals)
        if not np.all(np.isnan(vals)):
            colors[np.nanargmax(np.abs(vals))] = 'darkred'
        ax.bar(range(len(lag_periods)), vals, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_title(f"Gage {row['gage_id']}", fontweight='bold')
        ax.set_ylabel('Slope (ft/year)')
        ax.set_xticks(range(len(lag_periods)))
        ax.set_xticklabels(lag_periods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Individual Gage Slope Patterns Across Lag Periods',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('slope_individual_gages.png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_r2_lag_analysis(data: pd.DataFrame) -> None:
    """
    Create 4-panel R² analysis across lag periods (line, heatmap, boxplot, faceted).

    Parameters
    ----------
    data : pd.DataFrame
        R² table with columns: gage_id, no lag, 3 month, 6 month, 1 yr, 2 yr, 3 yr
    """
    df = data.copy()
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['gage_id'] = df['gage_id'].astype(str).str.replace('.0', '', regex=False)

    lag_periods = ['no lag', '3 month', '6 month', '1 yr', '2 yr', '3 yr']

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        ax.plot(lag_periods, row[lag_periods].values, marker='o',
                label=f"Gage {row['gage_id']}", linewidth=2)
    ax.set_xlabel('Lag Period', fontsize=12)
    ax.set_ylabel('R² Value', fontsize=12)
    ax.set_title('R² Changes Across Different Lag Periods', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rsquare_by_lag_lines.png', dpi=600, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.set_index('gage_id')[lag_periods], annot=True, fmt='.3f',
                cmap='YlOrRd', cbar_kws={'label': 'R² Value'}, ax=ax, vmin=0, vmax=1)
    ax.set_title('R² Heatmap Across Different Lag Periods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rsquare_heatmap.png', dpi=600, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    df_melt = df.melt(id_vars=['gage_id'], value_vars=lag_periods,
                      var_name='Lag Period', value_name='R²')
    sns.boxplot(data=df_melt, x='Lag Period', y='R²', ax=ax)
    ax.set_title('Distribution of R² Across Different Lag Periods', fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rsquare_boxplot.png', dpi=600, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes.flatten()[idx]
        vals = row[lag_periods].values.astype(float)
        colors = ['steelblue'] * len(vals)
        if not np.all(np.isnan(vals)):
            colors[np.nanargmax(vals)] = 'darkred'
        ax.bar(range(len(lag_periods)), vals, alpha=0.7, color=colors)
        ax.set_title(f"Gage {row['gage_id']}", fontweight='bold')
        ax.set_ylabel('R² Value')
        ax.set_xticks(range(len(lag_periods)))
        ax.set_xticklabels(lag_periods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(0.1, np.nanmax(vals) * 1.1))
    plt.suptitle('Individual Gage R² Patterns Across Lag Periods',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('rsquare_individual_gages.png', dpi=600, bbox_inches='tight')
    plt.show()