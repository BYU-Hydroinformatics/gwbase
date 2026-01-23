"""
Visualization functions for GWBASE.

This module provides plotting functions for:
- Well time series plots
- Delta metrics scatter plots
- Regression analysis visualizations
- MI comparison plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import linregress
import os
from typing import Tuple


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
            slope, intercept, r_value, p_value, std_err = linregress(
                group[delta_wte_col], group[delta_q_col]
            )

            x_range = [group[delta_wte_col].min(), group[delta_wte_col].max()]
            y_range = [intercept + slope * x for x in x_range]
            plt.plot(x_range, y_range, 'r-', linewidth=2)

            class_value = None
            if class_col and class_col in group.columns:
                class_value = group[class_col].iloc[0]

            stats_data.append({
                'gage_id': gage_id,
                'n_wells': group[well_id_col].nunique(),
                'n_observations': len(group),
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
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
