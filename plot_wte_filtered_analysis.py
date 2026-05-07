#!/usr/bin/env python3
"""
Filter wells with WTE < threshold and plot analysis.

This script:
1. Filters pairs with R² > threshold
2. For specified gage, finds wells with WTE < threshold
3. Plots time series for filtered wells
4. Removes filtered wells and plots updated scatter plot
"""

import argparse
import os
import sys
import importlib.util
import pandas as pd

# Set matplotlib to use non-interactive backend before importing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Directly import modules without triggering __init__.py
def load_module_from_file(module_name, file_path):
    """Load a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load modules directly
base_path = os.path.dirname(os.path.abspath(__file__))
metrics_module = load_module_from_file('metrics', os.path.join(base_path, 'gwbase', 'metrics.py'))
visualization_module = load_module_from_file('visualization', os.path.join(base_path, 'gwbase', 'visualization.py'))

# Import needed functions
filter_pairs_by_r_squared = metrics_module.filter_pairs_by_r_squared
plot_filtered_pairs_by_gage = visualization_module.plot_filtered_pairs_by_gage
format_p_value = visualization_module.format_p_value
from scipy.stats import linregress


def plot_well_wte_timeseries_filtered(
    data: pd.DataFrame,
    well_ids: list,
    gage_id: int,
    output_dir: str,
    wte_threshold: float = -10.0,
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    figsize: tuple = (15, 6)
):
    """Plot WTE time series for filtered wells."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Filter to specified gage and wells
    gage_data = data[data[gage_id_col] == gage_id]
    gage_data = gage_data[gage_data[well_id_col].isin(well_ids)]
    
    print(f"\nPlotting time series for {len(well_ids)} wells with WTE < -10...")
    
    for well_id in well_ids:
        well_data = gage_data[gage_data[well_id_col] == well_id].copy()
        well_data = well_data.sort_values(date_col)
        
        if len(well_data) < 10:
            continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot WTE time series
        ax.scatter(well_data[date_col], well_data[wte_col],
                  alpha=0.6, s=10, color='steelblue', edgecolors='black', linewidth=0.3)
        
        # Add trend line
        if len(well_data) > 1:
            z = np.polyfit(mdates.date2num(well_data[date_col]), well_data[wte_col], 1)
            p = np.poly1d(z)
            ax.plot(well_data[date_col], p(mdates.date2num(well_data[date_col])),
                   "r--", alpha=0.5, linewidth=1, label='Trend line')
        
        ax.set_ylabel('WTE (feet)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(f'Well {well_id} - Gage {gage_id}\n'
                    f'ΔWTE < {wte_threshold} ({len(well_data):,} observations)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format x-axis with detailed dates
        # Determine date range
        date_range = (well_data[date_col].max() - well_data[date_col].min()).days
        
        if date_range > 365 * 10:  # More than 10 years
            # Use year format for long time series
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
        elif date_range > 365 * 2:  # More than 2 years
            # Use year-month format
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        else:  # Less than 2 years
            # Use year-month-day format
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'gage_{gage_id}_well_{well_id}_wte_timeseries.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")


def plot_gage_scatter_filtered(
    data: pd.DataFrame,
    well_stats: pd.DataFrame,
    gage_id: int,
    excluded_well_ids: list,
    output_dir: str,
    wte_threshold: float = -10.0,
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    figsize: tuple = (12, 8)
):
    """Plot scatter plot for gage after removing excluded wells."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to specified gage
    gage_data = data[data[gage_id_col] == gage_id].copy()
    
    # Remove excluded wells
    gage_data_filtered = gage_data[~gage_data[well_id_col].isin(excluded_well_ids)].copy()
    gage_data_filtered = gage_data_filtered.dropna(subset=[delta_wte_col, delta_q_col])
    
    if len(gage_data_filtered) < 2:
        print(f"ERROR: Not enough data after filtering (only {len(gage_data_filtered)} records)")
        return
    
    # Get all wells for this gage (after filtering)
    wells = gage_data_filtered[well_id_col].unique()
    
    print(f"\nPlotting scatter plot for Gage {gage_id} (excluding {len(excluded_well_ids)} wells)...")
    print(f"  Wells remaining: {len(wells)}")
    print(f"  Total observations: {len(gage_data_filtered)}")
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Use a colormap to distinguish different wells
    colors = plt.cm.tab20(np.linspace(0, 1, len(wells)))
    
    # Plot each well with different color
    for i, well_id in enumerate(wells):
        well_data = gage_data_filtered[gage_data_filtered[well_id_col] == well_id]
        
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
    
    # Compute overall regression for the gage (all wells combined)
    x = gage_data_filtered[delta_wte_col].values
    y = gage_data_filtered[delta_q_col].values
    
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2
        
        # Plot overall regression line
        x_range = np.array([x.min(), x.max()])
        y_range = intercept + slope * x_range
        plt.plot(x_range, y_range, 'r-', linewidth=2.5, 
                label=f'Overall regression (R²={r_squared:.3f})', zorder=10)
    else:
        slope = intercept = r_squared = p_value = std_err = np.nan
    
    # Add statistics text
    stats_text = (
        f"Gage ID: {gage_id}\n"
        f"Number of wells: {len(wells)}\n"
        f"Total observations: {len(gage_data_filtered)}\n"
        f"Excluded wells (ΔWTE < {wte_threshold}): {len(excluded_well_ids)}\n"
        f"Overall slope: {slope:.2f}\n"
        f"Overall R²: {r_squared:.2f}\n"
        f"p-value: {format_p_value(p_value)}"
    )
    
    plt.text(0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            fontsize=11, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.xlabel('ΔWTE (ft)', fontsize=12)
    plt.ylabel('ΔQ (cfs)', fontsize=12)
    plt.title(f'Gage {gage_id}\nΔQ vs ΔWTE (R² > 0.3, excluding ΔWTE < {wte_threshold} wells)', 
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
    filename = f'gage_{gage_id}_scatter_filtered.png'
    plt.savefig(os.path.join(output_dir, filename),
               bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter wells with WTE < threshold and plot analysis'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to data_with_deltas.csv file'
    )
    parser.add_argument(
        '--well-stats-file',
        type=str,
        required=True,
        help='Path to regression_by_well.csv file'
    )
    parser.add_argument(
        '--r-squared-threshold',
        type=float,
        default=0.3,
        help='R² threshold (default: 0.3)'
    )
    parser.add_argument(
        '--gage-id',
        type=int,
        required=True,
        help='Gage ID to analyze'
    )
    parser.add_argument(
        '--wte-threshold',
        type=float,
        default=-10.0,
        help='WTE threshold (default: -10.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/wte_filtered_analysis',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Filter Wells with WTE < Threshold and Plot Analysis")
    print("="*60)
    print(f"Data file: {args.data_file}")
    print(f"Well stats file: {args.well_stats_file}")
    print(f"R² threshold: {args.r_squared_threshold}")
    print(f"Gage ID: {args.gage_id}")
    print(f"WTE threshold: {args.wte_threshold}")
    print(f"Output directory: {args.output_dir}")

    # Load data
    print("\nLoading data...")
    data = pd.read_csv(args.data_file)
    data['date'] = pd.to_datetime(data['date'])
    print(f"  Loaded {len(data):,} records")
    
    # Load well stats
    print("\nLoading regression statistics...")
    well_stats = pd.read_csv(args.well_stats_file)
    print(f"  Loaded statistics for {len(well_stats)} pairs")

    # Filter pairs by R² threshold
    print(f"\nFiltering pairs with R² > {args.r_squared_threshold}...")
    filtered_data = filter_pairs_by_r_squared(
        data,
        well_stats,
        r_squared_threshold=args.r_squared_threshold
    )

    if len(filtered_data) == 0:
        print("ERROR: No pairs meet the R² threshold")
        return

    # Filter to specified gage
    gage_data = filtered_data[filtered_data['gage_id'].astype(str).astype(int) == args.gage_id].copy()
    
    if len(gage_data) == 0:
        print(f"ERROR: No data found for Gage {args.gage_id}")
        return
    
    print(f"\nGage {args.gage_id} data:")
    print(f"  Total records: {len(gage_data):,}")
    print(f"  Unique wells: {gage_data['well_id'].nunique()}")
    
    # Find wells with delta_wte < threshold
    # Check if any delta_wte value for each well is below threshold
    wells_below_threshold = []
    for well_id in gage_data['well_id'].unique():
        well_data = gage_data[gage_data['well_id'] == well_id]
        if (well_data['delta_wte'] < args.wte_threshold).any():
            wells_below_threshold.append(well_id)
    
    print(f"\nWells with ΔWTE < {args.wte_threshold}:")
    print(f"  Found {len(wells_below_threshold)} wells")
    
    if len(wells_below_threshold) == 0:
        print("  No wells found with WTE < threshold")
        return
    
    # Plot time series for filtered wells
    timeseries_dir = os.path.join(args.output_dir, 'timeseries_wte_filtered')
    plot_well_wte_timeseries_filtered(
        filtered_data,
        wells_below_threshold,
        args.gage_id,
        timeseries_dir,
        wte_threshold=args.wte_threshold
    )
    
    # Plot scatter plot after removing filtered wells
    scatter_dir = os.path.join(args.output_dir, 'scatter_filtered')
    plot_gage_scatter_filtered(
        filtered_data,
        well_stats,
        args.gage_id,
        wells_below_threshold,
        scatter_dir,
        wte_threshold=args.wte_threshold
    )

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print(f"  - Time series: timeseries_wte_filtered/")
    print(f"  - Scatter plot: scatter_filtered/")


if __name__ == '__main__':
    main()
