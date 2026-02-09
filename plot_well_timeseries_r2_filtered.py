#!/usr/bin/env python3
"""
Plot groundwater time series for wells with R² > threshold for specific gages.

This script:
1. Loads paired data with delta metrics
2. Loads regression statistics
3. Filters to pairs with R² > threshold
4. Plots groundwater time series (WTE) for each well in specified gages
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

# Import needed functions
filter_pairs_by_r_squared = metrics_module.filter_pairs_by_r_squared


def plot_well_wte_timeseries(
    data: pd.DataFrame,
    well_stats: pd.DataFrame,
    gage_ids: list,
    output_dir: str,
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    figsize: tuple = (15, 6)
):
    """
    Plot groundwater time series (WTE) for each well in specified gages.
    
    Parameters
    ----------
    data : pd.DataFrame
        Filtered paired data with WTE
    well_stats : pd.DataFrame
        Regression statistics per well-gage pair
    gage_ids : list
        List of gage IDs to plot
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Convert gage_ids to match data type
    # Try to convert to int if possible, otherwise keep as string
    try:
        gage_ids_int = [int(gid) for gid in gage_ids]
        # Convert data gage_id to int for comparison
        data[gage_id_col] = data[gage_id_col].astype(str).astype(int)
        data = data[data[gage_id_col].isin(gage_ids_int)]
    except:
        # If conversion fails, use string comparison
        data[gage_id_col] = data[gage_id_col].astype(str)
        data = data[data[gage_id_col].isin([str(gid) for gid in gage_ids])]
    
    # Get R² for each well-gage pair
    data_with_r2 = data.merge(
        well_stats[[well_id_col, gage_id_col, 'r_squared']],
        on=[well_id_col, gage_id_col],
        how='inner'
    )
    
    for gage_id in gage_ids:
        gage_data = data_with_r2[data_with_r2[gage_id_col] == gage_id]
        wells = gage_data[well_id_col].unique()
        
        print(f"\nProcessing Gage {gage_id}: {len(wells)} wells")
        
        for well_id in wells:
            well_data = gage_data[gage_data[well_id_col] == well_id].copy()
            well_data = well_data.sort_values(date_col)
            
            if len(well_data) < 10:
                continue
            
            # Get R² for this well-gage pair
            well_r2 = well_data['r_squared'].iloc[0]
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot WTE time series
            ax.scatter(well_data[date_col], well_data[wte_col],
                      alpha=0.6, s=10, color='steelblue', edgecolors='black', linewidth=0.3)
            
            # Add trend line (optional)
            if len(well_data) > 1:
                z = np.polyfit(mdates.date2num(well_data[date_col]), well_data[wte_col], 1)
                p = np.poly1d(z)
                ax.plot(well_data[date_col], p(mdates.date2num(well_data[date_col])),
                       "r--", alpha=0.5, linewidth=1, label='Trend line')
            
            ax.set_ylabel('WTE (feet)', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title(f'Well {well_id} - Gage {gage_id}\n'
                        f'R² = {well_r2:.2f} ({len(well_data):,} observations)', 
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
    
    print(f"\nPlots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot groundwater time series for wells with R² > threshold'
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
        '--gage-ids',
        type=str,
        nargs='+',
        required=True,
        help='Gage IDs to plot (e.g., 10126000 10141000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/well_timeseries',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Plot Groundwater Time Series for Filtered Wells")
    print("="*60)
    print(f"Data file: {args.data_file}")
    print(f"Well stats file: {args.well_stats_file}")
    print(f"R² threshold: {args.r_squared_threshold}")
    print(f"Gage IDs: {args.gage_ids}")
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

    # Get filtered well_stats
    filtered_well_stats = well_stats[well_stats['r_squared'] > args.r_squared_threshold]

    # Plot time series
    print(f"\nGenerating time series plots...")
    # Convert gage_ids to int for consistency
    gage_ids = [int(gid) for gid in args.gage_ids]
    plot_well_wte_timeseries(
        filtered_data,
        filtered_well_stats,
        gage_ids=gage_ids,
        output_dir=args.output_dir
    )

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
