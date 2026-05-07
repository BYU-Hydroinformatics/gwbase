#!/usr/bin/env python3
"""
Filter pairs by R² threshold and plot scatter plots.

This script:
1. Loads paired data with delta metrics
2. Computes regression statistics by well-gage pair
3. Filters to pairs with R² > 0.1
4. Plots scatter plots for filtered pairs with regression lines and slopes
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
compute_regression_by_well = metrics_module.compute_regression_by_well
filter_pairs_by_r_squared = metrics_module.filter_pairs_by_r_squared
plot_filtered_pairs_scatter = visualization_module.plot_filtered_pairs_scatter
plot_filtered_pairs_by_gage = visualization_module.plot_filtered_pairs_by_gage
plot_pairs_by_r2_category = visualization_module.plot_pairs_by_r2_category


def main():
    parser = argparse.ArgumentParser(
        description='Filter pairs by R² and plot scatter plots'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to data_with_deltas.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/filtered_pairs',
        help='Output directory for plots and statistics'
    )
    parser.add_argument(
        '--r-squared-threshold',
        type=float,
        default=0.1,
        help='Minimum R² threshold (default: 0.1)'
    )
    parser.add_argument(
        '--min-observations',
        type=int,
        default=10,
        help='Minimum observations required for regression (default: 10)'
    )
    parser.add_argument(
        '--well-stats-file',
        type=str,
        default=None,
        help='Path to existing regression_by_well.csv file (if provided, will use this instead of recomputing)'
    )
    parser.add_argument(
        '--plot-by-gage',
        action='store_true',
        help='Plot by gage (one plot per gage with all wells) instead of by well-gage pair'
    )
    parser.add_argument(
        '--plot-by-r2-category',
        action='store_true',
        help='Plot by R² category (multiple plots per gage, one for each R² range)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Filter Pairs by R² and Plot Scatter Plots")
    print("="*60)
    print(f"Data file: {args.data_file}")
    print(f"R² threshold: {args.r_squared_threshold}")
    print(f"Output directory: {args.output_dir}")

    # Load data
    print("\nLoading data...")
    data = pd.read_csv(args.data_file)
    data['date'] = pd.to_datetime(data['date'])
    print(f"  Loaded {len(data):,} records")
    print(f"  Unique well-gage pairs: {data.groupby(['well_id', 'gage_id']).ngroups}")

    # Load or compute regression statistics by well-gage pair
    if args.well_stats_file and os.path.exists(args.well_stats_file):
        print(f"\nLoading existing regression statistics from: {args.well_stats_file}")
        well_stats = pd.read_csv(args.well_stats_file)
        print(f"  Loaded statistics for {len(well_stats)} pairs")
        print(f"  Mean R²: {well_stats['r_squared'].mean():.4f}")
        print(f"  Median R²: {well_stats['r_squared'].median():.4f}")
    else:
        print("\nComputing regression statistics by well-gage pair...")
        if args.well_stats_file:
            print(f"  Warning: Specified file '{args.well_stats_file}' not found, recomputing...")
        well_stats = compute_regression_by_well(
            data,
            min_observations=args.min_observations
        )

    if len(well_stats) == 0:
        print("ERROR: No valid pairs found for regression analysis")
        return

    print(f"\nRegression statistics computed for {len(well_stats)} pairs")
    print(f"  Mean R²: {well_stats['r_squared'].mean():.4f}")
    print(f"  Median R²: {well_stats['r_squared'].median():.4f}")
    print(f"  Pairs with R² > {args.r_squared_threshold}: {(well_stats['r_squared'] > args.r_squared_threshold).sum()}")

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save filtered data and statistics
    filtered_data.to_csv(
        os.path.join(args.output_dir, 'filtered_data.csv'),
        index=False
    )
    filtered_well_stats.to_csv(
        os.path.join(args.output_dir, 'filtered_well_stats.csv'),
        index=False
    )

    # Plot scatter plots for filtered pairs
    print(f"\nGenerating scatter plots...")
    if args.plot_by_r2_category:
        # Plot by R² category (multiple plots per gage, one for each R² range)
        stats_summary = plot_pairs_by_r2_category(
            data,  # Use full data, not filtered
            well_stats,
            output_dir=os.path.join(args.output_dir, 'scatter_plots_by_category')
        )
        plot_stats_file = 'scatter_plots_by_category/category_statistics.csv'
    elif args.plot_by_gage:
        # Plot by gage (one plot per gage with all wells)
        stats_summary = plot_filtered_pairs_by_gage(
            filtered_data,
            filtered_well_stats,
            output_dir=os.path.join(args.output_dir, 'scatter_plots_by_gage'),
            r_squared_threshold=args.r_squared_threshold
        )
        plot_stats_file = 'scatter_plots_by_gage/gage_statistics.csv'
    else:
        # Plot by well-gage pair (one plot per pair)
        stats_summary = plot_filtered_pairs_scatter(
            filtered_data,
            filtered_well_stats,
            output_dir=os.path.join(args.output_dir, 'scatter_plots')
        )
        plot_stats_file = 'scatter_plots/filtered_pairs_statistics.csv'

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    if not args.plot_by_r2_category:
        print(f"  - Filtered data: filtered_data.csv")
        print(f"  - Statistics: filtered_well_stats.csv")
    if args.plot_by_r2_category:
        print(f"  - Scatter plots (by R² category): scatter_plots_by_category/")
        print(f"  - Category statistics: scatter_plots_by_category/category_statistics.csv")
    elif args.plot_by_gage:
        print(f"  - Scatter plots (by gage): scatter_plots_by_gage/")
        print(f"  - Gage statistics: scatter_plots_by_gage/gage_statistics.csv")
    else:
        print(f"  - Scatter plots (by pair): scatter_plots/")
        print(f"  - Plot statistics: scatter_plots/filtered_pairs_statistics.csv")


if __name__ == '__main__':
    main()
