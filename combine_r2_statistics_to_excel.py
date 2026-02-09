#!/usr/bin/env python3
"""
Combine R² statistics from different thresholds into CSV files.

This script:
1. Loads statistics from R² > 0, 0.1, 0.2, 0.3 analyses
2. Combines them into CSV files
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress


def main():
    parser = argparse.ArgumentParser(
        description='Combine R² statistics into CSV files'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='reports/output_20260203_145601',
        help='Base directory containing filtered_pairs_r2_* folders'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='reports/output_20260203_145601/r2_statistics_summary.xlsx',
        help='Output file path (will create CSV files)'
    )
    parser.add_argument(
        '--well-stats-file',
        type=str,
        default='reports/output_20260203_145601/features/regression_by_well.csv',
        help='Path to regression_by_well.csv (for unfiltered statistics)'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='reports/output_20260203_145601/features/data_with_deltas.csv',
        help='Path to data_with_deltas.csv (for computing threshold=0 statistics)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Combine R² Statistics to CSV")
    print("="*60)
    print(f"Base directory: {args.base_dir}")
    print(f"Output file: {args.output_file}")

    # Define thresholds and corresponding directories
    thresholds = [0, 0.1, 0.2, 0.3]
    dir_names = {
        0: None,  # Will compute from data
        0.1: 'filtered_pairs_r2_0.1_final',
        0.2: 'filtered_pairs_r2_0.2_final',
        0.3: 'filtered_pairs_r2_0.3_final'
    }

    # Create output directory
    csv_dir = args.output_file.replace('.xlsx', '_csv').replace('.xls', '_csv')
    os.makedirs(csv_dir, exist_ok=True)
    print(f"CSV files will be saved to: {csv_dir}")

    # Load data for computing threshold=0 statistics
    print("\nLoading data for threshold=0 (all data)...")
    all_data_df = None
    if args.data_file and os.path.exists(args.data_file):
        all_data_df = pd.read_csv(args.data_file)
        print(f"  Loaded {len(all_data_df):,} records")
    else:
        print(f"  Warning: Data file not found: {args.data_file}")

    # Load unfiltered statistics (all data)
    print("\nLoading unfiltered statistics (all data)...")
    unfiltered_stats = None
    if args.well_stats_file and os.path.exists(args.well_stats_file):
        unfiltered_stats = pd.read_csv(args.well_stats_file)
        print(f"  Loaded {len(unfiltered_stats)} pairs (unfiltered)")
    else:
        print(f"  Warning: Unfiltered stats file not found: {args.well_stats_file}")

    # Process each threshold
    all_data = []
    summary_data = []

    # Process threshold=0 (all data, no filter)
    print(f"\nProcessing R² > 0 (all data)...")
    if all_data_df is not None:
        # Compute gage-level statistics for all data
        gage_stats_list = []
        for gage_id, gage_data in all_data_df.groupby('gage_id'):
            gage_data = gage_data.dropna(subset=['delta_wte', 'delta_q'])

            if len(gage_data) < 2:
                continue

            x = gage_data['delta_wte'].values
            y = gage_data['delta_q'].values

            if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                r_squared = r_value ** 2
            else:
                slope = intercept = r_squared = p_value = std_err = np.nan

            gage_stats_list.append({
                'r2_threshold': 0,
                'gage_id': gage_id,
                'n_wells': gage_data['well_id'].nunique() if 'well_id' in gage_data.columns else None,
                'n_observations': len(gage_data),
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'r_value': r_value if not np.isnan(r_squared) else np.nan,
                'p_value': p_value,
                'std_err': std_err
            })

        if gage_stats_list:
            threshold_0_df = pd.DataFrame(gage_stats_list)
            all_data.append(threshold_0_df)

            # Add to summary
            summary_data.append({
                'R² Threshold': f'>{0} (all data)',
                'Number of Gages': len(threshold_0_df),
                'Total Wells': threshold_0_df['n_wells'].sum() if 'n_wells' in threshold_0_df.columns else None,
                'Total Observations': threshold_0_df['n_observations'].sum() if 'n_observations' in threshold_0_df.columns else None,
                'Mean Slope': threshold_0_df['slope'].mean() if 'slope' in threshold_0_df.columns else None,
                'Median Slope': threshold_0_df['slope'].median() if 'slope' in threshold_0_df.columns else None,
                'Mean R²': threshold_0_df['r_squared'].mean() if 'r_squared' in threshold_0_df.columns else None,
                'Median R²': threshold_0_df['r_squared'].median() if 'r_squared' in threshold_0_df.columns else None,
                'Min R²': threshold_0_df['r_squared'].min() if 'r_squared' in threshold_0_df.columns else None,
                'Max R²': threshold_0_df['r_squared'].max() if 'r_squared' in threshold_0_df.columns else None
            })
            print(f"  Computed statistics for {len(threshold_0_df)} gages")

    # Process filtered thresholds
    for threshold in thresholds:
        if threshold == 0:
            continue  # Already processed above

        dir_name = dir_names[threshold]
        stats_dir = os.path.join(args.base_dir, dir_name, 'scatter_plots_by_gage')
        stats_file = os.path.join(stats_dir, 'gage_statistics.csv')

        if not os.path.exists(stats_file):
            print(f"\nWarning: File not found: {stats_file}")
            continue

        print(f"\nProcessing R² > {threshold}...")
        df = pd.read_csv(stats_file)

        # Add threshold column
        df.insert(0, 'r2_threshold', threshold)
        all_data.append(df)

        # Add to summary
        summary_data.append({
            'R² Threshold': f'>{threshold} (filtered)',
            'Number of Gages': len(df),
            'Total Wells': df['n_wells'].sum() if 'n_wells' in df.columns else None,
            'Total Observations': df['n_observations'].sum() if 'n_observations' in df.columns else None,
            'Mean Slope': df['slope'].mean() if 'slope' in df.columns else None,
            'Median Slope': df['slope'].median() if 'slope' in df.columns else None,
            'Mean R²': df['r_squared'].mean() if 'r_squared' in df.columns else None,
            'Median R²': df['r_squared'].median() if 'r_squared' in df.columns else None,
            'Min R²': df['r_squared'].min() if 'r_squared' in df.columns else None,
            'Max R²': df['r_squared'].max() if 'r_squared' in df.columns else None
        })

        print(f"  Loaded {len(df)} gages")

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Create combined DataFrame
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
    else:
        print("ERROR: No data found!")
        return

    print("\nCreating CSV files...")

    # Write summary
    summary_file = os.path.join(csv_dir, 'Summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"  Summary saved to: {summary_file}")

    # Write unfiltered well-gage pair data if available
    if unfiltered_stats is not None:
        unfiltered_file = os.path.join(csv_dir, 'All_Data_Unfiltered_Pairs.csv')
        unfiltered_stats.to_csv(unfiltered_file, index=False)
        print(f"  Unfiltered pairs data saved to: {unfiltered_file}")

    # Write individual threshold files
    for threshold in thresholds:
        if threshold == 0:
            # Threshold 0 data was computed above
            threshold_0_data = [df for df in all_data if 'r2_threshold' in df.columns and df['r2_threshold'].iloc[0] == 0]
            if threshold_0_data:
                df = threshold_0_data[0]
                csv_file = os.path.join(csv_dir, 'R2_gt_0.csv')
                df.to_csv(csv_file, index=False)
                print(f"  Saved to: {csv_file}")
        else:
            dir_name = dir_names[threshold]
            stats_dir = os.path.join(args.base_dir, dir_name, 'scatter_plots_by_gage')
            stats_file = os.path.join(stats_dir, 'gage_statistics.csv')

            if not os.path.exists(stats_file):
                continue

            df = pd.read_csv(stats_file)
            df.insert(0, 'r2_threshold', threshold)
            csv_file = os.path.join(csv_dir, f'R2_gt_{threshold}.csv')
            df.to_csv(csv_file, index=False)
            print(f"  Saved to: {csv_file}")

    # Write combined filtered data (excluding threshold=0)
    filtered_data = [df for df in all_data if 'r2_threshold' in df.columns and df['r2_threshold'].iloc[0] > 0]
    if filtered_data:
        combined_filtered_df = pd.concat(filtered_data, ignore_index=True)
        combined_file = os.path.join(csv_dir, 'All_Data_Filtered.csv')
        combined_filtered_df.to_csv(combined_file, index=False)
        print(f"  Combined filtered data saved to: {combined_file}")

    # Write all data (including threshold=0)
    combined_file = os.path.join(csv_dir, 'All_Data.csv')
    combined_df.to_csv(combined_file, index=False)
    print(f"  All data (including threshold=0) saved to: {combined_file}")

    print("\n" + "="*60)
    print("CSV files created successfully!")
    print("="*60)
    print(f"Output directory: {csv_dir}")
    print(f"\nFiles created:")
    print(f"  - Summary.csv: Overview statistics (including threshold=0)")
    if unfiltered_stats is not None:
        print(f"  - All_Data_Unfiltered_Pairs.csv: All well-gage pairs (no R² filter)")
    print(f"  - R2_gt_0.csv: Detailed gage statistics for all data (R² > 0)")
    print(f"  - R2_gt_0.1.csv: Detailed gage statistics for R² > 0.1")
    print(f"  - R2_gt_0.2.csv: Detailed gage statistics for R² > 0.2")
    print(f"  - R2_gt_0.3.csv: Detailed gage statistics for R² > 0.3")
    print(f"  - All_Data_Filtered.csv: Combined filtered data (R² > 0.1, 0.2, 0.3)")
    print(f"  - All_Data.csv: Combined data from all thresholds (including 0)")


if __name__ == '__main__':
    main()
