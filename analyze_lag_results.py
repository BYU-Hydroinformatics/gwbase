#!/usr/bin/env python3
"""
分析 features 文件夹中的 lag 结果，生成统计分析报告。

该脚本分析：
1. 不同 lag 期间的数据保留情况
2. 不同 lag 期间的相关性统计
3. lag vs no-lag 的比较
4. 按 gage 的详细分析
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_lag_data(features_dir):
    """加载所有 lag 相关的数据文件"""
    features_path = Path(features_dir)
    
    data = {}
    
    # 加载总体统计
    if (features_path / 'overall_lag_comparison_summary.csv').exists():
        data['overall_summary'] = pd.read_csv(features_path / 'overall_lag_comparison_summary.csv')
    
    # 加载按 gage 的统计
    if (features_path / 'gage_lag_comparison_summary.csv').exists():
        data['gage_summary'] = pd.read_csv(features_path / 'gage_lag_comparison_summary.csv')
    
    if (features_path / 'gage_lag_comparison_detailed.csv').exists():
        data['gage_detailed'] = pd.read_csv(features_path / 'gage_lag_comparison_detailed.csv')
    
    # 加载观测数据统计
    if (features_path / 'observations_by_gage_and_lag.csv').exists():
        data['observations'] = pd.read_csv(features_path / 'observations_by_gage_and_lag.csv')
    
    if (features_path / 'wells_by_gage_and_lag.csv').exists():
        data['wells'] = pd.read_csv(features_path / 'wells_by_gage_and_lag.csv')
    
    if (features_path / 'data_retention_by_gage_and_lag.csv').exists():
        data['retention'] = pd.read_csv(features_path / 'data_retention_by_gage_and_lag.csv')
    
    # 加载相关性统计
    if (features_path / 'correlation_summary_by_dataset.csv').exists():
        data['correlation_summary'] = pd.read_csv(features_path / 'correlation_summary_by_dataset.csv')
    
    if (features_path / 'correlation_stats_by_dataset.csv').exists():
        data['correlation_stats'] = pd.read_csv(features_path / 'correlation_stats_by_dataset.csv')
    
    return data


def analyze_overall_summary(data, output_dir):
    """分析总体统计摘要"""
    if 'overall_summary' not in data:
        return None
    
    df = data['overall_summary'].copy()
    
    print("\n" + "="*80)
    print("总体 Lag 分析摘要")
    print("="*80)
    
    # 重新格式化数据以便分析
    lag_periods = []
    for _, row in df.iterrows():
        lag_periods.append({
            'lag_period': row['lag_period'],
            'total_observations': row['total_observations'],
            'total_unique_wells': row['total_unique_wells'],
            'gages_with_data': row['gages_with_data'],
            'avg_observations_per_gage': row['avg_observations_per_gage'],
            'avg_wells_per_gage': row['avg_wells_per_gage'],
            'data_retention_pct': row.get('data_retention_pct', np.nan)
        })
    
    summary_df = pd.DataFrame(lag_periods)
    
    # 打印统计信息
    print("\n各 Lag 期间的数据统计:")
    print(summary_df.to_string(index=False))
    
    # 计算相对于 no_lag 的变化
    if 'no_lag' in summary_df['lag_period'].values:
        no_lag_row = summary_df[summary_df['lag_period'] == 'no_lag'].iloc[0]
        print("\n相对于 No-Lag 的变化:")
        print("-" * 80)
        for _, row in summary_df.iterrows():
            if row['lag_period'] != 'no_lag':
                obs_mult = row['total_observations'] / no_lag_row['total_observations'] if no_lag_row['total_observations'] > 0 else np.nan
                wells_mult = row['total_unique_wells'] / no_lag_row['total_unique_wells'] if no_lag_row['total_unique_wells'] > 0 else np.nan
                print(f"{row['lag_period']:15s}: "
                      f"观测数 = {obs_mult:.2f}x, "
                      f"井数 = {wells_mult:.2f}x")
    
    return summary_df


def analyze_gage_summary(data, output_dir):
    """分析按 gage 的统计"""
    if 'gage_summary' not in data:
        return None
    
    df = data['gage_summary'].copy()
    
    print("\n" + "="*80)
    print("按 Gage 的 Lag 分析")
    print("="*80)
    
    # 提取各 lag 期间的观测数和井数
    lag_periods = ['no_lag', '1_year_lag', '2_year_lag', '3_year_lag', 
                   '6_month_lag', '3_month_lag']
    
    gage_stats = []
    for _, row in df.iterrows():
        gage_id = row['gage_id']
        stats = {'gage_id': gage_id}
        
        for lag in lag_periods:
            obs_col = f'{lag}_observations'
            wells_col = f'{lag}_wells'
            
            if obs_col in row:
                stats[f'{lag}_obs'] = row[obs_col] if pd.notna(row[obs_col]) else 0
            else:
                stats[f'{lag}_obs'] = 0
            
            if wells_col in row:
                stats[f'{lag}_wells'] = row[wells_col] if pd.notna(row[wells_col]) else 0
            else:
                stats[f'{lag}_wells'] = 0
        
        gage_stats.append(stats)
    
    gage_df = pd.DataFrame(gage_stats)
    
    print(f"\n共有 {len(gage_df)} 个 gage")
    print("\n各 Gage 的观测数统计:")
    print(gage_df[['gage_id'] + [f'{lag}_obs' for lag in lag_periods]].to_string(index=False))
    
    return gage_df


def analyze_correlation_stats(data, output_dir):
    """分析相关性统计"""
    if 'correlation_stats' not in data:
        return None
    
    df = data['correlation_stats'].copy()
    
    print("\n" + "="*80)
    print("相关性统计分析")
    print("="*80)
    
    # 按 dataset 分组统计
    if 'dataset' in df.columns:
        print("\n各数据集的相关性统计:")
        print("-" * 80)
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            print(f"\n{dataset}:")
            print(f"  配对数量: {len(dataset_df)}")
            if 'correlation' in dataset_df.columns:
                print(f"  平均相关性: {dataset_df['correlation'].mean():.4f}")
                print(f"  中位数相关性: {dataset_df['correlation'].median():.4f}")
                print(f"  标准差: {dataset_df['correlation'].std():.4f}")
                print(f"  最小值: {dataset_df['correlation'].min():.4f}")
                print(f"  最大值: {dataset_df['correlation'].max():.4f}")
            if 'abs_correlation' in dataset_df.columns:
                print(f"  平均绝对相关性: {dataset_df['abs_correlation'].mean():.4f}")
                print(f"  中位数绝对相关性: {dataset_df['abs_correlation'].median():.4f}")
            if 'n_observations' in dataset_df.columns:
                print(f"  平均观测数: {dataset_df['n_observations'].mean():.1f}")
                print(f"  总观测数: {dataset_df['n_observations'].sum():,}")
    
    return df


def create_visualizations(data, output_dir):
    """创建可视化图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("生成可视化图表")
    print("="*80)
    
    # 1. 总体观测数和井数对比
    if 'overall_summary' in data:
        df = data['overall_summary'].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 观测数对比
        ax1 = axes[0, 0]
        lag_periods = df['lag_period'].values
        observations = df['total_observations'].values / 1e6  # 转换为百万
        colors = sns.color_palette("husl", len(lag_periods))
        bars = ax1.bar(lag_periods, observations, color=colors)
        ax1.set_ylabel('Total Observations (Millions)', fontsize=12)
        ax1.set_title('Total Observations by Lag Period', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        for i, (bar, val) in enumerate(zip(bars, observations)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}M', ha='center', va='bottom', fontsize=10)
        
        # Well count comparison
        ax2 = axes[0, 1]
        wells = df['total_unique_wells'].values
        bars = ax2.bar(lag_periods, wells, color=colors)
        ax2.set_ylabel('Total Unique Wells', fontsize=12)
        ax2.set_title('Total Wells by Lag Period', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        for i, (bar, val) in enumerate(zip(bars, wells)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(val)}', ha='center', va='bottom', fontsize=10)
        
        # Average observations per gage
        ax3 = axes[1, 0]
        avg_obs = df['avg_observations_per_gage'].values / 1e3  # Convert to thousands
        bars = ax3.bar(lag_periods, avg_obs, color=colors)
        ax3.set_ylabel('Average Observations per Gage (Thousands)', fontsize=12)
        ax3.set_title('Average Observations per Gage by Lag Period', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        for i, (bar, val) in enumerate(zip(bars, avg_obs)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}K', ha='center', va='bottom', fontsize=10)
        
        # Average wells per gage
        ax4 = axes[1, 1]
        avg_wells = df['avg_wells_per_gage'].values
        bars = ax4.bar(lag_periods, avg_wells, color=colors)
        ax4.set_ylabel('Average Wells per Gage', fontsize=12)
        ax4.set_title('Average Wells per Gage by Lag Period', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        for i, (bar, val) in enumerate(zip(bars, avg_wells)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / 'lag_overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: lag_overall_comparison.png")
    
    # 2. 相关性分布对比
    if 'correlation_stats' in data:
        df = data['correlation_stats'].copy()
        
        if 'dataset' in df.columns and 'correlation' in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            datasets = df['dataset'].unique()
            
            # 相关性分布箱线图
            ax1 = axes[0, 0]
            data_for_box = [df[df['dataset'] == d]['correlation'].values for d in datasets]
            bp = ax1.boxplot(data_for_box, labels=datasets, patch_artist=True)
            for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(datasets))):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax1.set_ylabel('Correlation Coefficient', fontsize=12)
            ax1.set_title('Correlation Distribution by Dataset', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Absolute correlation distribution boxplot
            if 'abs_correlation' in df.columns:
                ax2 = axes[0, 1]
                data_for_box = [df[df['dataset'] == d]['abs_correlation'].values for d in datasets]
                bp = ax2.boxplot(data_for_box, labels=datasets, patch_artist=True)
                for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(datasets))):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax2.set_ylabel('Absolute Correlation', fontsize=12)
                ax2.set_title('Absolute Correlation Distribution by Dataset', fontsize=14, fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
            
            # Correlation histogram
            ax3 = axes[1, 0]
            for dataset in datasets:
                dataset_df = df[df['dataset'] == dataset]
                ax3.hist(dataset_df['correlation'].values, bins=30, alpha=0.6, 
                        label=dataset, density=True)
            ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Correlation Coefficient', fontsize=12)
            ax3.set_ylabel('Density', fontsize=12)
            ax3.set_title('Correlation Distribution Histogram', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Observation count distribution
            if 'n_observations' in df.columns:
                ax4 = axes[1, 1]
                data_for_box = [df[df['dataset'] == d]['n_observations'].values for d in datasets]
                bp = ax4.boxplot(data_for_box, labels=datasets, patch_artist=True)
                for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(datasets))):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax4.set_ylabel('Number of Observations', fontsize=12)
                ax4.set_title('Observation Count Distribution by Dataset', fontsize=14, fontweight='bold')
                ax4.tick_params(axis='x', rotation=45)
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'lag_correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: lag_correlation_analysis.png")
    
    # 3. 按 gage 的观测数热力图
    if 'gage_summary' in data:
        df = data['gage_summary'].copy()
        
        lag_periods = ['no_lag', '1_year_lag', '2_year_lag', '3_year_lag', 
                       '6_month_lag', '3_month_lag']
        
        # 提取观测数数据
        obs_data = []
        for _, row in df.iterrows():
            obs_row = {'gage_id': row['gage_id']}
            for lag in lag_periods:
                col = f'{lag}_observations'
                if col in row:
                    val = row[col] if pd.notna(row[col]) else 0
                    obs_row[lag] = val / 1e3  # 转换为千
                else:
                    obs_row[lag] = 0
            obs_data.append(obs_row)
        
        obs_df = pd.DataFrame(obs_data)
        obs_df = obs_df.set_index('gage_id')
        
        if len(obs_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(obs_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                       cbar_kws={'label': '观测数 (千)'}, ax=ax)
            ax.set_title('Observations by Gage and Lag Period', fontsize=14, fontweight='bold')
            ax.set_xlabel('Lag Period', fontsize=12)
            ax.set_ylabel('Gage ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'lag_observations_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: lag_observations_heatmap.png")
    
    # 4. Lag period trend analysis
    if 'overall_summary' in data:
        df = data['overall_summary'].copy()
        
        # Create lag order for proper plotting
        lag_order = ['no_lag', '3_month_lag', '6_month_lag', '1_year_lag', '2_year_lag', '3_year_lag']
        df_ordered = df.set_index('lag_period').reindex([l for l in lag_order if l in df['lag_period'].values])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Trend: Observations vs Lag Period
        ax1 = axes[0, 0]
        x_pos = range(len(df_ordered))
        ax1.plot(x_pos, df_ordered['total_observations'].values / 1e6, 
                marker='o', linewidth=2, markersize=8, color='steelblue')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df_ordered.index, rotation=45, ha='right')
        ax1.set_ylabel('Total Observations (Millions)', fontsize=12)
        ax1.set_title('Observation Count Trend Across Lag Periods', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        for i, (x, y) in enumerate(zip(x_pos, df_ordered['total_observations'].values / 1e6)):
            ax1.text(x, y, f'{y:.2f}M', ha='center', va='bottom', fontsize=9)
        
        # Trend: Wells vs Lag Period
        ax2 = axes[0, 1]
        ax2.plot(x_pos, df_ordered['total_unique_wells'].values, 
                marker='s', linewidth=2, markersize=8, color='coral')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df_ordered.index, rotation=45, ha='right')
        ax2.set_ylabel('Total Unique Wells', fontsize=12)
        ax2.set_title('Well Count Trend Across Lag Periods', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        for i, (x, y) in enumerate(zip(x_pos, df_ordered['total_unique_wells'].values)):
            ax2.text(x, y, f'{int(y)}', ha='center', va='bottom', fontsize=9)
        
        # Data retention percentage
        ax3 = axes[1, 0]
        retention = df_ordered['data_retention_pct'].dropna()
        if len(retention) > 0:
            x_pos_ret = range(len(retention))
            bars = ax3.bar(x_pos_ret, retention.values, color='mediumseagreen', alpha=0.7)
            ax3.set_xticks(x_pos_ret)
            ax3.set_xticklabels(retention.index, rotation=45, ha='right')
            ax3.set_ylabel('Data Retention (%)', fontsize=12)
            ax3.set_title('Data Retention Percentage by Lag Period', fontsize=14, fontweight='bold')
            ax3.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% baseline')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            for i, (bar, val) in enumerate(zip(bars, retention.values)):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Average observations per well
        ax4 = axes[1, 1]
        if 'no_lag' in df_ordered.index:
            no_lag_obs = df_ordered.loc['no_lag', 'total_observations']
            no_lag_wells = df_ordered.loc['no_lag', 'total_unique_wells']
            if pd.notna(no_lag_obs) and pd.notna(no_lag_wells) and no_lag_wells > 0:
                no_lag_avg = no_lag_obs / no_lag_wells
                lag_periods_with_data = df_ordered[df_ordered['total_observations'].notna()].index
                avg_obs_per_well = []
                labels = []
                for lag in lag_periods_with_data:
                    if lag != 'no_lag':
                        obs = df_ordered.loc[lag, 'total_observations']
                        wells = df_ordered.loc[lag, 'total_unique_wells']
                        if pd.notna(obs) and pd.notna(wells) and wells > 0:
                            avg_obs_per_well.append(obs / wells)
                            labels.append(lag)
                
                if len(avg_obs_per_well) > 0:
                    x_pos_avg = range(len(avg_obs_per_well))
                    ax4.plot(x_pos_avg, avg_obs_per_well, marker='^', linewidth=2, 
                            markersize=8, color='purple', label='Lag periods')
                    ax4.axhline(y=no_lag_avg, color='r', linestyle='--', alpha=0.5, 
                              label=f'No-lag baseline ({no_lag_avg:.0f})')
                    ax4.set_xticks(x_pos_avg)
                    ax4.set_xticklabels(labels, rotation=45, ha='right')
                    ax4.set_ylabel('Average Observations per Well', fontsize=12)
                    ax4.set_title('Average Observations per Well by Lag Period', fontsize=14, fontweight='bold')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'lag_trend_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: lag_trend_analysis.png")
    
    # 5. Gage-level detailed comparison
    if 'gage_summary' in data and 'observations' in data:
        gage_df = data['gage_summary'].copy()
        obs_df = data['observations'].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Observations by gage (stacked bar)
        ax1 = axes[0, 0]
        lag_periods = ['no_lag', '3_month_lag', '6_month_lag', '1_year_lag', '2_year_lag', '3_year_lag']
        gage_ids = gage_df['gage_id'].astype(str).values
        
        bottom = np.zeros(len(gage_ids))
        colors_map = sns.color_palette("Set2", len(lag_periods))
        
        for i, lag in enumerate(lag_periods):
            col = f'{lag}_observations'
            if col in gage_df.columns:
                values = gage_df[col].fillna(0).values / 1e3  # Convert to thousands
                ax1.bar(gage_ids, values, bottom=bottom, label=lag.replace('_', ' ').title(), 
                       color=colors_map[i], alpha=0.8)
                bottom += values
        
        ax1.set_ylabel('Observations (Thousands)', fontsize=12)
        ax1.set_xlabel('Gage ID', fontsize=12)
        ax1.set_title('Observations Distribution by Gage and Lag Period', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Wells by gage (stacked bar)
        ax2 = axes[0, 1]
        bottom = np.zeros(len(gage_ids))
        
        for i, lag in enumerate(lag_periods):
            col = f'{lag}_wells'
            if col in gage_df.columns:
                values = gage_df[col].fillna(0).values
                ax2.bar(gage_ids, values, bottom=bottom, label=lag.replace('_', ' ').title(), 
                       color=colors_map[i], alpha=0.8)
                bottom += values
        
        ax2.set_ylabel('Number of Wells', fontsize=12)
        ax2.set_xlabel('Gage ID', fontsize=12)
        ax2.set_title('Well Count Distribution by Gage and Lag Period', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Relative increase compared to no-lag
        ax3 = axes[1, 0]
        if 'no_lag_observations' in gage_df.columns:
            no_lag_obs = gage_df['no_lag_observations'].fillna(0).values
            lag_cols = [f'{lag}_observations' for lag in lag_periods if lag != 'no_lag']
            
            x = np.arange(len(gage_ids))
            width = 0.15
            multiplier = 0
            
            for lag_col in lag_cols[:4]:  # Limit to 4 lag periods for clarity
                if lag_col in gage_df.columns:
                    lag_obs = gage_df[lag_col].fillna(0).values
                    # Calculate multiplier (how many times more than no-lag)
                    multiplier_vals = np.where(no_lag_obs > 0, lag_obs / no_lag_obs, np.nan)
                    offset = width * multiplier
                    bars = ax3.bar(x + offset, multiplier_vals, width, 
                                 label=lag_col.replace('_observations', '').replace('_', ' ').title(),
                                 alpha=0.8)
                    multiplier += 1
            
            ax3.set_ylabel('Multiplier (Relative to No-Lag)', fontsize=12)
            ax3.set_xlabel('Gage ID', fontsize=12)
            ax3.set_title('Observation Increase Multiplier by Gage', fontsize=14, fontweight='bold')
            ax3.set_xticks(x + width * 1.5)
            ax3.set_xticklabels(gage_ids, rotation=45, ha='right')
            ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='1x baseline')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Gage coverage (how many gages have data for each lag)
        ax4 = axes[1, 1]
        gage_coverage = []
        lag_labels = []
        
        for lag in lag_periods:
            if lag == 'no_lag':
                col = 'no_lag_observations'
            else:
                col = f'{lag}_observations'
            
            if col in gage_df.columns:
                coverage = (gage_df[col].notna() & (gage_df[col] > 0)).sum()
                gage_coverage.append(coverage)
                lag_labels.append(lag.replace('_', ' ').title())
        
        bars = ax4.bar(lag_labels, gage_coverage, color='teal', alpha=0.7)
        ax4.set_ylabel('Number of Gages with Data', fontsize=12)
        ax4.set_xlabel('Lag Period', fontsize=12)
        ax4.set_title('Gage Coverage by Lag Period', fontsize=14, fontweight='bold')
        ax4.set_ylim([0, max(gage_coverage) + 1])
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, gage_coverage):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(val)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / 'lag_gage_detailed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: lag_gage_detailed_comparison.png")
    
    # 6. Correlation improvement analysis
    if 'correlation_stats' in data:
        df = data['correlation_stats'].copy()
        
        if 'dataset' in df.columns and 'correlation' in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            datasets = df['dataset'].unique()
            
            # Violin plot for correlation distribution
            ax1 = axes[0, 0]
            data_for_violin = []
            labels_for_violin = []
            for d in datasets:
                dataset_df = df[df['dataset'] == d]
                if len(dataset_df) > 0:
                    data_for_violin.append(dataset_df['correlation'].values)
                    labels_for_violin.append(d)
            
            if len(data_for_violin) > 0:
                parts = ax1.violinplot(data_for_violin, positions=range(len(data_for_violin)), 
                                      showmeans=True, showmedians=True)
                ax1.set_xticks(range(len(labels_for_violin)))
                ax1.set_xticklabels(labels_for_violin, rotation=45, ha='right')
                ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax1.set_ylabel('Correlation Coefficient', fontsize=12)
                ax1.set_title('Correlation Distribution (Violin Plot)', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # Color the violins
                for pc, color in zip(parts['bodies'], sns.color_palette("husl", len(data_for_violin))):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
            
            # Absolute correlation comparison
            if 'abs_correlation' in df.columns:
                ax2 = axes[0, 1]
                abs_corr_data = []
                abs_labels = []
                for d in datasets:
                    dataset_df = df[df['dataset'] == d]
                    if len(dataset_df) > 0:
                        abs_corr_data.append(dataset_df['abs_correlation'].values)
                        abs_labels.append(d)
                
                if len(abs_corr_data) > 0:
                    bp = ax2.boxplot(abs_corr_data, labels=abs_labels, patch_artist=True)
                    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(abs_corr_data))):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    ax2.set_ylabel('Absolute Correlation', fontsize=12)
                    ax2.set_title('Absolute Correlation Comparison', fontsize=14, fontweight='bold')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3)
            
            # Correlation vs observations scatter
            if 'n_observations' in df.columns:
                ax3 = axes[1, 0]
                for d in datasets:
                    dataset_df = df[df['dataset'] == d]
                    if len(dataset_df) > 0:
                        ax3.scatter(dataset_df['n_observations'], dataset_df['abs_correlation'], 
                                  alpha=0.6, s=50, label=d)
                
                ax3.set_xlabel('Number of Observations', fontsize=12)
                ax3.set_ylabel('Absolute Correlation', fontsize=12)
                ax3.set_title('Correlation vs Sample Size', fontsize=14, fontweight='bold')
                ax3.set_xscale('log')
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3)
            
            # Cumulative distribution of correlations
            ax4 = axes[1, 1]
            for d in datasets:
                dataset_df = df[df['dataset'] == d]
                if len(dataset_df) > 0 and 'abs_correlation' in dataset_df.columns:
                    sorted_corr = np.sort(dataset_df['abs_correlation'].values)
                    y_vals = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
                    ax4.plot(sorted_corr, y_vals, label=d, linewidth=2, alpha=0.8)
            
            ax4.set_xlabel('Absolute Correlation', fontsize=12)
            ax4.set_ylabel('Cumulative Probability', fontsize=12)
            ax4.set_title('Cumulative Distribution of Absolute Correlations', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'lag_correlation_improvement.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: lag_correlation_improvement.png")
    
    # 7. Summary statistics visualization
    if 'overall_summary' in data:
        df = data['overall_summary'].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Radar chart style comparison (using bar chart instead)
        ax1 = axes[0]
        lag_periods = df['lag_period'].values
        metrics = ['total_observations', 'total_unique_wells', 'avg_observations_per_gage', 'avg_wells_per_gage']
        metric_labels = ['Total Obs\n(M)', 'Total Wells', 'Avg Obs/Gage\n(K)', 'Avg Wells/Gage']
        
        # Normalize each metric to 0-1 scale for comparison
        normalized_data = []
        for metric in metrics:
            values = df[metric].fillna(0).values
            if len(values) > 0 and max(values) > 0:
                normalized = values / max(values)
                normalized_data.append(normalized)
        
        if len(normalized_data) > 0:
            x = np.arange(len(lag_periods))
            width = 0.2
            for i, (norm_vals, label) in enumerate(zip(normalized_data, metric_labels)):
                offset = width * (i - len(metrics)/2 + 0.5)
                ax1.bar(x + offset, norm_vals, width, label=label, alpha=0.8)
            
            ax1.set_ylabel('Normalized Value', fontsize=12)
            ax1.set_xlabel('Lag Period', fontsize=12)
            ax1.set_title('Normalized Metrics Comparison Across Lag Periods', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(lag_periods, rotation=45, ha='right')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Efficiency metric: observations per well
        ax2 = axes[1]
        obs_per_well = []
        lag_labels = []
        
        for _, row in df.iterrows():
            obs = row['total_observations']
            wells = row['total_unique_wells']
            lag = row['lag_period']
            if pd.notna(obs) and pd.notna(wells) and wells > 0:
                obs_per_well.append(obs / wells)
                lag_labels.append(lag)
        
        if len(obs_per_well) > 0:
            bars = ax2.bar(lag_labels, obs_per_well, color='darkorange', alpha=0.7)
            ax2.set_ylabel('Observations per Well', fontsize=12)
            ax2.set_xlabel('Lag Period', fontsize=12)
            ax2.set_title('Data Efficiency: Observations per Well', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars, obs_per_well):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'lag_summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: lag_summary_statistics.png")
    
    print("\nAll visualizations generated successfully!")


def generate_summary_report(data, output_dir):
    """生成汇总报告"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / 'lag_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Lag 分析统计报告\n")
        f.write("="*80 + "\n\n")
        
        # 总体统计
        if 'overall_summary' in data:
            f.write("1. 总体统计摘要\n")
            f.write("-"*80 + "\n")
            df = data['overall_summary']
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # 计算相对于 no_lag 的变化
            if 'no_lag' in df['lag_period'].values:
                no_lag_row = df[df['lag_period'] == 'no_lag'].iloc[0]
                f.write("相对于 No-Lag 的变化:\n")
                for _, row in df.iterrows():
                    if row['lag_period'] != 'no_lag':
                        obs_mult = row['total_observations'] / no_lag_row['total_observations'] if no_lag_row['total_observations'] > 0 else np.nan
                        wells_mult = row['total_unique_wells'] / no_lag_row['total_unique_wells'] if no_lag_row['total_unique_wells'] > 0 else np.nan
                        f.write(f"  {row['lag_period']:15s}: 观测数 = {obs_mult:.2f}x, 井数 = {wells_mult:.2f}x\n")
                f.write("\n")
        
        # 相关性统计
        if 'correlation_stats' in data:
            f.write("2. 相关性统计分析\n")
            f.write("-"*80 + "\n")
            df = data['correlation_stats']
            if 'dataset' in df.columns:
                for dataset in df['dataset'].unique():
                    dataset_df = df[df['dataset'] == dataset]
                    f.write(f"\n{dataset}:\n")
                    f.write(f"  配对数量: {len(dataset_df)}\n")
                    if 'correlation' in dataset_df.columns:
                        f.write(f"  平均相关性: {dataset_df['correlation'].mean():.4f}\n")
                        f.write(f"  中位数相关性: {dataset_df['correlation'].median():.4f}\n")
                        f.write(f"  标准差: {dataset_df['correlation'].std():.4f}\n")
                        f.write(f"  最小值: {dataset_df['correlation'].min():.4f}\n")
                        f.write(f"  最大值: {dataset_df['correlation'].max():.4f}\n")
                    if 'abs_correlation' in dataset_df.columns:
                        f.write(f"  平均绝对相关性: {dataset_df['abs_correlation'].mean():.4f}\n")
                        f.write(f"  中位数绝对相关性: {dataset_df['abs_correlation'].median():.4f}\n")
                    if 'n_observations' in dataset_df.columns:
                        f.write(f"  平均观测数: {dataset_df['n_observations'].mean():.1f}\n")
                        f.write(f"  总观测数: {dataset_df['n_observations'].sum():,}\n")
            f.write("\n")
        
        # 按 gage 的统计
        if 'gage_summary' in data:
            f.write("3. 按 Gage 的统计\n")
            f.write("-"*80 + "\n")
            df = data['gage_summary']
            f.write(f"共有 {len(df)} 个 gage\n\n")
            f.write("各 Gage 的观测数:\n")
            lag_periods = ['no_lag', '1_year_lag', '2_year_lag', '3_year_lag', 
                          '6_month_lag', '3_month_lag']
            obs_cols = [f'{lag}_observations' for lag in lag_periods if f'{lag}_observations' in df.columns]
            if obs_cols:
                f.write(df[['gage_id'] + obs_cols].to_string(index=False))
            f.write("\n\n")
    
    print(f"\n报告已保存至: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='分析 features 文件夹中的 lag 结果'
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default='data/features',
        help='Features 文件夹路径 (默认: data/features)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/lag_analysis',
        help='输出目录 (默认: reports/lag_analysis)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Lag 结果统计分析")
    print("="*80)
    print(f"Features 目录: {args.features_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 加载数据
    print("\n加载数据...")
    data = load_lag_data(args.features_dir)
    print(f"  已加载 {len(data)} 个数据文件")
    
    # 执行分析
    overall_df = analyze_overall_summary(data, args.output_dir)
    gage_df = analyze_gage_summary(data, args.output_dir)
    corr_df = analyze_correlation_stats(data, args.output_dir)
    
    # 生成可视化
    create_visualizations(data, args.output_dir)
    
    # 生成报告
    generate_summary_report(data, args.output_dir)
    
    # 保存处理后的数据
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if overall_df is not None:
        overall_df.to_csv(output_path / 'processed_overall_summary.csv', index=False)
    if gage_df is not None:
        gage_df.to_csv(output_path / 'processed_gage_summary.csv', index=False)
    if corr_df is not None:
        corr_df.to_csv(output_path / 'processed_correlation_stats.csv', index=False)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
