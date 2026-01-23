"""
Data preprocessing functions for GWBASE.

This module implements Step 4 of the GWBASE workflow:
- Outlier detection using Z-score and IQR methods
- Filtering wells with insufficient data
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class SimpleOutlierDetector:
    """
    Simplified outlier detection for groundwater data.

    Uses combined Z-score and IQR methods to identify and remove
    outliers before interpolation.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing well measurements
    column : str
        Column name to check for outliers (e.g., 'wte' or 'q')

    Example
    -------
    >>> detector = SimpleOutlierDetector(data, 'q')
    >>> results = detector.detect_outliers()
    >>> clean_data = detector.get_clean_data()
    """

    DATE_COLUMN = 'date'
    WELL_ID_COLUMN = 'well_id'

    def __init__(self, data: pd.DataFrame, column: str):
        self.data = data.copy()
        self.column = column
        self._validate_columns()
        self.data[self.DATE_COLUMN] = pd.to_datetime(
            self.data[self.DATE_COLUMN], errors='coerce'
        )
        self.results = None

    def _validate_columns(self):
        """Validate that required columns exist in the data."""
        required_columns = [self.DATE_COLUMN, self.WELL_ID_COLUMN]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def detect_outliers(
        self,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect outliers using Z-score and IQR methods.

        Parameters
        ----------
        zscore_threshold : float, default 3.0
            Z-score threshold for outlier detection
        iqr_multiplier : float, default 1.5
            IQR multiplier for outlier detection

        Returns
        -------
        pd.DataFrame
            Data with outlier flags added
        """
        data = self.data.copy()

        # Initialize outlier flag columns
        data['is_outlier_zscore'] = False
        data['is_outlier_iqr'] = False

        # Z-score method
        try:
            z_scores = np.abs(stats.zscore(data[self.column], nan_policy='omit'))
            data['is_outlier_zscore'] = z_scores > zscore_threshold
        except Exception:
            pass

        # IQR method
        try:
            Q1 = np.nanpercentile(data[self.column], 25)
            Q3 = np.nanpercentile(data[self.column], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            data['is_outlier_iqr'] = (
                (data[self.column] < lower_bound) |
                (data[self.column] > upper_bound)
            )
        except Exception:
            pass

        # Combined outlier detection
        data['is_outlier_any'] = data[['is_outlier_zscore', 'is_outlier_iqr']].any(axis=1)

        self.results = data
        return self.results

    def get_clean_data(self) -> pd.DataFrame:
        """
        Get clean data with outliers removed.

        Returns
        -------
        pd.DataFrame
            Data with outliers removed
        """
        if self.results is None:
            raise ValueError("Please run detect_outliers() method first")
        return self.results[~self.results['is_outlier_any']].copy()


class GroundwaterOutlierDetector(SimpleOutlierDetector):
    """
    Specialized outlier detector for groundwater data.

    Processes data by well and provides interpolation readiness statistics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing groundwater measurements with
        date, well_id, and wte columns
    """

    def __init__(self, data: pd.DataFrame):
        super().__init__(data, 'wte')

    def detect_outliers(
        self,
        min_points: int = 5,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 2.0
    ) -> pd.DataFrame:
        """
        Detect outliers in groundwater data by well.

        Parameters
        ----------
        min_points : int, default 5
            Minimum points needed for statistical tests
        zscore_threshold : float, default 3.0
            Z-score threshold for outlier detection
        iqr_multiplier : float, default 2.0
            IQR multiplier for outlier detection

        Returns
        -------
        pd.DataFrame
            Data with outlier flags added
        """
        # Sort by well_id and date
        self.data = self.data.sort_values(
            [self.WELL_ID_COLUMN, self.DATE_COLUMN]
        ).reset_index(drop=True)

        results = []

        for well_id in self.data['well_id'].unique():
            well_data = self.data[self.data['well_id'] == well_id].copy()
            n_points = len(well_data)

            well_data['is_outlier'] = False

            if n_points >= min_points:
                wte_values = well_data['wte'].values

                # Z-score method
                z_scores = np.abs(stats.zscore(wte_values, nan_policy='omit'))
                is_zscore_outlier = z_scores > zscore_threshold

                # IQR method
                Q1, Q3 = np.nanpercentile(wte_values, [25, 75])
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    is_iqr_outlier = (
                        (wte_values < lower_bound) |
                        (wte_values > upper_bound)
                    )

                    # Combine methods
                    well_data['is_outlier'] = is_zscore_outlier | is_iqr_outlier

            results.append(well_data)

        if results:
            self.results = pd.concat(results, ignore_index=True)

        return self.results

    def get_clean_data(self) -> pd.DataFrame:
        """
        Get clean data suitable for interpolation.

        Returns
        -------
        pd.DataFrame
            Clean data with outliers removed and interpolation readiness stats
        """
        if self.results is None:
            return None

        clean_data = self.results[~self.results['is_outlier']].copy()

        # Print interpolation readiness stats
        well_stats = clean_data.groupby('well_id').size()
        print(f"\nInterpolation readiness summary:")
        print(f"- Wells with no data: {(well_stats == 0).sum()}")
        print(f"- Wells with 1-2 points: {((well_stats >= 1) & (well_stats <= 2)).sum()}")
        print(f"- Wells with 3+ points: {(well_stats >= 3).sum()} (suitable for PCHIP)")

        return clean_data


def detect_outliers(
    data: pd.DataFrame,
    column: str,
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers in a DataFrame column.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    column : str
        Column to check for outliers
    zscore_threshold : float, default 3.0
        Z-score threshold
    iqr_multiplier : float, default 1.5
        IQR multiplier

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier flags

    Example
    -------
    >>> data_with_flags = detect_outliers(streamflow_data, 'q')
    """
    result = data.copy()

    # Initialize flags
    result['is_outlier_zscore'] = False
    result['is_outlier_iqr'] = False

    # Z-score method
    z_scores = np.abs(stats.zscore(result[column], nan_policy='omit'))
    result['is_outlier_zscore'] = z_scores > zscore_threshold

    # IQR method
    Q1 = np.nanpercentile(result[column], 25)
    Q3 = np.nanpercentile(result[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    result['is_outlier_iqr'] = (
        (result[column] < lower_bound) |
        (result[column] > upper_bound)
    )

    # Combined flags
    result['is_outlier_any'] = result[['is_outlier_zscore', 'is_outlier_iqr']].any(axis=1)
    result['is_outlier_both'] = result[['is_outlier_zscore', 'is_outlier_iqr']].all(axis=1)

    # Print summary
    print(f"Outlier detection for '{column}':")
    print(f"  Total records: {len(result)}")
    print(f"  Z-score outliers: {result['is_outlier_zscore'].sum()}")
    print(f"  IQR outliers: {result['is_outlier_iqr'].sum()}")
    print(f"  Both methods: {result['is_outlier_both'].sum()}")

    return result


def filter_wells_by_data_quality(
    data: pd.DataFrame,
    min_measurements: int = 2,
    min_time_span_days: int = 365
) -> pd.DataFrame:
    """
    Filter wells based on data quality criteria.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with well_id, date, and measurement columns
    min_measurements : int, default 2
        Minimum number of measurements required
    min_time_span_days : int, default 365
        Minimum time span of measurements in days

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with wells meeting quality criteria

    Example
    -------
    >>> filtered = filter_wells_by_data_quality(well_data, min_measurements=5)
    """
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])

    # Calculate statistics per well
    well_stats = data.groupby('well_id').agg({
        'date': ['count', 'min', 'max']
    })
    well_stats.columns = ['n_measurements', 'first_date', 'last_date']
    well_stats['time_span_days'] = (
        well_stats['last_date'] - well_stats['first_date']
    ).dt.days

    # Apply filters
    valid_wells = well_stats[
        (well_stats['n_measurements'] >= min_measurements) &
        (well_stats['time_span_days'] >= min_time_span_days)
    ].index

    filtered_data = data[data['well_id'].isin(valid_wells)]

    print(f"Data quality filtering:")
    print(f"  Original wells: {data['well_id'].nunique()}")
    print(f"  Wells meeting criteria: {len(valid_wells)}")
    print(f"  Original records: {len(data)}")
    print(f"  Remaining records: {len(filtered_data)}")

    return filtered_data


def clean_well_data_for_interpolation(
    well_ts: pd.DataFrame,
    min_points: int = 5
) -> pd.DataFrame:
    """
    Clean groundwater data for interpolation.

    Convenience function that combines outlier detection and filtering.

    Parameters
    ----------
    well_ts : pd.DataFrame
        Groundwater time series data
    min_points : int, default 5
        Minimum points per well

    Returns
    -------
    pd.DataFrame
        Clean data ready for interpolation
    """
    detector = GroundwaterOutlierDetector(well_ts)
    detector.detect_outliers(min_points=min_points)
    return detector.get_clean_data()
