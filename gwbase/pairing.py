"""
Well-gage pairing functions for GWBASE.

This module implements Step 7 of the GWBASE workflow:
- Pair groundwater and streamflow records under baseflow-dominated conditions
- Filter data to BFD periods
"""

import pandas as pd


def pair_wells_with_streamflow(
    well_data: pd.DataFrame,
    streamflow_data: pd.DataFrame,
    bfd_classification: pd.DataFrame,
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Pair well time series with streamflow data under baseflow-dominated conditions.

    Parameters
    ----------
    well_data : pd.DataFrame
        Well time series with well_id, gage_id, date, wte columns
    streamflow_data : pd.DataFrame
        Streamflow data with gage_id, date, q columns
    bfd_classification : pd.DataFrame
        BFD classification with gage_id, date, bfd columns
    well_id_col : str, default 'well_id'
        Column name for well ID
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    date_col : str, default 'date'
        Column name for date

    Returns
    -------
    pd.DataFrame
        Paired data with well_id, gage_id, date, wte, q, bfd columns

    Example
    -------
    >>> paired = pair_wells_with_streamflow(well_data, streamflow, bfd_class)
    """
    # Ensure date columns are datetime
    well_data = well_data.copy()
    streamflow_data = streamflow_data.copy()
    bfd_classification = bfd_classification.copy()

    well_data[date_col] = pd.to_datetime(well_data[date_col])
    streamflow_data[date_col] = pd.to_datetime(streamflow_data[date_col])
    bfd_classification[date_col] = pd.to_datetime(bfd_classification[date_col])

    # Merge well data with streamflow
    paired = pd.merge(
        well_data,
        streamflow_data[[gage_id_col, date_col, 'q']],
        on=[gage_id_col, date_col],
        how='inner'
    )

    # Merge with BFD classification
    paired = pd.merge(
        paired,
        bfd_classification[[gage_id_col, date_col, 'bfd']],
        on=[gage_id_col, date_col],
        how='left'
    )

    # Fill missing BFD values with 0 (non-BFD)
    paired['bfd'] = paired['bfd'].fillna(0).astype(int)

    print(f"Paired data summary:")
    print(f"  Total records: {len(paired):,}")
    print(f"  Unique wells: {paired[well_id_col].nunique()}")
    print(f"  Unique gages: {paired[gage_id_col].nunique()}")
    print(f"  BFD records: {(paired['bfd'] == 1).sum():,} ({(paired['bfd'] == 1).mean()*100:.1f}%)")

    return paired


def filter_to_bfd_periods(
    paired_data: pd.DataFrame,
    bfd_col: str = 'bfd'
) -> pd.DataFrame:
    """
    Filter paired data to only baseflow-dominated periods.

    Parameters
    ----------
    paired_data : pd.DataFrame
        Paired well-streamflow data with bfd column
    bfd_col : str, default 'bfd'
        Column name for BFD indicator

    Returns
    -------
    pd.DataFrame
        Data filtered to BFD periods only

    Example
    -------
    >>> bfd_data = filter_to_bfd_periods(paired_data)
    """
    bfd_data = paired_data[paired_data[bfd_col] == 1].copy()

    print(f"Filtered to BFD periods:")
    print(f"  Input records: {len(paired_data):,}")
    print(f"  BFD records: {len(bfd_data):,}")
    print(f"  Retention rate: {len(bfd_data)/len(paired_data)*100:.1f}%")

    return bfd_data


def calculate_baseline_values(
    paired_data: pd.DataFrame,
    well_id_col: str = 'well_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    q_col: str = 'q',
    bfd_col: str = 'bfd'
) -> pd.DataFrame:
    """
    Calculate baseline (WTE0, Q0) values from first BFD=1 date per well.

    The baseline represents the initial reference point for computing
    changes in water table elevation and streamflow.

    Parameters
    ----------
    paired_data : pd.DataFrame
        Paired well-streamflow data
    well_id_col : str, default 'well_id'
        Column name for well ID
    date_col : str, default 'date'
        Column name for date
    wte_col : str, default 'wte'
        Column name for water table elevation
    q_col : str, default 'q'
        Column name for streamflow
    bfd_col : str, default 'bfd'
        Column name for BFD indicator

    Returns
    -------
    pd.DataFrame
        Data with wte0 and q0 columns added

    Example
    -------
    >>> data_with_baseline = calculate_baseline_values(paired_data)
    """
    data = paired_data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col)

    # Find first BFD=1 occurrence for each well
    first_bfd = data[data[bfd_col] == 1].groupby(well_id_col).first().reset_index()

    # Create mappings for baseline values
    wte0_map = first_bfd.set_index(well_id_col)[wte_col].to_dict()
    q0_map = first_bfd.set_index(well_id_col)[q_col].to_dict()

    # Apply baseline values
    data['wte0'] = data[well_id_col].map(wte0_map)
    data['q0'] = data[well_id_col].map(q0_map)

    # Count wells with valid baselines
    valid_wells = data['wte0'].notna().groupby(data[well_id_col]).any().sum()

    print(f"Baseline calculation:")
    print(f"  Wells with valid baseline: {valid_wells}")
    print(f"  Wells without BFD=1 data: {data[well_id_col].nunique() - valid_wells}")

    return data


def apply_date_range_filter(
    data: pd.DataFrame,
    start_date: str = None,
    end_date: str = None,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Filter data to a specific date range.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with date column
    start_date : str, optional
        Start date (YYYY-MM-DD format)
    end_date : str, optional
        End date (YYYY-MM-DD format)
    date_col : str, default 'date'
        Column name for date

    Returns
    -------
    pd.DataFrame
        Filtered data

    Example
    -------
    >>> filtered = apply_date_range_filter(data, '1990-01-01', '2020-12-31')
    """
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    if start_date:
        data = data[data[date_col] >= pd.to_datetime(start_date)]

    if end_date:
        data = data[data[date_col] <= pd.to_datetime(end_date)]

    print(f"Date range filter applied:")
    if start_date:
        print(f"  Start: {start_date}")
    if end_date:
        print(f"  End: {end_date}")
    print(f"  Records retained: {len(data):,}")

    return data


def get_well_gage_summary(
    paired_data: pd.DataFrame,
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Generate summary statistics for well-gage pairs.

    Parameters
    ----------
    paired_data : pd.DataFrame
        Paired well-streamflow data
    well_id_col : str, default 'well_id'
        Column name for well ID
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    date_col : str, default 'date'
        Column name for date

    Returns
    -------
    pd.DataFrame
        Summary statistics per well-gage pair

    Example
    -------
    >>> summary = get_well_gage_summary(paired_data)
    """
    data = paired_data.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    summary = data.groupby([well_id_col, gage_id_col]).agg({
        date_col: ['min', 'max', 'count'],
        'wte': ['mean', 'std', 'min', 'max'],
        'q': ['mean', 'std', 'min', 'max'],
        'bfd': 'sum'
    })

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Rename columns for clarity
    summary = summary.rename(columns={
        f'{date_col}_min': 'first_date',
        f'{date_col}_max': 'last_date',
        f'{date_col}_count': 'n_records',
        'bfd_sum': 'n_bfd_records'
    })

    # Calculate date span in days
    summary['date_span_days'] = (
        summary['last_date'] - summary['first_date']
    ).dt.days

    print(f"Well-gage pair summary:")
    print(f"  Total pairs: {len(summary)}")
    print(f"  Average records per pair: {summary['n_records'].mean():.1f}")
    print(f"  Average BFD records per pair: {summary['n_bfd_records'].mean():.1f}")

    return summary
