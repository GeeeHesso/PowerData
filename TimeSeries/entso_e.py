import pandas as pd
import numpy as np
import scipy as sp
import calendar
from tqdm import trange


def read_time_series(data_source: str, year: int, data_type: str,
                     column_name: str, data_filter: dict[str, str]) -> pd.DataFrame:
    """Reads a time series from 12 monthly CSV files of a given year and combine it into a single DataFrame.
    The data filter is a dictionary, e.g. {'MapCode': 'DE'} to select only entries belonging to Germany."""
    dataframes = []
    for month in trange(1, 13):
        data = pd.read_csv('%s/%d_%02d_%s.csv' % (data_source, year, month, data_type), sep='\t',
                           usecols=['DateTime', column_name] + list(data_filter.keys()),
                           parse_dates=['DateTime'])
        # apply filter
        for filter_key, filter_val in data_filter.items():
            data = data[data[filter_key] == filter_val]
        # NaN values are set to zero
        data = data.fillna({column_name: 0.})
        dataframes.append(data[['DateTime', column_name]])
    return pd.concat(dataframes, ignore_index=True).sort_values('DateTime')


def read_multiple_time_series(data_source: str, year: int, data_type: str, group_by: str,
                              column_name: str, data_filter: dict[str, str]) -> dict[hash, pd.DataFrame]:
    """Reads mulitple time series from 12 monthly CSV files of a given year and combine them into DataFrames.
    The data filter is a dictionary, e.g. {'MapCode': 'DE'} to select only entries belonging to Germany."""
    dataframes = {}
    for month in trange(1, 13):
        data = pd.read_csv('%s/%d_%02d_%s.csv' % (data_source, year, month, data_type), sep='\t',
                           usecols=['DateTime', group_by, column_name] + list(data_filter.keys()),
                           parse_dates=['DateTime'])
        # apply filter
        for filter_key, filter_val in data_filter.items():
            data = data[data[filter_key] == filter_val]
        # NaN values are set to zero
        data = data.fillna({column_name: 0.})
        for label, grouped_data in data.groupby(group_by):
            if label not in dataframes:
                dataframes[label] = []
            dataframes[label].append(grouped_data[['DateTime', column_name]])
    return {label: pd.concat(df_list, ignore_index=True).sort_values('DateTime')
            for label, df_list in dataframes.items()}


def smoothen_time_series(time_series: pd.Series, threshold: float = 30.0, filter_window: int = 101) -> np.ndarray:
    """Remove spikes in a time series by applying a median filter.
    The original value are preserved whenever they do not differ significantly from the filtered value."""
    filtered_time_series = sp.ndimage.median_filter(time_series, filter_window, mode='wrap')
    return np.where(np.abs(filtered_time_series - time_series) < threshold, time_series, filtered_time_series)


def interpolate_time_series(data: pd.DataFrame, column_name: str, year: int, daily_steps: int = 24) -> np.ndarray:
    """Returns a time series with a given number of daily steps by linear interpolation
    from a dataframe containing time stamps and values in a given column."""
    T = daily_steps * (366 if calendar.isleap(year) else 365)
    jan1st = pd.Timestamp('%d-01-01 00:00' % year)
    t = data['DateTime'].apply(lambda date: (date - jan1st) / pd.Timedelta(days=1/daily_steps))
    return np.interp(range(T), t, data[column_name], period=T)


def extract_time_series(data_source: str, year: int | list[int], data_type: str,
                        column_name: str, data_filter: dict[str, str], daily_steps: int = 24,
                        smoothen: bool = False, smoothen_threshold: float = 30.0,
                        smoothen_window: int = 101) -> np.ndarray | list[np.ndarray]:
    """Extracts a yearly time series of a given type for a given year.
    The data files must be stored in the folder 'data_source'.
    The data type and column name must match the entso-e format.
    The data filter is a dictionary, e.g. {'MapCode': 'DE'} to select only entries belonging to Germany.
    The time series is returned as a list. Missing value are set to 'None'."""
    if isinstance(year, list):
        return [extract_time_series(data_source, y, data_type, column_name, data_filter, daily_steps) for y in year]
    dataframe = read_time_series(data_source, year, data_type, column_name, data_filter)
    if smoothen:
        dataframe[column_name] = smoothen_time_series(dataframe[column_name], smoothen_threshold, smoothen_window)
    return interpolate_time_series(dataframe, column_name, year, daily_steps)


def extract_multiple_time_series(data_source: str, year: int, data_type: str, group_by: str, column_name: str,
                                 data_filter: dict[str, str], daily_steps: int = 24,
                                 smoothen: bool = False, smoothen_threshold: float = 30.0,
                                 smoothen_window: int = 101) -> dict[hash, np.ndarray]:
    """Extracts multiple yearly time series of a given type for a given year.
    The data files must be stored in the folder 'data_source'.
    The data type, group by, and column name must match the entso-e format.
    The data filter is a dictionary, e.g. {'MapCode': 'DE'} to select only entries belonging to Germany.
    The output is a dictionary of time series, each stored as a list with missing values set to 'None'."""
    dataframes = read_multiple_time_series(data_source, year, data_type, group_by, column_name, data_filter)
    if smoothen:
        for df in dataframes.values():
            df[column_name] = smoothen_time_series(df[column_name], smoothen_threshold, smoothen_window)
    return {label: interpolate_time_series(df, column_name, year, daily_steps) for label, df in dataframes.items()}


def extract_load_time_series(data_source: str, country_code: str, year: int | list,
                             daily_steps: int = 24) -> np.ndarray | list[np.ndarray]:
    """Extracts a yearly time series of the total load for a given country and a given year.
    The data files must be stored in the folder 'data_source'.
    'country_code' is 'DE' for Germany, 'FR' for France, and so on.
    The time series is returned as a list. Missing value are set to 'None'."""
    country_filter = {'AreaTypeCode': 'CTY', 'MapCode': country_code}
    return extract_time_series(data_source, year, 'ActualTotalLoad_6.1.A', 'TotalLoadValue',
                               country_filter, daily_steps)


def extract_border_flow_time_series(data_source: str, from_country: str, to_country: str,
                                    year: int | list, daily_steps: int = 24) -> np.ndarray | list[np.ndarray]:
    """Extracts a yearly time series of the border flow between two countries for a given year.
    The data files must be stored in the folder 'data_source'.
    'from_country' and 'to_country' are of the form 'DE' for Germany, 'FR' for France, and so on.
    The time series is returned as a list. Missing value are set to 'None'."""
    country_filter = {'OutAreaName': from_country + ' CTY', 'InAreaName': to_country + ' CTY'}
    return extract_time_series(data_source, year, 'PhysicalFlows_12.1.G', 'FlowValue',
                               country_filter, daily_steps)


def extract_production_by_type_time_series(data_source: str, gen_type: str, country_code: str,
                                           year: int | list, daily_steps: int = 24) -> np.ndarray | list[np.ndarray]:
    """Extracts a yearly time series of the total production of a given type, for a given country and a given year.
    The data files must be stored in the folder 'data_source'.
    'gen_type' is of the form 'Nuclear', 'Hydro Pumped Storage', 'Hydro Run-of-river and poundage', and so on.
    'country_code' is 'DE' for Germany, 'FR' for France, and so on.
    The time series is returned as a list. Missing values are set to 'None'."""
    data_filter = {'AreaTypeCode': 'CTY', 'MapCode': country_code, 'ProductionType': gen_type}
    return extract_time_series(data_source, year, 'AggregatedGenerationPerType_16.1.B_C',
                               'ActualGenerationOutput', data_filter, daily_steps)


def extract_production_by_unit_time_series(data_source: str, gen_type: str, year: int | list,
                                           daily_steps: int = 24) -> dict[hash, np.ndarray]:
    """Extracts yearly time series of the production of each individual generator of a given type for a given year.
    The data files must be stored in the folder 'data_source'.
    'gen_type' is of the form 'Nuclear', 'Hydro Pumped Storage', 'Hydro Run-of-river and poundage', and so on.
    The output is a dictionary of time series, each stored as a list with missing values set to 'None'."""
    return extract_multiple_time_series(data_source, year, 'ActualGenerationOutputPerGenerationUnit_16.1.A',
                                        'PowerSystemResourceName', 'ActualGenerationOutput',
                                        {'ProductionType': gen_type}, daily_steps, smoothen=True)
