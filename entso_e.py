import pandas as pd


def extract_time_series(data_source: str, data_type: str, column_name: str,
                        data_filter: dict, year: int | list, frequency) -> list:
    """Extracts a yearly time series of a given type, for a given country and a given year.
    The data files must be stored in the folder 'data_source'.
    The data type and column name must match the entso-e format.
    'country_code' is 'DE' for Germany, 'FR' for France, and so on.
    'frequency' uses the following format: 'H' means every hour, '2H' every 2 hours, '15T' every 15 minutes, and so on.
    The time series is returned as a list. Missing value are set to 'None'."""
    if isinstance(year, list):
        return [extract_time_series(data_source, data_type, column_name, data_filter, y, frequency) for y in year]
    timesteps = [str(t) for t in
                 pd.date_range('%s-01-01' % year, '%s-01-01' % (year + 1), freq=frequency, inclusive='left')]
    time_series_dict = {t: None for t in timesteps}
    for month in range(1, 13):
        data = pd.read_csv('%s/%d_%02d_%s.csv' % (data_source, year, month, data_type), sep='\t')
        for filter_key, filter_val in data_filter.items():
            data = data[data[filter_key] == filter_val]
        monthly_dict = dict(zip(data['DateTime'].apply(lambda x: x[:19]), data[column_name]))
        time_series_dict.update(monthly_dict)

    return [time_series_dict[t] for t in timesteps]


def extract_load_time_series(data_source: str, country_code: str, year: int | list, frequency: str = 'H') -> list:
    """Extracts a yearly time series of the total load for a given country and a given year.
    The data files must be stored in the folder 'data_source'.
    'country_code' is 'DE' for Germany, 'FR' for France, and so on.
    'frequency' uses the following format: 'H' means every hour, '2H' every 2 hours, '15T' every 15 minutes, and so on.
    The time series is returned as a list. Missing value are set to 'None'."""
    country_filter = {'AreaTypeCode': 'CTY', 'MapCode': country_code}
    return extract_time_series(data_source, 'ActualTotalLoad_6.1.A', 'TotalLoadValue',
                               country_filter, year, frequency)


def extract_border_flow_time_series(data_source: str, from_country: str, to_country: str,
                                    year: int | list, frequency: str = 'H') -> list:
    """Extracts a yearly time series of the border flow between two countries for a given year.
    The data files must be stored in the folder 'data_source'.
    'from_country' and 'to_country' are of the form 'DE' for Germany, 'FR' for France, and so on.
    'frequency' uses the following format: 'H' means every hour, '2H' every 2 hours, '15T' every 15 minutes, and so on.
    The time series is returned as a list. Missing value are set to 'None'."""
    country_filter = {'OutAreaName': from_country + ' CTY', 'InAreaName' : to_country + ' CTY'}
    return extract_time_series(data_source, 'PhysicalFlows_12.1.G', 'FlowValue',
                               country_filter, year, frequency)


def extract_production_by_type_time_series(data_source: str, gen_type: str, country_code: str,
                                           year: int | list, frequency: str = 'H') -> list:
    """Extracts a yearly time series of the total production of a given type, for a given country and a given year.
    The data files must be stored in the folder 'data_source'.
    'gen_type' is of the form 'Nuclear', 'Hydro Pumped Storage', 'Hydro Run-of-river and poundage', and so on.
    'country_code' is 'DE' for Germany, 'FR' for France, and so on.
    'frequency' uses the following format: 'H' means every hour, '2H' every 2 hours, '15T' every 15 minutes, and so on.
    The time series is returned as a list. Missing value are set to 'None'."""
    data_filter = {'AreaTypeCode': 'CTY', 'MapCode': country_code, 'ProductionType': gen_type}
    return extract_time_series(data_source, 'AggregatedGenerationPerType_16.1.B_C',
                               'ActualGenerationOutput', data_filter, year, frequency)