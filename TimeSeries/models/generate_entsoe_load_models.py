import sys
sys.path.append('..')

import time_series as ts
import entso_e as ee
import os

data_source = os.path.expanduser('~/data/entso-e/raw')
default_frequencies = ['15T', '30T', 'H']


def generate_entsoe_load_models(country: str, years_min: int, years_max: int, frequency: str = None):
    """Generate load models for a given country, based on years between years_min and years_max (both included)
    and using time steps with the largest possible frequency
    (e.g. use ['15T', '30T', 'H'] to try building a model with 15 min, 30 min, or 1 hour time steps successively)"""
    years = range(years_max, years_min - 1, -1)
    frequencies = [frequency] if frequency is not None else default_frequencies
    for freq in frequencies:
        print(country, '-', freq)
        data = {}
        ok_count = 0
        fail_count = 0
        for year in years:
            print(' - %s:' % year, end=' ')
            year_data = ee.extract_load_time_series(data_source, country, year, freq)
            missing_values = year_data.count(None)
            if missing_values < 364:
                print('ok with %d missing values' % missing_values)
                ok_count += 1
                data[year] = ts.interpolate_missing_values(year_data, verbose=False)
            else:
                print('X')
                fail_count += 1
            if fail_count - ok_count >= 2:
                break
        if ok_count >= 5:
            valid_years = list(data.keys())
            print(' => Creating model based on years', valid_years)
            model = ts.create_model([data[year] for year in valid_years], valid_years, verbose=False)
            filename = 'entsoe_load_%s_%d_%d' % (country, min(valid_years), max(valid_years))
            print(' => Writing to file', filename)
            ts.export_model(filename, model)
            break


def generate_all_entsoe_load_models(years_min: int, years_max: int, frequency: str = None):
    """Generate load models for a list of 21 ENTSO-E countries., based on years between years_min and years_max
    (both included) and using time steps with the largest possible frequency
    (e.g. use ['15T', '30T', 'H'] to try building a model with 15 min, 30 min, or 1 hour time steps successively)"""
    countries = ["AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "ES",
                 "FR", "GR", "HR", "HU", "IT", "LU", "LV", "ME",
                 "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK"]
    for country in countries:
        generate_entsoe_load_models(country, years_min, years_max, frequency)

