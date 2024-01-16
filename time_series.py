import numpy as np
import datetime


def interpolate_missing_values(time_series: list | np.ndarray, verbose: bool = True) -> list | np.ndarray:
    """Fill in the missing values (None) in a time series by interpolating between the nearest existing values.
    Wraps around if necessary."""

    series_length = len(time_series)

    # if the first value is missing, roll the series to make it non-missing
    if time_series[0] is None:
        first_existing_value = next(t for t, val in enumerate(time_series) if val is not None)
        new_time_series = time_series[first_existing_value:] + time_series[:first_existing_value]
        new_time_series = interpolate_missing_values(new_time_series)
        return new_time_series[-first_existing_value:] + new_time_series[:-first_existing_value]

    # find all the gaps in the data, characterized by the beginning and end index
    gaps = []
    gap_length = 0
    for t, val in enumerate(time_series):
        if val is None:
            gap_length += 1
        elif gap_length > 0:
            gaps.append((t - gap_length, t))
            gap_length = 0
    if gap_length > 0:
        gaps.append((len(time_series) - gap_length, len(time_series)))

    if verbose:
        print('Interpolating %d missing values spread over %d gap(s)' %
              (sum(gap[1] - gap[0] for gap in gaps), len(gaps)))

    # fill the gaps with interpolation
    new_time_series = time_series.copy()
    for (gap_begin, gap_end) in gaps:
        last_value = time_series[gap_begin - 1]
        next_value = time_series[gap_end % series_length]
        gap_width = gap_end - gap_begin + 1
        for t in range(gap_begin, gap_end):
            new_time_series[t] = (last_value * (gap_end - t) + next_value * (t - gap_begin + 1)) / gap_width

    return new_time_series


def crop_364(time_series: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Crops a time series to 364 days (52 weeks * 7 days).
    Works with different time steps."""
    verbose_print = print if verbose else lambda *a, **k: None
    input_length = len(time_series)
    if input_length % 365 == 0:
        day_length = input_length // 365
        verbose_print("Cropping regular year (365 days) to 364 = 52 weeks * 7 days.")
    elif input_length % 366 == 0:
        day_length = input_length // 366
        verbose_print("Cropping leap year (366 days) to 364 = 52 weeks * 7 days.")
    elif input_length % 364 == 0:
        day_length = input_length // 364
    else:
        raise ValueError("The length of the time series is not compatible with a year (364 to 366 days)")
    return time_series[0:364 * day_length]


def find_first_monday(time_series: np.ndarray) -> int:
    """Find the position of the first Monday in the series (integer between 0 and 6),
    based on the assumption that the Saturdays and Sundays are distinguishable from other days of the week."""
    # change type of time_series to numpy array, in case it was a list
    time_series = np.array(time_series)
    # reorganize the time series into weekdays (7 series of length 364 * n_per_day)
    weekday_series = time_series.reshape((52, 7, -1)).transpose((1, 0, 2)).reshape((7, -1))
    # compute the covariance between all week days
    weekday_covariance = np.cov(weekday_series).sum(axis=0)
    # select the two least correlated days
    weekend_days, = np.where(np.isin(weekday_covariance, np.sort(weekday_covariance)[0:2]))
    # pick sunday as the second of the least correlated days, provided that they are in a row (otherwise raise an error)
    if (weekend_days[0] + 1) % 7 == weekend_days[1]:
        sunday = weekend_days[1]
    elif (weekend_days[1] + 1) % 7 == weekend_days[0]:
        sunday = weekend_days[0]
    else:
        raise Exception("The periodicity properties of the time series does not permit to find the first Monday")
    return (sunday + 1) % 7


def roll_monday(time_series: np.ndarray, year: int = None, verbose: bool = True) -> np.ndarray:
    """Roll the time series to make it start on a Monday.
    If year is provided, the offset is computed from the calendar.
    If it is not provided, the first Monday is found from the weekly periodicity."""
    # change type of time_series to numpy array, in case it was a list
    time_series = np.array(time_series)
    # find the number of days until the first monday in the series
    first_monday = (7 - datetime.date(year, 1, 1).weekday()) % 7 if year is not None \
        else find_first_monday(time_series)
    if verbose:
        print("Rolling the series backward by %d days to make it start on a Monday" % first_monday)
    steps_per_day = len(time_series) // 364
    return np.roll(time_series, -steps_per_day * first_monday)


def make_364_periodic(time_series: np.ndarray, year: int = None, verbose: bool = True) -> np.ndarray:
    """Turns a time series into a periodic one, with 52 weeks * 7 days, and starting on a Monday.
    If year is provided, the offset is computed from the calendar.
    If it is not provided, the first Monday is found from the weekly periodicity."""
    cropped_series = crop_364(time_series, verbose=verbose)
    return roll_monday(cropped_series, year=year, verbose=verbose)