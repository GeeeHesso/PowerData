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


def fourier_transform(time_series: np.ndarray) -> np.ndarray:
    """Computes the discrete Fourier transform of a time series with 2T steps.
    The result is stored in an array of length 2T.
    The first T + 1  entries are the real parts of the Fourier coefficients.
    The last T - 1 entries are the imaginary parts of all coefficients except the first and the last."""
    complex_modes = np.fft.rfft(time_series)
    return np.concatenate([complex_modes.real, np.delete(complex_modes.imag, [0, -1], axis=-1)], axis=-1)


def inverse_fourier_transform(real_modes: np.ndarray) -> np.ndarray:
    """Computes the inverse of the function 'fourier_transform'.
    The input is a real array representing T + 1 real parts of the Fourier coefficients and T - 1 imaginary parts.
    The output is a real time series with 2T steps."""
    dim = len(real_modes.shape)
    padding_range = np.array([(0, 0) for _ in range(dim - 1)] + [(0, 2)])
    real_and_imaginary_modes = np.pad(real_modes, padding_range)
    real_modes, imaginary_modes = np.split(real_and_imaginary_modes, 2, axis=-1)
    imaginary_modes = np.roll(imaginary_modes, 1, axis=-1)
    complex_modes = real_modes + 1j * imaginary_modes
    return np.fft.irfft(complex_modes)


def create_model(time_series: list, years: list = None, verbose: bool = True) -> np.ndarray:
    """Create a model from a list of time series"""
    if len(time_series) == 0:
        raise ValueError("At least 2 times series are needed to create a model")
    if years is not None and len(years) != len(time_series):
        raise ValueError("The number of years should be the same as the number of time series.")
    # create periodic time series from the input
    periodic_time_series = [make_364_periodic(series, verbose=verbose) for series in time_series] if years is None \
        else [make_364_periodic(time_series[i], years[i], verbose=verbose) for i in range(len(time_series))]
    # compute the Fourier transform
    x = fourier_transform(np.array(periodic_time_series))
    # compute the mean and covariance
    x_mean = x.mean(axis=0)
    x_cov = np.cov(x, rowvar=False, bias=True)
    if verbose:
        print("Computing the square root of a %d x %d matrix (this may take some time)" % (len(x_mean), len(x_mean)))
    # decompose the covariance matrix into eigenvalues and eigenvectors
    x_cov_eigval, x_cov_eigvec = np.linalg.eigh(x_cov)
    # set to zero all negative eigenvalues resulting from numerical errors
    x_cov_eigval[x_cov_eigval < 0.0] = 0.0
    # compute the square root of the covariance matrix
    x_std = x_cov_eigvec @ np.diag(np.sqrt(x_cov_eigval)) @ x_cov_eigvec.transpose()
    # return a matrix in which the first row is the mean value and the rest the standard deviation with correlations
    return np.concatenate((x_mean.reshape(1, -1), x_std))


def generate_time_series(model: np.ndarray, n: int = 1, std_scaling: float = 1.0) -> np.ndarray:
    """Generate a given number n of time series from the model.
    The standard deviation can be adjusted with the parameter 'std_scaling'."""
    x_mean = model[0]
    x_std = model[1:]
    x_size = len(x_mean)
    return np.array([inverse_fourier_transform(x_mean + std_scaling * x_std @ np.random.normal(size=x_size))
                     for _ in range(n)])