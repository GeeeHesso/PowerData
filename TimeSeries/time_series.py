import numpy as np
import scipy as sp
import datetime
import scipy.ndimage


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


def smoothen(time_series: np.ndarray, threshold: float = 30.0, filter_window: int = 73) -> np.ndarray:
    """Remove spikes in a time series by applying a median filter.
    The original value are preserved whenever they do not differ significantly from the filtered value."""
    filtered_time_series = scipy.ndimage.median_filter(time_series, filter_window, mode='wrap')
    return np.where(np.abs(filtered_time_series - time_series) < threshold, time_series, filtered_time_series)


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
    if time_series.shape[-1] % 2 != 0:
        raise ValueError("The fourier transform requires a series of even length")
    complex_modes = np.fft.rfft(time_series)
    return np.concatenate([complex_modes.real, np.delete(complex_modes.imag, [0, -1], axis=-1)], axis=-1)


def inverse_fourier_transform(real_modes: np.ndarray) -> np.ndarray:
    """Computes the inverse of the function 'fourier_transform'.
    The input is a real array representing T + 1 real parts of the Fourier coefficients and T - 1 imaginary parts.
    The output is a real time series with 2T steps."""
    dim = len(real_modes.shape)
    if real_modes.shape[-1] % 2 != 0:
        raise ValueError("The inverse fourier transform requires a series of even length")
    padding_range = np.array([(0, 0) for _ in range(dim - 1)] + [(0, 2)])
    real_and_imaginary_modes = np.pad(real_modes, padding_range)
    real_modes, imaginary_modes = np.split(real_and_imaginary_modes, 2, axis=-1)
    imaginary_modes = np.roll(imaginary_modes, 1, axis=-1)
    complex_modes = real_modes + 1j * imaginary_modes
    return np.fft.irfft(complex_modes)


def create_model(time_series: list | np.ndarray, reduce: bool = True,
                 eps: float = 1e-8, verbose: bool = True) -> sp.sparse.csr_array:
    """Create a model from a list of time series with T steps each.
    The output is in the form of a T x 2T sparse matrix,
    in which the first T x T block is diagonal and contains the mean value of the model,
    and the second T x T block parameterize the standard deviation."""
    if len(time_series) < 2:
        raise ValueError("At least 2 times series are needed to create a model")
    # compute the Fourier transform
    x = fourier_transform(np.array(time_series))
    # compute the mean and covariance
    x_mean = x.mean(axis=0)
    x_cov = np.cov(x, rowvar=False, bias=True)
    if verbose:
        print("Computing the Cholesky decomposition of a %d x %d matrix (this may take some time)" % x_cov.shape)
    # use the LDL decomposition that is compatible with semi-positive definite matrices
    lu, d, perm = sp.linalg.ldl(x_cov, overwrite_a=True)
    # set to zero all negative diagonal elements resulting from numerical errors
    d[d < 0.0] = 0.0
    # define the lower-diagonal "square root" L of the covariance matrix, such that L * L' = cov
    L = lu @ np.sqrt(d)
    # this matrix has typically many zeros, so it can be reduced without altering the performance of the model
    if reduce:
        L_max = eps * np.abs(L).max()
        L[np.abs(L) < L_max] = 0.0
    # arrange the result into a T x 2T sparse matrix
    return sp.sparse.csr_array(np.concatenate((np.diag(x_mean), L), axis=1))


def generate_time_series(model: sp.sparse.csr_array, n: int = 1, std_scaling: float = 1.0,
                         normalize: bool = False) -> np.ndarray:
    """Generate a given number n of time series from the model.
    The standard deviation can be adjusted with the parameter 'std_scaling'.
    If 'normalize' is set to True, the series is rescaled such that its mean value is equal to one."""
    T = model.shape[0]
    x = np.array([model @ np.concatenate((np.ones(T), np.random.normal(size=T, scale=std_scaling)))
                  for _ in range(n)])
    if normalize:
        x *= T / x[:, 0].reshape(-1, 1)
    return inverse_fourier_transform(x)


def get_mu(model: sp.sparse.csr_array) -> np.ndarray:
    """Return a time series that corresponds to the mean value of the model."""
    return inverse_fourier_transform(model.diagonal())


def get_sigma(model: sp.sparse.csr_array) -> np.ndarray:
    """Return the covariance matrix of time series according to the model."""
    T = model.shape[0]
    # compute sigma, the covariance matrix of the Fourier modes
    L = model[:, T:]
    sigma = (L * L.transpose()).toarray()
    # perform the inverse Fourier transform with respect to both axes
    cov = inverse_fourier_transform(inverse_fourier_transform(sigma).transpose())
    # symmetrize the matrix to reduce numerical errors
    return 0.5 * (cov + cov.transpose())


def get_sigma_variance(model: sp.sparse.csr_array, truncate: int = 10) -> float:
    """Return the variance in time of the covariance of time series according to the model."""
    T = model.shape[0]
    Lt = model[:, T:].transpose()
    return sum(np.var(inverse_fourier_transform(Lt[t, :].toarray())) for t in range(min(T, truncate)))


def get_optimal_std_scaling(model: sp.sparse.csr_array, rho: float, truncate: int = 10) -> float:
    """Return a scaling factor for the standard deviation such that the time series obey a given average correlation."""
    return np.sqrt(np.var(get_mu(model)) / get_sigma_variance(model, truncate) * (1.0 / rho - 1.0))


def export_model(filename: str, model: sp.sparse.csr_array) -> None:
    """Save model to file 'filename.npz' after transforming it into a sparse matrix"""
    sp.sparse.save_npz(filename, model)
    return


def import_model(filename: str) -> sp.sparse.csr_array:
    """Load model from file 'filename'"""
    return sp.sparse.load_npz(filename)


def compute_pairwise_correlations(time_series: list | np.ndarray):
    """Computes a list of all correlation coefficients between pairs of time series."""
    correlation_matrix = np.corrcoef(time_series)
    upper_triangular_indices = np.triu_indices(correlation_matrix.shape[0], k=1)
    return correlation_matrix[upper_triangular_indices]


def generate_noise_with_frequencies(daily_steps: int, frequencies: list | np.ndarray, count: int = 1):
    """Generates a time series of 364 days with a given number of daily steps (typically 24).
    The time series is a superposition of oscillatory series (cosines)
    with the given frequencies, and with random phases distributed uniformly.
    Each oscillatory series is weighted by a random factor drawn from a normal distribution.
    The times series are normalized to have a standard deviation of one on average."""
    T = 364 * daily_steps  # length of the time series
    n = len(frequencies)  # number of frequencies defining the noise
    freq_times_t = np.outer(range(T), frequencies).reshape(1, T, n)
    theta = np.random.rand(count, 1, n)  # random phases
    A = np.random.normal(size=(count, 1, n))  # random amplitude
    return np.sqrt(2./n) * np.sum(A * np.cos(2. * np.pi * (freq_times_t + theta)), axis=2)


def generate_noise(daily_steps: int = 24, count: int = 1, daily_frequencies: int = 10,
                   weekly_frequencies: int = 6, yearly_frequencies: int = 10):
    """Generates a time series of 364 days with a given number of daily steps (typically 24).
    The time series is a superposition of oscillatory series (cosines)
    with a number of daily, weekly, and yearly frequencies, and with random phases distributed uniformly.
    Each oscillatory series is weighted by a random factor drawn from a normal distribution.
    The times series are normalized to have a standard deviation of one on average."""
    frequencies = np.array([364 * (n + 1) for n in range(daily_frequencies)]
                           + [52 * (n + 1) for n in range(weekly_frequencies)]
                           + [n + 1 for n in range(yearly_frequencies)]) / (364 * daily_steps)
    return generate_noise_with_frequencies(daily_steps, frequencies, count)

