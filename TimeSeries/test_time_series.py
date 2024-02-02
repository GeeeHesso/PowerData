import pytest
import time_series as ts
import numpy as np


@pytest.mark.parametrize("test_input,expected", [
    ([0., 1., None, None, 4., 5., 6., 7., None, 9.], [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]),
    ([0., 1., 2., None, 4., 5., 6., None, None], [0., 1., 2., 3., 4., 5., 6., 4., 2.]),  # last value missing
    ([None, None, 2., 3., 4., 5.], [4., 3., 2., 3., 4., 5.]),  # first values missing
    ([None, 1., 2., 3., None, None], [1.5, 1., 2., 3., 2.5, 2.]),  # both first and last values missing
    ([0, 1, 2, 3], [0, 1, 2, 3]),  # no missing values
    (np.array([0., 1., None, None, 4.]), np.array([0., 1., 2., 3., 4.]))
])
def test_interpolate_missing_values(test_input, expected):
    interpolated = ts.interpolate_missing_values(test_input)
    assert len(interpolated) == len(expected)
    assert all([x == y for x, y in zip(interpolated, expected)])


@pytest.mark.parametrize("days,steps", [(365, 1), (366, 24), (364, 24 * 4)])
def test_crop_364(days, steps):
    series = np.ones(days * steps)
    result = ts.crop_364(series)
    expected = np.ones(364 * steps)
    assert np.all(result == expected)


def test_crop_364_fails():
    series = np.ones(123)
    with pytest.raises(Exception) as e:
        result = ts.crop_364(series)


def test_find_first_monday():
    sample_weekday = [1, 2, 3, 2, 1]
    sample_saturday = [1, 2, 1, 1, 1]
    sample_sunday = [1, 1, 1, 1, 1]
    sample_week = (sample_weekday + sample_weekday + sample_saturday + sample_sunday +
                   sample_weekday + sample_weekday + sample_weekday)
    time_series = np.repeat(np.array([sample_week]), 52, axis=0).flatten()
    assert ts.find_first_monday(time_series) == 4


def test_find_first_monday_fails():
    sample_weekday = [1, 2, 3, 2, 1]
    sample_saturday = [1, 2, 1, 1, 1]
    sample_sunday = [1, 1, 1, 1, 1]
    sample_week = (sample_weekday + sample_saturday + sample_weekday + sample_sunday +
                   sample_weekday + sample_weekday + sample_weekday)  # saturday and sunday do not follow each other
    time_series = np.repeat(np.array([sample_week]), 52, axis=0).flatten()
    with pytest.raises(Exception) as e:
        result = ts.find_first_monday(time_series)


@pytest.mark.parametrize("year,expected", [(2024, 0), (2023, 1), (2022, 2), (2021, 3), (2020, 5), (2019, 6)])
def test_roll_monday(year, expected):
    time_series = np.array([t for t in range(364 * 24)])
    result = ts.roll_monday(time_series, year)
    assert result[0] == 24 * expected


def test_fourier_transform_and_back():
    time_series = np.random.random((10, 364*24))
    fourier_modes = ts.fourier_transform(time_series)
    assert len(fourier_modes) == len(time_series)
    assert np.max(np.abs(ts.inverse_fourier_transform(fourier_modes) / time_series - 1)) < 10e-6


def test_inverse_fourier_transform_and_back():
    fourier_modes = np.random.random((10, 364*24))
    time_series = ts.inverse_fourier_transform(fourier_modes)
    assert len(fourier_modes) == len(time_series)
    assert np.max(np.abs(ts.fourier_transform(time_series) / fourier_modes - 1)) < 10e-6


@pytest.mark.parametrize("T,n", [(365, 8), (365 * 4, 3)])
def test_create_model(T, n):
    # verify that the model's mean is equal to the Fourier transform of the mean of its input
    time_series = [np.random.random(T) for _ in range(n)]
    years = list(range(2000, 2000 + n))
    # create the model
    model = ts.create_model(time_series, years, reduce=False)
    # extract the mean value
    result = model[:, :T].diagonal()
    # compare with the Fourier transform of the mean of the time series
    periodic_time_series = [ts.make_364_periodic(time_series[i], years[i]) for i in range(n)]
    periodic_time_series_mean = np.array(periodic_time_series).mean(axis=0)
    expected = ts.fourier_transform(periodic_time_series_mean)
    assert np.max(np.abs(result - expected)) < 1e-6


@pytest.mark.parametrize("time_series,years", [
    ([np.random.random(365)], None),  # single time series
    ([np.random.random(365), np.random.random(366), np.random.random(365)], [2000, 2001]) # number of years not matching
])
def test_create_model_fail(time_series, years):
    with pytest.raises(Exception) as e:
        result = ts.create_model(time_series, years)


@pytest.mark.parametrize("T,n", [(365*2, 8), (365 * 3, 3)])
def test_generate_time_series(T, n):
    # verify that time series generated with zero variance reproduce the input's average
    time_series = np.random.random((n, T))
    years = list(range(2000, 2000 + n))
    # create the model
    model = ts.create_model(time_series, years)
    # generate a synthetic time series with zero variance
    result = ts.generate_time_series(model, std_scaling=0.0)[0]
    # compare with the average of time series after making them periodic
    periodic_time_series = np.array([ts.make_364_periodic(time_series[i], years[i]) for i in range(n)])
    expected = periodic_time_series.mean(axis=0)
    assert np.max(np.abs(result - expected)) < 1e-6


@pytest.mark.parametrize("series_count, series_length, series_type", [
    (10, 364, "list"),
    (16, 4*364, "np.ndarray")
])
def test_compute_pairwise_correlations(series_count, series_length, series_type):
    # generate a number of random time series
    if series_type == "list":
        time_series = [np.random.normal(size=(series_length)) for _ in range(series_count)]
    else:
        time_series = np.random.normal(size=(series_count, series_length))
    # compute the correlation coefficients
    correlations = ts.compute_pairwise_correlations(time_series)
    # verify that the number of pairwise correlation is correct and that all correlations are between +1 and -1
    assert correlations.shape == (series_count * (series_count - 1) // 2, ) \
        and np.abs(correlations).max() <= 1.0


@pytest.mark.parametrize("daily_steps,frequencies", [
    (24, (10, 6, 10)),
    (24*2, (10, 6, 10)),
    (24*4, (10, 6, 10)),
    (24, (20, 6, 20)),
])
def test_generate_noise(daily_steps, frequencies):
    # verify that the mean value of noise time series is zero and the standard deviation one
    daily_freqs, weekly_freqs, yearly_freqs = frequencies
    noise_series = ts.generate_noise(daily_steps, 100, daily_freqs, weekly_freqs, yearly_freqs)
    mean = noise_series.mean()
    std = noise_series.std(axis=1).mean()
    assert np.abs(mean) < 1e-8 and np.abs(std - 1.0) < 0.05
