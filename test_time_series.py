import pytest
import time_series as ts
import numpy as np


@pytest.mark.parametrize("test_input,expected", [
    ([0., 1., None, None, 4., 5., 6., 7., None, 9.], [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]),
    ([0., 1., 2., None, 4., 5., 6., None, None], [0., 1., 2., 3., 4., 5., 6., 4., 2.]),  # last value missing
    ([None, None, 2., 3., 4., 5.], [4., 3., 2., 3., 4., 5.]),   # first values missing
    ([None, 1., 2., 3., None, None], [1.5, 1., 2., 3., 2.5, 2.]),   # both first and last values missing
    ([0, 1, 2, 3], [0, 1, 2, 3]),   # no missing values
    (np.array([0., 1., None, None, 4.]), np.array([0., 1., 2., 3., 4.]))
])
def test_interpolate_missing_values(test_input, expected):
    interpolated = ts.interpolate_missing_values(test_input)
    assert len(interpolated) == len(expected)
    assert all([x == y for x, y in zip(interpolated, expected)])
