def interpolate_missing_values(time_series: list, verbose: bool = True) -> list:
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
        print('Interpolating missing values:')
        print(' -> number of gaps:', len(gaps))
        print(' -> number of missing values:', sum(gap[1] - gap[0] for gap in gaps))

    # fill the gaps with interpolation
    new_time_series = time_series.copy()
    for (gap_begin, gap_end) in gaps:
        last_value = time_series[gap_begin - 1]
        next_value = time_series[gap_end % series_length]
        gap_width = gap_end - gap_begin + 1
        for t in range(gap_begin, gap_end):
            new_time_series[t] = (last_value * (gap_end - t) + next_value * (t - gap_begin + 1)) / gap_width

    return new_time_series
