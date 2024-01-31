Time Series
===========

This is a Python toolset for generating (possibly many) synthetic time series of various type.
The main tools are contained in the module [time_series.py](./time_series.py).


Existing models
---------------

The [models](./models) directory contains several models for loads based on the 
[public ENTSO-E data](https://transparency.entsoe.eu/).
There are models for most European countries. Some of them have 15-minutes time steps (e.g. Germany),
while others have 1-hour time steps (e.g. France).

To generate random time series based on these models, just use the following code:

```python
import time_series
model = time_series.import_model('models/entsoe_load_CH_2015_2023.npz')
time_series.generate_time_series(model, 20)
```

This code outputs 20 series of 364 x 24 time steps in a *NumPy* array format.
It is based on the Swiss model constructed from historic data of the years 2015 to 2023.

To generate time series with a broader distribution simulating more extreme conditions,
the standard deviation can be scaled up, for instance here by a factor 2:

```python
time_series.generate_time_series(model, 20, std_scaling=2.0)
```

Normalized time series with a mean value of one can be obtained with:

```python
time_series.generate_time_series(model, 20, normalize=True)
```


Creating models
---------------

In addition to existing models, it is possible to create new models based on any type of existing time series.

Assuming that you have gathered 3 historic time series for a particular power system
and stored them in lists named `time_series_1`, `time_series_2`, and `time_series_3`,
then creating a model from these lists is as simple as

```python
model = time_series.create_model([time_series_1, time_series_2, time_series_3])
```

This model can be used to generate new time series resembling the historic ones.

The procedure for creating new models is detailed in [a Jupyter Notebook](../doc/ENTSO-E_models.ipynb).
Synthetic time series for power generation or cross-border flows can for instance be easily
generated based on ENTSO-E data with the help of the [entso_e.py](./entso_e.py) module.


Installation
------------

This project only relies on the standard Python modules [NumPy](https://numpy.org/)
and [SciPy](https://scipy.org/), as well as [PyTest](https://pytest.org/) for testing
and [Pandas](https://pandas.pydata.org/) for dealing with datasets in the examples.

All dependencies can be installed with the command:

```console
pip install -r requirements.txt
```

The installation can be verified by running tests:

```console
pytest
```


Future developments
-------------------

- More models could be added to the existing list.
- The *SciPy* support for linear algebra with sparse matrices is limited.
A more efficient implementation could be written in Julia.


