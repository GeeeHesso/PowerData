Power Data
==========

This repository contains a toolset for generating data for machine learning applications in power systems. It consists in two parts:

- [Time Series](./TimeSeries) is a Python module for generating realistic time series based on historical data.
- [Temperate Optimal Power Flow (TOPF)](https://github.com/gillioz/TemperateOptimalPowerFlow.jl) is a Julia framework for simulating a realistic dispatch of power generation on a transmission grid.

Several Jupyter notebooks documenting the usage of these packages can be found in the [doc](./doc) directory.

The [run](./run) directory contains more notebooks used to generate data for a particular model of the European transmission grid.
This model can be found in the [models](./models) directory, along with a few other models restricted to individual countries.
