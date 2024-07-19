PowerData: data generation for the European model
=================================================

This directory presents the complete workflow to generate data for the European transmission grid model.
The following notebooks must be run in this order:

1. [generate_loads](./generate_loads.ipynb):
   generate synthetic time series for all the loads in the model
2. [import_nuclear_series](./import_nuclear_series.ipynb):
   import actual data for non-dispatchable generators of nuclear type
3. [import_border_flows](./import_border_flows.ipynb):
   calculate the import/export balance for each country in the model
4. [generate_cost_noise](./generate_cost_noise.ipynb):
   ...
5. [setup_OPF](./setup_OPF) (Julia):
   ...
