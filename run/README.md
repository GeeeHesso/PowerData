PowerData: data generation for the European model
=================================================

This directory presents the complete workflow to generate data for the European transmission grid model.
The following notebooks must be run in this order:

1. [generate_loads](./generate_loads.ipynb):
   generate synthetic time series for all the loads in the model
2. [import_nuclear_series](./import_nuclear_series.ipynb):
   import actual data for non-dispatchable generators of nuclear type
3. [import_border_flows](./import_border_flows.ipynb):
   calculate the import/export balance for each country in the model based on actual data
4. [generate_cost_noise](./generate_cost_noise.ipynb):
   generate noise series for the generation costs
5. [setup_TOPF](./setup_TOPF.ipynb) (Julia):
   collect all the data generated above and setup the optimal power flow computation
6. [TOPF_test](./TOPF_test.ipynb) (Julia):
   run a small-scale version of the optimal power flow as a test

After that, the optimal power flow computation can be launched with:
```bash
julia run/run_TOPF.jl
```
