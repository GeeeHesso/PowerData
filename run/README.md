PowerData: data generation for the European model
=================================================

This directory presents the complete workflow to generate data for the European transmission grid model.
The following notebooks must be run in this order:

1. [import_nuclear_series](./import_nuclear_series.ipynb) (Python):
   import actual data for non-dispatchable generators of nuclear type
2. [import_border_flows](./import_border_flows.ipynb) (Python):
   calculate the import/export balance for each country in the model based on actual data
3. [generate_loads](./generate_loads.ipynb) (Python):
   generate synthetic time series for all the loads in the model
4. [generate_cost_noise](./generate_cost_noise.ipynb) (Python):
   generate noise series for the generation costs
5. [TOPF_setup](./TOPF_setup.ipynb) (Julia):
   collect all the data generated above and setup the optimal power flow computation
6. [TOPF_test](./TOPF_test.ipynb) (Julia):
   run a small-scale version of the optimal power flow as a test;
   the actual computation can be launched with
   ```bash
   julia run_TOPF.jl
   ```
7. [TOPF_analysis](./TOPF_analysis.ipynb) (Julia):
   analyse the results of the optimal power flow computation and export tables containing the data
8. [TOPF_noise_comparison](./TOPF_noise_comparison.ipynb) (Julia):
   comparison between different noise levels for the generation cost, which must be generated before with 
   ```bash
   julia run_TOPF.jl <noise-level-value>
   ```

