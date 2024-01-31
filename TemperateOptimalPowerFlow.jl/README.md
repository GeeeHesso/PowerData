TOPF: Temperate Optimal Power Flow
==================================

This package relies on a [fork](https://github.com/gillioz/PowerModels.jl)
of the [PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl) package
to perform an Optimal Power Flow (OPF) that disfavors heavily-loaded lines.

In addition to the conventional cost associated with power generation,
the "temperate" OPF includes a cost propotional to the square of each line's loading rate.

Line cost function
------------------

Given the power $P_i$ flowing through a line with index $i$,
and the thermal limit of the line $P_i^{th}$, the loading rate $R_i$ of the line is given by the formula

$R_i = | P_i | / P_i^{th}$

We define the cost associated with line $i$ to be the square of the loading rate,
weighted by the line's thermal limit, and multiplied by a constant $c_i$:

$c_i P_i^{th} R_i^2$

Using the definition of the loading rate, this is equivalent to:

$c_i P_i^2 / P_i^{th}$

This is independent of the sign of the power $P_i$, which can be positive or negative
depending on the definition of the line's direction.
The constant $c_i$ defines the cost per unit of power.
It is typically the same for all lines of a network.


Usage
-----

To begin, import a model in [MatPower](https://www.pserc.cornell.edu/matpower/)
or [PowerModels](https://lanl-ansi.github.io/PowerModels.jl/stable/network-data/) format:

```julia
using PowerModels
network = parse_file("models/pantagruel.json")
```

The model can be modified to add line costs, here specifying a unique cost for all the lines (the constant $c_i$ above):
```julia
add_line_costs!(network, 1000)
```

Time series for all generators of the model can be computed with a (modified) OPF. This can be done with the help of an optimizer such as [Gurobi](https://www.gurobi.com/) (commercial)
or [Ipopt](https://github.com/coin-or/Ipopt) (open source).
Note that there is one OPF computation for each time step
of a whole year, so for large networks this can take quite some time (several hours!):
```julia
using Gurobi  # alternatively: using Ipopt
optimizer = get_silent_optimizer()

iterate_dc_opf(network, "load_series.csv", "gen_cost_series.csv", "gens_list.csv", "output.csv", optimizer)
```

Besides the network and optimizer objects, the parameters of the function `iterate_dc_opf` are CSV files:
- "load_series.csv" contains a dataframe whose columns are the time steps of the series
and whose rows are the loads of the model;
- "gen_cost_series.csv" contains a similar dataframe, but the rows indicate the variable production cost
of some or all generators;
- "output.csv" is the file to which the result of the OPF is written;
if this file exists already, only missing time steps are added.

A more complete description of the package's possibilities can be found in the
Jupyter Notebook [doc/PanTaGruEl_example.ipynb](./doc/PanTaGruEl_example.ipynb).


Installation
------------

This package relies on an unregistered fork of PowerModels that needs to be installed first:

```julia
] add "https://github.com/gillioz/PowerModels.jl"
```

Then the package itself can be installed with
```julia
] dev <relative path>/TemperateOptimalPowerFlow.jl
```

You can test the installation by running

```julia
] test TemperateOptimalPowerFlow
```

