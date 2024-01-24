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

Then we define the cost associated with line $i$ to be the square of the loading rate,
weighted by the line's thermal limit, and multiplied by a constant $c_i$:

$c_i P_i^{th} R_i^2$

Using the definition of the loading rate, this is equivalent to:

$c_i P_i^2 / P_i^{th}$

This is independent of the sign of the power $P_i$, which can be positive or negative,
depending on the definition of the line's direction.
The constant $c_i$ defines the cost per unit of power.

Installation
------------

Since this package relies on an unregistered fork of PowerModels,
you first need to install it:

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


**This repository is still under construction.**

