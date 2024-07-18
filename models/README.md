PowerData: models
=================

This directory contains several realistic transmission grid models, including

- a large model of continental Europe, with more than 4000 buses, 8000 lines, and 800 generators, and
- smaller models of individual countries (Germany, France, Spain, Italy, and Switzerland) based on the European one.

The models are given in the [PowerModels](https://lanl-ansi.github.io/PowerModels.jl/) format, which is itself derived from the [MatPower](https://matpower.org/) file format. In addition to the standard data structure, our models contain a few additional fields such as:
- line susceptances needed for DC power flow computations;
- for some generators, a list of names matching those used in the [ENTSO-E database](https://transparency.entsoe.eu/), so that actual production data can be used when needed;
- an aggregated production type to simplify the grouping of generators by type;
- for each generator, an expected average production value computed from actual data.


Sources
-------

The European model is based on [PanTaGruEl](https://github.com/laurentpagnier/PanTaGruEl.jl/), a model of the transmission grid of continental Europe, available as a Julia package developed by Laurent Pagnier.

PanTaGruEl can also be found in the [Zenodo data repository](https://zenodo.org/records/2642175), and it has been documented in the publications
- L. Pagnier,  P. Jacquod, *Inertia location and slow network modes determine disturbance propagation in large-scale power grids*, [PLoS ONE 14(3): e0213550 (2019)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0213550), and
- M. Tyloo, L. Pagnier, P. Jacquod, *The Key Player Problem in Complex Oscillator Networks and Electric Power Grids*, [Science Advances 5(11): eaaw8359 (2019)](https://www.science.org/doi/full/10.1126/sciadv.aaw8359) [arXiv:1810.09694](https://arxiv.org/abs/1810.09694).

The notebooks used to generate the model can be found in the [src](./src) directory.
