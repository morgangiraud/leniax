# Leniax

Leniax is a [Lenia](https://chakazul.github.io/lenia.html) simulation library powered by JAX.
It can efficiently simulate, render Lenia worlds and compute statistics.

It can also be used to search for creatures, initial conditions, kernels and growth functions using Quality-Diveristy algorithms and/or direct optimisation as the simulation engine is also fully differentiable.

## Installation

### From source
To install a version from source, clone the repo
```
git clone https://github.com/morgangiraud/leniax
cd leniax
```

Install Leniax library with conda (make sure you have it before typing the following command): `make install`

Then activate the environment: `conda activate leniax`

Finally, install the lib itself: `pip install .`

### Verification
You can make sure that everything is working fine by running the following command: `make ci`

## Usage

### Examples
We provide multiple commented examples that you can explore in the [examples](examples) folder. Its `README` contains more information

### Tutorials
You can also find multiple tutorials mirroring the examples as notebooks. Check the `Tutorials` section in the [documentation](https://leniax.github.com).

## Documentation
See here for the documentation: https://leniax.github.com

To serve the documentation locally, run the following commands:
```
pip install -r docs/requirements.txt
make docs
```
Then use your favorite browser to access the localhost url of the `docs/_build/html` folder.

## APIs

**Remarks**
This library intend to provide an optimized way to search for creatures either by using quality-diveristy algorithm or gradient based algorithm.

Those requires different tradeofs and that's why 2 APIs are built. Instead of craming everything into a single complicated API, we decided to make 2 simpler APIs. 

This also means, it is not straightforward to mix the 2 currently.

### Search API
The goal of the non-differentiable search API is to provide an optimized way to search among a lof of variations.
Lenia is considered a blackbox in this settings and you want to query the simulation as fast as possible for as many as possible inputs.

For more information about how to experiments and limitations check the [experiments](experiments) folder.

### Gradient API
The gradient API does not allow one to explore multiple solutions nor multiple initialization at the same time but it allows one yo compute the gradient related to any parameters including the initialization state.

## Research
To know more about past and present research, check the [experiments](experiments) folder and its `README`.

## About Lenia
In Lenia we have multiple things interacting with each other:
- the state composed of shape `[C, spatial_dims...]` where C is the number of channels.
- kernels used by convolutions
- Growth functions

A state contains `C > 0` channels and `D > 0` dimensions `(1D, 2D, ...)`

The update function use  `K > 0` kernels and growth functions. 

The original Lenia aims to be a generalisation of conway's game of life and so, is a 2d world with only 1 channel to which is applied 1 kernel.

It has been rapidly extended to the generic version (C>=1, D>=2, K>=1).

> This library aims to be fully generic. For now, it can handle any kind of simulations but can't render more than 3 dimensions. How to perceive/understand and or interact with more dimensions is still an open challenge.

For more information about Lenia, please check the [open science Lenia website](https://openlenia.github.io/).

## About Quality-diversity algorithms
As one can read on the main website about QD algorithms:

```
A fascinating aspect of natural evolution is its ability to
produce a diversity of organisms that are all high performing in
their niche.

Quality-Diversity optimization (or illumination) is a new type of
evolutionary algorithm that aims at generating large collections
of diverse solutions that are all high-performing. 
```

QD algorithms are an absolute perfect match to explore and discover stable patterns in the Lenia world.  We can consider stability as a measurement of quality and use any other Lenia statistics to ensure diversity.

Check the [website](https://quality-diversity.github.io/) for more information about those fantaastic algorithms!

## References
- [Differentiable Quality Diversity](https://arxiv.org/abs/2106.03894)
- [Quality-Diversity Optimization: a novel branch of stochastic optimization](https://arxiv.org/abs/2012.04322)
- [Lenia - Biology of Artificial Life](https://arxiv.org/abs/1812.05433)
- [Lenia and Expanded Universe](https://arxiv.org/abs/2005.03742)

# Citing Leniax
To cite this repository:
```
@software{leniax2022github,
  author = {Morgan Giraud},
  title = {{Leniax}: efficient and differentiable Lenia simulators},
  url = {http://github.com/morgangiraud/leniax},
  version = {1.0.0},
  year = {2022},
}
```

In the above bibtex entry, version number is intended to be that from leniax/setup.py, and the year corresponds to the project's open-source release.

# Contributing
Leniax has been built thanks to the [lenia research](https://lenia.world) initiative. 

To contribute you can directly interact with issues in this repository and come discuss with us in our [discord](https://discord.gg/4TFNnCkJta)