# Leniax
Leniax is a [Lenia](https://chakazul.github.io/lenia.html) simulation library powered by JAX.
It can efficiently simulate and render Lenia worlds and can also be used to search for creatures, initial conditions, kernels and growth functions.
It is also fully differentiable!

## Install
Install Leniax library with conda (make sure you have it before typing the following command): `make install`
**Currently, only an OSX environment is provided**

Then activate the environment: `conda activate leniax`
Now, install the lib itself in the conda env: `pip install -e .`

Finally, make sure, everything is fine by running the following command: `make ci`

## About Lenia
In Lenia we have multiple things interacting with each other:
- the world
- kernels
- Growth functions

A world contains one or more channels (C), one or more dimensions (1D, 2D, 3D -> D), and one or more kernel functions (K). The original Lenia is a 2d world with only one channel and any number of functions (C=1, D=2, K>=1)  but it has been rapidly extended to the generic version (C>=1, D>=2, K>=1).

This library aims to be fully generic. For now, it can handle any number of channels and functions but is limited to 2d worlds. It also supports the *asymptotic update* variation.

## APIs

**Remarks**
This library intend to provide an optimized way to search for creatures either by using quality-diveristy algorithm or gradient based algorithm.
Those requires different tradeofs and that's why 2 APIs are built. Instead of craming everything into a single complicated API, I decided to make 2 simpler APIs. 
This also means, it is not straightforward to mix the 2.

### Search API
The goal of the non-differentiable search API is to provide an optimized way to search among a lof of variations.
Lenia is considered a blackbox in this settings and you want to query the simulation as fast as possible for as many as possible inputs.

Currently one can search over a set of configurations (as big as memory allows) for which multiple initializations can be tested. Because Lenia is a chaotic system, given a fixed world, one can only sample some initial conditions to see if a stable pattern can emerge.

This introduces limitations, all configurations mush share:
- number of world dimensions (2d, 3d, etc...)
- Size of each dimensions
- Number of channels
- Number of initializations tested
- Number of kernels (and so, number of growth functions)
- Shape of the growth functions parameters

### Gradient API
The gradient API does not allow one to explore multiple solutions nor multiple initialization at the same time.

### What search algorith mare you using
I'm using the family of Quality-Diversity algorithm to search for species. More precisely, I'm mainly using the CMA-ME algorithm with multiple emitters.

I believe those are particularly adapted to search for Lenia creatures as we can search over some *genetic* parameters and store high-potential creatures using some *behavioral* parameters.

The only drawback is the fitness function used to rank results. I'm currently searching for stable local patterns (SLPs), but SLP is not a mathematical definition. So far, I can only approximate what is potentially an SLP thanks to heuristics.

To give an idea of the current search capacity:
- 500k element searched
- 4k potential creatures (as defined by the heuristics)
- ~1000 SLPs
- ~202 unique creatures after classification

### I want to know more
I'm glad you do! Check the issues to see what we are working on and join our community: https://discord.gg/4TFNnCkJta


## Performances
### About memory
If you want to run very long simulations and compute its statistics, you should the runner function `run_scan_mem_optimized`. It doesn't store intermediary data but compute statistics along the way which allows one to have as many iterations as one want.

*Note: about QD*
QD algorithms require to compute the statistics of multiple candidate solutions at the same time. In our case, we also need to compute multipe initial state to have a good-enough approximation of the quality of candidate.
The total memory necessary is not easily computable so we provide an helper function to get an estimate of the required memory `leniax.utils.get_needed_memory`. 


## Research

### Experiments and tooling
To create an experiments, explore the results and create a collection out of it, follow those steps:
- Create a folder named 'XXX_my_exp' in the experiments folder and a add a config file
- Run the experiment
- Use the script `python scripts/dump_best.py` to dump all the best solutions
- Use the script `python tools/prepare_exp_folder.py` to gather all vizualisation data
- Launch the webserver to classify `python ui.server.py` (go to http://127.0.0.1:5000/classifier.html)
- Switch the comments in the script `python tools/prepare_exp_folder.py` and run it again to have your categories folder
- Use the originals folder to create your collection
- Call the script `python scripts/render_collection.py` to prepare all the data necessary for a collection
- If needed, update the metadata manually to suit your needs

## Tools
### The classifier


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