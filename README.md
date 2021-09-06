# Leniax
A Lenia library powered by JAX

## Install
Install Leniax library with conda (make sure you have it before typing the following command): `make install`
**Currently, only an OSX environment is provided**

Then activate the environment: `conda activate leniax`
Now, install the lib itself in the conda env: `pip install -e .`

Finally, make sure, everything is fine by running the following command: `make ci`

## About
In Lenia we have multiple things interacting with each other:
- the world
- kernels
- Growth functions

A world contains one or more channels (C), one or more dimensions (1D, 2D, 3D -> D), and one or more kernel functions (K). The original Lenia is a 2d world with only one channel and any number of functions (C=1, D=2, K>=1)  but it has been rapidly extended to the generic version (C>=1, D>=2, K>=1).

This library aims to be fully generic. For now, it can handle any number of channels and functions but is limited to 2d worlds. It also supports the *asymptotic update* variation.


### I want to know more
Look into the TODO file to follow the research and check issues associated with this repository.

