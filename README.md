# Leniax
A [Lenia](https://chakazul.github.io/lenia.html) library powered by JAX.
It also provides a set of tools to search, classify and organize creatures.

```
Warning: The current library is in beta. It's not yet fully battle-tested!
```

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

### What search algorith mare you using
I'm using the family of Quality-Diversity algorithm to search for species. More precisely, I'm mainly using the CMA-ME algorithm with multiple emitters.

I believe those are particularly adapted to search for Lenia creatures as we can search over some *genetic* parameters and store high-potential creatures using some *behavioral* parameters.

The only drawback is the fitness function used to rank results. I'm currently searching for stable local patterns (SLPs), but SLP is not a mathematical definition. So far, I can only approximate what is potentially an SLP thanks to heuristics.

To give an idea of the current search capacity:
- 500k element searched
- 4k potential creatures (as defined by the heuristics)
- ~1000 SLPs
- ~200 unique creatures after classification


### I want to know more
Look into the TODO file to follow the research and check issues associated with this repository.

## Research

### Experiments and tooling
To create an experiments, explore the results and create a collection out of it, follow those steps:
- Create a folder named 'XXX_my_exp' in the experiments folder and a add a config file
- Run the experiment 
- Use the script `python ui/prepare_exp_folder.py` to gather all vizualisation data
- Launch the webserver to classify `python ui.server.py` (go to http://127.0.0.1:5000/classifier.html)
- Switch the comments in the script `python ui/prepare_exp_folder.py` and run it again to have your categories folder
- Use the originals folder to create your collection
- Call the script `python scripts/render_collection.py` to prepare all the data necessary for a collection
- If needed, update the metadata manually to suit your needs