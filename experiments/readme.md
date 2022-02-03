# Research
Leniax is used extensively to search for stable local patterns in Lenia.

## Limitations

### The evaluation function
Because we have yet to found a formula to define "interestingness" in Lenia nor we have a dataset to learn one. We must resort to heuristics to evaluate if a simulation contains something interesting and in our case, a Stable Local Pattern (SLP).

But because Lenia is a chaotic system, it is not simple to determine if a configuration contains one or more SLPs. Or, if we want QD to be efficient we need a robust evaluation function.

One way to do it is to sample multiple well-chosen initializations per confguration. The more you sample, the more you increase the odds of finding potential stable patterns.

Currently, we consider that sampling Perlin noise at different level of noise is a decent heuristics.

### Configurations
To be able to search efficiently over configurations, a set of configuration must share some properties. 

The way JAX works is by pre-allocating a computation graph with its memory on selected devices so you can push your data later at full speed. To avoid slow graph recompilation, one has to keep the graph and tensors shape contant.

This introduces limitations, all configurations mush share:
- Number of channels
- number of world dimensions (2d, 3d, etc...)
- Size of each dimensions
- Number of initializations
- Number of kernels (and so, number of growth functions)
- Shape of the kernels parameters (can be padded if needed)
- Shape of the growth functions parameters (can be padded if needed)

### Search capacity
To give an idea of the current search capacity:
- 5 days spent running the algorithm on Google Collab
- 500k element searched
- 4k potential creatures (as defined by the heuristics)
- ~1000 SLPs discovered
- ~202 unique and interesting creatures gathered after manual curation.

## Experiments and tooling
If you want to contrivute to research, please create an experiments as follow:
- Create a folder named 'XXX_my_exp' in the experiments folder, add a `config.yaml` file and a `README` with  the description of your experiment
- Run the experiment
- Update the `README` with your findings and make a PR.

In the case of a QD search, you can use some raw tools built to explore its results:
- Use the script `python examples/dump_best.py` to dump all the best solutions
- Use the script `python tools/prepare_exp_folder.py` to gather all vizualisation data
- Launch the webserver to classify `python ui.server.py` (go to http://127.0.0.1:5000/classifier.html)
- Switch the comments in the script `python tools/prepare_exp_folder.py` and run it again to have your categories folder
- Use the originals folder to create your collection
- Call the script `python examples/render_collection.py` to prepare all the data necessary for a collection
- If needed, update the metadata manually to suit your needs

## Research notes
Following is some raw and random research notes, I've gathered along the way. Organized in no particular order.

### Leniax's parameters
Criticial parameters force JAX to recompile the evaluation function, either because they change the memory allocation of the computation graph or the graph itself. Those are:
- world_size        (change cells shape)
- nb_channels       (change cells shape)
- nb_kernels        (change kernels shape)
- gf_slug             (change computation graph)
- c_in              (change computation graph)
- c_out             (change computation graph)
- nb_init_search    (change cells shape)
- max_run_iter      (might change computation graph, depends on the behaviour of the scan function)

Half-critical parameters (those can be made non-criticial with some engineering) and those are non critical when using FFT):
- R
- r

On the other hand, non-critical parameters are parameters that can be changed without having to recompile anything. Those are the ones that can be searched over efficiently during a QD search:
- T
- b
- k_params
- kf_params
- gf_params

Rendering paremeters are unrelated with search performance.

### About growth functions
- Any growth functions that would boost the 0 value is doomed to fail. This would create a process that generates mass from an empty world.

### About initializations
- A random uniform initialisation convolved with a big enough kernel wil result in a neat constant field around the mean of the uniform distribution.

### Replication issues
- There is a very strange behaviour that happens rarely but surely: it might happens that when running the vmap version of run_scan the check_heurisitics function returns different values N than when running each init separately. It seems to occur mainly on the last element of the set.

## About QD search
After plotting all possible statistics of more than 4000 selected creatures, we observed that the current best 2d behaviours spaces is to gather diversity on the following statistics:
- Simulation running for more than 1024 timesteps
- Using the last 128 timesteps to compute average statistics
- Using `mass_density_mean`
- Using `mass_volume_mean`

The third interesting statistic is `mass_speed_mean`.

