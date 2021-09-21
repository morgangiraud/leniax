# Todos, ideas and notes 

## Notes

### Leniax's parameters
Criticial parameters force JAX to recompile the evaluation function, either because they change the memory allocation of the computation graph or the graph itself. Those are:
- world_size        (change cells shape)
- nb_channels       (change cells shape)
- nb_kernels        (change kernels shape)
- gf_id             (change computation graph)
- c_in              (change computation graph)
- c_out             (change computation graph)
- nb_init_search    (change cells shape)
- max_run_iter ?    (might change computation graph, depends on the behaviour of the scan function)

Half-critical parameters (those can be made non-criticial with some engineering) and those are non critical when using FFT):
- R
- r

On the other hand, non-critical parameters are parameters that can be changed without having to recompile anything. Those are the ones that can be searched over during one QD search:
- T
- b
- k_id
- q
- m
- s
- h

Rendering paremeters are unrelated with the search.

### Growth functions
Any growth functions that would boost the 0 value is doomed to fail. This would create a process that generates mass from an empty world.

### 1c2d1k worlds
- A random uniform initialisation convolved with a big enough kernel wil result in a neat constant field around the mean of the uniform distribution.

### Replication issues
There is a very strange behaviour that happens rarely but surely: it might happens that when running the vmap version of run_scan the check_heurisitics function returns different values N than when running each init separately.
It seems to occur mainly on the last element of the set.

### QD search
After doing an analysis on more than 4000 selected creatures, I concluded that the current best 2d behaviours statistics to select new creatures are the following:
- mass_density_mean on last 128 steps
- mass_volume_mean on last 128 steps

To handle them nicely in the QD search, it's good that the heuristics bound them

The next one is:
- mass_speed_mean

## Now
- Cleanup

## Maybe next?
- Explore other kernels
- Add heuristic for static species (no speed & no inertia change)
- For heuristics, add that each channel should be within mass bounds separately
- Move on to 3d behaviours? 
    - Dump cdf of all fitness, behaviours and genome
- define a qd state
- save/load qd search states
-  Go beyond perlin noise with simplex noise ? (https://www.bit-101.com/blog/2021/07/perlin-vs-simplex/)
