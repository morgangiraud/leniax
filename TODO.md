# TODO 

## Parameters
Criticial parameters are parameters that forces JAX to recompile the evaluation function, either because they change the memory allocation of the graph or they change the computation graph itself. Those are:
- world_size        (change cells shape)
- nb_channels       (change cells shape)
- nb_kernels        (change kernels shape)
- gf_id             (change computation graph)
- c_in              (change computation graph)
- c_out             (change computation graph)
- nb_init_search    (change cells shape)
- max_run_iter ?    (might change computation graph, depends on the behaviour of the scan function)

Half-critical parameters (those can be made non-criticial with some engineering):
- R
- r

Non-critical parameters:
- T
- b
- k_id
- q
- m
- s
- h

Rendering paremeters are unrelated with the search.

### Remarks
Any growth functions that would boost the 0 value is doomed to fail. This would create a process that generates mass from an empty world.

##### 1c1k notes
- A random uniform initialisation convolved with a big enough kernel wil result in a neat constant field around the mean of the uniform distribution.

##### Replication note
There is a very strange behaviour that happens rarely but surely: it might happens that when running the vmap version of run_scan the check_heurisitics function returns different values N than when running each init separately.
It seems to occur mainly on the last element of the set.

## Now
- On failed search, test altering T
- Add heuristic for static species (no speed, no inertia change)

# For the launch
- We fix ourselves to D=2, C=1, nK=1 R=13, T= 10, gf_id=0, k_id=0, q=4
- We are looking at the beta cube with the fraction 1/4 which leads to 61 (5 * 5 + 4 * 5 + 4 * 4) cases
- For the launch we need one side (25)
- The goal is to provide the community to choose whiche piece of the world to explore next (be part of cutting edge AI research, Each layer necessits more compute power)

Hopefull goal at the end: Find 1337 species!

## Maybe next?
- For heuristics, add that each channel should be within mass bounds
- Move on to 3d behaviours? 
    - Dump cdf of all fitness, behaviours and genome
- Let's explore with T and R
- Let's explore 1c-2k, 1c-3k for now (looking for 10 species with interelation)
- Spend some time to get intimate with all parameters
- explore other kernels
- define a qd state
- save/load qd search states
-  Go beyond perlin noise with simplex noise ? (https://www.bit-101.com/blog/2021/07/perlin-vs-simplex/)
