# TODO 

## Parameters
Criticial parameters are parameters that forces JAX to recompile the evaluation function, either because they change the memory allocation of the graph or they change the computation graph itself. Those are:
- world_size
- nb_channels
- nb_kernels
- gf_id
- c_in
- c_out
- nb_init_search
- max_run_iter ? (no sure yet, depends on the behaviour of the scan function)

Half-critical parameters (those can be made non-criticial with some engineering):
- K_id
- r

Non-critical parameters:
- R
- T
- b
- q
- m
- s
- h

Rendering paremeters are unrelated with the search.

### Remarks
I made a surprising progress on the init search reducing the need to ~128 initializations.
-> Delay the need to explore CPPN or gradient based init search

The progress on JAX opens up some nice potential. I can parallelize initializations and parameters as long as they all shared the same critical parameters.
Current perf on Collab with a t4 nvidia gpu
- 1 function call
- 32 parameters set
- 128 initialization
- 1024 iterations
-> compiled time 110s (25ms per run)
and
- 4 function call
- 8 parameters set
- 128 initialization
- 1024 iterations
-> compiled time
It might be interesting to spot the perfect

Any growth functions that would boost the 0 value is doomed to fail. This would create a process that generates mass from an empty world.

##### 1c1k notes
- A random uniform initialisation convolved with a big enough kernel wil result in a neat constant field around the mean of the uniform distribution.

##### Replication note
There is a very strange behaviour that happens rarely but surely: it might happens that when running the vmap version of run_scan the check_heurisitics function returns different values N than when running each init separately.
It seems to occur mainly on the last element of the set.

## Now
- Fix the progress bar for collab
- Define a 2D set of behaviours (mass_speed ?, etc.)
- Dump species (center world + crop of last frame)
- Dump cdf of all fitness, behaviours and genome
- Do validation number 3
- Let's explore 1c-1k, 1c-2k, 1c-3k for now (looking for 10 species with interelation)
- Let's explore with T and R
- Let's expplore with different bs

Validation:
1. a big qd_search (OK!)
2. a big qd_search with the new update rule (OK! need a more focused search)
3. a big qd search using real behaviours

## Next
- Spend some time to get intimate with all parameters
- explore other kernels
- define a qd state
- save/load qd search states
-  Go beyond perlin noise with simplex noise ? (https://www.bit-101.com/blog/2021/07/perlin-vs-simplex/)

## To be able to launch
Find 1337 species!

# Listo of experiments
for bs
- [1] Done

- [0.5, 1] Done
- [1, 0,5]
- [1, 1]

- [0.25, 0.5, 1]
- [1, 0.5, 0.25]
- [0.5, 1, 0.5]
- [1, 1, 1]

- [1, 0, 0.5]
- [0.5, 0, 1]
- [1, 0, 1]