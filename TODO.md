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
- Fix eval_lenia_config_mem_optimized to reuse the update_func and compute_stats_func
- Optimize get_mem_optimized_inputs (init_cells)
- Test unroll parameter bigger than 1 in lax.scan
- Move center_world inside compute_stats_fn
- replace compress/decompress functions for init_cells with a duo pickle + relative_path
- Let's explore 1c-1k, 1c-2k, 1d-3k for now (looking for 10 species with interelation)

## Next
- Spend some time to get intimate with all parameters
- explore other kernels
- debug qd video dump (sometimes the number of iterations do not match the one found during the optimization, reproducibility issue)
- define a qd state
- save/load qd search states
- Make sure that we do not need multiple modes because of multiple kernels
- Computate statistics for all species already found to ge a sense of good behaviours statistics and their ranges/
- Define a 2D set of behaviours (mass_speed ?, etc.)
-  Go beyond perlin noise with simplex noise ? (https://www.bit-101.com/blog/2021/07/perlin-vs-simplex/)