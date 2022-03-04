# 012 - Orbium, k2, Beta cube 4

### Decription
In this experiment, we are fixing the Orbium kernel and searching over a second kernel.

Variations consist of looking at what we can call the beta cube. 
We will iterate over all possible trio of beta values by increasing values by `1/4`.

The idea is to find a way to search more complex environment without increasing too much the number of variables.
We can do that by hierarchically increasing the complexity.

The setting is:
- The original Lenia update function
- 1 channel
- 2 kernels
- Different beta values from [1.] t0 [1., 1., 1.]

We are looking at the following variables:
- k0 - h
- k1 - h
- k1 - m
- k1 - s
- k1 - r

Behaviors:
- Mass speed
- Mass density

### Hypothesis
New interesting solitons can emerge by adding a kernel to an existing configuration.

### Results
Wrong.

We've done quite extensive search and honestly, there is not much new diversity. 
I wonder if adding kernels in a 1-channel world can allow for a new kind of soliton.