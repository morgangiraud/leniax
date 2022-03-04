# 019 - Neural Lenia

### Description
In this experiment, we try to replicate the Neural CA growing task with the gecko using the Lenia variation.

Settings:
- 16 channels 
- RGBA channels + 12 hidden channels
- Kernels of size 10
- Stochastic update, rate 0f 50%
- Alive mask using the alpha channel (alpha > 0.1 -> alive)
- We use a 2-layers NN to parametrize the continuous function
- the update `dstate` is computed directly by the continuous function `fixed step size = 1`

### Hypothesis
The gecko experiment can be replicated with non-local kernels.

### Results
True.

The learned 32 kernels are interesting. It looks like the first 6th which are applied to the RGB channels are close to being some weighted identity kernels.
The 7th and 8th which are applied to the alpha channel are do also look like some weighted identity kernels but of opposite signs,

Other kernels are less clearly defined but some seem to encode a directional gradient. For example: 13, 15, 27, 31.
