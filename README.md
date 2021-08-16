# Leniax
Lenia JAX library


## Install
`make install`

## About
In lenia we have multiple things interacting with each other:
- the world
- kernels
- Growth functions

The world is composed of a certain number of dimensions (1D, 2D, 3D) and somes channels. The original Lenia is a 2d world with only one channel. 

The more general case allows for a N-dimensional world with C channels on which K kernels are used for the update.

Bert Chan expose different properties emerging from those characteristics:
- Multi-channel: division of labor + polymorphism
- Multi-kernel: Individuality (self-containment, self-defense) + self-replication
- Multi-dimensionnal: 3D pyshiology, etc 

### More on each element
For the world:
- We have the number of dimension
- We have the total size of the world (power of 2 for simplicity)
We fake an infinite world by wrapping borders on each other (important 
for convolution padding)
- We have the number of channels

For the kernels
- We have the number of modes
- We have the kernel function
- We have the number of kernels
- We have the growth function

For fields:
- Fields outputs should be between 1 and -1, it decreases/increases value in the cells.