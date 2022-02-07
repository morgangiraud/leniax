# Overview

## Background: JAX

[JAX](https://github.com/google/jax) is NumPy + autodiff + GPU/TPU

It allows for fast scientific computing and machine learning with the normal NumPy API (+ additional APIs for special accelerator ops when needed).
## Leniax

[Leniax](https://github.com/morgangiraud/leniax) is a high-performance CA simulator library supporting variations like:
    - [Lenia](https://chakazul.github.io/lenia.html)
    - [Multi-neighbourhood CA](https://slackermanz.com/understanding-multiple-neighborhood-cellular-automata/)
    - [Neural CA](https://distill.pub/2020/growing-ca/)
    - Hopefully even more variations in the future

Leniax comes with everything you need to simulate, evolve and differentiate Cellular Automatata. 
It includes:

* **Evolution API** (`leniax.qd`): You can thousands of simulations in parallel and compute statistics to apply heuristics.

* **Differentiability**: Thanks to JAX, all the core components are differentiable making it easy to compute the gradients of any part of your CA.

* **Educational examples** See our examples.


## CPU/GPU/TPU support

All of our examples can run on CPU, GPU or TPU. 

Following is an example of TPU and GPU scripts to look for interesting initialization conditions:

* [Initialization search -  GPU](https://colab.research.google.com/drive/1vIOfp3zJa5GLYevE_W3eOS1BJ-FOfegd)
* [Initialization search -  TPU](https://colab.research.google.com/drive/1Kk1GXUGD6GiCdMy_fgL6LrwR2doDbXP_)