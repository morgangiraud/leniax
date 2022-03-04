# 016 - Differentiability

### Description
In this experiment, we explore how Lenia differentiability can be harnessed to learn its kernels from a dataset of an existing simulation.

### Hypothesis
The gradient signal at `t+1` is sufficient to learn the kernel.

### Results
It's a success using SGD + Adam. 

We take the opportunity to see how stable Lenia #0 is under its kernel approximation.

Following is the training log, with the 1-step average prediction errors of the simulation every 50 training steps:

| Training iteration | 1-step average prediction error |
| 0     |       11.114595 |
| 10    |      6.894103 |
| 20    |      4.105942 |
| 30    |      3.3381333 |
| 40    |      1.3709949  <--- First approximation leading to orbium stability |
| 50    |      0.5414693 |
| 100   |     0.09676437 |
| 150   |     0.027196145 |
| 200   |     0.014700023 |
| 250   |     0.0074412944 |
| 300   |     0.0035753988 |
| 350   |     0.0012906718 |
| 400   |     0.00088418333 |
| 450   |     0.0004377925 |
| 500   |     0.00021446226 |
| 550   |     8.197943e-05 |
| 600   |     5.5799173e-05 |
| 650   |     2.859323e-05 |
| 700   |     1.280467e-05 |
| 750   |     5.044243e-06 |
| 800   |     2.93275e-06 |
| 850   |     1.3505874e-06 |
| 900   |     5.074782e-07 |
| 950   |     2.1115437e-07 |
| 1000  |  1.0350308e-07 |

Interesting to see how Lenia #0 is robust in respect to its kernel. This result gives more ground to the fact that Lenia #0 is an attractor of its dynamic.