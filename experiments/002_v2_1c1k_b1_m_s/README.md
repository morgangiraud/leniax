# 002 - v2, 1c1k, b1, m, s

### Decription
In this experiment, we will iterate until the QD algorithm succeeds at finding solitons on Lenia automatically.
The setting is:
- The asymptotic Lenia update function
- 1 channel
- 1 kernel
- beta values: [1]

We are only looking at the following variables:
- m
- s

### Hypothesis
QD algorithms can solve this problem.

### Results
It works but much less efficiently.

It finds only 5 potential stable solitons.

One possible explanation:
- My current QD algorithm used heuristics that have been tailored to the original update function
- Those heuristics limit the generalization capacities of the QD algorithm.

An obvious possible way of moving forward would be to reduce the specificities of those heuristics but I don't see a solution for now.