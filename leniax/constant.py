"""Leniax constants

Those constants are used at different places in the code to ensure consistency and sstability.
"""

# Epsilon value used to avoid division by 0
EPSILON = 1e-7
# Define a threshold at which we start to apply our statistic heuristics
START_CHECK_STOP = 10
# The number of steps plotted in the truncated graph
NB_STATS_STEPS = 128
# Number of characters used of the custom encoding scheme
NB_CHARS = (ord('Z') - ord('A')) + (ord('z') - ord('a')) + (ord('þ') - ord('À'))
