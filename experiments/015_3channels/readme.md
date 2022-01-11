# 015 - 3 channels

In this experiment we explore how the current QD algorithm works in the following settings
- 3 channels
- 9 kernels (kernels can influence eachothers)
- 4 kernels frozen to one of the children of orbium found in experiment 014_2channels

Goal: See if the current QD algorithm scale well to this setting

Results: ok

Notes:
Surprisingly it feels as if it is easier to find interesting and diverse creatures in this more complex setting.
To validate this inituition, I would need to run a raw search over the 9 kernels. It might be possible, that because some of the kernels are already frozen to an existing creature, it eases the search for more complex one.