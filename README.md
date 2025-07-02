# variance-bandits-experiments

This repository contains simple simulations of the multi-armed bandit problem using UCB algorithms:
- **Known variance UCB**
- **Unknown variance UCB**

## What's Included
- Randomly generated means and variances
- Regret plots comparing known and unknown variance settings

## How to Customize
Change the constants at the top of the notebook:

```python
T = 100000          # Total number of rounds
K = 5               # Number of arms
mean_min = 0        # Minimum arm mean
mean_max = 1        # Maximum arm mean
variance_min = 1.1  # Minimum arm variance
variance_max = 5    # Maximum arm variance
