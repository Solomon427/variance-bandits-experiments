# variance-bandits-experiments

This repository contains simple simulations of the multi-armed bandit problem using algorithms:
- **VarUCB-Known**
- **VarUCB-Unknown**
- **UCB**

## Current version: experiment3.py

## What's Included
- Randomly generated means and variances
- Regret plots comparing variance settings

## How to Customize
Change the constants:

```python
T = 1000000         # Total number of rounds
K = 10              # Number of arms
mean_min = 0        # Minimum arm mean
mean_max = 1        # Maximum arm mean
variance_min = 1  # Minimum arm variance
variance_max = 5    # Maximum arm variance
