## MACNS: A Generic Graph Neural Network Integrated Deep Reinforcement Learning based Multi-Agent Collaborative Navigation System for Dynamic Trajectory Planning

Code and data used in the paper MACNS, DOI: https://doi.org/10.1016/j.inffus.2024.102250

## Code

In `gppo` directory.

The code is based on (with inheritance of) [`stablebaselines3`](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html), the main documents/usage can be found on their pages.

### Usage

Please install libraries in `requirements.txt` before running. Please **Exactly** match the version described in this file, as the higher version may report errors.

#### Run Example
1. clone the repository
2. run `examples.py`

#### Run Your Agent

1. clone the repository
2. copy `gppo/sb3_extend_gppo` directory and `gppo/examples.py` to your project directory
3. modify parameters in `examples.py`, including how to build your graph function `create_torch_graph_data`, GPPO parameters `policy_kwargs`, and define your environment `env`
4. run `examples.py`

### Data

In `data` directory.
