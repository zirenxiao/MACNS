import os
import shutil
import sys

import gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from torch_geometric.data import Data
import torch
from stable_baselines3.common.logger import configure

from gppo.sb3_extend_gppo.ActorCriticPolicyWithGNN import MaskableActorCriticPolicyWithGNN, \
    ActorCriticPolicyWithGNN


def create_torch_graph_data(s: torch.Tensor, num_players=None):
    """
    Transform state into graph data point
    :param num_players:
    :param s: a state from the environment
    :return: a data point with node features, edge indexes and number of features each node
    """
    ei = [(0, 1)]
    ei = torch.tensor(ei, dtype=torch.long)
    # node_feature = torch.reshape(s[:, [0, 1, 2, 0, 3, 4]], (-1, 3))
    # node_feature = torch.reshape(s[:, [0, 1, 2, 3, 0, 4, 5, 6]], (-1, 4))
    node_feature = torch.reshape(s[:, [0, 0]], (-1, 1))
    data = Data(x=node_feature, edge_index=ei.clone().t().contiguous(), feature_size=node_feature.size()[0])
    return data




if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = 'logs/'

    device = 'cuda'

    env = gym.make("CliffWalking-v0")

    shutil.copyfile(os.path.basename(__file__), f'{log_path}{os.path.basename(__file__)}')
    print(f"backup current script to {log_path}")

    goal_episodes = 1000
    # env = gym.make("CartPole-v1")
    policy_kwargs = dict(share_features_extractor=False,
                         # state_space=env.observation_space.shape[0],
                         state_space=1,
                         graph_fn=create_torch_graph_data,
                         )

    # agent = MaskablePPO(MaskableActorCriticPolicyWithGNN, env, verbose=2, policy_kwargs=policy_kwargs,
    #                     device=device, batch_size=64)

    agent = PPO(ActorCriticPolicyWithGNN, env, verbose=2, policy_kwargs=policy_kwargs,
                device=device, batch_size=64)

    agent.set_logger(configure(log_path, ["stdout", "csv"]))
    # output explanation: https://stable-baselines3.readthedocs.io/en/master/common/logger.html
    try:
        agent.learn(total_timesteps=goal_episodes * 2048)
    except KeyboardInterrupt:
        print("detected keyboard interrupt...saving model...")
    agent.save(f"{log_path}model")
