import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv, GATConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.pool import global_add_pool
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import torch
from typing import Tuple, Optional


class MaskableActorCriticPolicyWithGNN(MaskableActorCriticPolicy):
    def __init__(self, *args, state_space=None, graph_fn=None, **kwargs):
        self.graph_fn = graph_fn
        self.state_space = state_space
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GNNNetwork(self.features_dim, graph_fn=self.graph_fn, device='cuda')

    def evaluate_actions(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = torch.cat(
                [self.mlp_extractor.forward_actor(torch.reshape(each_pi_features, (-1, self.state_space)))
                 for each_pi_features in pi_features])
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


class ActorCriticPolicyWithGNN(ActorCriticPolicy):
    def __init__(self, *args, state_space=None, graph_fn=None, **kwargs):
        self.graph_fn = graph_fn
        self.state_space = state_space
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GNNNetwork(self.features_dim, graph_fn=self.graph_fn, device='cuda')

    def evaluate_actions(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = torch.cat(
                [self.mlp_extractor.forward_actor(torch.reshape(each_pi_features, (-1, self.state_space)))
                 for each_pi_features in pi_features])
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


class GNNNetwork(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 32,
            last_layer_dim_vf: int = 64,
            hidden_layer_dim: int = 128,
            graph_fn=None,
            device='cuda'
    ):
        super().__init__()

        self.graph_fn = graph_fn
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            GNN(hidden_layer_dim, last_layer_dim_pi, graph_fn=graph_fn, device=device)
        ).to(device)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(64, last_layer_dim_vf), nn.ReLU(),
        ).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class GNN(nn.Module):
    def __init__(self, hidden_dim, action_space, graph_fn=None, device="cuda"):
        super(GNN, self).__init__()
        assert graph_fn is not None
        self.graph_fn = graph_fn
        self.device = device
        self.conv1 = GATConv(1, hidden_dim)
        # self.conv1 = GCNConv(number_players, action_space)
        self.conv2 = GATConv(hidden_dim, action_space)
        # self.conv2 = GCNConv(hidden_dim, action_space)
        self.linear = nn.Linear(hidden_dim, action_space)
        self.linear2 = nn.Linear(hidden_dim, action_space)
        self.layer_norm = LayerNorm(action_space)

    def forward(self, state):
        data = self.graph_fn(state)
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        x = nn.functional.relu(self.conv1(x, edge_index))
        x = nn.functional.relu(self.conv2(x, edge_index))
        # print(x)
        # print(x.shape)
        # print(self.layer_norm(x))
        # sys.exit()
        x = global_add_pool(self.layer_norm(x), torch.LongTensor([0 for _ in range(data.feature_size)]).to(self.device))
        # x = nn.functional.relu(self.linear(x))
        # x = self.linear(x)
        return x
