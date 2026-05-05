"""
model_gnn.py — GNN Policy cho High-level CTDE (dùng với Ray RLlib PPO).

Kiến trúc:
  Input  : nodes (B, N, 7) + edge_index (B, 2, N²) + agent_idx (B, 1)
  GNN    : 2 lớp GATConv (Graph Attention Network)
  Output : action_out (B, 3) — delta_xyz cho drone agent_idx
           value_out  (B,)   — giá trị trạng thái toàn đồ thị (centralized critic)

CTDE đạt được vì:
  - Critic (value_head) nhìn toàn bộ đồ thị → centralized training
  - Actor (action_head) chỉ dùng embedding của node agent_idx → decentralized execution
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch_geometric.nn import GATConv


class HighLevelGNNModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_channels = 7           # pos(3) + vel(3) + bat(1)
        hidden_dim  = 64
        heads       = 4

        # ── GNN layers (shared — dùng cho cả actor lẫn critic) ──
        self.conv1 = GATConv(in_channels,       hidden_dim, heads=heads, concat=True,  dropout=0.1)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,    concat=False, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ── Actor head: chỉ dùng embedding của 1 node (decentralized) ──
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),  # num_outputs = 3 (delta_xyz)
        )

        # ── Critic head: dùng MEAN embedding toàn đồ thị (centralized) ──
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        obs        = input_dict["obs"]
        nodes      = obs["nodes"]       # (B, N, 7)
        edge_index = obs["edge_index"]  # (B, 2, N²)
        agent_idx  = obs["agent_idx"]   # (B, 1)

        batch_size = nodes.shape[0]
        edge_index = edge_index.long().contiguous()

        action_outs = []
        value_outs  = []

        for b in range(batch_size):
            x     = nodes[b]         # (N, 7)
            edges = edge_index[b]    # (2, N²)
            idx   = int(agent_idx[b, 0].item())

            # Lọc padding edges (src == dst là self-loop padding)
            mask  = edges[0] != edges[1]
            valid = edges[:, mask]

            if valid.shape[1] == 0:
                # Fallback: self-loop để GATConv không crash
                n = x.shape[0]
                loop = torch.arange(n, device=x.device)
                valid = torch.stack([loop, loop])

            # ── Message passing ─────────────────────────────────
            h = self.conv1(x, valid)
            h = torch.relu(self.norm1(h))
            h = self.conv2(h, valid)
            h = torch.relu(self.norm2(h))
            # h shape: (N, hidden_dim)

            # ── Actor: embedding của drone idx ──────────────────
            # Decentralized: chỉ lấy node của agent này
            action_outs.append(self.action_head(h[idx]))  # (num_outputs,)

            # ── Critic: mean toàn đồ thị ────────────────────────
            # Centralized: tổng hợp thông tin toàn bộ swarm
            global_h = h.mean(dim=0)  # (hidden_dim,)
            value_outs.append(self.value_head(global_h))  # (1,)

        action_out = torch.stack(action_outs)            # (B, num_outputs)
        self._last_value = torch.stack(value_outs).squeeze(-1)  # (B,)

        return action_out, state

    def value_function(self):
        return self._last_value