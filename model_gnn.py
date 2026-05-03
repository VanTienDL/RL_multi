import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch_geometric.nn import GATConv


class DroneGNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 7 features: pos(3) + vel(3) + battery(1)
        in_channels = 7
        hidden_dim = 64

        self.conv1 = GATConv(in_channels, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False)

        # Policy head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        nodes = obs["nodes"]           # (batch, num_drones, 7)
        edge_index = obs["edge_index"] # (batch, 2, max_edges)

        batch_size = nodes.shape[0]
        num_drones = nodes.shape[1]

        # FIX: edge_index phải là long (int64) và contiguous
        edge_index = edge_index.long().contiguous()

        out_actions = []
        out_values = []

        for b in range(batch_size):
            x = nodes[b]        # (num_drones, 7)
            edges = edge_index[b]  # (2, max_edges)

            # Lọc bỏ padding edges (cả src và dst đều = 0 là padding)
            # Để tránh self-loop giả từ padding, chỉ giữ edges có src != dst
            mask = edges[0] != edges[1]
            valid_edges = edges[:, mask]  # (2, num_valid_edges)

            # Nếu không có edge nào hợp lệ, dùng self-loop để GATConv không bị lỗi
            if valid_edges.shape[1] == 0:
                self_loops = torch.arange(num_drones, device=x.device)
                valid_edges = torch.stack([self_loops, self_loops], dim=0)

            h = torch.relu(self.conv1(x, valid_edges))
            h = torch.relu(self.conv2(h, valid_edges))

            # FIX: Mỗi agent là 1 sample trong batch của RLlib
            # → lấy embedding của TẤT CẢ drone, stack lại
            out_actions.append(self.action_head(h))   # (num_drones, num_outputs)
            out_values.append(self.value_head(h))     # (num_drones, 1)

        # FIX: RLlib gọi forward 1 lần per agent per batch
        # batch_size ở đây = số agent * số env steps
        # nodes shape (batch, num_drones, 7) vì env trả về toàn bộ graph cho mỗi agent
        # Ta cần trả về (batch, num_outputs) — dùng embedding của drone tương ứng
        # Nhưng vì shared policy, tất cả agent đều dùng chung model này
        # và Ray sẽ gọi forward riêng cho từng agent → batch_size=N, num_drones=3
        # → ta lấy mean embedding của các drone trong graph làm đại diện
        # (hoặc có thể lấy embedding drone 0 nếu muốn local policy)

        # Stack: (batch, num_drones, num_outputs)
        action_out = torch.stack(out_actions)   # (batch, num_drones, num_outputs)
        values_out = torch.stack(out_values)     # (batch, num_drones, 1)

        # FIX: Lấy MEAN qua tất cả drone → (batch, num_outputs) và (batch,)
        # Đây là centralized approach — mọi agent trong graph đều tạo ra cùng output
        action_mean = action_out.mean(dim=1)     # (batch, num_outputs)
        self._last_value = values_out.mean(dim=1).squeeze(-1)  # (batch,)

        return action_mean, state

    def value_function(self):
        return self._last_value