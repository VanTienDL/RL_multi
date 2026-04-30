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

        # Cấu hình: 7 features đầu vào (pos, vel, bat)
        in_channels = 7
        hidden_dim = 64

        # Tầng GNN: Học cách giao tiếp giữa các drone
        self.conv1 = GATConv(in_channels, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False)

        # Policy Head: Quyết định hành động
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs) 
        )

        # Value Head: Đánh giá trạng thái (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        # 1. Lấy dữ liệu
        obs = input_dict["obs"]
        nodes = obs["nodes"]           # (Batch, Num_Drones, 7)
        edge_index = obs["edge_index"] # (Batch, 2, Max_Edges)

        batch_size = nodes.shape[0]
        num_drones = nodes.shape[1]

        # 2. Xử lý ép kiểu quan trọng nhất ở đây
        # Chuyển edge_index sang torch.long (int64) để GATConv không báo lỗi
        # Sử dụng .contiguous() để đảm bảo bộ nhớ liên tục cho Ray
        edge_index = edge_index.long().contiguous() 

        out_actions = []
        out_values = []

        # 3. Message Passing
        # Tớ vẫn giữ vòng lặp for của cậu để khớp với logic xử lý batch hiện tại
        for b in range(batch_size):
            x = nodes[b]      # (Num_Drones, 7)
            edges = edge_index[b] # (2, Max_Edges)
            
            # GNN layers
            h = torch.relu(self.conv1(x, edges))
            h = torch.relu(self.conv2(h, edges))
            
            out_actions.append(self.action_head(h))
            out_values.append(self.value_head(h))

        # 4. Gom kết quả về dạng RLlib mong muốn
        action_out = torch.stack(out_actions).view(batch_size * num_drones, -1)
        values = torch.stack(out_values) # Shape: (Batch_size, Num_Drones, 1)
        
        # Lấy trung bình giá trị của các drone trong một batch 
        # để trả về 1 giá trị duy nhất cho mỗi step (Batch_size,)
        self._last_value = torch.mean(values, dim=1).squeeze(-1) 

        return action_out, state

    def value_function(self):
        return self._last_value