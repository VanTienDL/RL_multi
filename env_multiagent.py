from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from myAviary import MyAviary
import numpy as np

class RayMultiDroneEnv(MyAviary, MultiAgentEnv):
    def __init__(self, config):
        # 1. Thiết lập các thông số cơ bản trước
        self.num_drones = config.get("num_drones", 3)
        super().__init__(
            num_drones=self.num_drones,
            gui=config.get("gui", False),
            windRandom=True
        )
        self._agent_ids = [f"drone_{i}" for i in range(self.num_drones)]

        # 2. Định nghĩa lại Action Space cho Ray (phẳng)
        # Lấy từ MyAviary._actionSpace() lúc này đã là shape (3,)
        self.action_space = self._actionSpace() 

        # 3. Định nghĩa lại Observation Space cho Ray (dạng Dict cho GNN)
        max_edges = self.num_drones**2
        obs_dim = 3 + 3 + 1 # pos, vel, battery
        
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_drones, obs_dim), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=self.num_drones, shape=(2, max_edges), dtype=np.int64)
        })

    def reset(self, *, seed=None, options=None):
        # MyAviary.reset() trả về (obs, info)
        obs_raw, info = super().reset(seed=seed, options=options)
        return self._format_obs(obs_raw), {}

    def step(self, action_dict):
        # Chuyển action_dict từ Ray {'drone_0': [x,y,z], ...} 
        # thành mảng numpy [[x,y,z], [x,y,z], ...] để MyAviary hiểu
        actions = np.array([action_dict[aid] for aid in self._agent_ids])
        
        # Gọi step của lớp cha (MyAviary)
        obs_raw, reward, done, trunc, info = super().step(actions)
        
        # Format lại theo chuẩn Multi-Agent của Ray
        rewards = {aid: reward for aid in self._agent_ids}
        # Lưu ý: Ray RLlib yêu cầu 'terminateds' và 'truncateds' riêng
        terminateds = {aid: done for aid in self._agent_ids}
        terminateds["__all__"] = done
        
        truncateds = {aid: False for aid in self._agent_ids}
        truncateds["__all__"] = False
        
        return self._format_obs(obs_raw), rewards, terminateds, truncateds, info

    def _format_obs(self, obs_raw):
        # Tính toán edge_index động dựa trên khoảng cách comm_radius
        adj = []
        for i in range(self.num_drones):
            for j in range(self.num_drones):
                if i != j and np.linalg.norm(obs_raw[i][:3] - obs_raw[j][:3]) < self.comm_radius:
                    adj.append([i, j])
        
        edges = np.array(adj).T if adj else np.zeros((2, 0), dtype=np.int32)
        
        # Ray cần shape cố định, nên ta padding cạnh trắng
        max_edges = self.num_drones**2
        padded_edges = np.zeros((2, max_edges), dtype=np.int32)
        padded_edges[:, :edges.shape[1]] = edges

        return {
            f"drone_{i}": {
                "nodes": obs_raw.astype(np.float32), 
                "edge_index": padded_edges.astype(np.int64) # Torch Geometric thường dùng int64/Long cho index
            } 
            for i in range(self.num_drones)
        }