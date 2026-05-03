from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from myAviary import MyAviary
import numpy as np


class RayMultiDroneEnv(MyAviary, MultiAgentEnv):
    def __init__(self, config):
        self.num_drones = config.get("num_drones", 3)

        # FIX: Gọi MyAviary.__init__ trực tiếp, không dùng super() dây chuyền
        # vì MultiAgentEnv.__init__ không nhận tham số
        MyAviary.__init__(
            self,
            num_drones=self.num_drones,
            gui=config.get("gui", False),
            windRandom=True
        )
        MultiAgentEnv.__init__(self)

        self._agent_ids = set([f"drone_{i}" for i in range(self.num_drones)])

        # FIX: observation_space và action_space phải là của 1 agent
        # Ray sẽ tự map cho tất cả agent qua shared policy
        obs_dim = 7  # pos(3) + vel(3) + battery(1)
        max_edges = self.num_drones ** 2

        # Observation space cho 1 agent: dict gồm nodes (toàn bộ graph) + edge_index
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_drones, obs_dim),
                dtype=np.float32
            ),
            "edge_index": spaces.Box(
                low=0, high=self.num_drones,
                shape=(2, max_edges),
                dtype=np.int64
            )
        })

        # Action space cho 1 agent
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs_raw, info = MyAviary.reset(self, seed=seed, options=options)
        # obs_raw shape: (num_drones, 7)
        graph_obs = self._build_graph_obs(obs_raw)
        # Trả về dict per-agent
        obs_dict = {f"drone_{i}": graph_obs for i in range(self.num_drones)}
        return obs_dict, {}

    def step(self, action_dict):
        # Ghép action từ dict về array (num_drones, 3)
        actions = np.array([action_dict[f"drone_{i}"] for i in range(self.num_drones)])

        obs_raw, reward, done, trunc, info = MyAviary.step(self, actions)
        # obs_raw shape: (num_drones, 7)

        graph_obs = self._build_graph_obs(obs_raw)
        obs_dict = {f"drone_{i}": graph_obs for i in range(self.num_drones)}

        rewards = {f"drone_{i}": float(reward) for i in range(self.num_drones)}

        done = bool(done)
        terminateds = {f"drone_{i}": done for i in range(self.num_drones)}
        terminateds["__all__"] = done

        truncateds = {f"drone_{i}": False for i in range(self.num_drones)}
        truncateds["__all__"] = False

        return obs_dict, rewards, terminateds, truncateds, {}

    def _build_graph_obs(self, obs_raw):
        """
        Xây dựng graph observation dùng chung cho tất cả agent.
        obs_raw: (num_drones, 7)
        Trả về dict {"nodes": ..., "edge_index": ...}
        """
        max_edges = self.num_drones ** 2

        # Tính edge_index dựa trên comm_radius
        adj = []
        for i in range(self.num_drones):
            for j in range(self.num_drones):
                if i != j:
                    dist = np.linalg.norm(obs_raw[i][:3] - obs_raw[j][:3])
                    if dist < self.comm_radius:
                        adj.append([i, j])

        if adj:
            edges = np.array(adj, dtype=np.int64).T  # shape (2, num_edges)
        else:
            edges = np.zeros((2, 0), dtype=np.int64)

        # Padding về max_edges để shape cố định
        padded_edges = np.zeros((2, max_edges), dtype=np.int64)
        n_edges = edges.shape[1]
        if n_edges > 0:
            padded_edges[:, :n_edges] = edges

        return {
            "nodes": obs_raw.astype(np.float32),      # (num_drones, 7)
            "edge_index": padded_edges                  # (2, max_edges)
        }