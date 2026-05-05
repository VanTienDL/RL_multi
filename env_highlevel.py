"""
env_highlevel.py — High-level GNN Env (CTDE).

Thay đổi so với v1:
  - Drone về sạc QUAY LẠI đội hình sau khi đầy pin
  - High-level cập nhật rally_points liên tục để vá topo khi drone vắng mặt
  - Obs thêm drone_state (8 features thay vì 7)
  - Reward tích hợp logic "vá lỗ hổng topology" khi drone về sạc
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from myAviary import MyAviary, COMM_RADIUS, LOW_BATTERY, DroneState

# ──────────────────────────────────────────
# Siêu tham số
# ──────────────────────────────────────────
MAX_DELTA      = 1.0    # m — bước dịch chuyển tương đối tối đa
HL_MAX_STEPS   = 300    # steps mỗi episode
BANDWIDTH_DIST = 1.5    # m — khoảng cách lý tưởng cho băng thông cao

W_CONNECT   = 8.0
W_REDUNDANCY= 2.0
W_BANDWIDTH = 1.0
W_PATCH     = 3.0   # thưởng khi drone khác di chuyển vào vị trí drone đang sạc
W_ALIVE     = 0.2


class HighLevelGNNEnv(MyAviary, gym.Env):
    """
    Single-agent wrapper (dùng để debug / train PPO đơn giản).
    Action = (N, 3) delta_xyz cho TẤT CẢ drone ACTIVE.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict = None):
        config = config or {}
        MyAviary.__init__(
            self,
            num_drones=config.get("num_drones", 3),
            gui=config.get("gui", False),
            wind_random=config.get("wind_random", True),
            base_random=config.get("base_random", False),
        )
        gym.Env.__init__(self)
        self._step_count = 0

        N = self.num_drones
        # Obs: nodes (N, 8) + edge_index (2, N²)
        self.observation_space = spaces.Dict({
            "nodes":      spaces.Box(low=-np.inf, high=np.inf, shape=(N, 8), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=N, shape=(2, N * N), dtype=np.int64),
        })
        # Action: delta_xyz cho từng drone (N, 3)
        self.action_space = spaces.Box(low=-MAX_DELTA, high=MAX_DELTA,
                                       shape=(N, 3), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        obs_matrix = self._base_reset(seed=seed, options=options)
        return self._format_obs(obs_matrix), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        obs_matrix = self._get_obs_matrix()

        # Tính target tuyệt đối; drone không ACTIVE thì bỏ qua action
        targets = obs_matrix[:, :3].copy()
        for i in range(self.num_drones):
            if self.drone_states[i] == DroneState.ACTIVE:
                delta = np.clip(action[i], -MAX_DELTA, MAX_DELTA)
                targets[i] = obs_matrix[i, :3] + delta

        # Cập nhật rally_points = targets của drone ACTIVE
        # (khi drone RETURNING về, nó bay đến vị trí HIGH-LEVEL mới nhất)
        self._update_rally_points(targets, obs_matrix)

        self._step_physics(targets)

        new_obs = self._get_obs_matrix()
        reward  = self._compute_hl_reward(new_obs, targets, obs_matrix)

        terminated = self._is_done(new_obs)
        truncated  = self._step_count >= HL_MAX_STEPS

        info = {
            "connected":   self._graph_is_connected(new_obs),
            "redundancy":  self._approx_redundancy(new_obs),
            "states":      self.drone_states.tolist(),
        }
        return self._format_obs(new_obs), reward, terminated, truncated, info

    # ── Rally point management ────────────────────────────────

    def _update_rally_points(self, targets: np.ndarray, obs_matrix: np.ndarray):
        """
        Rally point của drone i = target của drone i khi nó ACTIVE.
        Khi drone i về sạc, rally_point[i] giữ nguyên vị trí cuối cùng nó được giao,
        nên khi quay lại nó sẽ bay về đúng vị trí trong topology.

        Nếu drone i đang CHARGING/DOCKED, high-level có thể gán rally_point mới
        (ví dụ: vị trí khác phù hợp hơn sau khi drone khác đã dịch chuyển vá lỗ).
        """
        for i in range(self.num_drones):
            if self.drone_states[i] == DroneState.ACTIVE:
                self.rally_points[i] = targets[i]
            # CHARGING/DOCKED: giữ nguyên rally_point hiện tại
            # High-level có thể truyền rally_point mới qua action
            # nếu muốn mở rộng — nhưng đơn giản thì giữ nguyên

    # ── Reward ───────────────────────────────────────────────

    def _compute_hl_reward(
        self,
        new_obs:    np.ndarray,
        targets:    np.ndarray,
        prev_obs:   np.ndarray,
    ) -> float:
        r = 0.0

        # 1. Kết nối A–B
        connected = self._graph_is_connected(new_obs)
        r += W_CONNECT if connected else -W_CONNECT * 2

        # 2. Dự phòng (mean degree)
        redundancy = self._approx_redundancy(new_obs)
        r += W_REDUNDANCY * min(redundancy, 3.0)  # cap ở 3

        # 3. Băng thông: thưởng khi các cặp kết nối gần nhau
        bw, n_pairs = 0.0, 0
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                d = np.linalg.norm(new_obs[i, :3] - new_obs[j, :3])
                if d < COMM_RADIUS:
                    n_pairs += 1
                    bw += max(0.0, 1.0 - d / BANDWIDTH_DIST)
        r += W_BANDWIDTH * (bw / max(n_pairs, 1))

        # 4. Vá lỗ hổng: thưởng khi drone ACTIVE di chuyển vào khoảng trống
        #    do drone CHARGING/DOCKED để lại
        #    Metric: với mỗi drone đang sạc, xem thử drone khác có tiến lại
        #    gần rally_point của nó không (so với step trước)
        for i in range(self.num_drones):
            if self.drone_states[i] in (DroneState.CHARGING, DroneState.DOCKED):
                target_rally = self.rally_points[i]
                for j in range(self.num_drones):
                    if j == i or self.drone_states[j] != DroneState.ACTIVE:
                        continue
                    prev_dist = np.linalg.norm(prev_obs[j, :3] - target_rally)
                    new_dist  = np.linalg.norm(new_obs[j, :3]  - target_rally)
                    improvement = prev_dist - new_dist
                    # Thưởng nếu drone j tiến lại gần rally_point của drone i đang sạc
                    r += W_PATCH * max(0.0, improvement)

        # 5. Thưởng tồn tại theo số drone đang bay
        n_active = int(np.sum(self.flying_mask))
        r += W_ALIVE * n_active

        return float(r)

    def _is_done(self, obs_matrix: np.ndarray) -> bool:
        # Done khi TẤT CẢ drone đều DOCKED (không còn ai bay)
        return bool(np.all(self.drone_states == DroneState.DOCKED))

    def _format_obs(self, obs_matrix: np.ndarray) -> dict:
        return {
            "nodes":      obs_matrix.astype(np.float32),
            "edge_index": self._build_edge_index(obs_matrix, COMM_RADIUS),
        }


# ──────────────────────────────────────────────────────────────
# Multi-Agent wrapper (Ray RLlib MAPPO)
# ──────────────────────────────────────────────────────────────

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class HighLevelMAEnv(HighLevelGNNEnv, MultiAgentEnv):
    """
    Mỗi agent = 1 drone.
    Shared GNN policy: tất cả drone dùng cùng 1 model.

    CTDE:
      - Obs: mỗi agent nhận TOÀN BỘ đồ thị + agent_idx (centralized input)
      - Action: mỗi agent chỉ output delta_xyz cho chính nó (decentralized output)
      - Critic trong GNN dùng global embedding (centralized value)
    """

    def __init__(self, config: dict = None):
        HighLevelGNNEnv.__init__(self, config)
        MultiAgentEnv.__init__(self)

        N = self.num_drones
        self._agent_ids = set(f"drone_{i}" for i in range(N))

        # Mỗi agent nhận: đồ thị đầy đủ + index của mình
        self.observation_space = spaces.Dict({
            "nodes":      spaces.Box(low=-np.inf, high=np.inf, shape=(N, 8), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=N, shape=(2, N * N), dtype=np.int64),
            "agent_idx":  spaces.Box(low=0, high=N, shape=(1,), dtype=np.int64),
        })
        # Mỗi agent output delta_xyz cho chính nó
        self.action_space = spaces.Box(low=-MAX_DELTA, high=MAX_DELTA,
                                       shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        obs_matrix = self._base_reset(seed=seed, options=options)
        return self._make_agent_obs(obs_matrix), {}

    def step(self, action_dict: dict):
        self._step_count += 1
        obs_matrix = self._get_obs_matrix()

        # Ghép action
        targets = obs_matrix[:, :3].copy()
        for i in range(self.num_drones):
            aid = f"drone_{i}"
            if aid in action_dict and self.drone_states[i] == DroneState.ACTIVE:
                delta = np.clip(action_dict[aid], -MAX_DELTA, MAX_DELTA)
                targets[i] = obs_matrix[i, :3] + delta

        self._update_rally_points(targets, obs_matrix)
        self._step_physics(targets)

        new_obs  = self._get_obs_matrix()
        reward   = self._compute_hl_reward(new_obs, targets, obs_matrix)
        done     = self._is_done(new_obs)
        trunc    = self._step_count >= HL_MAX_STEPS

        obs_dict  = self._make_agent_obs(new_obs)
        rewards   = {f"drone_{i}": reward for i in range(self.num_drones)}
        terms     = {f"drone_{i}": done for i in range(self.num_drones)}
        terms["__all__"] = done
        truncs    = {f"drone_{i}": trunc for i in range(self.num_drones)}
        truncs["__all__"] = trunc

        return obs_dict, rewards, terms, truncs, {}

    def _make_agent_obs(self, obs_matrix: np.ndarray) -> dict:
        edge_index = self._build_edge_index(obs_matrix, COMM_RADIUS)
        return {
            f"drone_{i}": {
                "nodes":      obs_matrix.astype(np.float32),
                "edge_index": edge_index,
                "agent_idx":  np.array([i], dtype=np.int64),
            }
            for i in range(self.num_drones)
        }