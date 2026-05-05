"""
env_lowlevel.py — Low-level DQN env: Single agent, discrete actions, 1 drone.

─────────────────────────────────────────────────────────
  Nhiệm vụ: Di chuyển drone đến tọa độ đích được giao từ high-level.

  State  : [pos(3), vel(3), battery(1), target_delta(3), neighbor_rel(K*3)]
             - pos/vel/battery: 7 biến cơ bản
             - target_delta: vector từ vị trí hiện tại → tọa độ đích (hướng cần đi)
             - neighbor_rel: vị trí tương đối của K drone lân cận trong sensor_radius
               (padding zeros nếu không đủ K)

  Action : 7 hành động rời rạc (Discrete(7))
             0: Lên     (+z)
             1: Xuống   (-z)
             2: Trái    (+x)  ← theo quy ước local frame
             3: Phải    (-x)
             4: Tới     (+y)
             5: Lùi     (-y)
             6: Đứng im (không di chuyển)

  Reward : Cosine similarity reward + collision avoidance penalty
─────────────────────────────────────────────────────────

  NOTE về SIM_STEPS_PER_ACTION:
  Mỗi discrete action = 20 bước PyBullet (≈ 0.083s thực).
  Drone CF2X bay ~0.5 m/s → trong 20 bước dịch ~0.04m.
  Ta set step_size=0.25m như target offset để PID tự điều chỉnh tốc độ.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from myAviary import MyAviary, COMM_RADIUS, SENSOR_RADIUS, SIM_STEPS_PER_ACTION, LOW_BATTERY

# ──────────────────────────────────────────
# Cấu hình Low-level
# ──────────────────────────────────────────
STEP_SIZE       = 0.25   # m — bước dịch chuyển của mỗi action
MAX_NEIGHBORS   = 3      # số drone lân cận tối đa quan sát được
LL_MAX_STEPS    = 300    # bước tối đa mỗi episode low-level
REACH_THRESHOLD = 0.3    # m — xem như đã tới đích
COLLISION_DIST  = 0.5    # m — bán kính va chạm nguy hiểm

# Vector dịch chuyển cho 7 action (x, y, z offsets)
ACTION_DELTAS = np.array([
    [ 0,  0,  STEP_SIZE],   # 0: Lên
    [ 0,  0, -STEP_SIZE],   # 1: Xuống
    [ STEP_SIZE, 0, 0],     # 2: Trái  (theo trục X+)
    [-STEP_SIZE, 0, 0],     # 3: Phải  (theo trục X-)
    [0,  STEP_SIZE, 0],     # 4: Tới   (trục Y+)
    [0, -STEP_SIZE, 0],     # 5: Lùi   (trục Y-)
    [0,  0,  0],            # 6: Đứng im
], dtype=np.float32)

NUM_ACTIONS = len(ACTION_DELTAS)  # = 7

# Trọng số reward
W_GOAL      = 2.0   # cosine similarity với vector đích
W_COLLISION = 5.0   # penalty va chạm (lấn át W_GOAL)
W_REACH     = 10.0  # thưởng khi tới đích
W_BATTERY   = 0.5   # penalty mỗi step (tiết kiệm pin)


class LowLevelDQNEnv(MyAviary, gym.Env):
    """
    Env low-level cho 1 drone.
    Drone index được chỉ định qua `drone_idx` trong config.

    Dùng riêng để train DQN trước, sau đó plug vào hệ thống CTDE.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict = None):
        config = config or {}

        MyAviary.__init__(
            self,
            num_drones=config.get("num_drones", 3),
            gui=config.get("gui", False),
            wind_random=config.get("wind_random", False),
            base_random=config.get("base_random", False),
        )
        gym.Env.__init__(self)

        # Drone nào là agent chính (các drone khác = obstacle / neighbor)
        self.agent_idx  = config.get("drone_idx", 0)
        self._step_count = 0

        # Tọa độ đích được giao từ high-level (hoặc set thủ công khi train độc lập)
        self.target_pos = np.zeros(3, dtype=np.float32)

        # ── Observation Space ──────────────────────────────────
        # [pos(3) + vel(3) + bat(1) + state(1) + target_delta(3) + neighbor_rel(K*3)]
        # NOTE: obs matrix từ myAviary giờ là 8 features (thêm drone_state ở index 7)
        obs_dim = 8 + 3 + MAX_NEIGHBORS * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ── Action Space ───────────────────────────────────────
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Cache action vectors để tính reward nhanh
        self._action_deltas = ACTION_DELTAS

    # ──────────────────────────────────────────────────────────
    # Public: set target từ high-level
    # ──────────────────────────────────────────────────────────

    def set_target(self, target_pos: np.ndarray):
        """High-level env gọi hàm này để truyền tọa độ đích xuống."""
        self.target_pos = np.array(target_pos, dtype=np.float32)

    # ──────────────────────────────────────────────────────────
    # Gymnasium interface
    # ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        obs_matrix = self._base_reset(seed=seed, options=options)

        # Khi train độc lập: random target trong không gian bay
        if options and "target_pos" in options:
            self.target_pos = np.array(options["target_pos"], dtype=np.float32)
        else:
            self.target_pos = np.array([
                np.random.uniform(1, 4),
                np.random.uniform(-2, 2),
                np.random.uniform(0.5, 3),
            ], dtype=np.float32)

        return self._build_obs(obs_matrix), {}

    def step(self, action: int):
        self._step_count += 1

        obs_matrix   = self._get_obs_matrix()
        agent_pos    = obs_matrix[self.agent_idx, :3].copy()
        action_delta = self._action_deltas[action]

        # ── Tính reward TRƯỚC khi thực hiện action ────────────
        # (Đánh giá CHẤT LƯỢNG của lựa chọn action, không phụ thuộc kết quả)
        reward = self._compute_lowlevel_reward(agent_pos, action_delta, obs_matrix)

        # ── Thực hiện action: target = pos + delta ─────────────
        # Các drone khác giữ nguyên vị trí (target = current pos)
        targets = obs_matrix[:, :3].copy()
        targets[self.agent_idx] = agent_pos + action_delta

        self._step_physics(targets, n_steps=SIM_STEPS_PER_ACTION)

        # ── Obs mới ────────────────────────────────────────────
        new_obs_matrix = self._get_obs_matrix()
        obs_out        = self._build_obs(new_obs_matrix)

        # ── Kiểm tra tới đích ─────────────────────────────────
        new_pos    = new_obs_matrix[self.agent_idx, :3]
        dist_to_goal = np.linalg.norm(new_pos - self.target_pos)
        reached = dist_to_goal < REACH_THRESHOLD

        if reached:
            reward += W_REACH

        # ── Battery penalty ────────────────────────────────────
        reward -= W_BATTERY

        # ── Done ───────────────────────────────────────────────
        terminated = reached or bool(self.battery[self.agent_idx] <= 0)
        truncated  = self._step_count >= LL_MAX_STEPS

        info = {
            "dist_to_goal": float(dist_to_goal),
            "reached":      reached,
            "battery":      float(self.battery[self.agent_idx]),
        }

        return obs_out, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────
    # Reward: Cosine similarity + Collision avoidance
    # ──────────────────────────────────────────────────────────

    def _compute_lowlevel_reward(
        self,
        agent_pos:    np.ndarray,
        action_delta: np.ndarray,
        obs_matrix:   np.ndarray,
    ) -> float:
        """
        Tính reward dựa trên hướng action so với:
          1. Vector đích (muốn maximize cosine similarity)
          2. Vector va chạm với obstacle/drone lân cận (muốn minimize)

        Công thức:
          reward = W_GOAL * cos(action, goal_vec)
                 - W_COLLISION * sum(max(0, cos(action, collision_vec_k)))
                    [chỉ khi drone k trong COLLISION_DIST]
        """

        # ── Vector đích ────────────────────────────────────────
        goal_vec  = self.target_pos - agent_pos
        goal_norm = np.linalg.norm(goal_vec)

        if goal_norm < 1e-6:
            # Đã tới đích → không cần di chuyển → action "đứng im" tốt nhất
            goal_cosine = 1.0 if np.allclose(action_delta, 0) else -0.5
        else:
            action_norm = np.linalg.norm(action_delta)
            if action_norm < 1e-6:
                # Action = đứng im, vector đích chưa = 0 → neutral
                goal_cosine = 0.0
            else:
                goal_cosine = np.dot(action_delta, goal_vec) / (action_norm * goal_norm)

        reward = W_GOAL * goal_cosine

        # ── Collision avoidance ────────────────────────────────
        # Duyệt tất cả drone khác (và có thể obstacle cố định nếu muốn mở rộng)
        collision_penalty = 0.0

        for j in range(self.num_drones):
            if j == self.agent_idx:
                continue

            other_pos = obs_matrix[j, :3]
            dist      = np.linalg.norm(agent_pos - other_pos)

            if dist < SENSOR_RADIUS:  # trong tầm cảm biến
                # Vector từ agent → drone khác (hướng va chạm tiềm năng)
                collision_vec  = other_pos - agent_pos
                collision_norm = np.linalg.norm(collision_vec)

                if collision_norm < 1e-6:
                    # Hai drone ở cùng vị trí → penalty tối đa
                    collision_penalty += 1.0
                    continue

                action_norm = np.linalg.norm(action_delta)
                if action_norm < 1e-6:
                    # Đứng im → không tiến về phía collision → không penalty
                    continue

                # Cosine giữa action và hướng va chạm
                cos_col = np.dot(action_delta, collision_vec) / (action_norm * collision_norm)

                # Chỉ penalty khi action đang DI VỀ PHÍA drone kia (cos > 0)
                # Scale theo mức độ nguy hiểm: drone càng gần penalty càng lớn
                danger_factor = 1.0 - dist / SENSOR_RADIUS  # [0, 1]
                if cos_col > 0:
                    collision_penalty += cos_col * danger_factor

        # Collision penalty lấn át goal reward (quan trọng hơn)
        reward -= W_COLLISION * collision_penalty

        return float(reward)

    # ──────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────

    def _build_obs(self, obs_matrix: np.ndarray) -> np.ndarray:
        """
        Xây dựng vector observation cho agent.
        Shape: (8 + 3 + MAX_NEIGHBORS*3,) = (8+3+9,) = (20,) với MAX_NEIGHBORS=3
          [pos(3), vel(3), battery(1), drone_state(1), target_delta(3), neighbor_rel(K*3)]
        """
        agent_state = obs_matrix[self.agent_idx]   # (8,) — pos,vel,bat,state
        agent_pos   = agent_state[:3]

        # Vector từ vị trí hiện tại → đích (relative)
        target_delta = self.target_pos - agent_pos  # (3,)

        # Neighbor positions relative (trong SENSOR_RADIUS)
        neighbors = []
        for j in range(self.num_drones):
            if j == self.agent_idx:
                continue
            other_pos = obs_matrix[j, :3]
            dist      = np.linalg.norm(agent_pos - other_pos)
            if dist < SENSOR_RADIUS:
                neighbors.append(other_pos - agent_pos)

        # Padding / truncation đến MAX_NEIGHBORS
        neighbor_obs = np.zeros((MAX_NEIGHBORS, 3), dtype=np.float32)
        for k, nv in enumerate(neighbors[:MAX_NEIGHBORS]):
            neighbor_obs[k] = nv

        obs = np.concatenate([
            agent_state,              # (8,)
            target_delta,             # (3,)
            neighbor_obs.flatten(),   # (MAX_NEIGHBORS*3,)
        ]).astype(np.float32)

        return obs


# ──────────────────────────────────────────────────────────────
# Wrapper tiện lợi: Train Low-level KHÔNG cần high-level
# Tự random target mỗi episode, phù hợp để pre-train DQN
# ──────────────────────────────────────────────────────────────

class LowLevelStandaloneDQN(LowLevelDQNEnv):
    """
    Standalone version: target được random mỗi episode.
    Dùng để train DQN độc lập trước khi tích hợp vào hệ thống 2 tầng.
    """

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        obs_matrix = self._base_reset(seed=seed, options=options)

        # Random target trong vùng bay hợp lệ
        self.target_pos = np.array([
            np.random.uniform(0.5, 4.5),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(0.5, 2.5),
        ], dtype=np.float32)

        return self._build_obs(obs_matrix), {"target": self.target_pos.tolist()}