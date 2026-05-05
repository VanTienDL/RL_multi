"""
myAviary.py — Lớp nền vật lý dùng chung.

Thay đổi so với v1:
  - Drone có 4 trạng thái: ACTIVE → CHARGING → DOCKED → RETURNING → ACTIVE
  - CHARGING : drone đang bay về trạm sạc (vẫn tốn pin)
  - DOCKED   : đã đáp, đang sạc (không bay, không tốn pin, pin tăng dần)
  - RETURNING: đủ pin, đang bay về rally_point (vị trí do high-level cập nhật)
  - ACTIVE   : hoạt động bình thường, high-level điều khiển
"""

import numpy as np
import pybullet as p
from collections import deque
from enum import IntEnum

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


# ──────────────────────────────────────────────────────────────
# Hằng số toàn cục
# ──────────────────────────────────────────────────────────────
COMM_RADIUS          = 2.0    # m
SENSOR_RADIUS        = 1.0    # m — bán kính cảm biến va chạm (low-level)
BATTERY_DRAIN        = 0.001  # per sim-step khi đang bay
BATTERY_CHARGE_RATE  = 0.003  # per sim-step khi DOCKED
LOW_BATTERY          = 0.15   # ngưỡng → chuyển CHARGING
FULL_BATTERY         = 0.90   # ngưỡng → chuyển RETURNING
DOCK_THRESHOLD       = 0.25   # m — xem như đã đáp/đến nơi
SIM_STEPS_PER_ACTION = 20     # bước PyBullet mỗi RL action


class DroneState(IntEnum):
    ACTIVE    = 0
    CHARGING  = 1   # đang bay về trạm
    DOCKED    = 2   # đang sạc tại trạm
    RETURNING = 3   # đang bay về rally point


# ──────────────────────────────────────────────────────────────
class MyAviary(CtrlAviary):

    def __init__(
        self,
        num_drones: int = 3,
        gui: bool = False,
        wind_random: bool = False,
        base_random: bool = False,
    ):
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=num_drones,
            physics=Physics.PYB,
            gui=gui,
            record=False,
        )
        self.num_drones  = num_drones
        self.wind_random = wind_random
        self.base_random = base_random

        self.ctrl = [DSLPIDControl(DroneModel.CF2X) for _ in range(num_drones)]

        self.battery      = np.ones(num_drones, dtype=np.float32)
        self.drone_states = np.full(num_drones, DroneState.ACTIVE, dtype=np.int32)

        # Rally points: high-level cập nhật liên tục; default = dàn đều A→B
        self.rally_points = np.zeros((num_drones, 3), dtype=np.float32)

        self.base_A = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        self.base_B = np.array([5.0, 0.0, 0.5], dtype=np.float32)

        # Phân tải trạm sạc: drone chẵn → A, lẻ → B
        self._charge_stations = self._assign_charge_stations()

        self._set_default_wind()

    # ── Helpers ───────────────────────────────────────────────

    def _assign_charge_stations(self) -> np.ndarray:
        stations = np.array([
            self.base_A if i % 2 == 0 else self.base_B
            for i in range(self.num_drones)
        ], dtype=np.float32)
        return stations

    @property
    def active_mask(self) -> np.ndarray:
        return self.drone_states == DroneState.ACTIVE

    @property
    def flying_mask(self) -> np.ndarray:
        """Drone đang bay (không DOCKED) — tốn pin."""
        return self.drone_states != DroneState.DOCKED

    # ── Wind ──────────────────────────────────────────────────

    def _set_default_wind(self):
        self.wind_zone_min = np.array([2.0, -2.0, 0.0])
        self.wind_zone_max = np.array([4.0,  2.0, 3.0])
        self.wind_vector   = np.array([0.2,  0.0, 0.0])

    def _randomize_wind(self):
        center = np.array([np.random.uniform(2, 5),
                           np.random.uniform(-2, 2), 1.5])
        size   = np.random.uniform(1.0, 2.0, size=3)
        self.wind_zone_min = center - size / 2
        self.wind_zone_max = center + size / 2
        d = np.random.randn(3)
        self.wind_vector = (d / (np.linalg.norm(d) + 1e-8)) * np.random.uniform(0.1, 0.5)

    def _in_wind_zone(self, pos: np.ndarray) -> bool:
        return bool(np.all(pos >= self.wind_zone_min) and np.all(pos <= self.wind_zone_max))

    # ── Observation matrix: (N, 8) ────────────────────────────
    # [pos(3), vel(3), battery(1), state(1)]

    def _get_obs_matrix(self) -> np.ndarray:
        raw = super()._computeObs()  # (N, 20)
        obs = np.zeros((self.num_drones, 8), dtype=np.float32)
        for i in range(self.num_drones):
            obs[i, 0:3] = raw[i, 0:3]
            obs[i, 3:6] = raw[i, 10:13]
            obs[i, 6]   = self.battery[i]
            obs[i, 7]   = float(self.drone_states[i]) / 3.0  # normalize 0→1
        return obs

    # ── State machine ─────────────────────────────────────────

    def _update_drone_states(self, obs_matrix: np.ndarray):
        """
        Transition logic (gọi sau mỗi sim step):
          ACTIVE    → CHARGING  : battery < LOW_BATTERY
          CHARGING  → DOCKED    : dist_to_charger < DOCK_THRESHOLD
          DOCKED    → RETURNING : battery >= FULL_BATTERY  (pin sạc đủ)
          RETURNING → ACTIVE    : dist_to_rally < DOCK_THRESHOLD
        """
        for i in range(self.num_drones):
            pos   = obs_matrix[i, :3]
            bat   = self.battery[i]
            state = DroneState(self.drone_states[i])

            if state == DroneState.ACTIVE:
                if bat < LOW_BATTERY:
                    self.drone_states[i] = DroneState.CHARGING

            elif state == DroneState.CHARGING:
                if np.linalg.norm(pos - self._charge_stations[i]) < DOCK_THRESHOLD:
                    self.drone_states[i] = DroneState.DOCKED

            elif state == DroneState.DOCKED:
                self.battery[i] = min(1.0, bat + BATTERY_CHARGE_RATE)
                if self.battery[i] >= FULL_BATTERY:
                    self.drone_states[i] = DroneState.RETURNING

            elif state == DroneState.RETURNING:
                if np.linalg.norm(pos - self.rally_points[i]) < DOCK_THRESHOLD:
                    self.drone_states[i] = DroneState.ACTIVE

    # ── Physics step ──────────────────────────────────────────

    def _step_physics(self, target_positions: np.ndarray, n_steps: int = SIM_STEPS_PER_ACTION):
        """
        target_positions: (N, 3) tọa độ đích tuyệt đối cho drone ACTIVE.
        CHARGING  → target = charge_station
        RETURNING → target = rally_point
        DOCKED    → không gửi RPM (đứng im, sạc pin)
        """
        for _ in range(n_steps):
            raw      = super()._computeObs()
            full_rpm = np.zeros((self.num_drones, 4), dtype=np.float32)

            for i in range(self.num_drones):
                state = DroneState(self.drone_states[i])

                if state == DroneState.DOCKED:
                    continue  # không bay, pin tăng trong _update_drone_states

                pos     = raw[i, 0:3]
                quat    = raw[i, 3:7]
                vel     = raw[i, 10:13]
                ang_vel = raw[i, 13:16]

                if   state == DroneState.CHARGING:
                    target = self._charge_stations[i]
                elif state == DroneState.RETURNING:
                    target = self.rally_points[i]
                else:  # ACTIVE
                    target = target_positions[i]

                rpm, _, _ = self.ctrl[i].computeControl(
                    control_timestep=1.0 / 240.0,
                    cur_pos=pos, cur_quat=quat,
                    cur_vel=vel, cur_ang_vel=ang_vel,
                    target_pos=target,
                )
                full_rpm[i] = rpm

                if self._in_wind_zone(pos):
                    p.applyExternalForce(
                        self.DRONE_IDS[i], -1,
                        self.wind_vector.tolist(), pos.tolist(), p.WORLD_FRAME,
                    )

                self.battery[i] = max(0.0, self.battery[i] - BATTERY_DRAIN)

            super().step(full_rpm)
            self._update_drone_states(self._get_obs_matrix())

    # ── Reset ─────────────────────────────────────────────────

    def _base_reset(self, seed=None, options=None) -> np.ndarray:
        self.battery      = np.ones(self.num_drones, dtype=np.float32)
        self.drone_states = np.full(self.num_drones, DroneState.ACTIVE, dtype=np.int32)

        if self.base_random:
            self.base_A = np.array([np.random.uniform(0, 2),
                                    np.random.uniform(-2, 2), 0.5], dtype=np.float32)
            self.base_B = np.array([np.random.uniform(4, 8),
                                    np.random.uniform(-2, 2), 0.5], dtype=np.float32)
            self._charge_stations = self._assign_charge_stations()

        if self.wind_random:
            self._randomize_wind()

        # Rally points ban đầu = dàn đều giữa A và B
        vec   = self.base_B - self.base_A
        dist  = np.linalg.norm(vec) + 1e-8
        direc = vec / dist
        step  = dist / (self.num_drones + 1)
        for i in range(self.num_drones):
            self.rally_points[i] = self.base_A + direc * step * (i + 1)
            self.rally_points[i, 2] = 1.0

        super().reset(seed=seed, options=options)
        return self._get_obs_matrix()

    # ── Graph utilities ───────────────────────────────────────

    def _build_edge_index(self, obs_matrix: np.ndarray, radius: float) -> np.ndarray:
        """Edge chỉ giữa drone đang bay (không DOCKED)."""
        max_edges = self.num_drones ** 2
        edges = []
        for i in range(self.num_drones):
            if self.drone_states[i] == DroneState.DOCKED:
                continue
            for j in range(self.num_drones):
                if i == j or self.drone_states[j] == DroneState.DOCKED:
                    continue
                if np.linalg.norm(obs_matrix[i, :3] - obs_matrix[j, :3]) < radius:
                    edges.append([i, j])

        ei = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        padded = np.zeros((2, max_edges), dtype=np.int64)
        padded[:, :ei.shape[1]] = ei
        return padded

    def _graph_is_connected(self, obs_matrix: np.ndarray) -> bool:
        """BFS — tránh recursion limit của Python."""
        nodes = [self.base_A, self.base_B] + [obs_matrix[i, :3] for i in range(self.num_drones)]
        # Node index: 0=base_A, 1=base_B, 2..N+1=drones
        active_ids = {0, 1}
        for i in range(self.num_drones):
            if self.drone_states[i] != DroneState.DOCKED:
                active_ids.add(i + 2)

        visited = {0}
        q = deque([0])
        while q:
            u = q.popleft()
            for v in active_ids - visited:
                if np.linalg.norm(nodes[u] - nodes[v]) < COMM_RADIUS:
                    visited.add(v)
                    q.append(v)
        return 1 in visited

    def _approx_redundancy(self, obs_matrix: np.ndarray) -> float:
        """Mean degree của drone ACTIVE/RETURNING — xấp xỉ nhanh cho số đường dự phòng."""
        degrees = np.zeros(self.num_drones)
        for i in range(self.num_drones):
            if self.drone_states[i] == DroneState.DOCKED:
                continue
            for j in range(self.num_drones):
                if i == j or self.drone_states[j] == DroneState.DOCKED:
                    continue
                if np.linalg.norm(obs_matrix[i, :3] - obs_matrix[j, :3]) < COMM_RADIUS:
                    degrees[i] += 1
        flying = self.flying_mask
        return float(degrees[flying].mean()) if flying.any() else 0.0