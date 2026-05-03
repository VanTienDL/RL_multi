import numpy as np
import pybullet as p

from gymnasium import spaces
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class MyAviary(CtrlAviary):

    def __init__(self,
                 num_drones=3,
                 comm_radius=2,
                 gui=True,
                 baseRandom=False,
                 droneRandom=False,
                 windRandom=False
                 ):
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=num_drones,
            physics=Physics.PYB,
            gui=gui
        )

        self.num_drones = num_drones
        self.comm_radius = comm_radius

        self.baseRandom = baseRandom
        self.droneRandom = droneRandom
        self.windRandom = windRandom

        self.ctrl = [DSLPIDControl(DroneModel.CF2X) for _ in range(num_drones)]
        self.targets = np.zeros((num_drones, 3))
        self.battery = np.ones(num_drones)

        self.base_A = np.array([0, 0, 1])
        self.base_B = np.array([5, 0, 1])
        self._set_default_wind()

    def _set_default_wind(self):
        self.wind_zone_min = np.array([2, -2, 0])
        self.wind_zone_max = np.array([4, 2, 3])
        self.wind_vector = np.array([0.2, 0.0, 0.0])

    def _random_bases(self):
        A = np.array([np.random.uniform(0, 2), np.random.uniform(-2, 2), 1])
        B = np.array([np.random.uniform(4, 8), np.random.uniform(-2, 2), 1])
        return A, B

    def _random_wind(self):
        center = np.array([
            np.random.uniform(2, 5),
            np.random.uniform(-2, 2),
            1.5
        ])
        size = np.array([
            np.random.uniform(1, 2),
            np.random.uniform(1, 2),
            np.random.uniform(1, 2)
        ])
        self.wind_zone_min = center - size / 2
        self.wind_zone_max = center + size / 2

        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        strength = np.random.uniform(0.1, 0.5)
        self.wind_vector = direction * strength

    def _init_drones_bridge(self):
        vec = self.base_B - self.base_A
        dist = np.linalg.norm(vec)
        direction = vec / dist
        step_dist = dist / (self.num_drones + 1)
        positions = [self.base_A + direction * (i + 1) * step_dist for i in range(self.num_drones)]
        self.INIT_XYZS = np.array(positions)
        self.agent_idx = np.random.randint(self.num_drones)
        self.hole_position = self.INIT_XYZS[self.agent_idx].copy()
        self.INIT_XYZS[self.agent_idx] += np.array([0, 1.0, 2.0])

    def _actionSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _observationSpace(self):
        obs_dim = 3 + 3 + 1  # pos, vel, battery
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _computeObs(self):
        obs = []
        raw = super()._computeObs()
        for i in range(self.num_drones):
            state = raw[i]
            pos = state[0:3]
            vel = state[10:13]
            obs.append(np.concatenate([pos, vel, [self.battery[i]]]))
        return np.array(obs, dtype=np.float32)  # shape: (num_drones, 7)

    def step(self, action):
        raw = super()._computeObs()
        rpm_all = []

        for i in range(self.num_drones):
            state = raw[i]
            pos = state[0:3]
            quat = state[3:7]
            vel = state[10:13]
            ang_vel = state[13:16]

            direction = action[i]
            self.targets[i] = pos + direction * 0.2

            rpm, _, _ = self.ctrl[i].computeControl(
                control_timestep=1 / 240,
                cur_pos=pos,
                cur_quat=quat,
                cur_vel=vel,
                cur_ang_vel=ang_vel,
                target_pos=self.targets[i]
            )
            rpm_all.append(rpm)

            if self._in_wind_zone(pos):
                p.applyExternalForce(
                    self.DRONE_IDS[i], -1,
                    self.wind_vector, pos, p.WORLD_FRAME
                )

            self.battery[i] -= 0.001
            self.battery[i] = max(0, self.battery[i])

        rpm_all = np.array(rpm_all)
        _, _, _, _, info = super().step(rpm_all)

        reward = self._computeReward()
        done = self._computeTerminated()

        return self._computeObs(), reward, done, False, info

    def reset(self, seed=None, options=None):
        # FIX: Reset battery trước khi khởi tạo lại môi trường
        self.battery = np.ones(self.num_drones)

        if self.baseRandom:
            self.base_A, self.base_B = self._random_bases()
        if self.windRandom:
            self._random_wind()
        if self.droneRandom:
            self._init_drones_bridge()

        # FIX: Gọi super().reset() rồi override obs bằng _computeObs() của chính mình
        _, info = super().reset(seed=seed, options=options)
        obs = self._computeObs()  # shape (num_drones, 7) — đúng format
        return obs, info

    def _in_wind_zone(self, pos):
        return np.all(pos >= self.wind_zone_min) and np.all(pos <= self.wind_zone_max)

    def _is_connected(self):
        nodes = [self.base_A, self.base_B]
        raw = super()._computeObs()
        for i in range(self.num_drones):
            nodes.append(raw[i][0:3])

        N = len(nodes)
        visited = [False] * N

        def dfs(u):
            visited[u] = True
            for v in range(N):
                if not visited[v]:
                    if np.linalg.norm(nodes[u] - nodes[v]) < self.comm_radius:
                        dfs(v)

        dfs(0)
        return visited[1]

    def _computeReward(self):
        reward = 0
        if self._is_connected():
            reward += 10
        else:
            reward -= 20
        for b in self.battery:
            if b <= 0:
                reward -= 50
        return reward

    def _computeTerminated(self):
        return bool(np.all(self.battery <= 0))