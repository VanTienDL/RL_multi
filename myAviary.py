import numpy as np
import pybullet as p

from gymnasium import spaces
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class MyAviary(CtrlAviary):

    #Constructor của MyAviary
    def __init__(self,
                 num_drones=3,
                 comm_radius=2,
                 gui=True,
                 baseRandom=False,
                 droneRandom=False,
                 windRandom=False
                 ):
        #Gọi tới constructor của CtrlAviary
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=num_drones,
            physics=Physics.PYB,
            gui=gui
        )

        self.num_drones = num_drones
        self.comm_radius = comm_radius

        # flags
        self.baseRandom = baseRandom
        self.droneRandom = droneRandom
        self.windRandom = windRandom

        # PID controller cho từng drone
        self.ctrl = [DSLPIDControl(DroneModel.CF2X) for _ in range(num_drones)]

        # target tạm thời ma trận tọa độ cho mỗi hàng drone toàn 0
        self.targets = np.zeros((num_drones, 3))

        # battery là mảng toàn 1 ứng với số drone
        self.battery = np.ones(num_drones)

        # ===== BASE ===== (sẽ bị reset đè lên nếu chọn Random)
        self.base_A = np.array([0, 0, 1]) #Trạm A
        self.base_B = np.array([5, 0, 1]) #Trạm B
        self._set_default_wind()

    # wind zone lấy tọa độ hai điểm đối diện nhau của hộp
    # ===== WIND =====
    def _set_default_wind(self):
        self.wind_zone_min = np.array([2, -2, 0])
        self.wind_zone_max = np.array([4, 2, 3])
        self.wind_vector = np.array([0.2, 0.0, 0.0])

    def _apply_physics_step(self, action_all):
        #Hàm lõi để chạy vật lý cho N drone, trả về rpm_all
        raw_obs = super()._computeObs() #Env gốc trả về
        rpm_all = []
        for i in range(self.num_drones): #Lấy các thông số cho input PID
            state = raw_obs[i]
            pos = state[0:3] #Lọc lấy tọa độ
            quat = state[3:7] #Lấy hướng (4 số)
            vel = state[10:13] #Lấy tốc độ
            ang_vel = state[13:16] #Lấy tốc độ góc

            # ===== ACTION → target position =====
            direction = action_all[i] #Ta tạo một mồi nhử tọa độ ngắn để dụ drone tới, biến input tọa độ thành input hướng đi
            self.targets[i] = pos + direction * 0.2

            # ===== PID control =====
            rpm, _, _ = self.ctrl[i].computeControl( #Nối tới hàm của PID
                control_timestep=1/240, #delta time để đạo hàm
                cur_pos=pos, 
                cur_quat=quat,
                cur_vel=vel, 
                cur_ang_vel=ang_vel, 
                target_pos=self.targets[i]
            )
            rpm_all.append(rpm) #Apply chuyển động cho mọi drone

            # ===== WIND =====
            if self._in_wind_zone(pos):
                p.applyExternalForce(self.DRONE_IDS[i],
                                      -1, # Đẩy vào trọng tâm drone
                                      self.wind_vector, 
                                      pos, # Điểm đặt lực là vị trí drone
                                      p.WORLD_FRAME) #Gió luôn fixed hệ trục XYZ, kệ trục drone
            # ===== BATTERY =====
            self.battery[i] -= 0.001 #Cứ mỗi step dù di chuyển hay đứng im đều tốn pin
            self.battery[i] = max(0, self.battery[i]) #pin không âm

            # Class con cần kế thừa thêm khúc: obs tác động env, check done?, trucated? và trả về state mới, reward, done?, trucated?, thông tin phụ
                
        return np.array(rpm_all)
    
    # =========================
    # Random func
    # =========================
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
        positions = [self.base_A + direction * (i+1) * step_dist for i in range(self.num_drones)]
        self.INIT_XYZS = np.array(positions)
        # Agent lẻ loi (sẽ được lớp con sử dụng)
        self.agent_idx = np.random.randint(self.num_drones)
        self.hole_position = self.INIT_XYZS[self.agent_idx].copy()
        self.INIT_XYZS[self.agent_idx] += np.array([0, 1.0, 2.0])

    # =========================
    # ACTION SPACE (Sửa lại cho 1 drone)
    # =========================
    def _actionSpace(self):
        # Mỗi drone nhận 3 giá trị (hướng x, y, z)
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,), # Chuyển từ (self.num_drones, 3) thành (3,)
            dtype=np.float32
        )

    # =========================
    # OBSERVATION SPACE (Sửa lại cho 1 drone)
    # =========================
    def _observationSpace(self):
        # Một drone quan sát: pos(3) + vel(3) + battery(1) = 7
        obs_dim = 3 + 3 + 1
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,), # Chuyển từ (self.num_drones, obs_dim) thành (obs_dim,)
            dtype=np.float32
        )

    # =========================
    # OBS
    # =========================
    def _computeObs(self): #Để lọc lấy các thông tin trả về từ class env cha
        obs = []

        raw = super()._computeObs() #Các thông số env trả về

        for i in range(self.num_drones):
            state = raw[i]

            pos = state[0:3] #Lấy ptu 0-3 của raw chính là tọa độ x,y,z
            vel = state[10:13] # Lấy ptu 10-13 là tốc độ vx, vy,vz

            obs.append(np.concatenate([ #Mix thêm state pin vào cho MyAviary
                pos,
                vel,
                [self.battery[i]]
            ]))

        return np.array(obs) # Trả ra ma trận n hàng với mỗi hàng (pos,vel,battery)

    # =========================
    # STEP
    # =========================
    def step(self, action):

        raw = super()._computeObs() #Env gốc trả về

        rpm_all = []

        for i in range(self.num_drones): #Lấy các thông số cho input PID

            state = raw[i]

            pos = state[0:3] #Lọc lấy tọa độ
            quat = state[3:7] #Lấy hướng (4 số)
            vel = state[10:13] #Lấy tốc độ
            ang_vel = state[13:16] #Lấy tốc độ góc

            # ===== ACTION → target position =====
            direction = action[i] #Ta tạo một mồi nhử tọa độ ngắn để dụ drone tới, biến input tọa độ thành input hướng đi
            self.targets[i] = pos + direction * 0.2

            # ===== PID control =====
            rpm, _, _ = self.ctrl[i].computeControl( #Nối tới hàm của PID
                control_timestep=1 / 240, #delta time để đạo hàm
                cur_pos=pos,
                cur_quat=quat,
                cur_vel=vel,
                cur_ang_vel=ang_vel,
                target_pos=self.targets[i]
            )

            rpm_all.append(rpm) #Apply chuyển động cho mọi drone

            # ===== WIND =====
            if self._in_wind_zone(pos):
                p.applyExternalForce(
                    self.DRONE_IDS[i],
                    -1, # Đẩy vào trọng tâm drone
                    self.wind_vector,
                    pos, # Điểm đặt lực là vị trí drone
                    p.WORLD_FRAME #Gió luôn fixed hệ trục XYZ, kệ trục drone
                )

            # ===== BATTERY =====
            self.battery[i] -= 0.001 #Cứ mỗi step dù di chuyển hay đứng im đều tốn pin
            self.battery[i] = max(0, self.battery[i]) #pin không âm

        rpm_all = np.array(rpm_all)

        obs, _, _, _, info = super().step(rpm_all) # Gửi tác động rpm vào env mô phỏng

        reward = self._computeReward() #check reward
        done = self._computeTerminated() #check dừng

        #Hàm trả về state mới, reward, done?, trucated?, thông tin phụ
        return self._computeObs(), reward, done, False, info 

    # =========================
    # RESET
    # =========================
    def reset(self, seed=None, options=None):
        if self.baseRandom:
            self.base_A, self.base_B = self._random_bases()

        if self.windRandom:
            self._random_wind()

        if self.droneRandom:
            self._init_drones_bridge()

        return super().reset(seed=seed, options=options)

    # =========================
    # WIND CHECK
    # =========================
    def _in_wind_zone(self, pos):
        return np.all(pos >= self.wind_zone_min) and np.all(pos <= self.wind_zone_max)

    # =========================
    # NETWORK CONNECTIVITY
    # =========================
    def _is_connected(self):

        nodes = [self.base_A, self.base_B]

        raw = super()._computeObs()
        for i in range(self.num_drones):
            nodes.append(raw[i][0:3]) #Lấy tọa độ mới mỗi drone

        N = len(nodes)

        visited = [False] * N

        def dfs(u):
            visited[u] = True
            for v in range(N):
                if not visited[v]:
                    if np.linalg.norm(nodes[u] - nodes[v]) < self.comm_radius:
                        dfs(v)

        dfs(0)  # từ base A

        return visited[1]  # base B reachable?

    # =========================
    # REWARD
    # =========================
    def _computeReward(self):

        reward = 0

        # connectivity
        if self._is_connected():
            reward += 10
        else:
            reward -= 20

        # penalty nếu drone chết pin
        for b in self.battery:
            if b <= 0:
                reward -= 50

        return reward

    # =========================
    # DONE
    # =========================
    def _computeTerminated(self):

        # terminate nếu tất cả drone chết
        return np.all(self.battery <= 0)